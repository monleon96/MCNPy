"""The format-agnostic sampler: what it guarantees, and what it refuses.

``draw_samples`` is meant to be the one place this library draws correlated
realisations of a covariance, whatever file format the covariance came out of.
These tests are about the properties that make that possible — it must know
nothing about the caller, reproduce the covariance it was given, and be
reproducible block by block — plus the two things it deliberately will not do.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.sampling.core import BLOCK_SEED_STRIDE, draw_samples


def psd(n: int, rank: int, seed: int, zero_sum: bool = False) -> np.ndarray:
    """A rank-deficient PSD matrix, optionally with ``C·1 = 0``."""
    rng = np.random.default_rng(seed)
    modes = rng.normal(size=(n, rank))
    matrix = modes @ modes.T
    if zero_sum:
        projector = np.eye(n) - np.ones((n, n)) / n
        matrix = projector @ matrix @ projector.T
    return 0.5 * (matrix + matrix.T)


# ---------------------------------------------------------------------------
# Input shapes: the core holds no covariance class
# ---------------------------------------------------------------------------

def test_blocks_may_be_a_mapping_a_pair_sequence_or_bare_matrices():
    """Three spellings, one meaning. The key is opaque and never inspected."""
    a, b = psd(6, 3, 0), psd(4, 2, 1)

    as_map, _ = draw_samples({"a": a, "b": b}, 8, seed=1, verbose=False)
    as_pairs, _ = draw_samples([("a", a), ("b", b)], 8, seed=1, verbose=False)
    as_bare, _ = draw_samples([a, b], 8, seed=1, verbose=False)

    assert set(as_map) == {"a", "b"}
    assert set(as_bare) == {0, 1}
    np.testing.assert_array_equal(as_map["a"], as_pairs["a"])
    np.testing.assert_array_equal(as_map["a"], as_bare[0])


def test_tuple_keys_survive():
    """The PFNS caller keys on ``(isotope, mt, band)``; MF34 would use an order."""
    key = ("U235", 18, 3)
    samples, diagnostics = draw_samples({key: psd(5, 2, 0)}, 8, seed=1, verbose=False)
    assert list(samples) == [key] and list(diagnostics) == [key]


# ---------------------------------------------------------------------------
# What it guarantees
# ---------------------------------------------------------------------------

def test_output_is_float64_and_absolute_by_default():
    """``1 + δ`` in float32 is how an absolute perturbation vanishes silently."""
    samples, _ = draw_samples([psd(6, 3, 0)], 64, seed=1, verbose=False)
    drawn = samples[0]
    assert drawn.dtype == np.float64
    assert abs(float(drawn.mean())) < 1.0, "deltas should centre on 0, not on 1"


def test_returns_factors_offsets_by_one():
    """The shape the existing ENDF/ACE drivers multiply by, ready for migration."""
    matrix = psd(6, 3, 0)
    deltas, _ = draw_samples([matrix], 64, seed=1, verbose=False)
    factors, _ = draw_samples([matrix], 64, seed=1, returns="factors", verbose=False)
    np.testing.assert_allclose(factors[0], deltas[0] + 1.0, rtol=0, atol=0)


def test_the_realised_covariance_reproduces_the_input():
    """On the retained subspace, which is the only place it can be checked."""
    matrix = psd(20, 12, 3)
    _, diagnostics = draw_samples([matrix], 20000, sampling_method="random",
                                  seed=5, verbose=False)
    assert diagnostics[0]["realised_covariance_error"] < 0.05


def test_seeds_differ_per_block_by_a_fixed_stride():
    """Otherwise every block draws the same Z and the blocks are identical.

    The quiet failure this prevents: each block looks right on its own, and the
    ensemble is perfectly correlated across blocks.
    """
    matrix = psd(8, 4, 0)
    samples, diagnostics = draw_samples([matrix, matrix, matrix], 32,
                                        seed=100, verbose=False)
    assert [diagnostics[i]["seed"] for i in range(3)] == [
        100, 100 + BLOCK_SEED_STRIDE, 100 + 2 * BLOCK_SEED_STRIDE
    ]
    assert not np.allclose(samples[0], samples[1])
    assert not np.allclose(samples[1], samples[2])


def test_the_same_seed_reproduces_the_same_draw():
    matrix = psd(8, 4, 0)
    first, _ = draw_samples([matrix], 32, seed=11, verbose=False)
    second, _ = draw_samples([matrix], 32, seed=11, verbose=False)
    np.testing.assert_array_equal(first[0], second[0])


def test_rank_deficiency_is_recorded_and_not_repaired():
    matrix = psd(20, 6, 2)
    _, diagnostics = draw_samples([matrix], 32, seed=1, verbose=False)
    info = diagnostics[0]
    assert info["n"] == 20
    assert info["rank"] == 6
    assert info["n_null"] == 14
    assert info["null_fraction"] == pytest.approx(0.7)


def test_null_tol_none_draws_in_every_direction_and_still_counts_the_null_ones():
    """The migration escape hatch, and the honesty it is required to keep.

    A migrated call site has to be able to prove it draws what the code it
    replaces drew, and truncation makes that impossible to prove: it changes the
    QMC dimension from *n* to the rank, so every drawn column moves and none of
    the difference can be attributed. ``null_tol=None`` retains everything so
    the comparison is a real one.

    What it must **not** do is report full rank. The counts stay at
    ``DEFAULT_NULL_TOL`` and describe the matrix, not the subspace this draw
    happened to use — otherwise the diagnostics of a gate run would claim a rank
    deficiency had gone away.
    """
    matrix = psd(20, 6, 2)

    kept, info = draw_samples([matrix], 32, seed=1, null_tol=None, verbose=False)
    truncated, _ = draw_samples([matrix], 32, seed=1, verbose=False)

    assert info[0]["rank"] == 6
    assert info[0]["n_null"] == 14

    # Drawn in 20 directions rather than 6, so the draw itself is a different
    # one -- which is the whole reason the two cannot land in the same commit.
    assert not np.allclose(kept[0], truncated[0])


def test_a_component_with_no_stated_variance_gets_no_delta():
    """The reason the decomposition is truncated to the retained rank.

    Component 0 is uncorrelated with everything and has zero variance. Keeping
    the null columns of the factor gives it a delta of ~1e-8 of the leading
    scale — nothing in absolute terms, but PFNS divides deltas by group
    probabilities as small as 1e-17, which turns that debris into a
    perturbation factor of 1e+7. Truncation makes it exactly zero, because
    every retained eigenvector vanishes there.
    """
    matrix = np.zeros((7, 7))
    matrix[1:, 1:] = psd(6, 3, 4)
    assert matrix[0, 0] == 0.0

    samples, _ = draw_samples([matrix], 256, seed=1, verbose=False)
    assert np.all(samples[0][:, 0] == 0.0)


def test_a_zero_sum_covariance_draws_zero_sum_deltas():
    """``C·1 = 0`` in, ``1ᵀδ = 0`` out — the property PFNS normalisation rests on."""
    matrix = psd(30, 10, 7, zero_sum=True)
    samples, _ = draw_samples([matrix], 256, seed=1, verbose=False)
    scale = np.sqrt(np.trace(matrix))
    assert np.max(np.abs(samples[0].sum(axis=1))) < 1e-10 * scale


def test_input_order_is_never_sorted():
    """Reordering the blocks must reorder the draws, not silently repair itself.

    The block index sets the seed offset, so a sampler that sorted its input
    would give different answers for inputs the caller considers equivalent —
    and would do so only when the keys happened to be unsorted.
    """
    a, b = psd(5, 2, 0), psd(5, 2, 1)
    forward, _ = draw_samples([("b", b), ("a", a)], 16, seed=3, verbose=False)
    reverse, _ = draw_samples([("a", a), ("b", b)], 16, seed=3, verbose=False)
    assert not np.allclose(forward["a"], reverse["a"])


# ---------------------------------------------------------------------------
# What it refuses
# ---------------------------------------------------------------------------

def test_cholesky_is_refused_by_name():
    """These matrices are rank deficient by construction; a factor is a fiction."""
    with pytest.raises(ValueError, match="Cholesky is refused"):
        draw_samples([psd(6, 3, 0)], 8, decomposition_method="cholesky")


def test_an_unknown_decomposition_is_refused():
    with pytest.raises(ValueError, match="must be 'svd' or 'eigen'"):
        draw_samples([psd(6, 3, 0)], 8, decomposition_method="qr")


def test_log_space_deltas_are_refused_as_meaningless():
    with pytest.raises(ValueError, match="not a thing"):
        draw_samples([psd(6, 3, 0)], 8, space="log", returns="deltas")


def test_a_non_square_block_is_refused_naming_the_key():
    with pytest.raises(ValueError, match=r"block 'bad'.*square"):
        draw_samples({"bad": np.zeros((3, 4))}, 8)


def test_eigen_and_svd_agree_on_the_covariance_they_realise():
    """Two routes to the same factor; a disagreement means one has an ordering bug."""
    matrix = psd(15, 8, 9)
    _, svd = draw_samples([matrix], 8000, sampling_method="random", seed=2,
                          verbose=False)
    _, eig = draw_samples([matrix], 8000, sampling_method="random", seed=2,
                          decomposition_method="eigen", verbose=False)
    assert svd[0]["rank"] == eig[0]["rank"]
    assert eig[0]["realised_covariance_error"] < 0.1
    assert svd[0]["realised_covariance_error"] < 0.1
