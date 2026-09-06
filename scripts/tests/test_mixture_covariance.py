"""Phase 3 — the per-bin mixture covariance.

The evaluation reported model-degree uncertainty as exactly zero. Two
mechanisms caused that:

  1. the MC drew one degree per sample and pooled the results, so a_6 (median
     inclusion probability 0.089) was reconstructed from ~9 % of the samples;
  2. the valid-parameter mask keyed on ``frozen_degree`` — the WINNER's degree —
     and hard-zeroed every order above it. Measured on the tau-GLS run this is
     the dominant one: it keeps 62.6 % of slots and zeroes 2738 (26 % of all)
     whose inclusion probability exceeds 0.10.

Phase 3 replaces both with the law of total covariance over the IC weights. The
tests here pin the arithmetic, the mask rule, and — most importantly — that the
near-zero guard still fires on the new path. That guard is the only active
protection against relative-sigma blow-up in the production config, and the
mixture path would bypass it if the moments were not routed through
``compute_covariance_from_samples``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.model_averaging import (  # noqa: E402
    mixture_moments, nested_masks, inclusion_probabilities,
)
from scripts.exfor_utils import compute_covariance_from_samples  # noqa: E402
from scripts.exfor_to_endf_research import (  # noqa: E402
    _stratified_counts, bin_valid_orders, endf_normalize_legendre_coeffs,
)


class _NR:
    """Minimal stand-in for NominalFitResult — bin_valid_orders only reads two
    attributes, and building a real one would drag in the whole fit pipeline."""

    def __init__(self, frozen_degree, degree_weights):
        self.frozen_degree = frozen_degree
        self.degree_weights = degree_weights


# --------------------------------------------------------------------------
# The arithmetic
# --------------------------------------------------------------------------

def test_degenerate_weights_reproduce_the_single_model() -> None:
    """w = (1, 0, ...) ⇒ the mixture IS that model. The anchor: if this fails
    nothing else in the phase is trustworthy."""
    rng = np.random.default_rng(0)
    means = rng.normal(0, 1, (3, 6))
    A = rng.normal(0, 1, (3, 6, 6))
    covs = np.einsum('kij,klj->kil', A, A)  # PSD by construction
    out = mixture_moments([1.0, 0.0, 0.0], means, covs)
    np.testing.assert_allclose(out['mean'], means[0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(out['total'], covs[0], rtol=0, atol=1e-12)
    assert np.max(np.abs(out['between'])) < 1e-12, (
        "a degenerate mixture has no between-model spread"
    )


def test_scalar_identity_q_m_and_q_s2_plus_q1mq_m2() -> None:
    """For one order shared by a subset of candidates the mixture must give
    E[a] = q·m and Var(a) = q·s² + q(1−q)·m², exactly.

    This is the identity the whole phase rests on — it is what says the
    between-model term is model-selection uncertainty and not an artefact.
    """
    q, m, s = 0.3, 2.0, 0.5
    # Two candidates: one carries the order (mean m, var s²), one does not
    # (exact zero, and zero variance — an absent order is absent, not noisy).
    weights = np.array([q, 1.0 - q])
    means = np.array([[m], [0.0]])
    covs = np.array([[[s ** 2]], [[0.0]]])
    out = mixture_moments(weights, means, covs)
    assert out['mean'][0] == pytest.approx(q * m, rel=1e-14)
    assert out['total'][0, 0] == pytest.approx(
        q * s ** 2 + q * (1 - q) * m ** 2, rel=1e-14
    )
    # And the split is the diagnostic Phase 3 reports:
    assert out['within'][0, 0] == pytest.approx(q * s ** 2, rel=1e-14)
    assert out['between'][0, 0] == pytest.approx(q * (1 - q) * m ** 2, rel=1e-14)


def test_absent_orders_are_zero_padded_not_truncated() -> None:
    """A degree-3 candidate contributes an exact 0 to a_4..a_6, and its
    *between* contribution there is q(1−q)m² — not zero.

    Truncating instead of padding is the bug that makes high orders look
    certain: the candidates that omit the order would simply not be counted.
    """
    n_out = 6
    degrees = [3, 6]
    w = np.array([0.7, 0.3])
    masks = nested_masks(degrees, n_out)
    q = inclusion_probabilities(w, masks)
    assert q[5] == pytest.approx(0.3), "only the degree-6 candidate carries a_6"
    assert q[2] == pytest.approx(1.0), "both candidates carry a_3"

    means = np.zeros((2, n_out))
    means[0, :3] = [0.4, 0.2, 0.1]      # degree 3: a_4..a_6 are exact zeros
    means[1, :6] = [0.4, 0.2, 0.1, 0.05, 0.02, 0.6]
    covs = np.zeros((2, n_out, n_out))
    out = mixture_moments(w, means, covs)
    assert out['mean'][5] == pytest.approx(0.3 * 0.6)
    assert out['between'][5, 5] == pytest.approx(0.3 * 0.7 * 0.6 ** 2)


@pytest.mark.parametrize("seed", range(8))
def test_total_is_psd(seed: int) -> None:
    rng = np.random.default_rng(seed)
    k, n = int(rng.integers(2, 6)), 6
    w = rng.dirichlet(np.ones(k))
    means = rng.normal(0, 1, (k, n))
    A = rng.normal(0, 1, (k, n, n))
    covs = np.einsum('kij,klj->kil', A, A)
    out = mixture_moments(w, means, covs)
    ev = np.linalg.eigvalsh(out['total'])
    assert ev.min() > -1e-10 * max(1.0, ev.max())


# --------------------------------------------------------------------------
# The mask — the dominant mechanism
# --------------------------------------------------------------------------

def test_mask_follows_q_not_the_winner() -> None:
    """A bin whose winner is degree 4 but whose degree-6 candidate carries real
    weight must keep a_6. Under the legacy rule it was hard-zeroed."""
    nr = _NR(frozen_degree=4, degree_weights={3: 0.2, 4: 0.5, 5: 0.2, 6: 0.1})
    assert bin_valid_orders(nr, 6) == 6
    assert min(nr.frozen_degree, 6) == 4, "the legacy rule would have said 4"


def test_mask_drops_orders_with_negligible_inclusion_probability() -> None:
    nr = _NR(frozen_degree=4, degree_weights={3: 0.3, 4: 0.699, 6: 0.001})
    assert bin_valid_orders(nr, 6) == 4


def test_mask_falls_back_to_the_winner_without_weights() -> None:
    """Interpolated bins carry no candidate set; they must not silently gain
    orders they have no evidence for."""
    assert bin_valid_orders(_NR(4, None), 6) == 4
    assert bin_valid_orders(_NR(4, {}), 6) == 4


# --------------------------------------------------------------------------
# Allocation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_total", [10, 137, 10000])
def test_stratified_counts_sum_exactly(n_total: int) -> None:
    rng = np.random.default_rng(3)
    for _ in range(20):
        p = rng.dirichlet(np.ones(int(rng.integers(2, 8))))
        out = _stratified_counts(p, n_total)
        assert out.sum() == n_total
        assert (out >= 0).all()


def test_stratified_counts_can_give_a_candidate_zero_slots() -> None:
    """A candidate with w·N < 0.5 gets no pooled slot. That is fine and must not
    raise — it still needs a batch, because its between-model contribution does
    not vanish just because it drew no samples."""
    out = _stratified_counts(np.array([0.997, 0.002, 0.001]), 100)
    assert out.sum() == 100
    assert out[2] == 0


# --------------------------------------------------------------------------
# Wiring — the guard, and inertness
# --------------------------------------------------------------------------

def _samples(n_samples, energy_indices, max_order, rng):
    return {
        s: {e: rng.normal(0.5, 0.05, max_order) for e in energy_indices}
        for s in range(n_samples)
    }


def test_mixture_blocks_none_is_a_pure_no_op() -> None:
    """Gate A in miniature: passing no mixture must leave the function's output
    bit-identical, or USE_MIXTURE_COVARIANCE=False cannot reproduce earlier
    runs."""
    rng = np.random.default_rng(11)
    eidx, mo = [0, 1, 2], 4
    smp = _samples(200, eidx, mo, rng)
    a = compute_covariance_from_samples(smp, eidx, mo, snr_threshold=0.0)
    b = compute_covariance_from_samples(smp, eidx, mo, snr_threshold=0.0,
                                        mixture_blocks=None)
    for x, y in zip(a[:2] + (a[3], a[4]), b[:2] + (b[3], b[4])):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_mixture_block_replaces_the_diagonal_block_only() -> None:
    """The listed bin's own block and mean are replaced; the other bins and
    every cross-bin block keep their sample estimates."""
    rng = np.random.default_rng(12)
    eidx, mo = [0, 1], 3
    smp = _samples(400, eidx, mo, rng)
    base = compute_covariance_from_samples(smp, eidx, mo, snr_threshold=0.0)
    mix_cov = np.diag([0.04, 0.09, 0.16])
    mix_mean = np.array([1.0, 2.0, 4.0])
    out = compute_covariance_from_samples(
        smp, eidx, mo, snr_threshold=0.0,
        mixture_blocks={0: {'mean': mix_mean, 'cov': mix_cov}},
    )
    cov_abs_base, cov_abs_new = base[4], out[4]
    np.testing.assert_allclose(cov_abs_new[:mo, :mo], mix_cov, atol=1e-12)
    # bin 1's own block untouched
    np.testing.assert_allclose(cov_abs_new[mo:, mo:], cov_abs_base[mo:, mo:],
                               atol=1e-12)
    # cross-bin block untouched
    np.testing.assert_allclose(cov_abs_new[:mo, mo:], cov_abs_base[:mo, mo:],
                               atol=1e-12)
    # mean replaced for bin 0 only
    np.testing.assert_allclose(out[3][:mo], mix_mean, atol=1e-12)
    np.testing.assert_allclose(out[3][mo:], base[3][mo:], atol=1e-12)
    # and the relative conversion used the mixture mean as denominator
    np.testing.assert_allclose(
        np.diag(out[0])[:mo], np.diag(mix_cov) / mix_mean ** 2, rtol=1e-10)


def test_near_zero_guard_fires_on_the_mixture_path() -> None:
    """The load-bearing one. The guard lives inside
    ``compute_covariance_from_samples``; routing the mixture through that
    function is what keeps it applied. A near-zero mean with real variance must
    NOT come back as a naive cov/mean² blow-up.
    """
    rng = np.random.default_rng(13)
    eidx, mo = [0, 1, 2], 3
    smp = _samples(400, eidx, mo, rng)
    # Bin 1, order 1: mean three orders of magnitude below its sigma.
    tiny = 1e-4
    blocks = {
        e: {'mean': np.array([0.5, 0.4, 0.3]), 'cov': np.diag([1e-4, 1e-4, 1e-4])}
        for e in eidx
    }
    blocks[1]['mean'] = np.array([tiny, 0.4, 0.3])
    out = compute_covariance_from_samples(
        smp, eidx, mo, snr_threshold=1.0, n_neighbors=1,
        mixture_blocks=blocks,
    )
    cov_rel = out[0]
    slot = 1 * mo + 0
    naive = 1e-4 / tiny ** 2          # = 1e4, the blow-up the guard exists for
    assert cov_rel[slot, slot] < naive / 10.0, (
        f"near-zero guard did not fire on the mixture path: relative variance "
        f"{cov_rel[slot, slot]:.3e} vs naive {naive:.3e}"
    )
    assert np.isfinite(cov_rel).all()


def test_mixture_block_shape_mismatch_is_an_error_not_a_silent_skip() -> None:
    """A 5 h run must not discover a shape bug by quietly producing the legacy
    covariance for that bin."""
    rng = np.random.default_rng(14)
    eidx, mo = [0], 3
    smp = _samples(50, eidx, mo, rng)
    with pytest.raises(ValueError, match="mixture block"):
        compute_covariance_from_samples(
            smp, eidx, mo,
            mixture_blocks={0: {'mean': np.zeros(2), 'cov': np.eye(2)}},
        )


# --------------------------------------------------------------------------
# Shipping the mixture mean
# --------------------------------------------------------------------------

def test_a_space_to_c_space_inversion_round_trips() -> None:
    """SHIP_MIXTURE_MEAN rebuilds c_l from the averaged a_l as
    c_l = a_l·(2l+1)·c_0. If that inverse is wrong the shipped MF4 is wrong in
    a way no covariance test would catch."""
    c = np.array([2.5, 0.9, 0.6, 0.2, 0.05, 0.01, 0.002])
    a = endf_normalize_legendre_coeffs(c, include_a0=False)
    back = np.zeros(7)
    back[0] = c[0]
    for l in range(1, 7):
        back[l] = a[l - 1] * (2 * l + 1) * c[0]
    np.testing.assert_allclose(back, c, rtol=0, atol=1e-14)
