"""Unit tests for kika.sampling.mf33_sampling.

Covers the pure factor-application math and slice extraction. End-to-end
``load_mf33_covariance`` and PENDF rewrite require a real ENDF tape and an
NJOY install; those are integration tests in
``test_pendf_perturbation.py`` and skip when the dependencies are absent.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.cov.cross_section_covariance import CrossSectionCovariance
from kika.sampling.mf33_sampling import (
    extract_mt_param_blocks,
    perturb_pointwise_xs,
)


# ---------------------------------------------------------------------------
# perturb_pointwise_xs — piecewise-constant factor application
# ---------------------------------------------------------------------------

def test_perturb_pointwise_xs_identity_block_is_noop():
    bins = np.array([1.0, 10.0, 100.0, 1000.0])  # 3 groups
    block = np.ones(3)
    energies = np.array([2.0, 50.0, 200.0, 500.0])
    xs = np.array([5.0, 4.0, 3.0, 2.0])

    xs_new, factors, frac_out = perturb_pointwise_xs(energies, xs, block, bins)

    np.testing.assert_array_equal(xs_new, xs)
    np.testing.assert_array_equal(factors, np.ones_like(energies))
    assert frac_out == 0.0


def test_perturb_pointwise_xs_applies_per_bin_factor():
    bins = np.array([1.0, 10.0, 100.0, 1000.0])  # 3 groups: [1,10), [10,100), [100,1000)
    block = np.array([1.10, 1.50, 0.80])
    energies = np.array([2.0, 9.99, 10.0, 50.0, 99.0, 100.0, 500.0])
    xs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    xs_new, factors, frac_out = perturb_pointwise_xs(energies, xs, block, bins)

    # searchsorted(side='right') - 1 → right-open bins:
    #   E=2.0 → bin 0 (factor 1.10)
    #   E=9.99 → bin 0 (factor 1.10)
    #   E=10.0 → bin 1 (factor 1.50)
    #   E=50.0 → bin 1 (factor 1.50)
    #   E=99.0 → bin 1 (factor 1.50)
    #   E=100.0 → bin 2 (factor 0.80)
    #   E=500.0 → bin 2 (factor 0.80)
    expected = np.array([1.10, 1.10, 1.50, 1.50, 1.50, 0.80, 0.80])
    np.testing.assert_allclose(factors, expected)
    np.testing.assert_allclose(xs_new, xs * expected)
    assert frac_out == 0.0


def test_perturb_pointwise_xs_below_coverage_uses_factor_one():
    bins = np.array([10.0, 100.0, 1000.0])  # 2 groups, lowest at 10
    block = np.array([2.0, 3.0])
    energies = np.array([1.0, 5.0, 50.0, 500.0])
    xs = np.array([1.0, 1.0, 1.0, 1.0])

    xs_new, factors, frac_out = perturb_pointwise_xs(energies, xs, block, bins)

    # E=1.0, 5.0 below first edge → factor=1
    np.testing.assert_allclose(factors, [1.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(xs_new, [1.0, 1.0, 2.0, 3.0])
    assert frac_out == pytest.approx(0.5)


def test_perturb_pointwise_xs_above_coverage_uses_factor_one():
    bins = np.array([1.0, 10.0])  # 1 group, highest at 10
    block = np.array([2.5])
    energies = np.array([2.0, 9.0, 10.0, 50.0])
    xs = np.array([1.0, 1.0, 1.0, 1.0])

    xs_new, factors, frac_out = perturb_pointwise_xs(energies, xs, block, bins)

    # E=10.0, 50.0 at/above last edge → factor=1
    np.testing.assert_allclose(factors, [2.5, 2.5, 1.0, 1.0])
    assert frac_out == pytest.approx(0.5)


def test_perturb_pointwise_xs_size_mismatch_raises():
    bins = np.array([1.0, 10.0, 100.0])  # 2 groups
    block = np.array([1.0, 1.0, 1.0])    # 3 factors
    with pytest.raises(ValueError, match="implies"):
        perturb_pointwise_xs(np.array([5.0]), np.array([1.0]), block, bins)


def test_perturb_pointwise_xs_zero_xs_passes_through():
    bins = np.array([1.0, 10.0])
    block = np.array([5.0])
    energies = np.array([2.0, 5.0])
    xs = np.array([0.0, 1.0])  # threshold reaction starts at zero

    xs_new, _, _ = perturb_pointwise_xs(energies, xs, block, bins)

    np.testing.assert_array_equal(xs_new, [0.0, 5.0])


# ---------------------------------------------------------------------------
# extract_mt_param_blocks — slicing into the flat factors vector
# ---------------------------------------------------------------------------

def _make_unified_cov(zaid: int, mts, n_groups: int) -> CrossSectionCovariance:
    edges = np.linspace(1.0, 100.0, n_groups + 1).tolist()
    cov = CrossSectionCovariance(num_groups=n_groups, energy_grid=edges)
    block = np.eye(n_groups) * 0.01
    for mt in mts:
        cov.add_matrix(zaid, mt, zaid, mt, block.copy(),
                       energy_grid=edges, is_relative=True)
    return cov


def test_extract_mt_param_blocks_orders_by_param_pairs():
    cov = _make_unified_cov(zaid=2056, mts=[2, 4, 102], n_groups=5)
    blocks = extract_mt_param_blocks(cov)

    assert set(blocks) == {2, 4, 102}
    # All slices have the same length (= num_groups)
    for sl in blocks.values():
        assert sl.stop - sl.start == 5

    # Slices partition [0, 3*5=15) without overlap or gap
    starts = sorted(sl.start for sl in blocks.values())
    stops = sorted(sl.stop for sl in blocks.values())
    assert starts == [0, 5, 10]
    assert stops == [5, 10, 15]


def test_extract_mt_param_blocks_correct_indexing_into_flat_factors():
    cov = _make_unified_cov(zaid=2056, mts=[2, 102], n_groups=4)
    blocks = extract_mt_param_blocks(cov)

    # Build a synthetic flat factors vector that encodes (mt, group)
    flat = np.zeros(8)
    flat[blocks[2]] = [10, 11, 12, 13]
    flat[blocks[102]] = [20, 21, 22, 23]

    np.testing.assert_array_equal(flat[blocks[2]], [10, 11, 12, 13])
    np.testing.assert_array_equal(flat[blocks[102]], [20, 21, 22, 23])


# ---------------------------------------------------------------------------
# _resolve_mt_request — positive/negative MT semantics with MT_GROUPS expansion
# ---------------------------------------------------------------------------

from kika.sampling.pendf_perturbation import _resolve_mt_request


def test_resolve_mt_request_empty_returns_all_in_mf33():
    resolved, log = _resolve_mt_request([], [1, 2, 4, 102])
    assert resolved == [1, 2, 4, 102]
    assert log == []


def test_resolve_mt_request_positives_only():
    resolved, log = _resolve_mt_request([2, 102], [1, 2, 4, 16, 51, 102])
    # Resolver does NOT filter to MF33-presence — that happens later in
    # load_mf33_covariance. It only resolves positive/negative semantics.
    assert resolved == [2, 102]
    assert log == []


def test_resolve_mt_request_negative_composite_expands_via_mt_groups():
    resolved, log = _resolve_mt_request([-4], [1, 2, 4, 51, 52, 91, 102])
    # -4 should drop 4 AND 51..91 (inelastic levels)
    assert 4 not in resolved
    assert 51 not in resolved
    assert 91 not in resolved
    assert 1 in resolved and 2 in resolved and 102 in resolved
    assert any("composite expansion" in line for line in log)


def test_resolve_mt_request_mixed_positives_and_negatives():
    resolved, log = _resolve_mt_request([4, -52, -53], [1, 2, 4, 51, 52, 53, 102])
    # User said: perturb MT4, except MT52 and MT53 specifically.
    # Negatives operate on the positive set, so 52 and 53 are removed
    # (though they weren't there to begin with — only MT4 is positive).
    assert resolved == [4]
    assert log and "Excluding MTs: [52, 53]" in log[0]


def test_resolve_mt_request_negative_only_drops_from_full_mf33():
    resolved, log = _resolve_mt_request([-102], [1, 2, 4, 102])
    assert resolved == [1, 2, 4]
    assert "Excluding MTs: [102]" in log[0]


def test_resolve_mt_request_dedup_and_overlap():
    # If the user passes 2 and -2, it should be empty (positive cancelled).
    resolved, _ = _resolve_mt_request([2, 2, -2], [1, 2, 4])
    assert resolved == []


# ---------------------------------------------------------------------------
# Composite expansion in apply_factors_to_pendf_mf3
# ---------------------------------------------------------------------------

def test_composite_partials_table_matches_mt_groups():
    """The internal composite map must equal MT_GROUPS verbatim — guards
    against accidental drift if MT_GROUPS is extended."""
    from kika._constants import MT_GROUPS
    from kika.sampling.mf33_sampling import _COMPOSITE_PARTIALS

    expected = {int(c): frozenset(int(p) for p in rng) for c, rng in MT_GROUPS}
    assert _COMPOSITE_PARTIALS == expected
    assert 4 in _COMPOSITE_PARTIALS  # total inelastic
    assert 51 in _COMPOSITE_PARTIALS[4]
    assert 91 in _COMPOSITE_PARTIALS[4]
    assert 103 in _COMPOSITE_PARTIALS  # (n,p)
    assert 600 in _COMPOSITE_PARTIALS[103]
    assert 649 in _COMPOSITE_PARTIALS[103]
