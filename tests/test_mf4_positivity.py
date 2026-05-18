"""Tests for kika.sampling.mf4_positivity."""

import numpy as np
import pytest

from kika.sampling.mf4_positivity import (
    check_mf4_positivity,
    project_mf4_to_positive,
)


def _sigma_un_normalized(a_endf: np.ndarray, n: int = 1001) -> np.ndarray:
    """Evaluate sum_l (2l+1) * a_l * P_l(mu) on a dense grid."""
    from numpy.polynomial.legendre import legval
    b = (2 * np.arange(len(a_endf)) + 1) * a_endf
    return legval(np.linspace(-1.0, 1.0, n), b)


def test_already_positive_returns_unchanged():
    a = np.array([1.0, 0.1, 0.05])
    is_pos, sigma_min = check_mf4_positivity(a)
    assert is_pos
    assert sigma_min >= 0.0
    projected = project_mf4_to_positive(a)
    np.testing.assert_array_equal(projected, a)


def test_forward_peaked_violates_positivity_at_backscatter():
    # a_0 = 1, a_1 = 0.9 → at mu = -1, f ∝ 1 + 3*0.9*(-1) = -1.7 < 0.
    a = np.array([1.0, 0.9, 0.0])
    is_pos, sigma_min = check_mf4_positivity(a)
    assert not is_pos
    assert sigma_min < 0.0

    projected = project_mf4_to_positive(a, n_points=101)

    # a_0 must remain pinned to 1.0.
    assert projected[0] == pytest.approx(1.0, abs=1e-10)

    # Projected vector satisfies positivity on the constraint grid exactly,
    # and on a denser grid up to small between-grid leakage.
    sigma_grid = _sigma_un_normalized(projected, n=101)
    assert sigma_grid.min() >= -1e-7
    sigma_dense = _sigma_un_normalized(projected, n=1001)
    # Allow tiny between-grid negativity (numerical artefact of a
    # finite-grid SLSQP constraint), bounded relative to sigma_max.
    assert sigma_dense.min() >= -1e-3 * sigma_dense.max()


def test_frozen_indices_are_pinned():
    a = np.array([1.0, 0.9, 0.2, 0.05, 0.01])
    projected = project_mf4_to_positive(
        a, n_points=101, frozen_indices={4: 0.123}
    )
    assert projected[0] == pytest.approx(1.0, abs=1e-10)
    assert projected[4] == pytest.approx(0.123, abs=1e-10)


def test_endf_form_matches_raw_c_form():
    """Sanity vs the eval pipeline's projector. Build an a-vector that needs
    projection, then verify that projecting in a-space with the (2l+1)
    weighting gives the same physical distribution as projecting in raw
    c-space using the existing scripts/resample_AD.py projector."""
    pytest.importorskip("scipy.optimize")

    a = np.array([1.0, 0.95, 0.4, 0.1])
    # Make sure this is non-physical so projection actually fires.
    is_pos, _ = check_mf4_positivity(a)
    assert not is_pos

    a_star = project_mf4_to_positive(a, n_points=101)

    # Build equivalent raw c-form (b_l = (2l+1) * a_l) and project there.
    # The eval pipeline's project_to_positive_distribution is the reference.
    from scripts.resample_AD import project_to_positive_distribution
    weights = 2 * np.arange(len(a)) + 1
    b = weights * a
    b_star = project_to_positive_distribution(b, n_points=101)

    # Both projections should yield equivalent physical distributions on
    # the un-normalized polynomial. Compare sigma(mu) reconstructed from each.
    from numpy.polynomial.legendre import legval
    mu = np.linspace(-1.0, 1.0, 257)
    sigma_a = legval(mu, weights * a_star)
    sigma_b = legval(mu, b_star)

    # Both must be non-negative (within between-grid SLSQP leakage,
    # bounded relative to sigma_max — the projector is exact on its
    # own 101-point constraint grid but the comparison grid here is denser).
    assert sigma_a.min() >= -1e-3 * sigma_a.max()
    assert sigma_b.min() >= -1e-3 * sigma_b.max()

    # Compare shapes — the a-space projection (weighted L²) and c-space
    # projection (unweighted L²) minimize different norms, so they don't
    # land on the same point. They should still produce qualitatively
    # similar non-negative distributions (peak heights within ~25%).
    rel_diff = np.abs(sigma_a - sigma_b) / (np.abs(sigma_b).max() + 1e-12)
    assert rel_diff.max() < 0.25


def test_projection_preserves_low_order_when_possible():
    """If only the high-order tail makes the vector unphysical, freezing
    a_0 (always) and not adding extra pins should still allow the projector
    to fix it."""
    # Construct: a_0=1, a_1=0 (isotropic baseline), a_3 = 0.5 — the
    # backward-pointing P_3 dip at mu ≈ -0.78 plus 7 * 0.5 = 3.5 weight
    # drops f below zero.
    a = np.array([1.0, 0.0, 0.0, 0.5])
    is_pos, _ = check_mf4_positivity(a)
    assert not is_pos
    projected = project_mf4_to_positive(a, n_points=101)
    sigma = _sigma_un_normalized(projected, n=1001)
    assert sigma.min() >= -1e-3 * sigma.max()
    assert projected[0] == pytest.approx(1.0, abs=1e-10)
