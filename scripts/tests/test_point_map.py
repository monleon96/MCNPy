"""`PointMap` is ONE map, so its two code paths must be one answer.

The class exists because five hand-written copies of "put a point on a grid"
drifted (roadmap §10.7-7). It earns that only if the fast path it dispatches to
for one-hot pairs is *exactly* the dense product — otherwise it has replaced
five copies with two.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.point_map import PointMap

GRID = np.array([1.0e6, 2.0e6, 3.0e6, 4.0e6, 5.0e6])


@pytest.fixture
def rng():
    return np.random.default_rng(20260808)


def _sym(rng, n):
    a = rng.normal(size=(n, n))
    return a @ a.T


# ── the dispatch is not a second implementation ──────────────────────────────

def test_the_one_hot_fast_path_is_exactly_the_dense_product(rng):
    e = rng.uniform(GRID[0], GRID[-1], 40)
    pm = PointMap.nearest(GRID, e)
    mat = _sym(rng, pm.n_bins)
    np.testing.assert_allclose(
        pm.sandwich(mat), pm.dense() @ mat @ pm.dense().T, rtol=0, atol=1e-13)


def test_the_fast_path_is_exact_with_points_off_the_grid(rng):
    """The masking is where the two paths could most easily disagree."""
    e = rng.uniform(0.2e6, 7.0e6, 60)          # deliberately spills both ends
    pm = PointMap.nearest(GRID, e)
    assert 0 < pm.n_off_grid < e.size, "the fixture must straddle both edges"
    mat = _sym(rng, pm.n_bins)
    np.testing.assert_allclose(
        pm.sandwich(mat), pm.dense() @ mat @ pm.dense().T, rtol=0, atol=1e-13)


@pytest.mark.parametrize("pad", [0.3e6, 2.5e6])
def test_the_mixed_gather_is_exactly_the_dense_product(rng, pad):
    """⚑ A cross block pairs an overlap map with a nearest map, and the one-hot
    leg is a column gather — so the second matmul must be skipped, EXACTLY.

    Not a speed test dressed as a correctness test. The triple product costs an
    (N, M') temporary plus an (N, M') @ (M', N') matmul; at the shipped geometry
    that is 28631 x 703 x 28631 = 5.8e11 flops with three 6.6 GB temporaries,
    and it OOM-killed a 12-core box. Skipping it is only legitimate if the
    answer does not move by one bit, which is why `atol=0`: every term the
    gather drops is exactly ``0.0 * A``.

    `pad = 2.5e6` widens the window past both ends of the grid so rows are
    partly and wholly uncovered — the masking branch, which is where an
    off-by-one would hide.
    """
    e = rng.uniform(GRID[0], GRID[-1], 30)
    pm_mag = PointMap.overlap(GRID, e - pad, e + pad)
    pm_shape = PointMap.nearest(GRID, e)
    mat = rng.normal(size=(pm_mag.n_bins, pm_shape.n_bins))

    np.testing.assert_array_equal(
        pm_mag.sandwich(mat, pm_shape),
        pm_mag.dense() @ mat @ pm_shape.dense().T)
    # ... and the other orientation, which the transpose leg of the cross term
    # exercises.
    np.testing.assert_array_equal(
        pm_shape.sandwich(mat.T, pm_mag),
        pm_shape.dense() @ mat.T @ pm_mag.dense().T)


def test_the_mixed_gather_masks_points_off_the_one_hot_grid(rng):
    """Off-grid on the ONE-HOT leg must still be an all-zero column.

    `PointMap.nearest` clips its index, so without the mask a point below the
    grid would silently borrow the first bin — §10.7-6's JEFF (2,6) defect,
    which fabricated Cov(a_2, a_6) for every point under 1 MeV.
    """
    e = np.array([0.5e6, 2.5e6, 9.0e6])          # outside, inside, outside
    pm_mag = PointMap.overlap(GRID, e - 0.2e6, e + 0.2e6)
    pm_shape = PointMap.nearest(GRID, e)
    mat = rng.normal(size=(pm_mag.n_bins, pm_shape.n_bins))
    out = pm_mag.sandwich(mat, pm_shape)
    assert not out[:, 0].any() and not out[:, 2].any()
    assert out[1, 1] != 0.0


def test_sandwich_diag_matches_the_full_sandwich(rng):
    for pm in (PointMap.nearest(GRID, rng.uniform(0.5e6, 6.0e6, 25)),
               PointMap.overlap(GRID, np.array([1.1e6, 2.4e6, 0.1e6]),
                                np.array([2.9e6, 3.1e6, 0.4e6]))):
        mat = _sym(rng, pm.n_bins)
        np.testing.assert_allclose(
            pm.sandwich_diag(mat), np.diag(pm.sandwich(mat)),
            rtol=0, atol=1e-13)


# ── the semantics the fold depends on ────────────────────────────────────────

def test_off_grid_is_an_all_zero_row_not_the_edge_interval():
    """⚑ §10.7-6, as an identity. Pinning an off-grid point to the first or last
    interval is what fabricated `Cov(a_2, a_6)` for JEFF in every chi2 from run
    82 to 90."""
    e = np.array([0.5e6, 1.5e6, 9.9e6])        # below, inside, above
    P = PointMap.nearest(GRID, e).dense()
    assert P[0].sum() == 0.0 and P[2].sum() == 0.0
    assert P[1, 0] == 1.0

    W = PointMap.overlap(GRID, np.array([0.1e6]), np.array([0.4e6])).dense()
    assert W.sum() == 0.0


def test_a_covered_row_sums_to_one_under_both_rules():
    """Both rules are averages, so a covered point must not rescale the value."""
    e = np.array([1.2e6, 3.7e6])
    assert np.allclose(PointMap.nearest(GRID, e).dense().sum(axis=1), 1.0)
    W = PointMap.overlap(GRID, e - 0.4e6, e + 0.4e6).dense()
    assert np.allclose(W.sum(axis=1), 1.0)


def test_overlap_is_length_weighted():
    """A window covering 1/4 of bin 0 and 3/4 of bin 1 weights them 1:3."""
    W = PointMap.overlap(GRID, np.array([1.75e6]), np.array([2.75e6])).dense()
    np.testing.assert_allclose(W[0, :2], [0.25, 0.75], atol=1e-12)


def test_nesting_makes_the_two_rules_agree():
    """§10.7-2's measured claim, as a test: when the window sits inside one bin
    the overlap average IS the containing-bin lookup. This is why the *rule*
    half of the defect contributed exactly nothing and the fix went to the grid.
    """
    e = np.array([1.5e6, 2.5e6, 3.5e6, 4.5e6])
    W = PointMap.overlap(GRID, e - 0.1e6, e + 0.1e6).dense()
    S = PointMap.nearest(GRID, e).dense()
    np.testing.assert_allclose(W, S, atol=1e-12)


# ── guards ───────────────────────────────────────────────────────────────────

def test_a_matrix_that_does_not_match_the_maps_is_rejected(rng):
    pm = PointMap.nearest(GRID, rng.uniform(GRID[0], GRID[-1], 5))
    with pytest.raises(ValueError, match="expected"):
        pm.sandwich(np.zeros((pm.n_bins + 1, pm.n_bins)))


def test_an_empty_grid_contributes_nothing():
    pm = PointMap.nearest(np.array([1.0e6]), np.array([1.0e6, 2.0e6]))
    assert pm.n_bins == 0
    assert pm.sandwich(np.zeros((0, 0))).shape == (2, 2)
    assert not pm.sandwich(np.zeros((0, 0))).any()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
