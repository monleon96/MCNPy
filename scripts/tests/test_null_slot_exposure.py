"""Tests for the unsupported-parameter exposure diagnostic (roadmap §10.6-1).

The whole measurement rests on ONE property: `null_slot_exposure.mf34_diag` must
reproduce the diagonal of `eval_covariance.build_mf34_block` exactly. It is a
reimplementation written for memory (Cierjacks alone would be a 28631² block),
so it is a second copy of the fold and the only thing standing between it and
silent drift is this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kika.cov.legendre_covariance import LegendreCovariance

from scripts.eval_covariance import build_eval_cov_for_groups, build_mf34_block
from scripts.null_slot_exposure import L_MAX, mf34_diag


def _mf34(grid_ev, blocks, is_relative=True):
    """MF34 from {(l_row, l_col): matrix}, all on one grid."""
    cov = LegendreCovariance()
    for (l_r, l_c), mat in blocks.items():
        cov.add_matrix(
            isotope_row=26056, reaction_row=2, l_row=l_r,
            isotope_col=26056, reaction_col=2, l_col=l_c,
            matrix=np.asarray(mat, float), energy_grid=list(grid_ev),
            is_relative=is_relative, frame="LAB",
        )
    return cov


@pytest.fixture
def case():
    rng = np.random.default_rng(20260806)
    grid_ev = np.array([1.0e6, 2.0e6, 3.0e6, 4.0e6, 5.0e6])
    M = len(grid_ev) - 1

    def sym():
        a = rng.normal(size=(M, M))
        return a @ a.T

    # (1,2) is deliberately NOT symmetric: the l_r != l_c term is the one where
    # build_mf34_block adds the companion block's transpose, and the diagonal
    # shortcut (block.T[j,j] == block[j,j], hence a factor 2) has to hold there.
    blocks = {(1, 1): sym(), (1, 2): rng.normal(size=(M, M)), (2, 2): sym()}
    mf34 = _mf34(grid_ev, blocks)

    N = 25
    e_mev = rng.uniform(0.5, 5.5, N)          # spans past both grid ends: clamping
    mu = rng.uniform(-1.0, 1.0, N)
    c0 = rng.uniform(0.5, 2.0, N)
    a_l_per_pt = rng.normal(size=(N, L_MAX))
    # grid_ev and blocks come back too: `energy_grids` stores whatever it was
    # given (eV here, since that is what build_mf34_block searchsorts against),
    # and round-tripping through the object to rebuild a comparison case is how
    # a unit slip gets in.
    return mf34, e_mev, mu, c0, a_l_per_pt, grid_ev, blocks


def test_matches_build_mf34_block_diagonal(case):
    """The invariant the measurement depends on."""
    mf34, e_mev, mu, c0, a_l, _, _ = case
    ref = np.diag(build_mf34_block(mf34, e_mev, mu, c0, a_l))
    got, _ = mf34_diag(mf34, e_mev, mu, c0, a_l)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0.0)


def test_matches_when_absolute(case):
    """Same, with is_relative=False — the a_l scaling must switch off in step."""
    _, e_mev, mu, c0, a_l, grid_ev, blocks = case
    absolute = _mf34(grid_ev, blocks, is_relative=False)
    ref = np.diag(build_mf34_block(absolute, e_mev, mu, c0, a_l))
    got, _ = mf34_diag(absolute, e_mev, mu, c0, a_l)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0.0)


def test_dropping_everything_gives_zero(case):
    mf34, e_mev, mu, c0, a_l, _, _ = case
    drop = np.ones((e_mev.size, L_MAX), bool)
    got, per_order = mf34_diag(mf34, e_mev, mu, c0, a_l, drop=drop)
    assert np.all(got == 0.0)
    assert np.all(per_order == 0.0)


def test_dropping_nothing_equals_no_drop(case):
    mf34, e_mev, mu, c0, a_l, _, _ = case
    ref, _ = mf34_diag(mf34, e_mev, mu, c0, a_l)
    got, _ = mf34_diag(mf34, e_mev, mu, c0, a_l,
                       drop=np.zeros((e_mev.size, L_MAX), bool))
    np.testing.assert_array_equal(got, ref)


def test_a_term_needs_both_orders_kept(case):
    """Dropping order 2 must remove (1,2) and (2,2), and leave (1,1) untouched.

    This is the semantics of zeroing a parameter's row AND column in the
    covariance — not of zeroing only its variance.
    """
    mf34, e_mev, mu, c0, a_l, grid_ev, blocks = case
    only_11 = _mf34(grid_ev, {(1, 1): blocks[(1, 1)]})

    drop = np.zeros((e_mev.size, L_MAX), bool)
    drop[:, 1] = True                                    # order 2 unsupported
    got, _ = mf34_diag(mf34, e_mev, mu, c0, a_l, drop=drop)
    ref, _ = mf34_diag(only_11, e_mev, mu, c0, a_l)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0.0)


def test_drop_is_per_point(case):
    """A drop on half the points leaves the other half bit-identical."""
    mf34, e_mev, mu, c0, a_l, _, _ = case
    full, _ = mf34_diag(mf34, e_mev, mu, c0, a_l)
    drop = np.zeros((e_mev.size, L_MAX), bool)
    drop[::2, :] = True
    got, _ = mf34_diag(mf34, e_mev, mu, c0, a_l, drop=drop)
    assert np.all(got[::2] == 0.0)
    np.testing.assert_array_equal(got[1::2], full[1::2])


# ── the production path: build_mf34_block's own `drop` (roadmap §10.6-1) ──────

def test_block_drop_matches_the_diagnostic(case):
    """The re-score must remove exactly what the diagnostic sized.

    `null_slot_exposure` measured a SIZE off its own diagonal helper; the
    re-score removes it inside `build_mf34_block`. If those two disagree, the
    chi2 answers a different question from the one that motivated it.
    """
    mf34, e_mev, mu, c0, a_l, _, _ = case
    rng = np.random.default_rng(5)
    drop = rng.random((e_mev.size, L_MAX)) < 0.3
    ref, _ = mf34_diag(mf34, e_mev, mu, c0, a_l, drop=drop)
    got = np.diag(build_mf34_block(mf34, e_mev, mu, c0, a_l, drop=drop))
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0.0)


def test_block_drop_none_is_the_old_behaviour(case):
    """Default None must leave every existing number bit-identical."""
    mf34, e_mev, mu, c0, a_l, _, _ = case
    ref = build_mf34_block(mf34, e_mev, mu, c0, a_l)
    got = build_mf34_block(mf34, e_mev, mu, c0, a_l, drop=None)
    np.testing.assert_array_equal(got, ref)
    zeros = build_mf34_block(mf34, e_mev, mu, c0, a_l,
                             drop=np.zeros((e_mev.size, L_MAX), bool))
    np.testing.assert_array_equal(zeros, ref)


def test_block_drop_removes_rows_and_columns(case):
    """Off the diagonal too — a dropped point couples to nothing, either way.

    The diagonal helper can only ever check `drop[j] or drop[j]`; this is the
    half of the semantics it structurally cannot see.
    """
    mf34, e_mev, mu, c0, a_l, _, _ = case
    N = e_mev.size
    drop = np.zeros((N, L_MAX), bool)
    drop[3, :] = True
    S = build_mf34_block(mf34, e_mev, mu, c0, a_l, drop=drop)
    assert np.all(S[3, :] == 0.0), "the dropped point's row must vanish"
    assert np.all(S[:, 3] == 0.0), "and its column"
    keep = [i for i in range(N) if i != 3]
    ref = build_mf34_block(mf34, e_mev, mu, c0, a_l)
    np.testing.assert_allclose(S[np.ix_(keep, keep)], ref[np.ix_(keep, keep)],
                               rtol=1e-12, atol=0.0)


def test_eval_cov_honours_the_library_mask(case):
    """`build_eval_cov_for_groups` must place the mask the consumer's way.

    The mask is per (shape group, order); the fold places each point by
    `searchsorted(grid, e) - 1`, clamped. This asserts the orchestrator does the
    same, because a half-bin offset here would silently mask the wrong groups.
    """
    import pandas as pd

    mf34, e_mev, mu, c0, a_l, grid_ev, _ = case
    N = e_mev.size
    df = pd.DataFrame({
        "library": ["This_work"] * N, "experiment_id": ["X"] * N,
        "energy_mev": e_mev, "mu": mu, "c0": c0, "y_eval": np.ones(N),
    })
    n_g = grid_ev.size - 1
    mask = np.zeros((n_g, L_MAX), bool)
    mask[1, 0] = True                      # group 1 unsupported at order 1
    lib = {"mf34": mf34, "mf34_null_mask": mask, "mf34_null_grid_ev": grid_ev}

    out = build_eval_cov_for_groups(
        df, {"This_work": lib}, lambda k, l, e: a_l[np.argmin(np.abs(e_mev - e))],
        l_max=L_MAX)
    got = np.diag(out[("This_work", "X")])

    g_pt = np.clip(np.searchsorted(grid_ev, e_mev * 1e6, side="right") - 1,
                   0, n_g - 1)
    ref, _ = mf34_diag(mf34, e_mev, mu, c0, a_l, drop=mask[g_pt])
    assert (g_pt == 1).any(), "the fixture must actually hit the masked group"
    np.testing.assert_allclose(got, ref.astype(np.float32), rtol=1e-5, atol=0.0)
