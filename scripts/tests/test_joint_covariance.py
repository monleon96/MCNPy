"""`JointCov` must reproduce today's fold exactly, then make PSD unconditional.

Two jobs, in this order (roadmap §10.7-7):

1. **The equivalence gate.** With the cross block zero -- the factorization
   shipped since run 82 -- `JointCov.fold` must reproduce
   `build_mf34_block + build_mf33_block` on the nose. If it does not, the new
   object is wrong and the old numbers stand. This makes the whole refactor a
   demonstrated no-op before anything is switched on, which is the lesson of
   §L16 and §L18: one variable at a time, and prove the zero first.

2. **The payoff.** With a nonzero cross block, the legacy three-term sum is
   NOT a congruence and goes indefinite; `S C S^T` cannot. Both are asserted
   here on the same joint, because "the new way is PSD" is only interesting
   next to "the old way was not".
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval_covariance import (  # noqa: E402
    _legendre_base_sens as _legacy_base_sens,
    _mf4_bin_edges_for_points,
    build_mf33_block,
    build_mf33_mf34_cross_block,
    build_mf34_block,
)
from scripts.joint_covariance import (  # noqa: E402
    JointCov,
    assemble_a_block,
    nearest_weights,
    overlap_weights,
    width_weights,
)

L_MAX = 3
N_PTS = 40


# ── fixtures: a miniature of the real configuration ───────────────────────────

def _psd(rng, n, jitter=0.35):
    z = rng.normal(size=(n + 12, n))
    c = z.T @ z / (n + 12)
    return c + jitter * np.eye(n)


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(20260807)


@pytest.fixture(scope="module")
def setup(rng):
    """Nested MF34 grids and a separate MF33 grid, as the shipped file has.

    The a_l blocks sit on three grids of 8/4/2 bins over the same span plus one
    SHORT block that starts late -- JEFF's (2, 6) in miniature, so the off-grid
    rule is exercised by the equivalence test and not only by its own.
    """
    lo, hi = 0.85e6, 4.0e6
    g_fine = np.linspace(lo, hi, 9)
    g_mid = g_fine[::2]
    g_coarse = g_fine[::4]
    g_short = g_fine[4:]                      # starts at the midpoint

    grids, l_rows, l_cols, mats = [], [], [], []
    for (lr, lc), g in [((1, 1), g_fine), ((2, 2), g_mid), ((3, 3), g_coarse),
                        ((1, 2), g_mid), ((1, 3), g_coarse), ((2, 3), g_short)]:
        m = g.size - 1
        a = rng.normal(size=(m, m)) * 0.02
        mats.append(a + a.T if lr != lc else _psd(rng, m) * 0.02)
        grids.append(g)
        l_rows.append(lr)
        l_cols.append(lc)
    mf34 = SimpleNamespace(
        matrices=mats, l_rows=l_rows, l_cols=l_cols, energy_grids=grids,
        is_relative=[True] * len(mats),
    )

    grid_sigma = np.linspace(lo, hi, 13)       # 12 bins, its own grid
    c33 = _psd(rng, grid_sigma.size - 1) * 0.004

    # MF4 grid the magnitude is averaged over, and the points themselves.
    energies_mf4_mev = np.linspace(lo, hi, 25) / 1e6
    e_mev = rng.uniform(lo, hi, N_PTS) / 1e6
    mu = rng.uniform(-1.0, 1.0, N_PTS)
    c0 = rng.uniform(0.05, 0.6, N_PTS)
    a_l = rng.uniform(-0.4, 0.9, (N_PTS, L_MAX))
    y = c0 * (1.0 + rng.uniform(-0.2, 0.2, N_PTS))
    return SimpleNamespace(
        mf34=mf34, grid_sigma=grid_sigma, c33=c33,
        energies_mf4_mev=energies_mf4_mev,
        e_mev=e_mev, mu=mu, c0=c0, a_l=a_l, y=y, lo=lo, hi=hi,
    )


@pytest.fixture(scope="module")
def psd_setup(rng):
    """A genuinely PSD joint, on ONE a_l grid.

    Separate from `setup` on purpose. `setup`'s six blocks on four grids are
    what the equivalence tests need -- a miniature of the shipped file, and
    like it, not PSD once the off-diagonal (L, L1) blocks are drawn freely.
    A covariance that is PSD **cannot** be split across grids of different
    resolution and put back together, so the two properties cannot live in one
    fixture, and the tests that need PSD say so by using this one.
    """
    lo, hi = 0.85e6, 4.0e6
    grid_a = np.linspace(lo, hi, 7)                 # 6 bins x L_MAX orders
    c34 = _psd(rng, (grid_a.size - 1) * L_MAX) * 0.02
    grid_sigma = np.linspace(lo, hi, 13)
    c33 = _psd(rng, grid_sigma.size - 1) * 0.004
    # HALF A BIN OUT OF STEP, ON PURPOSE. `W` only differs from a containing-bin
    # lookup where the MF4 bin straddles a magnitude boundary; align the two
    # grids, or make either far finer than the other, and W collapses to
    # one-hot, the two maps agree and this fixture stops exercising the defect.
    # The shipped file is in exactly the straddling regime: 1735 MF4 points and
    # 1738 MF33 bins over the same span, on different edges.
    dw = (hi - lo) / 12.0
    energies_mf4_mev = (np.linspace(lo, hi, 13) + 0.5 * dw) / 1e6
    e_mev = rng.uniform(lo, hi, N_PTS) / 1e6
    return SimpleNamespace(
        mf34=_mf34_like(c34, grid_a, L_MAX), grid_a=grid_a, c34=c34,
        grid_sigma=grid_sigma, c33=c33, energies_mf4_mev=energies_mf4_mev,
        e_mev=e_mev, mu=rng.uniform(-1.0, 1.0, N_PTS),
        c0=rng.uniform(0.05, 0.6, N_PTS),
        a_l=rng.uniform(-0.4, 0.9, (N_PTS, L_MAX)),
        y=rng.uniform(0.05, 0.6, N_PTS), lo=lo, hi=hi,
    )


def _mf34_like(c34, grid, l_max):
    """Split a joint a_l block into the upper-triangle (L, L1) blocks ENDF stores."""
    m = grid.size - 1
    mats, l_rows, l_cols, grids = [], [], [], []
    for lr in range(1, l_max + 1):
        for lc in range(lr, l_max + 1):
            rows = np.arange(m) * l_max + (lr - 1)
            cols = np.arange(m) * l_max + (lc - 1)
            mats.append(c34[np.ix_(rows, cols)])
            l_rows.append(lr)
            l_cols.append(lc)
            grids.append(grid)
    return SimpleNamespace(matrices=mats, l_rows=l_rows, l_cols=l_cols,
                           energy_grids=grids, is_relative=[True] * len(mats))


def _joint(setup, cross=None):
    grid_a, c34, rel = assemble_a_block(setup.mf34, L_MAX)
    return JointCov.from_blocks(
        setup.c33, c34, setup.grid_sigma, grid_a, L_MAX,
        cross=cross, sigma_is_relative=True, a_is_relative=rel,
    )


def _psd_joint(psd_setup, cross=None):
    return JointCov.from_blocks(
        psd_setup.c33, psd_setup.c34, psd_setup.grid_sigma, psd_setup.grid_a,
        L_MAX, cross=cross,
    )


def _fold_kwargs(setup):
    lo_mev, hi_mev = _mf4_bin_edges_for_points(setup.energies_mf4_mev, setup.e_mev)
    return dict(sigma_window_ev=(lo_mev * 1e6, hi_mev * 1e6),
                sigma_map="overlap", a_map="nearest")


# ── 1. the equivalence gate ───────────────────────────────────────────────────

def test_the_base_sensitivity_is_the_same_function(setup):
    """Duplicated for import hygiene, so assert it is a duplicate."""
    np.testing.assert_array_equal(
        _legacy_base_sens(setup.mu, setup.c0, L_MAX),
        __import__("scripts.joint_covariance", fromlist=["_"])._legendre_base_sens(
            setup.mu, setup.c0, L_MAX),
    )


def test_fold_reproduces_the_legacy_mf34_block(setup):
    """Shape leg alone: zero the magnitude half and compare to the old code."""
    j = _joint(setup)
    j.matrix[:j.n_sigma, :] = 0.0
    j.matrix[:, :j.n_sigma] = 0.0
    got = j.fold(setup.e_mev, setup.mu, setup.c0, setup.a_l, setup.y,
                 dtype=np.float64, **_fold_kwargs(setup))
    want = build_mf34_block(setup.mf34, setup.e_mev, setup.mu, setup.c0, setup.a_l)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


def test_fold_reproduces_the_legacy_mf33_block(setup):
    """Magnitude leg alone."""
    j = _joint(setup)
    j.matrix[j.n_sigma:, :] = 0.0
    j.matrix[:, j.n_sigma:] = 0.0
    got = j.fold(setup.e_mev, setup.mu, setup.c0, setup.a_l, setup.y,
                 dtype=np.float64, **_fold_kwargs(setup))
    want = build_mf33_block(setup.grid_sigma, setup.c33,
                            setup.energies_mf4_mev, setup.e_mev, setup.y)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


def test_fold_reproduces_the_legacy_sum(setup):
    """THE GATE. Cross = 0 -> the object must be a no-op against runs 82-86."""
    j = _joint(setup)
    assert not j.has_cross
    got = j.fold(setup.e_mev, setup.mu, setup.c0, setup.a_l, setup.y,
                 dtype=np.float64, **_fold_kwargs(setup))
    want = (build_mf34_block(setup.mf34, setup.e_mev, setup.mu, setup.c0, setup.a_l)
            + build_mf33_block(setup.grid_sigma, setup.c33,
                               setup.energies_mf4_mev, setup.e_mev, setup.y))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


def test_the_null_slot_drop_still_removes_row_and_column(setup, rng):
    """§10.6-1's `drop`, which the legacy code applies in two places."""
    drop = rng.random((N_PTS, L_MAX)) < 0.3
    j = _joint(setup)
    j.matrix[:j.n_sigma, :] = 0.0
    j.matrix[:, :j.n_sigma] = 0.0
    got = j.fold(setup.e_mev, setup.mu, setup.c0, setup.a_l, setup.y,
                 dtype=np.float64, drop=drop, **_fold_kwargs(setup))
    want = build_mf34_block(setup.mf34, setup.e_mev, setup.mu, setup.c0,
                            setup.a_l, drop=drop)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


# ── 2. the payoff: PSD stops being something to check ─────────────────────────

def test_splitting_into_endf_blocks_and_back_is_exact(psd_setup):
    """`_mf34_like` -> `assemble_a_block` must be the identity, or the PSD
    fixture is testing something other than what it claims to hold."""
    grid, c34, rel = assemble_a_block(psd_setup.mf34, L_MAX)
    np.testing.assert_array_equal(grid, psd_setup.grid_a)
    np.testing.assert_allclose(c34, psd_setup.c34, atol=1e-15)
    assert rel is True


def test_both_folds_are_congruences_and_agree(psd_setup, rng):
    """Same joint, two folds, and since §10.7-7 they are the same map.

    ⚑ THIS TEST USED TO ASSERT THE OPPOSITE for the three-term sum, and the
    change is the result. It read
    `test_the_legacy_three_term_sum_breaks_psd_and_the_operator_does_not`, and
    it pinned the defect its own docstring named: the legacy path folded the
    magnitude leg of the cross term with a nearest-bin lookup while folding the
    same variable in the MF33 self block with `W`, which is not a congruence and
    it showed. Now that both legs go through `_mf33_magnitude_map`, the sum is a
    congruence too — measured, that moved lam_min/scale from < -1e-8 to
    **-1.2e-16**, i.e. from indefinite to machine zero.

    The cross block is still scaled down until the JOINT is PSD, so any
    indefiniteness in either folded Sigma would be manufactured by the fold
    rather than inherited. Neither manufactures any.
    """
    s = psd_setup
    n0, na = s.c33.shape[0], s.c34.shape[0]
    raw = rng.normal(size=(n0, na)) * 0.004

    scale = 1.0
    for _ in range(60):
        j = _psd_joint(s, cross=raw * scale)
        if j.check(eigen=True).min_eig >= 0:
            break
        scale *= 0.7
    else:                                            # pragma: no cover
        pytest.fail("could not scale the cross block into a PSD joint")
    assert j.check(eigen=True).ok

    lo_mev, hi_mev = _mf4_bin_edges_for_points(s.energies_mf4_mev, s.e_mev)
    sigma_new = j.fold(s.e_mev, s.mu, s.c0, s.a_l, s.y, dtype=np.float64,
                       sigma_window_ev=(lo_mev * 1e6, hi_mev * 1e6),
                       sigma_map="overlap", a_map="nearest")

    cross_blocks = [
        {"l": l, "shape_grid_ev": s.grid_a,
         "matrix": j.cross.reshape(n0, s.grid_a.size - 1, L_MAX)[:, :, l - 1],
         "is_relative": True}
        for l in range(1, L_MAX + 1)
    ]
    sigma_old = (
        build_mf34_block(s.mf34, s.e_mev, s.mu, s.c0, s.a_l)
        + build_mf33_block(s.grid_sigma, s.c33, s.energies_mf4_mev,
                           s.e_mev, s.y)
        + build_mf33_mf34_cross_block(cross_blocks, s.e_mev, s.mu,
                                      s.c0, s.a_l, s.y,
                                      mf33_grid_ev=s.grid_sigma,
                                      energies_mf4_mev=s.energies_mf4_mev)
    )

    scale_new = max(np.abs(np.diag(sigma_new)).max(), 1e-300)
    scale_old = max(np.abs(np.diag(sigma_old)).max(), 1e-300)
    lam_new = np.linalg.eigvalsh(sigma_new).min() / scale_new
    lam_old = np.linalg.eigvalsh(sigma_old).min() / scale_old

    assert lam_new >= -1e-10, (
        f"S C S^T is not PSD (lam_min/scale = {lam_new:.3e}); the congruence "
        f"argument is wrong or the operator is not one matrix"
    )
    assert lam_old >= -1e-10, (
        f"the three-term sum is indefinite (lam_min/scale = {lam_old:.3e}). "
        f"Its magnitude leg must have stopped sharing `_mf33_magnitude_map` "
        f"with the MF33 self block — that is §10.7-2(a) regressing."
    )
    # Stronger than "both PSD": one map means one answer. If these ever drift
    # apart, two folds exist again and the whole point has been lost.
    assert np.allclose(sigma_old, sigma_new, rtol=1e-9, atol=1e-12), (
        f"the two folds disagree by "
        f"{np.abs(sigma_old - sigma_new).max():.3e} — they are meant to be the "
        f"same congruence written two ways"
    )


def test_psd_survives_every_experiment_not_just_this_one(psd_setup, rng):
    """Congruence is unconditional, so assert it over many random point sets."""
    s = psd_setup
    n0, na = s.c33.shape[0], s.c34.shape[0]
    j = _psd_joint(s, cross=rng.normal(size=(n0, na)) * 0.0005)
    assert j.check(eigen=True).min_eig >= 0

    for trial in range(8):
        n = int(rng.integers(5, 30))
        e = rng.uniform(s.lo, s.hi, n) / 1e6
        lo_mev, hi_mev = _mf4_bin_edges_for_points(s.energies_mf4_mev, e)
        sig = j.fold(
            e, rng.uniform(-1, 1, n), rng.uniform(0.05, 0.6, n),
            rng.uniform(-0.4, 0.9, (n, L_MAX)), rng.uniform(0.05, 0.6, n),
            dtype=np.float64,
            sigma_window_ev=(lo_mev * 1e6, hi_mev * 1e6),
            sigma_map="overlap", a_map="nearest",
        )
        lam = (np.linalg.eigvalsh(sig).min()
               / max(np.abs(np.diag(sig)).max(), 1e-300))
        assert lam >= -1e-10, f"trial {trial}: lam_min/scale = {lam:.3e}"


# ── grids, maps, masking ──────────────────────────────────────────────────────

def test_a_block_that_does_not_span_the_points_contributes_zero_there(setup):
    """JEFF's (2, 6) in miniature: the short block must not reach below itself."""
    grid_a, c34, _ = assemble_a_block(setup.mf34, L_MAX)
    short = np.asarray(setup.mf34.energy_grids[-1], float)   # the (2, 3) block
    below = grid_a[:-1] < short[0]
    assert below.any(), "fixture no longer has a late-starting block"
    rows = np.flatnonzero(below)[:, None] * L_MAX + 1        # a_2 slots
    cols = np.flatnonzero(below)[:, None] * L_MAX + 2        # a_3 slots
    assert np.all(c34[np.ix_(rows.ravel(), cols.ravel())] == 0.0)


def test_nearest_weights_mask_rather_than_pin():
    grid = np.array([1.0, 2.0, 3.0])
    w = nearest_weights(grid, np.array([0.5, 1.5, 2.5, 3.5]))
    assert w[0].sum() == 0.0 and w[3].sum() == 0.0
    np.testing.assert_array_equal(w[1], [1.0, 0.0])
    np.testing.assert_array_equal(w[2], [0.0, 1.0])


def test_overlap_weights_are_row_normalised_and_mask_off_grid():
    grid = np.array([0.0, 1.0, 2.0, 3.0])
    w = overlap_weights(grid, np.array([0.5, 4.0]), np.array([2.5, 5.0]))
    np.testing.assert_allclose(w[0], [0.25, 0.5, 0.25])
    assert w[1].sum() == 0.0


# ── collapse and restriction ──────────────────────────────────────────────────

def test_collapse_is_a_congruence_and_preserves_psd(psd_setup, rng):
    s = psd_setup
    n0, na = s.c33.shape[0], s.c34.shape[0]
    j = _psd_joint(s, cross=rng.normal(size=(n0, na)) * 0.0005)
    assert j.check(eigen=True).min_eig >= 0

    g_sigma = s.grid_sigma[::3]                     # 12 -> 4 bins
    g_a = s.grid_a[::2]                             # 6 -> 3 bins
    k = j.collapse(g_sigma, g_a)

    a_s = width_weights(s.grid_sigma, g_sigma)
    a_a = width_weights(s.grid_a, g_a)
    big = np.zeros((k.n, j.n))
    big[:k.n_sigma, :j.n_sigma] = a_s
    big[k.n_sigma:, j.n_sigma:] = np.kron(a_a, np.eye(L_MAX))
    np.testing.assert_allclose(k.matrix, big @ j.matrix @ big.T, atol=1e-14)

    assert k.check(eigen=True).min_eig >= -1e-12, (
        "A C A^T went indefinite, which a congruence cannot do"
    )
    assert k.has_cross, "the collapse dropped the cross block"


def test_collapsing_the_blocks_separately_is_the_thing_this_replaces(psd_setup, rng):
    """The joint collapse and a per-block collapse are not the same operator.

    Asserting they DIFFER is the point: the pipeline collapses MF33 and MF34
    separately today, and a cross block collapsed with anything other than its
    two neighbours' own operators is what breaks the congruence.
    """
    s = psd_setup
    n0, na = s.c33.shape[0], s.c34.shape[0]
    j = _psd_joint(s, cross=rng.normal(size=(n0, na)) * 0.0005)
    g_sigma = s.grid_sigma[::3]
    k = j.collapse(g_sigma, s.grid_a)
    # Same magnitude collapse, but the cross block nearest-binned onto the
    # coarse grid instead of carried through A -- the shortcut.
    a_s = width_weights(s.grid_sigma, g_sigma)
    naive = nearest_weights(g_sigma, 0.5 * (s.grid_sigma[:-1]
                                            + s.grid_sigma[1:])).T @ j.cross
    assert not np.allclose(k.cross, naive, atol=1e-12)
    np.testing.assert_allclose(k.cross, a_s @ j.cross, atol=1e-14)


def test_restrict_is_a_principal_submatrix(setup):
    j = _joint(setup)
    r = j.restrict(setup.lo, 0.5 * (setup.lo + setup.hi))
    assert r.n < j.n
    assert r.grid_sigma_ev[0] == setup.grid_sigma[0]
    assert r.grid_sigma_ev[-1] <= 0.5 * (setup.lo + setup.hi) * (1 + 1e-9)
    # Round-trip a value to prove the index arithmetic is not transposed.
    assert r.matrix[0, 0] == j.matrix[0, 0]
    assert r.matrix[r.a_index(0, 2), r.a_index(0, 2)] == \
        j.matrix[j.a_index(0, 2), j.a_index(0, 2)]


def test_width_weights_refuse_a_boundary_that_cuts_a_fine_bin():
    fine = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="not a subset"):
        width_weights(fine, np.array([0.0, 1.5, 4.0]))


# ── layout ────────────────────────────────────────────────────────────────────

def test_layout_is_energy_slow_order_fast(setup):
    """The MF34 writer's documented layout, so the write is a reshape."""
    j = _joint(setup)
    assert j.a_index(0, 1) == j.n_sigma
    assert j.a_index(0, L_MAX) == j.n_sigma + L_MAX - 1
    assert j.a_index(1, 1) == j.n_sigma + L_MAX
    with pytest.raises(ValueError):
        j.a_index(0, L_MAX + 1)


def test_check_reports_the_blocks_but_the_verdict_is_the_joint(psd_setup, rng):
    """Healthy blocks + an indefinite joint is exactly the case that misled us."""
    s = psd_setup
    n0, na = s.c33.shape[0], s.c34.shape[0]
    j = _psd_joint(s, cross=rng.normal(size=(n0, na)) * 5.0)   # far too big
    rep = j.check(eigen=True)
    assert rep.min_eig_sigma >= -1e-12 and rep.min_eig_a >= -1e-12
    assert rep.min_eig < 0 and not rep.ok
    assert "NOT OK" in str(rep)


def test_the_correlation_tolerance_is_per_source_not_widened_globally(psd_setup):
    """1 + 1e-9 in memory; 1 + 5e-6 for a file, and only for a file.

    Run 86 measured both ends of this: the pre-write sidecars give
    max|rho| = 1.000000000 exactly and the same matrix read back out of the
    _mg file gives 1.000002. Widening the in-memory check to swallow that is
    how a real violation would hide, so the two are separate numbers.
    """
    from scripts.joint_covariance import CORR_TOL_FROM_ENDF, CORR_TOL_IN_MEMORY

    assert CORR_TOL_IN_MEMORY < CORR_TOL_FROM_ENDF
    j = _psd_joint(psd_setup)
    # A rho of exactly 1 nudged by the round trip's order of magnitude.
    j.matrix[0, 1] = j.matrix[1, 0] = (
        np.sqrt(j.matrix[0, 0] * j.matrix[1, 1]) * (1 + 2e-6))
    assert not j.check(eigen=False).ok
    assert j.check(eigen=False, correlation_tolerance=CORR_TOL_FROM_ENDF).ok
