"""Do both legs of a parameter reach the points by the SAME map? (roadmap §10.7-2)

WHAT THIS ADDS TO `test_fold_congruence.py`. That file isolates the *units*
mismatch (§L13): MF34 is folded relative so its leg carries `a_l(E_j)`, the
cross sidecar is absolute so its leg carries 1. To do that cleanly its fixture
deliberately makes the MF33/MF34 group edges a subset of the MF4 grid, so
`build_mf33_block`'s overlap average `W` comes out one-hot and the OTHER
mismatch disappears. Its own docstring says so.

This file tests the one it removed, with the units already made consistent
(`is_relative=True`, `r = 1`), so the two defects are measured apart.

THE PARAMETER SPACE, stated once, because getting it wrong is how this goes
wrong. The fold is `Sigma_eval = M J M^T` with

    p    = ( sigma_rel on the magnitude bins , a_l ABSOLUTE on the shape groups )
    M33  = diag(y_eval) @ W            -- against a RELATIVE MF33
    M34  = c0*(2l+1)*P_l(mu) * [g==g_j] -- against an ABSOLUTE C34, with NO a_l
                                          factor: `build_mf34_block` multiplies
                                          the relative block by a_l on BOTH
                                          sides, which cancels the 1/(a a) the
                                          file stores.

⚑ THE CONFIGURATION UNDER TEST IS RUN 86'S, and the relationship between the
two MF33 representations is the whole point. `build_group_cross` COLLAPSES the
magnitude axis (`C33_c = A C33_f A^T`, `Cx_c = A Cx_f`) and certifies
`[[C33_c, Cx_c], [., C34]]`, which is PSD by congruence. The file then ships the
FINE `C33_f` and the COARSE `Cx_c`, and the chi2 folds those two together.

⚠ A FIRST VERSION OF THIS FILE BUILT `C33_f` BY DUPLICATING `C33_c` AND FOUND NO
VIOLATION -- correctly, and it is worth recording why. Duplication makes
`C33_f = D C33_c D^T`, so the folded joint is `diag(D, I)` applied to the
certified one: a congruence, PSD, nothing to detect. Collapsing is not the
inverse of duplicating. What survives the round trip is `P C33_f P^T` with
`P = D A` a projection, and `C33_f - P C33_f P^T` is indefinite in general --
that difference is the defect, and it exists only when the fine matrix carries
structure its own collapse threw away. Which is exactly what the -45 % peak
sigma of §10.7-3 measures.
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

from scripts.eval_covariance import (
    _legendre_base_sens,
    _mf33_overlap_weights,
    _mf4_bin_edges_for_points,
    build_mf33_block,
    build_mf33_mf34_cross_block,
    build_mf34_block,
)

L = 2          # Legendre orders
G34 = 6        # MF34 shape groups
G0 = 4         # cross block's magnitude axis -- the COARSE one (run 86's 188)
FINE = 4       # MF33 self-grid bins per coarse group (run 86's 2317)
G33 = G0 * FINE
REFINE = 3     # MF4 bins per MF33 bin, so MF4 is the finest grid of all
N = 200
E_LO, E_HI = 1.0e6, 5.0e6


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(20260807)

    mag_ev = np.linspace(E_LO, E_HI, G0 + 1)          # cross magnitude axis
    fine_ev = np.linspace(E_LO, E_HI, G33 + 1)        # MF33 self grid
    shape_ev = np.linspace(E_LO, E_HI, G34 + 1)       # MF34 shape grid
    mf4_mev = np.linspace(E_LO, E_HI, G33 * REFINE + 1) / 1e6

    # ⚠ The grids NEST: every MF4 bin sits inside one fine bin and one coarse
    # group, so BOTH overlap averages come out one-hot and the `W`-vs-
    # -`searchsorted` *rule* difference is exactly zero here. Whatever this file
    # measures is therefore the GRID, isolated. `test_the_rule_contributes_
    # nothing_here` asserts that rather than trusting it.
    assert np.isin(fine_ev, mf4_mev * 1e6).all()
    assert np.isin(mag_ev, fine_ev).all()

    # A PSD joint on the FINE parameter space, from replicas -- singular, so it
    # sits ON the Cauchy-Schwarz boundary like a real 10000-replica covariance.
    dim = G33 + G34 * L
    n_rep = dim - 3
    Z = rng.normal(size=(n_rep, dim))
    # Give the fine magnitude axis structure its own collapse cannot carry:
    # alternating sign within each coarse group. Averaging kills it, which is
    # what makes C33_f differ from P C33_f P^T at all.
    within = np.tile(np.where(np.arange(FINE) % 2 == 0, 1.0, -1.0), G0)
    Z[:, :G33] *= (1.0 + 0.8 * within)[None, :]
    J_fine = Z.T @ Z / (n_rep - 1)
    J_fine = 0.5 * (J_fine + J_fine.T)
    assert np.linalg.eigvalsh(J_fine)[0] > -1e-12
    assert np.linalg.matrix_rank(J_fine) < dim

    c33_f, c34, cx_f = J_fine[:G33, :G33], J_fine[G33:, G33:], J_fine[:G33, G33:]

    # What `build_group_cross` does: collapse the magnitude axis, then certify.
    A = np.zeros((G0, G33))
    for g in range(G0):
        A[g, g * FINE:(g + 1) * FINE] = 1.0 / FINE
    c33_c, cx_c = A @ c33_f @ A.T, A @ cx_f
    J_coarse = np.block([[c33_c, cx_c], [cx_c.T, c34]])
    assert np.linalg.eigvalsh(0.5 * (J_coarse + J_coarse.T))[0] > -1e-12, \
        "the certified joint must be PSD -- it is a congruence of J_fine"

    a_nom = (rng.uniform(0.3, 0.8, size=(G34, L))
             * rng.choice([-1.0, 1.0], size=(G34, L)))
    c34_rel = c34 / np.outer(a_nom.reshape(-1), a_nom.reshape(-1))

    mf34 = LegendreCovariance()
    for lr in range(1, L + 1):
        for lc in range(lr, L + 1):
            blk = c34_rel[np.ix_([g * L + lr - 1 for g in range(G34)],
                                 [g * L + lc - 1 for g in range(G34)])]
            if lr == lc:
                blk = 0.5 * (blk + blk.T)
            mf34.add_matrix(isotope_row=26056, reaction_row=2, l_row=lr,
                            isotope_col=26056, reaction_col=2, l_col=lc,
                            matrix=blk, energy_grid=list(shape_ev),
                            is_relative=True, frame="LAB")

    e_mev = rng.uniform(E_LO, E_HI, N) / 1e6
    g34_pt = np.clip(
        np.searchsorted(shape_ev, e_mev * 1e6, side="right") - 1, 0, G34 - 1)
    a_pt = a_nom[g34_pt]        # r = 1 exactly: §L13's mismatch is switched OFF

    cross = [{"l": l + 1, "mag_grid_ev": mag_ev, "shape_grid_ev": shape_ev,
              "matrix": (cx_c / a_nom.reshape(-1)[None, :])[
                  :, [g * L + l for g in range(G34)]],
              "is_relative": True} for l in range(L)]

    return dict(mag_ev=mag_ev, fine_ev=fine_ev, shape_ev=shape_ev,
                mf4_mev=mf4_mev, c33_f=c33_f, c33_c=c33_c, c34=c34,
                cx_c=cx_c, J_coarse=J_coarse, A=A, mf34=mf34, cross=cross,
                a_nom=a_nom, a_pt=a_pt, e_mev=e_mev,
                mu=rng.uniform(-1.0, 1.0, N), c0=rng.uniform(0.8, 1.2, N),
                y=rng.uniform(0.5, 1.5, N))


def _fold(w, mf33_grid, mf33_cov):
    """Sigma_eval through the production fold, MF33 on whichever grid is passed.

    Since the maps were unified (§10.7-7) the cross term's magnitude leg is
    built from `mf33_grid` too — there is no second axis to disagree with.
    """
    S = (build_mf34_block(w["mf34"], w["e_mev"], w["mu"], w["c0"], w["a_pt"])
         + build_mf33_block(mf33_grid, mf33_cov, w["mf4_mev"], w["e_mev"], w["y"])
         + build_mf33_mf34_cross_block(w["cross"], w["e_mev"], w["mu"],
                                       w["c0"], w["a_pt"], w["y"],
                                       mf33_grid_ev=mf33_grid,
                                       energies_mf4_mev=w["mf4_mev"]))
    return 0.5 * (S + S.T)


def _fold_legacy(w, mf33_grid, mf33_cov):
    """Sigma_eval as the fold assembled it BEFORE the maps were unified.

    Kept because the diagnosis is worth more than the code that had it: the
    self block averaged over `mf33_grid` while the cross term's magnitude leg
    did a nearest-bin lookup on its own `mag_ev`. Two maps for one parameter,
    so no single `M` existed and `Sigma_eval = M J M^T` was simply false.

    Written out of the test's own `_M33`/`_M34` rather than by calling the
    production code, which can no longer be made to do this — that is the point
    of the change.
    """
    M33_self = _M33(w, mf33_grid)
    M33_cross = _M33(w, w["mag_ev"])          # ← the second, disagreeing map
    M34 = _M34(w)
    S = M34 @ w["c34"] @ M34.T + M33_self @ mf33_cov @ M33_self.T
    X = M33_cross @ w["cx_c"] @ M34.T
    S = S + X + X.T
    return 0.5 * (S + S.T)


def _lam_min_norm(S):
    return float(np.linalg.eigvalsh(S)[0]) / float(np.max(np.abs(np.diag(S))))


def _M33(w, grid_ev):
    e_lo, e_hi = _mf4_bin_edges_for_points(w["mf4_mev"], w["e_mev"])
    W = _mf33_overlap_weights(np.asarray(grid_ev, float), e_lo * 1e6, e_hi * 1e6)
    return w["y"][:, None] * W


def _M34(w):
    """NO a_l factor -- see the module docstring. `build_mf34_block` applies it
    on both sides of a relative block, cancelling the file's own 1/(a a)."""
    base = _legendre_base_sens(w["mu"], w["c0"], L)
    g_pt = np.clip(np.searchsorted(w["shape_ev"], w["e_mev"] * 1e6,
                                   side="right") - 1, 0, G34 - 1)
    M = np.zeros((N, G34 * L))
    for l in range(1, L + 1):
        M[np.arange(N), g_pt * L + (l - 1)] = base[l - 1]
    return M


# ── the maps are what the fold means: not circular, because these fail first ──

def test_m33_reproduces_the_self_block(world):
    M = _M33(world, world["fine_ev"])
    want = M @ world["c33_f"] @ M.T
    got = build_mf33_block(world["fine_ev"], world["c33_f"], world["mf4_mev"],
                           world["e_mev"], world["y"])
    assert np.allclose(got, 0.5 * (want + want.T), atol=1e-12)


def test_m34_reproduces_the_self_block(world):
    """Also re-confirms §10.7-2: the MF34 leg is a clean congruence."""
    M = _M34(world)
    want = M @ world["c34"] @ M.T
    got = build_mf34_block(world["mf34"], world["e_mev"], world["mu"],
                           world["c0"], world["a_pt"])
    assert np.allclose(got, 0.5 * (want + want.T), atol=1e-10)


def test_the_rule_contributes_nothing_here(world):
    """`W` vs `searchsorted` is exactly zero on these nested grids, so every
    violation below is attributable to the GRID and not to the averaging rule.

    This matters for where the fix goes: a rule problem would send it to
    `_mf33_overlap_weights`, a grid problem sends it to what we WRITE.
    """
    for grid, n in ((world["fine_ev"], G33), (world["mag_ev"], G0)):
        W = _M33(world, grid) / world["y"][:, None]
        nn = np.zeros_like(W)
        nn[np.arange(N), np.clip(np.searchsorted(grid, world["e_mev"] * 1e6,
                                                 side="right") - 1, 0, n - 1)] = 1.0
        assert np.abs(W - nn).max() < 1e-12


# ── the control: one grid for the magnitude axis ──────────────────────────────

def test_control_one_grid_is_exactly_the_certified_congruence(world):
    """⚑ THE FIX, AS A TEST. Fold MF33 on the grid the cross block's magnitude
    axis already uses and Sigma_eval is literally `M J_coarse M^T` -- so PSD
    transfers by algebra, not by luck.

    This is §10.7-3 in one assertion: the joint is certified with its magnitude
    axis on the coarse grid, so that is the grid the chi2 has to fold it on.
    """
    M = np.hstack([_M33(world, world["mag_ev"]), _M34(world)])
    want = M @ world["J_coarse"] @ M.T
    got = _fold(world, world["mag_ev"], world["c33_c"])
    assert np.allclose(got, 0.5 * (want + want.T), atol=1e-10)
    assert _lam_min_norm(got) > -1e-12
    assert (np.diag(got) >= 0).all()


# ── the defect ────────────────────────────────────────────────────────────────

def test_the_legacy_two_map_assembly_broke_psd(world):
    """Run 86's configuration under the OLD fold: certify on the coarse axis,
    ship and fold the fine MF33, and let the cross term keep its own magnitude
    axis. Same J, ONE variable against the control.

    The units are consistent here, so this is §L13 switched off and §10.7-2(a)
    standing alone. This is now a historical record — `_fold_legacy` is the only
    thing that can still produce it.
    """
    lam = _lam_min_norm(_fold_legacy(world, world["fine_ev"], world["c33_f"]))
    assert lam < -1e-6, f"the two-map assembly must break PSD, got {lam:.3e}"


def test_the_legacy_assembly_agrees_with_production_when_the_axes_coincide(world):
    """`_fold_legacy` is the old fold and not merely a broken function: on the
    control configuration, where both magnitude maps land on the same grid, it
    reproduces the production fold exactly. Without this the test above would
    prove nothing about what the code used to do.
    """
    got = _fold_legacy(world, world["mag_ev"], world["c33_c"])
    want = _fold(world, world["mag_ev"], world["c33_c"])
    assert np.allclose(got, want, atol=1e-10)


def test_a_cross_block_off_the_mf33_grid_is_now_rejected(world):
    """⚑ THE FIX. The configuration that produced four runs with no χ² is no
    longer foldable: a cross term whose magnitude axis is not the shipped MF33
    grid is refused, loudly, instead of folded into an indefinite Σ_eval.

    This is the honest reading of §10.1 — `Cx` is Cauchy–Schwarz-compatible only
    with the marginals it was built from — turned into a guard.
    """
    with pytest.raises(ValueError, match="magnitude bins"):
        _fold(world, world["fine_ev"], world["c33_f"])


def test_the_two_defects_are_independent(world):
    """Fixing the units alone could not have been enough, because the map broke
    it too — measured on the legacy assembly, one variable at a time."""
    ok = _lam_min_norm(_fold_legacy(world, world["mag_ev"], world["c33_c"]))
    bad = _lam_min_norm(_fold_legacy(world, world["fine_ev"], world["c33_f"]))
    assert ok > -1e-12, ok
    assert bad < -1e-6, bad
    assert abs(bad) > 1e4 * abs(ok) + 1e-9, (ok, bad)


def test_a_block_that_does_not_span_the_points_contributes_zero_there(world):
    """⚑ §10.7-6. A block whose grid does not reach a point must contribute
    NOTHING at that point, not the value of its nearest edge interval.

    THIS IS NOT HYPOTHETICAL. JEFF-4.0 publishes 20 of its 21 MF34 blocks from
    1e-05 eV and **(2,6) only from 1 MeV**, while the EXFOR points run down to
    0.85 MeV — so every chi2 from run 82 to run 90 folded a fabricated
    Cov(a_2, a_6) for JEFF below 1 MeV. `build_group_cross` had the identical
    line and it manufactured a lam_min of -6.21e-02 out of nothing (§L18).
    Sec. L18.7's "it never fires in the chi2" was checked against This_work,
    whose own (2,6) starts at 0.846822 MeV and covers every point.

    The assertion is an identity, not a tolerance: masking the points must give
    exactly what deleting those rows and columns from the block gives.
    """
    w = world
    lo_cut = float(np.median(w["e_mev"]) * 1e6)     # truncate half the points away
    trunc = LegendreCovariance()
    for k in range(w["mf34"].num_matrices):
        lr, lc = int(w["mf34"].l_rows[k]), int(w["mf34"].l_cols[k])
        grid = np.asarray(w["mf34"].energy_grids[k], float)
        mat = np.asarray(w["mf34"].matrices[k], float)
        if (lr, lc) == (1, 2):                      # truncate ONE off-diagonal block
            keep = np.searchsorted(grid, lo_cut, side="right") - 1
            grid, mat = grid[keep:], mat[keep:, keep:]
        trunc.add_matrix(isotope_row=26056, reaction_row=2, l_row=lr,
                         isotope_col=26056, reaction_col=2, l_col=lc,
                         matrix=mat, energy_grid=list(grid),
                         is_relative=True, frame="LAB")

    got = build_mf34_block(trunc, w["e_mev"], w["mu"], w["c0"], w["a_pt"])

    # What the file says, built independently: the (1,2) term simply absent for
    # the uncovered points, every other block untouched.
    g12 = np.asarray(trunc.energy_grids[
        [i for i in range(trunc.num_matrices)
         if (int(trunc.l_rows[i]), int(trunc.l_cols[i])) == (1, 2)][0]], float)
    covered = (w["e_mev"] * 1e6 >= g12[0]) & (w["e_mev"] * 1e6 <= g12[-1])
    assert 0 < covered.sum() < N, "the fixture must leave some points uncovered"

    M = _M34(w)
    c34 = world["c34"].copy()
    r1 = np.arange(G34) * L + 0
    c2 = np.arange(G34) * L + 1
    want = M @ c34 @ M.T
    # subtract the (1,2)+(2,1) contribution, then add it back only where covered
    cross_only = np.zeros_like(c34)
    cross_only[np.ix_(r1, c2)] = c34[np.ix_(r1, c2)]
    cross_only[np.ix_(c2, r1)] = c34[np.ix_(c2, r1)]
    term = M @ cross_only @ M.T
    want = want - term + term * np.outer(covered, covered)
    assert np.allclose(got, 0.5 * (want + want.T), atol=1e-10)

    # And the uncovered points must not have picked up the edge interval.
    pinned = build_mf34_block(w["mf34"], w["e_mev"], w["mu"], w["c0"], w["a_pt"])
    assert not np.allclose(got, pinned, atol=1e-10), \
        "truncating a block must change the fold — otherwise nothing is tested"


def test_the_violation_is_the_structure_the_collapse_discards(world):
    """WHY it breaks, so the fix is not guessed at.

    The folded joint is `[[C33_f, P Cx_f], [., C34]]` with `P = D A`, and
    `xᵀ J' x = uᵀ(C33_f − P C33_f Pᵀ)u + J_coarse(Au, v)`. The second term is
    non-negative by certification; the first is the fine structure averaging
    threw away, and it is indefinite. Kill that structure and the violation goes
    with it — which is the dose-response the diagnosis needs.
    """
    A = world["A"]
    D = np.repeat(np.eye(G0), FINE, axis=0)     # A @ D == I, so P = D @ A
    P = D @ A
    flat = P @ world["c33_f"] @ P.T             # the piecewise-constant part

    # ⚑ SINGLE VARIABLE, and this is why the blend is the right one: since
    # A @ P == A, the COLLAPSE of every blend is identical, so `c33_c`, `cx_c`
    # and the certified joint do not move at all. Only how much fine structure
    # the shipped matrix carries above its own collapse changes. And each blend
    # is PSD, being a convex combination of two PSD matrices.
    lams = []
    for t in (0.0, 0.25, 0.5, 1.0):
        c33_f = (1.0 - t) * flat + t * world["c33_f"]
        assert np.linalg.eigvalsh(0.5 * (c33_f + c33_f.T))[0] > -1e-10
        assert np.allclose(A @ c33_f @ A.T, world["c33_c"], atol=1e-12), \
            "the blend must not move the collapse -- that is the control"
        lams.append(_lam_min_norm(_fold_legacy(world, world["fine_ev"], c33_f)))

    assert lams[0] > -1e-9, (
        f"t=0 is the piecewise-constant matrix, i.e. exactly the duplication of "
        f"its own collapse, so the fold IS a congruence and must be PSD; got "
        f"{lams[0]:.3e}")
    # Anything at 1e-12 IS zero here; comparing two machine zeros for order
    # would be reading noise as a trend.
    eff = [0.0 if lam > -1e-12 else lam for lam in lams]
    assert all(b <= a for a, b in zip(eff, eff[1:])), lams
    assert eff[-1] < -1e-6, lams

    # ⚑ AND THERE IS A THRESHOLD, which is the practically useful part: the
    # violation is not linear in the discarded structure. It stays at machine
    # zero while the fine matrix is close to piecewise-constant and only then
    # turns on. So "how far is the shipped MF33 from its own collapse" is the
    # quantity that decides whether this bites — not "are the grids equal".
    assert eff[1] == 0.0 and eff[2] < -1e-3, lams
