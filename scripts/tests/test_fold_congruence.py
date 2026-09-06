"""Is Sigma_eval a congruence of the parameter-space joint? (roadmap §10.1.8-L13)

THE PROPERTY EVERYTHING RESTS ON. `build_group_cross.py` certifies a joint
`J = [[c33, cx], [cx.T, c34]]` as PSD in parameter space. That certification is
worth something to the chi2 only if the fold is a congruence `M J M^T`, because
a congruence cannot change the sign of an eigenvalue. It is one iff the cross
term's legs use the same maps to points as the self blocks do.

THEY DID NOT, FOR RUNS 87 THROUGH 90. MF34 is shipped RELATIVE, so
`build_mf34_block` scales its leg by `a_l_per_pt`; the cross SIDECAR was
absolute, so `build_mf33_mf34_cross_block` did not. The chi2 folded `cx` against
`r^2 * Var34` where it was certified against `Var34`, with
`r(j,l) = a_l(E_j)/a_nom(g)`. And because `J` is a SINGULAR sample covariance it
sits ON the Cauchy-Schwarz boundary, so there is no margin: any `|r| < 1`
violates PSD. All four runs died on `potrf`, and every diagnosis until §L13 was
taken in parameter space, where this is invisible.

⚑ THE FILE NOW ANSWERS IT, and this suite is in two halves.

`_lam_min_norm` keeps the defect switched ON deliberately (`a_is_relative=False`
against a relative family) and the dose-response tests below are the evidence
that the guard added in §L13 is guarding something real. The production
orchestrator can no longer assemble that at all -- it reads the convention off
the MF34 family and refuses a block that disagrees.

`_lam_min_norm_congruent` is the repair: the cross term divided by `a_nom` on
the shape axis, which is what `write_consistent_mf34` writes into the MF34 a_0
blocks. Both legs then carry `a_l(E_j)`, one `M` exists, and
`test_psd_transfers_for_any_a_l` -- a STRICT XFAIL until 2026-08-08, with
"flip this when the a_0 blocks land" as its stated condition -- passes, at
spreads up to 90 %. Note what turned out not to matter: `a_l(E_j)` still varies
freely inside a group. It never had to be constant. Both legs had to be
multiplied by the SAME thing, which one file in one convention guarantees.
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
    _mf34_family_is_relative,
    build_mf33_mf34_cross_block,
    build_mf34_block,
)

L = 2
G = 6            # groups, shared by the MF33 and MF34-shape axes
REFINE = 4       # MF4 bins per group
N = 240


@pytest.fixture(scope="module")
def world():
    """A PSD parameter joint, an MF34 that publishes it relative, and points.

    ⚠ The MF33/MF34 group edges are a SUBSET of the MF4 grid on purpose, so the
    MF4 bin holding a point lies inside one group and `build_mf33_block`'s
    overlap average `W` is one-hot. That removes the *other* mismatch of §L13
    (`F != G`, the cross block's magnitude leg using nearest-bin where the MF33
    self block averages) and leaves only the shape-leg factor under test. The
    real run has the same nesting: its 188-group MF33 grid is exactly a subset
    of the 1738-bin fine grid.
    """
    rng = np.random.default_rng(20260806)
    grp_ev = np.linspace(1.0e6, 5.0e6, G + 1)
    mf4_mev = np.unique(np.concatenate(
        [np.linspace(grp_ev[i], grp_ev[i + 1], REFINE + 1)
         for i in range(G)])) / 1e6

    # SINGULAR by construction (n_rep < dim), so it saturates Cauchy-Schwarz
    # exactly the way a real sample covariance of 10000 replicas over 4218
    # rank-2658 parameters does. A joint with margin would hide the effect.
    dim = G + G * L
    n_rep = dim - 4
    Z = rng.normal(size=(n_rep, dim))
    J = Z.T @ Z / (n_rep - 1)
    J = 0.5 * (J + J.T)
    assert np.linalg.eigvalsh(J)[0] > -1e-12, "the joint must start PSD"
    assert np.linalg.matrix_rank(J) < dim, "the joint must be singular"

    c33, c34, cx = J[:G, :G], J[G:, G:], J[:G, G:]
    a_nom = (rng.uniform(0.3, 0.8, size=(G, L))
             * rng.choice([-1.0, 1.0], size=(G, L)))
    c34_rel = c34 / np.outer(a_nom.reshape(-1), a_nom.reshape(-1))

    mf34 = LegendreCovariance()
    for lr in range(1, L + 1):
        for lc in range(lr, L + 1):
            blk = c34_rel[np.ix_([g * L + lr - 1 for g in range(G)],
                                 [g * L + lc - 1 for g in range(G)])]
            if lr == lc:
                blk = 0.5 * (blk + blk.T)
            mf34.add_matrix(isotope_row=26056, reaction_row=2, l_row=lr,
                            isotope_col=26056, reaction_col=2, l_col=lc,
                            matrix=blk, energy_grid=list(grp_ev),
                            is_relative=True, frame="LAB")

    # The magnitude axis is the MF33 grid itself now, so the block no longer
    # carries one of its own; `grp_ev` reaches the fold as `mf33_grid_ev`.
    cross = [{"l": l + 1, "shape_grid_ev": grp_ev,
              "matrix": cx[:, [g * L + l for g in range(G)]],
              "is_relative": False} for l in range(L)]

    e_mev = rng.uniform(grp_ev[0], grp_ev[-1], N) / 1e6
    pts = dict(e_mev=e_mev, mu=rng.uniform(-1.0, 1.0, N),
               c0=rng.uniform(0.8, 1.2, N), y=rng.uniform(0.5, 1.5, N))
    pts["g_pt"] = np.clip(
        np.searchsorted(grp_ev, e_mev * 1e6, side="right") - 1, 0, G - 1)
    return dict(grp_ev=grp_ev, mf4_mev=mf4_mev, c33=c33, c34=c34,
                cx_abs=cx, cross=cross, mf34=mf34, a_nom=a_nom, **pts)


def _lam_min_norm(w, a_l_per_pt):
    """Sigma_eval with §L13's units mismatch deliberately switched ON.

    ⚑ `a_is_relative=False` is what makes this the DEFECT and not the fix. The
    MF34 family here is relative, so its self block carries `a_l(E_j)`; passing
    False keys the cross term's shape leg on the block's own flag instead, which
    is what `build_mf33_mf34_cross_block` used to do unconditionally
    (§10.7-2(b)). Two units for one parameter.

    The production orchestrator can no longer assemble this — it reads the flag
    off the MF34 family and the mismatch raises. See
    `test_the_production_fold_now_refuses_the_units_mismatch`. These tests keep
    reaching past that guard on purpose, because the dose-response below is the
    evidence that the guard is guarding something real.
    """
    s34 = build_mf34_block(w["mf34"], w["e_mev"], w["mu"], w["c0"], a_l_per_pt)
    s33 = build_mf33_block(w["grp_ev"], w["c33"], w["mf4_mev"], w["e_mev"], w["y"])
    sx = build_mf33_mf34_cross_block(
        w["cross"], w["e_mev"], w["mu"], w["c0"], a_l_per_pt, w["y"],
        mf33_grid_ev=w["grp_ev"], energies_mf4_mev=w["mf4_mev"],
        a_is_relative=False)
    S = s34 + s33 + sx
    lam = float(np.linalg.eigvalsh(0.5 * (S + S.T))[0])
    return lam / float(np.max(np.abs(np.diag(S)))), S


def _lam_min_norm_congruent(w, a_l_per_pt):
    """Sigma_eval with the cross term in the marginals' OWN coordinates.

    ⚑ The repair §L13 pointed at, and the reason it had to be upstream. The
    cross block is divided by `a_nom` on the shape axis -- exactly what
    `write_consistent_mf34` does before writing the MF34 a_0 blocks -- so it
    declares `is_relative=True` and matches the family. Then BOTH legs of the
    shape parameter carry `a_l(E_j)`, one `M` exists, and

        Sigma_eval = M J M^T,   M = [ diag(y).W | base_l * a_l(E_j) * S ]

    is a congruence for ANY `a_l_per_pt`, however it varies inside a group.
    That is the whole claim, and `test_psd_transfers_for_any_a_l` below is now
    a plain test rather than an xfail because of it.
    """
    a_flat = w["a_nom"].reshape(-1)
    cross_rel = [dict(b, matrix=b["matrix"] / a_flat[None, [g * L + b["l"] - 1
                                                           for g in range(G)]],
                      is_relative=True)
                 for b in w["cross"]]
    s34 = build_mf34_block(w["mf34"], w["e_mev"], w["mu"], w["c0"], a_l_per_pt)
    s33 = build_mf33_block(w["grp_ev"], w["c33"], w["mf4_mev"], w["e_mev"], w["y"])
    sx = build_mf33_mf34_cross_block(
        cross_rel, w["e_mev"], w["mu"], w["c0"], a_l_per_pt, w["y"],
        mf33_grid_ev=w["grp_ev"], energies_mf4_mev=w["mf4_mev"],
        a_is_relative=True)
    S = s34 + s33 + sx
    lam = float(np.linalg.eigvalsh(0.5 * (S + S.T))[0])
    return lam / float(np.max(np.abs(np.diag(S)))), S


def test_the_production_fold_now_refuses_the_units_mismatch(world):
    """⚑ §L13's HALF OF THE FIX. The fold no longer keys the shape leg on the
    cross block's own flag, so an absolute cross against a relative MF34 is
    refused instead of folded into an indefinite Sigma_eval.

    Refused rather than converted: the repair is to build the cross in the
    marginals' coordinates, because converting means dividing by an `a_l_nom`
    that `test_a_l_crossing_zero_is_the_unbounded_case` shows passes through
    zero.
    """
    a_pt = world["a_nom"][world["g_pt"]]
    with pytest.raises(ValueError, match="two units"):
        build_mf33_mf34_cross_block(
            world["cross"], world["e_mev"], world["mu"], world["c0"], a_pt,
            world["y"], mf33_grid_ev=world["grp_ev"],
            energies_mf4_mev=world["mf4_mev"],
            a_is_relative=True,          # what the MF34 family actually declares
        )


def test_the_orchestrator_reads_the_convention_off_the_family(world):
    """The guard is only worth having if the orchestrator cannot bypass it, so
    pin where the flag comes from."""
    assert _mf34_family_is_relative(world["mf34"]) is True
    assert _mf34_family_is_relative(None) is None

    # A file mixing conventions across blocks has no family answer, and
    # guessing which one the cross term belongs to is the §L13 mistake itself.
    mixed = LegendreCovariance()
    for lr, rel in ((1, True), (2, False)):
        mixed.add_matrix(isotope_row=26056, reaction_row=2, l_row=lr,
                         isotope_col=26056, reaction_col=2, l_col=lr,
                         matrix=np.eye(G), energy_grid=list(world["grp_ev"]),
                         is_relative=rel, frame="LAB")
    with pytest.raises(ValueError, match="mixes relative and absolute"):
        _mf34_family_is_relative(mixed)


def test_congruence_holds_when_both_legs_share_the_factor(world):
    """r = 1: Sigma_eval IS [F, S] J [F, S]^T, so PSD transfers exactly.

    This is the control, and it is what makes the failures below attributable
    to the factor rather than to the blocks, the grids or the code.
    """
    a_pt = world["a_nom"][world["g_pt"]]
    lam, S = _lam_min_norm(world, a_pt)
    assert lam > -1e-12, f"the control must be PSD, got {lam:.3e}"
    assert (np.diag(S) >= 0).all(), "no point may have negative variance"


@pytest.mark.parametrize("spread, worse_than", [(0.05, 1e-2), (0.5, 1e-1)])
def test_a_l_varying_within_a_group_breaks_psd(world, spread, worse_than):
    """r != 1 breaks it, and the damage grows with the spread.

    5 % of variation inside a group already costs more than the entire margin
    the real baseline has (run 86 sits at -0.008, §L12).
    """
    rng = np.random.default_rng(7)
    a_pt = world["a_nom"][world["g_pt"]] * (
        1.0 + spread * rng.uniform(-1, 1, size=(N, L)))
    lam, _ = _lam_min_norm(world, a_pt)
    assert lam < -worse_than, f"expected a violation < -{worse_than}, got {lam:.3e}"


def test_violation_grows_monotonically_with_the_spread(world):
    """Dose-response. A one-off negative eigenvalue could be many things."""
    rng = np.random.default_rng(11)
    base = world["a_nom"][world["g_pt"]]
    lams = []
    for spread in (0.05, 0.2, 0.5, 0.9):
        a_pt = base * (1.0 + spread * rng.uniform(-1, 1, size=(N, L)))
        lams.append(_lam_min_norm(world, a_pt)[0])
    assert all(b < a for a, b in zip(lams, lams[1:])), lams


def test_a_l_crossing_zero_is_the_unbounded_case(world):
    """The limit the roadmap cares about: r -> 0 gives negative VARIANCES.

    Grouping pulls sign-changing coefficients toward zero, which is what
    REGULARIZE_NEAR_ZERO_REL_UNC exists to contain on the variance side. There
    is no counterpart on the cross side.
    """
    a_pt = world["a_nom"][world["g_pt"]] * np.linspace(-1.0, 1.0, N)[:, None]
    lam, S = _lam_min_norm(world, a_pt)
    assert lam < -1.0, f"expected an unbounded-looking violation, got {lam:.3e}"
    assert (np.diag(S) < 0).any(), "some point must get negative variance"


@pytest.mark.parametrize("spread", [0.05, 0.5, 0.9])
def test_psd_transfers_for_any_a_l(world, spread):
    """⚑⚑ WHAT THE WHOLE TRACK WAS FOR: certify J once, and have the chi2
    inherit it. This was a STRICT XFAIL until the cross term moved into the
    MF34 a_0 blocks; it is now a plain test, at the spreads that broke the
    absolute assembly by 1e-2 and 1e-1 above.

    Note what is NOT required: `a_l(E_j)` still varies inside a group by up to
    90 %. It never mattered. What mattered was that both legs be multiplied by
    the SAME thing, which is what one file in one convention guarantees.
    """
    rng = np.random.default_rng(3)
    a_pt = world["a_nom"][world["g_pt"]] * (
        1.0 + spread * rng.uniform(-1, 1, size=(N, L)))
    lam, S = _lam_min_norm_congruent(world, a_pt)
    assert lam > -1e-12, f"Sigma_eval must inherit J's PSD, got {lam:.3e}"
    assert (np.diag(S) >= 0).all(), "no point may have negative variance"


def test_the_congruent_fold_is_exactly_M_J_MT(world):
    """PSD is the consequence; the identity is the reason. A sign check passes
    on folds that are not congruences -- 10.7-9's baseline sits at -0.008 and
    is still 'PSD enough' -- so assert the algebra, not its symptom."""
    rng = np.random.default_rng(5)
    a_pt = world["a_nom"][world["g_pt"]] * (
        1.0 + 0.5 * rng.uniform(-1, 1, size=(N, L)))
    _, S = _lam_min_norm_congruent(world, a_pt)

    e_lo, e_hi = _mf4_bin_edges_for_points(world["mf4_mev"], world["e_mev"])
    W = _mf33_overlap_weights(world["grp_ev"], e_lo * 1e6, e_hi * 1e6)
    base = _legendre_base_sens(world["mu"], world["c0"], L)
    M = np.zeros((N, G + G * L))
    M[:, :G] = world["y"][:, None] * W
    for l in range(1, L + 1):
        M[np.arange(N), G + world["g_pt"] * L + (l - 1)] = (
            base[l - 1] * a_pt[:, l - 1])

    a_flat = world["a_nom"].reshape(-1)
    c34_rel = world["c34"] / np.outer(a_flat, a_flat)
    cx_rel = world["cx_abs"] / a_flat[None, :]
    J_rel = np.block([[world["c33"], cx_rel], [cx_rel.T, c34_rel]])

    want = M @ J_rel @ M.T
    want = 0.5 * (want + want.T)
    err = float(np.abs(S - want).max()) / float(np.abs(want).max())
    assert err < 1e-10, f"Sigma_eval is not M J M^T: relative {err:.3e}"
