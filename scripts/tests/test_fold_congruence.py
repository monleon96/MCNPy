"""Is Sigma_eval a congruence of the parameter-space joint? (roadmap §10.1.8-L13)

THE PROPERTY EVERYTHING RESTS ON. `build_group_cross.py` certifies a joint
`J = [[c33, cx], [cx.T, c34]]` as PSD in parameter space. That certification is
worth something to the chi2 only if the fold is a congruence `M J M^T`, because
a congruence cannot change the sign of an eigenvalue. It is one iff the cross
term's legs use the same maps to points as the self blocks do.

They do not. MF34 is shipped RELATIVE, so `build_mf34_block` scales its leg by
`a_l_per_pt`; the cross block is shipped ABSOLUTE, so
`build_mf33_mf34_cross_block` does not. The chi2 therefore folds `cx` against
`r^2 * Var34` where it was certified against `Var34`, with
`r(j,l) = a_l(E_j)/a_nom(g)`. And because `J` is a SINGULAR sample covariance it
sits ON the Cauchy-Schwarz boundary, so there is no margin: any `|r| < 1`
violates PSD.

These tests are the controlled demonstration -- same `J`, same code, one
variable. Runs 87 through 90 all died on `potrf` in the chi2 and every diagnosis
until now was taken in parameter space, where this is invisible.
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
    build_mf33_block,
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
    return dict(grp_ev=grp_ev, mf4_mev=mf4_mev, c33=c33, cross=cross,
                mf34=mf34, a_nom=a_nom, **pts)


def _lam_min_norm(w, a_l_per_pt):
    s34 = build_mf34_block(w["mf34"], w["e_mev"], w["mu"], w["c0"], a_l_per_pt)
    s33 = build_mf33_block(w["grp_ev"], w["c33"], w["mf4_mev"], w["e_mev"], w["y"])
    sx = build_mf33_mf34_cross_block(
        w["cross"], w["e_mev"], w["mu"], w["c0"], a_l_per_pt, w["y"],
        mf33_grid_ev=w["grp_ev"], energies_mf4_mev=w["mf4_mev"])
    S = s34 + s33 + sx
    lam = float(np.linalg.eigvalsh(0.5 * (S + S.T))[0])
    return lam / float(np.max(np.abs(np.diag(S)))), S


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


@pytest.mark.xfail(
    strict=True,
    reason="roadmap §10.6-3: the cross term must be read from MF34's a_0 blocks "
           "so both legs carry a_l(E_j). Until then the fold is not a congruence "
           "and this cannot pass. Flip to a plain test when that lands -- strict "
           "xfail makes it fail loudly the moment it starts passing.",
)
def test_psd_transfers_for_any_a_l(world):
    """WHAT WE WANT: certify J once, and have the chi2 inherit it."""
    rng = np.random.default_rng(3)
    a_pt = world["a_nom"][world["g_pt"]] * (
        1.0 + 0.5 * rng.uniform(-1, 1, size=(N, L)))
    lam, _ = _lam_min_norm(world, a_pt)
    assert lam > -1e-12, f"Sigma_eval must inherit J's PSD, got {lam:.3e}"
