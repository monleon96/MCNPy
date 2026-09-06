"""A PSD joint, written to ENDF, read back, folded — is Sigma_eval still M J M^T?

⚑ THE GATE THAT WOULD HAVE CAUGHT RUNS 87 THROUGH 90, and which did not exist.

Every check in this track has covered one link: `build_group_cross` certifies the
joint in parameter space, `test_group_cross_endf_writer` checks the file
round-trips, `test_fold_maps` and `test_fold_congruence` isolate the two fold
defects on synthetic blocks. Four runs still died on `potrf`, because a chain of
locally-correct links is not a correct chain — the joint that was certified was
never the joint that was folded.

This runs the whole chain on one object:

    PSD J  ->  MF34 with a_0 blocks  ->  ENDF text  ->  read back
           ->  build_eval_cov_for_groups  ->  compare to M J M^T

and the comparison is against the matrices READ OUT OF THE FILE, so ENDF's six
significant digits never enter as a tolerance. The identity is exact or the fold
is wrong.

THE PARAMETER SPACE, in the coordinates the file publishes:

    p    = ( delta_sigma/sigma per MF33 bin , delta_a_l/a_l per (group, order) )
    M33  = diag(y_eval) @ W                        W = overlap over the MF4 bin
    M34  = c0*(2l+1)*P_l(mu) * a_l(E_j) * [g == g_j]

⚠ The a_l factor IS present here, unlike `test_fold_maps._M34`, because there
the shape block is absolute and here it is relative — `build_mf34_block`
multiplies a relative block by a_l on both sides.

⚠ The MF4 grid is deliberately COARSER than the MF33 grid, so `W` carries two
nonzeros per row. That is the shipped geometry, measured: 46805 of 46819 points
have exactly 2, and none has 1 (roadmap §10.7-10, 0.4). A fixture where `W` came
out one-hot would test a map the production fold never takes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kika.endf.writers.mf34_writer import (
    create_mf34_from_covariance,
    write_mf34_to_file,
)

from scripts.eval_covariance import (
    _legendre_base_sens,
    _mf33_overlap_weights,
    _mf4_bin_edges_for_points,
    build_eval_cov_for_groups,
)
from scripts.mf34_cross_reader import read_mf34_split

ZA, AWR, MAT, MT = 26056.0, 55.454, 2631, 2
ISO = 26056
L = 3
G_MAG, G_SHAPE = 8, 4
N = 50

# ⚑ EXACTLY REPRESENTABLE IN SIX SIGNIFICANT DIGITS, on purpose. A `linspace`
# grid here gives 1116666.667 -> "1.116667+6" -> 1116667.0, a 3e-7 relative
# move, and `read_mf34_split` then refuses the block -- correctly, because in
# production the a_0 row grid is written from the array MF33 was READ BACK
# into, which is idempotent, and a reconstruction is not (§10.7-10, 0.7).
# Sidestepping that here keeps this file about the fold; the guard itself is
# tested in `test_mf34_cross_reader.py`.
MAG_EV = 850_000.0 + 400_000.0 * np.arange(G_MAG + 1)

_TEMPLATE = (Path(__file__).resolve().parents[2] / "kika" / "endf" / "tests"
             / "data" / "micro_fe56_cov.endf")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    rng = np.random.default_rng(20260808)

    mag_ev = MAG_EV                                     # == the MF33 grid
    shape_ev = mag_ev[::G_MAG // G_SHAPE]               # nested, 4 groups
    mf4_mev = mag_ev[::2] / 1e6                         # COARSER than MF33
    assert shape_ev.size == G_SHAPE + 1

    # A PSD joint directly in RELATIVE coordinates -- the space the file
    # publishes -- so this tests the fold and not the absolute->relative
    # conversion, which `test_group_cross_endf_writer` already pins.
    #
    # SINGULAR by construction, so it sits ON the Cauchy-Schwarz boundary the
    # way a 10000-replica sample covariance over 12166 parameters does. A joint
    # with margin would forgive a fold that is not quite a congruence.
    dim = G_MAG + G_SHAPE * L
    n_rep = dim - 4
    Z = rng.normal(size=(n_rep, dim)) * 0.05
    J = Z.T @ Z / (n_rep - 1)
    J = 0.5 * (J + J.T)
    assert np.linalg.eigvalsh(J)[0] > -1e-14, "the joint must start PSD"
    assert np.linalg.matrix_rank(J) < dim, "and singular, or there is margin"

    c33, c34, cx = J[:G_MAG, :G_MAG], J[G_MAG:, G_MAG:], J[:G_MAG, G_MAG:]

    # Write it: shape blocks on the shape grid, a_0 blocks LB=6 rectangular on
    # (MF33 grid) x (shape grid) -- the geometry §10.7-9 vindicated.
    cross_cov = {l: cx[:, [g * L + (l - 1) for g in range(G_SHAPE)]]
                 for l in range(1, L + 1)}
    mf34_obj = create_mf34_from_covariance(
        c34, shape_ev, L, ZA, AWR, MAT, MT,
        ltt=1, cross_cov=cross_cov, cross_energy_grid_ev=mag_ev,
    )
    path = tmp_path_factory.mktemp("e2e") / "joint_mg.endf"
    write_mf34_to_file(str(_TEMPLATE), mf34_obj, str(path))

    res = read_mf34_split(path, isotope=ISO, mt=MT, l_max=L,
                          mf33_grid_ev=mag_ev)
    assert len(res.cross) == L, "the a_0 blocks must survive the round trip"

    # ⚑ Rebuild the joint FROM THE FILE. Comparing against the pre-write arrays
    # would make ENDF's 6 significant digits a tolerance in a test whose whole
    # point is an exact identity; comparing against what came back makes the
    # round trip a separate, honest question (asserted below).
    c34_file = np.zeros_like(c34)
    for k in range(res.mf34.num_matrices):
        lr, lc = int(res.mf34.l_rows[k]), int(res.mf34.l_cols[k])
        blk = np.asarray(res.mf34.matrices[k], float)
        ri = [g * L + lr - 1 for g in range(G_SHAPE)]
        ci = [g * L + lc - 1 for g in range(G_SHAPE)]
        c34_file[np.ix_(ri, ci)] = blk
        if lr != lc:
            c34_file[np.ix_(ci, ri)] = blk.T
    cx_file = np.zeros_like(cx)
    for b in res.cross:
        cx_file[:, [g * L + (b["l"] - 1) for g in range(G_SHAPE)]] = b["matrix"]
    J_file = np.block([[c33, cx_file], [cx_file.T, c34_file]])
    J_file = 0.5 * (J_file + J_file.T)

    e_mev = rng.uniform(mag_ev[0] * 1.001, mag_ev[-1] * 0.999, N) / 1e6
    mu = rng.uniform(-1.0, 1.0, N)
    c0 = rng.uniform(0.8, 1.2, N)
    y = rng.uniform(0.5, 1.5, N)
    a_nom = (rng.uniform(0.3, 0.8, (G_SHAPE, L))
             * rng.choice([-1.0, 1.0], (G_SHAPE, L)))
    g_pt = np.clip(np.searchsorted(shape_ev, e_mev * 1e6, side="right") - 1,
                   0, G_SHAPE - 1)
    a_pt = a_nom[g_pt]

    return dict(path=path, mag_ev=mag_ev, shape_ev=shape_ev, mf4_mev=mf4_mev,
                c33=c33, J=J, J_file=J_file, res=res, e_mev=e_mev, mu=mu,
                c0=c0, y=y, a_pt=a_pt, g_pt=g_pt)


def _M(w):
    """The single map from the parameter vector to the data points."""
    e_lo, e_hi = _mf4_bin_edges_for_points(w["mf4_mev"], w["e_mev"])
    W = _mf33_overlap_weights(w["mag_ev"], e_lo * 1e6, e_hi * 1e6)
    base = _legendre_base_sens(w["mu"], w["c0"], L)
    M = np.zeros((N, G_MAG + G_SHAPE * L))
    M[:, :G_MAG] = w["y"][:, None] * W
    for l in range(1, L + 1):
        M[np.arange(N), G_MAG + w["g_pt"] * L + (l - 1)] = (
            base[l - 1] * w["a_pt"][:, l - 1])
    return M


def _sigma_from_production(w):
    """Sigma_eval through the orchestrator the chi2 actually calls."""
    df = pd.DataFrame({
        "library": ["This_work"] * N, "experiment_id": ["E1"] * N,
        "energy_mev": w["e_mev"], "mu": w["mu"], "c0": w["c0"],
        "y_eval": w["y"],
    })
    lib = {
        "mf34": w["res"].mf34,
        "mf33_grid_ev": w["mag_ev"],
        "mf33_rel_cov": w["c33"],
        "energies_mf4_mev": w["mf4_mev"],
        "mf33_mf34_cross": w["res"].cross,
    }
    out = build_eval_cov_for_groups(
        df, {"This_work": lib},
        lambda key, l, e: w["a_pt"][int(np.argmin(np.abs(w["e_mev"] - e)))],
        l_max=L,
    )
    return np.asarray(out[("This_work", "E1")], dtype=np.float64)


# ── the chain ────────────────────────────────────────────────────────────────

def test_W_is_not_one_hot_here_because_it_is_not_one_hot_in_production(world):
    """Guard the fixture. §10.7-2(a) assumed the MF4 grid was finer than MF33's;
    it is coarser, and 0 of 46819 real points give a one-hot row (§10.7-10)."""
    e_lo, e_hi = _mf4_bin_edges_for_points(world["mf4_mev"], world["e_mev"])
    W = _mf33_overlap_weights(world["mag_ev"], e_lo * 1e6, e_hi * 1e6)
    nnz = (W != 0).sum(axis=1)
    assert nnz.min() >= 2, f"fixture degenerated to one-hot rows: {nnz.min()}"


def test_sigma_eval_is_exactly_M_J_MT(world):
    """⚑ THE IDENTITY. Not a sign check — a sign check passes on a fold that has
    quietly stopped being a congruence, and that is how runs 87-90 got launched.
    """
    M = _M(world)
    want = M @ world["J_file"] @ M.T
    want = 0.5 * (want + want.T)
    got = _sigma_from_production(world)
    scale = float(np.abs(want).max())
    err = float(np.abs(got - want).max())
    assert err / scale < 1e-6, (
        f"Sigma_eval is not M J M^T: max|diff| = {err:.3e}, "
        f"relative {err / scale:.3e}. The fold has stopped being a congruence."
    )


def test_psd_transfers_from_the_joint_through_the_file_to_sigma_eval(world):
    """What the whole track has been trying to buy: certify J once, and have the
    chi2 inherit it. The bar is -1e-6 relative, agreed in advance, because ENDF
    carries 6 significant digits (§L17, §10.7-10 item 0.2) — not zero.
    """
    lam_j = float(np.linalg.eigvalsh(world["J_file"])[0])
    scale_j = float(np.abs(np.diag(world["J_file"])).max())
    assert lam_j / scale_j > -1e-6, (
        f"the joint did not survive the ENDF round trip: {lam_j / scale_j:.3e}")

    S = _sigma_from_production(world)
    lam = float(np.linalg.eigvalsh(0.5 * (S + S.T))[0])
    scale = float(np.abs(np.diag(S)).max())
    assert lam / scale > -1e-6, f"Sigma_eval is indefinite: {lam / scale:.3e}"


def test_dropping_the_cross_term_still_folds_and_lowers_nothing_structurally(world):
    """The cross term must be the ONLY difference, so the same chain with an
    empty block list is the control for the two tests above."""
    M = _M(world)
    J0 = world["J_file"].copy()
    J0[:G_MAG, G_MAG:] = 0.0
    J0[G_MAG:, :G_MAG] = 0.0

    df = pd.DataFrame({
        "library": ["This_work"] * N, "experiment_id": ["E1"] * N,
        "energy_mev": world["e_mev"], "mu": world["mu"], "c0": world["c0"],
        "y_eval": world["y"],
    })
    lib = {"mf34": world["res"].mf34, "mf33_grid_ev": world["mag_ev"],
           "mf33_rel_cov": world["c33"], "energies_mf4_mev": world["mf4_mev"]}
    out = build_eval_cov_for_groups(
        df, {"This_work": lib},
        lambda key, l, e: world["a_pt"][int(np.argmin(np.abs(world["e_mev"] - e)))],
        l_max=L,
    )
    got = np.asarray(out[("This_work", "E1")], dtype=np.float64)
    want = M @ J0 @ M.T
    assert np.abs(got - want).max() / np.abs(want).max() < 1e-6


def test_a_cross_term_without_its_magnitude_self_block_is_refused(world):
    """[[0, Cx], [Cx.T, C34]] has a negative eigenvalue for every nonzero Cx.
    The MF34-only scenarios are safe by never setting the key; what must never
    happen is keeping Cx and losing C33."""
    df = pd.DataFrame({
        "library": ["This_work"] * N, "experiment_id": ["E1"] * N,
        "energy_mev": world["e_mev"], "mu": world["mu"], "c0": world["c0"],
        "y_eval": world["y"],
    })
    lib = {"mf34": world["res"].mf34, "energies_mf4_mev": world["mf4_mev"],
           "mf33_mf34_cross": world["res"].cross}
    with pytest.raises(ValueError, match="no MF33 self block"):
        build_eval_cov_for_groups(
            df, {"This_work": lib},
            lambda key, l, e: world["a_pt"][0], l_max=L,
        )
