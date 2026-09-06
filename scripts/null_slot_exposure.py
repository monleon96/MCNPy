#!/usr/bin/env python
"""How much of Sigma_eval comes from MF34 parameters the fit never determined?

ROADMAP Sec. 10.6-1. THIS IS THE MEASUREMENT THAT CAN MOVE PUBLISHED NUMBERS,
and it has nothing to do with the MF33<->MF34 cross block.

`row_aggregator` returns an all-zero row for any (shape group, order) with no
valid fine bin, so the MC constrains nothing at ~37 % of MF34's parameters
(39 of 703 groups at l=1, rising to 572 at l=6). The SHIPPED file nevertheless
declares nonzero relative variance at ~95 % of them -- up to 7.6, i.e.
sigma_rel ~ 276 %.

Sec. 10.1.8-L11 established those slots are NOT invisible to the chi2. The
fold scales a relative MF34 block by `a_l_per_pt`, which
`precompute_chi2_predictive._a_l_lookup` fills from `interp_a_l_to_energy` --
the file's own MF4 interpolated onto each EXFOR energy -- and that is NOT zero
there. What is empty at those slots is the MC validity mask, not the
coefficient. So they enter at FULL weight.

Run 86 folds the same file, and run 86 is the baseline every chi2 in the roadmap
is quoted against. If the contribution is material, `This_work` has been carrying
evaluated variance from parameters its own fit never determined -- which inflates
Sigma_eval and therefore FLATTERS `This_work` against JEFF and JENDL, because a
bigger Sigma gives a smaller chi2.

WHAT THIS SCRIPT DOES AND DOES NOT SETTLE. It computes the DIAGONAL of the MF34
contribution twice -- as shipped, and with the unsupported slots removed -- over
the very points the run scored. That sizes the effect for the cost of one ENDF
parse. It does NOT give a chi2: the chi2 inverts the full per-experiment block,
and a diagonal share does not map linearly onto it. The escalation, if this bites,
is a full re-score with the slots zeroed. Sizing first is the cheap half.

Why the diagonal and not the block: Cierjacks alone is 28631 x 28631, ~6.5 GB in
float64. The diagonal is exact and free -- `block.T[j,j] == block[j,j]`, so the
l_r != l_c terms simply double, exactly as `build_mf34_block` sums them.

THE MASK IS AN INPUT, NOT A RE-DERIVATION. Pass the npz written by
`build_group_cross.py --write-null-mask`, which computes it from the same
collapsed replicas the joint is built from. Deriving it locally from
`frozen_degree` gives 1412 slots against that one's 1542 -- the same object, not
the same array, because the run's MC-spread mask is slightly stricter.

Run:
  <venv>/bin/python null_slot_exposure.py \
      --parquet .../chi2_data_predictive_86.parquet \
      --endf    .../26-Fe-56g_nominal_mg.endf \
      --mask    .../mf34_null_mask.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_kika_root = Path(__file__).resolve().parent.parent
if str(_kika_root) not in sys.path:
    sys.path.insert(0, str(_kika_root))

from scripts.eval_covariance import _legendre_base_sens
from scripts.point_map import PointMap
from scripts.precompute_chi2_library_c0 import (
    interp_a_l_to_energy,
    load_library_lib_c0,
)

L_MAX = 6
LIB_KEY = "This_work"


def mf34_diag(mf34, e_mev, mu, c0, a_l_per_pt, drop=None):
    """Diagonal of `build_mf34_block`, optionally dropping some (point, order).

    Mirrors `eval_covariance.build_mf34_block` term for term: same bin lookup
    (`searchsorted(grid, e, "right") - 1`, clipped), same `base_sens`, same
    relative-vs-absolute a_l scaling, same doubling for l_r != l_c. Any drift
    between the two makes this measurement meaningless, so it is written to be
    read side by side with it rather than to be clever.

    drop : (N, L_MAX) bool or None
        True where the (point, order) is to be excluded. A term contributes only
        if BOTH its orders are kept, which is what zeroing the parameter's row
        AND column in the covariance does.
    """
    N = e_mev.size
    e_ev = e_mev * 1e6
    base_sens = _legendre_base_sens(mu, c0, L_MAX)
    out = np.zeros(N)
    per_order = np.zeros(L_MAX)

    for idx in range(len(mf34.matrices)):
        l_r, l_c = int(mf34.l_rows[idx]), int(mf34.l_cols[idx])
        if l_r < 1 or l_c < 1 or l_r > L_MAX or l_c > L_MAX:
            continue
        grid = np.asarray(mf34.energy_grids[idx], float)
        mat = np.asarray(mf34.matrices[idx], float)
        M = mat.shape[0]
        if M == 0:
            continue
        # THE SAME MAP `build_mf34_block` uses, by construction rather than by
        # being written to match (roadmap §10.7-7). This used to be a hand copy
        # of the bin lookup plus the off-grid mask, and it drifted the moment
        # the first copy changed — `test_null_slot_exposure.py` caught it, which
        # is luck we should not need twice. `sandwich_diag` also skips the (N,N)
        # the diagonal never needed.
        diag = PointMap.nearest(grid, e_ev).sandwich_diag(mat)

        sens_r = base_sens[l_r - 1].copy()
        sens_c = base_sens[l_c - 1].copy()
        if mf34.is_relative[idx]:
            sens_r = sens_r * a_l_per_pt[:, l_r - 1]
            sens_c = sens_c * a_l_per_pt[:, l_c - 1]

        term = sens_r * diag * sens_c
        if l_r != l_c:
            term = 2.0 * term          # block.T[j,j] == block[j,j]
        if drop is not None:
            term = np.where(drop[:, l_r - 1] | drop[:, l_c - 1], 0.0, term)
        out += term
        per_order[l_r - 1] += float(np.abs(term).sum())
        if l_r != l_c:
            per_order[l_c - 1] += float(np.abs(term).sum())
    return out, per_order


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="a chi2_data_*.parquet")
    ap.add_argument("--endf", required=True, help="the This_work ENDF it scored")
    ap.add_argument("--mask", required=True,
                    help="npz from build_group_cross.py --write-null-mask")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    z = np.load(args.mask)
    mask, shape_ev = z["null_mask"], z["shape_grid_ev"]
    n_gs = mask.shape[0]
    print(f"mask: {int(mask.sum())} of {mask.size} (group, order) slots "
          f"unsupported, on {n_gs} shape groups")
    print("  by order: " + "  ".join(
        f"a_{l+1} {int(mask[:, l].sum())}/{n_gs}" for l in range(L_MAX)))

    df = pd.read_parquet(args.parquet)
    df = df[df["library"] == LIB_KEY].reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"no {LIB_KEY} rows in {args.parquet}")
    print(f"\npoints: {len(df)} {LIB_KEY} rows, "
          f"{df['experiment_id'].nunique()} experiments")

    lib = load_library_lib_c0(args.endf, "This work")
    mf34 = lib.get("mf34")
    if mf34 is None:
        raise SystemExit(f"no MF34 in {args.endf}")

    e_mev = df["energy_mev"].to_numpy(float)
    mu = df["mu"].to_numpy(float)
    c0 = df["c0"].to_numpy(float)

    a_l_per_pt = np.zeros((len(df), L_MAX))
    for i, e_i in enumerate(e_mev):
        a_l_per_pt[i] = interp_a_l_to_energy(lib, float(e_i), L_MAX)

    # THE MASK IS PER POINT, VIA THE BASE GRID. The shipped MF34 blocks sit on
    # four nested grids (673/676/684/703); the mask is on the finest. A point's
    # base group therefore fixes its bin in EVERY block consistently, so mapping
    # the mask through the points is exact and avoids remapping it per block.
    g_pt = np.clip(np.searchsorted(shape_ev, e_mev * 1e6, side="right") - 1,
                   0, n_gs - 1)
    drop = mask[g_pt]                       # (N, L_MAX)
    print(f"  points touching >=1 unsupported slot: "
          f"{int(drop.any(1).sum())} of {len(df)} "
          f"({100 * drop.any(1).mean():.1f} %)")

    full, per_order_full = mf34_diag(mf34, e_mev, mu, c0, a_l_per_pt)
    kept, per_order_kept = mf34_diag(mf34, e_mev, mu, c0, a_l_per_pt, drop=drop)
    removed = full - kept

    print("\n=== MF34 diagonal contribution ===")
    print(f"  sum as shipped : {full.sum():.6g}")
    print(f"  sum supported  : {kept.sum():.6g}")
    print(f"  sum removed    : {removed.sum():.6g}  "
          f"({100 * removed.sum() / full.sum():.2f} % of the MF34 diagonal)")
    print("\n  |contribution| by order, as shipped:")
    print("   " + "  ".join(f"a_{l+1} {per_order_full[l]:.4g}"
                            for l in range(L_MAX)))
    print("  |contribution| by order, supported only:")
    print("   " + "  ".join(f"a_{l+1} {per_order_kept[l]:.4g}"
                            for l in range(L_MAX)))

    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(full > 0, removed / full, 0.0)
    print("\n  per-point share of the MF34 diagonal that is unsupported:")
    for q in (50, 90, 95, 99, 100):
        print(f"    p{q:<3d} {np.percentile(share, q):.4f}")

    # AGAINST THE FULL Sigma_eval, WHICH IS WHAT ACTUALLY MATTERS. MF34 is one of
    # two (or three) terms; a large share of a small term is not a large effect.
    col = "sigma_eval_var_diag" if "sigma_eval_var_diag" in df.columns else None
    if col is None and "sigma_eval_diag" in df.columns:
        # Fallback for older parquets: the clipped sigma, so var = sigma^2 and
        # any negative diagonal has already been destroyed. Say so.
        print("\n[WARN] no sigma_eval_var_diag; falling back to sigma_eval_diag^2, "
              "which is CLIPPED at zero. Negative diagonals are invisible in it.")
        tot = df["sigma_eval_diag"].to_numpy(float) ** 2
    elif col is not None:
        tot = df[col].to_numpy(float)
    else:
        tot = None

    if tot is not None:
        good = tot > 0
        print(f"\n=== against the FULL Sigma_eval diagonal ===")
        print(f"  removed / Sigma_eval, summed : "
              f"{100 * removed[good].sum() / tot[good].sum():.2f} %")
        with np.errstate(divide="ignore", invalid="ignore"):
            s2 = np.where(good, removed / tot, 0.0)
        for q in (50, 90, 95, 99, 100):
            print(f"    p{q:<3d} {np.percentile(s2, q):.4f}")

        # Per experiment, because the chi2 is per experiment and an effect
        # concentrated in one dataset is a different problem from a diffuse one.
        g = pd.DataFrame({
            "experiment_id": df["experiment_id"].to_numpy(),
            "removed": removed, "total": tot, "n": 1,
        }).groupby("experiment_id").sum(numeric_only=True)
        g["share"] = g["removed"] / g["total"]
        g = g.sort_values("share", ascending=False)
        print("\n  worst 12 experiments by removed / Sigma_eval:")
        print(g.head(12)[["n", "share"]].to_string())
        if args.out_csv:
            g.to_csv(args.out_csv)
            print(f"\n  wrote {args.out_csv}")

    print("\n⚠ This is a SIZE, not a chi2. The chi2 inverts the full "
          "per-experiment\n  block; a diagonal share does not map linearly onto "
          "it. If this bites,\n  the escalation is a full re-score with the "
          "unsupported slots zeroed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
