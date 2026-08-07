#!/usr/bin/env python
"""Prove `JointCov` is a no-op against the shipped fold, on the real file.

ROADMAP §10.7-7, step 1's gate. `test_joint_covariance.py` asserts the same
identity on a miniature; this asserts it on the artefact a chi^2 actually
scored, which is the only version that licenses switching anything on.

With the cross block zero -- the factorization every run since 82 has shipped --

    JointCov.fold(...)  ==  build_mf34_block(...) + build_mf33_block(...)

must hold to float64 round-off. If it does not, the object is wrong and the
published numbers stand.

It also prints `JointCov.check()` on the joint restricted to the analysis
window, which is the "is our matrix good" report in one place: symmetry,
finiteness, max |rho|, and the min eigenvalue of the joint AND of each block.
Read the joint's, not the blocks': a principal submatrix of a PSD matrix is PSD,
so healthy blocks prove nothing about the whole, and that asymmetry is what made
§10.7-3 look settled when it was not.

Run (cluster):
  <venv>/bin/python verify_joint_equivalence.py \
      --endf    .../new_test_86_mgfix/26-Fe-56g_nominal_mg.endf \
      --parquet .../chi2_data_predictive_86.parquet \
      --library This_work --max-points 3000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.eval_covariance import (  # noqa: E402
    _mf4_bin_edges_for_points,
    build_mf33_block,
    build_mf34_block,
)
from scripts.joint_covariance import JointCov  # noqa: E402
from scripts.precompute_chi2_library_c0 import (  # noqa: E402
    interp_a_l_to_energy,
    load_library_lib_c0,
)

L_MAX = 6
E_MIN_MEV, E_MAX_MEV = 0.85, 4.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endf", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--library", default="This_work")
    ap.add_argument("--max-points", type=int, default=3000,
                    help="cap per experiment; the dense N x N is the cost")
    ap.add_argument("--n-experiments", type=int, default=6)
    ap.add_argument("--eigen", action="store_true",
                    help="also eigendecompose the joint (O(n^3), minutes)")
    args = ap.parse_args()

    t0 = time.time()
    lib = load_library_lib_c0(args.endf, args.library)
    if lib.get("mf34") is None:
        raise SystemExit(f"{args.endf}: no MF34")

    print(f"\n=== JointCov.from_endf ({time.time() - t0:.0f} s so far) ===",
          flush=True)
    joint = JointCov.from_endf(args.endf, mt=2, l_max=L_MAX)
    print(f"  sigma : {joint.n_sigma} bins on "
          f"[{joint.grid_sigma_ev[0]:.6g}, {joint.grid_sigma_ev[-1]:.6g}] eV")
    print(f"  a_l   : {joint.n_a_bins} bins x {joint.l_max} orders on "
          f"[{joint.grid_a_ev[0]:.6g}, {joint.grid_a_ev[-1]:.6g}] eV "
          f"(union of every stored block's grid)")
    print(f"  joint : {joint.n} parameters, "
          f"{joint.matrix.nbytes / 1e6:.0f} MB, cross={'yes' if joint.has_cross else 'ZERO'}")

    # ── the report, on OUR range only ────────────────────────────────────────
    print(f"\n=== check() on [{E_MIN_MEV}, {E_MAX_MEV}] MeV ===", flush=True)
    ours = joint.restrict(E_MIN_MEV * 1e6, E_MAX_MEV * 1e6)
    print(f"  restricted to {ours.n_sigma} sigma + {ours.n_a_bins} a_l bins "
          f"({ours.n} params) — the host remainder is NOT modelled here, it is "
          f"the merge's business (§10.7-7)")
    print(ours.check(eigen=args.eigen))

    # ── the gate ─────────────────────────────────────────────────────────────
    df = pd.read_parquet(args.parquet)
    df = df[df["library"] == args.library]
    if df.empty:
        raise SystemExit(f"no {args.library} rows in {args.parquet}")
    order = (df.groupby("experiment_id").size()
             .sort_values(ascending=False).index[:args.n_experiments])

    print(f"\n=== equivalence gate, cross = 0 ===", flush=True)
    worst = 0.0
    for exp in order:
        sub = df[df["experiment_id"] == exp]
        if len(sub) > args.max_points:
            sub = sub.iloc[:args.max_points]
        e = sub["energy_mev"].to_numpy(float)
        mu = sub["mu"].to_numpy(float)
        c0 = sub["c0"].to_numpy(float)
        y = sub["y_eval"].to_numpy(float)
        a_l = np.array([interp_a_l_to_energy(lib, float(x), L_MAX) for x in e])

        legacy = (
            build_mf34_block(lib["mf34"], e, mu, c0, a_l)
            + build_mf33_block(lib.get("mf33_grid_ev"), lib.get("mf33_rel_cov"),
                               lib["energies_mf4_mev"], e, y)
        )
        lo_mev, hi_mev = _mf4_bin_edges_for_points(lib["energies_mf4_mev"], e)
        new = joint.fold(
            e, mu, c0, a_l, y, dtype=np.float64,
            sigma_window_ev=(lo_mev * 1e6, hi_mev * 1e6),
            sigma_map="overlap", a_map="nearest",
        )
        scale = max(np.abs(legacy).max(), 1e-300)
        rel = float(np.abs(new - legacy).max() / scale)
        worst = max(worst, rel)
        lam = float(np.linalg.eigvalsh(new).min()
                    / max(np.abs(np.diag(new)).max(), 1e-300))
        print(f"  {exp:<12} N={len(sub):>6}  max|diff|/scale = {rel:.3e}  "
              f"lam_min(Sigma)/scale = {lam:+.3e}")

    print(f"\n  WORST over {len(order)} experiments: {worst:.3e}")
    ok = worst < 1e-10
    print(f"  VERDICT: {'PASS — the object is a no-op, step 2 may proceed' if ok else 'FAIL — do NOT switch anything on'}")
    print(f"\n({time.time() - t0:.0f} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
