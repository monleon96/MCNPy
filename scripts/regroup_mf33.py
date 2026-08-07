#!/usr/bin/env python
"""Re-write a shipped `_mg.endf` with MF33/MT2 on the adaptive GROUP grid.

WHAT THIS IS FOR (roadmap §10.7-4 step 2). §10.7-3 argues that grouping MF33 in
the multigroup file is the *enabling* step for the cross term, not a size
optimisation. That argument costs something — grouping takes the relative sigma
from median 0.0616 to 0.0526 and its PEAK from 0.204 to 0.113 — and the price
has to be paid in chi2, not asserted. This script produces the single-variable
artefact that prices it.

**It is not a pipeline run.** Both MF33 representations already exist in the run
directory; the multigroup one is simply merged into a COPY of the shipped file,
replacing the fine self-covariance inside its own energy span and leaving the
host's values outside it untouched (`merge_mf33_covariance_into_host`). MF34,
MF4, MF3 and every other section are byte-identical to the source, so scoring
this against run 86 moves exactly one thing.

⚠ MT2 ONLY, and that is deliberate. `precompute_chi2_library_c0` reads the MT=2
self-block and nothing else ("No MT=2 self-block in MF33 covariance" is a hard
error there), so regrouping MT1 as well would change the file without changing
the chi2 — two variables for the price of one. MT1 stays as shipped.

WHY GROUPING IS ALSO A CONDITIONING FIX, which is the part that is easy to miss.
Both matrices are PSD and full rank *before* the file. The fine one has
lam_max/lam_min = 6.365/4.00e-09, a condition number of **1.6e9**, and ENDF-6 is
a six-significant-digit ASCII format: it cannot carry that, which is why the
shipped MF33 comes back with 237 negative eigenvalues and rank 1916/2317. The
grouped one sits at 0.356/3.62e-07 — condition **1.0e6** — and survives. This
script re-reads what it wrote and prints both spectra so the claim is checked on
the artefact rather than on the inputs.

Run:
  <venv>/bin/python regroup_mf33.py --source <_mg.endf> --run-dir <dir> \
      --out <path_mg_mf33grouped.endf>
  ... add --check to measure and write nothing.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ZA, MT = 26056, 2
NULL_TOL = 1e-10


def _spectrum(name, A):
    A = np.asarray(A, float)
    w = np.linalg.eigvalsh(0.5 * (A + A.T))
    keep = w > NULL_TOL * max(w.max(), 0.0)
    kmin = float(w[keep].min()) if keep.any() else float("nan")
    return {
        "matrix": name, "n": A.shape[0], "rank": int(keep.sum()),
        "lam_max": float(w.max()), "lam_min_kept": kmin,
        "cond": float(w.max()) / kmin if kmin > 0 else np.inf,
        "n_neg": int((w < 0).sum()), "lam_most_neg": float(w.min()),
    }


def _table(rows):
    cols = list(rows[0])
    fmt = {"lam_max": "{:.4e}", "lam_min_kept": "{:.4e}", "cond": "{:.3e}",
           "lam_most_neg": "{:+.4e}"}
    w = {c: len(c) for c in cols}
    txt = []
    for r in rows:
        t = {c: (fmt[c].format(r[c]) if c in fmt and np.isfinite(r[c])
                 else str(r[c])) for c in cols}
        for c in cols:
            w[c] = max(w[c], len(t[c]))
        txt.append(t)
    out = ["  ".join(c.ljust(w[c]) for c in cols),
           "  ".join("-" * w[c] for c in cols)]
    out += ["  ".join(t[c].ljust(w[c]) for c in cols) for t in txt]
    return "\n".join(out)


def _sqrt_diag(name, grid_ev, C, lo=0.85e6, hi=4.0e6):
    d = np.sqrt(np.maximum(np.diag(C), 0.0))
    cen = 0.5 * (np.asarray(grid_ev)[:-1] + np.asarray(grid_ev)[1:])
    m = (cen >= lo) & (cen <= hi)
    s = d[m] if m.any() else d
    return (f"  {name:<34} n={len(d):5d}  sqrt-diag over [0.85,4] MeV: "
            f"min {s.min():.5f}  median {np.median(s):.5f}  max {s.max():.5f}")


def _discarded_structure(grid_f, C_f, grid_c):
    """||C_f - P C_f P^T|| / ||C_f||, the quantity §10.7-2(a) says decides it.

    P = D A with A the width-weighted collapse and D the duplication. This is
    the fine structure the collapse throws away; it is invisible to a joint
    certified on the coarse axis, it is indefinite in general, and
    `scripts/tests/test_fold_maps.py` measures the fold's violation turning on
    as a function of exactly this.
    """
    nf, nc = C_f.shape[0], len(grid_c) - 1
    cen, wid = 0.5 * (grid_f[:-1] + grid_f[1:]), np.diff(grid_f)
    g = np.clip(np.searchsorted(grid_c, cen, side="right") - 1, 0, nc - 1)
    A = np.zeros((nc, nf))
    A[g, np.arange(nf)] = wid
    A /= np.maximum(A.sum(1, keepdims=True), 1e-300)
    D = np.zeros((nf, nc))
    D[np.arange(nf), g] = 1.0
    R = C_f - (D @ A) @ C_f @ (D @ A).T
    w = np.linalg.eigvalsh(0.5 * (R + R.T))
    return np.linalg.norm(R) / np.linalg.norm(C_f), float(w.min()), float(w.max())


def _read_mf33_mt2(path):
    """(grid_ev, relative matrix) of the MT=2 self block, as the chi2 reads it.

    Same selection as `precompute_chi2_library_c0`: the MT=2 self-block, and
    the grids come back in eV whatever `energy_unit` says (that parameter is a
    label for MF34 and a no-op — roadmap §10.7-2(c)).
    """
    from kika.cov import CrossSectionCovariance

    cov = CrossSectionCovariance.from_endf(str(path), mf=33)
    for i in range(cov.num_matrices):
        if (int(cov.reaction_rows[i]) == MT and int(cov.reaction_cols[i]) == MT
                and int(cov.isotope_rows[i]) == ZA):
            if not bool(cov.is_relative[i]):
                raise SystemExit("MT2 self block came back absolute; expected relative")
            return (np.asarray(cov.energy_grids[i], float),
                    np.asarray(cov.matrices[i], float))
    raise SystemExit("no MT=2 self block in the written MF33")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="the shipped _mg.endf")
    ap.add_argument("--run-dir", required=True, help="supplies the two MF33 .npy")
    ap.add_argument("--out", help="destination; required unless --check")
    ap.add_argument("--check", action="store_true",
                    help="measure the inputs and write nothing")
    args = ap.parse_args()

    src, run_dir = Path(args.source), Path(args.run_dir)
    grid_c = np.load(run_dir / "mf33_multigroup_grid_ev.npy").astype(float)
    cov_c = np.load(run_dir / "mf33_multigroup_relative_covariance.npy")
    grid_f = np.load(run_dir / "mf33_energy_grid_ev.npy").astype(float)
    cov_f = np.load(run_dir / "mf33_relative_covariance.npy")

    print(f"source : {src.name}  ({src.stat().st_size} bytes)")
    print(f"run dir: {run_dir}\n")
    print("=== the two representations, as produced (before any ENDF) ===")
    print(_sqrt_diag("fine  (what run 86 ships)", grid_f, cov_f))
    print(_sqrt_diag("group (what this writes)", grid_c, cov_c))
    print()
    print(_table([_spectrum("fine, as produced", cov_f),
                  _spectrum("group, as produced", cov_c)]))

    frac, lo, hi = _discarded_structure(grid_f, cov_f, grid_c)
    print(f"\n  ||C_f - P C_f P^T|| / ||C_f|| = {frac:.4f}   "
          f"discarded part's spectrum: [{lo:+.4g}, {hi:+.4g}]")
    print("  That is what a joint certified on the coarse axis cannot see, and\n"
          "  what makes the fold stop being a congruence. §10.7-2(a).")

    if args.check:
        print("\n--check: nothing written.")
        return 0
    if not args.out:
        raise SystemExit("--out is required unless --check")

    out = Path(args.out)
    print(f"\ncopying {src.name} -> {out.name} ...", flush=True)
    shutil.copyfile(src, out)

    from kika.endf.writers.mf33_writer import (
        merge_mf33_covariance_into_host, write_mf33_to_file,
    )

    print("merging the GROUP matrix into MT2 (MT1 and every other section "
          "stay as shipped) ...", flush=True)
    section = merge_mf33_covariance_into_host(
        host_endf=str(out), cov_matrix=cov_c, energy_grid_ev=grid_c, mt=MT,
    )
    write_mf33_to_file(str(out), section, str(out))
    print(f"  written: {out}  ({out.stat().st_size} bytes, "
          f"{out.stat().st_size - src.stat().st_size:+d})")

    # ── verify on the ARTEFACT, not on the inputs ────────────────────────────
    print("\n=== re-reading what was written ===", flush=True)
    g_new, c_new = _read_mf33_mt2(out)
    g_old, c_old = _read_mf33_mt2(src)
    print(_sqrt_diag("MF33/MT2 from the SOURCE file", g_old, c_old))
    print(_sqrt_diag("MF33/MT2 from the NEW file", g_new, c_new))
    print()
    print(_table([_spectrum("source file, post-ENDF", c_old),
                  _spectrum("new file, post-ENDF", c_new)]))
    print("\n  THE POINT: the source row carries negative eigenvalues because a\n"
          "  condition number of ~1e9 cannot survive six significant digits.\n"
          "  If the new row does not, grouping bought a strictly valid MF33 as\n"
          "  well as the size. §10.7-3.")

    s_new = _spectrum("", c_new)
    if s_new["n_neg"] and abs(s_new["lam_most_neg"]) > 1e-6 * s_new["lam_max"]:
        print("\n  *** the regrouped MF33 is NOT clean — do not score it "
              "before explaining this.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
