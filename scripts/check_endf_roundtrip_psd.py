#!/usr/bin/env python
"""Re-run the PSD diagnostic on what the WRITTEN FILE says, not on what we meant.

WHY THIS IS A SEPARATE GATE (roadmap Sec. 10.1.8-L). Two things happen between
`build_group_cross --write-endf` and a consumer reading the result, and each can
break a matrix that was PSD in memory:

1. SERIALISATION. ENDF's 11-character float carries ~6-7 significant figures, so
   every entry is perturbed at ~1e-7 relative. The joint sits at
   sigma_max(K) = 0.99994 -- essentially ON the Cauchy-Schwarz boundary by
   construction -- and 1e-7 of slop is the same order as the 6e-5 of headroom.

2. WINDOWING. The (L=0, L1) blocks are written only over the range where EXFOR
   constrains them, so every magnitude group outside it has a zero cross row.
   Zeroing rows of the off-diagonal block of a PSD matrix is NOT in general
   PSD-preserving, so it has to be measured rather than assumed.

Everything is done in the file's own RELATIVE units. Converting to absolute is
the diagonal congruence diag(I, D_a) with D_a = diag(a_l group means).

⚠⚠ THE ARGUMENT THIS FILE WAS WRITTEN ON IS NO LONGER TRUE, AND IT MATTERS.
It said D_a "has no zero entries (the writer refuses otherwise)", so that the
congruence preserved the sign of every eigenvalue and the relative verdict WAS
the absolute verdict. Sec. 10.1.8-L9 killed both halves:

  * D_a has **1542 zero entries of 4218** (37 %) -- (group, order) slots where no
    fine bin is valid, so `row_aggregator` returns an all-zero row by design.
  * The writer no longer refuses. It preserves the SHIPPED relative values there,
    because the absolute round trip (multiply by 0, divide by 0) destroys them
    and the rebuild spans no such direction.

With a SINGULAR D_a the congruence is one-way: M PSD => D_a M D_a PSD, but NOT
the converse -- the null directions are annihilated rather than sign-preserved.
So a NEGATIVE eigenvalue found here can be a **false alarm** for the chi2, which
only ever sees the absolute joint, and it will most likely live exactly in the
1542 preserved-value directions that absolute space cannot see.

**Before trusting this gate again:** restrict it to the live sub-block
(a_l != 0) and diagnose that, or apply D_a and diagnose in absolute units. As it
stands it over-reports. It is NOT on run 90's path -- nothing calls it.

Run 89's lesson is that a construction is only PSD together with the marginals it
is actually shipped with. The only trustworthy check is the one performed on the
artefact.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_group_cross import L_MAX, diagnose  # noqa: E402

ZA, MT = 26056, 2


def read_mf34_blocks(endf_path: Path):
    """Shape (L>=1) and cross (L=0) blocks as stored, keeping the L=0 row.

    `build_group_cross.mf34_group_edges_ev` deliberately drops L=0; here it is
    the point of the exercise, so this reader keeps both and returns them apart.
    """
    from kika.cov import MF34CovMat

    m = MF34CovMat.from_endf(str(endf_path), energy_unit="MeV")
    m = m.filter_by_isotope_reaction(ZA, MT)
    shape, cross = {}, {}
    for k in range(m.num_matrices):
        lr, lc = int(m.l_rows[k]), int(m.l_cols[k])
        if not bool(m.is_relative[k]):
            raise SystemExit(f"block ({lr},{lc}) came back absolute; expected relative")
        grid = np.asarray(m.energy_grids[k], float)
        mat = np.asarray(m.matrices[k], float)
        if lr >= 1 and lc >= 1:
            shape[(lr, lc)] = (grid, mat)
        elif lr == 0 and lc >= 1:
            cross[lc] = (grid, mat)
        elif lc == 0 and lr >= 1:
            cross[lr] = (grid, mat.T)
    return shape, cross


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endf", required=True, help="the file to re-read")
    ap.add_argument("--run-dir", required=True, help="run dir supplying MF33")
    args = ap.parse_args()

    endf, run_dir = Path(args.endf), Path(args.run_dir)

    mag_ev = np.load(run_dir / "mf33_multigroup_grid_ev.npy").astype(float)
    c33 = np.load(run_dir / "mf33_multigroup_relative_covariance.npy")
    n_gm = len(mag_ev) - 1

    print(f"reading {endf.name} ({endf.stat().st_size} bytes) ...", flush=True)
    shape, cross = read_mf34_blocks(endf)
    if not cross:
        raise SystemExit("no (L=0, L1) cross blocks in the file — nothing to check")
    print(f"  {len(shape)} shape blocks, {len(cross)} cross blocks")

    shape_ev = max((g for g, _ in shape.values()), key=len)
    n_gs = len(shape_ev) - 1
    # The parser's energy unit only has to be consistent between the two axes;
    # infer the factor that puts the cross rows on the magnitude grid's scale.
    row_ev = next(iter(cross.values()))[0]
    scale = 1.0 if row_ev.max() > 1e4 else 1e6
    print(f"  shape grid {n_gs} groups, magnitude grid {n_gm} groups "
          f"(row-grid unit factor {scale:g})")

    # --- reassemble the joint, in the file's relative units -------------------
    c34 = np.zeros((n_gs * L_MAX, n_gs * L_MAX))
    for (lr, lc), (_, mat) in shape.items():
        r = np.arange(n_gs) * L_MAX + (lr - 1)
        c = np.arange(n_gs) * L_MAX + (lc - 1)
        c34[np.ix_(r, c)] = mat
    c34 = 0.5 * (c34 + c34.T)

    cx = np.zeros((n_gm, n_gs * L_MAX))
    centres = 0.5 * (mag_ev[:-1] + mag_ev[1:])
    for l1, (grid, mat) in sorted(cross.items()):
        rev = grid * scale
        # Put the windowed rows back on the full magnitude grid; groups the file
        # does not cover keep a zero cross row, which is what the file asserts.
        inside = (centres >= rev[0]) & (centres <= rev[-1])
        ridx = np.clip(np.searchsorted(rev, centres[inside], side="right") - 1,
                       0, mat.shape[0] - 1)
        cols = np.arange(n_gs) * L_MAX + (l1 - 1)
        cx[np.ix_(np.where(inside)[0], cols)] = mat[ridx, :]

    n_rows = int((np.abs(cx).sum(1) > 0).sum())
    print(f"  cross rows carried by the file: {n_rows}/{n_gm}")
    print(f"  ||Cx||_F (relative, from file) = {np.linalg.norm(cx):.4g}")

    rows = [
        diagnose("AS WRITTEN: file MF34 + file cross", c33, c34, cx),
        diagnose("file MF34, Cx = 0 (isolates serialisation)",
                 c33, c34, np.zeros_like(cx)),
    ]
    print("\n=== PSD of the joint the FILE describes (relative units) ===")
    print(pd.DataFrame(rows).set_index("case").to_string())

    lam = rows[0]["lam_min_norm"]
    tol = -1e-8
    print(f"\n  lam_min/scale = {lam:.3e}   gate: >= {tol:.0e}")
    if not np.isfinite(lam) or lam < tol:
        print("  *** FAIL — the artefact is not PSD. Do not score it.")
        return 1
    print("  PASS — the written file is a valid joint covariance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
