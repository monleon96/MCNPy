#!/usr/bin/env python
"""Build the joint (sigma, a_1..a_L) on the FINE grid, from one set of replicas.

⚠⚠ THIS IS A DIAGNOSTIC. THE PRODUCER IS `build_group_cross.py --mag-grid fine`.
Two scripts that both build the joint is how §L3 happened -- a matrix was
certified that was not the matrix being shipped -- so nothing downstream may
read this one's output, and it deliberately writes no files.

⚠⚠ AND EVERY CROSS-BLOCK NUMBER IT PRINTED BEFORE 2026-08-08 WAS TAKEN ON NOISE.
`_a_replicas` joined the TMC parquet on its raw `sample_idx`, which is the MC
draw PLUS ONE (row 0 is the nominal fit), so c0 replica k was paired with a_l
draw k-1 and row 0 was all zeros. Measured: within-bin corr(c0, a_l) 0.0060
median, below the 0.0100 noise floor, against 0.386 once the shift is applied
(roadmap §10.7-10, 0.5). **§10.7-8's `lam_min = -2.48e-16` therefore certified a
joint whose cross block carried no signal** -- true, because a sample covariance
of any stacked rows is PSD, but not evidence about the cross term. The
construction argument survives; that specific matrix does not. Fixed below; the
marginals were never affected.

ROADMAP §10.7-8. The point Juan made and the measurement that follows from it:
**on the fine grid MF33 and MF34 are already the same mesh** -- 1738 analysis
bins, [846822, 4.075e+06] eV -- and so is the cross term. The 188/660/703
divergence is created entirely by the adaptive collapse and the host splice at
WRITE time. So every cross-block experiment so far has been run downstream of a
divergence that does not exist upstream.

Here all three blocks come from ONE covariance over the stacked parameter vector

    z = [ c0(bin 0) .. c0(bin M-1) | a_1(bin 0) .. a_L(bin 0) | a_1(bin 1) .. ]

so the joint is a sample covariance of paired draws and is therefore **PSD by
construction**. Nothing is checked into being PSD and nothing is repaired. The
pipeline today computes the Pass-1 correlation TWICE -- once for c0 in
``combine_c0_covariance`` and once for a_l -- so the correlation BETWEEN the two
families is never formed as part of a single object. That is the hole.

MEASURED ABSOLUTE, ON PURPOSE. The relative form is ``D C D`` with D diagonal,
a congruence, so PSD carries over exactly; and where a nominal is ~0 the
pipeline zeroes the row and column, which is a projection and also preserves it.
Measuring absolute answers the same question without importing the near-zero
regularisation policy, which is a separate argument.

RANK. 12166 parameters from 10000 replicas, so the joint is PSD but SINGULAR:
about 2166 eigenvalues are exactly zero. That is arithmetic, not a defect, and
it costs nothing here -- the joint is never inverted. The chi^2 inverts
Sigma_eval per experiment, which is far smaller and is a congruence of this.

Run:
  <venv>/bin/python build_fine_joint.py --run-dir .../new_test_88_fullcross
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.joint_covariance import JointCov  # noqa: E402

L_MAX = 6


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def _c0_replicas(path: Path, n_bins: int, t0: float) -> np.ndarray:
    """The c0 replicas in the same family as the shipped MF33.

    ``combine_c0_covariance`` ships ``corr_pass1 * outer(std2, std2)``. The
    replica set whose covariance IS that is the affine rescale of Pass 1 onto
    Pass 2's marginals -- exactly what the a_l side already stores as its "TMC"
    samples. Building it here rather than reading it keeps the two families in
    one family, which is the whole point.
    """
    tbl = pq.read_table(path, columns=["sample_idx", "energy_index", "pass", "c0"])
    s = tbl.column("sample_idx").to_numpy()
    e = tbl.column("energy_index").to_numpy()
    p = np.asarray(tbl.column("pass").to_pylist())
    c = tbl.column("c0").to_numpy()
    del tbl
    n_s = int(s.max()) + 1
    # energy_index is the pipeline's bin id, not necessarily 0..M-1; map it.
    uniq = np.unique(e)
    assert uniq.size == n_bins, f"{uniq.size} energy_index values, {n_bins} bins"
    col = np.searchsorted(uniq, e)
    out = {}
    for tag in ("pass1", "pass2"):
        m = p == tag
        z = np.full((n_s, n_bins), np.nan)
        z[s[m], col[m]] = c[m]
        out[tag] = z
        _log(f"  {tag}: {np.isfinite(z).all(1).sum()}/{n_s} complete replicas", t0)
    p1, p2 = out["pass1"], out["pass2"]
    m1, m2 = np.nanmean(p1, 0), np.nanmean(p2, 0)
    s1, s2 = np.nanstd(p1, 0, ddof=1), np.nanstd(p2, 0, ddof=1)
    scale = np.where(s1 > 0, s2 / np.where(s1 > 0, s1, 1.0), 0.0)
    return m2[None, :] + (p1 - m1[None, :]) * scale[None, :], uniq


def _a_replicas(path: Path, uniq_e: np.ndarray, t0: float) -> np.ndarray:
    cols = ["sample_idx", "is_nominal", "energy_index"] + [f"a_{l}" for l in range(1, L_MAX + 1)]
    tbl = pq.read_table(path, columns=cols)
    keep = ~np.asarray(tbl.column("is_nominal").to_pylist(), dtype=bool)
    s = tbl.column("sample_idx").to_numpy()[keep]
    e = tbl.column("energy_index").to_numpy()[keep]
    a = np.column_stack([tbl.column(f"a_{l}").to_numpy()[keep]
                         for l in range(1, L_MAX + 1)])
    del tbl
    # ⚑ THE -1 IS LOAD-BEARING, AND ITS ABSENCE WAS A REAL BUG (roadmap
    # §10.7-10, 0.5). `save_all_legendre_coefficients` writes MC draw s as
    # `sample_idx = s + 1`, keeping row 0 for the nominal fit, while
    # `mf33_c0_samples.parquet` writes 0..N-1. Joining on the raw column
    # therefore paired c0 replica k with a_l draw k-1 and left row 0 all zero.
    #
    # Measured rather than argued: within-bin corr(c0, a_l) was 0.0060 median
    # without the shift -- BELOW the 1/sqrt(10000) = 0.0100 noise floor -- and
    # 0.386 with it, inside the 0.18-0.59 band §10.1 measured independently.
    # So this is what produced the "max|diff|/scale = 1.009, a total mismatch"
    # against `mf33_mf34_cross_covariance_full.npy`, and the Pass-1/Pass-2
    # family explanation is retracted.
    #
    # ⚠ EVERY CROSS-BLOCK NUMBER THIS SCRIPT PRINTED BEFORE 2026-08-08 WAS
    # TAKEN ON NOISE, §10.7-8's included. The joint was PSD because a sample
    # covariance always is, not because its cross term was right.
    #
    # `build_group_cross._read_a_chunk` has always done this correctly; that is
    # the producer. This script is a DIAGNOSTIC and must not become a second
    # one -- see the module docstring.
    s = s - 1
    n_s = int(s.max()) + 1
    col = np.searchsorted(uniq_e, e)
    z = np.zeros((n_s, uniq_e.size, L_MAX))
    z[s, col, :] = a
    _log(f"  a_l: {n_s} replicas x {uniq_e.size} bins x {L_MAX} orders", t0)
    # Energy slow, order fast -- JointCov's layout.
    return z.reshape(n_s, uniq_e.size * L_MAX)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--no-eigen", action="store_true")
    args = ap.parse_args()
    d = Path(args.run_dir)
    t0 = time.time()

    grid = np.load(d / "mf33_energy_grid_ev.npy")
    n_bins = grid.size - 1
    _log(f"fine grid: {n_bins} bins [{grid[0]:.6g}, {grid[-1]:.6g}] eV", t0)

    _log("loading c0 replicas ...", t0)
    z_c0, uniq_e = _c0_replicas(d / "mf33_c0_samples.parquet", n_bins, t0)
    _log("loading a_l replicas ...", t0)
    z_a = _a_replicas(d / "legendre_samples_tmc.parquet", uniq_e, t0)
    n_s = min(z_c0.shape[0], z_a.shape[0])
    z = np.concatenate([z_c0[:n_s], z_a[:n_s]], axis=1)
    del z_c0, z_a
    gc.collect()
    n_par = z.shape[1]
    _log(f"stacked replicas: {n_s} x {n_par} ({z.nbytes / 1e6:.0f} MB)", t0)

    z -= z.mean(0, keepdims=True)
    cov = (z.T @ z) / (n_s - 1)
    _log(f"joint covariance: {cov.shape} ({cov.nbytes / 1e6:.0f} MB)", t0)

    # ── does it reproduce what the pipeline shipped? ─────────────────────────
    ref = np.load(d / "mf33_absolute_covariance.npy")
    scale = max(np.abs(ref).max(), 1e-300)
    _log(f"c33 vs mf33_absolute_covariance.npy : "
         f"max|diff|/scale = {np.abs(cov[:n_bins, :n_bins] - ref).max() / scale:.3e}",
         t0)
    del ref
    xf = d / "mf33_mf34_cross_covariance_full.npy"
    if xf.exists():
        ref = np.load(xf)                        # (M, M, L) Cov(c0_i, a_l(j))
        got = cov[:n_bins, n_bins:].reshape(n_bins, n_bins, L_MAX)
        scale = max(np.abs(ref).max(), 1e-300)
        _log(f"cross vs mf33_mf34_cross_covariance_full.npy : "
             f"max|diff|/scale = {np.abs(got - ref).max() / scale:.3e}", t0)
        del ref, got
    gc.collect()

    joint = JointCov(grid, grid, L_MAX, cov,
                     sigma_is_relative=False, a_is_relative=False)
    print(f"\n=== FINE joint, one grid for everything ===")
    print(f"  {joint.n} parameters = {joint.n_sigma} sigma "
          f"+ {joint.n_a_bins} bins x {joint.l_max} orders")

    if not args.no_eigen:
        # The nonzero spectrum of Z^T Z equals that of Z Z^T, and the Gram
        # matrix is 10000 x 10000 against the covariance's 12166 x 12166 -- same
        # answer, less memory, and it makes the rank deficiency explicit instead
        # of leaving 2166 numerical zeros to be argued about.
        gram = (z @ z.T) / (n_s - 1)
        ev = np.linalg.eigvalsh(gram)
        del gram
        gc.collect()
        n_zero_exact = n_par - n_s
        lam_max = float(ev.max())
        lam_min = float(ev.min())
        print(f"\n  lambda_max        : {lam_max:.6e}")
        print(f"  lambda_min        : {lam_min:+.6e}   "
              f"(relative: {lam_min / lam_max:+.3e})")
        print(f"  negatives         : {int((ev < -1e-12 * lam_max).sum())} "
              f"of {n_s} possibly-nonzero")
        print(f"  structural zeros  : {n_zero_exact}  "
              f"({n_par} parameters from {n_s} replicas — PSD but singular)")
        print(f"  VERDICT           : "
              f"{'PSD' if lam_min >= -1e-10 * lam_max else 'NOT PSD'}")
        _log("spectrum done", t0)

    del z
    gc.collect()

    # ── and now group it, with ONE operator over the whole joint ─────────────
    gpath = d / "mf33_multigroup_grid_ev.npy"
    if gpath.exists():
        gg = np.load(gpath)
        print(f"\n=== collapsed to ONE group grid for both families "
              f"({gg.size - 1} groups) ===")
        k = joint.collapse(gg, gg)
        print(f"  {k.n} parameters = {k.n_sigma} sigma "
              f"+ {k.n_a_bins} bins x {k.l_max} orders")
        rep = k.check(eigen=not args.no_eigen)
        print(rep)
        if rep.min_eig is not None:
            print(f"  NB: PSD here is a THEOREM, not a measurement — A C A^T is "
                  f"a congruence. This line exists to catch a coding error, not "
                  f"to discover a property.")
        _log("collapse done", t0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
