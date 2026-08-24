#!/usr/bin/env python3
"""P1.3 step 1 — build the split-combine tape on the cluster, and gate it.

`docs/post_run92_verification_roadmap.md` §2.6.  Runs on the ASNR cluster (the
11 GB chi2 sidecars and the host tape live there; the workstation cannot read
them).  Writes one 206 MB ENDF and a report.  Scores nothing.

WHAT IT DOES
    1. Stages a rebuild directory: the run's own MF33 sidecars, its
       `nominal_fits.parquet` and its `run_metadata.json` -- so every knob
       (near-zero guard, rho_min, PENDF tolerance, host tape, NJOY) comes from
       the run itself and the rebuild is SINGLE-VARIABLE -- with
       `mf33_absolute_covariance.npy` replaced by P1.2's product.
    2. Calls `scripts.rebuild_mf33`, the same code the pipeline calls.
    3. ⚑ GATES THE WRITTEN TAPE before any cluster time is spent scoring it.

⚠ ONLY THE `_mg` TAPE IS STAGED, and that is deliberate on two counts:
    * the share sits near 91 % full, and the fine tape is 843 MB against 206 MB;
    * `precompute_chi2_predictive.py` reads `26-Fe-56g_nominal_mg.endf` by
      default (`KIKA_THIS_WORK_ENDF`), so it is the tape that gets scored.

⚠⚠ AND IT IS THE NO-CROSS TAPE, WHICH IS THE WHOLE POINT OF PRICING HERE FIRST.
    Changing MF33's marginals breaks the Cauchy-Schwarz compatibility of the a0
    cross block, which is compatible only with the marginals it was built from
    (§10.1 -- this is what killed runs 89 and 90).  So the cross tape cannot be
    reused and must not be.  Pricing on the no-cross tape is also where
    §10.8-19 measured energy-local freedom at -68 % / 5 %, so the comparison
    lands on the number it should.  The baseline is therefore `91_rewrite`.

THE GATE, and it is cheap insurance against an hour of scoring
    The tape must read back carrying the REDISTRIBUTION, not the shipped
    structure.  Expected from P1.2, in relative units:

        shipped     coh 0.900   PR  1.78   L_corr 1.73 MeV
        split-loc   coh 0.391   PR  8.06   L_corr 0.098 MeV     <- this one

    The diagonal must be unchanged (this declares no new uncertainty), and the
    matrix must be PSD.  If the near-zero guard or the ENDF writer flattens the
    redistribution, PR comes back near 1.8 and the job is not worth launching.

    python p13_build_split_tape.py --run RUN_DIR --split-dir P12_DIR --out OUT_DIR
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

# The cluster runs these from inside `EXFOR/scripts/`, where `scripts` is not on
# sys.path as a package even though `__init__.py` is there.  Same three lines
# `precompute_chi2_predictive.py:113` uses; without them `from
# scripts.rebuild_mf33 import rebuild` raises ModuleNotFoundError (job 8480875).
_kika_root = Path(__file__).resolve().parent.parent
if str(_kika_root) not in sys.path:
    sys.path.insert(0, str(_kika_root))

# Acceptance thresholds, fixed before the build.
PR_MIN = 4.0            # shipped is 1.78; P1.2 predicts 8.06
COH_MAX = 0.60          # shipped is 0.900; P1.2 predicts 0.391
DIAG_RTOL = 1e-6
PSD_TOL = -1e-8         # relative to the matrix scale


def descriptors(cov_rel: np.ndarray) -> dict:
    n = cov_rel.shape[0]
    sd = np.sqrt(np.clip(np.diag(cov_rel), 0.0, None))
    sigma_pt = float(np.median(sd))
    sigma_coh = float(np.sqrt(max(float(cov_rel.sum()), 0.0)) / n)
    tr = float(np.trace(cov_rel))
    fro2 = float(np.sum(cov_rel ** 2))
    return {
        "n": n,
        "sigma_pt": sigma_pt,
        "sigma_coh": sigma_coh,
        "coh": sigma_coh / sigma_pt if sigma_pt > 0 else float("nan"),
        "PR": (tr ** 2 / fro2) if fro2 > 0 else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="completed run dir (run 92)")
    ap.add_argument("--split-dir", required=True,
                    help="P1.2 output dir holding the replacement "
                         "mf33_absolute_covariance.npy")
    ap.add_argument("--out", required=True, help="where the patched tape goes")
    ap.add_argument("--stage", default=None,
                    help="staging dir (default <out>/_stage)")
    ap.add_argument("--variant", default="mf33_absolute_covariance.npy",
                    help="which P1.2 matrix to use; pass "
                         "mf33_absolute_covariance_splitdiag.npy for the "
                         "upper-bound variant")
    ap.add_argument("--skip-gate", action="store_true",
                    help="build without reading the tape back (NOT recommended)")
    ap.add_argument("--pendf-cache-dir",
                    default="/share_snc/snc/JuanMonleon/cache/kika_pendf_cache",
                    help="durable PENDF cache; patched into the STAGED "
                         "run_metadata.json only")
    args = ap.parse_args()

    run = Path(args.run)
    split = Path(args.split_dir)
    out = Path(args.out)
    stage = Path(args.stage) if args.stage else out / "_stage"
    out.mkdir(parents=True, exist_ok=True)
    stage.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("P1.3 step 1 — build the split-combine tape")
    print("=" * 78)
    print(f"  run        {run}")
    print(f"  split      {split / args.variant}")
    print(f"  out        {out}")

    # ---- stage ----------------------------------------------------------
    needed = ["mf33_energy_grid_ev.npy", "mf33_c0_nominal.npy",
              "nominal_fits.parquet", "run_metadata.json"]
    for f in needed:
        src = run / f
        if not src.exists():
            print(f"missing required input: {src}", file=sys.stderr)
            return 2
        shutil.copy2(src, stage / f)

    # Point the rebuild at the durable PENDF cache on the share.  Run 92 recorded
    # MF33_PENDF_CACHE_DIR = None, so `build_mf33_denominator` falls back to the
    # library default and job 8480883 paid a full RECONR (~8 min) for a tape that
    # is already reconstructed.  The cache is keyed on sha256+tolerance, so this
    # is INERT for the result and only buys wall-clock -- and it is patched into
    # the STAGED copy of run_metadata.json, never into the run's own.
    if args.pendf_cache_dir:
        import json
        meta_p = stage / "run_metadata.json"
        meta = json.loads(meta_p.read_text())
        meta["config"]["MF33_PENDF_CACHE_DIR"] = args.pendf_cache_dir
        meta_p.write_text(json.dumps(meta, indent=2))
        cached = Path(args.pendf_cache_dir)
        n_hit = len(list(cached.glob("*.pendf"))) if cached.is_dir() else 0
        print(f"\n  PENDF cache -> {args.pendf_cache_dir}  ({n_hit} tapes cached)")
        if n_hit == 0:
            print("   ⚠ empty or unreadable: RECONR will run (~8 min), not fatal")

    new_cov = split / args.variant
    if not new_cov.exists():
        print(f"missing P1.2 product: {new_cov}", file=sys.stderr)
        return 2
    shutil.copy2(new_cov, stage / "mf33_absolute_covariance.npy")

    mg = sorted(run.glob("*_nominal_mg.endf"))
    if not mg:
        print(f"no *_nominal_mg.endf in {run}", file=sys.stderr)
        return 2
    mg_src = mg[0]
    print(f"\n  staging the _mg tape only ({mg_src.stat().st_size / 1e6:.0f} MB); "
          f"the 843 MB fine tape is deliberately not staged")
    shutil.copy2(mg_src, stage / mg_src.name)

    # Sanity: the replacement must have the run's own shape and diagonal.
    cov_new = np.load(stage / "mf33_absolute_covariance.npy")
    cov_old = np.load(run / "mf33_absolute_covariance.npy")
    if cov_new.shape != cov_old.shape:
        print(f"shape mismatch {cov_new.shape} vs {cov_old.shape}", file=sys.stderr)
        return 2
    d_new, d_old = np.diag(cov_new), np.diag(cov_old)
    dmax = float(np.max(np.abs(d_new - d_old) / np.maximum(np.abs(d_old), 1e-300)))
    print(f"  diagonal vs the shipped matrix: max |rel diff| = {dmax:.3e}"
          f"   ({'unchanged, as designed' if dmax < DIAG_RTOL else '⚠ CHANGED'})")
    if dmax >= DIAG_RTOL:
        print("   ⚠ the split combine must not alter the declared variance; "
              "this is a redistribution only", file=sys.stderr)
        return 3
    del cov_old

    # ---- rebuild --------------------------------------------------------
    print("\n-- rebuilding MF33 (same code path as the pipeline)")
    from scripts.rebuild_mf33 import rebuild

    written = rebuild(
        stage,
        out_dir=out,
        in_place=False,
        mg_representation="fine",   # run 92: MF33_MG_REPRESENTATION = 'fine'
        rebuild_mt1=True,           # run 92: MF33_REBUILD_MT1 = True
    )
    print("\n  written:")
    for p in written:
        print(f"    {p}")

    if args.skip_gate:
        print("\n⚠ gate skipped by request")
        return 0

    # ---- gate -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("GATE — read the written tape back")
    print("=" * 78)
    from kika.endf import read_endf

    tape = next((Path(p) for p in written if str(p).endswith("_mg.endf")), None)
    if tape is None:
        print("no _mg tape among the written products", file=sys.stderr)
        return 3

    def read_mf33_mt2(path: Path):
        """Largest MF33 MT2 block on a tape, with all block sizes reported.

        The section is range-MERGED into the host, so it can carry several
        blocks; taking `matrices[0]` blindly could describe a host block outside
        the analysis window and fail the gate for the wrong reason.
        """
        h = read_endf(str(path), mf_numbers=[33])
        dec = h.get_file(33).sections[2].to_xs_covmat()
        mats = [np.asarray(x, dtype=float) for x in dec.matrices]
        sizes = [x.shape[0] for x in mats]
        k = int(np.argmax(sizes))
        m = 0.5 * (mats[k] + mats[k].T)
        return m, sizes, k

    m, sizes, k = read_mf33_mt2(tape)
    print(f"  new tape   MF33 MT2 blocks {sizes}, using block {k} ({m.shape[0]})")

    # ⚑ The gate is RELATIVE, against run 92's own tape read the same way. The
    #   recentring, the near-zero guard and the ENDF grid all act on both, so a
    #   like-for-like comparison is the only thing that isolates what changed.
    ref_tape = next(iter(sorted(Path(args.run).glob("*_nominal_mg.endf"))), None)
    if ref_tape is None:
        print("  ⚠ no reference _mg tape in the run dir; gate falls back to "
              "absolute thresholds")
        d_ref = None
    else:
        m_ref, sizes_ref, k_ref = read_mf33_mt2(ref_tape)
        print(f"  run 92     MF33 MT2 blocks {sizes_ref}, using block {k_ref} "
              f"({m_ref.shape[0]})")
        d_ref = descriptors(m_ref)
        scale_ref = float(np.trace(m_ref) / m_ref.shape[0])
        psd_ref = float(np.linalg.eigvalsh(m_ref)[0]) / max(scale_ref, 1e-300)
        if m_ref.shape != m.shape:
            print(f"  ⚠ block shapes differ ({m.shape[0]} vs {m_ref.shape[0]}); "
                  f"the comparison below is not like-for-like")
        else:
            dg = float(np.max(np.abs(np.diag(m) - np.diag(m_ref))
                              / np.maximum(np.abs(np.diag(m_ref)), 1e-300)))
            print(f"  diagonal on the TAPE vs run 92: max |rel diff| = {dg:.3e}")
        del m_ref

    d = descriptors(m)
    lam = float(np.linalg.eigvalsh(m)[0])
    scale = float(np.trace(m) / m.shape[0])
    print(f"\n  {'':<10}{'sigma_pt':>10}{'sigma_coh':>11}{'coh':>8}{'PR':>9}")
    if d_ref:
        print(f"  {'run 92':<10}{d_ref['sigma_pt']:>10.4f}{d_ref['sigma_coh']:>11.4f}"
              f"{d_ref['coh']:>8.3f}{d_ref['PR']:>9.2f}")
    print(f"  {'new':<10}{d['sigma_pt']:>10.4f}{d['sigma_coh']:>11.4f}"
          f"{d['coh']:>8.3f}{d['PR']:>9.2f}")
    print(f"  lambda_min/scale = {lam / max(scale, 1e-300):.3e}")
    print("  P1.2, before the recentring/guard/grid: coh 0.900 -> 0.391, "
          "PR 1.78 -> 8.06")

    ok = True
    if d_ref and np.isfinite(d_ref["PR"]) and d_ref["PR"] > 0:
        # Relative form: the redistribution must survive to the tape.
        if not (d["PR"] / d_ref["PR"] >= 2.0):
            print(f"\n  ❌ PR rose only {d['PR'] / d_ref['PR']:.2f}x over run 92 "
                  f"({d_ref['PR']:.2f} -> {d['PR']:.2f}).")
            ok = False
        if not (d["coh"] < d_ref["coh"]):
            print(f"  ❌ coh did not fall ({d_ref['coh']:.3f} -> {d['coh']:.3f}).")
            ok = False
    else:
        if not (d["PR"] >= PR_MIN):
            print(f"\n  ❌ PR {d['PR']:.2f} < {PR_MIN}.")
            ok = False
        if not (d["coh"] <= COH_MAX):
            print(f"  ❌ coh {d['coh']:.3f} > {COH_MAX}.")
            ok = False
    # ⚑ PSD, and it must be RELATIVE too -- the third instance of the same
    #   design error (job 8480883 failed here and the matrix was innocent).
    #   No merged MF33 tape is PSD to 1e-8: the ENDF round-trip writes ~6
    #   significant digits and `merge_mf33_covariance_into_host` says so itself
    #   ("Merged MT2 relative covariance not PSD ... warn only, not repaired").
    #   MEASURED 2026-08-12 on run 92's own shipped deliverable:
    #       lambda_min/scale = -3.132045e-05
    #   and the split tape returned -3.132e-05 -- the same number to four
    #   figures. The question is therefore never "is it PSD to 1e-8" but "is it
    #   as PSD as the tape we already ship, read back the same way".
    psd_new = lam / max(scale, 1e-300)
    if d_ref is not None and psd_ref < 0.0:
        print(f"  PSD floor: run 92 {psd_ref:.6e}   new {psd_new:.6e}"
              f"   ({psd_new / psd_ref:.2f}x the shipped floor)")
        if psd_new < 2.0 * psd_ref:
            print(f"  ❌ more than 2x more negative than the shipped tape's own "
                  f"round-trip floor.")
            ok = False
    elif psd_new < PSD_TOL:
        print(f"  ❌ not PSD to tolerance ({psd_new:.3e}).")
        ok = False

    if ok:
        print("\n  ✅ GATE PASSED — the tape carries the redistribution, the "
              "diagonal is unchanged and it is PSD.")
        print("     Proceed to step 2 (precompute + score). See run_p13.sh.")
        return 0
    print("\n  ⛔ GATE FAILED — do NOT launch the scoring job. Most likely the "
          "near-zero guard\n     (REGULARIZE_NEAR_ZERO_REL_UNC) or the MF33 "
          "group grid is flattening the\n     redistribution; diagnose that "
          "before spending an hour of cluster time.")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
