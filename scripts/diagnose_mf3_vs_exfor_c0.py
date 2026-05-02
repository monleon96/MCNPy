"""Diagnostic — MF3 vs EXFOR-implied integrated σ, per AD bin.

For every fitted bin in v3's grid, computes:

    σ_implied_EXFOR,i = 4π · c_0_EXFOR,i      # free-c_0 Legendre fit on EXFOR
    σ_MF3,i           = bin-average MF3 cross section
    R_i               = σ_implied_EXFOR,i / σ_MF3,i − 1

R_i quantifies the disagreement between the cross section EXFOR data implies
in bin i and the cross section MF3 carries there. The choice between v3's
free-c_0 fit (Mode 1) and a pinned-c_0 fit (Mode 2) shifts every published
a_l in that bin by factor (1+R_i), and amplifies the reconstructed angular
variation by R_i/(1+R_i). If R_i is small everywhere, the choice is
cosmetic; if large in some bins, pinning distorts shapes there.

Reuses v3's CONFIG and STEP 1–5 helpers, so the diagnostic runs on EXACTLY
the same bins / EXFOR filtering / fit settings as production.

Outputs (under OUTPUT_DIR):
  - r_i_per_bin.csv     : per-bin table
  - r_i_summary.txt     : percentiles + threshold counts + decision aid
  - r_i_histogram.png   : if matplotlib is available
  - diagnose_r_i_<ts>.log
"""
from __future__ import annotations

import os
import sys
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Pin BLAS threads before numpy import (mirror v3 — fits use BLAS internally).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

_kika_path = Path(__file__).parent.parent
if str(_kika_path) not in sys.path:
    sys.path.insert(0, str(_kika_path))

from kika.exfor import read_all_exfor
from scripts.exfor_utils import (
    DualLogger,
    build_exfor_cache_from_objects,
    build_union_energy_grid,
    compute_energy_bins_with_tof_resolution,
)
from scripts import exfor_to_endf_sampling_v3 as v3


# =========================================================================
# CONFIG — overrides for v3 defaults (None = use v3.* value)
# =========================================================================
OUTPUT_DIR_OVERRIDE: Optional[str] = None   # None → use v3.OUTPUT_DIR


# =========================================================================
def main() -> int:
    out_dir = Path(OUTPUT_DIR_OVERRIDE or v3.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = out_dir / f'diagnose_r_i_{timestamp}.log'

    file_logger = DualLogger(log_file=str(log_path))

    class _EchoLogger:
        def info(self, msg: str): file_logger.info(msg, console=True)
        def warning(self, msg: str): file_logger.warning(msg, console=True)
        def error(self, msg: str): file_logger.error(msg, console=True)
        def debug(self, msg: str): file_logger.debug(msg)
    logger = _EchoLogger()

    logger.info(f"R_i diagnostic — {datetime.now().isoformat()}")
    logger.info(f"v3 ENDF       : {v3.ENDF_FILE}")
    logger.info(f"v3 EXFOR DB   : {v3.EXFOR_DB_PATH}")
    logger.info(f"v3 MT         : {v3.MT_NUMBER}")
    logger.info(f"v3 E range    : [{v3.ENERGY_RANGE_MEV[0]}, {v3.ENERGY_RANGE_MEV[1]}] MeV")
    logger.info(f"Output dir    : {out_dir}")
    logger.info("")

    # ---- STEP 1: Load EXFOR ------------------------------------------------
    logger.info("STEP 1: Load EXFOR")
    exfor_dict, _status = read_all_exfor(
        source='database',
        db_path=v3.EXFOR_DB_PATH,
        target_zaid=v3.TARGET_ZAIDS,
        projectile='N',
        mt=v3.MT_NUMBER,
        energy_range=v3.ENERGY_RANGE_MEV,
        supplementary_json_files=v3.SUPPLEMENTARY_JSON_FILES,
        exclude_experiments=v3.EXCLUDE_EXPERIMENTS,
        group_by_energy=False,
        return_load_status=True,
    )
    exfor_cache, sorted_energies = build_exfor_cache_from_objects(
        list(exfor_dict.values()),
        exclude_experiments=v3.EXCLUDE_EXPERIMENTS,
    )
    n_datasets = sum(len(v) for v in exfor_cache.values())
    logger.info(f"  {n_datasets} datasets at {len(sorted_energies)} unique energies")

    # ---- STEP 2: Load ENDF + reconstruct σ ---------------------------------
    logger.info("STEP 2: Load ENDF + NJOY reconstruct")
    _endf, xs_source, _mf33_mt, _mf33_sections, _xs_map = (
        v3._load_endf_with_reconstruction(
            v3.ENDF_FILE, v3.MT_NUMBER, v3.NJOY_EXECUTABLE, v3.NJOY_TOLERANCE,
            logger, pendf_cache_dir=v3.NJOY_PENDF_CACHE_DIR,
        )
    )

    # ---- STEP 3: Build AD energy bins --------------------------------------
    logger.info("STEP 3: Build AD energy bins")
    if v3.UNION_GRID_SUBENTRIES:
        grid_energies_ev = build_union_energy_grid(
            exfor_cache=exfor_cache,
            subentries=v3.UNION_GRID_SUBENTRIES,
            energy_min_mev=v3.ENERGY_RANGE_MEV[0],
            energy_max_mev=v3.ENERGY_RANGE_MEV[1],
        )
    else:
        all_e_mev = sorted(
            e for e in exfor_cache
            if v3.ENERGY_RANGE_MEV[0] <= e <= v3.ENERGY_RANGE_MEV[1]
        )
        grid_energies_ev = np.array(
            sorted(set(all_e_mev) | set(v3.ENERGY_RANGE_MEV))
        ) * 1e6
    energy_bins = compute_energy_bins_with_tof_resolution(
        energies_ev=grid_energies_ev,
        energy_min_mev=v3.ENERGY_RANGE_MEV[0],
        energy_max_mev=v3.ENERGY_RANGE_MEV[1],
        delta_t_ns=v3.DELTA_T_NS,
        flight_path_m=v3.FLIGHT_PATH_M,
    )
    logger.info(f"  built {len(energy_bins)} bins")

    # ---- STEP 4: σ_MF3 per bin (bin-average) -------------------------------
    logger.info("STEP 4: σ_MF3 bin-average per bin")
    sigma_mf3 = np.array([
        v3._bin_average_xs(xs_source, b.bin_lower_mev * 1e6, b.bin_upper_mev * 1e6)
        for b in energy_bins
    ], dtype=float)

    # ---- STEP 5: Per-bin nominal fits + R_i --------------------------------
    logger.info("STEP 5: Nominal fits + R_i")
    rows: List[dict] = []
    n_skip_c0 = n_skip_qg = n_skip_fit = 0
    FOUR_PI = 4.0 * np.pi

    for bin_idx, bin_info in enumerate(energy_bins):
        if sigma_mf3[bin_idx] < v3.MIN_C0_MF3_BARN:
            n_skip_c0 += 1
            continue
        res, expansion_used = v3._filter_with_quality_gate(
            exfor_cache, sorted_energies, energy_bins, bin_idx, logger,
        )
        if res is None:
            n_skip_qg += 1
            continue
        exfor_df, _exp_info, kernel_weights, _diag, _floor = res
        exfor_df = v3._attach_sigma_sys(exfor_df)
        try:
            nominal_coeffs, info = v3._nominal_fit_one_bin(exfor_df, kernel_weights)
        except Exception as e:
            logger.warning(
                f"  E={bin_info.energy_mev:.4f} MeV: fit failed ({e}) — skip"
            )
            n_skip_fit += 1
            continue
        c0_exfor = float(nominal_coeffs[0])           # b/sr
        sigma_implied = FOUR_PI * c0_exfor             # b
        sigma_m = float(sigma_mf3[bin_idx])            # b
        R_i = sigma_implied / sigma_m - 1.0
        rows.append({
            "bin_idx": bin_idx,
            "energy_mev": bin_info.energy_mev,
            "bin_lower_mev": bin_info.bin_lower_mev,
            "bin_upper_mev": bin_info.bin_upper_mev,
            "expansion_used": expansion_used,
            "n_exfor_points": int(len(exfor_df)),
            "frozen_degree": int(info["degree"]),
            "c0_exfor_b_per_sr": c0_exfor,
            "sigma_implied_exfor_b": sigma_implied,
            "sigma_mf3_b": sigma_m,
            "R_i": R_i,
            "abs_R_i": abs(R_i),
        })

    logger.info(
        f"  fitted: {len(rows)}, skipped (c0): {n_skip_c0}, "
        f"skipped (QG): {n_skip_qg}, skipped (fit): {n_skip_fit}"
    )

    # ---- DUMP per-bin CSV --------------------------------------------------
    csv_path = out_dir / "r_i_per_bin.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        logger.info(f"  wrote {csv_path}")
    else:
        logger.error("No bins fitted — aborting summary.")
        return 1

    # ---- SUMMARY -----------------------------------------------------------
    R = np.array([r["R_i"] for r in rows], dtype=float)
    absR = np.abs(R)
    n = R.size

    summary: List[str] = []

    def add(line: str = "") -> None:
        summary.append(line)
        logger.info(line)

    add()
    add("=" * 72)
    add("R_i SUMMARY")
    add("=" * 72)
    add(f"Bins fitted        : {n}")
    add(f"Bins skipped (c0)  : {n_skip_c0}")
    add(f"Bins skipped (QG)  : {n_skip_qg}")
    add(f"Bins skipped (fit) : {n_skip_fit}")
    add()
    add(f"Mean R_i           : {R.mean():+.4f}")
    add(f"Median R_i         : {np.median(R):+.4f}")
    add(f"Std R_i            : {R.std():.4f}")
    add(f"Mean |R_i|         : {absR.mean():.4f}")
    add(f"Median |R_i|       : {np.median(absR):.4f}")
    add()
    add("R_i percentiles:")
    for pct in (5, 25, 50, 75, 95):
        add(f"  {pct:>2}th : {np.percentile(R, pct):+.4f}")
    add()
    add("|R_i| threshold counts:")
    thresholds = [0.01, 0.03, 0.05, 0.10, 0.20, 0.50]
    prev = 0.0
    for t in thresholds:
        c = int(np.sum((absR >= prev) & (absR < t)))
        add(f"  {prev*100:>4.1f}% <= |R| < {t*100:>4.1f}% : "
            f"{c:>4} ({100*c/n:>5.1f}%)")
        prev = t
    c_above = int(np.sum(absR >= thresholds[-1]))
    add(f"  {thresholds[-1]*100:>4.1f}% <= |R|           : "
        f"{c_above:>4} ({100*c_above/n:>5.1f}%)")
    add()

    sorted_idx = np.argsort(-absR)[:10]
    add("Top 10 |R_i| outliers:")
    add(f"  {'E (MeV)':>10}  {'σ_MF3 (b)':>10}  {'σ_EXFOR (b)':>12}  "
        f"{'R_i':>8}  {'n_pts':>5}  {'L_AICc':>6}")
    for k in sorted_idx:
        r = rows[k]
        add(f"  {r['energy_mev']:>10.4f}  {r['sigma_mf3_b']:>10.4f}  "
            f"{r['sigma_implied_exfor_b']:>12.4f}  {r['R_i']:>+8.4f}  "
            f"{r['n_exfor_points']:>5d}  {r['frozen_degree']:>6d}")
    add()

    add("=" * 72)
    add("DECISION AID")
    add("=" * 72)
    pct_within_3  = 100 * np.sum(absR < 0.03) / n
    pct_within_10 = 100 * np.sum(absR < 0.10) / n
    n_above_20    = int(np.sum(absR >= 0.20))
    add(f"  {pct_within_3:>5.1f}% of bins have |R_i| < 3%")
    add(f"  {pct_within_10:>5.1f}% of bins have |R_i| < 10%")
    add(f"  {n_above_20} bin(s) have |R_i| >= 20%")
    add()
    if pct_within_3 >= 95:
        add("  → R_i is small almost everywhere. Mode 2 (pin c_0) is essentially")
        add("    free of cost. Free vs pinned fits will publish nearly identical")
        add("    a_l. The MF34 L=0 cross block is honest and small.")
    elif pct_within_10 >= 90 and n_above_20 == 0:
        add("  → R_i is moderate. Mode 2 will shift published a_l by a few % in")
        add("    most bins. Acceptable if you want the σ↔a_l cross block to be")
        add("    meaningful. A soft prior with small u_disc (~3%) is recommended.")
    else:
        u_disc_suggest = float(np.percentile(absR, 75))
        add("  → R_i is large in some bins. Mode 2 with a hard constraint will")
        add("    distort shapes by R_i/(1+R_i) in those bins. Either:")
        add(f"    (a) use a soft prior with u_disc ≈ {u_disc_suggest:.0%} to soften the pull,")
        add("    (b) drop the L=0 row for bins with large |R_i|, or")
        add("    (c) stick with Mode 1 (EXFOR-only nominal, no L=0 row).")
    add()
    add("=" * 72)

    summary_path = out_dir / "r_i_summary.txt"
    summary_path.write_text("\n".join(summary) + "\n")
    logger.info(f"wrote {summary_path}")

    # ---- Optional histogram ------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.hist(R * 100, bins=40, edgecolor="black", alpha=0.75)
        ax1.axvline(0, color="red", linestyle="--", linewidth=1)
        ax1.set_xlabel("R_i (%)")
        ax1.set_ylabel("# bins")
        ax1.set_title(f"R_i distribution (N={n})")
        ax1.grid(alpha=0.3)

        e_arr = np.array([r["energy_mev"] for r in rows])
        ax2.scatter(e_arr, R * 100, s=20)
        ax2.axhline(0, color="red", linestyle="--", linewidth=1)
        ax2.axhspan(-3, 3, color="green", alpha=0.15, label="±3%")
        ax2.axhspan(-10, 10, color="orange", alpha=0.10, label="±10%")
        ax2.set_xlabel("E (MeV)")
        ax2.set_ylabel("R_i (%)")
        ax2.set_title("R_i vs energy")
        ax2.legend(loc="best")
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        png_path = out_dir / "r_i_histogram.png"
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        logger.info(f"wrote {png_path}")
    except ImportError:
        logger.info("matplotlib not available — skipping histogram")

    logger.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
