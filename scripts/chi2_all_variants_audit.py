"""
Run all four chi^2 variants on the cluster-side parquet+npz, in one pass.

Variants (Σ_g for one experiment of n_g points). u = σ_indep·y (per-experiment
normalization rank-1), v = σ_dep·y (per-experiment shape rank-1, per-row
amplitude). Both are correlated within an experiment per the manifest:

    V1 textbook diag : Σ = diag(σ_stat² + (σ_indep·y)² + (σ_dep·y)² + σ_eval_diag²)
                       (no rank-1 marginalisation; σ_eval_diag only — diagonal of
                        the full MF34 sandwich, stored in parquet column)
    V2 rank-2 only   : Σ = D + u uᵀ + v vᵀ              (D excludes σ_eval)
    V3 rank-2 + diag : Σ = (D + diag σ_eval²) + u uᵀ + v vᵀ
    V4 rank-2 + dense: Σ = D + u uᵀ + v vᵀ + Σ_eval     (full dense block;
                                                         cross-energy MF34 +
                                                         cross-order MF34 +
                                                         cross-energy MF33 for library_c0)

What you get out
----------------
A small Markdown report at <REPORT_DIR>/all_variants_audit.md with:

  * Per-(library × subset × variant) chi²/N table (4 chi²/N values × 3
    subsets × N_libraries — single source of truth for deciding which
    variant goes in the chapter).
  * Per-(library × KS subentry × variant) breakdown.
  * Top-15 chi² contributors per library, with V1/V3/V4 side-by-side, so the
    user can see whether the offending experiments are the same under each
    variant.
  * σ_stat-floor sensitivity sweep on V3 (0.5%, 1%, 2%) to bracket
    robustness of the numbers we'd put in the chapter.

Memory profile
--------------
Loads parquet (~4 MB) eagerly. Streams npz BLOCKS LAZILY one at a time
(np.load on a compressed npz decompresses on access), so the resident
memory never exceeds (parquet + one block). Even Cierjacks's 3.3 GB block
fits a 32 GB cluster node easily, but if you want to skip it set
MAX_EXP_POINTS below.

Usage on the cluster
--------------------
    python scripts/chi2_all_variants_audit.py
        # runs both methodologies (exfor_c0 + library_c0) by default

    METHODOLOGIES=exfor_c0  python scripts/chi2_all_variants_audit.py
        # override via env var

Output goes to /share_snc/snc/JuanMonleon/CHI_Figures/<methodology>/
alongside the existing report.md.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from scripts.chi2_metrics import (
    _components, _per_group_sigma, _rank2_chi2_woodbury, _solve,
)


# ── Configuration ─────────────────────────────────────────────────────────────

# Run versioning. Outputs land in `<report_dir>/run_<RUN_ID>/`; bump this to
# start a fresh run without overwriting earlier ones. Match the value in
# chi2_analysis_cluster.py to keep both reports under the same run directory.
RUN_ID: str = "001"

PATHS = {
    "exfor_c0": {
        "parquet":   "/share_snc/snc/JuanMonleon/chi2/chi2_data_exfor_c0_73.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_exfor_c0",
        "title": "exfor_c0 (c_0 from K&S fit, σ_eval = MF34 sandwich)",
    },
    "library_c0": {
        "parquet":   "/share_snc/snc/JuanMonleon/chi2/chi2_data_library_c0_73.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_library_c0",
        "title": "library_c0 (c_0 from MF3 bin avg, σ_eval = MF34 sandwich + MF33)",
    },
}

E_MIN_MEV = 0.85
E_MAX_MEV = 4.0
KINNEY_SMITH_IDS = ("10571002", "10886002")

# Lib display names, ordered for the table.
LIB_LABELS = {"JEFF": "JEFF-4.0", "JENDL": "JENDL-5", "This_work": "This work"}
LIB_ORDER = ("JEFF", "JENDL", "This_work")

# σ_stat-floor sensitivity sweep on V3 (1% is the resolver/pipeline default).
SIGMA_STAT_FLOOR_SWEEP = (0.005, 0.01, 0.02)

# Skip the V4 (dense block) computation for experiments larger than this many
# points. Cluster nodes with >32 GB free can handle Cierjacks (28k → 3.3 GB);
# set lower if running on a smaller node. None = no skip.
MAX_EXP_POINTS_FOR_V4: Optional[int] = None

DEFAULT_METHODOLOGIES = ("exfor_c0", "library_c0")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _v4_chi2_for_group(
    D: np.ndarray, u: np.ndarray, v: np.ndarray, r: np.ndarray,
    block: np.ndarray, label: str,
) -> float:
    sigma = _per_group_sigma(D, u, v, block)
    chi2, _z = _solve(sigma, r, label=label)
    return float(chi2)


def _v1(D: np.ndarray, u: np.ndarray, v: np.ndarray,
        r: np.ndarray, sed: np.ndarray) -> float:
    denom = np.maximum(D + u ** 2 + v ** 2 + sed ** 2, 1e-300)
    return float(np.sum(r ** 2 / denom))


def _v2(D: np.ndarray, u: np.ndarray, v: np.ndarray, r: np.ndarray) -> float:
    return _rank2_chi2_woodbury(D, u, v, r)


def _v3(D: np.ndarray, u: np.ndarray, v: np.ndarray,
        r: np.ndarray, sed: np.ndarray) -> float:
    return _rank2_chi2_woodbury(D + sed ** 2, u, v, r)


def _refloor_D(sigma_stat: np.ndarray, y: np.ndarray, floor_rel: float) -> np.ndarray:
    """Re-apply a custom σ_stat floor to D = σ_stat². σ_dep is no longer on
    the diagonal — it enters chi² as a rank-1 correlated mode v = σ_dep·y.
    Used by the floor-sensitivity sweep so we don't have to rebuild the parquet."""
    sigma_stat_f = np.maximum(sigma_stat, floor_rel * np.abs(y))
    D = sigma_stat_f ** 2
    diag_floor = (1e-3 * np.abs(y)) ** 2 + 1e-30
    return np.maximum(D, diag_floor)


# ── Per-experiment runner ─────────────────────────────────────────────────────


def compute_per_experiment(
    df: pd.DataFrame,
    npz_path: Path,
    max_points_for_v4: Optional[int],
    log: List[str],
) -> pd.DataFrame:
    """Walk experiments, stream npz blocks lazily, compute V1..V4 per (lib, exp)."""
    cols_required = {"y_exp", "y_eval", "sigma_exp_stat",
                     "sigma_sys_indep_rel", "sigma_sys_dep_rel"}
    missing = cols_required - set(df.columns)
    if missing:
        raise ValueError(f"Parquet missing required columns: {missing}")

    rows: List[dict] = []
    skipped_v4: List[Tuple[str, str, int]] = []

    with np.load(str(npz_path)) as data:
        keys_in_npz = set(data.files)

        for (lib, exp), sub in df.groupby(["library", "experiment_id"],
                                          observed=True, sort=False):
            sub = sub.reset_index(drop=True)
            n = len(sub)
            y = sub["y_exp"].to_numpy(dtype=float)
            r = (sub["y_exp"] - sub["y_eval"]).to_numpy(dtype=float)
            sigma_stat = sub["sigma_exp_stat"].to_numpy(dtype=float)
            indep_rel = sub["sigma_sys_indep_rel"].to_numpy(dtype=float)
            dep_rel = sub["sigma_sys_dep_rel"].to_numpy(dtype=float)
            sed = (
                sub["sigma_eval_diag"].to_numpy(dtype=float)
                if "sigma_eval_diag" in sub.columns else np.zeros(n)
            )

            # Default σ_stat floor matching the precompute behaviour (1% of |y|).
            # σ_dep is rank-1 (v), not diagonal — see Fix 1a docstring.
            D = _refloor_D(sigma_stat, y, floor_rel=0.01)
            u = indep_rel * y
            v_arr = dep_rel * y

            v1 = _v1(D, u, v_arr, r, sed)
            v2 = _v2(D, u, v_arr, r)
            v3 = _v3(D, u, v_arr, r, sed)

            v4 = float("nan")
            if max_points_for_v4 is None or n <= max_points_for_v4:
                key_str = f"{lib}@@{exp}"
                if key_str in keys_in_npz:
                    block = np.asarray(data[key_str], dtype=np.float64)
                    if block.shape != (n, n):
                        log.append(f"  ! {lib}/{exp}: block shape {block.shape} "
                                   f"vs df rows {n}; V4 skipped")
                    else:
                        v4 = _v4_chi2_for_group(D, u, v_arr, r, block, f"{lib}/{exp}")
                    del block
                else:
                    log.append(f"  ! {lib}/{exp}: no block in npz; V4 skipped")
            else:
                skipped_v4.append((lib, exp, n))

            rows.append({
                "library": lib, "experiment_id": exp,
                "author": sub["author"].iloc[0],
                "year": int(sub["year"].iloc[0]),
                "ks_subentry": sub["ks_subentry"].iloc[0],
                "N": n,
                "chi2_v1": v1, "chi2_v2": v2, "chi2_v3": v3, "chi2_v4": v4,
                "chi2_v1_per_N": v1 / n if n > 0 else np.nan,
                "chi2_v2_per_N": v2 / n if n > 0 else np.nan,
                "chi2_v3_per_N": v3 / n if n > 0 else np.nan,
                "chi2_v4_per_N": v4 / n if n > 0 else np.nan,
            })

    if skipped_v4:
        log.append(f"\nSkipped V4 for {len(skipped_v4)} experiments above "
                   f"MAX_EXP_POINTS_FOR_V4={max_points_for_v4}:")
        for lib, exp, n in skipped_v4:
            log.append(f"  {lib}/{exp}: N={n}")
    return pd.DataFrame(rows)


# ── Sensitivity sweep on σ_stat floor (V1/V3 only — V4 requires block) ────────


def floor_sensitivity_v3(
    df: pd.DataFrame, floor_sweep: Tuple[float, ...],
) -> pd.DataFrame:
    """Recompute V3 chi²/N per (library, subset) under different σ_stat floors."""
    out = []
    df = df.copy()
    df["is_KS"] = df["experiment_id"].isin(KINNEY_SMITH_IDS)

    for floor in floor_sweep:
        y = df["y_exp"].to_numpy(dtype=float)
        D = _refloor_D(
            df["sigma_exp_stat"].to_numpy(dtype=float),
            y, floor_rel=floor,
        )
        u = df["sigma_sys_indep_rel"].to_numpy(dtype=float) * y
        v = df["sigma_sys_dep_rel"].to_numpy(dtype=float) * y
        sed = df["sigma_eval_diag"].to_numpy(dtype=float)
        r = (df["y_exp"] - df["y_eval"]).to_numpy(dtype=float)

        # Per (library, experiment_id) rank-2 Woodbury V3
        work = df.assign(_D=D + sed ** 2, _u=u, _v=v, _r=r)

        per_exp_chi2 = []
        for (lib, exp), g in work.groupby(["library", "experiment_id"],
                                           observed=True, sort=False):
            d_ = g["_D"].to_numpy()
            u_ = g["_u"].to_numpy()
            v_ = g["_v"].to_numpy()
            r_ = g["_r"].to_numpy()
            per_exp_chi2.append({
                "library": lib, "experiment_id": exp,
                "N": len(g),
                "chi2_v3": _rank2_chi2_woodbury(d_, u_, v_, r_),
                "is_KS": exp in KINNEY_SMITH_IDS,
            })
        per = pd.DataFrame(per_exp_chi2)

        for subset_label, mask in (
            ("all",       pd.Series(True, index=per.index)),
            ("no_KS",     ~per["is_KS"]),
            ("only_KS",    per["is_KS"]),
        ):
            sub = per[mask]
            for lib in LIB_ORDER:
                sl = sub[sub["library"] == lib]
                n_total = int(sl["N"].sum())
                c_total = float(sl["chi2_v3"].sum())
                out.append({
                    "floor": floor, "subset": subset_label, "library": lib,
                    "N": n_total, "chi2_v3_per_N":
                        c_total / n_total if n_total > 0 else float("nan"),
                })
    return pd.DataFrame(out)


# ── Reporting ─────────────────────────────────────────────────────────────────


def _fmt(v: float) -> str:
    if not np.isfinite(v):
        return "  —  "
    if v >= 100:
        return f"{v:.1f}"
    return f"{v:.3f}"


def aggregate_by_library_subset(per_exp: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """For each subset (all/no_KS/only_KS), sum chi² and N over experiments and
    compute chi²/N for each variant. Returns dict of DataFrames keyed by subset."""
    per_exp = per_exp.copy()
    per_exp["is_KS"] = per_exp["experiment_id"].isin(KINNEY_SMITH_IDS)
    out: Dict[str, pd.DataFrame] = {}
    for subset_label, mask in (
        ("all",     pd.Series(True, index=per_exp.index)),
        ("no_KS",  ~per_exp["is_KS"]),
        ("only_KS", per_exp["is_KS"]),
    ):
        sub = per_exp[mask]
        rows = []
        for lib in LIB_ORDER:
            sl = sub[sub["library"] == lib]
            n = int(sl["N"].sum())
            row = {"Library": LIB_LABELS[lib], "library_key": lib, "N": n}
            for v in ("v1", "v2", "v3", "v4"):
                c = float(sl[f"chi2_{v}"].sum())
                row[f"chi2_{v}/N"] = c / n if n > 0 else float("nan")
            rows.append(row)
        out[subset_label] = pd.DataFrame(rows)
    return out


def render_report(
    title: str,
    per_exp: pd.DataFrame,
    floor_sweep_df: pd.DataFrame,
    log: List[str],
) -> str:
    R = [f"# {title}\n"]
    R.append("Variants: V1 textbook diag · V2 rank-2 only · V3 rank-2 + diag σ_eval · V4 rank-2 + dense Σ_eval.\n")
    R.append(f"Energy window: [{E_MIN_MEV}, {E_MAX_MEV}] MeV.\n")
    R.append(f"Total points per library: {int(per_exp[per_exp['library']==LIB_ORDER[0]]['N'].sum())}, "
             f"experiments: {per_exp['experiment_id'].nunique()}.\n")

    # ── 1. Global chi²/N table — all libraries × subsets × variants ──────────
    R.append("\n## 1. Global chi²/N — all four variants\n")
    by_sub = aggregate_by_library_subset(per_exp)
    for subset_label in ("all", "no_KS", "only_KS"):
        ag = by_sub[subset_label]
        n_show = int(ag["N"].iloc[0])
        R.append(f"\n**Subset: {subset_label}** "
                 f"({n_show} points/library, "
                 f"{per_exp[(per_exp.library==LIB_ORDER[0])].pipe(lambda d: d if subset_label=='all' else (d[~d.experiment_id.isin(KINNEY_SMITH_IDS)] if subset_label=='no_KS' else d[d.experiment_id.isin(KINNEY_SMITH_IDS)]))['experiment_id'].nunique()} experiments)\n")
        R.append("| Library | V1 textbook | V2 rank-2 only | V3 rank-2 + diag eval | V4 rank-2 + dense eval |\n")
        R.append("|---|---:|---:|---:|---:|\n")
        for _, row in ag.iterrows():
            R.append(f"| {row['Library']} | {_fmt(row['chi2_v1/N'])} | "
                     f"{_fmt(row['chi2_v2/N'])} | {_fmt(row['chi2_v3/N'])} | "
                     f"{_fmt(row['chi2_v4/N'])} |\n")
    R.append("\n")

    # ── 2. K&S subentry split, all variants ──────────────────────────────────
    R.append("\n## 2. Kinney/Smith/OTHER subentry split — all variants\n\n")
    R.append("| Library | Subentry | N | V1 | V2 | V3 | V4 |\n")
    R.append("|---|---|---:|---:|---:|---:|---:|\n")
    for lib in LIB_ORDER:
        for ks in ("Kinney_1976", "Smith_1980", "OTHER"):
            sub = per_exp[(per_exp["library"] == lib) &
                          (per_exp["ks_subentry"] == ks)]
            if sub.empty:
                continue
            n = int(sub["N"].sum())
            vals = {v: float(sub[f"chi2_{v}"].sum()) / n if n > 0 else float("nan")
                    for v in ("v1", "v2", "v3", "v4")}
            R.append(f"| {LIB_LABELS[lib]} | {ks} | {n} | "
                     f"{_fmt(vals['v1'])} | {_fmt(vals['v2'])} | "
                     f"{_fmt(vals['v3'])} | {_fmt(vals['v4'])} |\n")
    R.append("\n")

    # ── 3. Top-15 contributors per library under each variant ────────────────
    R.append("\n## 3. Top contributors per library — V1/V3/V4 side-by-side\n")
    R.append("\nRanked by V3 chi² (the recommended diagonal-eval metric); "
             "shown to compare whether the same experiments dominate across variants.\n")
    for lib in LIB_ORDER:
        R.append(f"\n### {LIB_LABELS[lib]}\n\n")
        sub = (per_exp[per_exp["library"] == lib]
               .copy()
               .sort_values("chi2_v3", ascending=False)
               .head(15))
        R.append("| exp_id | author (year) | KS | N | V1/N | V2/N | V3/N | V4/N |\n")
        R.append("|---|---|---|---:|---:|---:|---:|---:|\n")
        for _, r in sub.iterrows():
            R.append(f"| {r['experiment_id']} | {r['author']} ({r['year']}) | "
                     f"{r['ks_subentry']} | {int(r['N'])} | "
                     f"{_fmt(r['chi2_v1_per_N'])} | {_fmt(r['chi2_v2_per_N'])} | "
                     f"{_fmt(r['chi2_v3_per_N'])} | {_fmt(r['chi2_v4_per_N'])} |\n")

    # ── 4. σ_stat-floor sensitivity sweep on V3 ──────────────────────────────
    R.append("\n## 4. σ_stat-floor sensitivity (V3)\n\n")
    R.append("Default floor is 1% of |y|; this table sweeps 0.5% / 1% / 2% "
             "to bracket robustness of the V3 numbers we'd put in the chapter.\n\n")
    R.append("| floor | subset | " + " | ".join(LIB_LABELS[l] for l in LIB_ORDER) + " |\n")
    R.append("|---|---|" + "---:|" * len(LIB_ORDER) + "\n")
    for floor in sorted(floor_sweep_df["floor"].unique()):
        for subset in ("all", "no_KS", "only_KS"):
            row = floor_sweep_df[(floor_sweep_df["floor"] == floor) &
                                 (floor_sweep_df["subset"] == subset)]
            vals = [float(row[row["library"] == l]["chi2_v3_per_N"].iloc[0])
                    for l in LIB_ORDER]
            R.append(f"| {floor*100:.1f}% | {subset} | " +
                     " | ".join(_fmt(v) for v in vals) + " |\n")
    R.append("\n")

    if log:
        R.append("\n## Run log\n```\n")
        R.append("\n".join(log))
        R.append("\n```\n")
    return "".join(R)


# ── Top-level driver ──────────────────────────────────────────────────────────


def run(methodology: str) -> None:
    cfg = PATHS[methodology]
    parquet_path = Path(cfg["parquet"])
    npz_path = parquet_path.with_suffix(parquet_path.suffix + ".eval_cov.npz")
    report_dir = Path(cfg["report_dir"]) / f"run_{RUN_ID}"
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {methodology} (run_{RUN_ID}) ===")
    print(f"  parquet: {parquet_path}")
    print(f"  npz:     {npz_path}")
    print(f"  output:  {report_dir}")

    df = pd.read_parquet(parquet_path)
    df = df[(df["energy_mev"] >= E_MIN_MEV) & (df["energy_mev"] <= E_MAX_MEV)].copy()
    df = df[df["sigma_exp"] > 0].copy()
    df["sigma_eval_diag"] = df["sigma_eval_diag"].fillna(0.0)
    print(f"  filtered rows: {len(df)} (energy [{E_MIN_MEV},{E_MAX_MEV}] MeV, σ_exp>0)")

    log: List[str] = []
    log.append(f"parquet path: {parquet_path}")
    log.append(f"npz path:     {npz_path}")
    log.append(f"MAX_EXP_POINTS_FOR_V4 = {MAX_EXP_POINTS_FOR_V4}")
    log.append(f"σ_stat-floor sweep: {SIGMA_STAT_FLOOR_SWEEP}")

    per_exp = compute_per_experiment(df, npz_path, MAX_EXP_POINTS_FOR_V4, log)
    floor_sweep_df = floor_sensitivity_v3(df, SIGMA_STAT_FLOOR_SWEEP)

    report = render_report(cfg["title"], per_exp, floor_sweep_df, log)
    out_md = report_dir / "all_variants_audit.md"
    out_md.write_text(report)
    out_csv = report_dir / "all_variants_per_experiment.csv"
    per_exp.to_csv(out_csv, index=False)
    out_floor = report_dir / "all_variants_floor_sweep.csv"
    floor_sweep_df.to_csv(out_floor, index=False)

    print(f"  wrote: {out_md}")
    print(f"  wrote: {out_csv}")
    print(f"  wrote: {out_floor}")


def main() -> int:
    env = os.environ.get("METHODOLOGIES", "").strip()
    if env:
        methodologies = tuple(m.strip() for m in env.split(",") if m.strip())
    else:
        methodologies = DEFAULT_METHODOLOGIES
    for m in methodologies:
        if m not in PATHS:
            raise SystemExit(f"unknown methodology: {m}; choose from {list(PATHS)}")
        run(m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
