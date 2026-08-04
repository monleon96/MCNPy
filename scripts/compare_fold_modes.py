"""Compare the five resolution-fold conventions of the predictive scenario.

The notebook `kika_dev/exfor/uncertainty/energy_resolution_impact.ipynb` shows
that averaging the product <sigma . F(a_l)> differs from averaging the factors
<sigma> . F(<a_l>) by up to ~18% below 2.5 MeV — but only at a handful of
hand-picked energies, with no data attached. This script turns that into a
statement over every (experiment, energy, angle) bin in the database: which
convention actually reproduces the measurements best, and does the ordering hold
everywhere or only in places.

Reads the summary.json each analysis run writes, so it costs nothing and needs
no covariance sidecar. Run after run_fold_sweep.sh.

The headline is V2 (EXFOR uncertainties only): that isolates the central values,
which is what the fold convention actually changes. V4 is reported alongside
because the fold also shifts c0 and therefore the Sigma_eval linearization.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

FIG_ROOT = Path("/share_snc/snc/JuanMonleon/CHI_Figures")
CHI2_DIR = Path("/share_snc/snc/JuanMonleon/chi2")
RUN_ID = "082"

# Ordered worst-model-first so the table reads as "what does adding each piece
# of the resolution model buy".
MODES: List[tuple] = [
    ("none",    "chi2_predictive_nofold",     "sigma(E0)*F(a_l(E0))   no fold"),
    ("al",      "chi2_predictive_foldal",     "sigma(E0)*F(<a_l>)     a_l only"),
    ("sigma",   "chi2_predictive_foldsigma",  "<sigma>*F(a_l(E0))     sigma only"),
    ("factors", "chi2_predictive_factors",    "<sigma>*F(<a_l>)       factors"),
    ("product", "chi2_predictive",            "<sigma*F(a_l)>         PRODUCT"),
]

LIBS = ["JEFF", "JENDL", "This_work"]
SUBSETS = ["all", "no_Cierjacks", "no_KS", "no_KS_no_Cierjacks",
           "only_KS", "only_Cierjacks"]


def load_summary(report_dir: str) -> Optional[dict]:
    p = FIG_ROOT / report_dir / f"run_{RUN_ID}" / "summary.json"
    if not p.exists():
        print(f"  [missing] {p}")
        return None
    return json.loads(p.read_text())


def table(summaries: Dict[str, dict], variant: str) -> pd.DataFrame:
    """chi2/N per (mode x subset x library) for one variant."""
    recs = []
    for mode, _, label in MODES:
        s = summaries.get(mode)
        if s is None:
            continue
        for subset in SUBSETS:
            for row in s["per_subset_all_variants"].get(subset, []):
                recs.append({
                    "mode": mode, "label": label, "subset": subset,
                    "library": row["library_key"],
                    "chi2_per_N": row[f"chi2_{variant.lower()}_per_N"],
                })
    return pd.DataFrame(recs)


def show(df: pd.DataFrame, variant: str) -> None:
    if df.empty:
        print(f"  no data for {variant}")
        return
    print(f"\n{'=' * 100}")
    print(f"  {variant}  chi2/N  —  " +
          ("central values only, EXFOR uncertainties (isolates the fold)"
           if variant == "V2" else "full budget, EXFOR + dense MF34/MF33"))
    print("=" * 100)
    for subset in SUBSETS:
        sub = df[df.subset == subset]
        if sub.empty:
            continue
        print(f"\n  subset: {subset}")
        print(f"    {'fold convention':38s}" +
              "".join(f"{l:>13s}" for l in LIBS))
        base = {}
        for mode, _, label in MODES:
            row = sub[sub["mode"] == mode]
            if row.empty:
                continue
            vals = {l: float(row[row.library == l]["chi2_per_N"].iloc[0])
                    for l in LIBS if not row[row.library == l].empty}
            if mode == "none":
                base = dict(vals)
            cells = ""
            for l in LIBS:
                if l not in vals:
                    cells += f"{'--':>13s}"
                    continue
                delta = ""
                if base.get(l):
                    delta = f" ({100 * (vals[l] / base[l] - 1):+5.1f}%)"
                cells += f"{vals[l]:7.2f}{delta:>6s}"
            print(f"    {label:38s}{cells}")
        # which convention wins, per library
        best = []
        for l in LIBS:
            r = sub[sub.library == l]
            if r.empty:
                continue
            best.append(f"{l}={r.loc[r.chi2_per_N.idxmin(), 'mode']}")
        print(f"    {'-> lowest chi2/N':38s}  " + "  ".join(best))


def per_experiment_split(summaries: Dict[str, dict]) -> None:
    """Does the fold help where the TOF resolution is measured, or everywhere?

    Only a few subentries carry measured flight-path/timing; the rest fall back
    to L=27.037 m, dt=5 ns. If the product fold is doing real physics the gain
    should concentrate on the measured ones — if it helps uniformly, the model
    is mostly fitting the default width and that is worth knowing.
    """
    prod = CHI2_DIR / "chi2_data_predictive_82.parquet"
    nofold = CHI2_DIR / "chi2_data_predictive_82_nofold.parquet"
    if not (prod.exists() and nofold.exists()):
        print("\n[skip] per-experiment TOF split needs both the product and "
              "nofold parquets.")
        return

    cols = ["library", "experiment_id", "author", "y_exp", "y_eval",
            "sigma_exp", "tof_source", "energy_mev", "sigma_E_mev"]
    a = pd.read_parquet(prod, columns=cols)
    b = pd.read_parquet(nofold, columns=cols)

    def z2(d):
        return ((d.y_exp - d.y_eval) / d.sigma_exp) ** 2

    a = a.assign(z2=z2(a))
    b = b.assign(z2=z2(b))

    key = ["library", "experiment_id", "author", "tof_source"]
    ga = a.groupby(key, observed=True).agg(chi2_prod=("z2", "mean"),
                                           N=("z2", "size")).reset_index()
    gb = b.groupby(key, observed=True).agg(chi2_nofold=("z2", "mean")).reset_index()
    g = ga.merge(gb, on=key)
    g["gain_pct"] = 100.0 * (g.chi2_prod / g.chi2_nofold - 1.0)

    print(f"\n{'=' * 100}")
    print("  Does folding help where the TOF resolution is actually measured?")
    print("  (V2-style chi2/N, product vs no-fold; negative = folding helps)")
    print("=" * 100)
    for lib in LIBS:
        sub = g[g.library == lib]
        if sub.empty:
            continue
        print(f"\n  {lib}")
        for src, s in sub.groupby("tof_source", observed=True):
            print(f"    tof_source={str(src):24s} n_exp={len(s):3d}  "
                  f"median gain {s.gain_pct.median():+7.2f}%  "
                  f"helped on {int((s.gain_pct < 0).sum())}/{len(s)}")

    print("\n  Ten experiments where folding changes chi2/N most (This_work):")
    tw = g[g.library == "This_work"].reindex(
        g[g.library == "This_work"].gain_pct.abs().sort_values(
            ascending=False).index)
    print(tw[["experiment_id", "author", "N", "tof_source",
              "chi2_nofold", "chi2_prod", "gain_pct"]].head(10).to_string(index=False))


def main() -> None:
    print("Loading fold-mode summaries...")
    summaries = {}
    for mode, report_dir, _ in MODES:
        s = load_summary(report_dir)
        if s is not None:
            summaries[mode] = s
    if not summaries:
        raise SystemExit("No fold-mode summaries found. Run run_fold_sweep.sh first.")
    print(f"  found: {sorted(summaries)}")

    for variant in ("V2", "V4"):
        show(table(summaries, variant), variant)

    per_experiment_split(summaries)

    out = FIG_ROOT / "fold_mode_comparison.csv"
    frames = []
    for variant in ("V1", "V2", "V3", "V4"):
        t = table(summaries, variant)
        if not t.empty:
            frames.append(t.assign(variant=variant))
    if frames:
        pd.concat(frames).to_csv(out, index=False)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
