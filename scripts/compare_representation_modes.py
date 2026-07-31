"""Compare the MF34 representation modes — fine vs grouped, and order truncation.

Turns the `repr_*` sweep into the two numbers the file-design decision actually
needs:

  1. What does multigroup collapse cost in chi^2?  Grouping is irreversible and
     it modifies the result (width-averaged means, then a percentile variance
     compensation driven by l=1 heterogeneity applied to every order). A
     consumer can always regroup a fine file; nobody can ungroup a collapsed
     one. So the fine grid is the default and the collapse has to justify
     itself. This measures what it is buying and what it is costing.

  2. What do Legendre orders 4-6 buy?  They are what makes the file large —
     MF34 is quadratic in (bins x orders), so dropping 5 and 6 is a 2.25x
     reduction and dropping 4-6 is 4x. Two versions of the question:
     `cov` truncates covariance only (file size), `eval` truncates the
     evaluation itself (physics).

Reads the summary.json each analysis run writes, so it costs nothing and needs
no covariance sidecar. Run after precompute_chi2_representation.py and
chi2_analysis_cluster.py.

A warning about how to read Q2. Truncation is NOT monotonic in the propagated
variance: Sigma_eval carries cross-order blocks, P_l(mu) alternates sign, so a
negative cross term between a low and a high order is *cancelling* variance and
removing the high order removes the cancellation too. Dropping orders can
therefore raise sigma_eval at some angles and lower it at others (there is a
unit test pinning this: scripts/tests/test_chi2_representation.py). A chi^2/N
that falls after truncation is not by itself evidence the orders were useless —
check which direction sigma_eval moved before concluding anything.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

FIG_ROOT = Path("/share_snc/snc/JuanMonleon/CHI_Figures")
CHI2_DIR = Path("/share_snc/snc/JuanMonleon/chi2")
RUN_ID = "082"

LIBS = ["JEFF", "JENDL", "This_work"]
SUBSETS = ["all", "no_Cierjacks", "no_KS", "no_KS_no_Cierjacks",
           "only_KS", "only_Cierjacks"]

# (mode, label). Reference first in each block.
BLOCK_GROUPING: List[Tuple[str, str]] = [
    ("fine", "fine grid, L=6            REFERENCE"),
    ("mg",   "multigroup, L=6           collapsed"),
]
BLOCK_COV_FINE: List[Tuple[str, str]] = [
    ("fine",      "fine, MF34 l<=6          REFERENCE"),
    ("fine_cov5", "fine, MF34 l<=5"),
    ("fine_cov4", "fine, MF34 l<=4"),
    ("fine_cov3", "fine, MF34 l<=3"),
]
BLOCK_COV_MG: List[Tuple[str, str]] = [
    ("mg",      "mg, MF34 l<=6            REFERENCE"),
    ("mg_cov5", "mg, MF34 l<=5"),
    ("mg_cov4", "mg, MF34 l<=4"),
    ("mg_cov3", "mg, MF34 l<=3"),
]
BLOCK_EVAL: List[Tuple[str, str]] = [
    ("fine",       "fine, MF4+MF34 l<=6      REFERENCE"),
    ("fine_eval5", "fine, MF4+MF34 l<=5"),
    ("fine_eval4", "fine, MF4+MF34 l<=4"),
    ("fine_eval3", "fine, MF4+MF34 l<=3"),
]

# The evaluation grid, for the analytic size table. 1738 fine bins is the
# run-82 number (docs/mf3_mf33_roadmap.md); 598 is the l=1 collapse.
N_BINS = {"fine": 1738, "mg": 598}


def load_summary(mode: str) -> Optional[dict]:
    p = FIG_ROOT / f"chi2_repr_{mode}" / f"run_{RUN_ID}" / "summary.json"
    if not p.exists():
        print(f"  [missing] {p}")
        return None
    return json.loads(p.read_text())


def table(summaries: Dict[str, dict], variant: str) -> pd.DataFrame:
    recs = []
    for mode, s in summaries.items():
        for subset in SUBSETS:
            for row in s["per_subset_all_variants"].get(subset, []):
                recs.append({
                    "mode": mode, "subset": subset,
                    "library": row["library_key"],
                    "chi2_per_N": row[f"chi2_{variant.lower()}_per_N"],
                })
    return pd.DataFrame(recs)


def show_block(df: pd.DataFrame, block: List[Tuple[str, str]],
               variant: str, heading: str, note: str) -> None:
    present = [m for m, _ in block if m in set(df["mode"])]
    if len(present) < 2:
        print(f"\n[skip] {heading} — needs at least the reference and one "
              f"other mode; have {present}")
        return

    print(f"\n{'=' * 100}")
    print(f"  {heading}  —  {variant} chi2/N")
    print(f"  {note}")
    print("=" * 100)

    ref_mode = block[0][0]
    for subset in SUBSETS:
        sub = df[df.subset == subset]
        if sub.empty:
            continue
        print(f"\n  subset: {subset}")
        print(f"    {'representation':38s}" + "".join(f"{l:>15s}" for l in LIBS))
        base: Dict[str, float] = {}
        for mode, label in block:
            row = sub[sub["mode"] == mode]
            if row.empty:
                continue
            vals = {l: float(row[row.library == l]["chi2_per_N"].iloc[0])
                    for l in LIBS if not row[row.library == l].empty}
            if mode == ref_mode:
                base = dict(vals)
            cells = ""
            for l in LIBS:
                if l not in vals:
                    cells += f"{'--':>15s}"
                    continue
                delta = ""
                if base.get(l):
                    delta = f" ({100 * (vals[l] / base[l] - 1):+5.1f}%)"
                cells += f"{vals[l]:8.3f}{delta:>7s}"
            print(f"    {label:38s}{cells}")


def check_v2_invariance(summaries: Dict[str, dict]) -> None:
    """Q1 and Q2a must not move V2 at all.

    fine and mg ship the same MF4; `cov` truncation touches only MF34. V2 uses
    central values with EXFOR uncertainties only, so it cannot see either. If
    it moves, something other than the intended knob changed and the whole
    sweep is suspect.
    """
    df = table(summaries, "V2")
    if df.empty:
        return
    print(f"\n{'=' * 100}")
    print("  INVARIANCE CHECK — V2 must be identical across fine/mg and all "
          "`cov` modes")
    print("=" * 100)
    invariant = [m for m in
                 ["fine", "mg", "fine_cov3", "fine_cov4", "fine_cov5",
                  "mg_cov3", "mg_cov4", "mg_cov5"]
                 if m in set(df["mode"])]
    if len(invariant) < 2:
        print("  [skip] fewer than two invariant modes present")
        return
    worst = 0.0
    for subset in SUBSETS:
        for lib in LIBS:
            vals = df[(df.subset == subset) & (df.library == lib)
                      & (df["mode"].isin(invariant))]["chi2_per_N"]
            if len(vals) < 2 or not np.isfinite(vals).all():
                continue
            v = vals.to_numpy(dtype=float)
            spread = float(np.ptp(v)) / max(abs(float(np.mean(v))), 1e-30)
            worst = max(worst, spread)
    print(f"  modes checked: {invariant}")
    print(f"  worst relative V2 spread: {worst:.3e}")
    if worst > 1e-6:
        print("  *** FAIL — V2 moved. fine/mg and `cov` truncation must not "
              "touch the central values. Investigate before reading anything "
              "below as a result.")
    else:
        print("  OK — the knobs did what they claim.")


def validation_gate(summaries: Dict[str, dict]) -> None:
    """`repr_mg` must reproduce the `predictive` numbers it derives from.

    The reference run is overridable because the default one went stale. The
    fold in precompute_chi2_predictive.py uses σ_E from
    exfor_tof_parameters.json, which was rebuilt on 2026-07-30 to read the
    declared EXFOR EN-RSL* widths (~3.5× wider on 74 subentries). σ_E is part of
    the *scoring*, not just the evaluation, so every χ² built after that date —
    including this whole repr_* sweep — sits on a different footing from
    run_082, which was built 2026-07-28. Comparing across that boundary measures
    the resolution change, not the representation code, and the mismatch shows up
    on JEFF and JENDL too, which the sweep never touches.

    Point this at a reference rebuilt under the same σ_E:

        KIKA_GATE_REF_RUN=082_enrsl python compare_representation_modes.py
    """
    ref_run = os.environ.get("KIKA_GATE_REF_RUN", RUN_ID).strip()
    print(f"\n{'=' * 100}")
    print(f"  VALIDATION GATE — repr_mg vs `predictive` run_{ref_run}")
    print("=" * 100)
    if "mg" not in summaries:
        print("  [skip] mode `mg` was not run. Run it: it is the only check "
              "that this script's code path is faithful.")
        return
    ref_path = FIG_ROOT / "chi2_predictive" / f"run_{ref_run}" / "summary.json"
    if not ref_path.exists():
        print(f"  [skip] reference not found: {ref_path}")
        if ref_run == RUN_ID:
            print("       If this gate FAILED against run_082, that reference "
                  "predates the EN-RSL σ_E change — rebuild it as "
                  "`predictive_82_enrsl` and re-check with "
                  "KIKA_GATE_REF_RUN=082_enrsl.")
        return
    ref = json.loads(ref_path.read_text())

    worst = 0.0
    worst_where = ""
    for variant in ("V1", "V2", "V3", "V4"):
        for subset in SUBSETS:
            a = {r["library_key"]: r[f"chi2_{variant.lower()}_per_N"]
                 for r in summaries["mg"]["per_subset_all_variants"].get(subset, [])}
            b = {r["library_key"]: r[f"chi2_{variant.lower()}_per_N"]
                 for r in ref["per_subset_all_variants"].get(subset, [])}
            for lib in set(a) & set(b):
                if not (np.isfinite(a[lib]) and np.isfinite(b[lib])) or b[lib] == 0:
                    continue
                d = abs(a[lib] / b[lib] - 1.0)
                if d > worst:
                    worst, worst_where = d, f"{variant}/{subset}/{lib}"
    print(f"  worst relative difference: {worst:.3e}  ({worst_where})")
    if worst > 1e-6:
        print("  *** FAIL — repr_mg does not reproduce `predictive`.")
        if ref_run == RUN_ID:
            print("      Before concluding the code is wrong, check WHICH "
                  "libraries moved. JEFF and JENDL are never touched by this "
                  "sweep, so if they moved too, the difference is in the "
                  "scoring and not in the representation — almost certainly the "
                  "EN-RSL σ_E change (see the docstring). Rebuild the reference "
                  "and re-check with KIKA_GATE_REF_RUN=082_enrsl.")
        else:
            print("      Reference was rebuilt under matching σ_E, so this is a "
                  "real failure: precompute_chi2_representation.py is not a "
                  "faithful derivative. Fix it before reading any other mode.")
    else:
        print(f"  OK — the representation script reproduces run_{ref_run}.")


def size_table() -> None:
    """What each representation would actually cost on disk.

    Analytic, from the parameter count: MF34 stores the upper triangle of an
    (n_bins x n_orders) square matrix, six numbers per ENDF record line, 81
    bytes per line. Calibrated against the measured run-82 multigroup section
    (1,138,222 records, 92 MB) so the scaling is anchored, not guessed.
    """
    print(f"\n{'=' * 100}")
    print("  What each representation costs on disk (analytic, anchored on the")
    print("  measured run-82 multigroup MF34: 1,138,222 records / 92 MB)")
    print("=" * 100)
    print(f"    {'representation':28s}{'params':>10s}{'entries':>14s}"
          f"{'records':>14s}{'MB':>10s}{'vs fine L=6':>13s}")

    def rec(n_bins: int, n_orders: int):
        p = n_bins * n_orders
        entries = p * (p + 1) // 2
        records = entries / 6.0
        mb = records * 81 / 1e6
        return p, entries, records, mb

    _, _, _, mb_fine6 = rec(N_BINS["fine"], 6)
    for grid in ("fine", "mg"):
        for n_orders in (6, 5, 4, 3):
            p, entries, records, mb = rec(N_BINS[grid], n_orders)
            label = f"{grid} grid, l<={n_orders}"
            print(f"    {label:28s}{p:>10d}{entries:>14,d}{records:>14,.0f}"
                  f"{mb:>10.0f}{mb / mb_fine6:>12.2f}x")
    print("\n  Read this against the chi2 tables above: the question is not "
          "which\n  representation is smallest, it is whether the chi2 "
          "difference is worth\n  the factor in the last column.")


def main() -> None:
    print("Loading representation-mode summaries...")
    summaries = {}
    for mode in [m for m, _ in
                 BLOCK_GROUPING + BLOCK_COV_FINE + BLOCK_COV_MG + BLOCK_EVAL]:
        if mode in summaries:
            continue
        s = load_summary(mode)
        if s is not None:
            summaries[mode] = s
    if not summaries:
        raise SystemExit(
            "No representation summaries found. Run "
        "precompute_chi2_representation.py, then chi2_analysis_cluster.py with "
        "KIKA_CHI2_METHODOLOGIES=repr_<mode>,... first."
        )
    print(f"  found: {sorted(summaries)}")

    validation_gate(summaries)
    check_v2_invariance(summaries)

    v4 = table(summaries, "V4")
    v2 = table(summaries, "V2")

    show_block(v4, BLOCK_GROUPING, "V4",
               "Q1 — what multigroup collapse costs",
               "same MF4 and same MF33 in both files, so this is MF34 alone")
    show_block(v4, BLOCK_COV_FINE, "V4",
               "Q2a — what MF34 orders above N are worth (fine grid)",
               "MF4 central held at L=6; V4 only, V2 cannot move")
    show_block(v4, BLOCK_COV_MG, "V4",
               "Q2a — what MF34 orders above N are worth (multigroup)",
               "MF4 central held at L=6; V4 only, V2 cannot move")
    show_block(v2, BLOCK_EVAL, "V2",
               "Q2b — what evaluating above order N is worth (central values)",
               "MF4 and MF34 truncated together; V2 isolates the central values")
    show_block(v4, BLOCK_EVAL, "V4",
               "Q2b — what evaluating above order N is worth (full budget)",
               "MF4 and MF34 truncated together")

    size_table()

    out = FIG_ROOT / "representation_comparison.csv"
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
