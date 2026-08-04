"""Separate the two things the EN-RSL resolution fix changed.

The fix (scripts/tof_parameters.py + a rebuilt exfor_tof_parameters.json,
2026-07-30) made the pipeline read the incident-energy resolution EXFOR actually
declares. 74 of 98 subentries carry an EN-RSL* width; the median σ_E at 2 MeV is
~3.5x wider than the L=27.037 m / δt=5 ns default it replaced.

σ_E enters the problem in **two independent places**, and the whole point of
this script is that they must not be read as one number:

  1. THE SCORING. precompute_chi2_predictive.py folds each library's MF3+MF4
     over σ_E before comparing to the measurement. Change σ_E and every library
     scores differently — JEFF and JENDL included, even though neither has
     anything to do with our evaluation. This is why the repr_* sweep failed its
     validation gate against CHI_Figures/chi2_predictive/run_082: that reference
     was built 2026-07-28, before the fix, so the comparison spanned the change.

  2. THE EVALUATION. σ_E feeds compute_overlap_weight (exfor_utils.py:453),
     which decides which datasets constrain which energy bin in the
     kernel-weight MC. That is run 83.

Reading run 83 against run_082 would confound the two. So three runs, differing
one at a time:

    run_082         run-82 evaluation, OLD σ_E   (the existing reference)
    run_082_enrsl   run-82 evaluation, NEW σ_E   -> block A isolates SCORING
    run_083         run-83 evaluation, NEW σ_E   -> block B isolates EVALUATION

## What block B must show, and why it is self-checking

Measured 2026-07-31, before this ran: run 83's nominal_fits.parquet is
**byte-identical** to run 82's (md5 2ba03ec0...). The MF4 central values, the
fitted degrees, n_pts and τ did not move at all. That is not a null result — it
is a structural fact about where σ_E is used. compute_overlap_weight has exactly
one call site, reached only from the MC covariance stage (v2:3031); the nominal
fits never see it. Run 83 is a **covariance-only** re-evaluation.

The two covariance channels then responded **differently**, which was measured
directly on the shipped `_mg.endf` files (2026-07-31):

  MF33 (magnitude, c0)  diagonal BIT-IDENTICAL; only correlations moved
                        (49 % of entries by >0.01, max 0.31)
  MF34 (Legendre shape) diagonal NOT identical. Pooled over the six self-blocks,
                        the relative-sigma ratio 83/82 has median 1.0000 but
                        p10 0.18 / p90 1.73: ~51 % of bins are untouched and
                        ~41 % move by more than 20 %. Correlation changes reach
                        2.0, i.e. a full sign reversal.

So run 83 is **not** a uniform perturbation. It leaves roughly half the bins
exactly as they were and substantially rewrites the other half — which is what
you would expect, since Cierjacks alone is ~61 % of the points and its sigma_E
did not change, while the 74 EN-RSL subentries moved a lot.

Given the variant definitions in chi2_metrics.py --

    V1  diag(D + u² + v² + σ_eval_diag²)      uses eval variances
    V2  D + uuᵀ + vvᵀ                          NO σ_eval at all
    V3  (D + diag σ_eval_diag²) + uuᵀ + vvᵀ    uses eval variances
    V4  D + uuᵀ + vvᵀ + Σ_eval_block           uses the dense eval covariance

-- block B has three hard invariances and one expected mover:

    JEFF / JENDL, every variant   IDENTICAL (neither file changed)
    This_work V2                  IDENTICAL (V2 excludes σ_eval; centrals equal)
    This_work V1 / V3             EXPECTED TO MOVE — MF34's variances did change
    This_work V4                  EXPECTED TO MOVE — variances and correlations

Gates 1 and 2 are hard: a violation there is a real finding, not a tolerance
problem, because it means a knob moved that nobody intended to move.

Gate 3 (V1/V3) is reported as information, not as a pass/fail. An earlier
version of this file predicted V1/V3 would hold still, reasoning by analogy from
MF33's bit-identical diagonal. **That prediction was wrong** and the direct MF34
measurement above retired it. V1/V3 moving is now the expected outcome; V1/V3
holding *still* would be the surprise, and would mean the MF34 variance change
happens to cancel in the σ_eval diagonal the χ² actually consumes.

Reads only the summary.json each analysis run writes, so it costs nothing and
touches no covariance sidecar.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

FIG_ROOT = Path("/share_snc/snc/JuanMonleon/CHI_Figures")

LIBS = ["JEFF", "JENDL", "This_work"]
SUBSETS = ["all", "no_Cierjacks", "no_KS", "no_KS_no_Cierjacks",
           "only_KS", "only_Cierjacks"]
VARIANTS = ["V1", "V2", "V3", "V4"]

# Relative tolerance for "identical". The quantities compared are ratios of
# sums of ~10^5 floats, so exact equality is not the right test even when the
# inputs are bit-identical.
TOL = 1e-9

RUNS: List[Tuple[str, str]] = [
    ("082",       "run-82 evaluation, OLD sigma_E"),
    ("082_enrsl", "run-82 evaluation, NEW sigma_E"),
    ("083",       "run-83 evaluation, NEW sigma_E"),
]


def load_run(run_id: str) -> Optional[dict]:
    p = FIG_ROOT / "chi2_predictive" / f"run_{run_id}" / "summary.json"
    if not p.exists():
        print(f"  [missing] {p}")
        return None
    return json.loads(p.read_text())


def value(summary: dict, subset: str, lib: str, variant: str) -> Optional[float]:
    for row in summary["per_subset_all_variants"].get(subset, []):
        if row["library_key"] == lib:
            v = row.get(f"chi2_{variant.lower()}_per_N")
            return float(v) if v is not None and np.isfinite(v) else None
    return None


def rel(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None or old == 0:
        return None
    return new / old - 1.0


def comparison_block(a: dict, b: dict, title: str, why: str) -> None:
    """Print every (subset, library, variant) change from `a` to `b`."""
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"  {why}")
    print("=" * 100)
    for subset in SUBSETS:
        print(f"\n  subset: {subset}")
        print("    %-10s %s" % ("library", "  ".join(f"{v:>22}" for v in VARIANTS)))
        for lib in LIBS:
            cells = []
            for variant in VARIANTS:
                old, new = value(a, subset, lib, variant), value(b, subset, lib, variant)
                d = rel(new, old)
                if new is None:
                    cells.append(f"{'--':>22}")
                elif d is None:
                    cells.append(f"{new:>10.4f}{'':>12}")
                else:
                    cells.append(f"{new:>10.4f} ({100 * d:+7.2f}%)")
            print("    %-10s %s" % (lib, "  ".join(cells)))


def invariance_gates(enrsl: dict, r83: dict) -> bool:
    """The self-checks described in the module docstring. True if all pass."""
    print(f"\n{'=' * 100}")
    print("  INVARIANCE GATES — run_082_enrsl vs run_083")
    print("  Run 83 changed MF34 correlations only. Everything else must hold still.")
    print("=" * 100)

    ok = True

    # Gate 1 — the other two libraries did not change at all.
    worst, where = 0.0, ""
    for subset in SUBSETS:
        for lib in ("JEFF", "JENDL"):
            for variant in VARIANTS:
                d = rel(value(r83, subset, lib, variant),
                        value(enrsl, subset, lib, variant))
                if d is not None and abs(d) > worst:
                    worst, where = abs(d), f"{variant}/{subset}/{lib}"
    print(f"\n  [1] JEFF and JENDL unchanged, every variant")
    print(f"      worst relative difference: {worst:.3e}  ({where})")
    if worst > TOL:
        ok = False
        print("      *** FAIL — a library this run never touched moved. Something")
        print("          other than the This_work file differs between the two runs;")
        print("          suspect a changed sigma_E, manifest or EXFOR selection.")
    else:
        print("      OK")

    # Gate 2 — V2 has no eval covariance in it, and the centrals are identical.
    worst, where = 0.0, ""
    for subset in SUBSETS:
        d = rel(value(r83, subset, "This_work", "V2"),
                value(enrsl, subset, "This_work", "V2"))
        if d is not None and abs(d) > worst:
            worst, where = abs(d), f"V2/{subset}/This_work"
    print(f"\n  [2] This_work V2 unchanged (V2 excludes sigma_eval)")
    print(f"      worst relative difference: {worst:.3e}  ({where})")
    if worst > TOL:
        ok = False
        print("      *** FAIL — V2 depends only on the residuals and the EXFOR")
        print("          uncertainties. run 83's nominal_fits.parquet is byte-identical")
        print("          to run 82's, so the MF4 central values cannot have moved.")
        print("          If V2 moved, the MF3 host or the fold changed too.")
    else:
        print("      OK — the central values really are identical.")

    # Gate 3 — variances (soft: predicted from MF33, not measured on MF34).
    worst, where = 0.0, ""
    for subset in SUBSETS:
        for variant in ("V1", "V3"):
            d = rel(value(r83, subset, "This_work", variant),
                    value(enrsl, subset, "This_work", variant))
            if d is not None and abs(d) > worst:
                worst, where = abs(d), f"{variant}/{subset}/This_work"
    print(f"\n  [3] This_work V1/V3 — informational, NOT a pass/fail")
    print(f"      worst relative difference: {worst:.3e}  ({where})")
    if worst > TOL:
        print("      As expected. MF34's variances did change between run 82 and")
        print("      run 83 (measured on the _mg files: ~41 % of bins move by >20 %,")
        print("      p10 0.18 / p90 1.73), unlike MF33's, whose diagonal is")
        print("      bit-identical. The two covariance channels responded")
        print("      differently to the membership change; say so in the write-up")
        print("      rather than calling run 83 a correlation-only change.")
    else:
        print("      SURPRISING — MF34's variances demonstrably changed, so V1/V3")
        print("      holding still means that change cancels in the sigma_eval")
        print("      diagonal the chi2 consumes. Worth understanding before")
        print("      quoting V4.")

    return ok


def headline(a: dict, b: dict, label_a: str, label_b: str) -> None:
    """The V4 numbers on the full dataset, which is what gets quoted."""
    print(f"\n{'=' * 100}")
    print("  HEADLINE — V4 chi2/N, subset `all`")
    print("=" * 100)
    print("    %-12s %14s %14s %10s" % ("library", label_a, label_b, "change"))
    for lib in LIBS:
        old, new = value(a, "all", lib, "V4"), value(b, "all", lib, "V4")
        d = rel(new, old)
        print("    %-12s %14.4f %14.4f %9s" % (
            lib,
            old if old is not None else float("nan"),
            new if new is not None else float("nan"),
            f"{100 * d:+.2f}%" if d is not None else "--",
        ))


def main() -> None:
    print("=== EN-RSL: scoring effect vs evaluation effect ===")
    summaries: Dict[str, dict] = {}
    for run_id, desc in RUNS:
        s = load_run(run_id)
        if s is not None:
            summaries[run_id] = s
            print(f"  loaded run_{run_id:<10} {desc}")

    if "082_enrsl" not in summaries or "083" not in summaries:
        raise SystemExit(
            "\nNeed both run_082_enrsl and run_083 to say anything. Build them:\n"
            "  KIKA_RUN_TAG=82_enrsl python precompute_chi2_predictive.py\n"
            "  KIKA_THIS_WORK_DIR=.../new_test_83 KIKA_RUN_TAG=83 "
            "python precompute_chi2_predictive.py"
        )

    # Block A — the scoring effect. Needs the pre-fix reference to exist.
    if "082" in summaries:
        comparison_block(
            summaries["082"], summaries["082_enrsl"],
            "BLOCK A — what the sigma_E fix did to the SCORING",
            "Same run-82 evaluation in both. Every library moves, ours included.",
        )
        headline(summaries["082"], summaries["082_enrsl"], "old sigma_E", "new sigma_E")
    else:
        print("\n  [skip] BLOCK A — run_082 not found, cannot measure the scoring effect.")

    # Block B — the evaluation effect. This is the run-83 result.
    comparison_block(
        summaries["082_enrsl"], summaries["083"],
        "BLOCK B — what run 83 did to the EVALUATION",
        "Same sigma_E in both. Only the This_work file differs.",
    )
    headline(summaries["082_enrsl"], summaries["083"], "run 82", "run 83")

    ok = invariance_gates(summaries["082_enrsl"], summaries["083"])

    print(f"\n{'=' * 100}")
    print("  HOW TO READ THIS")
    print("=" * 100)
    print("""
  Block A is not a result about our evaluation — it is the size of the
  correction to the measuring instrument. It also tells you how much of the
  thesis chi2 table would move if the declared resolutions were adopted for
  scoring, which is a manuscript decision (docs/thesis_chi2_review.md S12),
  not a research one.

  Block B is the run-83 result. Because run 83 left the central values
  byte-identical, V4 is the ONLY place its work can show up: a better-matched
  set of datasets per bin changes the MF34 correlations, and V4 is the only
  variant that reads the dense evaluated covariance. If V4 barely moves, the
  honest conclusion is that the resolution fix does not materially change the
  shipped covariance -- which would retire it as a research question and leave
  it a correctness fix worth one sentence.

  What run 83 does NOT do, despite what the roadmap assumed: it does not move
  a_l(E). compute_overlap_weight is only reached from the MC covariance stage,
  so the nominal fit never sees sigma_E. Every order-treatment question in
  S4 of the roadmap is therefore unaffected by run 83, and Gate 2 on run 83 is
  identical to Gate 2 on run 82 -- already confirmed, same parquet.
""")
    if not ok:
        raise SystemExit("Invariance gates FAILED — read them before the tables.")


if __name__ == "__main__":
    main()
