"""
Headless cluster runner for the chi^2 analysis with all four chi^2 variants.

Replicates `chi2_analysis_exfor_c0.ipynb` and `chi2_analysis_library_c0.ipynb`
without a Jupyter kernel: loads the per-row parquet + the per-experiment dense
Σ_eval .npz sidecar (both produced by `precompute_chi2_*_c0.py`), runs all
four chi^2 variants (V1/V2/V3/V4) on six subsets, and writes:

  - **Headline summary** at the top of `report.md`: χ²/N per (library × subset)
    under PRIMARY_VARIANT and the per-experiment win count
    (best / mid / worst) per library. This is the chapter headline data.
  - Per-subset decision table (V1/V2/V3/V4 × libraries side-by-side).
  - Per-subset residual centering table — mean/median/std/p10/p90 of r/y per
    library (σ-free; exposes uniform multiplicative biases that the rank-2
    variants would otherwise hide).
  - Per-subset coverage table — % of points within k·σ (V1 textbook σ) per
    library, target N(0,1) 68/95/99.7%; tells you whether each library's
    uncertainty actually covers the data.
  - Per-subset per-experiment ranking — count of experiments where each
    library has best / mid / worst χ²/N under each variant.
  - Per-experiment CSV per subset with all 4 variants per library
    (per_experiment_<subset>.csv) — fine-grained comparison data.
  - The full 9-figure diagnostic set for one PRIMARY_VARIANT (default V1
    textbook diag): chi² bar, per-experiment histogram, contribution,
    waterfall, vs-year, whitened-residual stats + histograms + vs-energy
    scatter, plus a `per_experiment_win_<subset>_<variant>` bar chart showing
    log-ratio of This_work vs min(JEFF, JENDL) per experiment.
  - A clean primary-variant summary figure (χ²/N per subset × library) and a
    combined-comparison figure spanning every (subset × variant) cell.

Subsets:
  - `all`                  — every experiment.
  - `no_Cierjacks`         — recommended thesis headline (Cierjacks 1978 has
                             ~61% of points and is a known difficult dataset).
  - `no_KS`                — old default (Kinney 1976 + Smith 1980 excluded).
  - `no_KS_no_Cierjacks`   — both K&S and Cierjacks excluded.
  - `only_KS`              — Kinney + Smith only.
  - `only_Cierjacks`       — Cierjacks 1978 only (analyse the difficult dataset
                             on its own).

Re-running with a different PRIMARY_VARIANT (and a bumped RUN_ID) generates
the chapter figures for that variant without touching the precompute step.
RUN_ID isolates each run in its own directory so reruns never overwrite.

Output layout per methodology:

    <REPORT_DIR>/run_<RUN_ID>/
      report.md           (headline summary + per-subset detail)
      summary.json        (machine-readable; includes residual/coverage/win
                           tally tables)
      figures/            (PNG)
      per_experiment/per_experiment_<subset>.csv

All knobs are configured in the CONFIGURATION block below — run with:

    python scripts/chi2_analysis_cluster.py
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

import chi2_metrics


# ── CONFIGURATION ────────────────────────────────────────────────────────────
# Edit these constants in-place; the script takes no CLI arguments.

# Run versioning. Outputs land in `<REPORT_DIR>/run_<RUN_ID>/`; bump this to
# start a fresh run without overwriting earlier ones.
#
# Overridable from the environment so scoring a different pipeline run lands in
# its own directory instead of overwriting run_082 — which the thesis rests on.
RUN_ID: str = os.environ.get("KIKA_CHI2_RUN_ID", "082").strip()

# Which methodologies to run. Any non-empty subset of the PATHS keys. Each entry
# needs its parquet + .eval_cov.npz sidecar built first by the matching
# scripts/precompute_chi2_*.py. The folded_al_c0_ns{1,3} entries are built by
# scripts/precompute_chi2_folded_al_c0.py (set N_SIGMA=1.0 then 3.0 and rerun).
# Original headline set: ["exfor_c0", "library_c0", "folded_c0"].
METHODOLOGIES_TO_RUN: List[str] = ["predictive", "exfor_c0"]

# Overridable from the environment so a sweep script (run_fold_sweep.sh) can
# select a different set without editing this file. Comma-separated.
if os.environ.get("KIKA_CHI2_METHODOLOGIES"):
    METHODOLOGIES_TO_RUN = [
        m.strip() for m in os.environ["KIKA_CHI2_METHODOLOGIES"].split(",")
        if m.strip()
    ]

# Per-methodology I/O. The .npz sidecar is inferred as `<parquet>.eval_cov.npz`.
#
# `systematic_block_col` controls how the EXFOR rank-1 systematics (u, v) are
# correlated in the rank-2/dense variants (V2/V3/V4):
#   - "energy_mev" → correlated only *within* each incident energy.
#   - None         → one experiment-wide correlated mode (across all energies).
# Use "energy_mev" for exfor_c0: there c₀ is fit independently per incident
# energy, so a single experiment-wide normalization mode would penalise the
# energy-to-energy level differences the per-energy fit already removed —
# inflating χ² for multi-energy experiments (Kinney by ~5×). library_c0 takes
# c₀ from the library's own MF3 (not fit per energy), so its 8% normalization is
# a genuine global mode and stays None.
PATHS: Dict[str, Dict[str, Optional[str]]] = {
    # PRIMARY. Each library judged by what it ships: its own MF3 + MF4 forward-
    # folded through each experiment's TOF resolution (product fold), covariance
    # = its own MF34 + MF33. Written by scripts/precompute_chi2_predictive.py,
    # which supersedes library_c0 / folded_c0 / folded_al_c0 (each a special
    # case of it). c0 is a smooth function of energy, not a per-energy fit, so
    # the normalization is a genuine experiment-wide mode → block col None.
    "predictive": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_82.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — shipped MF3+MF4 resolution-folded, MF34+MF33 eval σ",
        "systematic_block_col": None,
    },
    # ── The EN-RSL pair. Same code, same libraries, same everything as
    # `predictive` above; they differ only in which σ_E the *fold* uses and
    # which pipeline run supplies This_work.
    #
    # `predictive` itself was built 2026-07-28, before the declared EXFOR
    # resolutions were read (scripts/tof_parameters.py, deployed 2026-07-30).
    # σ_E is now ~3.5× wider for the 74 subentries that declare EN-RSL*, and the
    # fold is part of the *scoring*, not just the evaluation — so every library,
    # JEFF and JENDL included, scores differently now. That is why `predictive`
    # cannot be compared against anything built after 2026-07-30, and why
    # repr_mg failed its validation gate against it.
    #
    #   predictive_82_enrsl : run-82 evaluation, new σ_E  → isolates the effect
    #                         of the fix on SCORING. Also the repaired reference
    #                         the repr_* gate should be checked against.
    #   predictive_83       : run-83 evaluation, new σ_E  → isolates the effect
    #                         of the fix on the EVALUATION, read against
    #                         predictive_82_enrsl.
    #
    # Two invariances make this pair self-checking. Run 83's nominal_fits.parquet
    # is byte-identical to run 82's, so between these two entries:
    #   • JEFF and JENDL must be identical in every variant (neither file moved);
    #   • This_work V2 must be identical (V2 excludes σ_eval, and the MF4 central
    #     values did not move).
    # Only V4 should move — run 83 changed MF34 *correlations*, not variances.
    # Anything else moving means an unintended knob moved with it.
    "predictive_82_enrsl": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_82_enrsl.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — run-82 evaluation, declared EN-RSL σ_E in the fold",
        "systematic_block_col": None,
    },
    "predictive_83": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_83.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — run-83 evaluation (EN-RSL), declared σ_E in the fold",
        "systematic_block_col": None,
    },
    # Run 84 = run 83 re-evaluated with the τ-IRLS refit on the GLS kernel
    # (TAU_REFIT_USE_GLS=True). Single-variable against predictive_83: same
    # σ_E, same EN-RSL, and the runner pins DEGREE_WEIGHT_FLOOR=0.01 and
    # MC_CAP_FROM_SUPPORT_ONLY=0 back to v2 so the fork's other two research
    # knobs stay out of it. The τ-GLS solver is the ONLY difference.
    #
    # Unlike run 83 this DOES move the central values — median |Δa|/|a| ≈ 5 %
    # at ℓ=1 and ≈ 13 % at ℓ=5, with frozen_degree changing in 12 % of bins —
    # so V2 is expected to move here, and V2 is the variant that actually
    # answers the question (it is the pure central-value metric; V4 mixes in
    # the covariance, which the solver change also perturbs).
    "predictive_84": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_84.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — run-84 evaluation (τ-IRLS on GLS), declared σ_E in the fold",
        "systematic_block_col": None,
    },
    # Run 85 = run 84 + the Phase-3 mixture. Single-variable against
    # predictive_84: they share the τ-GLS solver, σ_E, the manifest and
    # DEGREE_WEIGHT_FLOOR=0.01. Verified on the parquets: run 85's `win_c_*` is
    # bit-identical to run 84's `c_*` and `frozen_degree` moves in 0 % of bins,
    # so the underlying fit is the same and only what is *shipped* differs.
    #
    # ⚠ READ V2 **AND** V4, and do NOT carry run 83's reading over. Run 83 left
    # the centrals byte-identical, which made V2 an invariance gate there. Run 85
    # moves both channels — MF4 ships the mixture mean (median |Δa|/|a| 2 % at
    # ℓ=1 rising to 20 % at ℓ=5,6) and MF34 ships the mixture covariance — so:
    #   V2 → did the mixture MEAN help the central value?
    #   V4 → did the mixture COVARIANCE change the fit?
    # Neither is a gate. JEFF and JENDL remain hard invariances.
    "predictive_85": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_85.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — run-85 evaluation (τ-GLS + Phase-3 mixture), declared σ_E in the fold",
        "systematic_block_col": None,
    },
    # Run 85 again, but scoring the FINE ENDF instead of the multigroup one.
    # This is the first measurement of the Phase-3 mixture that is not empty.
    #
    # WHY IT IS NEEDED. `predictive_85` scored `_mg.endf`, and the mixture never
    # reached that file: dead parameters at ℓ=6 go 92.3 % → 1.0 % on the fine
    # grid and 81.6 % → 81.7 % on the multigroup one (roadmap §8.4). So run 85's
    # V4 (−2.25 %) and its flat calibration measured the mixture's *absence*.
    # The fine ENDF already carries it — no re-evaluation required.
    #
    # The two files carry the SAME MF4 centrals and the SAME MF33 (both ship the
    # ungrouped 1738-group MF33), so `predictive_85_fine` vs `predictive_85` is a
    # pure MF34-representation difference — the same clean contrast §3.1 relies
    # on. Its own baseline is `repr_fine` from the representation sweep, which is
    # the fine product without the mixture.
    #
    # Needs 300G: the fine MF34 is ~734 MB on disk, ~0.5 GB parsed.
    "predictive_85_fine": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_85_fine.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — run-85 evaluation, FINE MF34 (τ-GLS + Phase-3 mixture), declared σ_E in the fold",
        "systematic_block_col": None,
    },
    # Run 86 — the §8.4 fix, in the product we actually ship.
    #
    # Run 85 proved Phase 3 reached the fine MF34 and not the multigroup one,
    # because `perform_adaptive_multigroup_collapse` rebuilt the legacy
    # winner-take-all mask internally from `nr.frozen_degree`. That is now
    # overridable via `valid_orders_fn` (default = legacy, so v2 stays
    # bit-identical) and the fork passes its q_ℓ rule through. Run 86 is the
    # re-evaluation that makes the fix take effect.
    #
    # Read it against `predictive_85_fine` (the mixture, fine grid — what the
    # covariance does when it is present) and `predictive_85` (the mixture
    # absent from the multigroup file — the bug). Run 86 should land near the
    # former, not the latter.
    #
    # Also carries the first §9.4.2 [XCORR] measurement: Cov(c₀, a_ℓ) over the
    # shared Pass-2 replicas. Diagnostic-only, so it cannot move any number in
    # this table — read it from run 86's log, not from here.
    #
    # 100G suffices: this scores `_mg.endf` (~205 MB), not the fine file.
    "predictive_86": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_86.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — run-86 evaluation (τ-GLS + Phase-3 mixture, multigroup mask fixed), declared σ_E in the fold",
        "systematic_block_col": None,
    },
    # Fold-mode sweep. Identical to `predictive` in every respect except which
    # part of the forward model is resolution-averaged (FOLD_MODE in
    # precompute_chi2_predictive.py). Compare V2 across these to decide, over
    # every bin in the database rather than a handful of hand-picked energies,
    # whether the product fold is really the right convention.
    "predictive_factors": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_82_factors.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive_factors",
        "title":      "χ² analysis — predictive, factors averaged <σ>·F(<a_l>)",
        "systematic_block_col": None,
    },
    "predictive_foldsigma": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_82_foldsigma.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive_foldsigma",
        "title":      "χ² analysis — predictive, magnitude averaged only <σ>·F(a_l(E0))",
        "systematic_block_col": None,
    },
    "predictive_foldal": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_82_foldal.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive_foldal",
        "title":      "χ² analysis — predictive, Legendre shape averaged only σ(E0)·F(<a_l>)",
        "systematic_block_col": None,
    },
    "predictive_nofold": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_82_nofold.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive_nofold",
        "title":      "χ² analysis — predictive, no resolution model σ(E0)·F(a_l(E0))",
        "systematic_block_col": None,
    },
    "exfor_c0": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_exfor_c0_82.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_exfor_c0",
        "title":      "χ² analysis — c₀ from EXFOR Kinney/Smith fit, MF34 eval σ",
        "systematic_block_col": "energy_mev",
    },
    "library_c0": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_library_c0_82.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_library_c0",
        "title":      "χ² analysis — c₀ from library MF3, MF34+MF33 eval σ",
        "systematic_block_col": None,
    },
    # Fair-c₀ variant: each library's own MF3 elastic σ folded over each
    # experiment's TOF energy resolution → c₀; MF34-only (value-only) eval σ.
    # c₀ is a smooth function of energy (not a per-energy fit), so the 8%-style
    # normalization is a genuine experiment-wide mode → systematic_block_col=None,
    # same as library_c0. Parquet written by scripts/precompute_chi2_folded_c0.py.
    "folded_c0": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_folded_c0_82.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_folded_c0",
        "title":      "χ² analysis — c₀ from resolution-folded library MF3, MF34 eval σ",
        "systematic_block_col": None,
    },
    # Both-folded variant: like folded_c0 but the angular shape a_l is also folded
    # over each experiment's TOF resolution window (truncated Gaussian, ±N_SIGMA·σ_E).
    # One entry per window half-width; parquets written by
    # scripts/precompute_chi2_folded_al_c0.py (edit N_SIGMA and rerun for each).
    # Same global-normalization reasoning as folded_c0 → systematic_block_col=None.
    "folded_al_c0_ns1": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_folded_al_c0_82_ns1.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_folded_al_c0_ns1",
        "title":      "χ² analysis — c₀ and a_l resolution-folded (±1σ window), MF34 eval σ",
        "systematic_block_col": None,
    },
    "folded_al_c0_ns3": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_folded_al_c0_82_ns3.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_folded_al_c0_ns3",
        "title":      "χ² analysis — c₀ and a_l resolution-folded (±3σ window), MF34 eval σ",
        "systematic_block_col": None,
    },
}

# MF34 representation study — fine vs multigroup, and Legendre-order truncation.
# Registered programmatically because the mode table lives in the precompute
# script and duplicating a dozen near-identical dict literals here would
# guarantee they drift apart. Written by
# scripts/precompute_chi2_representation.py (REPR_MODE=<mode>); everything
# except the This_work MF4/MF34 representation is identical to `predictive`,
# so systematic_block_col matches it.
#
# Read the results as:
#   repr_fine  vs repr_mg          -> what multigroup collapse costs (V4 only;
#                                     the two files share MF4 and MF33)
#   repr_fine  vs repr_fine_covN   -> what MF34 orders > N are worth (V4 only)
#   repr_fine  vs repr_fine_evalN  -> what evaluating above order N is worth
#                                     (V2 and V4)
# repr_mg must reproduce `predictive` run 082. It is the validation gate, not
# a result.
_REPR_MODES = {
    "fine":       "fine grid, L=6 — the reference",
    "mg":         "multigroup grid, L=6 — must reproduce `predictive`",
    "fine_cov3":  "fine grid, MF4 L=6, MF34 l<=3",
    "fine_cov4":  "fine grid, MF4 L=6, MF34 l<=4",
    "fine_cov5":  "fine grid, MF4 L=6, MF34 l<=5",
    "mg_cov3":    "multigroup grid, MF4 L=6, MF34 l<=3",
    "mg_cov4":    "multigroup grid, MF4 L=6, MF34 l<=4",
    "mg_cov5":    "multigroup grid, MF4 L=6, MF34 l<=5",
    "fine_eval3": "fine grid, MF4 and MF34 both l<=3",
    "fine_eval4": "fine grid, MF4 and MF34 both l<=4",
    "fine_eval5": "fine grid, MF4 and MF34 both l<=5",
}
for _mode, _desc in _REPR_MODES.items():
    PATHS[f"repr_{_mode}"] = {
        "parquet":    f"/share_snc/snc/JuanMonleon/chi2/chi2_data_repr_82_{_mode}.parquet",
        "report_dir": f"/share_snc/snc/JuanMonleon/CHI_Figures/chi2_repr_{_mode}",
        "title":      f"χ² analysis — MF34 representation: {_desc}",
        "systematic_block_col": None,
    }

# Analysis window (MeV), K&S anchor experiment IDs, Cierjacks anchor IDs.
E_MIN_MEV = 0.85
E_MAX_MEV = 4.0
KINNEY_SMITH_IDS: List[str] = ["10571002", "10886002"]
# Cierjacks 1978 (20743002) — 28,631 points (~61% of the dataset), high TOF
# resolution but known angular-shape disagreement with all modern evaluations
# at backward angles; broken out so it does not dominate the global chi² aggregate.
CIERJACKS_IDS: List[str] = ["20743002"]

# Figure output. PNG @ 200 dpi keeps each report dir under ~50 MB; bump to 300
# to match the notebooks' on-screen quality if you intend to publish from these.
FIGURE_FORMAT = "png"
FIGURE_DPI = 200

# Library display config. Keys must match the `library` column in the parquet.
LIB_LABELS = {"JEFF": "JEFF-4.0", "JENDL": "JENDL-5", "This_work": "This work"}
LIB_COLORS = {"JEFF": "#1f77b4", "JENDL": "#2ca02c", "This_work": "#d62728"}
LIB_MARKERS = {"JEFF": "o", "JENDL": "s", "This_work": "D"}
LIB_YEAR_OFFSETS = {"JEFF": -0.3, "JENDL": 0.0, "This_work": 0.3}

# Chi² variants — all four are computed for every subset. u = σ_indep·y is
# the per-experiment normalization rank-1; v = σ_dep·y is the per-experiment
# shape rank-1 (per-row amplitude). Both are correlated within an experiment
# (manifest ``correlated: true``); putting either on the diagonal would treat
# correlated systematics as independent noise.
# - V1 textbook diag    : Σ = diag(σ_stat² + (σ_indep·y)² + (σ_dep·y)² + σ_eval_diag²)
# - V2 rank-2 only      : Σ = D + u uᵀ + v vᵀ          (no eval cov)
# - V3 rank-2 + diag    : Σ = (D + diag σ_eval_diag²) + u uᵀ + v vᵀ
# - V4 rank-2 + dense   : Σ = D + u uᵀ + v vᵀ + Σ_eval_block
#
# REPORTED PAIR: V2 and V4. Chapter 3 ships chi2_V2.png (EXFOR-only budget) and
# chi2_V4.png (full budget), and the per-experiment strip figure is V4. V1 and V3
# are the diagonal approximations of V2 and V4 — they treat correlated
# systematics (V1) or the evaluation's energy/order correlations (V3) as
# independent noise — so they are kept as cross-checks and not reported.
# Σ_eval being rank-deficient on its own is not a problem for V4: it enters only
# as D + u uᵀ + v vᵀ + Σ_eval, and D is strictly positive.
# NOTE: an earlier version of this comment called V3 the chapter headline and V4
# "diagnostic only". That contradicts what the thesis actually ships; see
# docs/thesis_chi2_review.md before acting on either claim.
# Reported variants. V1 and V3 are DELIBERATELY OMITTED (Juan, 2026-08-03): the
# two that carry the argument are V2 (rank-2 only — the pure central-value
# metric, σ_eval excluded) and V4 (rank-2 + dense eval — the only one that sees
# the off-diagonal structure Phase 3 actually adds). V1/V3 are diagonal
# statistics that mostly restate the coverage table.
#
# ⚠ What this does and does not save. It halves the report tables and the
# per-(subset × variant) figures — the bulk of the wall-clock in the reporting
# stage. It does NOT save the χ² computation: the per-experiment chi2_v1..v4
# columns are still built upstream, and the expensive one is V4's dense eval
# blocks, which we keep. Add "V1"/"V3" back here to restore them; nothing else
# needs changing.
#
# Coverage is unaffected: coverage_table() computes σ_total itself and never
# reads VARIANTS, so |z|<1 stays even though its heading names V1's σ.
VARIANTS: Tuple[str, ...] = ("V2", "V4")
VARIANT_LABELS: Dict[str, str] = {
    "V1": "V1 textbook diag",
    "V2": "V2 rank-2 only",
    "V3": "V3 rank-2 + diag eval",
    "V4": "V4 rank-2 + dense eval",
}
# Which variant gets the full 9-figure diagnostic set per subset. V1 textbook
# diag is the field-standard chi² formulation (every uncertainty on the
# diagonal); it is the conservative interpretation of correlated systematics
# and avoids the rank-1/rank-2 absorption artefact that lets libraries with
# uniform multiplicative biases (e.g. JEFF vs Cierjacks) get a free pass.
# V2/V3/V4 are still computed and tabulated for cross-check.
PRIMARY_VARIANT: str = "V4"

# Subsets — six slices of the dataset. `no_Cierjacks` is the recommended
# headline because Cierjacks 1978 contributes 61% of all points and is a
# known difficult dataset for all modern Fe-56 angular evaluations.
SUBSETS: List[Tuple[str, str]] = [
    ("all",                "All experiments"),
    ("no_Cierjacks",       "Excluding Cierjacks 1978"),
    ("no_KS",              "Excluding K&S"),
    ("no_KS_no_Cierjacks", "Excluding K&S and Cierjacks 1978"),
    ("only_KS",            "K&S only"),
    ("only_Cierjacks",     "Cierjacks 1978 only"),
]

# Contribution-pie threshold (%): experiments below this are merged into
# "Other" in the contribution bar chart.
CONTRIB_THRESHOLD_PCT = 1.0

# ── End of CONFIGURATION ─────────────────────────────────────────────────────


# ── Helpers ──────────────────────────────────────────────────────────────────


@dataclass
class RunPaths:
    parquet: Path
    eval_cov: Path
    figures_dir: Path
    report_path: Path
    summary_json: Path
    per_exp_csv_dir: Path
    run_dir: Path
    title: str
    systematic_block_col: Optional[str]


def build_paths(methodology: str) -> RunPaths:
    """Resolve all output paths for one methodology, scoped under run_<RUN_ID>."""
    cfg = PATHS[methodology]
    parquet_path = Path(cfg["parquet"])
    eval_cov_path = parquet_path.with_suffix(parquet_path.suffix + ".eval_cov.npz")
    run_dir = Path(cfg["report_dir"]) / f"run_{RUN_ID}"
    return RunPaths(
        parquet=parquet_path,
        eval_cov=eval_cov_path,
        figures_dir=run_dir / "figures",
        report_path=run_dir / "report.md",
        summary_json=run_dir / "summary.json",
        per_exp_csv_dir=run_dir / "per_experiment",
        run_dir=run_dir,
        systematic_block_col=cfg.get("systematic_block_col"),
        title=cfg["title"],
    )


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "figure.dpi": FIGURE_DPI, "savefig.dpi": FIGURE_DPI,
        "axes.grid": True, "grid.alpha": 0.3,
        "font.size": 14, "axes.labelsize": 15, "axes.titlesize": 15,
        "legend.fontsize": 12, "xtick.labelsize": 13, "ytick.labelsize": 13,
    })


def save_and_close(fig, figures_dir: Path, name: str, log: List[str]) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.{FIGURE_FORMAT}"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.append(f"  saved {path.name}")
    return path


def _fmt_cell(val, float_format: str) -> str:
    if isinstance(val, float):
        if np.isnan(val):
            return ""
        return float_format % val
    if isinstance(val, (np.floating,)):
        v = float(val)
        return "" if np.isnan(v) else float_format % v
    return str(val)


def _df_to_md(df: pd.DataFrame, *, float_format: str = "%.3f",
              index: bool = False) -> str:
    """pandas → GitHub-flavored Markdown table. No `tabulate` dependency."""
    work = df.copy()
    if index:
        work = work.reset_index()
    headers = [str(c) for c in work.columns]
    rows = [[_fmt_cell(v, float_format) for v in row]
            for row in work.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines) + "\n"


def _df_to_text(df: pd.DataFrame, *, float_format: str = "%.3f",
                index: bool = False) -> str:
    return df.to_string(index=index, float_format=float_format.__mod__)


def _v_low(variant: str) -> str:
    return variant.lower()


# ── All-variants aggregation ────────────────────────────────────────────────


def aggregate_all_variants_by_library(
    per_exp_av: pd.DataFrame, libraries: List[str],
) -> pd.DataFrame:
    """Sum chi²_v{1..4} and N across experiments within each library.

    Returns columns: Library (display), library_key, N,
    chi2_v1..v4, chi2_v1_per_N..v4_per_N.
    """
    agg_spec = {"N": ("N", "sum")}
    for v in ("v1", "v2", "v3", "v4"):
        agg_spec[f"chi2_{v}"] = (f"chi2_{v}", "sum")
    grp = per_exp_av.groupby("library", observed=True).agg(**agg_spec).reset_index()
    for v in ("v1", "v2", "v3", "v4"):
        grp[f"chi2_{v}_per_N"] = np.where(
            grp["N"] > 0, grp[f"chi2_{v}"] / grp["N"], np.nan
        )
    grp = grp.rename(columns={"library": "library_key"})
    grp["Library"] = grp["library_key"].map(LIB_LABELS).fillna(grp["library_key"])
    # Preserve library display order.
    grp["__order"] = grp["library_key"].map({lib: i for i, lib in enumerate(libraries)})
    grp = grp.sort_values("__order").drop(columns="__order").reset_index(drop=True)
    return grp


def write_decision_table(
    decision_summary: pd.DataFrame, subset_key: str, subset_label: str,
    report: List[str],
) -> None:
    """Markdown decision table: libraries × {V1, V2, V3, V4} chi²/N."""
    cols = ["Library", "N"] + [f"chi2_{v.lower()}_per_N" for v in VARIANTS]
    display = decision_summary[cols].copy()
    display.columns = ["Library", "N"] + [f"{v} χ²/N" for v in VARIANTS]
    report.append(
        f"\n**Decision table — χ²/N per (library × variant), "
        f"subset `{subset_key}` ({subset_label}):**\n\n"
    )
    report.append(_df_to_md(display, float_format="%.3f") + "\n")


def write_subset_csv(
    per_exp_av: pd.DataFrame, df_subset: pd.DataFrame, libraries: List[str],
    subset_key: str, per_exp_csv_dir: Path, fig_log: List[str],
) -> Path:
    """Per-experiment CSV with all four variants per library, one CSV per subset.

    Columns:  experiment_id | author | year | E_min | E_max | N |
              <lib>_V1 χ²/N | <lib>_V2 χ²/N | <lib>_V3 χ²/N | <lib>_V4 χ²/N
              for each library in `libraries` (ordered to match LIB_LABELS).
    """
    meta = (
        df_subset.groupby(["experiment_id", "library"], observed=True)
        .agg(author=("author", "first"), year=("year", "first"),
             E_min=("energy_mev", "min"), E_max=("energy_mev", "max"))
        .reset_index()
    )
    exp_df = per_exp_av.merge(meta, on=["experiment_id", "library"])
    exp_df["year"] = exp_df["year"].apply(lambda y: int(y) if pd.notna(y) else "")

    pivots = []
    for v in ("v1", "v2", "v3", "v4"):
        p = exp_df.pivot_table(
            index=["experiment_id", "author", "year", "E_min", "E_max"],
            columns="library", values=f"chi2_{v}_per_N",
        )
        ordered = [lib for lib in libraries if lib in p.columns]
        p = p[ordered]
        rename = {lib: f"{lib}_{v.upper()}" for lib in ordered}
        pivots.append(p.rename(columns=rename))

    wide = pd.concat(pivots, axis=1)
    n_per_exp = exp_df.groupby("experiment_id")["N"].first()
    wide = wide.join(n_per_exp, on="experiment_id")
    val_cols = [c for c in wide.columns if c != "N"]
    if val_cols:
        wide["max_chi2_per_N"] = wide[val_cols].max(axis=1)
        wide = wide.sort_values("max_chi2_per_N", ascending=False)
        wide = wide.drop(columns="max_chi2_per_N")

    per_exp_csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = per_exp_csv_dir / f"per_experiment_{subset_key}.csv"
    wide.to_csv(csv_path)
    fig_log.append(f"  saved {csv_path.relative_to(per_exp_csv_dir.parent)}")
    return csv_path


# ── Per-subset diagnostic tables (residual, coverage, win tally) ────────────


def mean_residual_table(
    df_subset: pd.DataFrame, libraries: List[str], subset_key: str,
    report: List[str],
) -> pd.DataFrame:
    """Per-library mean/median/std/p10/p90 of r/y and mean of |r|/|y|.

    Centering metric independent of σ. JEFF's uniform 22% Cierjacks bias is
    invisible to V3 chi² (rank-1 absorbs it) but glaring here.
    """
    rows = []
    for lib in libraries:
        sub = df_subset[df_subset["library"] == lib]
        if len(sub) == 0:
            continue
        y_abs = sub["y_exp"].abs()
        ok = y_abs > 0
        rey = ((sub["y_exp"] - sub["y_eval"]) / sub["y_exp"].where(ok, np.nan))
        rows.append({
            "Library":       LIB_LABELS[lib],
            "N":             int(len(sub)),
            "mean(r/y)":     float(rey.mean()),
            "median(r/y)":   float(rey.median()),
            "std(r/y)":      float(rey.std()),
            "mean(|r|/|y|)": float(rey.abs().mean()),
            "p10(r/y)":      float(rey.quantile(0.1)),
            "p90(r/y)":      float(rey.quantile(0.9)),
        })
    out = pd.DataFrame(rows)
    report.append(
        f"\n**Residual centering (r/y) per library, subset `{subset_key}`** "
        "(σ-free — measures how close each library's central value is to data):\n\n"
    )
    report.append(_df_to_md(out, float_format="%.4f") + "\n")
    return out


def coverage_table(
    df_subset: pd.DataFrame, libraries: List[str], subset_key: str,
    report: List[str],
) -> pd.DataFrame:
    """% of points within k·σ_total for k=1,2,3 (V1 textbook σ).

    σ_total² = σ_stat² + (σ_indep·y)² + (σ_dep·y)² + σ_eval_diag²
    Target N(0,1): 68.3% / 95.4% / 99.7%. Values below target = under-coverage
    (uncertainty too tight); values above = over-coverage.
    """
    rows = []
    for lib in libraries:
        sub = df_subset[df_subset["library"] == lib]
        if len(sub) == 0:
            continue
        y = sub["y_exp"].to_numpy()
        r = y - sub["y_eval"].to_numpy()
        sstat = np.maximum(sub["sigma_exp_stat"].to_numpy(),
                           0.01 * np.abs(y))
        u = sub["sigma_sys_indep_rel"].to_numpy() * y
        v = sub["sigma_sys_dep_rel"].to_numpy() * y
        sed = sub["sigma_eval_diag"].to_numpy()
        sigma = np.sqrt(np.maximum(sstat ** 2 + u ** 2 + v ** 2 + sed ** 2,
                                   1e-300))
        z = r / sigma
        rows.append({
            "Library":   LIB_LABELS[lib],
            "N":         int(len(sub)),
            "|z|<1 (%)": 100.0 * float(np.mean(np.abs(z) < 1)),
            "|z|<2 (%)": 100.0 * float(np.mean(np.abs(z) < 2)),
            "|z|<3 (%)": 100.0 * float(np.mean(np.abs(z) < 3)),
            "mean(z)":   float(np.mean(z)),
            "std(z)":    float(np.std(z, ddof=1)) if len(z) > 1 else float("nan"),
        })
    out = pd.DataFrame(rows)
    report.append(
        f"\n**Coverage under V1 textbook σ, subset `{subset_key}`** "
        "(target N(0,1): 68.3% / 95.4% / 99.7%; std(z)=1):\n\n"
    )
    report.append(_df_to_md(out, float_format="%.2f") + "\n")
    return out


def per_experiment_win_tally(
    per_exp_av: pd.DataFrame, libraries: List[str], subset_key: str,
    report: List[str],
) -> pd.DataFrame:
    """For each variant, count experiments where each lib has best / mid / worst χ²/N.

    Excludes experiments with NaN for any library. For 3 libraries with no ties,
    Σ_lib best = Σ_lib mid = Σ_lib worst = N_exp.
    """
    rows = []
    n_total_per_v: Dict[str, int] = {}
    for variant in VARIANTS:
        v_lo = _v_low(variant)
        piv = per_exp_av.pivot_table(
            index="experiment_id", columns="library",
            values=f"chi2_{v_lo}_per_N",
        )
        cols_present = [l for l in libraries if l in piv.columns]
        piv = piv[cols_present].dropna()
        n_total_per_v[variant] = int(len(piv))
    for lib in libraries:
        row = {"Library": LIB_LABELS[lib], "library_key": lib}
        for variant in VARIANTS:
            v_lo = _v_low(variant)
            piv = per_exp_av.pivot_table(
                index="experiment_id", columns="library",
                values=f"chi2_{v_lo}_per_N",
            )
            cols_present = [l for l in libraries if l in piv.columns]
            piv = piv[cols_present].dropna()
            if lib not in piv.columns:
                row[f"{variant} best"] = 0
                row[f"{variant} mid"]  = 0
                row[f"{variant} worst"] = 0
                continue
            others = [l for l in cols_present if l != lib]
            if not others:
                row[f"{variant} best"] = len(piv)
                row[f"{variant} mid"]  = 0
                row[f"{variant} worst"] = 0
                continue
            best  = int((piv[lib] <= piv[others].min(axis=1)).sum())
            worst = int((piv[lib] >= piv[others].max(axis=1)).sum())
            mid   = int(len(piv) - best - worst)
            row[f"{variant} best"]  = best
            row[f"{variant} mid"]   = mid
            row[f"{variant} worst"] = worst
        rows.append(row)
    out = pd.DataFrame(rows)
    # Was hardcoded to V1, which now reports 0 whenever V1 is not in VARIANTS.
    n_ref = n_total_per_v.get(PRIMARY_VARIANT, 0)
    report.append(
        f"\n**Per-experiment ranking, subset `{subset_key}`** "
        f"(count of experiments where each library has best/mid/worst χ²/N; "
        f"N_experiments under {PRIMARY_VARIANT} = {n_ref}):\n\n"
    )
    show_cols = ["Library"] + [
        f"{v} {kind}" for v in VARIANTS for kind in ("best", "mid", "worst")
    ]
    report.append(_df_to_md(out[show_cols], float_format="%.0f") + "\n")
    return out


def per_experiment_win_figure(
    per_exp_av: pd.DataFrame, libraries: List[str], variant: str,
    subset_key: str, subset_label: str,
    figures_dir: Path, report: List[str], fig_log: List[str],
) -> None:
    """Bar chart per experiment: log10(χ²/N(This_work) / min(χ²/N(others))).

    Bars below 0 = This_work strictly best of three. Bars above 0 = This_work
    not best. Title reports the win count. The most readable single figure
    summarising "where does This work win".
    """
    v_lo = _v_low(variant)
    piv = per_exp_av.pivot_table(
        index="experiment_id", columns="library",
        values=f"chi2_{v_lo}_per_N",
    )
    if "This_work" not in piv.columns:
        return
    others = [l for l in libraries if l != "This_work" and l in piv.columns]
    if not others:
        return
    piv = piv[["This_work"] + others].dropna()
    if len(piv) == 0:
        return
    other_min = piv[others].min(axis=1)
    ratio = piv["This_work"] / other_min
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    sorted_ratio = ratio.sort_values()
    colors = [LIB_COLORS["This_work"] if r < 1 else "#7f7f7f"
              for r in sorted_ratio.values]
    n_win = int((sorted_ratio < 1).sum())
    fig_w = max(10, 0.18 * len(sorted_ratio))
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    ax.bar(range(len(sorted_ratio)),
           np.log10(np.maximum(sorted_ratio.values, 1e-300)),
           color=colors, edgecolor="k", linewidth=0.3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Experiment (sorted by ratio)")
    ax.set_ylabel(
        r"$\log_{10}\!\left(\chi^2/N_{\mathrm{This\ work}} \,/\, "
        r"\min_{\mathrm{JEFF, JENDL}}\chi^2/N\right)$"
    )
    ax.set_title(
        f"{subset_label} — {variant}: This work strictly best on "
        f"{n_win}/{len(sorted_ratio)} experiments"
    )
    ax.set_xticks([])
    fig.tight_layout()
    save_and_close(fig, figures_dir,
                   f"per_experiment_win_{subset_key}_{variant}", fig_log)
    report.append(
        f"\n![per_experiment_win_{subset_key}_{variant}]"
        f"(figures/per_experiment_win_{subset_key}_{variant}.{FIGURE_FORMAT})\n"
    )


# ── PRIMARY_VARIANT detail figures ──────────────────────────────────────────


def run_subset_primary_detail(
    df_s: pd.DataFrame,
    per_exp_av: pd.DataFrame,
    z_primary: pd.Series,
    subset_key: str,
    subset_label: str,
    variant: str,
    libraries: List[str],
    figures_dir: Path,
    report: List[str],
    fig_log: List[str],
) -> pd.DataFrame:
    """Generate the 9-figure diagnostic set + tables for one (subset, variant)."""
    v_lo = _v_low(variant)
    suffix = f"{subset_key}_{variant}"

    meta = (
        df_s.groupby(["experiment_id", "library"], observed=True)
        .agg(author=("author", "first"), year=("year", "first"),
             E_min=("energy_mev", "min"), E_max=("energy_mev", "max"))
        .reset_index()
    )
    exp_df = per_exp_av.merge(meta, on=["experiment_id", "library"])
    exp_df["year"] = exp_df["year"].apply(lambda y: int(y) if pd.notna(y) else "")
    exp_df["chi2"] = exp_df[f"chi2_{v_lo}"]
    exp_df["chi2_per_N"] = exp_df[f"chi2_{v_lo}_per_N"]

    n_libs = len(libraries)

    report.append(
        f"\n### Detail figures — subset `{subset_key}`, variant {variant} "
        f"({VARIANT_LABELS[variant]})\n"
    )
    report.append(
        f"- Data points: **{len(df_s)}**, experiments: "
        f"**{df_s['experiment_id'].nunique()}**\n"
    )

    # 1. Global χ²/N summary + bar
    summary = exp_df.groupby("library", observed=True).agg(
        N=("N", "sum"), chi2=("chi2", "sum"),
    ).reset_index().rename(columns={"library": "library_key"})
    summary["chi2_per_N"] = np.where(
        summary["N"] > 0, summary["chi2"] / summary["N"], np.nan
    )
    summary["Library"] = summary["library_key"].map(LIB_LABELS).fillna(summary["library_key"])
    summary["__order"] = summary["library_key"].map({lib: i for i, lib in enumerate(libraries)})
    summary = summary.sort_values("__order").drop(columns="__order").reset_index(drop=True)

    report.append(f"\n**Global χ²/N (variant {variant})**:\n\n")
    report.append(_df_to_md(summary[["Library", "N", "chi2", "chi2_per_N"]],
                            float_format="%.3f") + "\n")

    fig_w = max(5, 2 + n_libs * 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    values = [summary.loc[summary["library_key"] == lib, "chi2_per_N"].values[0]
              for lib in libraries]
    colors = [LIB_COLORS[lib] for lib in libraries]
    bars = ax.bar([LIB_LABELS[lib] for lib in libraries], values,
                  color=colors, edgecolor="k", linewidth=0.5)
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label=r"$\chi^2/N = 1$")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2f}", ha="center", va="bottom", fontsize=12)
    ax.set_ylabel(r"$\chi^2 / N$")
    ax.set_title(f"{subset_label} — {VARIANT_LABELS[variant]}")
    ax.legend()
    fig.tight_layout()
    save_and_close(fig, figures_dir, f"chi2_global_bar_{suffix}", fig_log)

    # 2. Per-experiment χ²/N pivot (printed inline; full CSV is written per subset)
    pivot = exp_df.pivot_table(
        index=["experiment_id", "author", "year", "E_min", "E_max"],
        columns="library", values="chi2_per_N",
    )
    libs_in_pivot = [l for l in libraries if l in pivot.columns]
    if libs_in_pivot:
        pivot["avg"] = pivot[libs_in_pivot].mean(axis=1)
        pivot = pivot.sort_values("avg", ascending=False).drop(columns="avg")
    n_per_exp = exp_df.groupby("experiment_id")["N"].first()
    pivot = pivot.join(n_per_exp, on="experiment_id")

    report.append(
        f"\n**Per-experiment χ²/N (variant {variant}, top 20)** — full table in "
        f"`per_experiment/per_experiment_{subset_key}.csv` (all variants):\n\n"
    )
    report.append("```\n" + _df_to_text(pivot.reset_index().head(20),
                                        float_format="%.2f") + "\n```\n")
    if len(pivot) > 20:
        report.append(f"_…{len(pivot) - 20} more experiments in the CSV._\n")

    # 3. Per-experiment χ²/N histogram
    fig, axes = plt.subplots(1, n_libs, figsize=(5 * n_libs, 4.5),
                             sharey=True, squeeze=False)
    for ax, lib in zip(axes.flat, libraries):
        vals = exp_df.loc[exp_df["library"] == lib, "chi2_per_N"].dropna()
        if len(vals) == 0:
            ax.set_title(LIB_LABELS[lib]); continue
        median_val = float(vals.median())
        ax.hist(vals, bins=max(5, len(vals) // 2), color=LIB_COLORS[lib],
                edgecolor="k", alpha=0.7)
        ax.axvline(1.0, color="k", ls="--", lw=1, label=r"$\chi^2/N=1$")
        ax.axvline(median_val, color="gray", ls=":", lw=1.2,
                   label=f"median = {median_val:.2f}")
        ax.set_xlabel(r"$\chi^2/N$"); ax.set_title(LIB_LABELS[lib])
        ax.legend(fontsize=11)
    axes.flat[0].set_ylabel("Number of experiments")
    fig.suptitle(f"{subset_label} — {variant}", fontsize=14, y=1.02)
    fig.tight_layout()
    save_and_close(fig, figures_dir, f"chi2_per_experiment_histogram_{suffix}", fig_log)

    # 4. χ² contribution per experiment (%)
    fig, axes = plt.subplots(1, n_libs, figsize=(5 * n_libs + 1, 7),
                             sharey=False, squeeze=False)
    for ax, lib in zip(axes.flat, libraries):
        sub = exp_df[exp_df["library"] == lib].copy()
        total_chi2 = float(sub["chi2"].sum())
        if total_chi2 <= 0 or len(sub) == 0:
            ax.set_title(LIB_LABELS[lib]); continue
        sub["pct"] = 100.0 * sub["chi2"] / total_chi2
        above = sub[sub["pct"] >= CONTRIB_THRESHOLD_PCT].sort_values("pct", ascending=True)
        below = sub[sub["pct"] < CONTRIB_THRESHOLD_PCT]
        bar_labels = [f"{r['author']} ({r['year']})" for _, r in above.iterrows()]
        if len(below) > 0:
            bar_labels = [f"Other ({len(below)} exp.)"] + bar_labels
            vals = [float(below["pct"].sum())] + above["pct"].tolist()
            cols = ["#cccccc"] + [LIB_COLORS[lib]] * len(above)
        else:
            vals = above["pct"].tolist()
            cols = [LIB_COLORS[lib]] * len(above)
        ax.barh(bar_labels, vals, color=cols, edgecolor="k", linewidth=0.5)
        ax.set_xlabel(r"Contribution to total $\chi^2$ (%)")
        ax.set_title(LIB_LABELS[lib])
    fig.suptitle(f"{subset_label} — {variant}", fontsize=14, y=1.02)
    fig.tight_layout()
    save_and_close(fig, figures_dir, f"chi2_contribution_{suffix}", fig_log)

    # 5. Cumulative χ² waterfall
    fig, ax = plt.subplots(figsize=(10, 5))
    for lib in libraries:
        sub = (exp_df[exp_df["library"] == lib]
               .sort_values("chi2", ascending=False).reset_index(drop=True))
        total = float(sub["chi2"].sum())
        if total <= 0 or len(sub) == 0:
            continue
        sub["cum_pct"] = 100.0 * sub["chi2"].cumsum() / total
        ax.plot(range(1, len(sub) + 1), sub["cum_pct"], "o-",
                color=LIB_COLORS[lib], label=LIB_LABELS[lib], markersize=5)
    ax.axhline(80, color="#333333", ls="--", lw=2.0, alpha=0.9, label="80%")
    ax.axhline(95, color="#333333", ls=":", lw=2.0, alpha=0.9, label="95%")
    ax.set_xlabel(r"Experiment rank (by $\chi^2$ contribution)")
    ax.set_ylabel(r"Cumulative $\chi^2$ (%)")
    ax.set_title(f"{subset_label} — {variant}")
    ax.legend()
    fig.tight_layout()
    save_and_close(fig, figures_dir, f"chi2_waterfall_{suffix}", fig_log)

    # 6. χ²/N vs publication year
    fig, ax = plt.subplots(figsize=(10, 5))
    for lib in libraries:
        sub = exp_df[exp_df["library"] == lib].copy()
        sub["year_num"] = pd.to_numeric(sub["year"], errors="coerce")
        sub = sub.dropna(subset=["year_num"])
        offset = LIB_YEAR_OFFSETS.get(lib, 0.0)
        ax.scatter(sub["year_num"] + offset, sub["chi2_per_N"],
                   c=LIB_COLORS[lib], marker=LIB_MARKERS.get(lib, "o"),
                   s=50, alpha=0.7, edgecolors="k", linewidths=0.3,
                   label=LIB_LABELS[lib], zorder=3)
    ax.axvspan(1982, 2016, alpha=0.10, color="gray", zorder=0)
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label=r"$\chi^2/N = 1$")
    ax.set_xlabel("Publication year")
    ax.set_ylabel(r"$\chi^2 / N$")
    ax.set_yscale("log")
    ax.set_title(f"{subset_label} — {variant}")
    ax.legend()
    fig.tight_layout()
    save_and_close(fig, figures_dir, f"chi2_vs_year_{suffix}", fig_log)

    # 7. Whitened-residual statistics
    df_z = df_s.assign(z=z_primary.values)
    res_rows = []
    for lib in libraries:
        zr = df_z.loc[df_z["library"] == lib, "z"].to_numpy()
        if zr.size == 0:
            continue
        res_rows.append({
            "Library": LIB_LABELS[lib],
            "mean": float(np.mean(zr)),
            "std": float(np.std(zr, ddof=1)) if zr.size > 1 else float("nan"),
            "skewness": float(stats.skew(zr)),
            "kurtosis": float(stats.kurtosis(zr)),
            "|z|>2 (%)": 100.0 * float(np.mean(np.abs(zr) > 2)),
            "|z|>3 (%)": 100.0 * float(np.mean(np.abs(zr) > 3)),
        })
    res_df = pd.DataFrame(res_rows)
    report.append(
        f"\n**Whitened-residual statistics (variant {variant})** "
        "(N(0,1) target: mean=0, std=1, skew=0, kurt=0, "
        "|z|>2: 4.6%, |z|>3: 0.3%):\n\n"
    )
    report.append(_df_to_md(res_df, float_format="%.3f") + "\n")

    # 8. Whitened-residual histograms
    x_grid = np.linspace(-6, 6, 300)
    fig, axes = plt.subplots(1, n_libs, figsize=(5 * n_libs, 4.5),
                             sharey=True, squeeze=False)
    for ax, lib in zip(axes.flat, libraries):
        zr = df_z.loc[df_z["library"] == lib, "z"].to_numpy()
        if zr.size == 0:
            ax.set_title(LIB_LABELS[lib]); continue
        mu_fit, sigma_fit = stats.norm.fit(zr)
        ax.hist(zr, bins=80, density=True, color=LIB_COLORS[lib],
                edgecolor="k", linewidth=0.3, alpha=0.7, label=LIB_LABELS[lib])
        ax.plot(x_grid, stats.norm.pdf(x_grid, 0, 1), "k--", lw=2.0,
                label=r"$\mathcal{N}(0,1)$")
        ax.plot(x_grid, stats.norm.pdf(x_grid, mu_fit, sigma_fit), "r-", lw=1.5,
                label=f"fit $\\mathcal{{N}}$({mu_fit:.2f},{sigma_fit:.2f})")
        ax.set_xlabel("Whitened residual $z$", fontsize=15)
        ax.set_xlim(-6, 6)
        ax.legend(fontsize=12)
    axes.flat[0].set_ylabel("Density", fontsize=15)
    fig.suptitle(f"{subset_label} — {variant}", fontsize=14, y=1.02)
    fig.tight_layout()
    save_and_close(fig, figures_dir, f"whitened_residual_histograms_{suffix}", fig_log)

    # 9. Whitened residual vs energy scatter
    fig, axes = plt.subplots(1, n_libs, figsize=(5 * n_libs, 4.5),
                             sharey=True, squeeze=False)
    for ax, lib in zip(axes.flat, libraries):
        sub = df_z[df_z["library"] == lib]
        ax.scatter(sub["energy_mev"], sub["z"], s=3, alpha=0.2,
                   color=LIB_COLORS[lib], rasterized=True)
        ax.axhline(0, color="k", lw=0.8)
        ax.axhline(2, color="gray", ls="--", lw=0.6, alpha=0.7)
        ax.axhline(-2, color="gray", ls="--", lw=0.6, alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("Energy (MeV)")
        ax.set_title(LIB_LABELS[lib])
    axes.flat[0].set_ylabel("Whitened residual $z$")
    fig.suptitle(f"{subset_label} — {variant}", fontsize=14, y=1.02)
    fig.tight_layout()
    save_and_close(fig, figures_dir, f"whitened_residual_vs_energy_{suffix}", fig_log)

    # Inline figure references in the markdown report.
    report.append("\nFigures:\n\n")
    for stem in (
        f"chi2_global_bar_{suffix}",
        f"chi2_per_experiment_histogram_{suffix}",
        f"chi2_contribution_{suffix}",
        f"chi2_waterfall_{suffix}",
        f"chi2_vs_year_{suffix}",
        f"whitened_residual_histograms_{suffix}",
        f"whitened_residual_vs_energy_{suffix}",
    ):
        report.append(f"![{stem}](figures/{stem}.{FIGURE_FORMAT})\n")

    return summary


# ── Per-subentry breakdown (all variants) ───────────────────────────────────


def per_subentry_table_all_variants(
    df_in: pd.DataFrame, eval_cov: Optional[Dict],
    libraries: List[str], report: List[str],
    systematic_block_col: Optional[str] = None,
) -> pd.DataFrame:
    """χ²/N per (subentry × variant), all four variants in one table.

    Rows are (Kinney_1976, Smith_1980, OTHER, BOTH/ALL) × (V1, V2, V3, V4);
    columns are libraries. Each slice goes through chi2_per_experiment_variants
    once with the full eval_cov, so V1/V2/V3/V4 are consistent within a row.
    """
    subentry_labels = ("Kinney_1976", "Smith_1980", "OTHER", "BOTH/ALL")
    rows = []
    for ks_label in subentry_labels:
        sub_df = df_in if ks_label == "BOTH/ALL" else df_in[df_in["ks_subentry"] == ks_label]
        if len(sub_df) == 0:
            continue
        per_exp = chi2_metrics.chi2_per_experiment_variants(
            sub_df, eval_cov, systematic_block_col=systematic_block_col,
        )
        n_first = int(per_exp[per_exp["library"] == libraries[0]]["N"].sum()) if libraries else 0
        for variant in VARIANTS:
            v_lo = _v_low(variant)
            row = {"subentry": ks_label, "variant": variant, "N": n_first}
            for lib in libraries:
                sl = per_exp[per_exp["library"] == lib]
                n = int(sl["N"].sum())
                chi2 = float(sl[f"chi2_{v_lo}"].sum())
                row[LIB_LABELS[lib]] = chi2 / n if n > 0 else float("nan")
            rows.append(row)
    out = pd.DataFrame(rows)
    cols = ["subentry", "variant", "N"] + [LIB_LABELS[l] for l in libraries]
    out = out[cols]

    report.append("\n**χ²/N per (subentry × variant):**\n\n")
    report.append(_df_to_md(out, float_format="%.3f") + "\n")
    return out


# ── Combined comparison across all 12 cells ─────────────────────────────────


def combined_comparison_figure(
    all_variants_summaries: Dict[str, pd.DataFrame], libraries: List[str],
    figures_dir: Path, report: List[str], fig_log: List[str],
) -> None:
    """Grouped bar chart of χ²/N for every (subset × variant) per library.

    Subset alpha is auto-spaced from 1.0 (first) down to 0.3 (last) so it
    extends gracefully when SUBSETS grows beyond the original three.
    """
    n_libs = len(libraries)
    subsets_present = [(k, l) for k, l in SUBSETS if k in all_variants_summaries]
    n_cells = len(subsets_present) * len(VARIANTS)
    if n_cells == 0:
        return
    x = np.arange(n_libs)
    width = 0.9 / n_cells

    hatch_for_variant = {"V1": "", "V2": "///", "V3": "...", "V4": "xxx"}
    n_sub = max(len(subsets_present), 1)
    alpha_for_subset = {
        k: 1.0 - 0.7 * (i / max(n_sub - 1, 1))
        for i, (k, _) in enumerate(subsets_present)
    }

    fig, ax = plt.subplots(figsize=(max(14, 1.2 * n_cells), 6))
    cell_idx = 0
    for subset_key, _ in subsets_present:
        summary = all_variants_summaries[subset_key]
        for variant in VARIANTS:
            col = f"chi2_{_v_low(variant)}_per_N"
            vals = [
                summary.loc[summary["library_key"] == lib, col].values[0]
                if lib in summary["library_key"].values else np.nan
                for lib in libraries
            ]
            label = f"{subset_key} · {variant}"
            bars = ax.bar(
                x + (cell_idx - (n_cells - 1) / 2) * width, vals, width,
                color=[LIB_COLORS[lib] for lib in libraries],
                edgecolor="k", linewidth=0.4,
                alpha=alpha_for_subset[subset_key],
                hatch=hatch_for_variant[variant], label=label,
            )
            for bar, val in zip(bars, vals):
                if np.isfinite(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{val:.1f}", ha="center", va="bottom", fontsize=5,
                            rotation=90)
            cell_idx += 1

    ax.set_xticks(x)
    ax.set_xticklabels([LIB_LABELS[lib] for lib in libraries])
    ax.set_ylabel(r"$\chi^2 / N$")
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_title("Combined comparison — all subsets × variants")
    ax.legend(title="Subset · Variant", fontsize=6, ncol=4,
              loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    save_and_close(fig, figures_dir, "chi2_combined_comparison", fig_log)
    report.append("\n![chi2_combined_comparison]"
                  f"(figures/chi2_combined_comparison.{FIGURE_FORMAT})\n")


def primary_variant_summary_figure(
    all_variants_summaries: Dict[str, pd.DataFrame], libraries: List[str],
    variant: str, figures_dir: Path, report: List[str], fig_log: List[str],
) -> None:
    """One grouped bar chart of PRIMARY_VARIANT χ²/N per library across subsets.

    The clean single-figure summary for the thesis chapter: x = subset,
    grouped bars = library, height = χ²/N. Easier to read than the 24-cell
    combined figure.
    """
    subsets_present = [(k, l) for k, l in SUBSETS if k in all_variants_summaries]
    if not subsets_present:
        return
    n_libs = len(libraries)
    n_sub  = len(subsets_present)
    width  = 0.8 / n_libs
    x = np.arange(n_sub)
    col = f"chi2_{_v_low(variant)}_per_N"
    fig, ax = plt.subplots(figsize=(max(10, 2.0 * n_sub), 5.5))
    for j, lib in enumerate(libraries):
        vals = [
            float(all_variants_summaries[k].loc[
                all_variants_summaries[k]["library_key"] == lib, col
            ].values[0])
            if lib in all_variants_summaries[k]["library_key"].values
            else np.nan
            for k, _ in subsets_present
        ]
        offsets = (j - (n_libs - 1) / 2) * width
        bars = ax.bar(x + offsets, vals, width,
                      color=LIB_COLORS[lib], edgecolor="k", linewidth=0.5,
                      label=LIB_LABELS[lib])
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([k for k, _ in subsets_present], rotation=20, ha="right")
    ax.set_ylabel(rf"$\chi^2 / N$ ({variant})")
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_title(
        f"χ²/N per (subset × library) — {variant} ({VARIANT_LABELS[variant]})"
    )
    ax.legend()
    fig.tight_layout()
    save_and_close(fig, figures_dir,
                   f"primary_summary_{variant}", fig_log)
    report.append(
        f"\n![primary_summary_{variant}]"
        f"(figures/primary_summary_{variant}.{FIGURE_FORMAT})\n"
    )


# ── Top-level driver ─────────────────────────────────────────────────────────


def run_methodology(methodology: str, paths: RunPaths) -> Dict:
    print(f"\n=== {methodology}: {paths.title} ===")
    print(f"  parquet     : {paths.parquet}")
    print(f"  eval_cov    : {paths.eval_cov}")
    print(f"  output dir  : {paths.run_dir}")
    print(f"  primary var : {PRIMARY_VARIANT}")

    if not paths.parquet.exists():
        raise FileNotFoundError(f"Parquet not found: {paths.parquet}")
    if not paths.eval_cov.exists():
        raise FileNotFoundError(f"Eval-cov sidecar not found: {paths.eval_cov}")
    if PRIMARY_VARIANT not in VARIANTS:
        raise SystemExit(
            f"PRIMARY_VARIANT={PRIMARY_VARIANT!r} not in {VARIANTS}"
        )

    eval_cov = chi2_metrics.load_eval_cov(str(paths.eval_cov))
    print(f"  loaded {len(eval_cov)} per-(library, experiment) Σ_eval blocks")

    df_raw = pd.read_parquet(paths.parquet)
    df = df_raw[(df_raw["energy_mev"] >= E_MIN_MEV)
                & (df_raw["energy_mev"] <= E_MAX_MEV)].copy()
    bad_mask = (df["sigma_exp"] <= 0) | ~np.isfinite(df["sigma_exp"])
    n_bad = int(bad_mask.sum())
    if n_bad:
        df = df[~bad_mask].copy()
    df["sigma_eval_diag"] = df["sigma_eval_diag"].fillna(0.0)
    df["is_KS"] = df["experiment_id"].isin(KINNEY_SMITH_IDS)
    df["is_Cierjacks"] = df["experiment_id"].isin(CIERJACKS_IDS)

    libs_in_data = sorted(df["library"].unique())
    libraries = [lib for lib in libs_in_data if lib in LIB_LABELS]

    is_KS = df["is_KS"]
    is_C  = df["is_Cierjacks"]
    subsets_df: Dict[str, pd.DataFrame] = {
        "all":                df.copy(),
        "no_Cierjacks":       df[~is_C].copy(),
        "no_KS":              df[~is_KS].copy(),
        "no_KS_no_Cierjacks": df[~is_KS & ~is_C].copy(),
        "only_KS":            df[is_KS].copy(),
        "only_Cierjacks":     df[is_C].copy(),
    }
    # Guard against SUBSETS containing keys not built above.
    missing = [k for k, _ in SUBSETS if k not in subsets_df]
    if missing:
        raise SystemExit(
            f"SUBSETS contains keys with no builder: {missing}. "
            f"Available builders: {sorted(subsets_df)}"
        )

    report: List[str] = []
    fig_log: List[str] = []
    report.append(f"# {paths.title}\n")
    report.append(
        f"\n_Run ID: `{RUN_ID}` — primary variant for detail figures: "
        f"**{PRIMARY_VARIANT}** ({VARIANT_LABELS[PRIMARY_VARIANT]})_  \n"
        f"_Parquet: `{paths.parquet}`_  \n"
        f"_Eval-cov: `{paths.eval_cov}`_  \n"
        f"_EXFOR systematic correlation: "
        f"{('per-' + paths.systematic_block_col) if paths.systematic_block_col else 'experiment-wide (global)'} "
        f"(V2/V3/V4)_  \n"
        f"_Energy window: [{E_MIN_MEV}, {E_MAX_MEV}] MeV; "
        f"L_max truncation: see precompute script._\n"
    )
    report.append(
        f"\n## Dataset\n\n"
        f"- Rows after filter: **{len(df)}** "
        f"(dropped {n_bad} with σ_exp ≤ 0 or non-finite)\n"
        f"- Experiments: **{df['experiment_id'].nunique()}**\n"
        f"- Libraries: {', '.join(LIB_LABELS[l] for l in libraries)}\n\n"
        f"Subsets:\n\n"
    )
    for subset_key, subset_label in SUBSETS:
        sd = subsets_df[subset_key]
        report.append(
            f"- `{subset_key}` ({subset_label}): {len(sd)} rows, "
            f"{sd['experiment_id'].nunique()} experiments\n"
        )
    report.append(
        f"\nChi² variants computed for every subset:\n\n"
    )
    for v in VARIANTS:
        report.append(f"- **{v}** — {VARIANT_LABELS[v]}\n")

    # Reserve a placeholder for the headline summary; populated after all
    # subsets are computed, then spliced in at this position.
    headline_slot = len(report)
    report.append("")  # placeholder

    # Per-subset: compute all 4 variants once, write decision table + CSV,
    # diagnostic tables (residual, coverage, win tally), then the detail set
    # for PRIMARY_VARIANT.
    all_variants_summaries: Dict[str, pd.DataFrame] = {}
    primary_summaries: Dict[str, pd.DataFrame] = {}
    mean_residual_per_subset: Dict[str, pd.DataFrame] = {}
    coverage_per_subset: Dict[str, pd.DataFrame] = {}
    win_tally_per_subset: Dict[str, pd.DataFrame] = {}
    for subset_key, subset_label in SUBSETS:
        df_subset = subsets_df[subset_key]
        report.append(f"\n## Subset: `{subset_key}` — {subset_label}\n")
        report.append(
            f"- {len(df_subset)} rows, "
            f"{df_subset['experiment_id'].nunique()} experiments\n"
        )
        if len(df_subset) == 0:
            report.append("\n_(subset is empty — skipped)_\n")
            continue

        per_exp_av = chi2_metrics.chi2_per_experiment_variants(
            df_subset, eval_cov, systematic_block_col=paths.systematic_block_col,
        )
        decision_summary = aggregate_all_variants_by_library(per_exp_av, libraries)
        all_variants_summaries[subset_key] = decision_summary
        write_decision_table(decision_summary, subset_key, subset_label, report)
        write_subset_csv(per_exp_av, df_subset, libraries, subset_key,
                         paths.per_exp_csv_dir, fig_log)

        mean_residual_per_subset[subset_key] = mean_residual_table(
            df_subset, libraries, subset_key, report,
        )
        coverage_per_subset[subset_key] = coverage_table(
            df_subset, libraries, subset_key, report,
        )
        win_tally_per_subset[subset_key] = per_experiment_win_tally(
            per_exp_av, libraries, subset_key, report,
        )

        z_primary = chi2_metrics.whitened_residuals_per_experiment_variant(
            df_subset, eval_cov, variant=PRIMARY_VARIANT,
            systematic_block_col=paths.systematic_block_col,
        )
        primary_summaries[subset_key] = run_subset_primary_detail(
            df_s=df_subset,
            per_exp_av=per_exp_av,
            z_primary=z_primary,
            subset_key=subset_key,
            subset_label=subset_label,
            variant=PRIMARY_VARIANT,
            libraries=libraries,
            figures_dir=paths.figures_dir,
            report=report,
            fig_log=fig_log,
        )

        # Win figure for PRIMARY_VARIANT — single most informative plot.
        per_experiment_win_figure(
            per_exp_av, libraries, PRIMARY_VARIANT,
            subset_key, subset_label,
            paths.figures_dir, report, fig_log,
        )

    # ── Build & splice in the headline summary ───────────────────────────────
    headline: List[str] = []
    headline.append(
        f"\n## Headline summary — primary variant **{PRIMARY_VARIANT}** "
        f"({VARIANT_LABELS[PRIMARY_VARIANT]})\n"
    )
    headline.append(
        f"\n**Recommended thesis headline subset: `no_Cierjacks`** — Cierjacks "
        f"1978 contributes ~61% of all points and is a known difficult dataset "
        f"shared by all three modern Fe-56 evaluations; reported separately in "
        f"the `only_Cierjacks` and `Cierjacks 1978` per-experiment tables.\n"
    )
    headline.append(
        f"\n**χ²/N per (library × subset) under {PRIMARY_VARIANT}:**\n\n"
    )
    head_rows = []
    for subset_key, subset_label in SUBSETS:
        if subset_key not in all_variants_summaries:
            continue
        s = all_variants_summaries[subset_key]
        head_rows.append({
            "Subset":    subset_key,
            "N_points":  int(s["N"].iloc[0]) if len(s) else 0,
            **{
                str(s.iloc[i]["Library"]):
                    float(s.iloc[i][f"chi2_{_v_low(PRIMARY_VARIANT)}_per_N"])
                for i in range(len(s))
            },
        })
    head_df = pd.DataFrame(head_rows)
    headline.append(_df_to_md(head_df, float_format="%.3f") + "\n")

    headline.append(
        f"\n**Per-experiment ranking under {PRIMARY_VARIANT} "
        f"(count of experiments where each library has best/mid/worst χ²/N):**\n\n"
    )
    tally_rows = []
    for subset_key, _ in SUBSETS:
        if subset_key not in win_tally_per_subset:
            continue
        wt = win_tally_per_subset[subset_key].set_index("library_key")
        row: Dict = {"Subset": subset_key}
        n_total = None
        for lib in libraries:
            if lib in wt.index:
                b = int(wt.loc[lib, f"{PRIMARY_VARIANT} best"])
                m = int(wt.loc[lib, f"{PRIMARY_VARIANT} mid"])
                w = int(wt.loc[lib, f"{PRIMARY_VARIANT} worst"])
                row[f"{LIB_LABELS[lib]} best/mid/worst"] = f"{b}/{m}/{w}"
                if n_total is None:
                    n_total = b + m + w
        row["N_experiments"] = n_total if n_total is not None else 0
        tally_rows.append(row)
    tally_df = pd.DataFrame(tally_rows)
    cols = ["Subset", "N_experiments"] + [
        f"{LIB_LABELS[lib]} best/mid/worst" for lib in libraries
        if f"{LIB_LABELS[lib]} best/mid/worst" in tally_df.columns
    ]
    headline.append(_df_to_md(tally_df[cols], float_format="%.0f") + "\n")
    headline.append(
        "\n_Best = strictly smallest χ²/N of the three libraries on that experiment. "
        "Sum of `best` across libraries equals N_experiments only when there are no ties._\n"
    )
    report[headline_slot] = "".join(headline)

    # Per-subentry split — all four variants in one table.
    report.append(
        "\n## Per-subentry breakdown (Kinney 1976 / Smith 1980 / OTHER) — all variants\n"
    )
    per_subentry_table_all_variants(
        df, eval_cov, libraries, report,
        systematic_block_col=paths.systematic_block_col,
    )

    # Combined comparison: every (subset × variant) cell per library.
    report.append("\n## Combined comparison — all subsets × variants\n")
    combined_rows = []
    for subset_key, _ in SUBSETS:
        if subset_key not in all_variants_summaries:
            continue
        s = all_variants_summaries[subset_key]
        for _, row in s.iterrows():
            entry = {"Subset": subset_key, "Library": row["Library"],
                     "library_key": row["library_key"], "N": int(row["N"])}
            for v in VARIANTS:
                entry[f"{v} χ²/N"] = float(row[f"chi2_{_v_low(v)}_per_N"])
            combined_rows.append(entry)
    combined_df = pd.DataFrame(combined_rows)
    report.append("\n**All-variants χ²/N table:**\n\n")
    show_cols = ["Subset", "Library", "N"] + [f"{v} χ²/N" for v in VARIANTS]
    report.append(_df_to_md(combined_df[show_cols], float_format="%.3f") + "\n")

    # Best library per (subset × variant)
    report.append("\n**Best library per (subset × variant)** (lowest χ²/N):\n\n")
    best_records: Dict[str, Dict[str, Dict]] = {}
    for subset_key, _ in SUBSETS:
        if subset_key not in all_variants_summaries:
            continue
        best_records[subset_key] = {}
        sub_rows = combined_df[combined_df["Subset"] == subset_key]
        for v in VARIANTS:
            col = f"{v} χ²/N"
            if sub_rows[col].dropna().empty:
                continue
            idx = sub_rows[col].astype(float).idxmin()
            best_lib = sub_rows.loc[idx, "Library"]
            best_val = float(sub_rows.loc[idx, col])
            report.append(
                f"- `{subset_key}` · {v} → **{best_lib}** ({best_val:.3f})\n"
            )
            best_records[subset_key][v] = {"library": best_lib,
                                           "chi2_per_N": best_val}

    primary_variant_summary_figure(
        all_variants_summaries, libraries, PRIMARY_VARIANT,
        paths.figures_dir, report, fig_log,
    )
    combined_comparison_figure(
        all_variants_summaries, libraries, paths.figures_dir, report, fig_log
    )

    # Machine-readable summary
    summary_json = {
        "methodology": methodology,
        "title": paths.title,
        "run_id": RUN_ID,
        "primary_variant": PRIMARY_VARIANT,
        "variants": list(VARIANTS),
        "systematic_block_col": paths.systematic_block_col,
        "parquet": str(paths.parquet),
        "eval_cov": str(paths.eval_cov),
        "e_min_mev": E_MIN_MEV, "e_max_mev": E_MAX_MEV,
        "n_rows": int(len(df)), "n_experiments": int(df["experiment_id"].nunique()),
        "libraries": libraries,
        "kinney_smith_ids": list(KINNEY_SMITH_IDS),
        "cierjacks_ids":    list(CIERJACKS_IDS),
        "per_subset_all_variants": {
            subset_key: all_variants_summaries[subset_key].to_dict(orient="records")
            for subset_key, _ in SUBSETS
            if subset_key in all_variants_summaries
        },
        "primary_per_subset": {
            subset_key: primary_summaries[subset_key].to_dict(orient="records")
            for subset_key, _ in SUBSETS
            if subset_key in primary_summaries
        },
        "mean_residual_per_subset": {
            subset_key: mean_residual_per_subset[subset_key].to_dict(orient="records")
            for subset_key, _ in SUBSETS
            if subset_key in mean_residual_per_subset
        },
        "coverage_per_subset": {
            subset_key: coverage_per_subset[subset_key].to_dict(orient="records")
            for subset_key, _ in SUBSETS
            if subset_key in coverage_per_subset
        },
        "win_tally_per_subset": {
            subset_key: win_tally_per_subset[subset_key].to_dict(orient="records")
            for subset_key, _ in SUBSETS
            if subset_key in win_tally_per_subset
        },
        "best_per_subset_variant": best_records,
    }
    paths.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with paths.summary_json.open("w") as fh:
        json.dump(summary_json, fh, indent=2, default=float)

    paths.report_path.parent.mkdir(parents=True, exist_ok=True)
    paths.report_path.write_text("".join(report))

    n_png = len(list(paths.figures_dir.glob(f"*.{FIGURE_FORMAT}")))
    n_csv = len(list(paths.per_exp_csv_dir.glob("*.csv")))
    print(f"  wrote report:  {paths.report_path}")
    print(f"  wrote summary: {paths.summary_json}")
    print(f"  figures:       {paths.figures_dir} ({n_png} PNG)")
    print(f"  per-exp CSVs:  {paths.per_exp_csv_dir} ({n_csv} files)")
    return summary_json


def main() -> int:
    if not METHODOLOGIES_TO_RUN:
        raise SystemExit("METHODOLOGIES_TO_RUN is empty; nothing to do.")
    unknown = [m for m in METHODOLOGIES_TO_RUN if m not in PATHS]
    if unknown:
        raise SystemExit(
            f"Unknown methodology entry in METHODOLOGIES_TO_RUN: {unknown}. "
            f"Allowed keys: {sorted(PATHS)}"
        )

    configure_matplotlib()
    for m in METHODOLOGIES_TO_RUN:
        run_methodology(m, build_paths(m))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
