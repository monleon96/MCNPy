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
  - `no_Cierjacks`         — Cierjacks 1978 excluded (aporta ~61% de los puntos
                             y es un conjunto dificil). NO es el subconjunto
                             titular: no hay ninguno.
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
    # Run 87 — the SAME run-86 ENDF, scored with the measured MF33↔MF34 cross
    # block switched on. No re-evaluation: the sidecars run 86 already wrote are
    # fed to `build_mf33_mf34_cross_block`, which every run from 82 to 86 fed
    # None. So `predictive_87` vs `predictive_86` isolates the cross term and
    # nothing else — the MF4 centrals, the MF33 and the MF34 are the same file.
    #
    # V2 MUST NOT MOVE: it excludes Σ_eval entirely. JEFF and JENDL must not move
    # either — they keep a zero cross block. Anything else is a knob that moved
    # unintentionally. V1/V3 (diagonal Σ_eval) and V4 (dense) are the answer.
    #
    # What is being tested: run 86 measured Cov(c₀, a_ℓ) in PARAMETER space —
    # median |ρ| = 0.594 at a₁, but sign-alternating in energy. Σ_cross carries
    # P_ℓ(μ) on top of that, which alternates in ANGLE, so whether the net per
    # datapoint inflates or deflates σ_eval cannot be read off ρ. This run is how
    # that question gets an answer.
    #
    # ⚠ Read the [PSD] lines in the precompute log before trusting any number
    # here. Σ^MF33 and Σ^MF34 come from the shipped (collapsed, guarded) file
    # while Σ^cross comes from the raw MC, so the sum is not PSD by construction.
    #
    # 300G: the cross block is materialised densely per (experiment, order).
    "predictive_87": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_87.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — run-86 evaluation + measured MF33↔MF34 cross block, declared σ_E in the fold",
        "systematic_block_col": None,
    },
    # Damped cross block. Run 87 at full strength put 1005 of 46819 This_work
    # points (2.15 %) at a NEGATIVE Σ_eval diagonal — Σ^MF33 and Σ^MF34 went
    # through the multigroup collapse and the near-zero guard while Σ^cross came
    # from the raw MC, so the per-point Cauchy–Schwarz bound is violable and it
    # is violated. Those points lose their evaluated uncertainty in V1/V3 and
    # make V4 indefinite, so run 87's headline numbers are a diagnostic, not a
    # result.
    #
    # Σ_eval(s) = Σ^MF33 + Σ^MF34 + s·Σ^cross is EXACTLY LINEAR in s, so damping
    # brackets the largest self-consistent cross term instead of guessing one.
    # Two points, and the pair is a self-check as well as a scan: the s=0.25
    # diagonal must equal the s=0.5 one linearly interpolated, or something other
    # than the cross block moved.
    #
    # These runs carry `sigma_eval_var_diag` (the SIGNED, unclipped diagonal),
    # which is what makes the exact PSD-safe scale computable offline from a
    # 5 MB parquet rather than by further scanning.
    "predictive_87_s050": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_87_s050.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — run-86 evaluation + MF33↔MF34 cross block damped to 0.50, declared σ_E in the fold",
        "systematic_block_col": None,
    },
    "predictive_87_s025": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_87_s025.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — run-86 evaluation + MF33↔MF34 cross block damped to 0.25, declared σ_E in the fold",
        "systematic_block_col": None,
    },
    # Run 88 — the COMPLETE cross block (roadmap §10.1.6). The first evaluation
    # to ship Cov(c₀(E_i), a_ℓ(E_j)) with its cross-ENERGY structure instead of
    # the within-bin diagonal.
    #
    # WHY THIS RUN EXISTS. Runs 87/87_s050/87_s025 all failed to produce a χ² at
    # all: Σ_V4 was not positive definite and `chi2_metrics._solve` died on
    # Cholesky. §10.1.4 measured the PSD-safe damping scale at ≤ 0.052 against a
    # diagonal ceiling of 0.276, i.e. a token rather than a measurement, and
    # §10.1.5 then proved damping and representation work are both dead ends:
    # with Cx = 0 the joint parameter-space covariance is PSD at −1e-11, and
    # enforcing Cauchy–Schwarz — which is exactly what a consistent collapse
    # delivers — moves λ_min by 12 % in the best energy window and ~0.01 % in the
    # rest. The defect was never the representation. It was that we shipped a
    # PARTIAL Level A (within-bin only) against complete MF33/MF34 diagonals.
    #
    # Proved on run 81's paired replicas: the full joint sample covariance is PSD
    # at −1e-18, and zeroing ONLY the cross-energy entries makes it non-PSD in
    # every window, 13 orders worse, with those entries the same size as the ones
    # kept and ~159× more numerous.
    #
    # ⚠ WHAT MUST NOT MOVE. Run 88 is a fresh evaluation, so unlike run 87 it is
    # NOT an isolation of the cross term — the centrals, MF33 and MF34 are all
    # rebuilt. JEFF and JENDL must still be identical (they keep a zero cross
    # block); anything else moving in them is a knob that moved unintentionally.
    # V2 vs run 86 measures whatever else changed in the evaluation, not the
    # cross term, so do not read it as the cross term's cost.
    #
    # ⚠ READ THE [PSD] LINES FIRST. The joint is PSD by construction only in the
    # raw MC; Σ^MF33 and Σ^MF34 still pass through the collapse and the near-zero
    # guard, neither of which is a congruence. If the [PSD] lines are clean, the
    # §10.1.3 theorem is back and this is the shippable file. If they are not,
    # the residual is the representation term §10.1.5 priced at ~12 %.
    #
    # ⚠ AND CHECK `[XCROSS] form=full` IN THE PRECOMPUTE LOG. The loader falls
    # back to the within-bin block with a RuntimeWarning when the full sidecar is
    # absent, which would silently reproduce run 87.
    "predictive_88": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_88.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — run-88 evaluation with the COMPLETE MF33↔MF34 cross block, declared σ_E in the fold",
        "systematic_block_col": None,
    },
    # ⚠ RUN 89 HAS NO ENTRY ON PURPOSE, AND THAT IS THE POINT. It was never
    # added, so the job died on `Unknown methodology entry: ['predictive_89']`
    # after the 11 GB precompute had already succeeded — the identical omission
    # that cost run 85 its analysis (roadmap "Run 85's χ² — the recovery"). The
    # runner's `|| echo "[FAIL] not PD in chi2"` fired on the non-zero exit and
    # misreported it as a Cholesky failure. Nothing was scored at tag 89.
    #
    # Run 90 is the repair, and it is a different evaluation, not a re-run:
    # run 89 shipped cx_post as a sidecar next to the FILE's MF34, a pairing
    # that had never been diagnosed and measures σ_max(K) = 41.27,
    # λ_min/scale = −0.447 (§10.1.8-L). Run 90 ships the marginals the cross
    # block was built with — MF34 rebuilt from the same collapsed Pass-1
    # replicas, published σ's preserved to 2.8e-17 — so the joint is PSD by
    # construction rather than by repair.
    #
    # vs run 86: same MF3, MF4 and MF33; MF34 differs in its CORRELATIONS only
    # (median |Δρ| 0.0000, p95 0.0376), plus the cross term. So `predictive_90`
    # against `predictive_86` prices exactly one thing: making the file
    # internally consistent and carrying σ↔shape correlation.
    "predictive_90": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_90.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — run-90 consistent joint: MF34 from the collapsed Pass-1 replicas + the group-space MF33↔MF34 cross block",
        "systematic_block_col": None,
    },
    # Run 86 re-scored with the unsupported MF34 parameters removed (roadmap
    # §10.6-1). Same ENDF, same centrals, same MF33, NO cross block — the only
    # difference from run 86 is that the 1542 (group, order) slots the MC never
    # populated no longer contribute. Compare V2/V4 against `predictive_86`:
    # that difference IS the bias, and it is one-sided in This_work's favour
    # because JEFF and JENDL carry no such term. Sized at 2.14 % of the Sigma_eval
    # diagonal before this ran (§10.1.8-L14.2); a diagonal share does not map
    # linearly onto a chi2, which is why it is re-scored rather than scaled.
    #
    # REGISTERED BEFORE THE JOB WAS LAUNCHED. Runs 85 and 89 both died after
    # their precompute because their entry was missing here (§10.1.8-L1).
    "predictive_86_nonull": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_86_nonull.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — run 86 with MF34 parameters the fit never determined removed",
        "systematic_block_col": None,
    },
    # Run 86 re-scored with ONE change: MF34 blocks no longer extrapolate off
    # their own energy grid (roadmap §10.7-6). Same ENDF as run 86, same MF33,
    # same everything — only the fold changed.
    #
    # WHAT IT PRICES, and it is not ours. JEFF-4.0 publishes 20 of its 21 MF34
    # blocks from 1e-05 eV and **(2,6) only from 1 MeV**, while the EXFOR points
    # run down to 0.85 MeV. `np.clip` on the bin index pinned those points to
    # (2,6)'s first interval, so runs 82–90 all folded a **fabricated
    # Cov(a_2, a_6) for JEFF** below 1 MeV. This_work's own (2,6) starts at
    # 0.846822 MeV and covers every point, so it never had the problem.
    #
    # ⚠ SO THIS MOVES JEFF AND (probably) NOT US, which is the opposite of every
    # other single-variable job in this document. `This_work` must come out
    # IDENTICAL to run_086; if it does not, the mask is reaching something it
    # should not. JENDL has one MF34 block — check whether it moves at all.
    #
    # ⚠ AND IT MUST BE READ BEFORE `predictive_86_mf33grouped`, because that job
    # carries this change too. Scoring the grouping against run_086 directly
    # would confound the two.
    "predictive_86_maskfix": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_86_maskfix.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — run 86, MF34 blocks masked outside their own grid",
        "systematic_block_col": None,
    },
    # Run 86 with MF33/MT2 written on the 188-group ADAPTIVE grid instead of the
    # 1738-bin fine one (roadmap §10.7-4 step 2, `regroup_mf33.py`).
    #
    # ⚠ COMPARE THIS AGAINST `predictive_86_maskfix`, NOT against run_086. Both
    # carry the §10.7-6 mask; only this one also regroups MF33. Against run_086
    # it would be two variables.
    #
    # WHAT IT PRICES. §10.7-3 says grouping MF33 is the enabling step for the
    # cross term, not a size optimisation: it is what makes the fold a
    # congruence (`test_fold_maps.py`), it takes the cross block from +132 MB to
    # +11 MB, and it is what lets MF33 survive ENDF's six digits at all
    # (condition 1.6e9 fine vs 1.0e6 grouped). It costs relative sigma: median
    # 0.0616 -> 0.0528, PEAK 0.204 -> 0.113. Less evaluated variance means a
    # LARGER chi2 for This_work, so expect this to look worse and the JEFF gap
    # to narrow.
    #
    # SINGLE VARIABLE. Same run-86 directory, same MF34, same MF4, same MF3,
    # same centrals, no cross block in either — `regroup_mf33.py` copies the
    # shipped file and replaces only the MT2 self-block. JEFF and JENDL must
    # move by exactly 0.0; if they do not, something else moved.
    #
    # REGISTERED BEFORE THE JOB WAS LAUNCHED. Runs 85 and 89 both died after
    # their precompute because their entry was missing here (§10.1.8-L1).
    "predictive_86_mf33grouped": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_86_mf33grouped.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — run 86 with MF33 on the 188-group adaptive grid",
        "systematic_block_col": None,
    },
    # Run 86, NOTHING GROUPED — the third row of the grouping table (§10.7-9).
    #
    # Scores `26-Fe-56g_nominal.endf` (843 MB): MF33 on its 1738 fine bins AND
    # MF34 on its 1738 fine bins, against the shipped `_mg.endf` which is MF33
    # fine + MF34 grouped to 660, and against `_mf33grouped` which is 188 + 660.
    # Three points on one axis, same evaluation, same centrals:
    #
    #   predictive_86_fine         MF33 1738   MF34 1738     843.4 MB
    #   predictive_86              MF33 1738   MF34  660     205.8 MB
    #   predictive_86_mf33grouped  MF33  188   MF34  660     173.5 MB
    #
    # ⚠ `run_085_fine` is NOT this row. It is run 85's evaluation, and run 86 is
    # the re-evaluation that fixed the multigroup mask — different centrals, so
    # putting the two in one column would compare two things at once.
    #
    # WHAT IT ANSWERS. Job B priced the MF33 step at -57 % on Cierjacks, -52 %
    # on K&S and +22 % on the other 64 experiments, with the centrals BYTE-
    # IDENTICAL — so grouping cannot have improved the model, only softened
    # Sigma_eval. This row separates the MF34 grouping (the 111.7 MB half of the
    # file, and the one nobody has priced) from the MF33 grouping (the 75.7 MB
    # half, priced by B).
    #
    # Resources: the 843 MB parse and 21 blocks of 1738^2 need the 300G; the
    # walltime is raised to 8 h because the `_mg` precompute alone took 76 min
    # on a file a quarter the size.
    "predictive_86_fine": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_86_fine.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — run 86, NOTHING grouped (MF33 and MF34 both on the 1738 fine bins)",
        "systematic_block_col": None,
    },
    # ── ITEM 5: the MF33↔MF34 cross term, shipped in MF34's a_0 blocks ──────
    #
    # ONE FILE, TWO ROWS, and that is what makes it single-variable. The a_0
    # blocks are in the file for both; `build_mf34_block` skips `l_r < 1`, so
    # with KIKA_MF33_MF34_CROSS_FROM_FILE unset they are simply not read. The
    # marginals are therefore BYTE-IDENTICAL between the two rows — not "the
    # same construction", the same bytes.
    #
    #   predictive_91_rewrite   MF34 rebuilt from the collapsed replicas, cross NOT read
    #   predictive_91_cross     the same file, cross READ from the a_0 blocks
    #
    # ⚠ READ 91_rewrite AGAINST predictive_86_nonull, NOT against predictive_86.
    # `--null-fill zero` and the null mask remove the SAME 1542 (group, order)
    # slots — `build_group_cross`'s mask is `|a_nom_group| < tiny` and
    # `write_consistent_mf34`'s `live` is its exact complement — so the rewrite
    # has already had §L14.2's 2.14 % of the diagonal removed. Against run 86
    # the comparison would move two things.
    #
    # ⚠ 91_rewrite IS NOT A NULL RESULT BY CONSTRUCTION. It carries two changes
    # against 086_nonull: the MF34 correlations now come from the collapsed
    # replicas rather than the shipped file, and the 39 shape groups outside
    # 0.8468-4.075 MeV lose the host's merged MF34 (178 of 46819 points bin
    # into them). Its job is to absorb both so that 91_cross - 91_rewrite is
    # the cross term alone.
    #
    # §10.7-5 predicts the cross term at 1-3 pp. Measured before launch:
    # sigma_max(K) 0.999993 and lam_min/scale -1.4e-16 in the space the chi2
    # folds (§10.7-10, row F), against 1055 and -113 for the sidecar route that
    # killed runs 87-90.
    "predictive_91_rewrite": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_91_rewrite.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — MF34 rebuilt from the collapsed replicas, cross term NOT read",
        "systematic_block_col": None,
    },
    "predictive_91_cross": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_91_cross.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — MF33↔MF34 cross term read from MF34's a_0 blocks",
        "systematic_block_col": None,
    },
    # ── P1.3: the SPLIT-COMBINE MF33 (post-run-92 roadmap §2.6) ──────────────
    #
    # Covariance-only. Same centrals, same MF3, same MF4, same MF34 — the ONLY
    # thing that moves is MF33's off-diagonal, rebuilt so that the Pass-1
    # correlation carries only the variance Pass 1 actually measured and the
    # Pass-2 excess carries the measured local-residual correlation instead.
    # The tape's MF33 diagonal is byte-identical to run 92's (verified on the
    # written tape: max |rel diff| = 0.000e+00), so this declares NO extra
    # uncertainty — it redistributes it. Descriptors on the written tape:
    # coh 0.779 -> 0.344, PR 2.13 -> 9.47.
    #
    # ⚠ Base is 91_rewrite, NOT 91_cross: changing MF33's marginals breaks the
    # Cauchy-Schwarz compatibility of the a_0 cross block, which is only valid
    # for the marginals it was built from (§10.1 — this is what killed runs
    # 89 and 90). Read every subset against predictive_91_rewrite.
    #
    # ❌ DISQUALIFYING (§6e): if Kinney improves while n_absorbed FALLS
    # elsewhere in the corpus, that is redistribution dressed as information.
    # `corpus_absorbed.py` runs in the same chain and is not optional.
    "predictive_92split": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_92split.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — MF33 split-combine: Pass-1 correlation at its own scale + measured-correlation Pass-2 excess",
        "systematic_block_col": None,
    },
    # ── P1.4/(b2): the SPLIT-COMBINE MF34 (post-run-92 roadmap §2.7-quater) ──
    #
    # Run 95 = run 94 + ONE knob. Verified in run_metadata.json, not assumed:
    # 154 config keys against run 94's 153, and the single addition is
    # MF34_CORR_OVERRIDE -> chi2/p16_mf34_split/mf34_corr_split.npy. The runner's
    # own gate passed in all six parts: MF33's six .npy byte-identical,
    # nominal_fits.parquet byte-identical, mf34_std_perbin.npy byte-identical
    # (so this DECLARES NO EXTRA SIGMA — it redistributes), the saved corr_kw
    # equal to the input at 0.000e+00, the pipeline rebuilding the p16 product to
    # 2.2e-16, and all three tapes differing as required.
    #
    # What moved, on the shipped object: <|rho|> over the whole MF34 matrix
    # 0.1532 -> 0.0601, and coh 0.209 -> 0.054 / PR 15.26 -> 19.73 over the 6184
    # usable columns.
    #
    # ⚠⚠ MF33 IS NOT SPLIT IN THIS TAPE. Measured, not inferred: run 95's
    # mf33_absolute_covariance.npy has the same md5 (c68ac0ac...) as run 92's,
    # while the split MF33 object is a different file (4511d2b7..., in
    # myworkspace/chi2/p12_new_test_92_integrated/). So `predictive_95` is
    # MF34's repair ALONE, with the cross term present — it is the mirror of
    # `predictive_92split`, which was MF33's repair alone with the cross NOT
    # read. NEITHER IS THE DELIVERABLE. The tape carrying both repairs plus the
    # cross does not exist yet.
    #
    # ⚠⚠ TWO CHANGES TRAVEL IN THIS OBJECT, and the write-up cannot attribute a
    # move to the redistribution alone until they are separated. Besides the
    # redistribution, the split ERASES a +-1 correlation block that roundoff had
    # put across the 2517 columns that are exactly constant in Pass 1
    # (<|rho|> = 0.9992 between them). Measured bound on the ambiguity: those
    # 2517 columns carry 2.26 % of MF34's declared variance (the 6184 usable
    # ones carry 67.56 %). So a large move cannot plausibly be the roundoff
    # block; a small one needs the tiebreaker, which is NOT built.
    #
    # ⚑ THE BASE IS `predictive_91_cross`, AND IT IS ALREADY ON DISK. Measured
    # 2026-08-14 by cmp over all 346,800,528 bytes: the tape 91_cross scored
    # (86_a0cross/26-Fe-56g_nominal_a0cross_mg.endf) is BYTE-IDENTICAL to run
    # 92's, which run 94's gate proved identical to runs 93 and 94. So run 95 is
    # that same tape with one knob, and no run-94 scoring is needed — its parquet
    # and 11 GB sidecar exist as chi2_data_predictive_91_cross.*.
    #
    # ⚠ NOT 92split as the base: 92split moved MF33's off-diagonal and did not
    # read the cross, so against it two things move at once.
    #
    # ⚠ READING CRITERION, not a failure: the MF34 multigroup grid is chosen
    # adaptively from the l=1 correlation, which is exactly what this knob
    # changes. Run 94 grouped 1738 fine bins into 660; run 95 gives 896. That is
    # downstream of the single knob, not a second knob, and it must be reported.
    #
    # ❌ DISQUALIFYING (§6e): if Kinney improves while n_absorbed FALLS elsewhere
    # in the corpus, that is redistribution dressed as information.
    # `corpus_absorbed.py` runs in the same chain and is not optional.
    # ── RUN 99, PASOS C DEL PLAN: la rejilla fina pasa a ser la referencia ───
    #
    # Plan: kika-workspace/docs/chi2-mf4/plan_fine_reference_and_100k.md §C. Tres puntos
    # del MISMO objeto (la run 99), que se leen en DOS parejas y cada pareja
    # mueve UNA sola cosa:
    #
    #   99c0mg    mg 660 + cruzado, dead=drop, UNA rejilla   <- el ancla
    #   99c1fine  fino 1738 + cruzado, dead=drop             <- 99c0mg vs 99c1fine
    #                                                           = el cambio de REJILLA
    #   99c2fine  fino 1738 + cruzado, dead=carry            <- 99c1fine vs 99c2fine
    #                                                           = la RESTITUCION de a_5/a_6
    #
    # ⚠⚠ NO se leen contra `predictive_91_cross` ni contra `predictive_98raw`.
    # Aquellas son OTRA run del pipeline; comparar contra ellas mueve la run y
    # el objeto a la vez, que es exactamente el confound por el que se
    # retiraron §L16 y §L18. El ancla de estas tres es 99c0mg y nada mas.
    #
    # ⚠ Los tres se emiten con `--per-order-mesh off`, INCLUIDO el mg. La run 99
    # escribio `mf34_per_order_mesh.npz` sobre la rejilla de 660 (w_l es
    # 649x660 ... 646x660), asi que `auto` la aplicaria — al mg legitimamente,
    # pero al FINO le colapsaria 1738 bins sobre una malla elegida encima de los
    # 660, que es el regroup-de-un-regroup que este plan elimina. Con `off` los
    # tres estan en una sola rejilla y la unica diferencia entre c0 y c1 es fina
    # contra mg.
    #
    # ⚠ Sin `KIKA_MF34_NULL_MASK`: la mascara vive en la malla de 703 de la
    # run 86 y `eval_covariance` la coloca POR ENERGIA, asi que una mascara
    # caduca se aplica en silencio. La retirada ya la lleva la cinta via
    # `--null-fill zero` + el complemento de `live` en write_consistent_mf34.
    #
    # ⚠ VETO, igual que siempre: JEFF y JENDL al 0.00000 % entre los tres. Si
    # se mueven, ninguno de los dos pares es de una sola variable.
    #
    # PREDICCION de 99c1fine vs 99c0mg: la run 86 midio el mismo cambio de
    # rejilla SIN cruzado y V4 se movia hasta un 8 % (only KS -8.0 %,
    # no_KS_no_Cierjacks +6.2 %), con V2 identico. Aqui hay cruzado ademas.
    # PREDICCION de 99c2fine vs 99c1fine: `carry` restituye 1307 de 4218
    # parametros de forma y rank34 sube de 2660 a 3481, o sea mete
    # incertidumbre REAL donde hoy hay ceros ⇒ V4 tiene que BAJAR. V2 identico
    # a cuatro decimales en los tres, que es el control gratis.
    "predictive_99c0mg": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_99c0mg.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — run 99, mg 660 + cruzado a_0, una rejilla: el ancla de los pasos C",
        "systematic_block_col": None,
    },
    "predictive_99c1fine": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_99c1fine.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — run 99, rejilla FINA 1738 + cruzado a_0, parámetros muertos DROP: aísla el cambio de rejilla",
        "systematic_block_col": None,
    },
    "predictive_99c2fine": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_99c2fine.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — run 99, rejilla FINA 1738 + cruzado a_0, 1307 parámetros de forma restituidos (hilo B): aísla a_5/a_6",
        "systematic_block_col": None,
    },
    # MEDIDO 2026-08-20. Las dos predicciones aciertan: c1/c0 sube V4 +1.2 a
    # +9.2 %, c2/c1 lo baja 3.3-5.8 % en los SEIS subconjuntos. JEFF y JENDL al
    # 0.00000 % y V2 identico en los tres, o sea las dos parejas son limpias.
    # ⚠ EL NETO c2/c0 ES UN EMPATE (-0.51 % en no_Cierjacks, +1 a +3 % en cuatro
    # de seis): la rejilla fina NO gana al mg en puntuacion. Parte de la ventaja
    # del mg es `apply_percentile_variance_scaling` re-inflando su diagonal, que
    # se ve en V1 (+3.5 % y +9.8 % en los dos subconjuntos finos). La referencia
    # fina se defiende por CORRECCION (2.4 % de sigma=0 contra 37-42 %), no por
    # chi2, y asi hay que escribirla.

    # ── LAS MALLAS: cuanto se puede AGRUPAR el objeto fino ────────────────
    #
    # Los tres son `c2` MAS una malla por orden -- misma cinta de origen, mismo
    # cruzado, mismo `carry`, mismo `--mag-grid fine`. La UNICA variable contra
    # `predictive_99c2fine` es el agrupamiento, asi que la pareja m*/c2 mueve una
    # sola cosa y c2 es su ancla. Cualquier lectura contra `99c0mg` mueve ADEMAS
    # `apply_percentile_variance_scaling`, que solo vive en el camino multigrupo
    # del pipeline.
    #
    # La malla la fijan dos restricciones derivadas del dato, no elegidas
    # (`kika-workspace/notebooks/mf34_mesh/`): anchura <= res_factor * sigma_E
    # (la resolucion TOF del experimento) y a_l constante dentro de k sigmas de
    # su miembro mas preciso (el fichero MF34 es RELATIVO: lo homogeneo tiene que
    # ser el DENOMINADOR).
    #
    # ⚑ MEDIDO ANTES DE LANZAR. Criterio de aceptacion = CONSERVADURISMO, o sea
    # que el fichero no declare MENOS varianza de la que el objeto fino tiene a
    # la resolucion a la que se lee:
    #
    #   malla    params   MiB    sub-declara   sobre-declara
    #   m1k3       7949   773       0.65          1.28
    #   m2k10      6746   621       0.27          1.97
    #   m3r2       6006   538       0.34          2.54
    #   c2 fino   10428  1145       1.00          1.00
    #   c0 HOY     3960   327       0.01           354
    #
    # Ninguna es estrictamente conservadora; la de HOY es con diferencia la peor.
    # Y quitar el tope de resolucion del todo solo baja a 5582 params (-17 %) a
    # cambio de 165x en a_5: lo que ata mas alla de la resolucion es la
    # consistencia de a_l, no la resolucion.
    #
    # PREDICCION FIJADA ANTES DE MEDIR: los tres caen dentro de +-2 % de c2. Si
    # acierta, queda MEDIDO que el chi2 no puede arbitrar la malla -- se sienta
    # en la mediana del cociente plegado, ~1 en todas -- y §10.8 se cierra con un
    # negativo medido. El control agresivo ya esta medido y es `c0`.
    "predictive_99m1k3": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_99m1k3.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — run 99, fino + cruzado, malla por orden σ_E×1 k=3 (7949 params): la más fiel",
        "systematic_block_col": None,
    },
    "predictive_99m2k10": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_99m2k10.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — run 99, fino + cruzado, malla por orden σ_E×1 k=10 (6746 params): la malla física",
        "systematic_block_col": None,
    },
    "predictive_99m3r2": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_99m3r2.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — run 99, fino + cruzado, malla por orden σ_E×2 k=10 (6006 params): agrupa MÁS allá de la resolución",
        "systematic_block_col": None,
    },

    # ── RUN 100, LOS MISMOS TRES PUNTOS CON 100 000 MUESTRAS ──────────────
    #
    # ⛔ NO LANZAR HASTA QUE LA RUN 100 HAYA TERMINADO. Se lanzan con
    #   sbatch -J kika-c0-100 run_c_fine.sh c0 /share_snc/snc/JuanMonleon/ENDF_samples/new_test_100_100k
    #   sbatch -J kika-c1-100 run_c_fine.sh c1 /share_snc/snc/JuanMonleon/ENDF_samples/new_test_100_100k
    #   sbatch -J kika-c2-100 run_c_fine.sh c2 /share_snc/snc/JuanMonleon/ENDF_samples/new_test_100_100k
    # El runner deriva el prefijo `100` del nombre del directorio.
    #
    # QUE PREGUNTA NUEVA CONTESTAN. Las dos parejas internas (c0/c1 la rejilla,
    # c1/c2 la restitucion) se repiten como control. La pregunta NUEVA es la
    # TERCERA pareja, entre runs y sobre el mismo punto:
    #
    #   99c2fine  vs  100c2fine   <-  p/n 1.043 -> 0.104
    #
    # A 10k, p = 10 428 parametros contra n = 10 000 replicas: el soporte
    # Marchenko-Pastur de los autovalores de la covarianza muestral es [0, 4.1]
    # y la ESTRUCTURA DE CORRELACION fina es en buena parte ruido de MC (la
    # diagonal esta bien, error relativo sqrt(2/n) ~ 1.4 %). A 100k el soporte
    # es [0.46, 1.75]. Si V4 se mueve entre 99c2fine y 100c2fine, lo que se
    # mueve es ruido de correlacion, no fisica.
    #
    # ⚠ ES LA UNICA COMPARACION DE ESTA TANDA QUE CRUZA DOS RUNS DEL PIPELINE, y
    # se permite SOLO porque las dos son el mismo punto del mismo objeto con la
    # misma config salvo N_SAMPLES (verificado en run_metadata.json). No leer
    # 100c1fine contra 99c0mg ni ninguna otra pareja cruzada: eso mueve la run y
    # la representacion a la vez, que es §L16/§L18 otra vez.
    #
    # ⚑ LO QUE 100k NO ARREGLA: `mc_order_cap` es el soporte angular del bin y
    # no depende de n, asi que a_5/a_6 siguen muertos en las replicas en los
    # mismos bins y `carry` (c2) sigue siendo necesario exactamente igual.
    "predictive_100c0mg": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_100c0mg.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — run 100 (100k), mg 660 + cruzado a_0, una rejilla: el ancla",
        "systematic_block_col": None,
    },
    "predictive_100c1fine": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_100c1fine.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — run 100 (100k), rejilla FINA 1738 + cruzado a_0, parámetros muertos DROP",
        "systematic_block_col": None,
    },
    "predictive_100c2fine": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_100c2fine.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — run 100 (100k), rejilla FINA 1738 + cruzado a_0, forma restituida (carry): el entregable candidato",
        "systematic_block_col": None,
    },
    # ── SERIE 103 (22/23-ago): la malla en UNA etapa, contra la de dos ──────
    #
    # ⚑ PAREJA DE UNA SOLA VARIABLE. 103R2 y 103R4 salen del MISMO codigo, la
    # MISMA semilla y la MISMA config salvo `KIKA_MF34_MESH_SINGLE_STAGE`. Las
    # dos se puntuan por la cinta FINA con cruzado (`_a0cross.endf`, dead=carry),
    # asi que lo unico que cambia entre las dos es la MALLA: 3 311 parametros
    # (dos etapas) contra 7 820 (una etapa desde el fino).
    #
    # ⛔ Y NO SE PUNTUA PARA ELEGIR MALLA. Medido en la serie 99: el chi2 baja un
    # 17 % monotono segun se engruesa la malla, y todo el efecto viene de la
    # correlacion -- discrimina AL REVES del criterio de conservadurismo. Esto se
    # puntua para REPORTAR el numero del entregable, no para decidir.
    #
    # ⚠ 103R4 es ademas el ENSAYO DE MEMORIA: 7 820 parametros es la MF34 mas
    # fina que se ha puntuado nunca (91_cross fueron 703 grupos / 347 MB), asi
    # que si el sidecar de ~11 GB o las eigendescomposiciones no caben, se quiere
    # descubrir aqui y no con el entregable definitivo.
    "predictive_103R2": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_103R2.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² — run 103R2, malla de DOS etapas (3 311 params), fina + cruzado a_0 (carry)",
        "systematic_block_col": None,
    },
    "predictive_103R4": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_103R4.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² — run 103R4, malla en UNA etapa desde el fino (7 820 params), fina + cruzado a_0 (carry)",
        "systematic_block_col": None,
    },
    # ── SERIE 104: la malla en una etapa CON el arreglo del singleton ─────
    #
    # Las tres salen del MISMO codigo y la MISMA covarianza (la de 103R4); lo
    # unico que cambia entre ellas son los dos parametros del criterio fisico:
    #   S1  k=10  c=3   5 964 params   <- 103R4 + SOLO el arreglo del singleton
    #   S2  k= 3  c=3   6 349 params   <- el entregable pre-registrado
    #   S3  k= 3  c=2   6 888 params   <- variante conservadora
    # Las tres reprodujeron su prediccion offline EXACTA y no dieron ni un aviso
    # de no-cancelacion.
    #
    # ⛔ NO SE PUNTUA PARA ELEGIR MALLA. Medido en la serie 99 y CONFIRMADO el
    # 23-ago sobre la pareja 103R2/103R4: la malla 2,4x mas fina puntua un 4,0 %
    # PEOR en V4 (5,69 -> 5,92) y un 2,4 % MEJOR en V1, y todo el delta viene de
    # la correlacion. El chi2 discrimina AL REVES del criterio de
    # conservadurismo. Esto se puntua para REPORTAR el numero del entregable.
    #
    # ⚠ Las tres se puntuan por la cinta FINA con cruzado (`_a0cross.endf`,
    # dead=carry), igual que 103R2 y 103R4, para que la comparacion siga siendo
    # de una sola variable contra ellas.
    "predictive_104S1": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_104S1.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² — run 104S1, malla 1 etapa + arreglo singleton, k=10 c=3 (5 964 params), fina + cruzado a_0 (carry)",
        "systematic_block_col": None,
    },
    "predictive_104S2": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_104S2.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² — run 104S2, malla 1 etapa + arreglo singleton, k=3 c=3 (6 349 params), fina + cruzado a_0 (carry)",
        "systematic_block_col": None,
    },
    # ── RUN 105T1: 104S2 con los sigma_E de entrada corregidos ─────────────
    # MISMOS flags que 104S2. La UNICA variable son los inputs de sigma_E:
    # el JSON corregido (Smith 5.25 m, Perey 200.191 m, Salnikov como caja,
    # Cox corroborado), los dos canales nuevos del resolver (cuarentena como
    # caja, default relativo 1.31 % de E) y el marco CM de Becker 11511009.
    # Se lee CONTRA predictive_104S2 y contra nada mas.
    "predictive_105T1": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_105T1.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² — run 105T1, flags de 104S2 con sigma_E de entrada corregida (Smith 5.25 m, cuarentena como caja, default 1.31 % de E, Becker en CM)",
        "systematic_block_col": None,
    },
    # ── RUN 106T2: 105T1 + los dos arreglos aguas arriba del ajuste ────────
    # 1. Se rechazan los candidatos Legendre IMPOSIBLES (|a_l| > 1) antes de
    #    pesarlos por AIC. a_l = <P_l(mu)> con |P_l| <= 1, asi que un ajuste que
    #    afirma |a_1| > 1 no es una hipotesis rival sino un artefacto de minimos
    #    cuadrados sin restriccion en un bin con n_eff ~ 1. En 105T1 contaminaba
    #    2 bins (1,558 y 1,560 MeV): volteaba avg_a_1 contra todos sus vecinos y
    #    dejaba sigma(a_1) = 1,667, mas que el rango fisico del coeficiente.
    # 2. La sintesis de DATA-ERR de Gkatis 27673002, que corria con sigma 1 %
    #    plana porque la guarda era `is None` y el cargador JSON pone `[]`.
    # ⚠ Los DOS mueven el central, asi que esto NO es una comparacion de una
    #   sola variable contra 104S2. Se lee contra predictive_105T1.
    "predictive_106T2": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_106T2.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² — run 106T2, 105T1 + rechazo de candidatos Legendre imposibles (|a_l| > 1) y sintesis DATA-ERR de Gkatis 27673002",
        "systematic_block_col": None,
    },
    "predictive_104S3": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_104S3.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² — run 104S3, malla 1 etapa + arreglo singleton, k=3 c=2 (6 888 params), fina + cruzado a_0 (carry)",
        "systematic_block_col": None,
    },
    # ── CINTA B-SPLINE v3 (2-sep-2026): la via de ajuste directo, no el pipeline ──
    # MF4 MT2 = a_1..a_6(E) del ajuste B-spline plegado (receta v3 de la Fase W:
    # eficiencias por detector contra el consenso de terceros, escala de energia
    # de Kinney por detector, Cierjacks multiplicativo), 631 puntos a 5 keV en
    # 0,850-4,000 MeV, lambda por validacion fina (1 030 edf). MF34 MT2 = 63 bins
    # x 6 ordenes (50 keV), sandwich + nuisances de la receta, LTT 1.
    # MF3 y MF33 son los del HOST (JEFF-4.0): la cinta no lleva magnitud propia.
    # SIN cruzado a_0 (KIKA_MF33_MF34_CROSS_FROM_FILE=0 en el brazo B1).
    # ⚠ NO es comparable de una sola variable con 104S2/106T2: cambia el metodo
    # entero (ajuste directo contra mezcla AIC + MC). Se lee contra JEFF/JENDL
    # dentro de su propio informe, y contra 106T2 solo como "que via va mejor".
    # Cinta: /share_snc/snc/JuanMonleon/splines/deliverable/26-Fe-56g_bspline_v3.endf
    # Origen: kika-workspace/myworkspace/chi2/bspline_window/w16_write_tape.py
    "predictive_bspline_v3": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_bspline_v3.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² — cinta B-spline v3: MF4+MF34 del ajuste plegado (1 030 edf, 63x6 a 50 keV), MF3/MF33 del host JEFF, sin cruzado",
        "systematic_block_col": None,
    },
    # ── CINTA B-SPLINE v3, MF34 SOLO ESTADISTICA (2-sep noche) ───────────────
    # MISMO MF4 que predictive_bspline_v3; la MF34 es solo el sandwich a lambda
    # fija, sin los 27 terminos nuisance (eficiencias por detector, escala de
    # energia de Kinney, Cierjacks aditivo/multiplicativo). Pregunta que responde:
    # el V4 de bspline_v3 (1,06; Kinney 0,24, Cierjacks 0,18 con V2 79 y 27) es
    # bueno porque el central esta bien, o porque los modos nuisance de rango bajo
    # y totalmente correlados en E apuntan justo hacia donde la cinta se aparta de
    # Kinney y Cierjacks (que es de donde salieron)? Se lee CONTRA
    # predictive_bspline_v3 y contra nada mas: si Kinney/Cierjacks V4 se quedan
    # cerca de 0,3 el central aguanta; si suben a varios, los modos hacian el trabajo.
    "predictive_bspline_v3_statonly": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_bspline_v3_statonly.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² — cinta B-spline v3 con MF34 solo estadistica (sandwich, sin nuisances); mismo MF4 que bspline_v3, MF3/MF33 del host JEFF, sin cruzado",
        "systematic_block_col": None,
    },
    # ── CINTA B-SPLINE v4 (3-sep-2026): nivel libre, plan_revision §9 ─────────
    # MF4 MT2 = a_1..a_6 = beta_l / m del ajuste con el nivel m(E) libre (M2, lambda_m =
    # 0,01 lambda_a, ancla de sigma_tot de Cornelis 1995, tau_e recalibrado una vez), 631
    # puntos a 5 keV. MF34 MT2 = bloques (l,l') del sandwich CONJUNTO (m, a_l) + nuisances
    # (eficiencias, escala de energia, Cierjacks, variantes de P1 y la eleccion de tau),
    # 63 bins x 6 ordenes. MF33 MT2 = bloque (0,0) del mismo sandwich, RELATIVO al nivel
    # ajustado y aplicado sobre el MF3 del host (inconsistencia declarada, medida:
    # m/sigma_JEFF por bin en el manifiesto). Cruzado = bloques (0,l) como fila L=0 de MF34
    # (LTT 3), la convencion de 104S2: CHICROSS=1 en el brazo B3.
    # Se lee contra bspline_v3 (B1): mismo corpus, misma receta de observacion; cambia el
    # nivel (libre) y la covarianza (conjunta). Y contra 106T2, JEFF y JENDL como B1.
    "predictive_bspline_v5_tau1": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_bspline_v5_tau1.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² — cinta B-spline v4 (nivel libre, tau recalibrado): MF4+MF34+MF33+cruzado del sandwich conjunto; MF3 del host JEFF",
        "systematic_block_col": None,
    },
    # La MISMA receta SIN corregir a Kinney (LOEO la prefiere un 11 %; los dos finos dejan de ser
    # compatibles): se lee contra predictive_bspline_v5_tau1; V2 es el estadistico libre de banda.
    "predictive_bspline_v5_tau1_nokinney": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_bspline_v5_tau1_nokinney.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² — cinta B-spline v5 sin las correcciones de Kinney (misma lambda, mismos nuisances)",
        "systematic_block_col": None,
    },
    # La MISMA cinta sin cruzado (MF34 LTT 1, MF33 propia): se lee contra predictive_bspline_v5_tau1.
    "predictive_bspline_v5_tau1_noxs": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_bspline_v5_tau1_noxs.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² — cinta B-spline v4 sin cruzado (misma MF4/MF34/MF33, bloques a_0 a cero)",
        "systematic_block_col": None,
    },
    # 3-sep (manana): las mismas tres cintas con la MF34 en una malla POR ORDEN (el DP m2k10c3 de la
    # pipeline anterior sobre la rejilla de 5 keV del ajuste; colapso 'max' + margen plegado, la barra
    # de conservadurismo de 104S2). Misma MF4 bit a bit: se leen contra v5_tau1 / _noxs / _nokinney.
    "predictive_bspline_v5_tau1_perorder": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_bspline_v5_tau1_perorder.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² — cinta B-spline v5 con MF34 en malla por orden (DP m2k10c3, colapso conservador); MF33 + cruzado",
        "systematic_block_col": None,
    },
    "predictive_bspline_v5_tau1_perorder_noxs": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_bspline_v5_tau1_perorder_noxs.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² — cinta B-spline v5 malla por orden, sin cruzado (bloques a_0 a cero)",
        "systematic_block_col": None,
    },
    # 3-sep (mediodia): B6 con el ancla del borde de 4 MeV (estudio §18.19). Se lee contra B6.
    "predictive_bspline_v5_tau1_edge_perorder": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_bspline_v5_tau1_edge_perorder.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² — cinta B-spline v5 malla por orden, con ancla del anfitrión en 4 MeV",
        "systematic_block_col": None,
    },
    "predictive_bspline_v5_tau1_nokinney_perorder": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_bspline_v5_tau1_nokinney_perorder.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² — cinta B-spline v5 malla por orden, sin las correcciones de Kinney",
        "systematic_block_col": None,
    },
    # ── 4-sep: la base de datos CORREGIDA (Juan: «si esos puntos deben corregirse, la comparativa
    # debería hacerse con los puntos corregidos; si no, V2 claro que sale peor»). El precompute con
    # KIKA_EXFOR_CORRECTIONS divide los puntos de Kinney y Cierjacks por su eficiencia de detector y
    # pliega cada librería a la energía reescalada de cada detector de Kinney (la MISMA corrección para
    # JEFF, JENDL y This_work: variable única). Cada entrada `_corr` se lee contra su gemela cruda.
    "predictive_bspline_v5_tau1_edge_perorder_corr": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_bspline_v5_tau1_edge_perorder_corr.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² — cinta B-spline v5 (por orden, ancla del borde) contra la base EXFOR CORREGIDA (eficiencias por detector y escala de energía de Kinney, aplicadas a las tres evaluaciones)",
        "systematic_block_col": None,
    },
    "predictive_106T2_corr": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_106T2_corr.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² — pipeline por bins 106T2 contra la base EXFOR CORREGIDA (mismas correcciones que _bspline_*_corr; se lee contra predictive_106T2)",
        "systematic_block_col": None,
    },
    # 4-sep: la cinta con la malla del NIVEL decidida por el DP (W16_LEVEL_MESH=dp, sufijo _lvdp) en vez
    # de los 50 keV heredados de v3. Misma MF4 y MF34 que edge_perorder: se lee contra ella (cruda) y
    # contra su gemela corregida.
    "predictive_bspline_v5_tau1_edge_perorder_lvdp": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_bspline_v5_tau1_edge_perorder_lvdp.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² — cinta B-spline v5 (por orden, ancla del borde) con la MF33 en la malla del DP (misma regla que los órdenes)",
        "systematic_block_col": None,
    },
    "predictive_bspline_v5_tau1_edge_perorder_lvdp_corr": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_bspline_v5_tau1_edge_perorder_lvdp_corr.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² — cinta B-spline v5 (por orden, ancla del borde, MF33 en malla del DP) contra la base EXFOR CORREGIDA",
        "systematic_block_col": None,
    },
    # ── 6-sep: la cinta v6 SIN host (sin ancla de sigma_tot ni de borde, ventana derivada de los datos
    # 0,820-4,035 MeV, nivel/modos/correcciones por iteración de punto fijo en 5 pasadas, regla de malla
    # m2k10c3 medida). B11 crudo, C11 corregido con SU tabla (w18_v6_x5_efficiencies.csv: las eficiencias
    # y la escala de Kinney de la v6 no son las de la v5). C11 se lee contra B11; B11/C11 contra
    # B10/C10 (la v5 con la misma malla de nivel).
    "predictive_bspline_v6_x5_perorder_m2_c3_lvdp": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_bspline_v6_x5_perorder_m2_c3_lvdp.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² — cinta B-spline v6 sin host (punto fijo x5, ventana derivada, por orden m2k10c3, MF33 en malla del DP)",
        "systematic_block_col": None,
    },
    "predictive_bspline_v6_x5_perorder_m2_c3_lvdp_corr": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_bspline_v6_x5_perorder_m2_c3_lvdp_corr.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² — cinta B-spline v6 sin host contra la base EXFOR CORREGIDA con las eficiencias de la propia v6",
        "systematic_block_col": None,
    },
    # ── 6-sep (noche): la MISMA cinta v6 con el término `level_vs_host` en la MF33 (w16 docstring (vi)): la
    # diferencia entre el nivel ajustado y el MT2 del host que embarca la cinta, suavizados a 100 keV, como
    # término totalmente correlado del nivel (MF4, MF34 y cruzados iguales salvo la malla del nivel que
    # re-decide el DP). Diagnóstico de B11: V1/V3/centro iguales que v5 y V4 2,08 → 3,20 porque la MF33 de v6
    # perdió el modo común (nivel promedio 2,2 % contra +6,5 % de diferencia con el host). B12 crudo, C12
    # corregido (tabla v6). Se leen contra B11/C11 y contra B10/C10.
    "predictive_bspline_v6_x5_perorder_m2_c3_lvdp_lh": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_bspline_v6_x5_perorder_m2_c3_lvdp_lh.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² — cinta B-spline v6 sin host, MF33 con el término nivel-ajustado contra MT2 del host (level_vs_host)",
        "systematic_block_col": None,
    },
    "predictive_bspline_v6_x5_perorder_m2_c3_lvdp_lh_corr": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_bspline_v6_x5_perorder_m2_c3_lvdp_lh_corr.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² — cinta B-spline v6 sin host con level_vs_host, contra la base EXFOR CORREGIDA (eficiencias de la v6)",
        "systematic_block_col": None,
    },
    # ── RUN 97: una malla por orden Legendre (roadmap §10.8) ────────────────
    #
    # Candidata frente a `predictive_91_cross`, la misma base contra la que se
    # leyo `predictive_95`, y por la MISMA ruta: los bloques a_0 del fichero
    # (`KIKA_MF33_MF34_CROSS_FROM_FILE=1`), nunca el sidecar.
    #
    # ⚠ SIN `KIKA_MF34_NULL_MASK`, y por la razon que dejo escrita la run 95,
    # solo que aqui es mas fuerte: la mascara vive en la malla de 703 grupos de
    # la run 86 y el MF34 de la 97 esta en 679/703/637/472/299/105 -- SEIS
    # mallas, ninguna de ellas esa. `eval_covariance` coloca la mascara POR
    # ENERGIA y precompute solo valida el npz contra si mismo, asi que una
    # mascara caduca se aplicaria en silencio en vez de fallar. La cinta ya
    # lleva la retirada por `CROSS_NULL_FILL='zero'` mas el complemento de
    # `live` en write_consistent_mf34.
    #
    # ⚑ MEMORIA: riesgo MENOR que el de la run 95, no mayor. Aquel job temia la
    # OOM porque su MF34 era mas FINO que nada puntuado hasta entonces (896
    # grupos, 485 MB); el de la 97 es mas GRUESO en cinco de los seis ordenes.
    # Si aun asi la mata la OOM, descomentar `--mem` en la cabecera de run_chi.sh.
    # La run 98 es un cambio de REPRESENTACION sobre la misma covarianza: la malla
    # por orden colapsa MF34 y el cruzado con la MISMA U, y el colapso es una
    # contraccion demostrada (amp = 1 en los seis ordenes, max|c34_rel| 0.9945 ->
    # 0.9945 exacto). ⇒ EL CHI2 TIENE QUE SALIR IGUAL QUE EL DE LA RUN 96/94.
    # Eso es lo que este job comprueba; una diferencia no seria una mejora, seria
    # un fallo del colapso. La base de lectura es `predictive_91_cross`.
    "predictive_98raw": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_98raw.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "\u03c7\u00b2 analysis \u2014 malla por orden emitida en 691/699/690/690/699/703 grupos, colapso demostrado contractivo",
        "systematic_block_col": None,
    },
    "predictive_97mesh": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_97mesh.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "\u03c7\u00b2 analysis \u2014 una malla por orden Legendre: MF34 a 679/703/637/472/299/105 grupos, cruzado colapsado con la misma U",
        "systematic_block_col": None,
    },
    "predictive_95": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_95.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — MF34 split-combine: Pass-1 correlation redistributed at its own scale, declared σ unchanged",
        "systematic_block_col": None,
    },
    # ⛔ RESULT (job 8488732, 2026-08-14): `predictive_95` IS DISQUALIFIED and is
    # kept here only so the measurement stays reproducible. The veto passed
    # exactly (JEFF/JENDL 0.00000 % on all 24 subset x variant comparisons, and
    # max |delta n_absorbed| = 0.000e+00 on 62 experiments each), so the run was
    # single-variable and the reading is sound — but §6e trips:
    #     Kinney            n_absorbed 1447.22 -> 1596.39   +10.31 %
    #     THE OTHER 61      n_absorbed 1563.55 -> 1535.82    -1.77 %   <-- FALLS
    # 25 of the other 61 fall and only 9 rise. That is Kinney gaining directions
    # by taking them from the rest of the corpus, which is the exact pattern §6e
    # was written to disqualify, and the chi2 agrees rather than fighting it:
    # no_Cierjacks V4/N 8.4437 -> 8.5445 (+1.19 %), and every subset except
    # only_KS (-0.69 %) gets worse.
    #
    # ⚑ WHY, measured on the shipped object: the 4240 columns that are exactly
    # constant in Pass 1 carry 32.4 % of MF34's declared variance and come back
    # with EXACTLY ZERO off-diagonal mass (sampled directly out of
    # mf34_corr_split.npy), against ~1080 per row on the usable ones. The repair
    # made a third of MF34's declared variance perfectly independent — not
    # because independence was measured, but because there is no data there.
    # That is the 40.7 % the roadmap already declared before the run.
    #
    # ⛔ DO NOT score `_splitloc_plain.npy` "to compare". Choosing between two
    # variants of one repair on the scoreboard is §10.8-6, and it is the thing
    # this work holds against JENDL. The choice was made on a measured artefact
    # and a failed score does not reopen it.
    # ── THE DELIVERABLE CANDIDATE: MF33's repair WITH the cross term ─────────
    #
    # The one combination that has never been scored, and the only one that puts
    # a repair which passed BOTH gates next to the term that ships by physics:
    #   92split      MF33 repaired, MF34 original, cross NOT read   V4 4.4358 ✅
    #   91_cross     MF33 original, MF34 original, cross read       V4 8.4437
    #   95           MF33 original, MF34 repaired, cross read       V4 8.5445 ⛔
    #   THIS ONE     MF33 repaired, MF34 original, cross read       never measured
    #
    # Two free readings, both bases already on disk: against `92split` it prices
    # the cross term on top of the repair; against `91_cross` it prices the
    # repair with the cross present.
    #
    # ⚠⚠ IT IS GATED, AND THE GATE IS NOT OPTIONAL. §10.1: the a_0 cross block is
    # Cauchy-Schwarz-compatible only with the marginals it was built from, and
    # skipping that is what killed runs 89-90. Reading build_group_cross.py
    # settles what is and is not at risk here:
    #   * the rescaling `jj = d_tar / d_mc` uses ONLY the diagonals, and the
    #     split preserves MF33's diagonal byte-for-byte, so `cx_post` comes out
    #     numerically IDENTICAL either way — the cross block itself is not the
    #     risk;
    #   * but the certified joint is [c33_post, cx_post; ...] where `c33_post`
    #     is the MC matrix, while the tape ships the SPLIT MF33, whose
    #     off-diagonal is very different (coh 0.900 -> 0.391, PR 1.78 -> 8.06).
    #     The joint the chi2 actually folds is therefore NOT the PSD-by-
    #     construction object, and the split moves it in the risky direction:
    #     it SHRINKS the coherent variance (sigma_coh 0.0554 -> 0.0225) that the
    #     cross block leans on.
    # `build_group_cross.py --c33-from-file --check` diagnoses exactly that, on
    # the MF33 the chi2 folds, and writes nothing. It runs FIRST, as its own
    # short job, and is READ before the hour is spent.
    "predictive_92split_cross": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_92split_cross.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — MF33 split-combine WITH the MF33↔MF34 cross term read from the a_0 blocks",
        "systematic_block_col": None,
    },
    # ── THE ANALYTIC (R2) EVALUATION ─────────────────────────────────────────
    #
    # `docs/chi2-mf4/cross_term_two_pass_investigation.md` §10-11. MF33, MF34 and the
    # cross block all come from ONE object: the closed-form covariance of the
    # estimator that produced the central values,
    #
    #     C = A A^T  +  sum_b B_b(model)      A = d(param)/d(xi)
    #
    # read out of the pipeline's own solvers by identity probing (c0 and a_l are
    # affine in the data). PSD by construction, so the cross term needs no
    # rescaling and no repair: sigma_max(K) = 0.99949, lam_min = -6.5e-19 in the
    # units written, against 1.0205 / -1.3e-07 for the shipped file and
    # 84.72 / -4.38 for the split (which is what killed MF33-split + cross).
    #
    # Built by myworkspace/chi2/r2_{dump_nominal,analytic_joint,group_joint,
    # stage_tape_inputs,write_cross_tape}.py; tape in chi2/r2_analytic_tape/.
    #
    # ⚑ The declared MF33 sigma barely moves (median ratio 0.9994 against the
    # shipped absolute), so this is a REATTRIBUTION of the correlations, not an
    # inflation: V2 stays a free control and it must come back unchanged.
    # coh 0.657 against 0.845 shipped and 0.368 split -- the exact number lands
    # between the two candidates the deliverable was choosing from.
    #
    # ⛔ SCORE IT TO REPORT, NOT TO CHOOSE. Sec. 10.8-6 applies unchanged: the
    # object was built and gated on structural criteria (marginal reproduction,
    # PSD, Cauchy-Schwarz), never on V4.
    #
    # Precompute:
    #   KIKA_THIS_WORK_DIR=/share_snc/snc/JuanMonleon/chi2/r2_analytic_tape \
    #   KIKA_THIS_WORK_ENDF=26-Fe-56g_nominal_a0cross_mg.endf \
    #   KIKA_MF33_MF34_CROSS_FROM_FILE=1 KIKA_RUN_TAG=r2analytic \
    #   python precompute_chi2_predictive.py
    "predictive_r2analytic": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_r2analytic.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — analytic joint (R2): MF33, MF34 and the cross term from one PSD object",
        "systematic_block_col": None,
    },
    # ⛔ EXPOSURE MEASUREMENT, NOT A CANDIDATE DELIVERABLE (2026-08-17).
    #
    # The analytic tape declares σ_rel > 100 % on 201 LIVE (group, order) slots,
    # max 6337 %, because the group mean of a_l cancels there. Those 201 carry
    # 44.62 % of MF34's absolute variance. And `eval_covariance.build_mf34_block`
    # rescales the file's RELATIVE block by `a_l_per_pt` — MF4's a_l interpolated
    # AT EACH DATA POINT — not by the group mean that is actually the file's
    # denominator. That is a CORRECT reading of ENDF, which is what makes it a
    # measurement: an honest consumer reads something we did not build, and
    # a_pt/a_bar reaches ~200 there (~4e4 in variance).
    #
    # Same mechanism §10.6-1 priced for the NULL slots on run 86 (2.14 % of the
    # Σ_eval diagonal, 82.9 % of points, "flatters us"). Nobody had priced the
    # live-but-degenerate ones. Mask: myworkspace/chi2/audit_a7_degenerate_mask.py.
    #
    # HOW TO READ IT, fixed before the job ran (deliverable_tape_audit.md §6.0-bis):
    #   1. VETO  JEFF and JENDL at 0.00000 % — the mask is This_work only.
    #   2. VETO  V2 identical to four decimals — V2 carries no evaluated cov.
    #   3. V4 can only RISE (parameters removed => less declared σ). Against
    #      what: those slots carry 44.62 % of MF34's absolute variance, and MF34
    #      is only part of Σ_eval.
    #        rise ~ that share      -> the fold reconstructs what we built;
    #                                  P1/P2 are a PUBLICATION issue and the
    #                                  headline -23.8 % stands.
    #        rise >> that share     -> the fold reconstructs something we did
    #                                  not build; the excess is the (a_pt/a_bar)^2
    #                                  artefact and the headline CANNOT be
    #                                  written into the thesis unrepaired.
    #        ambiguous              -> run mf34_p1p2_mask_r2analytic.npz too
    #                                  (504 slots, 46.26 %).
    #
    # ⛔ In none of the three cases does the masked version ship. Removing
    # directions without measured structure in exchange is what disqualified the
    # MF34 split (§6e).
    "predictive_r2analytic_nodeg": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_r2analytic_nodeg.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — analytic joint (R2), the 201 degenerate MF34 slots removed: an EXPOSURE measurement",
        "systematic_block_col": None,
    },
    # The closing half of the same exposure. `nodeg` removed P1 (201 slots whose
    # group mean of a_l cancels) and cost only +2.67 % on no_Cierjacks -- the
    # degenerate groups are 1-3 keV wide, so few data points sit in them and the
    # local (a_pt/a_bar)^2 blow-up does not reach the total. This adds P2: the
    # 382 slots where the group mean averages ONLY the fine bins valid at that
    # order while MF4 declares a_l on the whole interval, so the denominator is
    # not reconstructable from the file at all (a_5 median 26 % off, max x24).
    #
    # ⚠ P2 is NOT a smaller P1. Its slots are not narrow -- 173/400 of a_4 and
    # 130/237 of a_5 -- so it can matter per-point even though it adds only
    # 1.64 points of variance share (44.62 % -> 46.26 %). That is exactly why it
    # is measured rather than inferred from `nodeg`.
    #
    # Same reading rules as `nodeg`: JEFF/JENDL at 0.00000 %, V2 identical, and
    # V4 can only rise. ⛔ The masked version never ships.
    "predictive_r2analytic_nodegp2": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_r2analytic_nodegp2.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — analytic joint (R2), P1+P2 MF34 slots removed: the closing EXPOSURE measurement",
        "systematic_block_col": None,
    },
    # ── ITEM 6: Cierjacks' backward points CORRECTED (roadmap §10.8-14) ──────
    #
    # Same evaluation, same Σ_eval, same everything — only the DATA move. The
    # parquets are written by `myworkspace/chi2/correct_cierjacks_backward.py`
    # from `chi2_data_predictive_91_cross.parquet`, and they must be scored
    # with KIKA_CHI2_EVAL_COV pointing at that file's existing 11 GB sidecar:
    # Σ_eval never sees y_exp, so re-precomputing would spend an hour and 11 GB
    # to reproduce it byte for byte.
    #
    # y_exp and sigma_exp_stat are multiplied by 1/(1+A) beyond 90°; the
    # RELATIVE systematics are untouched, so Cierjacks keeps its declared 5 %
    # normalization and 7 % ERR-T exactly. No global rescale (k = 1): the
    # forward-support magnitude agreement of §10.8-12 1c (+4.3 % against 5 %
    # declared, null control 2.2 %) pins k to ~1, and spending it would buy
    # shape amplitude at the price of that agreement.
    #
    # ⚠ THE CORRECTION IS APPLIED TO ALL THREE LIBRARIES' ROWS, because it is a
    # correction to the measurement. JEFF's and JENDL's χ² move too, and that is
    # the property that makes it non-self-serving. Report raw AND corrected.
    #
    # ⚠ `no_Cierjacks` MUST come back EXACTLY unchanged — it excludes every row
    # this touches. If it moves, the wrong rows were modified. That is the
    # control, and it is free.
    #
    # A = 0.30 is where the angle-integrated σ_el lands on the corpus consensus
    # with no rescale (six determinations, median 0.275); it delivers 31 % of
    # §10.8-10's measured Δa₁. A = 0.55 is the dose-response point: 46 % of the
    # displacement, and it takes the integral ~5 % BELOW the consensus.
    "predictive_91_cj030": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_91_cj030.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — run 91 + Cierjacks backward corrected, A = 0.30",
        "systematic_block_col": None,
    },
    "predictive_91_cj055": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_91_cj055.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — run 91 + Cierjacks backward corrected, A = 0.55",
        "systematic_block_col": None,
    },
    # ── ITEM 7: the cross-term dose-response (roadmap §10.8-5 step 4) ────────
    #
    # KIKA_MF33_MF34_CROSS_SCALE damps the whole cross block by s ∈ [0, 1] — a
    # convex combination of the joint and its block diagonal, hence PSD wherever
    # the joint is. ⚠ It is read inside `precompute_chi2_predictive.py`, so each
    # s needs its OWN precompute and its own 11 GB sidecar; the roadmap's
    # "re-scoring only" meant "no re-evaluation", not "no precompute".
    #
    # The endpoints are already measured: s = 0 is 91_rewrite (no_Cierjacks V4
    # 6.308) and s = 1 is 91_cross (8.444). One interior point therefore tests
    # linearity. LINEAR PREDICTION AT s = 0.5: 7.376. If the measurement comes
    # back materially above that, the chain amplifies and §10.8-1's certificate
    # has to be re-read before +33.9 % is quoted again.
    "predictive_91_cross_s50": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_91_cross_s50.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — cross term damped to s = 0.50",
        "systematic_block_col": None,
    },
    "predictive_91_cross_s25": {
        "parquet":    "/share_snc/snc/JuanMonleon/chi2/chi2_data_predictive_91_cross_s25.parquet",
        "report_dir": "/share_snc/snc/JuanMonleon/CHI_Figures/chi2_predictive",
        "title":      "χ² analysis — cross term damped to s = 0.25",
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
# docs/thesis/thesis_chi2_review.md before acting on either claim.
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

# Subsets — six slices of the dataset, y NINGUNA es la titular (Juan,
# 2026-08-21). Cierjacks 1978 aporta el 61 % de los puntos y es un conjunto
# dificil para todas las evaluaciones modernas de Fe-56, asi que se mira aislado
# y excluido; eso son dos vistas del mismo problema, no una recomendacion de
# cual reportar.
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
    # ⚑ A change confined to the DATA columns -- y_exp, sigma_exp_stat, the
    # relative systematics -- leaves Sigma_eval byte-identical, because
    # Sigma_eval is folded from (mu, E, c_0, a_l) and never sees y_exp. Such a
    # re-score therefore needs no precompute and no new 11 GB sidecar: point the
    # entry's parquet at the modified copy and this variable at the sidecar the
    # unmodified parquet already has. Roadmap §10.8-12 Phase 2 and §10.8-14.
    #
    # ⚠ It applies to EVERY methodology in one invocation, so run exactly one.
    override = os.environ.get("KIKA_CHI2_EVAL_COV", "").strip()
    if override:
        if len(METHODOLOGIES_TO_RUN) != 1:
            raise SystemExit(
                f"KIKA_CHI2_EVAL_COV is set but {len(METHODOLOGIES_TO_RUN)} "
                f"methodologies are selected ({METHODOLOGIES_TO_RUN}). One "
                f"sidecar cannot be right for several parquets — run them "
                f"one at a time."
            )
        eval_cov_path = Path(override)
        print(f"[EVAL_COV] overridden by KIKA_CHI2_EVAL_COV: {eval_cov_path}")
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
    # ⛔ NO SE RECOMIENDA NINGUN SUBCONJUNTO COMO TITULAR (Juan, 2026-08-21, y
    # reiterado el 23-ago). Esta linea decia «Recommended thesis headline
    # subset: no_Cierjacks» y de ahi salia la estrellita en todas las tablas
    # copiadas a los docs.
    #
    # Razon: marcar un subconjunto como titular fija la lectura de TODOS los
    # deltas que se reportan, y elegir el que mejor sale es la primera critica
    # que hace un referee. Ademas NINGUN subconjunto es fuera-de-muestra para
    # las tres evaluaciones a la vez -- JEFF y JENDL ajustaron Kinney <= 2,5 MeV
    # y Smith 2,5-4 MeV, y este trabajo ajusto TODO EXFOR -- asi que no hay uno
    # que se gane el papel por construccion.
    headline.append(
        f"\n**Los seis subconjuntos se presentan sin destacar ninguno.** No hay "
        f"un subconjunto titular: ninguno es fuera-de-muestra para las tres "
        f"evaluaciones a la vez (JEFF y JENDL ajustaron Kinney ≤ 2,5 MeV y Smith "
        f"2,5–4 MeV; este trabajo ajustó todo EXFOR), asi que destacar uno fija "
        f"la lectura de los deltas sin justificacion. Cierjacks 1978 aporta el "
        f"~61 % de los puntos y por eso aparece aislado en `only_Cierjacks` y "
        f"excluido en `no_Cierjacks`: son dos vistas, no una recomendacion.\n"
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
