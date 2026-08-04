"""
RESEARCH FORK of ``exfor_to_endf_sampling_v2.py`` — model-order treatment.

============================================================================
``exfor_to_endf_sampling_v2.py`` IS THE THESIS PIPELINE AND IS FROZEN.
It is never edited for research. This file is where research changes live.
============================================================================

Forked from v2 at its run-83 state (2026-07-31), which is the corrected
evaluation: run 83 read the declared EXFOR EN-RSL* resolutions, and its
``nominal_fits.parquet`` came out byte-identical to run 82's, so this fork's
fitted baseline is the same one the thesis rests on.

Why a fork and not a mode switch in v2: v2 is ~5250 lines and its covariance
combine is one long inline block interleaving ``KW_MC_INJECT``, PSD repair, dead
masks and near-zero flooring. Threading a second covariance construction through
it while guaranteeing the legacy path stays bit-reproducible is harder than
forking, and it puts a finished thesis result one bad merge away from changing.

What this fork changes (Phase 2 — the central value only)
---------------------------------------------------------
The production pipeline scores candidate Legendre degrees by AIC and then keeps
**only the winner's** coefficients as the central value. Gate 2 measured that
this throws away real evidence: median winning weight 0.505, a runner-up above
25 % in 58 % of bins, ~3.25 effectively supported models per bin.

1. ``DEGREE_WEIGHT_FLOOR`` — the arbitrary 1 % candidate cutoff becomes a knob,
   defaulting to 0.0 (keep every feasible candidate).
2. ``MC_CAP_FROM_SUPPORT_ONLY`` — the MC order cap comes from angular support
   rather than being bounded by whichever model happened to win.
3. Model-averaged central value and inclusion probabilities are computed and
   written as **new columns** in ``nominal_fits.parquet``.

What this fork deliberately does NOT change
-------------------------------------------
* **The ENDF output.** ``nominal_coeffs`` is never reassigned — the shipped MF4
  and MF34 are bit-identical to run 83's. The averaged central is a diagnostic
  column, nothing more. Runs 80/82/83 stay comparable.
* **The near-zero regularization.** ``REGULARIZE_NEAR_ZERO_REL_UNC`` stays ON.
  In the production config ``APPLY_COV_POSTPROCESSING`` and
  ``APPLY_COVARIANCE_CAP`` are both False, so ``ORDER_REL_STD_CAPS`` and every
  other ceiling are switched off and this guard is the *only* active protection
  against relative-σ blow-up. It also runs separately on the multigroup branch,
  where grouping pulls sign-changing coefficients toward zero and so makes the
  exposure worse, not better. Model averaging aggravates this by construction —
  ``ā_l = p·m`` glides through zero across a region instead of being present or
  absent. Relaxing it needs its own gate, measured in absolute DCS space.
* **``scripts/resample_AD.py`` and ``scripts/exfor_utils.py``**, which stay
  shared with v2. Everything here is computed from ``all_degrees_info``, which
  the fitter already returns.

The τ-IRLS solver mismatch (Gate 1b) is NOT fixed here. It lives in the shared
``resample_AD.py``, so fixing it needs an opt-in keyword rather than an in-place
edit, and it changes the covariance — it lands as its own change and its own run
so the τ shift stays attributable.

----------------------------------------------------------------------------

This script generates N samples of ENDF files by fitting Legendre coefficients
to EXFOR experimental angular distribution data at each energy point in the
original ENDF's energy grid, then replacing the original coefficients with
sampled values.

This is the migrated version that uses:
- kika.exfor module for EXFOR data loading (read_all_exfor)
- kika.exfor.transforms for frame conversions
- kika.endf.writers for MF34 creation

The workflow:
1. Read the reference ENDF file and extract the MF4 Legendre energy grid
2. For each energy bin in the specified range, compute adaptive tolerance
   based on neighboring energy points
3. Load EXFOR data within tolerance and fit Legendre polynomials
4. Generate N samples of coefficients using deterministic seeds for energy correlation
5. Create N output ENDF files with the sampled coefficients
6. Handle missing data by interpolating from neighboring bins

Per-point uncertainty decomposition
-----------------------------------
Per-point σ_stat (GLS fit weights, χ²) and per-experiment σ_sys (MC factor,
Kish ρ) come from ``scripts/uncertainty_manifest.py`` applied to each EXFOR
dataset at the cache-build boundary (``build_exfor_cache_from_objects``).
For datasets whose manifest stat-spec carries ``derive_stat_only: true`` —
i.e. the prose says the named column is a TOTAL that already includes the
correlated systematic (e.g. Barnard 30076004 DATA-ERR including the 5%
detector-efficiency systematic) — the resolver returns
    σ_stat = sqrt(max(σ_total² − σ_sys², 0))    per point,
so the per-experiment MC factor layered on top reproduces σ_total² without
double-counting. See ``uncertainty_manifest.py:resolve_for_dataset``.

Author: Generated for kika project
"""
from __future__ import annotations

import os
import sys

# Pin BLAS / threadpool libraries to a single thread per worker process BEFORE
# numpy/scipy/pandas are imported. With Pool(N_PROCS=40), each worker would
# otherwise spawn its own multi-threaded BLAS, oversubscribing the CPUs by
# 4–8× and cratering throughput. Use setdefault so users can still override
# from the environment (e.g. OMP_NUM_THREADS=2 python … for hybrid runs).
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import hashlib
import json
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from multiprocessing import Pool
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Add kika to path if needed (from scripts/ directory, kika is parent)
_kika_path = Path(__file__).parent.parent
if str(_kika_path) not in sys.path:
    sys.path.insert(0, str(_kika_path))

# Import kika modules - ENDF reading
from kika.endf.read_endf import read_endf
from kika.endf.classes.mf4.polynomial import MF4MTLegendre
from kika.endf.classes.mf4.mixed import MF4MTMixed

# Import kika modules - NEW EXFOR module (replaces AD_utils.load_all_exfor_data)
from kika.exfor import read_all_exfor

# Import kika modules - MF34 from library (replaces local implementation)
from kika.endf.writers import create_mf34_from_covariance, write_mf34_to_file, merge_mf34
from kika.endf.writers import (
    create_mf33_from_covariance,
    merge_mf33_covariance_into_host,
    write_mf33_to_file,
)

# Import local utility module (uses relative import from scripts package)
from scripts.exfor_utils import (
    # Logging
    DualLogger,
    _get_logger,
    _set_logger,
    _format_condensed_experiments,
    # Data classes
    EnergyBinInfo,
    SamplingResult,
    KernelDiagnostics,
    KWDiagnostics,
    compute_kw_diagnostics,
    compute_kw_reliability_alpha,
    # Energy binning
    compute_energy_bins_with_tof_resolution,
    build_union_energy_grid,
    remap_samples_to_endf_indices,
    # EXFOR data conversion (new API -> legacy format)
    build_exfor_cache_from_objects,
    # EXFOR filtering
    filter_exfor_with_energy_bin,
    # Covariance
    compute_covariance_from_samples,
    extract_ll_prime_correlations,
    build_gaussian_correlation_covariance,
    build_gaussian_relevance_matrix,
    _extract_correlation_matrix,
    inject_within_bin_correlations,
    log_psd_diagnostics,
    psd_repair_correlation_active,
    threshold_small_correlations,
    compute_bin_reliability_alpha,
    generate_cholesky_samples,
    cap_covariance_relative_uncertainty,
    absolute_to_nominal_relative,
    regularize_near_zero_relative_covariance,
    apply_between_experiment_floor,
    apply_between_experiment_floor_mg,
    smooth_absent_order_uncertainties,
    log_rel_std_profile,
    cap_order_relative_uncertainty,
    smooth_diagonal_median,
    forward_fill_rel_std,
    save_all_legendre_coefficients,
    # Kernel-weight MC (new)
    precompute_overlap_weights,
    run_mc_with_kernel_weights,
    stack_samples_to_matrix,
    build_mf33_channel,
    contiguous_grid_from_bins,
    recentre_relative_covariance,
    # ENDF writing
    write_nominal_endf,
    write_average_endf,
    write_endf_samples_batch,
    write_endf_sample,
    _write_sample_wrapper,
)

# Import kika sampling modules for ACE generation and MF34-based sampling
from kika.sampling.endf_perturbation import _process_njoy_for_sample, perturb_ENDF_files
from kika.sampling.reprocess_endf_to_ace import _ace_file_exists
from kika.sampling.utils import _set_logger as _set_kika_logger

# Import multigroup collapse module
from scripts.multigroup_collapse import (
    perform_adaptive_multigroup_collapse,
    try_merge_adjacent_multigroups,
    MultigroupResult,
)

# Import TOF parameters module (Improvement 1.4)
from scripts.mf33_diagnostics import (
    FOUR_PI as _MF33_FOUR_PI,
    bin_average_xs,
    fold_host_mf3_at_points,
    folded_c0_comparison_stats,
    log_folded_comparison,
)
from scripts.mf33_build import (
    build_mf33_denominator,
    build_mf33_matrices,
    write_mf33_products,
)
from scripts.tof_parameters import (
    load_tof_parameters_file,
    get_tof_parameters,
    compute_sigma_E,
    summarize_tof_parameters,
)

# Import resample_AD functions (relative import from scripts package)
from scripts.resample_AD import (
    sample_legendre_coefficients,
    endf_normalize_legendre_coeffs,
    compute_angular_band_discrepancy,
    sigma_eff_from_tau,
    smooth_tau_in_energy,
    apply_tau_prior_floor,
    compute_n_eff,
    compute_angular_support_diagnostics,
    compute_between_experiment_coeffs,
)

# RESEARCH FORK: the Phase-1 pure module. No file I/O, no global config, no
# pipeline imports — the maths lives there and is unit-tested there (24 tests),
# so this file only does the plumbing.
from scripts.model_averaging import (
    ic_weights,
    stack_padded,
    nested_masks,
    inclusion_probabilities,
    mixture_moments,
    conditional_mean,
    effective_n_models,
)

import time


# =============================================================================
# CONFIGURATION
# =============================================================================

# --- PATHS & I/O ----------------------------------------------------------- #
ENDF_FILE = "/share_snc/snc/JuanMonleon/jeff40_with_MF4_from_jeff33/26-Fe-56g.txt"
MF34_SOURCE_FILE = None                          # Separate MF34 source (None = use ENDF_FILE)
EXFOR_DIRECTORY = "/share_snc/snc/JuanMonleon/EXFOR/data_v1/"
EXFOR_DB_PATH = '/share_snc/snc/JuanMonleon/EXFOR/x4_iron_angular.db'
# Research fork output. Never point this at a NEW_FIT_8x directory — those are
# the v2 lineage and must stay reproducible.
#
# Correcting the note inherited from v2 at the fork point, which said sigma_E
# "feeds compute_overlap_weight, so which datasets constrain which bin moves and
# this is a re-evaluation, not a re-scoring": measured 2026-07-31, that is wrong
# for the nominal fit. compute_overlap_weight is reachable only from the MC
# covariance stage, so run 83's nominal_fits.parquet came out byte-identical to
# run 82's. Run 83 changed the covariance, not the central values.
# Writes straight to the share rather than /SCRATCH, so there is no copy step
# and the result is readable from WSL the moment the job ends. The v2 lineage
# keeps writing to /SCRATCH — this is a fork-only choice.
#
# ⚠ Fine for a STOP_AFTER_NOMINAL_FITS run, which emits ~1.5 MB (parquet + log).
# A FULL research run is a different matter: ~843 MB ENDF + ~690 MB TMC parquet
# + ~190 MB multigroup ENDF, written over a network mount that was at 91 % use
# (763 GB free) on 2026-07-31. Point KIKA_OUTPUT_DIR back at /SCRATCH for those,
# then copy, rather than streaming ~1.7 GB to the share through the job.
def _env_flag(name: str, default: bool) -> bool:
    """Boolean from the environment. Accepts 1/0, true/false, yes/no (any case).

    Unset → ``default``. An unrecognised value is a hard error rather than a
    silent fallback: a full research run costs ~5 h, and discovering afterwards
    that KIKA_MC_CAP_FROM_SUPPORT_ONLY=False was read as "unset, use True" would
    invalidate the whole run.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name}={raw!r} is not a recognised boolean")


OUTPUT_DIR = os.environ.get(
    "KIKA_OUTPUT_DIR",
    "/share_snc/snc/JuanMonleon/ENDF_samples/NEW_FIT_RESEARCH/",
)
TOF_PARAMETERS_FILE = "/share_snc/snc/JuanMonleon/EXFOR/exfor_tof_parameters.json"

# --- DATA SOURCE ----------------------------------------------------------- #
EXFOR_SOURCE = "database"                        # "json", "database", "auto", or "both"
TARGET_ZAIDS = [26056, 26000]                    # Fe-56 + natural iron
TARGET_PROJECTILE = "N"                          # Projectile (N for neutrons)
SUPPLEMENTARY_JSON_FILES = [                     # Extra JSON files loaded alongside main source
    '/share_snc/snc/JuanMonleon/EXFOR/data_v1/27673002.json',
]

# --- ENERGY RANGE & PHYSICS ------------------------------------------------ #
ENERGY_MIN_MEV = 0.847
ENERGY_MAX_MEV = 4
MT_NUMBER = 2                                    # MT reaction (2 = elastic scattering)
M_PROJ_U = 1.008665                              # Projectile mass in u (neutron)
M_TARG_U = 55.93494                              # Target mass in u (Fe-56)

# --- LEGENDRE FITTING ------------------------------------------------------ #
MAX_LEGENDRE_DEGREE = 6                          # Maximum Legendre order (capped at 8)
SELECT_DEGREE = "aic"                            # "aicc" | "aic" | "bic" | None (use max).
                                                  # "aic" drops the small-sample correction
                                                  # (penalty = 2k); use it when AICc over-
                                                  # penalises higher orders in low-n bins
                                                  # (e.g. single-experiment 8-pt bins where
                                                  # AICc's 2k(k+1)/(n−k−1) term ≈ 30 at k=5).
RIDGE_LAMBDA = 1e-4                              # Ridge regularization parameter
RIDGE_POWER = 4                                  # Power for ridge penalty (l^ridge_power)
DF_METHOD = "hat"                                # Degrees of freedom: "hat" or "naive"

# --- UNCERTAINTY & DISCREPANCY --------------------------------------------- #
# Band-based discrepancy
USE_BAND_DISCREPANCY = True                      # Use band-based uncertainty (vs global Birge)
MIN_POINTS_PER_BAND = 4                          # Min points to estimate s_b per band
                                                  # (5 sits at the median per-bin support; 4 captures
                                                  # the typical bin's left-edge without losing MAD's
                                                  # statistical meaning. Threshold 6+ collapses M-band
                                                  # derivation on this Fe-56 dataset.)
MAX_BAND_SCALE_FACTOR = 5.0                      # Max multiplicative scale per band
TAU_SMOOTHING_WINDOW = 1                         # Moving median window for s_b(E) (1 = disabled)
TAU_PRIOR_FLOOR = True                           # Apply tau prior floor from well-supported bins
TAU_PRIOR_NEFF_THRESHOLD = 5.0                   # Min N_eff to count as "well-supported" for the
                                                  # tau prior floor. Stays above the N_eff<3 zone
                                                  # where single-experiment residual collapse biases
                                                  # τ down, and gives a robust donor/recipient split
                                                  # on this dataset.
TAU_PRIOR_PERCENTILE = 50                        # Percentile of well-supported tau for baseline
TAU_IRLS_MAX_ITERS = 20                           # Max (τ, refit) iterations for band discrepancy
TAU_IRLS_TOL = 1e-2                              # Converge when max |Δτ_b| < tol
TAU_IRLS_DAMPING = 0.5                           # Geometric-mean damping α∈[0,1) on τ updates;
                                                  #  0 = no damping (legacy), 0.5 = half-step in
                                                  #  log-τ (breaks MAD limit cycles on bimodal bands).
BAND_SCALE_METHOD = "mad"                        # 'mad' (default) | 'rms' | 'hybrid' = max(MAD, RMS).
                                                  #  'rms' and 'hybrid' are research options.
SIGMA_SYS_AWARE_FIT = True                       # Fit weights = 1/σ_total² with σ_total² = σ_stat² +
                                                  #  σ_sys² (per-row, point-wise quadrature). τ is
                                                  #  computed against σ_total too, so it only absorbs
                                                  #  scatter beyond what stat *and* sys predict.
                                                  #  σ_eff returned is still τ·σ_stat (downstream
                                                  #  semantics unchanged). Set False for legacy
                                                  #  σ_stat-only behaviour.
RESCALE_UNC_BY_CHI2 = True                       # Apply Birge scaling when band discrepancy disabled
ALLOW_SHRINK_UNC = False                          # Allow uncertainties to shrink (chi2_red < 1)
# Normalization model (two physically distinct sources of multiplicative noise):
#   COMMON-MODE: uncertainty in a shared reference / monitor that ALL experiments
#                rely on. One factor per MC sample, applied globally to every data
#                point across every experiment (fully correlated → MF33 floor).
#   SYSTEMATIC:  per-experiment calibration / setup uncertainty. One factor per
#                experiment per MC sample (superseded by the manifest sigma_sys).
#                Also drives the Kish ρ for ESS collapse in the nominal fit.
NORM_COMMON_MODE_SIGMA = 0.0                      # Common-mode normalization uncertainty (0 = off).
                                                  # ONE global factor applied to ALL experiments together
                                                  # (shared monitor/reference lineage) — the fully-correlated
                                                  # floor of MF33 that between-experiment scatter can't see.
                                                  # 0 = experiment normalizations treated as independent.
NORM_SYSTEMATIC_SIGMA = 0.05                      # Per-experiment systematic (fallback only): the manifest's
                                                  # per-dataset sigma_sys supersedes this whenever present.
NORM_DIST = "lognormal"                          # "lognormal" (always positive) or "normal"
# Experiment exclusion & uncertainty floor
EXCLUDE_EXPERIMENTS = ["32246002", "400750022"]   # Tostkii 1957; Morozov 1972 pointer 2 (SPA-fitted sister of 400750021, double-counts raw data)
MIN_STAT_RELATIVE_UNCERTAINTY = 1e-4             # Numerical guard against literal-zero σ_stat
                                                  # (which would give infinite GLS weights). Set
                                                  # well below the smallest credible reported
                                                  # statistical uncertainty so that prose-stated
                                                  # values pass through unchanged: Korzh 1972
                                                  # ERR-S=0.7%, Cierjacks per-point ERR-S median
                                                  # ~0.05%, Tomita DATA-ERR ~0.5%. The legitimate
                                                  # "residual decomposition" floor is enforced at
                                                  # the resolver level (SIGMA_STAT_MIN_REL=0.01 in
                                                  # uncertainty_manifest.py) for Case B entries
                                                  # only — that's where σ_stat = √(σ_total² −
                                                  # σ_sys²) can saturate when the prose-named sys
                                                  # exceeds the lab's reported total. Set to 0
                                                  # to disable entirely (risks div-by-zero in GLS
                                                  # if any loader produces σ_stat=0 silently).
MIN_RELATIVE_UNCERTAINTY = MIN_STAT_RELATIVE_UNCERTAINTY  # backwards-compat alias
UNCERTAINTY_FLOOR_STRATEGY = 'bin_median'        # 'fixed' | 'bin_median' | 'band_median'
                                                  #  'band_median' uses per-F/M/B median with
                                                  #  fallback to bin median when <3 trustworthy
                                                  #  in-band points (research option).

# --- MODEL AVERAGING ------------------------------------------------------- #
USE_MODEL_AVERAGING = True                       # Enable model averaging over Legendre orders
                                                  # NOTE: a misnomer inherited from v2. The nominal
                                                  # central value is still the WINNING degree's
                                                  # coefficients; this flag only feeds MC degree
                                                  # draws. WRITE_AVERAGING_DIAGNOSTICS below is what
                                                  # actually computes an average.
MIN_DEGREE_FOR_AVERAGING = 1                     # Min degree to consider (1 = include all)
USE_DEGREE_SAMPLING_IN_MC = True                 # Sample degree from degree_weights in MC

# --- RESEARCH FORK: model-order treatment (Phase 2) ------------------------- #
# Minimum IC weight a candidate degree must carry to stay in the set. v2 uses a
# hard-coded 1 %, which is arbitrary — it discards candidates the evidence does
# support, and the threshold was never justified. 0.0 keeps every feasible
# candidate; set to 0.01 to reproduce v2 exactly.
#
# Env-overridable (KIKA_DEGREE_WEIGHT_FLOOR) so a single-variable run can pin it
# back to v2's 0.01 without editing this file. This knob is NOT inert in a full
# run: degree_weights feeds the MC degree sampling, hence the covariance.
DEGREE_WEIGHT_FLOOR = float(os.environ.get("KIKA_DEGREE_WEIGHT_FLOOR", "0.0"))

# Where the MC-only order cap comes from. v2 uses
# ``min(frozen_degree, recommended_mc_order)``, so the ceiling is bounded by
# whichever model won the IC scan. The ceiling should come from how much angular
# support the data actually provides — which is already computed — not from a
# model-selection outcome. True = support only; False = v2 behaviour.
#
# WARNING: this affects the MC, hence the covariance and the ENDF. It is inert
# under STOP_AFTER_NOMINAL_FITS (the MC never runs), which is how Phase 2's
# first run is done. The first FULL research run is where it starts to bite.
#
# Env-overridable (KIKA_MC_CAP_FROM_SUPPORT_ONLY=0) so a full run that is meant
# to isolate ONE change can pin it back to v2 behaviour.
MC_CAP_FROM_SUPPORT_ONLY = _env_flag("KIKA_MC_CAP_FROM_SUPPORT_ONLY", True)

# Compute the model-averaged central value, inclusion probabilities and the
# conditional mean, and write them as EXTRA columns in nominal_fits.parquet.
# The shipped MF4/MF34 are untouched — this is a diagnostic, and the whole point
# of Phase 2 is to look at the averaged central before deciding whether to adopt
# it. Costs a few ms per bin.
WRITE_AVERAGING_DIAGNOSTICS = True

# --- PHASE 3: the per-bin mixture covariance -------------------------------- #
# The evaluation currently reports model-degree uncertainty as exactly zero.
# Two independent mechanisms cause that, and the second is the dominant one:
#
#   1. The MC draws ONE degree per sample from degree_weights and pools the
#      results, so a_6 (median inclusion probability 0.089) is reconstructed
#      from the ~9 % of samples that happened to draw a degree >= 6.
#   2. The valid-parameter mask keys on nr.frozen_degree — the WINNER's degree —
#      and hard-zeroes every order above it, in BOTH covariance paths. Measured
#      on the tau-GLS run: the mask keeps 62.6 % of slots, and 2738 slots (26 %
#      of all) are zeroed despite an inclusion probability above 0.10. a_5
#      carries a median q of 0.293 and is zeroed in 73 % of bins.
#
# Phase 3 replaces both with the law of total covariance over the same IC
# weights that produce the averaged central:
#
#   V_mix = sum_L w_L V_L  +  sum_L w_L (mu_L - mu_bar)(mu_L - mu_bar)^T
#           \___within___/     \_____________between_____________________/
#
# so every candidate contributes to every order, zero-padded. The between term
# is the model-selection uncertainty that has never been in the file.
#
# SCOPE: per-bin blocks only. Cross-bin correlations still come from the Pass-1
# KW MC and are rescaled through the existing congruence
# cov = corr_pass1 * outer(std, std). A cross-energy mixture is the NEXT phase.
USE_MIXTURE_COVARIANCE = _env_flag("KIKA_USE_MIXTURE_COVARIANCE", True)
                                                  # False must reproduce the previous run bit-for-bit
                                                  # (the legacy rng.choice degree draw is kept intact
                                                  # for exactly that reason) — Gate A of the plan.

MIXTURE_MIN_SAMPLES_PER_MODEL = int(os.environ.get(
    "KIKA_MIXTURE_MIN_SAMPLES_PER_MODEL", "500"))
                                                  # Floor on the per-candidate batch size. This is a
                                                  # NON-SINGULARITY floor, not a precision one: the
                                                  # relative error on V_L is sqrt(2/n_L) and it enters
                                                  # multiplied by w_L, so a low-weight candidate needs
                                                  # few samples precisely because it barely counts.
                                                  # That is why the mixture is ~5-10 % more MC than
                                                  # today's flat N_SAMPLES, not 3-6x. Set 0 to use
                                                  # pure proportional allocation (Gate B).

MIXTURE_Q_MASK_THRESHOLD = float(os.environ.get(
    "KIKA_MIXTURE_Q_MASK_THRESHOLD", "0.01"))
                                                  # Replaces the frozen_degree mask: an order is a
                                                  # valid parameter when its inclusion probability
                                                  # q_l exceeds this. Deliberately LOW — the near-zero
                                                  # guard and absolute_to_nominal_relative's real 1e-6
                                                  # cutoff already drop numerically-empty parameters,
                                                  # and a second competing mechanism would make it
                                                  # impossible to attribute a dropped parameter to
                                                  # either one.
                                                  # ⚠ Valid slots go 62.6 % -> ~100 %, so the FINE
                                                  # MF34 grows roughly 1.6x (~734 MB -> ~1.2 GB). The
                                                  # multigroup file is the shipped product and scales
                                                  # the same way from a much smaller base.

MIXTURE_SPLICE_WITHIN_BIN_CORR = _env_flag(
    "KIKA_MIXTURE_SPLICE_WITHIN_BIN_CORR", False)
                                                  # Whether to force the mixture's within-bin
                                                  # CORRELATION blocks into the Pass-1 scaffold, on
                                                  # top of supplying the variances.
                                                  #
                                                  # Default False, and the reason is measured, not
                                                  # aesthetic. Pass-1 is a rank-<=n_samples estimate
                                                  # over 10428 parameters, so splicing full-rank
                                                  # blocks into it makes the matrix badly indefinite:
                                                  # the 300-sample smoke test hit min_eig -4.9 /
                                                  # neg_mass -5.2e3, and a sample/parameter sweep
                                                  # puts a 10000-sample run at min_eig ~-0.8 /
                                                  # neg_mass ~-1e3. That is NOT a low-sample artifact
                                                  # that disappears at production scale.
                                                  #
                                                  # A Higham repair of that much negative mass does
                                                  # not preserve the spliced blocks — it moves the
                                                  # whole matrix to restore PSD, so the result is
                                                  # neither Pass-1 nor the mixture, after hours of
                                                  # wall clock. The plain congruence
                                                  # D_mix * Corr_pass1 * D_mix is PSD by
                                                  # construction, needs no repair, and ships the
                                                  # mixture VARIANCES exactly.
                                                  #
                                                  # What that costs, stated plainly: Pass-1 fits at
                                                  # the frozen winner degree, so it has no
                                                  # information about within-bin cross-order
                                                  # correlations ABOVE that degree, and those are
                                                  # shipped as ~0. Conservative and honest. The real
                                                  # fix is degree sampling in Pass-1, which is a
                                                  # bigger change than Phase 3.
SHIP_MIXTURE_MEAN = _env_flag("KIKA_SHIP_MIXTURE_MEAN", True)
                                                  # MF4 central = the mixture mean, so MF4 and MF34
                                                  # describe ONE distribution. The between-model term
                                                  # is defined relative to the mixture mean; centring
                                                  # it on the winner would ship a mean and a
                                                  # covariance from different distributions.
                                                  # The winner's coefficients are preserved in
                                                  # nominal_fits.parquet as win_c_0..win_c_6 so every
                                                  # earlier run stays comparable.

# Handed to _mc_one_bin as its 30th arg. None is the legacy signal, so the
# whole Phase-3 path is switched by the presence of this object rather than by
# a flag read inside the worker — a worker that never sees the config cannot
# accidentally take the new branch.
#
# Note the mixture only forms where the bin actually has competing candidates
# (`use_degree_sampling` needs len(degree_weights) > 1). A single-candidate bin
# falls through to the legacy pooled covariance, which for one degree IS V_L —
# the same answer, so nothing is lost.
_MIXTURE_CFG = (
    {'min_samples_per_model': MIXTURE_MIN_SAMPLES_PER_MODEL}
    if USE_MIXTURE_COVARIANCE else None
)

RERUN_AICC_POST_TAU = True                       # Recompute AICc weights against τ-inflated σ_eff
                                                  # before sampling, so the degree distribution sent
                                                  # to MC reflects the τ-converged noise model
                                                  # (matches v3). False reverts to pre-τ stat-only
                                                  # weights — biases degree sampling for any bin
                                                  # where τ inflated σ noticeably.
USE_GLS_KERNEL = True                            # GLS kernel (block-correlated, Σ = D + u uᵀ + v vᵀ) for the
                                                  # IC model-selection scan, the initial nominal fit,
                                                  # and the post-τ rescan. Applies regardless of
                                                  # SELECT_DEGREE choice (aicc/aic/bic). freeze_c0
                                                  # refit and MC sampler stay diagonal; the τ-IRLS
                                                  # refit follows TAU_REFIT_USE_GLS below.
TAU_REFIT_USE_GLS = True                          # RESEARCH FORK: True. v2 leaves this at the
                                                  # library default False, which refits inside the
                                                  # τ-IRLS loop under diagonal WLS — σ_sys folded
                                                  # into the fit weights as if uncorrelated. Since
                                                  # that loop's coeffs0 IS the shipped nominal
                                                  # (the post-τ rescan replaces it only when the
                                                  # winning *degree* changes), v2 ships a WLS
                                                  # central whose degree was chosen by GLS AICc
                                                  # scores. τ is active in 98.4 % of run-83 bins
                                                  # (median τ_M 2.07), so this is not a corner case.
                                                  # True makes the refit use the same GLS kernel.

# --- ENERGY BINNING & CORRELATION ------------------------------------------ #
# Weighting and constraints
NORMALIZE_BY_N_POINTS = True                     # Enable study-level GLS-ESS weighting
BAND_AWARE_ESS = False                           # Kish ESS uses per-band counts (F/M/B)
MAX_EXP_WEIGHT_FRAC_BIN = 0.8                    # Safety cap per experiment weight fraction
FREEZE_C0 = True                                 # Freeze c0 in MC/KW shape refits (nominal AICc fit
                                                  # must float c0 to determine it; MC then refits
                                                  # a_l = c_l/c0 with c0 fixed at the nominal).
MAX_SAMPLE_ORDER = 6                             # Publish covariance for l=1..MAX_SAMPLE_ORDER only
# Angular quality gate
ANGULAR_QUALITY_GATE = True                      # Matches run77. A no-op on this database (every bin
                                                  # in the 0.847-4 MeV grid already passes the angular
                                                  # coverage check, so no bin is expanded/interpolated)
MIN_ANGULAR_POINTS = 4                           # Min total angular data points
MIN_BANDS_COVERED = 3                            # Must have data in all 3 bands (F/M/B)
MAX_BIN_EXPANSION = 3                            # Max expansion steps (1=+-1 bins, 2=+-2, etc.)
MEMBERSHIP_K_SIGMA = 0.0                         # Widen WHICH datasets may constrain a bin to
                                                  # target +- k*sigma_E (unioned with the bin edges).
                                                  # 0.0 = hard bin edges, i.e. runs <= 82.
                                                  #
                                                  # Motivation: sigma_E is 1.7-8.6 keV here against a
                                                  # 1 keV grid, so a bin-width window renews ~89% of
                                                  # the point set from one bin to the next, and an
                                                  # experiment with coarse resolution constrains a
                                                  # single 1 keV bin instead of the region it actually
                                                  # measured. This is a MEMBERSHIP knob only: the
                                                  # selected point per experiment is still the nearest
                                                  # to target and still carries weight 1.0. Applying
                                                  # Gaussian overlap WEIGHTS instead would convolve a
                                                  # second time (the data are already resolution-
                                                  # folded) and cost sqrt(2)*sigma_E of resolution.
# Energy grid source
ENERGY_GRID_SOURCE = "union"                     # "endf" (MF4 grid) or "union" (from EXFOR subentries)
UNION_GRID_SUBENTRIES = [                        # (subentry, min_MeV, max_MeV)
    ("10571002", 0.847, 2.5),                    # Kinney: up to 2.5 MeV
    ("23365005", 2.5, 4),                        # Pirovano: from 2.5 MeV onwards
]
# Correlation method
# "gaussian"         - Per-bin stochastic MC -> Gaussian parametric correlations -> Cholesky
# "kernel_weight_mc" - Kernel-weighted multi-bin MC -> correlations from shared perturbations
# "hybrid"           - KW two-pass + Gaussian blend weighted by per-bin reliability
CORRELATION_METHOD = "kernel_weight_mc"
KW_MC_TWO_PASS = True                            # True: per-bin variance + KW correlations
KW_MC_INJECT = False                             # False (default): congruence transform Cov = D*Corr_pass1*D
                                                  #   - PSD by construction; consistent with calibrated parquet.
                                                  # True: legacy splice (Pass-1 cross-bin + Pass-2 within-bin)
                                                  #   followed by Higham nearest-PSD repair. Kept for research
                                                  #   comparison; not PSD-preserving by construction.
KW_MC_MIN_WEIGHT = 1e-3                          # Overlap weight threshold
KW_MIN_POINTS_REF = None                         # Quality penalty threshold (set to max_order+1 at runtime)
# TOF energy resolution
DELTA_T_NS = 5.0                                 # Time resolution in nanoseconds
FLIGHT_PATH_M = 27.037                           # Flight path in meters
DELTA_T_IS_FWHM = True                           # Whether DELTA_T_NS (and the per-experiment
                                                  # time_resolution values in TOF_PARAMETERS_FILE)
                                                  # are FWHM rather than sigma. Timing spreads are
                                                  # normally quoted as FWHM — exfor_tof_parameters
                                                  # records e.g. "neutron_pulse_width_FWHM" — and
                                                  # sigma_E = FWHM/2.3548 accordingly.
                                                  #
                                                  # THIS CHANGES RESULTS. sigma_E sets how far each
                                                  # experimental point spreads across bins via the
                                                  # overlap weights, so it moves n_eff, the tau
                                                  # bands, degree selection and hence MF4/MF34/MF33.
                                                  # Runs <= 81 were produced with False; set False
                                                  # to reproduce them. Individual subentries may
                                                  # override via "is_fwhm" in the TOF JSON.
N_SIGMA_CUTOFF = 3.0                             # Gaussian kernel cutoff (+-n_sigma * sigma_E)

# --- COVARIANCE PIPELINE --------------------------------------------------- #
# Pre-processing: cap & near-zero regularization
APPLY_COVARIANCE_CAP = False                     # Global rel_std cap (True to enable)
MAX_RELATIVE_STD_CAP = 1.0                       # Max relative std when cap is enabled
REGULARIZE_NEAR_ZERO_REL_UNC = True              # Regularize explosive rel_std near zero means
NEAR_ZERO_SNR_THRESHOLD = 1.0                    # Flag if |mean|/sigma_abs < threshold
NEAR_ZERO_N_NEIGHBORS = 3                        # Valid neighbors to seek on each side
APPLY_BETWEEN_EXP_FLOOR = False                 # Between-experiment scatter floor (disabled: redundant with band discrepancy)
# Post-processing pipeline (set APPLY_COV_POSTPROCESSING=False to skip all)
APPLY_COV_POSTPROCESSING = False
SMOOTH_MIN_REL_STD = 0.005                       # Entries below this treated as absent (0.5%)
SMOOTH_DIP_FRACTION = 0.50                       # Flag dips < fraction*median (None = disabled)
SMOOTH_SPIKE_FACTOR = 3.0                        # Flag spikes > factor*median (None = disabled)
SMOOTH_DIP_N_NEIGHBORS = 3                       # Neighbors each side (fine-grid level)
SMOOTH_MEDIAN_FILL_THRESHOLD = 0.50              # If > fraction absent, use flat median fill
MG_SMOOTH_SPIKE_FACTOR = 2.0                     # Tighter spike threshold at MG level
MG_SMOOTH_DIP_N_NEIGHBORS = 5                    # Wider neighborhood at MG level
SMOOTH_DIAGONAL_WINDOW = 0                       # Gaussian kernel window (0=disabled, >=3 to enable)
ORDER_REL_STD_CAPS = {1: 0.50, 2: 0.50, 3: 0.40, 4: 0.35, 5: 0.25, 6: 0.20}  # Hard cap per order (None=disabled)
FORWARD_FILL_REL_STD_ENABLED = False             # Propagate last valid rel_std into absent bins
# Positivity projection
APPLY_POSITIVITY_PROJECTION = True               # Project MC samples for non-negative distributions
POSITIVITY_CHECK_POINTS = 101                    # Number of mu points in [-1, 1]

# --- MULTIGROUP COVARIANCE ------------------------------------------------- #
GENERATE_MULTIGROUP_COVARIANCE = True            # Enable adaptive multigroup collapse
MULTIGROUP_RHO_MIN = 0.85                        # Min l=1 adjacent correlation to merge groups
MULTIGROUP_SIGMA_RATIO_MAX = 5.0                 # Max running max(σ_l1)/min(σ_l1) within a group.
                                                  # Prevents merging strongly correlated bins whose l=1
                                                  # std differ by more than this factor (heterogeneity
                                                  # would force the percentile compensation to over-
                                                  # inflate group variance). Set to None to disable.
MULTIGROUP_USE_RAW_MC_CORR = True                # Feed multigroup collapse with raw KW correlations + Pass-2 std,
                                                  # bypassing the inject + Higham-smear path. No effect when
                                                  # KW_MC_TWO_PASS=False. See plan: ok-i-watn-you-floofy-hickey.md
MULTIGROUP_CORRELATION_THRESHOLD = 0        # Hard-zero |rho| < threshold in the multigroup correlation matrix.
                                                  # Set to 0.0 to disable, "auto" to use 1/sqrt(N_SAMPLES) (sampling-
                                                  # noise floor), or a positive float for an explicit threshold.
MF34_COVARIANCE_TYPE = "both"              # "fine", "multigroup", or "both"
USE_ORIGINAL_MF34_GRID = False                   # Force grid from original MF34
MERGE_ORIGINAL_MF34 = True                       # Merge pipeline MF34 with original (full range)
MULTIGROUP_VARIANCE_PCT_MIN = 67                 # Base percentile for homogeneous groups
MULTIGROUP_VARIANCE_PCT_MAX = 85                 # Max percentile for heterogeneous groups
MULTIGROUP_VARIANCE_RATIO_REF = 5.0              # Sigma ratio at which percentile saturates
MULTIGROUP_REGROUP_AFTER_SMOOTH = False          # Second-pass regrouping after smoothing

# --- MF33 ELASTIC MAGNITUDE CHANNEL ---------------------------------------- #
# Record the fixed-shape c0 (sigma_el = 4*pi*c0) and write its MF33 covariance.
# Read-only: MF4/MF34 are unchanged. MF3 itself is deliberately NOT written
# (despite the knob name, which records the eventual intent) — the central stays
# at the host value and the DCS magnitude 4*pi*c0 is saved as a sidecar. When on,
# MF33 goes into BOTH products: the nominal file on the fine MF4 grid, and the
# _mg file on its own adaptive grid (see MF33_MULTIGROUP_* below).
GENERATE_MF3_MF33 = 1                            # 0 = off, 1 = on

# The MF33 relative covariance is C_abs / sigma_host^2, so it needs a host
# central per bin. Two things make that non-trivial and both are handled here:
#
#   1. RESOLUTION. The fitted c0 is what a detector with a finite TOF resolution
#      measured (sigma_E = 4-41 keV here), not a box average over the 1 keV bin.
#      The denominator is folded through the same kernel — see fold_xs_over_bins.
#   2. RESONANCE RANGE. File 3 carries only the smooth background inside a
#      resolved resonance range, so MF3 MT2 is identically ZERO below the RRR
#      upper bound (850 keV for JEFF-4.0 Fe-56, while this grid starts at
#      846.8 keV). The denominator therefore comes from the RECONR-reconstructed
#      PENDF, not from File 3 — correct for any target whose RRR reaches into
#      the analysis window, and it costs one cached NJOY run.
#
# RECONR output is 0 K; BROADR is unnecessary because Doppler widths (~eV) are
# negligible against a 4-41 keV resolution kernel.
MF33_PENDF_TOLERANCE = 0.001                     # RECONR linearization tolerance
MF33_PENDF_CACHE_DIR = None                      # None -> kika default temp cache
                                                  # (keyed on ENDF sha256 + tolerance,
                                                  #  so repeat runs are free)

# Adaptive multigroup grid for MF33. Independent of the MF34 grid: the magnitude
# channel has its own correlation structure, and ENDF lets each MF/MT section
# carry its own grid. Defaults match the MF34 knobs so the first run is directly
# comparable.
MF33_MULTIGROUP_RHO_MIN = 0.85                   # Min adjacent c0 correlation to merge
MF33_MULTIGROUP_SIGMA_RATIO_MAX = 5.0            # Max intra-group max(sigma)/min(sigma)
MF33_MG_REPRESENTATION = "fine"                  # What MF33 the _mg file carries:
                                                  # "multigroup" -> the collapsed grid
                                                  # (as MF34), "fine" -> the full MF4-grid
                                                  # matrix while MF34 stays collapsed.
                                                  # MF34 is what makes that file large and
                                                  # genuinely needs grouping; the MF33
                                                  # collapse is irreversible and damped the
                                                  # peak elastic sigma 19.3% -> 12.2% on
                                                  # run 82. Grouping can always be done
                                                  # afterwards; un-grouping cannot.
MF33_REBUILD_MT1 = True                          # Rebuild MF33 MT1 (total) over the analysis
                                                  # window as the sandwich over the partials,
                                                  # cross terms zeroed, so the file's total
                                                  # stops contradicting its own elastic. The
                                                  # host ships MT1 within 1.4% of the
                                                  # uncorrelated partial sum, so this keeps
                                                  # the evaluator's own convention.
SAVE_MF33_MULTIGROUP_DIAGNOSTICS_CSV = True      # mf33_boundary_decisions.csv
COMPUTE_MF33_MF34_CROSS = True                   # Measure Cov(c0, a_l) across the shared Pass-2 MC
                                                  # replicas and write mf33_mf34_cross_*.npy
                                                  # (roadmap §9.4.1). DIAGNOSTIC ONLY — it is not fed
                                                  # into any covariance the ENDF ships, so turning it
                                                  # on cannot move a single number in the file. It
                                                  # exists to answer the one thing the Cauchy-Schwarz
                                                  # bound cannot: the SIGN of the magnitude-shape
                                                  # correlation. Costs no MC time; both channels are
                                                  # already in memory at that point.
                                                  # Publishing it as an MF34 LB=6 (L=0,L1) block
                                                  # stays gated on mf3_mf33_roadmap Phase 3 / LTT=3.
SAVE_MF33_C0_SAMPLES = True                      # mf33_c0_samples.parquet — the raw
                                                  # two-pass c0 draws (~300 MB at 10k
                                                  # samples).
                                                  # TURNED ON 2026-08-04 (roadmap §10.1.6).
                                                  # It was False through run 86, and that is
                                                  # the ONLY reason run 86's cross-energy
                                                  # magnitude<->shape structure cannot be
                                                  # recovered offline. Without these draws a
                                                  # complete cross block cannot be rebuilt
                                                  # after the fact — the covariance sidecars
                                                  # summarise each channel separately and
                                                  # throw the pairing away. ~300 MB against a
                                                  # share that sits near full: still worth it.

# --- OUTPUT: Pipeline A (fitting) ------------------------------------------ #
N_SAMPLES = int(os.environ.get("KIKA_N_SAMPLES", "10000"))
                                                  # Number of MC samples. Env-overridable so the
                                                  # FULL chain (MC -> covariance -> MF34 -> ENDF)
                                                  # can be smoke-tested in minutes at a few hundred
                                                  # samples before a ~5 h production run. A low
                                                  # value is NOT a valid evaluation — the covariance
                                                  # is badly under-sampled — it only proves the
                                                  # pipeline runs end to end. Anything published
                                                  # must be at the default.
BASE_SEED = 42                                   # Random seed for reproducibility
GENERATE_NOMINAL_ENDF = True                     # Best-fit coefficients ENDF
GENERATE_MC_MEAN_ENDF = False                    # MC mean coefficients ENDF
GENERATE_FITTING_SAMPLES = False                 # Individual MC sample ENDFs -> endf_direct/
GENERATE_FITTING_ACE = False                     # Generate ACE files for fitting samples
SAVE_COVARIANCE_FILES = False                    # Save fine + multigroup .npy files
                                                  # (cov, mean, boundaries). Redundant with MF34
                                                  # but convenient for analysis notebooks.
SAVE_CORRELATION_MATRICES = False                # Save correlation alongside covariance
                                                  # (only honored if SAVE_COVARIANCE_FILES=True).
# --- TMC sample parquets (Pipeline A outputs) ----
# Three independent representations of the MC samples can be emitted; only
# enable what your downstream consumer actually needs.
SAVE_TMC_PARQUET = True                          # legendre_samples_tmc.parquet
                                                  # — Pass-1 Pearson correlations × Pass-2 marginals
                                                  # (preserves non-Gaussian dependence). Recommended
                                                  # TMC input; consistent with the published MF34.
SAVE_RAW_KW_PARQUET = False                      # legendre_samples_raw_kw.parquet
                                                  # — raw KW Pass-1 samples (Pass-1 marginals).
                                                  # Research/diagnostic only.
SAVE_MULTIGROUP_DIAGNOSTICS_CSV = True           # multigroup_boundary_decisions.csv — small
                                                  # diagnostic with per-pair merge decisions;
                                                  # useful for tuning rho_min / sigma_ratio_max.
STOP_AFTER_NOMINAL_FITS = _env_flag("KIKA_STOP_AFTER_NOMINAL_FITS", True)
                                                  # RESEARCH FORK: True, not v2's False.
                                                  # Env-overridable (KIKA_STOP_AFTER_NOMINAL_FITS=0)
                                                  # so a full run is declared in the runner rather
                                                  # than by editing this file.
                                                  # Exit after Step 4 (~2 min) instead of running the
                                                  # MC (~5 h). Phase 2 only produces diagnostic
                                                  # COLUMNS in nominal_fits.parquet, which Step 4
                                                  # writes — so the MC would cost 5 h and add
                                                  # nothing. Stopping here also means no covariance
                                                  # is built, no ENDF is written, and the near-zero
                                                  # guard is never reached, so this run cannot
                                                  # perturb anything.
                                                  # Set False only for a FULL research run — and note
                                                  # MC_CAP_FROM_SUPPORT_ONLY starts biting there.
SAVE_NOMINAL_FITS = True                         # nominal_fits.parquet — per-bin c_0..c_L, chi2_red,
                                                  # frozen_degree, tau bands, AICc weights.

# --- OUTPUT: Pipeline B (MF34 sampling) ------------------------------------ #
GENERATE_MF34_SAMPLES = False                    # Perturbed ENDF samples from MF34 -> endf/
GENERATE_MF34_ACE = False                        # Generate ACE files for MF34 samples
SAMPLING_RESOLUTION = "multigroup"               # "fine" or "multigroup"
SAMPLING_SPACE = "linear"                        # "linear" or "log"
SAMPLING_DECOMPOSITION = "svd"                   # "svd", "cholesky", "eigen", "pca"
SAMPLING_METHOD = "random"                       # "sobol", "lhs", "random"

# --- ACE / NJOY ----------------------------------------------------------- #
ACE_TEMPERATURES = [293.6]                       # Temperature(s) in Kelvin
ACE_NJOY_EXE = "/soft_snc/NJOY/2016.78/bin/njoy"
ACE_LIBRARY_NAME = "jeff40"                      # Library name (e.g., 'endfb81', 'jeff40')
ACE_NJOY_VERSION = "NJOY 2016.78"               # NJOY version string
ACE_XSDIR_FILE = "/share_snc/snc/JuanMonleon/xsdir_MCNPy/xsdir40-irdff2"
ACE_SKIP_EXISTING = False                        # Skip samples with existing ACE files

# --- RUNTIME --------------------------------------------------------------- #
N_PROCS = 24                                     # Parallel processes (1 = sequential)
                                                  # Matches --cpus-per-task=24 in run_pyscript.sh.
                                                  # v2 has 40; keep these two in step or the pool
                                                  # oversubscribes the allocation.
N_EFF_WARNING_THRESHOLD = 5.0                    # Warning if effective sample size < threshold
VERBOSE_DIAGNOSTICS = True                       # Per-order percentile stats at every pipeline stage

# =============================================================================
# END OF CONFIGURATION
# =============================================================================


# Global logger reference (set by _set_logger from exfor_utils)
_logger = None


# =============================================================================
# RUN METADATA (audit trail)
# =============================================================================


def _sha256_of_file(path: str) -> Optional[str]:
    """Return hex SHA256 of a file, or None if unreadable."""
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _git_info(repo_dir: str) -> Dict[str, Any]:
    """Best-effort git commit + dirty status for the repo at repo_dir."""
    info: Dict[str, Any] = {'commit': None, 'dirty': None, 'branch': None}
    try:
        commit = subprocess.run(
            ['git', '-C', repo_dir, 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5,
        )
        if commit.returncode == 0:
            info['commit'] = commit.stdout.strip()
        branch = subprocess.run(
            ['git', '-C', repo_dir, 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, timeout=5,
        )
        if branch.returncode == 0:
            info['branch'] = branch.stdout.strip()
        status = subprocess.run(
            ['git', '-C', repo_dir, 'status', '--porcelain'],
            capture_output=True, text=True, timeout=5,
        )
        if status.returncode == 0:
            info['dirty'] = bool(status.stdout.strip())
    except Exception:
        pass
    return info


def _collect_config_constants() -> Dict[str, Any]:
    """Snapshot every ALL_CAPS module-level constant for the run audit."""
    g = globals()
    snapshot: Dict[str, Any] = {}
    for name, val in g.items():
        if not name.isupper():
            continue
        if name.startswith('_'):
            continue
        if callable(val):
            continue
        # Keep only JSON-friendly leaf values (str/int/float/bool/None/list/dict).
        try:
            json.dumps(val)
        except (TypeError, ValueError):
            snapshot[name] = repr(val)
            continue
        snapshot[name] = val
    return snapshot


def _write_run_metadata(output_dir: str) -> Dict[str, Any]:
    """
    Write OUTPUT_DIR/run_metadata.json capturing config + git + script/manifest hashes.

    Returns the metadata dict so the caller can log a short summary.
    """
    script_path = os.path.abspath(__file__)
    repo_dir = str(Path(__file__).resolve().parent.parent)

    # Resolve manifest path. Importing the module here avoids hard-coding the
    # path; if the manifest module isn't importable, build_exfor_cache_from_objects
    # will raise loudly anyway, so a None here is safe metadata.
    manifest_path: Optional[str] = None
    manifest_sha256: Optional[str] = None
    manifest_path_reachable: Optional[bool] = None
    try:
        from scripts.uncertainty_manifest import _MANIFEST_PATH as _mp
        manifest_path = str(_mp)
    except Exception:
        try:
            from uncertainty_manifest import _MANIFEST_PATH as _mp
            manifest_path = str(_mp)
        except Exception:
            manifest_path = None
    if manifest_path is not None:
        manifest_path_reachable = os.path.exists(manifest_path)
        if manifest_path_reachable:
            manifest_sha256 = _sha256_of_file(manifest_path)

    metadata: Dict[str, Any] = {
        'started_at': datetime.now().isoformat(),
        'script_path': script_path,
        'script_sha256': _sha256_of_file(script_path),
        'manifest_path': manifest_path,
        'manifest_path_reachable': manifest_path_reachable,
        'manifest_sha256': manifest_sha256,
        'git': _git_info(repo_dir),
        'config': _collect_config_constants(),
    }

    out = Path(output_dir) / 'run_metadata.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(metadata, f, indent=2, default=str, sort_keys=True)
    return metadata


# =============================================================================
# PARALLEL HELPERS (top-level for pickling)
# =============================================================================


def _discover_exfor_endf_samples(
    output_dir: str,
    n_samples: int,
    endf_file: str,
) -> List[str]:
    """
    Discover existing ENDF sample files written by Step 9.

    Used when generate_fitting_samples=False but generate_fitting_ace=True (reprocessing
    existing samples at new temperatures).

    Expected structure: {output_dir}/endf_direct/{sample_str}/{base}_{sample_str}.endf
    where sample_str is 4-digit zero-padded (1-based).

    Parameters
    ----------
    output_dir : str
        Base output directory containing the endf_direct/ subdirectory
    n_samples : int
        Number of samples expected
    endf_file : str
        Path to the original ENDF file (used for base filename)

    Returns
    -------
    List[str]
        List of discovered ENDF sample file paths, indexed by sample_index (0-based).
        Missing samples are represented as empty strings.
    """
    logger = _get_logger()
    base = Path(endf_file).stem
    endf_dir = Path(output_dir) / "endf_direct"

    if not endf_dir.exists():
        if logger:
            logger.warning(f"[ACE] [DISCOVERY] ENDF directory not found: {endf_dir}")
        return []

    discovered = []
    missing = []

    for sample_num in range(1, n_samples + 1):
        sample_str = f"{sample_num:04d}"
        sample_dir = endf_dir / sample_str
        expected = sample_dir / f"{base}_{sample_str}.endf"

        if expected.exists():
            discovered.append(str(expected))
        else:
            # Try any file matching the pattern in the sample directory
            found = False
            if sample_dir.is_dir():
                for f in sample_dir.iterdir():
                    if f.name.startswith(f"{base}_{sample_str}"):
                        discovered.append(str(f))
                        found = True
                        break
            if not found:
                discovered.append("")
                missing.append(str(expected))

    if missing and logger:
        logger.warning(f"[ACE] [DISCOVERY] {len(missing)}/{n_samples} expected ENDF files not found")
        for m in missing[:5]:
            logger.warning(f"[ACE] [DISCOVERY]   Missing: {m}")
        if len(missing) > 5:
            logger.warning(f"[ACE] [DISCOVERY]   ... and {len(missing) - 5} more")

    n_found = sum(1 for f in discovered if f)
    if logger:
        logger.info(f"[ACE] [DISCOVERY] Found {n_found}/{n_samples} ENDF sample files")

    return discovered


def _ace_worker(args):
    """
    Process a single ENDF sample through NJOY to generate ACE files.

    Top-level function (picklable for Pool.map), follows existing _mc_one_bin() pattern.

    Parameters
    ----------
    args : tuple
        (endf_file, sample_index, temperatures, output_dir, njoy_exe,
         library_name, njoy_version, xsdir_file, skip_existing, zaid)

    Returns
    -------
    tuple
        (sample_index, result_dict) where result_dict has keys:
        'success' (bool), 'temperatures_processed' (list), 'errors' (list),
        'skipped' (list)
    """
    (endf_file, sample_index, temperatures, output_dir, njoy_exe,
     library_name, njoy_version, xsdir_file, skip_existing, zaid) = args

    sample_str = f"{sample_index + 1:04d}"

    # Filter temperatures if skip_existing is enabled
    temps_to_process = []
    skipped_temps = []

    if skip_existing and zaid is not None:
        for temp in temperatures:
            if _ace_file_exists(output_dir, zaid, sample_index, temp, endf_file):
                skipped_temps.append(temp)
            else:
                temps_to_process.append(temp)
    else:
        temps_to_process = list(temperatures)

    if not temps_to_process:
        return (sample_index, {
            "success": True,
            "temperatures_processed": [],
            "skipped": skipped_temps,
            "errors": [],
        })

    try:
        result = _process_njoy_for_sample(
            out_endf=endf_file,
            sample_index=sample_index,
            njoy_exe=njoy_exe,
            temperatures=temps_to_process,
            library_name=library_name,
            njoy_version=njoy_version,
            output_dir=output_dir,
            xsdir_file=xsdir_file,
        )
        result["skipped"] = skipped_temps
        return (sample_index, result)
    except Exception as e:
        return (sample_index, {
            "success": False,
            "temperatures_processed": [],
            "skipped": skipped_temps,
            "errors": [f"Sample {sample_str}: {e}"],
        })


def _stratified_counts(probs, n_total):
    """Split ``n_total`` slots across candidates in proportion to ``probs``.

    Largest-remainder apportionment: floor everything, then hand the leftover
    slots to the largest fractional parts. Guarantees ``sum(out) == n_total``
    exactly, which a multinomial draw does not.

    Used instead of ``rng.choice`` on the mixture path because the pooled slot
    counts should reflect the weights, not the sampling noise on top of them —
    the weights are the quantity we are propagating.
    """
    p = np.asarray(probs, dtype=float)
    raw = p * float(n_total)
    base = np.floor(raw).astype(int)
    short = int(n_total) - int(base.sum())
    if short > 0:
        order = np.argsort(-(raw - base), kind='stable')
        base[order[:short]] += 1
    elif short < 0:
        # Can only happen through floating-point slop in p.sum().
        order = np.argsort(raw - base, kind='stable')
        for j in order:
            if short == 0:
                break
            if base[j] > 0:
                base[j] -= 1
                short += 1
    return base


def bin_valid_orders(nr, max_degree):
    """How many leading Legendre orders count as real parameters for this bin.

    Legacy rule: ``min(frozen_degree, max_degree)`` — every order above the
    WINNER's degree is a hard zero in the covariance. That is winner-take-all
    surviving in the covariance after Phase 2 removed it from the central value,
    and it is the dominant reason high orders carry no uncertainty. Measured on
    the τ-GLS run: it keeps 62.6 % of slots and zeroes 2738 slots (26 % of all)
    whose inclusion probability exceeds 0.10; a_5 has a median q of 0.293 and is
    zeroed in 73 % of bins.

    Mixture rule: an order is a real parameter when the total IC weight of the
    candidates that contain it, ``q_l``, clears ``MIXTURE_Q_MASK_THRESHOLD``.
    ``q_l`` is monotone non-increasing in ``l`` for a nested family, so counting
    the ones above threshold is the same as taking the leading run.
    """
    if USE_MIXTURE_COVARIANCE and getattr(nr, 'degree_weights', None):
        from scripts.model_averaging import nested_masks, inclusion_probabilities
        degs = sorted(nr.degree_weights)
        w = np.array([float(nr.degree_weights[d]) for d in degs], dtype=float)
        if w.sum() > 0:
            q = inclusion_probabilities(w / w.sum(), nested_masks(degs, max_degree))
            return int(np.sum(q > MIXTURE_Q_MASK_THRESHOLD))
    return min(nr.frozen_degree, max_degree)


def build_mixture_blocks(mixture_by_bin, nominal_results, max_degree, logger=None):
    """Collapse per-candidate moments into one mixture covariance per bin.

    Parameters
    ----------
    mixture_by_bin
        ``{energy_index: {degree: {'n', 'mean', 'cov'}}}`` from ``_mc_one_bin``.
    nominal_results
        Carries ``degree_weights`` — the SAME IC weights that produce the
        averaged central, so MF4 and MF34 cannot drift apart.

    Returns
    -------
    (blocks, diag) where ``blocks`` is ``{energy_index: {'mean', 'cov'}}`` ready
    for ``compute_covariance_from_samples(mixture_blocks=...)`` and ``diag`` is a
    per-bin record of the within/between split.

    The within/between split is the whole diagnostic story of Phase 3: if
    ``between`` dominates at the high orders, model-selection uncertainty is
    appearing in the file for the first time. If ``within`` dominates, the
    mixture is mostly re-weighting and the claim is much weaker. Keep both.
    """
    from scripts.model_averaging import mixture_moments

    weights_by_idx = {
        nr.energy_index: (nr.degree_weights or {}) for nr in nominal_results
    }
    blocks, diag = {}, {}
    n_skipped = 0
    for e_idx, by_deg in (mixture_by_bin or {}).items():
        if not by_deg:
            continue
        w_map = weights_by_idx.get(e_idx) or {}
        degs = sorted(by_deg.keys())
        w = np.array([float(w_map.get(d, 0.0)) for d in degs], dtype=float)
        if not np.isfinite(w).all() or w.sum() <= 0:
            # No usable weights for a bin that produced batches: fall through to
            # the sampled covariance rather than inventing a uniform mixture.
            n_skipped += 1
            continue
        means = np.vstack([by_deg[d]['mean'] for d in degs])
        covs = np.stack([by_deg[d]['cov'] for d in degs])
        mm = mixture_moments(w, means, covs)
        blocks[e_idx] = {'mean': mm['mean'], 'cov': mm['total']}
        diag[e_idx] = {
            'n_models': len(degs),
            'within_var': np.diag(mm['within']).copy(),
            'between_var': np.diag(mm['between']).copy(),
            'total_var': np.diag(mm['total']).copy(),
        }
    if logger:
        if n_skipped:
            logger.warning(
                f"  [MIX] {n_skipped} bin(s) had candidate batches but no usable "
                f"degree weights — left on the sampled covariance."
            )
        if diag:
            wv = np.vstack([d['within_var'] for d in diag.values()])
            bv = np.vstack([d['between_var'] for d in diag.values()])
            frac = bv / np.maximum(wv + bv, 1e-300)
            logger.info(
                "  [MIX] between-model share of the variance, median by order: "
                + ", ".join(f"a_{l+1} {np.nanmedian(frac[:, l]):.3f}"
                            for l in range(min(max_degree, frac.shape[1])))
            )
    return blocks, diag


def compute_mf33_mf34_cross(
    all_samples_perbin,
    c0_samples_perbin,
    energy_indices,
    max_order,
    min_samples=32,
    compute_full=True,
):
    """Per-bin Cov(c0, a_l) over the shared Pass-2 MC replicas (roadmap §9.4.1).

    WHY THIS IS COMPUTABLE AT ALL. Our Sigma_eval today is
    Sigma^MF34 + Sigma^MF33 with the cross term set to exactly zero, because the
    evaluation splits one Monte Carlo into two one-at-a-time channels: the
    magnitude channel varies c0 at frozen shape (-> MF33), the shape channel
    refits a_l at frozen c0 (-> MF34). That partition, not any external MF3, is
    what discards Cov(sigma, a_l).

    But the two channels are NOT independent draws. In
    ``resample_AD.sample_legendre_coefficients`` a single ``Y_perturbed`` matrix
    is built once per batch; ``_batch_mc_ridge_solve`` turns row i into that
    replica's a_l, and ``fixed_shape_c0_scale`` turns THE SAME row i into that
    replica's c0. Row i is one perturbed dataset. So ``all_samples_perbin[s]``
    and ``c0_samples_perbin[s]`` are the same replica, and their covariance
    across s is a real measurement of the correlation the shared data induces --
    no joint evaluation, no relaxing FREEZE_C0, no LTT=3 needed to MEASURE it
    (publication is still gated by mf3_mf33_roadmap Phase 3; see roadmap §6.4).

    Freezing c0 inside the shape refit does not break this: a_l does not depend
    on the c0 *drawn* in replica s, but both respond to the same perturbed data,
    and that is precisely the channel the correlation travels down.

    WHAT IT RETURNS, AND WHY NOT A RELATIVE BLOCK. Absolute covariance and
    correlation, never Cov(c0/c0_nom, a_l/a_l_nom). The relative form divides by
    a_l_nom, which passes through zero -- the same near-zero blow-up
    ``REGULARIZE_NEAR_ZERO_REL_UNC`` exists to contain, and
    ``absolute_to_nominal_relative`` drops outright at |a_l| < 1e-6. rho is
    bounded in [-1, 1] by construction and is the quantity the bound in §9.4.1
    is expressed in, so it is the honest thing to report and to gate on.

    This is the "Level A" estimator: paired one-at-a-time draws. A fully joint
    refit (c0 and a_l floating together per replica) is Level B and is a
    different, larger change.

    Parameters
    ----------
    all_samples_perbin : dict {s_idx: {energy_idx: a_l array of len >= max_order}}
        Pass-2 shape draws, post freeze/positivity/normalise -- i.e. the
        coefficients the file actually ships.
    c0_samples_perbin : dict {s_idx: {energy_idx: float}}
        Pass-2 magnitude draws, same replica index.
    energy_indices : sequence of int
        Bin order for the returned rows.
    max_order : int
        Legendre orders 1..max_order.
    min_samples : int
        Bins with fewer paired replicas return NaN rather than a noisy rho.

    THE CROSS-ENERGY BLOCK, AND WHY IT IS NOT OPTIONAL (roadmap §10.1.5/§10.1.6).
    The within-bin numbers above are a *partial* Level A: Cov(c0(E_i), a_l(E_i))
    only. Shipping that against complete MF33/MF34 diagonals is **not a valid
    covariance**. Measured 2026-08-04 on run 81's paired replicas: the full joint
    sample covariance is PSD at -1e-18, and zeroing ONLY the cross-energy entries
    makes it non-PSD in every energy window, 13 orders of magnitude worse -- with
    those discarded entries the same size as the ones kept (ratio 0.82-1.01) and
    ~159x more numerous. That is what killed run 87. So ``cov_full`` is computed
    here, as (n_bins, n_bins, max_order) = Cov(c0(E_i), a_l(E_j)).

    IT IS NOT SYMMETRIC. Row = magnitude bin, column = shape bin; the two axes
    are different quantities. Do not transpose it into place.

    A COMMON REPLICA SET IS REQUIRED, and this is the one thing that would
    silently reintroduce the original bug. A sample covariance is PSD *only* if
    every entry is computed from the same replicas. Pairwise-complete covariance
    -- intersecting per (i, j) as the within-bin loop does per bin -- carries no
    PSD guarantee at all, which is precisely the failure mode this block exists
    to fix. So ``cov_full`` uses the replicas complete in BOTH channels across
    ALL bins, and reports how many that keeps in ``n_common``. The per-bin
    ``cov``/``rho`` above keep their per-bin intersection and are unchanged, so
    run 86's sidecars stay reproducible.

    Returns
    -------
    dict with 'cov', 'rho' (n_bins, max_order), 'std_c0' (n_bins,),
    'std_a' (n_bins, max_order), 'n_pairs' (n_bins,), and -- when
    ``compute_full`` -- 'cov_full' (n_bins, n_bins, max_order) plus the scalar
    'n_common'.
    """
    energy_indices = list(energy_indices)
    n_bins = len(energy_indices)
    cov = np.full((n_bins, max_order), np.nan, dtype=float)
    rho = np.full((n_bins, max_order), np.nan, dtype=float)
    std_a = np.full((n_bins, max_order), np.nan, dtype=float)
    std_c0 = np.full(n_bins, np.nan, dtype=float)
    n_pairs = np.zeros(n_bins, dtype=int)

    # Replica indices present in both channels. Intersected per bin: a bin can
    # fail in one channel and not the other, and pairing the wrong replicas
    # would manufacture correlation out of nothing.
    for row, e_idx in enumerate(energy_indices):
        c0_vals, a_rows = [], []
        for s_idx, per_bin in c0_samples_perbin.items():
            if e_idx not in per_bin:
                continue
            a_bin = all_samples_perbin.get(s_idx, {}).get(e_idx)
            if a_bin is None:
                continue
            c0_v = float(per_bin[e_idx])
            a_v = np.asarray(a_bin, dtype=float)[:max_order]
            if not np.isfinite(c0_v) or a_v.size < max_order or not np.all(np.isfinite(a_v)):
                continue
            c0_vals.append(c0_v)
            a_rows.append(a_v)

        n = len(c0_vals)
        n_pairs[row] = n
        if n < min_samples:
            continue

        c = np.asarray(c0_vals, dtype=float)
        A = np.vstack(a_rows)
        cc = c - c.mean()
        Ac = A - A.mean(axis=0)
        # ddof=1 to match the mixture blocks' convention (np.cov ddof=1).
        cov[row] = cc @ Ac / (n - 1)
        s_c = float(np.sqrt(cc @ cc / (n - 1)))
        s_a = np.sqrt(np.einsum("ij,ij->j", Ac, Ac) / (n - 1))
        std_c0[row] = s_c
        std_a[row] = s_a
        # Orders restored from nominal (frozen above mc_order_cap / the mixture
        # mask) have zero spread: rho is genuinely undefined there, not zero.
        # The threshold is RELATIVE, not `> 0`: subtracting the mean of a
        # constant column leaves ~1e-17 of rounding, which passes `> 0` and
        # yields rho ~ 4e-16 -- a confident-looking zero for a quantity that was
        # never measured. That would bias any median taken over orders, and the
        # frozen orders are exactly the ones Phase 3 changes the count of.
        _eps = 1e-12
        scale_a = np.maximum(np.max(np.abs(A), axis=0), np.finfo(float).tiny)
        scale_c = max(float(np.max(np.abs(c))), np.finfo(float).tiny)
        good = (s_a > _eps * scale_a) & (s_c > _eps * scale_c)
        rho[row, good] = cov[row, good] / (s_c * s_a[good])

    out = dict(cov=cov, rho=rho, std_c0=std_c0, std_a=std_a, n_pairs=n_pairs)
    if compute_full:
        out.update(_cross_full(
            all_samples_perbin, c0_samples_perbin, energy_indices, max_order,
            min_samples=min_samples,
        ))
    return out


def _cross_full(
    all_samples_perbin,
    c0_samples_perbin,
    energy_indices,
    max_order,
    min_samples=32,
):
    """Cov(c0(E_i), a_l(E_j)) over ONE common replica set. See the caller's
    docstring for why the common set is mandatory rather than convenient.

    Returns {'cov_full': (n_bins, n_bins, max_order), 'n_common': int}.
    ``cov_full`` is None when too few replicas are complete everywhere -- a
    partial block is exactly what we are trying to stop shipping, so it is
    withheld rather than filled.
    """
    energy_indices = list(energy_indices)
    n_bins = len(energy_indices)

    def _complete(s_idx):
        """Does replica s_idx have finite c0 AND a_1..a_max in every bin?"""
        c_bins = c0_samples_perbin.get(s_idx) or {}
        a_bins = all_samples_perbin.get(s_idx) or {}
        for e_idx in energy_indices:
            if e_idx not in c_bins or not np.isfinite(float(c_bins[e_idx])):
                return False
            a = a_bins.get(e_idx)
            if a is None:
                return False
            a = np.asarray(a, dtype=float)
            if a.size < max_order or not np.all(np.isfinite(a[:max_order])):
                return False
        return True

    common = sorted(s for s in c0_samples_perbin if _complete(s))
    n = len(common)
    if n < min_samples:
        return dict(cov_full=None, n_common=n)

    # Order-major so each A[l] is C-contiguous and the products below are plain
    # BLAS GEMMs. An einsum over (r, i, j, l) is the obvious spelling and is
    # several times slower here, which matters inside a ~6 h pipeline run.
    C0 = np.empty((n, n_bins), dtype=float)
    A = np.empty((max_order, n, n_bins), dtype=float)
    for r, s_idx in enumerate(common):
        c_bins = c0_samples_perbin[s_idx]
        a_bins = all_samples_perbin[s_idx]
        for col, e_idx in enumerate(energy_indices):
            C0[r, col] = float(c_bins[e_idx])
            A[:, r, col] = np.asarray(a_bins[e_idx], dtype=float)[:max_order]

    C0 -= C0.mean(axis=0)
    A -= A.mean(axis=1, keepdims=True)
    # ddof=1, matching the per-bin path and np.cov's convention.
    cov_full = np.empty((n_bins, n_bins, max_order), dtype=float)
    for l in range(max_order):
        cov_full[:, :, l] = (C0.T @ A[l]) / (n - 1)
    return dict(cov_full=cov_full, n_common=n)


def _mc_one_bin(args):
    """
    Run MC sampling for a single energy bin (top-level for Pool.map pickling).

    Returns
    -------
    tuple
        (energy_idx, is_interpolated, results_by_sample, success, error_msg)
        where results_by_sample is Dict[s_idx, np.ndarray] of ENDF coefficients
        and error_msg is None on success or the exception summary when the bin
        fell back to nominal coefficients (zero MC variance).
    """
    (
        nr_energy_idx,
        nr_frozen_degree,
        nr_nominal_coeffs,
        nr_interpolated,
        nr_mc_df,
        nr_mc_weights,
        nr_degree_weights,
        n_samples,
        base_seed,
        max_degree,
        ridge_lambda,
        ridge_power,
        df_method,
        use_band_discrepancy,
        min_points_per_band,
        max_band_scale,
        use_degree_sampling_in_mc,
        rescale_unc_by_chi2,
        allow_shrink_unc,
        freeze_c0,
        normalization_sigma,
        sigma_norm_common_mode,
        norm_dist,
        max_sample_order,
        _apply_positivity_projection,
        _positivity_check_points,
        frozen_tau_info,
        mc_order_cap,
    ) = args[:28]
    # Phase-2 magnitude channel: optional 29th field. Absent (28-tuple) → off,
    # so the Gaussian-method builder needs no change and behavior is identical.
    record_c0 = bool(args[28]) if len(args) > 28 else False
    # Phase-3 mixture: optional 30th field, a dict or None. Absent → legacy
    # pooled degree-sampling, byte-for-byte (Gate A depends on this staying a
    # pure no-op when the knob is off).
    #   {'min_samples_per_model': int}
    mixture_cfg = args[29] if len(args) > 29 else None
    # Per-candidate moments in ENDF a-space, filled only under the mixture path:
    #   {degree: {'n': int, 'mean': (max_degree,), 'cov': (max_degree, max_degree)}}
    mixture_by_degree: dict = {}
    # Per-sample fixed-shape c0 for this bin; empty unless recording.
    c0_by_sample: dict = {}

    energy_idx = nr_energy_idx

    if nr_interpolated:
        endf_coeffs = endf_normalize_legendre_coeffs(nr_nominal_coeffs, include_a0=False)
        results = {s_idx: endf_coeffs for s_idx in range(n_samples)}
        if record_c0 and len(nr_nominal_coeffs) > 0:
            c0_by_sample = {s_idx: float(nr_nominal_coeffs[0]) for s_idx in range(n_samples)}
        # Interpolated bins have no candidate models, so there is no mixture to
        # form. An empty dict here (not a fabricated single-model entry) is what
        # makes the caller fall back to the legacy covariance for this bin.
        return (energy_idx, True, results, True, None, c0_by_sample, {})

    bin_seed = base_seed + energy_idx
    rng = np.random.default_rng(bin_seed)
    mc_weights = nr_mc_weights

    # If frozen tau_info is available, pre-inflate uncertainties and disable
    # band discrepancy re-estimation so MC uses the smoothed/floored tau
    if frozen_tau_info and use_band_discrepancy:
        from scripts.resample_AD import sigma_eff_from_tau
        mu_arr = nr_mc_df['mu'].to_numpy()
        sigma_arr = nr_mc_df['unc'].to_numpy()
        inflated_unc = sigma_eff_from_tau(mu_arr, sigma_arr, frozen_tau_info)
        nr_mc_df = nr_mc_df.copy()
        nr_mc_df['unc'] = inflated_unc
        use_band_discrepancy = False  # already applied via frozen tau

    use_degree_sampling = (
        use_degree_sampling_in_mc and
        nr_degree_weights is not None and
        len(nr_degree_weights) > 1
    )

    # Combine global MAX_SAMPLE_ORDER with per-bin mc_order_cap
    effective_sample_order = max_sample_order
    if mc_order_cap is not None:
        if effective_sample_order is not None:
            effective_sample_order = min(effective_sample_order, mc_order_cap)
        else:
            effective_sample_order = mc_order_cap

    # Pass through Level-2 sys-aware fitting if the column survived the
    # nominal-stage data flow (perform_nominal_fits adds '_sigma_sys_abs').
    sys_unc_col_arg = (
        '_sigma_sys_abs' if '_sigma_sys_abs' in nr_mc_df.columns else None
    )

    results = {}
    try:
        if use_degree_sampling:
            degrees = list(nr_degree_weights.keys())
            probs = np.array(list(nr_degree_weights.values()))
            probs = probs / probs.sum()

            from scripts.resample_AD import (
                check_angular_distribution_positivity,
                project_to_positive_distribution,
            )

            if mixture_cfg is None:
                # LEGACY: draw all degrees at once, group by value for batched
                # fitting. Multinomial counts. Untouched so that
                # USE_MIXTURE_COVARIANCE=False reproduces earlier runs exactly.
                sampled_degrees = rng.choice(degrees, size=n_samples, p=probs)
                degree_groups = {}
                for s_idx, d in enumerate(sampled_degrees):
                    degree_groups.setdefault(int(d), []).append(s_idx)
                fit_counts = {deg: len(idx) for deg, idx in degree_groups.items()}
            else:
                # PHASE 3: stratified allocation instead of a multinomial draw.
                # The POOLED slots still total exactly n_samples and are still
                # split in proportion to w_L, so the pooled sample set that
                # feeds the correlations keeps today's meaning. The batches are
                # additionally topped up to the non-singularity floor, and the
                # extra rows are used ONLY for V_L — never assigned a pooled
                # slot, or the pooled set would stop being a mixture draw.
                pool_counts = _stratified_counts(probs, n_samples)
                degree_groups = {}
                fit_counts = {}
                _cursor = 0
                for deg, n_pool in zip(degrees, pool_counts):
                    deg = int(deg)
                    degree_groups[deg] = list(range(_cursor, _cursor + int(n_pool)))
                    _cursor += int(n_pool)
                    fit_counts[deg] = max(
                        int(n_pool),
                        int(mixture_cfg.get('min_samples_per_model', 0)),
                    )
                # A candidate can legitimately get zero pooled slots when
                # w_L * n_samples < 0.5; it still needs a batch, because its
                # between-model contribution w_L (mu_L - mu)(...)^T does not
                # vanish just because it drew no samples.
                degree_groups = {d: idx for d, idx in degree_groups.items()
                                 if fit_counts.get(d, 0) > 0}

            for deg, s_indices in degree_groups.items():
                n_batch = int(fit_counts[deg])
                coef_df_batch, info_batch = sample_legendre_coefficients(
                    nr_mc_df,
                    value_col="value",
                    unc_col="unc",
                    sys_unc_col=sys_unc_col_arg,
                    degree=deg,
                    max_degree=max_degree,
                    select_degree=None,
                    ridge_lambda=ridge_lambda,
                    ridge_power=ridge_power,
                    df_method=df_method,
                    external_weights=mc_weights if len(mc_weights) > 0 else None,
                    n_samples=n_batch,
                    stochastic=True,
                    rescale_unc_by_chi2=rescale_unc_by_chi2,
                    allow_shrink_unc=allow_shrink_unc,
                    random_state=bin_seed + deg * 1000,
                    use_band_discrepancy=use_band_discrepancy,
                    min_points_per_band=min_points_per_band,
                    max_band_scale=max_band_scale,
                    freeze_c0=freeze_c0,
                    sigma_norm=normalization_sigma,
                    sigma_norm_common_mode=sigma_norm_common_mode,
                    norm_dist=norm_dist,
                    max_sample_order=effective_sample_order,
                    record_c0_scale=record_c0,
                    c0_scale_ref_coeffs=nr_nominal_coeffs if record_c0 else None,
                )
                if record_c0 and info_batch.get("c0_samples") is not None:
                    _c0_batch = np.atleast_1d(info_batch["c0_samples"])
                    for local_i, s_idx in enumerate(s_indices):
                        c0_by_sample[s_idx] = float(_c0_batch[local_i])
                # Under the mixture path n_batch can exceed len(s_indices): the
                # surplus rows exist only to keep V_L non-singular. Every row is
                # carried through the identical freeze/positivity/normalise
                # pipeline so mu_L and V_L describe the coefficients the file
                # would actually ship, not the raw fit output.
                _batch_a = [] if mixture_cfg is not None else None
                for local_i in range(n_batch):
                    s_idx = s_indices[local_i] if local_i < len(s_indices) else None
                    sample_coeffs = coef_df_batch.iloc[local_i].to_numpy()
                    if len(sample_coeffs) < max_degree + 1:
                        sample_coeffs = np.pad(sample_coeffs, (0, max_degree + 1 - len(sample_coeffs)))
                    # Restore frozen orders from nominal values
                    if effective_sample_order is not None:
                        for l in range(effective_sample_order + 1, len(sample_coeffs)):
                            if l < len(nr_nominal_coeffs):
                                sample_coeffs[l] = nr_nominal_coeffs[l]
                    if _apply_positivity_projection:
                        if not check_angular_distribution_positivity(sample_coeffs, _positivity_check_points):
                            frozen = {}
                            if freeze_c0:
                                frozen[0] = sample_coeffs[0]
                            if effective_sample_order is not None and effective_sample_order + 1 < len(sample_coeffs):
                                frozen.update({i: sample_coeffs[i] for i in range(effective_sample_order + 1, len(sample_coeffs))})
                            sample_coeffs = project_to_positive_distribution(sample_coeffs, _positivity_check_points, frozen_indices=frozen or None)
                    endf_coeffs = endf_normalize_legendre_coeffs(sample_coeffs, include_a0=False)
                    if len(endf_coeffs) < max_degree:
                        _p = np.zeros(max_degree, dtype=float)
                        _p[:len(endf_coeffs)] = endf_coeffs
                        endf_coeffs = _p
                    if _batch_a is not None:
                        _batch_a.append(np.asarray(endf_coeffs, dtype=float))
                    if s_idx is not None:
                        results[s_idx] = endf_coeffs

                if _batch_a is not None and len(_batch_a) > 0:
                    _A = np.vstack(_batch_a)
                    mixture_by_degree[int(deg)] = {
                        'n': int(_A.shape[0]),
                        'mean': _A.mean(axis=0),
                        # ddof=1: V_L is an estimate of the within-model
                        # covariance, not the covariance of these exact draws.
                        # With the 500-sample floor the bias is negligible
                        # either way, but ddof=0 would make the identity test
                        # in test_mixture_covariance fail for the right reason.
                        'cov': (np.cov(_A, rowvar=False, ddof=1)
                                if _A.shape[0] > 1
                                else np.zeros((_A.shape[1], _A.shape[1]))),
                    }
        else:
            from scripts.resample_AD import (
                check_angular_distribution_positivity,
                project_to_positive_distribution,
            )

            coef_df, info_nd = sample_legendre_coefficients(
                nr_mc_df,
                value_col="value",
                unc_col="unc",
                sys_unc_col=sys_unc_col_arg,
                degree=nr_frozen_degree,
                max_degree=max_degree,
                select_degree=None,
                ridge_lambda=ridge_lambda,
                ridge_power=ridge_power,
                df_method=df_method,
                external_weights=mc_weights if len(mc_weights) > 0 else None,
                n_samples=n_samples,
                rescale_unc_by_chi2=rescale_unc_by_chi2,
                allow_shrink_unc=allow_shrink_unc,
                random_state=bin_seed,
                use_band_discrepancy=use_band_discrepancy,
                min_points_per_band=min_points_per_band,
                max_band_scale=max_band_scale,
                freeze_c0=freeze_c0,
                sigma_norm=normalization_sigma,
                sigma_norm_common_mode=sigma_norm_common_mode,
                norm_dist=norm_dist,
                max_sample_order=effective_sample_order,
                record_c0_scale=record_c0,
                c0_scale_ref_coeffs=nr_nominal_coeffs if record_c0 else None,
            )
            if record_c0 and info_nd.get("c0_samples") is not None:
                _c0_nd = np.atleast_1d(info_nd["c0_samples"])
                for s_idx in range(n_samples):
                    c0_by_sample[s_idx] = float(_c0_nd[s_idx])
            for s_idx in range(n_samples):
                sample_coeffs = coef_df.iloc[s_idx].to_numpy()
                if _apply_positivity_projection:
                    if not check_angular_distribution_positivity(sample_coeffs, _positivity_check_points):
                        frozen = {}
                        if freeze_c0:
                            frozen[0] = sample_coeffs[0]
                        if effective_sample_order is not None and effective_sample_order + 1 < len(sample_coeffs):
                            frozen.update({i: sample_coeffs[i] for i in range(effective_sample_order + 1, len(sample_coeffs))})
                        sample_coeffs = project_to_positive_distribution(sample_coeffs, _positivity_check_points, frozen_indices=frozen or None)
                endf_coeffs = endf_normalize_legendre_coeffs(sample_coeffs, include_a0=False)
                if len(endf_coeffs) < max_degree:
                    padded = np.zeros(max_degree, dtype=float)
                    padded[:len(endf_coeffs)] = endf_coeffs
                    endf_coeffs = padded
                results[s_idx] = endf_coeffs

        return (energy_idx, False, results, True, None, c0_by_sample,
                mixture_by_degree)

    except Exception as exc:
        endf_coeffs = endf_normalize_legendre_coeffs(nr_nominal_coeffs, include_a0=False)
        results = {s_idx: endf_coeffs for s_idx in range(n_samples)}
        c0_fallback = (
            {s_idx: float(nr_nominal_coeffs[0]) for s_idx in range(n_samples)}
            if record_c0 and len(nr_nominal_coeffs) > 0 else {}
        )
        # A bin that fell back to nominal has zero MC variance; emitting no
        # mixture keeps it on the legacy path rather than handing the caller a
        # degenerate V_mix that would look like a real (zero) uncertainty.
        return (energy_idx, False, results, False,
                f"{type(exc).__name__}: {exc}", c0_fallback, {})


def _log_mc_bin_failures(bin_results, nominal_results, warning_counts, logger):
    """Log every per-bin MC failure (nominal fallback = zero MC variance).

    A failed bin contributes identical (nominal) coefficients to all samples,
    so its Pass-2 variance is ~0 and the published MF34 silently understates
    the uncertainty there unless the failure is surfaced.
    """
    energy_by_idx = {nr.energy_index: nr.energy_mev for nr in nominal_results}
    n_failed = 0
    for rec in bin_results:
        energy_idx, _interp, _results, success, error_msg = rec[:5]
        if success:
            continue
        n_failed += 1
        e_mev = energy_by_idx.get(energy_idx)
        e_str = f"E={e_mev:.4f} MeV" if e_mev is not None else f"idx={energy_idx}"
        if logger:
            logger.error(
                f"[ERROR] [MC] Bin {e_str}: per-bin MC failed, all samples "
                f"fell back to nominal (zero MC variance) — {error_msg}",
                console=True,
            )
    if n_failed:
        warning_counts['mc_bin_failures'] = (
            warning_counts.get('mc_bin_failures', 0) + n_failed
        )


# =============================================================================
# WORKFLOW DATACLASSES
# =============================================================================

@dataclass
class NominalFitResult:
    """Result from nominal fitting at one energy."""
    energy_mev: float
    energy_index: int
    exfor_df: pd.DataFrame
    experiments_info: List[Dict]
    kernel_weights: np.ndarray
    frozen_degree: int
    nominal_coeffs: np.ndarray
    sigma_eff: np.ndarray
    tau_info: Dict[str, float]
    chi2_red: float
    has_data: bool = True
    interpolated: bool = False  # Whether coefficients were interpolated from neighbors
    kernel_diagnostics: Optional[KernelDiagnostics] = None
    degree_weights: Optional[Dict[int, float]] = None
    all_degrees_info: Optional[Dict[int, Dict]] = None
    # RESEARCH FORK: averaging diagnostics + data-space chi2, computed in the
    # fitting loop where mu/y/sigma_eff/kernel_weights are still in scope. The
    # parquet writer only reads this, so there is exactly one computation site.
    averaging_diag: Optional[Dict[str, float]] = None
    # PHASE 3: the winning degree's coefficients, kept when SHIP_MIXTURE_MEAN
    # replaces nominal_coeffs with the mixture mean. Serialised as win_c_* so
    # every earlier run stays directly comparable against this one — without it
    # the c_* columns silently change meaning between runs.
    winner_coeffs: Optional[np.ndarray] = None
    mc_order_cap: Optional[int] = None              # Adaptive MC-only order cap from angular support
    between_exp_scatter: Optional[np.ndarray] = None  # Per-coefficient between-experiment scatter (ENDF a_l units)
    between_exp_L_common: int = 0                      # Max order with valid scatter
    skip_reason: Optional[str] = None
    expanded_bins: int = 0                             # 0=original, 1=±1 expanded, 2=±2 expanded
    endf_index: Optional[int] = None                   # Nearest ENDF grid index (for I/O when grid != ENDF)


# --------------------------------------------------------------------------- #
# RESEARCH FORK: model-order averaging diagnostics
# --------------------------------------------------------------------------- #

# Columns emitted per bin. Kept here so the writer and the tests agree on the
# schema in one place rather than by coincidence.
AVERAGING_COLUMNS = (
    [f'avg_a_{l}' for l in range(1, 7)]
    + [f'q_a_{l}' for l in range(1, 7)]
    + [f'cond_a_{l}' for l in range(1, 7)]
    + ['n_eff_models', 'weight_floor_applied']
)

# Data-space fit quality, one statistic evaluated for BOTH centrals.
CHI2_COLUMNS = ['chi2_pp_win', 'chi2_pp_avg', 'chi2_pp_ratio']


def data_space_chi2(
    mu: np.ndarray,
    y: np.ndarray,
    sigma_eff: np.ndarray,
    weights: Optional[np.ndarray],
    a_vec: np.ndarray,
    c0: float,
) -> float:
    """Kernel-weighted mean squared standardized residual for one central value.

    .. math::
        \\chi^2_{pp} = \\frac{\\sum_i w_i\\,[(y_i-\\hat y_i)/\\sigma_{{\\rm eff},i}]^2}
                            {\\sum_i w_i}

    **Why this and not the fit's own ``chi2_red``.** ``chi2_red`` comes out of
    the GLS solver with a per-experiment rank-1 normalisation and a ridge-aware
    effective parameter count. Scoring the model-averaged coefficients against
    *that* number would compare two different quadratic forms and two different
    dof conventions, and the difference would not be attributable to the central
    value. So this computes ONE explicit statistic and evaluates it for the
    winner and the average with the **same** ``sigma_eff``, the **same** kernel
    weights and the same points. Only the predicted curve differs, which is the
    whole question.

    It is a per-effective-point quantity, so winner and average are directly
    comparable and the ratio is the number to read. It is NOT comparable to
    ``chi2_red`` and must not be reported alongside it as if it were.

    The curve is rebuilt from ENDF-normalised coefficients at a **shared** ``c0``
    (the winner's), since :math:`y = c_0\\,[1 + \\sum_l (2l+1)\\,a_l P_l(\\mu)]`.
    Holding ``c0`` fixed isolates the angular *shape*, which is what the order
    treatment changes; the magnitude is the host MF3's business.
    """
    # Local import: v2 imports legval inside the functions that use it, and this
    # module has no top-level legendre import to rely on.
    from numpy.polynomial.legendre import legval

    if mu.size == 0 or not np.isfinite(a_vec).all():
        return float('nan')
    full = np.concatenate([[1.0], [(2 * l + 1) * a_vec[l - 1] for l in range(1, len(a_vec) + 1)]])
    y_hat = c0 * legval(mu, full)

    w = np.ones_like(y, dtype=float) if weights is None or len(weights) != len(y) \
        else np.asarray(weights, dtype=float)
    ok = np.isfinite(sigma_eff) & (sigma_eff > 0) & np.isfinite(y) & np.isfinite(y_hat) & (w > 0)
    if not ok.any():
        return float('nan')
    r = (y[ok] - y_hat[ok]) / sigma_eff[ok]
    return float(np.sum(w[ok] * r * r) / np.sum(w[ok]))


def compute_averaging_diagnostics(
    all_degrees_info: Optional[Dict[int, Dict]],
    degree_weights: Optional[Dict[int, float]],
    max_degree: int,
    weight_floor: float = 0.0,
) -> Dict[str, float]:
    """Model-averaged central value and inclusion probabilities for one bin.

    Implements roadmap §4.2. For candidate degrees L with IC support weights
    ``w_L``, the averaged central is ``ā = Σ_L w_L a_L`` where a lower-order
    candidate contributes an **exact zero** for an absent coefficient. That zero
    is a prediction, not a missing value: because the Legendre representation is
    linear, ``ā_l`` is the exact coefficient of the weighted-average angular
    distribution. It answers "what is a_5 after accounting for order
    uncertainty", not "what is a_5 given order 5 is certainly present" — the
    latter is ``cond_a_l``, kept as a diagnostic only.

    Averaging happens in **normalized a-space**, not raw c-space, because MF4
    ships a normalized shape and the host MF3 central is retained.

    Returns NaN for every averaged quantity when the bin has no candidate set —
    interpolated bins have no fitted models, and a silent fall-back to the
    winner would hide which bins the average is undefined for.
    """
    nan_row = {c: float('nan') for c in AVERAGING_COLUMNS}
    nan_row['weight_floor_applied'] = float(weight_floor)

    if not all_degrees_info or not degree_weights:
        return nan_row

    # Only degrees that survived the floor AND that we have coefficients for.
    degrees = sorted(d for d in degree_weights if d in all_degrees_info)
    if not degrees:
        return nan_row

    weights = np.array([degree_weights[d] for d in degrees], dtype=float)
    if not np.isfinite(weights).any() or weights.sum() <= 0:
        return nan_row

    # Normalize each candidate into ENDF a-space before averaging. A degree-L
    # candidate carries c_0..c_L, so this yields a_1..a_L, which pad_to then
    # zero-extends to max_degree. pad_to REFUSES to truncate, so a candidate
    # somehow longer than max_degree raises rather than silently averaging a
    # different model than the one that was scored.
    a_vectors = []
    for d in degrees:
        coeffs = np.asarray(all_degrees_info[d]['coeffs'], dtype=float)
        a_vectors.append(endf_normalize_legendre_coeffs(coeffs, include_a0=False))

    means = stack_padded(a_vectors, max_degree)
    masks = nested_masks(degrees, max_degree)

    moments = mixture_moments(weights, means)
    q = inclusion_probabilities(weights / weights.sum(), masks)
    cond = conditional_mean(weights, means, masks)

    out: Dict[str, float] = {}
    for l in range(1, max_degree + 1):
        out[f'avg_a_{l}'] = float(moments['mean'][l - 1])
        out[f'q_a_{l}'] = float(q[l - 1])
        out[f'cond_a_{l}'] = float(cond[l - 1])
    # Orders above max_degree are not evaluated at all; NaN, not 0.0, so they
    # are never mistaken for "averaged to zero".
    for l in range(max_degree + 1, 7):
        out[f'avg_a_{l}'] = float('nan')
        out[f'q_a_{l}'] = float('nan')
        out[f'cond_a_{l}'] = float('nan')

    out['n_eff_models'] = float(effective_n_models(weights))
    out['weight_floor_applied'] = float(weight_floor)
    return out


# Import the rest of the workflow functions from the original script
# These are the same as the original - just need to update the MF34 calls


def interpolate_missing_nominal_fits(
    nominal_results: List[NominalFitResult],
    logger=None,
) -> List[NominalFitResult]:
    """
    Interpolate coefficients for bins where EXFOR data was not available.

    IMPORTANT: This ensures we NEVER use original ENDF coefficients as fallback,
    which would compromise the independence of the evaluation.

    Uses linear interpolation in energy space between neighboring bins
    that have valid EXFOR data.

    Parameters
    ----------
    nominal_results : List[NominalFitResult]
        List of nominal fit results (some may have has_data=False)
    logger : optional
        Logger instance

    Returns
    -------
    List[NominalFitResult]
        Updated results with interpolated coefficients for missing bins
    """
    # Find indices with and without data
    valid_indices = []
    missing_indices = []

    for i, result in enumerate(nominal_results):
        if result.has_data:
            valid_indices.append(i)
        else:
            missing_indices.append(i)

    if not missing_indices:
        return nominal_results

    if len(valid_indices) < 2:
        if logger:
            logger.error(
                "CRITICAL: Not enough bins with EXFOR data for interpolation. "
                "Cannot proceed without at least 2 bins with data."
            )
            logger.error(
                "  -> Energy bins without EXFOR data will have ISOTROPIC (a0=1) coefficients."
            )
            logger.error(
                "  -> These bins should be excluded from final evaluation or data coverage improved."
            )
        # Set isotropic for missing bins rather than using original ENDF
        for miss_idx in missing_indices:
            nominal_results[miss_idx].nominal_coeffs = np.array([1.0])
            nominal_results[miss_idx].frozen_degree = 0
        return nominal_results

    # Get energies and coefficient arrays for valid bins
    valid_energies = np.array([nominal_results[i].energy_mev for i in valid_indices])

    # Determine max coefficient length across all valid bins
    max_n_coeffs = max(len(nominal_results[i].nominal_coeffs) for i in valid_indices)

    # Pad coefficient arrays to same size
    valid_coeffs = []
    for i in valid_indices:
        coeffs = nominal_results[i].nominal_coeffs
        if len(coeffs) < max_n_coeffs:
            padded = np.zeros(max_n_coeffs, dtype=float)
            padded[:len(coeffs)] = coeffs
            valid_coeffs.append(padded)
        else:
            valid_coeffs.append(coeffs)
    valid_coeffs = np.array(valid_coeffs)  # Shape: (n_valid, n_coeffs)

    # Also get frozen degrees for interpolation (round to nearest integer)
    valid_degrees = np.array([nominal_results[i].frozen_degree for i in valid_indices])

    # Interpolate for each missing bin
    n_interpolated = 0
    n_extrapolated = 0

    for miss_idx in missing_indices:
        miss_energy = nominal_results[miss_idx].energy_mev

        # Check if energy is within interpolation range
        if miss_energy < valid_energies.min():
            # Extrapolate below - use lowest valid bin's values
            if logger:
                logger.warning(
                    f"E={miss_energy:.4f} MeV: Below EXFOR data range [{valid_energies.min():.4f} MeV] "
                    f"- extrapolating from lowest valid bin"
                )
            # Use nearest (lowest energy with data)
            nearest_idx = valid_indices[0]
            nominal_results[miss_idx].nominal_coeffs = nominal_results[nearest_idx].nominal_coeffs.copy()
            nominal_results[miss_idx].frozen_degree = nominal_results[nearest_idx].frozen_degree
            nominal_results[miss_idx].interpolated = True
            nominal_results[miss_idx].has_data = True  # Mark as having data (interpolated)
            n_extrapolated += 1
            continue

        if miss_energy > valid_energies.max():
            # Extrapolate above - use highest valid bin's values
            if logger:
                logger.warning(
                    f"E={miss_energy:.4f} MeV: Above EXFOR data range [{valid_energies.max():.4f} MeV] "
                    f"- extrapolating from highest valid bin"
                )
            # Use nearest (highest energy with data)
            nearest_idx = valid_indices[-1]
            nominal_results[miss_idx].nominal_coeffs = nominal_results[nearest_idx].nominal_coeffs.copy()
            nominal_results[miss_idx].frozen_degree = nominal_results[nearest_idx].frozen_degree
            nominal_results[miss_idx].interpolated = True
            nominal_results[miss_idx].has_data = True  # Mark as having data (interpolated)
            n_extrapolated += 1
            continue

        # Interpolate each coefficient
        interp_coeffs = np.zeros(max_n_coeffs, dtype=float)
        for coeff_idx in range(max_n_coeffs):
            y_vals = valid_coeffs[:, coeff_idx]
            interp_coeffs[coeff_idx] = np.interp(miss_energy, valid_energies, y_vals)

        # Interpolate degree (round to nearest integer)
        interp_degree = int(round(np.interp(miss_energy, valid_energies, valid_degrees.astype(float))))

        nominal_results[miss_idx].nominal_coeffs = interp_coeffs
        nominal_results[miss_idx].frozen_degree = interp_degree
        nominal_results[miss_idx].interpolated = True
        nominal_results[miss_idx].has_data = True  # Mark as having data (interpolated)
        n_interpolated += 1

        if logger:
            logger.info(
                f"E={miss_energy:.4f} MeV: INTERPOLATED from neighboring bins (L={interp_degree})"
            )

    if logger:
        logger.info("")
        logger.info("[INTERPOLATION SUMMARY]")
        logger.info(f"  Bins with EXFOR data: {len(valid_indices)}")
        logger.info(f"  Bins interpolated: {n_interpolated}")
        logger.info(f"  Bins extrapolated: {n_extrapolated}")
        if n_extrapolated > 0:
            logger.warning(
                f"  WARNING: {n_extrapolated} bins were extrapolated outside EXFOR data range. "
                f"Consider expanding energy range or improving data coverage."
            )

    return nominal_results


def log_experiments_summary(
    nominal_results: List[NominalFitResult],
    logger=None,
) -> None:
    """
    Log a summary of all EXFOR experiments used across all energy bins.

    This provides a quick overview of data sources used in the evaluation.

    Parameters
    ----------
    nominal_results : List[NominalFitResult]
        List of nominal fit results
    logger : optional
        Logger instance
    """
    from collections import defaultdict

    if not logger:
        return

    # Aggregate experiments across all energy bins
    experiment_totals = defaultdict(lambda: {
        'author': '',
        'year': '',
        'n_energies': 0,
        'total_points': 0,
        'energy_min': float('inf'),
        'energy_max': float('-inf'),
    })

    for nr in nominal_results:
        if not nr.has_data or nr.interpolated:
            continue  # Skip interpolated bins - they don't contribute new data

        for exp in nr.experiments_info:
            key = (exp['entry'], exp['subentry'])
            experiment_totals[key]['author'] = exp.get('author', 'Unknown')
            experiment_totals[key]['year'] = exp.get('year', '????')
            experiment_totals[key]['n_energies'] += 1
            experiment_totals[key]['total_points'] += exp.get('n_points', 0)

            exp_energy = exp.get('exfor_energy_mev', nr.energy_mev)
            if exp_energy < experiment_totals[key]['energy_min']:
                experiment_totals[key]['energy_min'] = exp_energy
            if exp_energy > experiment_totals[key]['energy_max']:
                experiment_totals[key]['energy_max'] = exp_energy

    if not experiment_totals:
        logger.info("[EXPERIMENTS USED - SUMMARY]")
        logger.info("  No experiments found (all bins interpolated or no data)")
        return

    # Sort by total points (descending)
    sorted_experiments = sorted(
        experiment_totals.items(),
        key=lambda x: x[1]['total_points'],
        reverse=True
    )

    # Calculate totals
    total_experiments = len(sorted_experiments)
    total_points = sum(exp['total_points'] for _, exp in sorted_experiments)
    total_energy_bins = sum(1 for nr in nominal_results if nr.has_data and not nr.interpolated)

    logger.info("")
    logger.info("=" * 80)
    logger.info("[EXPERIMENTS USED - SUMMARY]")
    logger.info("=" * 80)
    logger.info(f"  Total experiments: {total_experiments}")
    logger.info(f"  Total data points: {total_points}")
    logger.info(f"  Energy bins with EXFOR data: {total_energy_bins}")
    logger.info("")
    logger.info("  Experiment details (sorted by total points):")
    logger.info("  " + "-" * 76)
    logger.info(f"  {'Entry.Sub':<12} {'Author':<20} {'Year':<6} {'Energies':<10} {'Points':<8} {'E range (MeV)'}")
    logger.info("  " + "-" * 76)

    for (entry, subentry), data in sorted_experiments:
        exp_id = f"{entry}.{subentry}"
        author = data['author'][:18] if len(data['author']) > 18 else data['author']
        year = str(data['year'])
        n_energies = data['n_energies']
        total_pts = data['total_points']

        if data['energy_min'] == data['energy_max']:
            e_range = f"{data['energy_min']:.4f}"
        else:
            e_range = f"{data['energy_min']:.4f}-{data['energy_max']:.4f}"

        logger.info(f"  {exp_id:<12} {author:<20} {year:<6} {n_energies:<10} {total_pts:<8} {e_range}")

    logger.info("  " + "-" * 76)
    logger.info(f"  {'TOTAL':<12} {'':<20} {'':<6} {'':<10} {total_points:<8}")
    logger.info("=" * 80)
    logger.info("")


from scripts.exfor_utils import check_angular_quality as _check_angular_quality  # noqa: E402


def perform_nominal_fits(
    energy_bins: List[EnergyBinInfo],
    exfor_cache: Dict[float, List[Tuple[pd.DataFrame, Dict]]],
    sorted_energies: List[float],
    max_degree: int,
    select_degree: Optional[str],
    ridge_lambda: float,
    m_proj_u: float,
    m_targ_u: float,
    use_band_discrepancy: bool,
    min_points_per_band: int,
    max_band_scale: float,
    tau_smoothing_window: int,
    n_eff_warning_threshold: float = 5.0,
    min_degree_for_averaging: int = 3,
    exclude_experiments: Optional[List[str]] = None,
    min_relative_uncertainty: float = 0.0,
    tau_prior_floor: bool = True,
    tau_prior_neff_threshold: float = 5.0,
    tau_prior_percentile: float = 50.0,
    angular_quality_gate: bool = True,
    min_angular_points: int = 4,
    min_bands_covered: int = 3,
    max_bin_expansion: int = 2,
    rerun_aicc_post_tau: bool = True,
    use_gls_kernel: bool = True,
    tau_refit_use_gls: bool = TAU_REFIT_USE_GLS,
    sigma_norm_systematic: float = NORM_SYSTEMATIC_SIGMA,
    membership_k_sigma: float = 0.0,
    logger = None,
) -> List[NominalFitResult]:
    """Phase 1: Perform nominal fits using energy_bin method.

    Returns
    -------
    List[NominalFitResult]
        List of nominal fit results for each energy bin.
    """
    from numpy.polynomial.legendre import legvander, legval

    logger = _get_logger()
    results = []
    total_floor_floored = 0
    total_floor_pts = 0
    total_floor_bins = 0
    floor_replacement_vals = []
    gate_expanded_1 = 0  # Bins expanded by ±1
    gate_expanded_2 = 0  # Bins expanded by ±2
    gate_failed = 0      # Bins that failed quality gate after max expansion

    for bin_idx, bin_info in enumerate(energy_bins):
        exfor_df, experiments_info, kernel_weights, diagnostics, floor_stats = filter_exfor_with_energy_bin(
            exfor_cache=exfor_cache,
            sorted_energies=sorted_energies,
            bin_lower_mev=bin_info.bin_lower_mev,
            bin_upper_mev=bin_info.bin_upper_mev,
            target_energy_mev=bin_info.energy_mev,
            m_proj_u=m_proj_u,
            m_targ_u=m_targ_u,
            dedupe_per_experiment=True,
            exclude_experiments=exclude_experiments,
            min_relative_uncertainty=min_relative_uncertainty,
            unc_floor_strategy=UNCERTAINTY_FLOOR_STRATEGY,
            normalize_by_n_points=NORMALIZE_BY_N_POINTS,
            sigma_norm=sigma_norm_systematic,
            band_aware_ess=BAND_AWARE_ESS,
            max_experiment_weight_fraction=MAX_EXP_WEIGHT_FRAC_BIN,
            membership_k_sigma=membership_k_sigma,
            sigma_E_mev=bin_info.sigma_E_mev,
            logger=_logger,
        )

        # Accumulate uncertainty floor stats
        if floor_stats['n_floored'] > 0:
            total_floor_floored += floor_stats['n_floored']
            total_floor_pts += floor_stats['n_total']
            total_floor_bins += 1
            floor_replacement_vals.append(floor_stats['replacement_rel_unc'])

        if exfor_df.empty or len(exfor_df) < 3:
            results.append(NominalFitResult(
                energy_mev=bin_info.energy_mev,
                energy_index=bin_info.index,
                endf_index=bin_info.endf_index,
                exfor_df=pd.DataFrame(),
                experiments_info=[],
                kernel_weights=np.array([]),
                frozen_degree=0,
                nominal_coeffs=np.array([1.0]),
                sigma_eff=np.array([]),
                tau_info={'tau_F': 1.0, 'tau_M': 1.0, 'tau_B': 1.0},
                chi2_red=0.0,
                has_data=False,
                kernel_diagnostics=None,
            ))
            if logger:
                logger.warning(f"E={bin_info.energy_mev:.4f} MeV: No EXFOR data (σE={bin_info.sigma_E_mev:.4f} MeV)")
            continue

        # --- Angular Quality Gate ---
        expansion_used = 0
        if angular_quality_gate:
            passes, reason = _check_angular_quality(exfor_df, min_angular_points, min_bands_covered)

            if not passes:
                expanded = False
                for expansion in range(1, max_bin_expansion + 1):
                    left_idx = max(0, bin_idx - expansion)
                    right_idx = min(len(energy_bins) - 1, bin_idx + expansion)
                    expanded_lower = energy_bins[left_idx].bin_lower_mev
                    expanded_upper = energy_bins[right_idx].bin_upper_mev

                    exfor_df_exp, exp_info_exp, kw_exp, diag_exp, floor_exp = filter_exfor_with_energy_bin(
                        exfor_cache=exfor_cache,
                        sorted_energies=sorted_energies,
                        bin_lower_mev=expanded_lower,
                        bin_upper_mev=expanded_upper,
                        target_energy_mev=bin_info.energy_mev,
                        m_proj_u=m_proj_u,
                        m_targ_u=m_targ_u,
                        dedupe_per_experiment=True,
                        exclude_experiments=exclude_experiments,
                        min_relative_uncertainty=min_relative_uncertainty,
                        unc_floor_strategy=UNCERTAINTY_FLOOR_STRATEGY,
                        normalize_by_n_points=NORMALIZE_BY_N_POINTS,
                        sigma_norm=sigma_norm_systematic,
                        band_aware_ess=BAND_AWARE_ESS,
                        max_experiment_weight_fraction=MAX_EXP_WEIGHT_FRAC_BIN,
                        membership_k_sigma=membership_k_sigma,
                        sigma_E_mev=bin_info.sigma_E_mev,
                        logger=_logger,
                    )

                    passes_exp, reason_exp = _check_angular_quality(exfor_df_exp, min_angular_points, min_bands_covered)
                    if passes_exp:
                        exfor_df = exfor_df_exp
                        experiments_info = exp_info_exp
                        kernel_weights = kw_exp
                        diagnostics = diag_exp
                        if floor_exp['n_floored'] > 0:
                            total_floor_floored += floor_exp['n_floored']
                            total_floor_pts += floor_exp['n_total']
                            total_floor_bins += 1
                            floor_replacement_vals.append(floor_exp['replacement_rel_unc'])
                        expansion_used = expansion
                        if expansion == 1:
                            gate_expanded_1 += 1
                        else:
                            gate_expanded_2 += 1
                        if logger:
                            logger.info(
                                f"E={bin_info.energy_mev:.4f} MeV: expanded ±{expansion} bins "
                                f"(idx {left_idx}-{right_idx}), now {len(exfor_df)} pts"
                            )
                        expanded = True
                        break

                if not expanded:
                    gate_failed += 1
                    if logger:
                        logger.warning(
                            f"E={bin_info.energy_mev:.4f} MeV: QUALITY GATE failed after "
                            f"max expansion ({reason}) → interpolate"
                        )
                    results.append(NominalFitResult(
                        energy_mev=bin_info.energy_mev,
                        energy_index=bin_info.index,
                        endf_index=bin_info.endf_index,
                        exfor_df=pd.DataFrame(),
                        experiments_info=[],
                        kernel_weights=np.array([]),
                        frozen_degree=0,
                        nominal_coeffs=np.array([1.0]),
                        sigma_eff=np.array([]),
                        tau_info={'tau_F': 1.0, 'tau_M': 1.0, 'tau_B': 1.0},
                        chi2_red=0.0,
                        has_data=False,
                        skip_reason=reason,
                        kernel_diagnostics=None,
                    ))
                    continue

        bin_info.has_exfor_data = True
        bin_info.exfor_n_points = len(exfor_df)
        bin_info.exfor_n_experiments = len(experiments_info)
        bin_info.experiments_used = experiments_info

        mu = exfor_df['mu'].to_numpy()
        y = exfor_df['value'].to_numpy()
        sigma = exfor_df['unc'].to_numpy()

        # Build absolute σ_sys per row from the relative manifest column.
        # Used only for fit weights and τ residual normalization; σ_eff
        # stored downstream stays τ·σ_stat (semantics unchanged).
        sys_unc_col_arg: Optional[str] = None
        if SIGMA_SYS_AWARE_FIT and 'sigma_sys_relative' in exfor_df.columns:
            exfor_df = exfor_df.copy()
            exfor_df['_sigma_sys_abs'] = (
                exfor_df['sigma_sys_relative'].to_numpy(dtype=float)
                * np.abs(exfor_df['value'].to_numpy(dtype=float))
            )
            sys_unc_col_arg = '_sigma_sys_abs'

        coef_df, fit_info = sample_legendre_coefficients(
            exfor_df,
            value_col="value",
            unc_col="unc",
            sys_unc_col=sys_unc_col_arg,
            degree=None,
            max_degree=max_degree,
            select_degree=select_degree,
            ridge_lambda=ridge_lambda,
            ridge_power=RIDGE_POWER,
            df_method=DF_METHOD,
            external_weights=kernel_weights,
            n_samples=1,
            rescale_unc_by_chi2=RESCALE_UNC_BY_CHI2,
            allow_shrink_unc=ALLOW_SHRINK_UNC,
            use_band_discrepancy=use_band_discrepancy,
            min_points_per_band=min_points_per_band,
            max_band_scale=max_band_scale,
            tau_irls_max_iters=TAU_IRLS_MAX_ITERS,
            tau_irls_tol=TAU_IRLS_TOL,
            tau_irls_damping=TAU_IRLS_DAMPING,
            band_scale_method=BAND_SCALE_METHOD,
            rerun_aicc_post_tau=rerun_aicc_post_tau,
            use_gls_kernel=use_gls_kernel,
            tau_refit_use_gls=tau_refit_use_gls,
        )

        frozen_degree = fit_info['degree']
        nominal_coeffs = coef_df.iloc[0].to_numpy()
        chi2_red = fit_info['chi2_red']
        tau_info = fit_info.get('tau_info', {'tau_F': 1.0, 'tau_M': 1.0, 'tau_B': 1.0})

        all_degrees_info = fit_info.get('all_degrees_info', None)
        degree_weights = None

        if all_degrees_info and len(all_degrees_info) > 1:
            # RESEARCH FORK: v2 hard-codes a 1 % cutoff here. It is now
            # DEGREE_WEIGHT_FLOOR, default 0.0 — keep every feasible candidate.
            # The weights themselves are unchanged: ic_weights computes the same
            # exp(-Δ/2)/Σ, but shifts by the minimum first (log-sum-exp), which
            # matters because a χ²-based IC routinely reaches a few hundred and
            # would otherwise overflow exp to NaN.
            _degrees = sorted(all_degrees_info.keys())
            _scores = [all_degrees_info[d]['aicc'] for d in _degrees]
            _w = ic_weights(_scores, floor=DEGREE_WEIGHT_FLOOR)
            degree_weights = {
                d: float(w) for d, w in zip(_degrees, _w)
                if w > 0.0 and d >= min_degree_for_averaging
            }
            if degree_weights:
                total = sum(degree_weights.values())
                degree_weights = {d: w / total for d, w in degree_weights.items()}
            else:
                degree_weights = {frozen_degree: 1.0}

        y_fit = legval(mu, nominal_coeffs)

        if use_band_discrepancy and tau_info:
            # Pass experiment IDs for per-experiment diagnostics on capped bands
            _exp_ids = exfor_df['entry'].values if 'entry' in exfor_df.columns else None
            _sigma_sys_arg = (
                exfor_df[sys_unc_col_arg].to_numpy(dtype=float)
                if sys_unc_col_arg is not None else None
            )
            sigma_eff, tau_info = compute_angular_band_discrepancy(
                mu=mu, y=y, sigma=sigma, y_fit=y_fit,
                min_points_per_band=min_points_per_band,
                max_band_scale=max_band_scale,
                experiment_ids=_exp_ids,
                sigma_sys=_sigma_sys_arg,
            )
            tau_F = tau_info.get('tau_F', 1.0)
            tau_M = tau_info.get('tau_M', 1.0)
            tau_B = tau_info.get('tau_B', 1.0)
        else:
            scale = max(1.0, np.sqrt(chi2_red))
            sigma_eff = sigma * scale
            tau_F = tau_M = tau_B = 1.0
            tau_info = {'tau_F': 1.0, 'tau_M': 1.0, 'tau_B': 1.0}

        bin_info.fitted_degree = frozen_degree
        bin_info.chi2_red = chi2_red
        bin_info.tau_F = tau_F
        bin_info.tau_M = tau_M
        bin_info.tau_B = tau_B

        final_n_eff = compute_n_eff(kernel_weights, sigma_eff)
        diagnostics.n_eff = final_n_eff

        if logger:
            if final_n_eff < n_eff_warning_threshold:
                logger.warning(
                    f"E={bin_info.energy_mev:.4f} MeV: Low N_eff={final_n_eff:.1f} "
                    f"(threshold: {n_eff_warning_threshold})"
                )
        # RESEARCH FORK: the averaging diagnostics and the data-space chi2 of
        # both centrals. Done here, not in the writer, because sigma_eff, the
        # kernel weights and the EXFOR points only exist inside this loop — and
        # comparing the two centrals is only meaningful against identical ones.
        _avg_diag: Optional[Dict[str, float]] = None
        if WRITE_AVERAGING_DIAGNOSTICS:
            _avg_diag = compute_averaging_diagnostics(
                all_degrees_info=all_degrees_info,
                degree_weights=degree_weights,
                max_degree=max_degree,
                weight_floor=DEGREE_WEIGHT_FLOOR,
            )
            _c0 = float(nominal_coeffs[0]) if len(nominal_coeffs) else float('nan')
            _win_a = endf_normalize_legendre_coeffs(nominal_coeffs, include_a0=False)
            # Absent orders are an exact zero — the same convention the average
            # uses, so the two curves are built the same way.
            _win_pad = np.zeros(max_degree)
            _win_pad[:min(len(_win_a), max_degree)] = _win_a[:max_degree]
            _avg_pad = np.array([_avg_diag[f'avg_a_{l}'] for l in range(1, max_degree + 1)])

            _cw = data_space_chi2(mu, y, sigma_eff, kernel_weights, _win_pad, _c0)
            _ca = data_space_chi2(mu, y, sigma_eff, kernel_weights, _avg_pad, _c0)
            _avg_diag['chi2_pp_win'] = _cw
            _avg_diag['chi2_pp_avg'] = _ca
            _avg_diag['chi2_pp_ratio'] = (
                float(_ca / _cw) if np.isfinite(_cw) and _cw > 0 else float('nan')
            )

        # PHASE 3: adopt the model-averaged central. Up to here `nominal_coeffs`
        # is the winner's; from here it is the mixture mean, and it is what the
        # ENDF ships.
        #
        # This has to use the analytic average of the NOMINAL candidate fits
        # (avg_a_*, already computed above), not the MC mixture mean assembled
        # later in build_mixture_blocks. The central value should be the
        # deterministic Sum_L w_L a_L, not that quantity plus MC noise. The two
        # agree in expectation; only the analytic one is reproducible.
        #
        # c_0 is deliberately kept from the winner. The average is taken in
        # a-space (shape), which is what MIN_DEGREE_FOR_AVERAGING and the
        # angular-equivalence property in Phase 1 are defined against; the
        # magnitude channel is MF33's business, not MF34's.
        winner_coeffs = np.asarray(nominal_coeffs, dtype=float).copy()
        if SHIP_MIXTURE_MEAN and _avg_diag is not None:
            _avg_a = [_avg_diag.get(f'avg_a_{l}') for l in range(1, max_degree + 1)]
            if all(a is not None and np.isfinite(a) for a in _avg_a):
                _c0v = float(winner_coeffs[0])
                _mix_c = np.zeros(max_degree + 1, dtype=float)
                _mix_c[0] = _c0v
                for _l in range(1, max_degree + 1):
                    # inverse of endf_normalize_legendre_coeffs: a_l = (c_l/c_0)/(2l+1)
                    _mix_c[_l] = float(_avg_a[_l - 1]) * (2 * _l + 1) * _c0v
                nominal_coeffs = _mix_c
            # A bin whose average is undefined (NaN) keeps the winner. That is
            # the honest fallback: Phase 2 established those bins have no
            # candidate set to average over, and fabricating one would hide
            # which bins the mixture does not cover.

        results.append(NominalFitResult(
            energy_mev=bin_info.energy_mev,
            energy_index=bin_info.index,
            endf_index=bin_info.endf_index,
            exfor_df=exfor_df,
            experiments_info=experiments_info,
            kernel_weights=kernel_weights,
            frozen_degree=frozen_degree,
            nominal_coeffs=nominal_coeffs,
            sigma_eff=sigma_eff,
            tau_info=tau_info if tau_info else {'tau_F': 1.0, 'tau_M': 1.0, 'tau_B': 1.0},
            chi2_red=chi2_red,
            has_data=True,
            kernel_diagnostics=diagnostics,
            degree_weights=degree_weights,
            all_degrees_info=all_degrees_info,
            expanded_bins=expansion_used,
            averaging_diag=_avg_diag,
            winner_coeffs=winner_coeffs,
        ))

        # Compute adaptive MC-only order cap from angular support diagnostics
        support_diag = compute_angular_support_diagnostics(mu, kernel_weights, max_degree)
        # RESEARCH FORK: v2 uses min(frozen_degree, recommended_mc_order), which
        # bounds the MC ceiling by whichever model won the IC scan. The ceiling
        # should come from how much angular structure the data can support — a
        # quantity already computed right above — not from a selection outcome.
        # Inert under STOP_AFTER_NOMINAL_FITS, since the MC never runs.
        mc_cap = (
            support_diag['recommended_mc_order'] if MC_CAP_FROM_SUPPORT_ONLY
            else min(frozen_degree, support_diag['recommended_mc_order'])
        )
        results[-1].mc_order_cap = mc_cap

        # Between-experiment scatter (for uncertainty floor)
        n_unique_entries = exfor_df['entry'].nunique() if 'entry' in exfor_df.columns else 0
        scatter_info = None
        if len(exfor_df) >= 3 and n_unique_entries >= 2:
            scatter_info = compute_between_experiment_coeffs(
                exfor_df=exfor_df, degree=frozen_degree,
                fixed_c0=nominal_coeffs[0],
            )
            if scatter_info is not None:
                results[-1].between_exp_scatter = scatter_info['scatter']
                results[-1].between_exp_L_common = scatter_info['L_common']

        if logger:
            exp_tag = f" [expanded ±{expansion_used}]" if expansion_used > 0 else ""
            logger.info(
                f"E = {bin_info.energy_mev:.4f} MeV (bin: [{bin_info.bin_lower_mev:.4f}, {bin_info.bin_upper_mev:.4f}] MeV){exp_tag}:"
            )

            # Experiments used (condensed - one line per experiment with ranges)
            condensed_lines = _format_condensed_experiments(experiments_info)
            for line in condensed_lines:
                logger.info(line)

            # Fit results
            logger.info(
                f"  Fit: L={frozen_degree}, χ²/dof={chi2_red:.2f}, {len(exfor_df)} pts, N_eff={final_n_eff:.1f}"
            )
            if degree_weights and len(degree_weights) > 1:
                dw_str = " ".join(f"L{d}:{w:.0%}" for d, w in sorted(degree_weights.items()))
                logger.info(f"  AICc weights: {dw_str}")
            raw_F = tau_info.get('raw_F', tau_F)
            raw_M = tau_info.get('raw_M', tau_M)
            raw_B = tau_info.get('raw_B', tau_B)
            any_capped = (raw_F > tau_F + 0.005) or (raw_M > tau_M + 0.005) or (raw_B > tau_B + 0.005)
            band_str = f"  Band scales: s_F={tau_F:.2f}, s_M={tau_M:.2f}, s_B={tau_B:.2f}"
            if any_capped:
                band_str += f" (raw: F={raw_F:.2f}, M={raw_M:.2f}, B={raw_B:.2f})"
            logger.info(band_str)
            # Per-experiment diagnostics for capped bands
            _exp_diag = tau_info.get('exp_diag')
            if _exp_diag:
                band_names = {'F': 'Forward', 'M': 'Mid', 'B': 'Backward'}
                for bnd, entries in _exp_diag.items():
                    # Sort experiments by mean_abs_r descending (worst first)
                    sorted_exps = sorted(entries.items(), key=lambda x: x[1]['mean_abs_r'], reverse=True)
                    parts = []
                    for exp_id, stats in sorted_exps:
                        parts.append(f"{exp_id}(n={stats['n']}, |r|={stats['mean_abs_r']:.1f}, "
                                     f"max|r|={stats['max_abs_r']:.1f})")
                    logger.info(f"    [{band_names.get(bnd, bnd)} capped] {', '.join(parts)}")
            if mc_cap < frozen_degree:
                logger.info(
                    f"  MC order cap: {mc_cap} (nominal L={frozen_degree}, "
                    f"n_unique_mu={support_diag['n_unique_mu']}, cond={support_diag['legendre_cond']:.0f})"
                )
            # Between-experiment scatter diagnostic
            r_last = results[-1]
            if r_last.between_exp_scatter is not None:
                sc = r_last.between_exp_scatter
                sc_str = ", ".join(f"l={l+1}:{sc[l]:.4f}" for l in range(len(sc)))
                n_qual = scatter_info['n_experiments'] if scatter_info else 0
                logger.info(f"  [Between-exp] L_common={r_last.between_exp_L_common}, n_qual={n_qual}, scatter: {sc_str}")
            if scatter_info is not None and scatter_info.get('skipped_experiments'):
                for s_entry, s_reason in scatter_info['skipped_experiments']:
                    logger.info(f"  [Between-exp skip] {s_entry}: {s_reason}")
            elif n_unique_entries >= 2 and r_last.between_exp_scatter is None:
                logger.info(f"  [Between-exp] skipped: <2 experiments passed angular quality gate")
            logger.info("")  # Blank line between bins

    if tau_smoothing_window > 1 and use_band_discrepancy:
        tau_by_energy = {r.energy_mev: r.tau_info for r in results if r.has_data}
        if len(tau_by_energy) >= tau_smoothing_window:
            smoothed_tau = smooth_tau_in_energy(tau_by_energy, window=tau_smoothing_window)
            for r in results:
                if r.has_data and r.energy_mev in smoothed_tau:
                    r.tau_info = smoothed_tau[r.energy_mev]

    if tau_prior_floor and use_band_discrepancy:
        baselines = apply_tau_prior_floor(
            results,
            n_eff_threshold=tau_prior_neff_threshold,
            percentile=tau_prior_percentile,
        )
        if logger:
            logger.info(f"  Band scale floor baselines: s_F={baselines['tau_F']:.2f}, "
                        f"s_M={baselines['tau_M']:.2f}, s_B={baselines['tau_B']:.2f}")

    # Recompute sigma_eff and N_eff after tau smoothing/floor
    if (tau_smoothing_window > 1 or tau_prior_floor) and use_band_discrepancy:
        for r in results:
            if not r.has_data or r.interpolated:
                continue
            mu = r.exfor_df['mu'].to_numpy()
            sigma = r.exfor_df['unc'].to_numpy()
            r.sigma_eff = sigma_eff_from_tau(mu, sigma, r.tau_info)
            r.kernel_diagnostics.n_eff = compute_n_eff(r.kernel_weights, r.sigma_eff)

    # Log global uncertainty floor summary
    if total_floor_floored > 0 and logger:
        valid_repls = [v for v in floor_replacement_vals if not np.isnan(v)]
        median_repl = float(np.median(valid_repls)) * 100 if valid_repls else 0.0
        logger.info(f"  [Unc floor] Summary: {total_floor_floored}/{total_floor_pts} total pts "
                    f"floored across {total_floor_bins} bins, "
                    f"median replacement={median_repl:.1f}% (strategy={UNCERTAINTY_FLOOR_STRATEGY})")

    # Log angular quality gate summary
    total_expanded = gate_expanded_1 + gate_expanded_2
    if angular_quality_gate and logger and (total_expanded > 0 or gate_failed > 0):
        logger.info(
            f"  [Quality gate] {total_expanded} bins expanded "
            f"({gate_expanded_1} by ±1, {gate_expanded_2} by ±2), "
            f"{gate_failed} bins failed → interpolate"
        )

    # ================================================================
    # End-of-fit summaries
    # ================================================================
    if logger and use_band_discrepancy:
        _log_band_capping_summary(results, max_band_scale, logger)
        _log_between_exp_summary(results, logger)
        _log_experiment_bias_summary(results, logger)

    return results


def _log_band_capping_summary(
    results: List[NominalFitResult],
    max_band_scale: float,
    logger,
) -> None:
    """Log aggregate band-capping statistics and repeat-offender experiments."""
    from collections import Counter

    band_names = {'F': 'Forward', 'M': 'Mid', 'B': 'Backward'}
    # Count capped bands and collect per-experiment stats
    n_capped = {'F': 0, 'M': 0, 'B': 0}
    # exp_id -> {band -> list of mean_abs_r}
    exp_stats = {}

    for r in results:
        if not r.has_data or r.interpolated:
            continue
        ti = r.tau_info
        for bnd in ('F', 'M', 'B'):
            raw = ti.get(f'raw_{bnd}', ti.get(f'tau_{bnd}', 1.0))
            if raw >= max_band_scale - 0.005:
                n_capped[bnd] += 1

        exp_diag = ti.get('exp_diag')
        if not exp_diag:
            continue
        for bnd, entries in exp_diag.items():
            for exp_id, stats in entries.items():
                if exp_id not in exp_stats:
                    exp_stats[exp_id] = {'F': [], 'M': [], 'B': [], 'abs_r': [], 'signed_r': []}
                exp_stats[exp_id][bnd].append(stats['mean_abs_r'])
                exp_stats[exp_id]['abs_r'].append(stats['mean_abs_r'])

    total_capped = sum(n_capped.values())
    if total_capped == 0:
        return

    logger.info("")
    logger.info("  === Band capping summary ===")
    logger.info(f"  {total_capped} band-scale values capped at {max_band_scale:.1f} "
                f"(F:{n_capped['F']}, M:{n_capped['M']}, B:{n_capped['B']})")

    if exp_stats:
        # Rank experiments by total number of capped-band appearances
        ranked = sorted(exp_stats.items(),
                        key=lambda x: len(x[1]['abs_r']), reverse=True)
        logger.info("  Experiments most frequently in capped bands:")
        for exp_id, stats in ranked[:10]:  # Top 10
            total = len(stats['abs_r'])
            band_counts = []
            for bnd in ('B', 'M', 'F'):
                cnt = len(stats[bnd])
                if cnt > 0:
                    band_counts.append(f"{bnd}:{cnt}")
            mean_r = float(np.mean(stats['abs_r']))
            logger.info(f"    {exp_id}: {total} bins ({', '.join(band_counts)}), "
                        f"mean|r|={mean_r:.1f}")


def _log_between_exp_summary(
    results: List[NominalFitResult],
    logger,
) -> None:
    """Log between-experiment scatter summary by energy region."""
    # Collect bins with scatter and their inflation potential
    scatter_bins = []
    for r in results:
        if not r.has_data or r.interpolated or r.between_exp_scatter is None:
            continue
        scatter_bins.append(r)

    n_total = sum(1 for r in results if r.has_data and not r.interpolated)
    n_with_scatter = len(scatter_bins)
    if n_with_scatter == 0:
        return

    logger.info("")
    logger.info("  === Between-experiment scatter summary ===")
    logger.info(f"  {n_with_scatter}/{n_total} bins had between-experiment scatter computed")

    # Find energy regions with largest scatter (l=1, most physically meaningful)
    l1_scatter = [(r.energy_mev, r.between_exp_scatter[0]) for r in scatter_bins
                  if len(r.between_exp_scatter) > 0]
    if not l1_scatter:
        return

    # Sort by scatter magnitude, report top energy regions
    l1_scatter.sort(key=lambda x: x[1], reverse=True)
    logger.info("  Bins with largest l=1 scatter (top 10):")
    for energy, scatter_val in l1_scatter[:10]:
        logger.info(f"    E={energy:.4f} MeV: scatter_l1={scatter_val:.4f}")


def _log_experiment_bias_summary(
    results: List[NominalFitResult],
    logger,
) -> None:
    """Log per-experiment systematic bias across all energy bins.

    For each experiment, compute the mean signed normalized residual
    across all bins.  A persistent positive or negative bias indicates
    a normalization offset that the band scale cannot distinguish from
    random scatter.
    """
    from collections import defaultdict

    exp_residuals = defaultdict(list)  # exp_id -> list of signed residuals

    for r in results:
        if not r.has_data or r.interpolated:
            continue
        df = r.exfor_df
        if 'entry' not in df.columns:
            continue
        mu = df['mu'].to_numpy()
        y = df['value'].to_numpy()
        sigma = r.sigma_eff
        y_fit = np.polynomial.legendre.legval(mu, r.nominal_coeffs)
        signed_r = (y - y_fit) / sigma
        entries = df['entry'].values

        for exp_id in np.unique(entries):
            mask = entries == exp_id
            exp_residuals[exp_id].extend(signed_r[mask].tolist())

    if not exp_residuals:
        return

    logger.info("")
    logger.info("  === Experiment systematic bias ===")
    # Compute mean signed residual and total points per experiment
    bias_data = []
    for exp_id, resids in exp_residuals.items():
        n = len(resids)
        mean_r = float(np.mean(resids))
        std_r = float(np.std(resids))
        bias_data.append((exp_id, n, mean_r, std_r))

    # Sort by absolute mean bias
    bias_data.sort(key=lambda x: abs(x[2]), reverse=True)
    logger.info(f"  {'Experiment':<12} {'N_pts':>7} {'mean_r':>8} {'std_r':>7}  note")
    for exp_id, n, mean_r, std_r in bias_data:
        # Flag if bias is statistically significant (|mean| > 2/sqrt(N))
        threshold = 2.0 / np.sqrt(max(n, 1))
        flag = ""
        if abs(mean_r) > threshold and n >= 10:
            direction = "high" if mean_r > 0 else "low"
            flag = f"  ← systematic {direction} ({abs(mean_r)/threshold:.1f}σ)"
        logger.info(f"  {str(exp_id):<12} {n:>7} {mean_r:>+8.3f} {std_r:>7.2f}{flag}")


# Import PrecomputedEnergyData, _precompute_energy_data, _sample_one_realization, run_mc_per_realization
# from original - these don't need changes
# For brevity, we reference them from the original implementation
# In practice, these would be copied here

# For the migration, we'll import the main run function logic


def load_exfor_with_new_api(
    exfor_directory: str = None,
    db_path: str = None,
    source: str = "auto",
    target_zaid: Union[int, List[int]] = None,
    projectile: str = "N",
    mt: int = None,
    energy_range: tuple = None,
    supplementary_json_files: List[str] = None,
    exclude_experiments: Optional[List[str]] = None,
    logger=None,
):
    """
    Load EXFOR data using the new kika.exfor module API.

    Supports multiple data sources: JSON files, X4Pro database, or automatic
    fallback (database with JSON fallback for missing entries).

    Returns data in the legacy format for compatibility with existing code.

    Parameters
    ----------
    exfor_directory : str, optional
        Path to EXFOR data directory (for JSON source or fallback)
    db_path : str, optional
        Path to X4Pro database. Uses KIKA_X4PRO_DB_PATH env var if None.
    source : str, optional
        Data source: "json", "database", "auto" (default), or "both"
    target_zaid : int or List[int], optional
        Target ZAID(s) for database queries. Can be:
        - Single ZAID (e.g., 26056 for Fe-56)
        - List of ZAIDs (e.g., [26056, 26000] for Fe-56 + natural iron)
    projectile : str, optional
        Projectile for database queries (default: "N")
    mt : int, optional
        ENDF MT number for database queries
    energy_range : tuple, optional
        (min, max) energy range in MeV for filtering
    supplementary_json_files : List[str], optional
        List of additional JSON file paths to load (for experiments not in database)
    logger : optional
        Logger instance

    Returns
    -------
    exfor_cache : Dict[float, List[Tuple[pd.DataFrame, Dict]]]
        Legacy format data cache
    sorted_energies : List[float]
        Sorted list of available energies
    """
    if logger:
        logger.info(f"  Using NEW kika.exfor module (read_all_exfor)")
        logger.info(f"  Data source: {source}")
        if source in ("database", "auto", "both"):
            logger.info(f"  Database path: {db_path or 'default (env var or builtin)'}")
            if target_zaid:
                if isinstance(target_zaid, list):
                    logger.info(f"  Target ZAIDs: {target_zaid}")
                else:
                    logger.info(f"  Target ZAID: {target_zaid}")
        if supplementary_json_files:
            logger.info(f"  Supplementary JSON files: {len(supplementary_json_files)}")
            for f in supplementary_json_files:
                logger.info(f"    - {f}")

    # Load with new API - get all objects by identifier
    exfor_dict, load_status = read_all_exfor(
        directory=exfor_directory,
        group_by_energy=False,
        source=source,
        db_path=db_path,
        target_zaid=target_zaid,
        projectile=projectile,
        mt=mt,
        energy_range=energy_range,
        supplementary_json_files=supplementary_json_files,
        exclude_experiments=exclude_experiments,
        return_load_status=True,
    )

    # Extract list of ExforAngularDistribution objects
    exfor_objects = list(exfor_dict.values())

    if logger:
        logger.info(f"  Loaded {len(exfor_objects)} EXFOR datasets")

        # Log supplementary file load results
        if supplementary_json_files:
            logger.info("  Supplementary file load results:")
            for item in load_status.get('loaded', []):
                logger.info(f"    LOADED: {item['id']} ({item.get('label', 'unknown')}, {item['n_energies']} energies)")
            for item in load_status.get('skipped', []):
                logger.warning(f"    SKIPPED: {item['id']} - {item['reason']}")
            for item in load_status.get('failed', []):
                logger.warning(f"    FAILED: {item['file']} - {item['error']}")

    # Convert to legacy format using build_exfor_cache_from_objects
    exfor_cache, sorted_energies = build_exfor_cache_from_objects(
        exfor_objects,
        exclude_experiments=exclude_experiments,
    )

    return exfor_cache, sorted_energies


def _extract_mf34_energy_grid(
    endf_file: str,
    mf34_source_file: Optional[str],
    mt_number: int,
    logger=None,
) -> Optional[np.ndarray]:
    """
    Extract the energy grid from the original MF34 section of an ENDF file.

    Returns the union energy grid boundaries in MeV, or None if MF34 is not
    found or cannot be parsed.

    Parameters
    ----------
    endf_file : str
        Path to the reference ENDF file.
    mf34_source_file : str, optional
        Path to an alternative ENDF file containing MF34. If None, uses endf_file.
    mt_number : int
        MT reaction number to look up in MF34.
    logger : optional
        Logger for diagnostics.

    Returns
    -------
    np.ndarray or None
        Energy grid boundaries in MeV (sorted, unique), or None if unavailable.
    """
    from kika.endf.read_endf import read_endf

    source = mf34_source_file or endf_file
    try:
        endf_data = read_endf(source, mf_numbers=34)
    except Exception as e:
        if logger:
            logger.warning(f"  Could not read MF34 from {source}: {e}")
        return None

    # Access MF34 sections
    mf34 = endf_data.mf.get(34)
    if mf34 is None:
        if logger:
            logger.warning(f"  No MF34 found in {source}")
        return None

    mf34_mt = mf34.sections.get(mt_number)
    if mf34_mt is None:
        if logger:
            logger.warning(f"  No MF34/MT{mt_number} found in {source}")
        return None

    try:
        ang_covmat = mf34_mt.to_ang_covmat(energy_unit='eV')
    except Exception as e:
        if logger:
            logger.warning(f"  Could not convert MF34/MT{mt_number} to covmat: {e}")
        return None

    # Get energy grid: use union grid for robustness
    if ang_covmat.has_uniform_energy_grid():
        if ang_covmat.energy_grids:
            grid_ev = np.array(ang_covmat.energy_grids[0], dtype=float)
        else:
            if logger:
                logger.warning(f"  MF34/MT{mt_number} has no energy grids")
            return None
    else:
        union_grids = ang_covmat.compute_union_energy_grids()
        # Collect all union grid points
        all_points = set()
        for grid in union_grids.values():
            all_points.update(np.asarray(grid, dtype=float).tolist())
        if not all_points:
            if logger:
                logger.warning(f"  MF34/MT{mt_number} union grids are empty")
            return None
        grid_ev = np.array(sorted(all_points))

    # Convert eV -> MeV
    grid_mev = grid_ev / 1e6

    if logger:
        logger.info(f"  Extracted MF34 energy grid: {len(grid_mev)} points, "
                    f"[{grid_mev[0]:.4f}, {grid_mev[-1]:.4f}] MeV")

    return grid_mev


def run_exfor_to_endf_sampling_v2(
    endf_file: str,
    exfor_directory: str = None,
    output_dir: str = None,
    n_samples: int = 10,
    energy_min_mev: float = 1.0,
    energy_max_mev: float = 3.0,
    mt_number: int = 2,
    max_degree: int = 8,
    select_degree: Optional[str] = "aicc",
    ridge_lambda: float = 0.0,
    m_proj_u: float = 1.008665,
    m_targ_u: float = 55.93494,
    use_band_discrepancy: bool = True,
    min_points_per_band: int = 3,
    max_band_scale: float = 3.0,
    tau_smoothing_window: int = 3,
    tau_prior_floor: bool = True,
    tau_prior_neff_threshold: float = 5.0,
    tau_prior_percentile: float = 50.0,
    sigma_norm_systematic: float = 0.05,
    sigma_norm_common_mode: float = 0.0,
    use_model_averaging: bool = True,
    min_degree_for_averaging: int = 3,
    n_eff_warning_threshold: float = 5.0,
    n_procs: int = 1,
    base_seed: int = 42,
    generate_nominal_endf: bool = True,
    generate_mc_mean_endf: bool = True,
    generate_fitting_samples: bool = True,
    generate_fitting_ace: bool = False,
    save_covariance_files: bool = True,
    # Multigroup covariance options
    generate_multigroup_covariance: bool = False,
    multigroup_rho_min: float = 0.90,
    multigroup_sigma_ratio_max: Optional[float] = None,
    multigroup_variance_pct_min: float = 67.0,
    multigroup_variance_pct_max: float = 85.0,
    multigroup_variance_ratio_ref: float = 5.0,
    multigroup_regroup_after_smooth: bool = False,
    mf34_covariance_type: str = "fine",
    use_original_mf34_grid: bool = False,
    # Database configuration (new parameters)
    exfor_db_path: str = None,
    exfor_source: str = "auto",
    target_zaid: Union[int, List[int]] = None,
    target_projectile: str = "N",
    supplementary_json_files: List[str] = None,
    # Experiment exclusion and uncertainty floor
    exclude_experiments: Optional[List[str]] = None,
    min_relative_uncertainty: float = 0.0,
    # Energy grid source
    energy_grid_source: str = ENERGY_GRID_SOURCE,
    union_grid_subentries: Optional[List[Tuple[str, Optional[float], Optional[float]]]] = None,
    # Covariance cap (Layer 1) and positivity projection (Layer 2)
    apply_covariance_cap: bool = False,
    max_relative_std_cap: float = 1.0,
    regularize_near_zero: bool = True,
    near_zero_snr_threshold: float = 1.0,
    near_zero_n_neighbors: int = 3,
    apply_between_exp_floor: bool = True,
    apply_cov_postprocessing: bool = True,
    apply_positivity_projection: bool = False,
    positivity_check_points: int = 50,
    # File output options
    save_correlation_matrices: bool = False,
    save_tmc_parquet: bool = True,
    save_raw_kw_parquet: bool = False,
    save_multigroup_diagnostics_csv: bool = True,
    save_nominal_fits: bool = True,
    # ACE common options (shared NJOY config)
    ace_temperatures: Optional[List[float]] = None,
    ace_njoy_exe: Optional[str] = None,
    ace_library_name: Optional[str] = None,
    ace_njoy_version: str = "NJOY 2016.78",
    ace_xsdir_file: Optional[str] = None,
    ace_skip_existing: bool = False,
    # MF34 merge source
    mf34_source_file: Optional[str] = None,
    # MF34 sampling (Pipeline B)
    generate_mf34_samples: bool = False,
    generate_mf34_ace: bool = False,
    sampling_resolution: str = "fine",  # "fine" or "multigroup"
    merge_original_mf34: bool = True,
    sampling_space: str = "linear",
    sampling_decomposition: str = "svd",
    sampling_method: str = "sobol",
    # Diagnostic output
    verbose_diagnostics: bool = False,
):
    """
    Main function to generate ENDF samples from EXFOR angular distribution data.

    This is the v2 version using the new kika.exfor module API.

    CHANGES FROM v1:
    - Uses read_all_exfor() from kika.exfor instead of load_all_exfor_data()
    - Uses create_mf34_from_covariance and write_mf34_to_file from kika.endf.writers
    - Supports X4Pro database backend in addition to JSON files

    Parameters
    ----------
    endf_file : str
        Path to reference ENDF file
    exfor_directory : str, optional
        Path to EXFOR JSON files directory
    output_dir : str
        Output directory for generated files
    exfor_db_path : str, optional
        Path to X4Pro database. Uses KIKA_X4PRO_DB_PATH env var if None.
    exfor_source : str, optional
        Data source: "json", "database", "auto" (default), or "both"
    target_zaid : int, optional
        Target ZAID for database queries (e.g., 26056 for Fe-56)
    target_projectile : str, optional
        Projectile for database queries (default: "N")
    (other parameters documented inline)
    """
    global _logger

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = output_path / f'exfor_to_endf_{timestamp}.log'
    _logger = DualLogger(str(log_file))
    _set_logger(_logger)

    _logger.info("EXFOR-to-ENDF Angular Distribution Sampling (v2)")
    _logger.info(f"Timestamp: {datetime.now().isoformat()}")
    _logger.info("")

    print(f"[INFO] Starting EXFOR-to-ENDF sampling (v2)")
    print(f"[INFO] Log file: {log_file}")

    # ── Run metadata (audit trail) ───────────────────────────────────────────
    try:
        _md = _write_run_metadata(output_dir)
        _logger.info("#== RUN METADATA ==========================================================")
        _logger.info(f"  metadata_file   = {Path(output_dir) / 'run_metadata.json'}")
        _logger.info(f"  script_sha256   = {_md.get('script_sha256')}")
        _logger.info(f"  manifest_path   = {_md.get('manifest_path')}")
        _logger.info(f"  manifest_sha256 = {_md.get('manifest_sha256')} "
                     f"(reachable={_md.get('manifest_path_reachable')})")
        _g = _md.get('git') or {}
        _logger.info(f"  git_branch      = {_g.get('branch')}")
        _logger.info(f"  git_commit      = {_g.get('commit')}")
        _logger.info(f"  git_dirty       = {_g.get('dirty')}")
        _logger.info("")
    except Exception as _e:
        _logger.warning(f"Failed to write run_metadata.json: {type(_e).__name__}: {_e}")

    # ── Model-order policy (deliberate; log so it is reviewer-auditable) ─────
    _logger.info("#== MODEL-ORDER POLICY ====================================================")
    _logger.info("  Nominal MF4 order = AICc winner per bin.")
    _logger.info("  MC samples may draw alternate AICc-supported orders (mixture).")
    _logger.info("  MF34 covariance is published only for orders present in nominal MF4")
    _logger.info("    (n_valid = min(frozen_degree, MAX_SAMPLE_ORDER)).")
    _logger.info("  Higher sampled orders affect retained-order variance but are not published.")
    _logger.info("")

    # Track warnings for dynamic summary at end
    _warning_counts = {}

    # ── CONFIG section ───────────────────────────────────────────────────────
    _logger.info("#== CONFIG ================================================================")
    _logger.info("")

    # -- Paths & I/O --
    _logger.info("  # Paths & I/O")
    _logger.info(f"  ENDF_FILE = {endf_file}")
    _logger.info(f"  MF34_SOURCE_FILE = {mf34_source_file or '(same as ENDF_FILE)'}")
    _logger.info(f"  EXFOR_DIRECTORY = {exfor_directory}")
    _logger.info(f"  EXFOR_DB_PATH = {exfor_db_path}")
    _logger.info(f"  OUTPUT_DIR = {output_dir}")
    _logger.info(f"  TOF_PARAMETERS_FILE = {TOF_PARAMETERS_FILE}")
    _logger.info("")

    # -- Data Source --
    _logger.info("  # Data Source")
    _logger.info(f"  EXFOR_SOURCE = {exfor_source}")
    _logger.info(f"  TARGET_ZAIDS = {target_zaid}")
    _logger.info(f"  TARGET_PROJECTILE = {target_projectile}")
    _logger.info(f"  SUPPLEMENTARY_JSON_FILES = {supplementary_json_files}")
    _logger.info("")

    # -- Energy Range & Physics --
    _logger.info("  # Energy Range & Physics")
    _logger.info(f"  ENERGY_MIN_MEV = {energy_min_mev}")
    _logger.info(f"  ENERGY_MAX_MEV = {energy_max_mev}")
    _logger.info(f"  MT_NUMBER = {mt_number}")
    _logger.info(f"  M_PROJ_U = {m_proj_u}")
    _logger.info(f"  M_TARG_U = {m_targ_u}")
    _logger.info("")

    # -- Legendre Fitting --
    _logger.info("  # Legendre Fitting")
    _logger.info(f"  MAX_LEGENDRE_DEGREE = {max_degree}")
    _logger.info(f"  SELECT_DEGREE = {select_degree if select_degree else 'None (use max)'}")
    _logger.info(f"  RIDGE_LAMBDA = {ridge_lambda}")
    _logger.info(f"  RIDGE_POWER = {RIDGE_POWER}")
    _logger.info(f"  DF_METHOD = {DF_METHOD}")
    _logger.info("")

    # -- Uncertainty & Discrepancy --
    _logger.info("  # Uncertainty & Discrepancy")
    _logger.info(f"  USE_BAND_DISCREPANCY = {use_band_discrepancy}")
    _logger.info(f"  MIN_POINTS_PER_BAND = {min_points_per_band}")
    _logger.info(f"  MAX_BAND_SCALE_FACTOR = {max_band_scale}")
    _logger.info(f"  TAU_SMOOTHING_WINDOW = {tau_smoothing_window}")
    _logger.info(f"  TAU_PRIOR_FLOOR = {tau_prior_floor}")
    if tau_prior_floor:
        _logger.info(f"  TAU_PRIOR_NEFF_THRESHOLD = {tau_prior_neff_threshold}")
        _logger.info(f"  TAU_PRIOR_PERCENTILE = {tau_prior_percentile}")
    _logger.info(f"  TAU_IRLS_MAX_ITERS = {TAU_IRLS_MAX_ITERS}")
    _logger.info(f"  TAU_IRLS_TOL = {TAU_IRLS_TOL}")
    _logger.info(f"  TAU_IRLS_DAMPING = {TAU_IRLS_DAMPING}")
    _logger.info(f"  BAND_SCALE_METHOD = {BAND_SCALE_METHOD}")
    _logger.info(f"  RESCALE_UNC_BY_CHI2 = {RESCALE_UNC_BY_CHI2}")
    _logger.info(f"  ALLOW_SHRINK_UNC = {ALLOW_SHRINK_UNC}")
    _logger.info(f"  NORM_COMMON_MODE_SIGMA = {sigma_norm_common_mode}  (global, all experiments)")
    _logger.info(f"  NORM_SYSTEMATIC_SIGMA = {sigma_norm_systematic}  (per-experiment, dist={NORM_DIST})")
    _logger.info(f"  EXCLUDE_EXPERIMENTS = {exclude_experiments if exclude_experiments else 'None'}")
    _logger.info(f"  MIN_RELATIVE_UNCERTAINTY = {min_relative_uncertainty} ({min_relative_uncertainty*100:.1f}%)")
    _logger.info(f"  UNCERTAINTY_FLOOR_STRATEGY = {UNCERTAINTY_FLOOR_STRATEGY}")
    _logger.info("")

    # -- Model Averaging --
    _logger.info("  # Model Averaging")
    _logger.info(f"  USE_MODEL_AVERAGING = {use_model_averaging}")
    _logger.info(f"  MIN_DEGREE_FOR_AVERAGING = {min_degree_for_averaging}")
    _logger.info(f"  USE_DEGREE_SAMPLING_IN_MC = {USE_DEGREE_SAMPLING_IN_MC}")
    _logger.info(f"  RERUN_AICC_POST_TAU = {RERUN_AICC_POST_TAU}")
    # -- Research fork knobs (differences from v2) --
    _logger.info("  # Research fork (model-order treatment)")
    _logger.info(f"  DEGREE_WEIGHT_FLOOR = {DEGREE_WEIGHT_FLOOR}"
                 f"{'  (v2 hard-codes 0.01)' if DEGREE_WEIGHT_FLOOR != 0.01 else ''}")
    _logger.info(f"  MC_CAP_FROM_SUPPORT_ONLY = {MC_CAP_FROM_SUPPORT_ONLY}"
                 f"{'  (v2 bounds the cap by the winning degree)' if MC_CAP_FROM_SUPPORT_ONLY else ''}")
    _logger.info(f"  WRITE_AVERAGING_DIAGNOSTICS = {WRITE_AVERAGING_DIAGNOSTICS}")
    _logger.info(f"  USE_MIXTURE_COVARIANCE = {USE_MIXTURE_COVARIANCE}"
                 f"{'  (PHASE 3: per-bin law-of-total-covariance mixture)' if USE_MIXTURE_COVARIANCE else '  (legacy pooled degree-sampling)'}")
    if USE_MIXTURE_COVARIANCE:
        _logger.info(f"    MIXTURE_MIN_SAMPLES_PER_MODEL = {MIXTURE_MIN_SAMPLES_PER_MODEL}"
                     f"{'  (0 = pure proportional allocation, Gate B)' if MIXTURE_MIN_SAMPLES_PER_MODEL == 0 else ''}")
        _logger.info(f"    MIXTURE_Q_MASK_THRESHOLD = {MIXTURE_Q_MASK_THRESHOLD}"
                     "  (replaces the frozen_degree mask; expect ~100% valid slots)")
        _logger.info(f"    MIXTURE_SPLICE_WITHIN_BIN_CORR = {MIXTURE_SPLICE_WITHIN_BIN_CORR}"
                     f"{'  (⚠ expect a large Higham repair; see the constant)' if MIXTURE_SPLICE_WITHIN_BIN_CORR else '  (congruence only: mixture variances, Pass-1 correlations, PSD by construction)'}")
        _logger.info(f"    SHIP_MIXTURE_MEAN = {SHIP_MIXTURE_MEAN}"
                     f"{'  (MF4 central = mixture mean; winner kept as win_c_*)' if SHIP_MIXTURE_MEAN else '  (MF4 keeps the winner - mean and covariance will NOT match)'}")
    _logger.info(f"  REGULARIZE_NEAR_ZERO_REL_UNC = {REGULARIZE_NEAR_ZERO_REL_UNC}"
                 "  (kept ON — the only active guard against rel-sigma blow-up)")
    _logger.info(f"  USE_GLS_KERNEL = {USE_GLS_KERNEL}")
    _logger.info(f"  TAU_REFIT_USE_GLS = {TAU_REFIT_USE_GLS}"
                 f"{'  (RESEARCH FORK: v2 refits the tau-IRLS loop under diagonal WLS)' if TAU_REFIT_USE_GLS else ''}")
    _logger.info("")

    # -- Energy Binning & Correlation --
    if union_grid_subentries is None:
        union_grid_subentries = UNION_GRID_SUBENTRIES
    _logger.info("  # Energy Binning & Correlation")
    _logger.info(f"  NORMALIZE_BY_N_POINTS = {NORMALIZE_BY_N_POINTS}")
    _logger.info(f"  BAND_AWARE_ESS = {BAND_AWARE_ESS}")
    _logger.info(f"  MAX_EXP_WEIGHT_FRAC_BIN = {MAX_EXP_WEIGHT_FRAC_BIN}")
    _logger.info(f"  FREEZE_C0 = {FREEZE_C0}")
    _logger.info(f"  MAX_SAMPLE_ORDER = {MAX_SAMPLE_ORDER}")
    _logger.info(f"  ANGULAR_QUALITY_GATE = {ANGULAR_QUALITY_GATE}")
    if ANGULAR_QUALITY_GATE:
        _logger.info(f"  MIN_ANGULAR_POINTS = {MIN_ANGULAR_POINTS}")
        _logger.info(f"  MIN_BANDS_COVERED = {MIN_BANDS_COVERED}")
        _logger.info(f"  MAX_BIN_EXPANSION = {MAX_BIN_EXPANSION}")
    _logger.info(f"  ENERGY_GRID_SOURCE = {energy_grid_source}")
    if energy_grid_source == "union":
        _logger.info(f"  UNION_GRID_SUBENTRIES = {union_grid_subentries}")
    _logger.info(f"  CORRELATION_METHOD = {CORRELATION_METHOD}")
    if CORRELATION_METHOD in ("kernel_weight_mc", "hybrid"):
        _logger.info(f"  KW_MC_TWO_PASS = {KW_MC_TWO_PASS}")
        if KW_MC_TWO_PASS:
            _logger.info(f"  KW_MC_INJECT = {KW_MC_INJECT}")
        _logger.info(f"  KW_MC_MIN_WEIGHT = {KW_MC_MIN_WEIGHT}")
        _logger.info(f"  KW_MIN_POINTS_REF = {KW_MIN_POINTS_REF}")
    _logger.info(f"  DELTA_T_NS = {DELTA_T_NS}")
    _logger.info(f"  FLIGHT_PATH_M = {FLIGHT_PATH_M}")
    _logger.info(
        f"  DELTA_T_IS_FWHM = {DELTA_T_IS_FWHM}"
        f"  (sigma_E{'  = FWHM/2.3548' if DELTA_T_IS_FWHM else ' taken directly; pre-82 behaviour'})"
    )
    _logger.info(f"  N_SIGMA_CUTOFF = {N_SIGMA_CUTOFF}")
    _logger.info("")

    # -- Covariance Pipeline --
    _logger.info("  # Covariance Pipeline")
    _logger.info(f"  APPLY_COVARIANCE_CAP = {apply_covariance_cap}")
    if apply_covariance_cap:
        _logger.info(f"  MAX_RELATIVE_STD_CAP = {max_relative_std_cap} ({max_relative_std_cap*100:.0f}%)")
    _logger.info(f"  REGULARIZE_NEAR_ZERO = {regularize_near_zero}")
    if regularize_near_zero:
        _logger.info(f"  NEAR_ZERO_SNR_THRESHOLD = {near_zero_snr_threshold}")
        _logger.info(f"  NEAR_ZERO_N_NEIGHBORS = {near_zero_n_neighbors}")
    _logger.info(f"  APPLY_BETWEEN_EXP_FLOOR = {apply_between_exp_floor}")
    _logger.info(f"  APPLY_COV_POSTPROCESSING = {apply_cov_postprocessing}")
    if apply_cov_postprocessing:
        _dip_str = "None (disabled)" if SMOOTH_DIP_FRACTION is None else f"{SMOOTH_DIP_FRACTION} ({SMOOTH_DIP_FRACTION*100:.0f}%)"
        _spike_str = "None (disabled)" if SMOOTH_SPIKE_FACTOR is None else f"{SMOOTH_SPIKE_FACTOR} ({SMOOTH_SPIKE_FACTOR:.1f}x)"
        _caps_str = "None (disabled)" if ORDER_REL_STD_CAPS is None else str(ORDER_REL_STD_CAPS)
        _smooth_w = SMOOTH_DIAGONAL_WINDOW or 0
        _logger.info(f"  SMOOTH_MIN_REL_STD = {SMOOTH_MIN_REL_STD} ({SMOOTH_MIN_REL_STD*100:.1f}%)")
        _logger.info(f"  SMOOTH_DIP_FRACTION = {_dip_str}")
        _logger.info(f"  SMOOTH_SPIKE_FACTOR = {_spike_str}")
        _logger.info(f"  SMOOTH_DIP_N_NEIGHBORS = {SMOOTH_DIP_N_NEIGHBORS}")
        _logger.info(f"  SMOOTH_MEDIAN_FILL_THRESHOLD = {SMOOTH_MEDIAN_FILL_THRESHOLD}")
        _logger.info(f"  MG_SMOOTH_SPIKE_FACTOR = {MG_SMOOTH_SPIKE_FACTOR}")
        _logger.info(f"  MG_SMOOTH_DIP_N_NEIGHBORS = {MG_SMOOTH_DIP_N_NEIGHBORS}")
        _logger.info(f"  SMOOTH_DIAGONAL_WINDOW = {_smooth_w}")
        _logger.info(f"  ORDER_REL_STD_CAPS = {_caps_str}")
        _logger.info(f"  FORWARD_FILL_REL_STD = {FORWARD_FILL_REL_STD_ENABLED}")
    _logger.info(f"  APPLY_POSITIVITY_PROJECTION = {apply_positivity_projection}")
    if apply_positivity_projection:
        _logger.info(f"  POSITIVITY_CHECK_POINTS = {positivity_check_points}")
    _logger.info("")

    # -- Multigroup Covariance --
    if generate_multigroup_covariance:
        _logger.info("  # Multigroup Covariance")
        _logger.info(f"  GENERATE_MULTIGROUP_COVARIANCE = {generate_multigroup_covariance}")
        _logger.info(f"  MULTIGROUP_RHO_MIN = {multigroup_rho_min}")
        _logger.info(f"  MULTIGROUP_SIGMA_RATIO_MAX = {multigroup_sigma_ratio_max}")
        _logger.info(f"  MF34_COVARIANCE_TYPE = {mf34_covariance_type}")
        _logger.info(f"  MULTIGROUP_VARIANCE_PCT_MIN = {multigroup_variance_pct_min}")
        _logger.info(f"  MULTIGROUP_VARIANCE_PCT_MAX = {multigroup_variance_pct_max}")
        _logger.info(f"  MULTIGROUP_VARIANCE_RATIO_REF = {multigroup_variance_ratio_ref}")
        _logger.info(f"  MULTIGROUP_REGROUP_AFTER_SMOOTH = {multigroup_regroup_after_smooth}")
        _logger.info("")

    # -- Output: Pipeline A (fitting) --
    _logger.info("  # Output: Pipeline A (fitting)")
    _logger.info(f"  N_SAMPLES = {n_samples}")
    _logger.info(f"  BASE_SEED = {base_seed}")
    _logger.info(f"  GENERATE_NOMINAL_ENDF = {generate_nominal_endf}")
    _logger.info(f"  GENERATE_MC_MEAN_ENDF = {generate_mc_mean_endf}")
    _logger.info(f"  GENERATE_FITTING_SAMPLES = {generate_fitting_samples}")
    _logger.info(f"  GENERATE_FITTING_ACE = {generate_fitting_ace}")
    _logger.info(f"  SAVE_COVARIANCE_FILES = {save_covariance_files}")
    _logger.info(f"  SAVE_CORRELATION_MATRICES = {save_correlation_matrices}")
    _logger.info(f"  SAVE_TMC_PARQUET = {save_tmc_parquet}")
    _logger.info(f"  SAVE_RAW_KW_PARQUET = {save_raw_kw_parquet}")
    _logger.info(f"  SAVE_MULTIGROUP_DIAGNOSTICS_CSV = {save_multigroup_diagnostics_csv}")
    _logger.info(f"  SAVE_NOMINAL_FITS = {save_nominal_fits}")
    _logger.info("")

    # -- Output: Pipeline B (MF34 sampling) --
    _logger.info("  # Output: Pipeline B (MF34 sampling)")
    _logger.info(f"  GENERATE_MF34_SAMPLES = {generate_mf34_samples}")
    _logger.info(f"  GENERATE_MF34_ACE = {generate_mf34_ace}")
    if generate_mf34_samples:
        _logger.info(f"  SAMPLING_RESOLUTION = {sampling_resolution}")
        _logger.info(f"  MERGE_ORIGINAL_MF34 = {merge_original_mf34}")
        _logger.info(f"  SAMPLING_SPACE = {sampling_space}")
        _logger.info(f"  SAMPLING_DECOMPOSITION = {sampling_decomposition}")
        _logger.info(f"  SAMPLING_METHOD = {sampling_method}")
    _logger.info("")

    # -- ACE / NJOY --
    _any_ace = generate_fitting_ace or generate_mf34_ace
    if _any_ace:
        # Normalize temperatures: single float -> list
        if ace_temperatures is None:
            ace_temperatures = [293.6]
        elif isinstance(ace_temperatures, (int, float)):
            ace_temperatures = [float(ace_temperatures)]

        # Validate required parameters
        if not ace_njoy_exe or not os.path.isfile(ace_njoy_exe):
            raise FileNotFoundError(
                f"ACE_NJOY_EXE not found: {ace_njoy_exe}. "
                "Set ACE_NJOY_EXE to a valid NJOY executable path."
            )
        if not ace_library_name:
            raise ValueError(
                "ACE_LIBRARY_NAME must be provided when ACE generation is enabled "
                "(e.g., 'endfb81', 'jeff40')."
            )
        if not ace_temperatures:
            raise ValueError(
                "ACE_TEMPERATURES must be a non-empty list when ACE generation is enabled."
            )

        _logger.info("  # ACE / NJOY")
        _logger.info(f"  ACE_TEMPERATURES = {ace_temperatures}")
        _logger.info(f"  ACE_NJOY_EXE = {ace_njoy_exe}")
        _logger.info(f"  ACE_LIBRARY_NAME = {ace_library_name}")
        _logger.info(f"  ACE_NJOY_VERSION = {ace_njoy_version}")
        _logger.info(f"  ACE_XSDIR_FILE = {ace_xsdir_file}")
        _logger.info(f"  ACE_SKIP_EXISTING = {ace_skip_existing}")
        _logger.info("")

    # -- Runtime --
    _logger.info("  # Runtime")
    _logger.info(f"  N_PROCS = {n_procs}")
    _logger.info(f"  N_EFF_WARNING_THRESHOLD = {n_eff_warning_threshold}")
    _logger.info(f"  VERBOSE_DIAGNOSTICS = {verbose_diagnostics}")
    _logger.info("")

    _logger.info("#== END CONFIG ============================================================")
    _logger.info("")

    # Validate inputs
    if not os.path.exists(endf_file):
        _logger.error(f"[ERROR] [ENDF] File not found: {endf_file}", console=True)
        return

    if not os.path.isdir(exfor_directory):
        _logger.error(f"[ERROR] [EXFOR] Directory not found: {exfor_directory}", console=True)
        return

    # Step 1: Pre-load EXFOR data
    t_exfor_start = time.time()
    t_step = time.time()
    _logger.info("")
    _logger.info("#-- STEP 1: Load EXFOR data ------------------------------------------------")
    _logger.info(f"  [INFO] [EXFOR] Source: {exfor_source}")
    if exfor_source in ("database", "auto", "both"):
        _logger.info(f"  [INFO] [EXFOR] Database: {exfor_db_path or 'default'}")
    if exfor_directory:
        _logger.info(f"  [INFO] [EXFOR] JSON directory: {exfor_directory}")

    _logger.info(f"  [INFO] [EXFOR] Pre-loading EXFOR data (source={exfor_source})", console=True)

    try:
        exfor_cache, sorted_exfor_energies = load_exfor_with_new_api(
            exfor_directory=exfor_directory,
            db_path=exfor_db_path,
            source=exfor_source,
            target_zaid=target_zaid,
            projectile=target_projectile,
            mt=mt_number,
            energy_range=(energy_min_mev, energy_max_mev) if energy_min_mev and energy_max_mev else None,
            supplementary_json_files=supplementary_json_files,
            exclude_experiments=exclude_experiments,
            logger=_logger,
        )
        t_exfor_elapsed = time.time() - t_step

        n_exfor_files = sum(len(entries) for entries in exfor_cache.values())
        _logger.info(f"  [INFO] [EXFOR] Loaded {n_exfor_files} experiments at {len(sorted_exfor_energies)} unique energies")
        _logger.info(f"  [INFO] [EXFOR] Energy range: [{min(sorted_exfor_energies):.4f}, {max(sorted_exfor_energies):.4f}] MeV")
        _logger.info(f">> exfor_experiments = {n_exfor_files}")
        _logger.info(f">> exfor_energies = {len(sorted_exfor_energies)}")

        # Manifest-flag summary: how many datasets/points came from each
        # manifest source bucket (curated/uncurated/default/excluded), plus
        # any per-dataset manifest application failures from the cache build.
        try:
            from collections import Counter as _Counter
            from scripts.exfor_utils import get_last_manifest_stats as _get_mstats
            flag_datasets: Dict[str, set] = {}
            flag_points: _Counter = _Counter()
            for _e_mev, _entries in exfor_cache.items():
                for _df, _meta in _entries:
                    _flag = str(_meta.get('uncertainty_manifest_flag', 'default'))
                    flag_datasets.setdefault(_flag, set()).add(
                        f"{_meta.get('entry')}/{_meta.get('subentry')}"
                    )
                    flag_points[_flag] += int(len(_df))
            _mstats = _get_mstats()
            _logger.info("  [INFO] [EXFOR] Manifest flag summary:")
            for _flag in sorted(set(list(flag_points.keys()) +
                                    ['curated', 'uncurated', 'default', 'excluded'])):
                _ndsets = len(flag_datasets.get(_flag, set()))
                _npts = flag_points.get(_flag, 0)
                _logger.info(f"    flag={_flag:<12s} datasets={_ndsets:>4d}  points={_npts:>6d}")
            _logger.info(
                f"    manifest_failed: attempted={_mstats.get('attempted', 0)}, "
                f"failed={_mstats.get('failed', 0)}"
            )
            for _ent, _sub, _exc in (_mstats.get('failures', []) or [])[:10]:
                _logger.warning(f"    manifest_failed: {_ent}/{_sub} — {_exc}")
        except Exception as _e:
            _logger.warning(f"Manifest flag summary failed: {type(_e).__name__}: {_e}")

        # Manifest excluded-flag desync check: any dataset the manifest marks
        # `flag: excluded` should also appear in EXCLUDE_EXPERIMENTS, otherwise
        # the manifest and the run config are out of sync. Warn-only; do not
        # raise (the run can still be valid if the desync is intentional).
        try:
            from scripts.uncertainty_manifest import load_manifest as _load_manifest
            from scripts.exfor_utils import _parse_exclusion_list, _is_experiment_excluded
            _m = _load_manifest()
            _excl_patterns = _parse_exclusion_list(exclude_experiments)
            _desync = []
            for _dsid, _entry in (_m.get('datasets') or {}).items():
                if _entry.get('flag') != 'excluded':
                    continue
                # _dsid is "ENTRYSUB" format (e.g. "32246002"); split into entry/subentry.
                if len(_dsid) >= 8:
                    _ent, _sub = _dsid[:5], _dsid[5:]
                else:
                    _ent, _sub = _dsid, ''
                if not _is_experiment_excluded(_ent, _sub, _excl_patterns):
                    _desync.append(_dsid)
            if _desync:
                _logger.warning(
                    f"  [WARN] [EXFOR] {len(_desync)} dataset(s) flagged 'excluded' in manifest "
                    f"but not in EXCLUDE_EXPERIMENTS: {_desync[:10]}"
                    + (" ..." if len(_desync) > 10 else "")
                )
        except Exception as _e:
            _logger.warning(f"Manifest excluded-flag desync check failed: {type(_e).__name__}: {_e}")

        _logger.info(f"#-- END STEP 1 (elapsed: {t_exfor_elapsed:.2f}s) -------------------------------------")
        _logger.info(f"  [INFO] [EXFOR] Loaded {n_exfor_files} experiments in {t_exfor_elapsed:.1f}s", console=True)
    except Exception as e:
        _logger.error(f"[ERROR] [EXFOR] Failed to load data: {str(e)}", console=True)
        return

    # Step 2: Read ENDF and extract energy grid
    t_step = time.time()
    _logger.info("")
    _logger.info("#-- STEP 2: Read ENDF file -------------------------------------------------")

    try:
        endf = read_endf(endf_file)
        mf4 = endf.get_file(4)

        if mf4 is None:
            _logger.error("[ERROR] [ENDF] MF4 section not found in ENDF file", console=True)
            return

        mt_data = mf4.sections.get(mt_number)
        if mt_data is None:
            _logger.error(f"[ERROR] [ENDF] MT{mt_number} not found in MF4", console=True)
            return

        if not isinstance(mt_data, (MF4MTLegendre, MF4MTMixed)):
            _logger.error(f"[ERROR] [ENDF] MT{mt_number} is not Legendre or Mixed type (LTT={mt_data._ltt})", console=True)
            return

        energies_ev = np.array(mt_data.legendre_energies)
        original_coeffs = mt_data.legendre_coefficients

        _logger.info(f"  [INFO] [ENDF] Found {len(energies_ev)} energy points in MF4/MT{mt_number}")
        _logger.info(f">> endf_energy_points = {len(energies_ev)}")
        _logger.info(f"#-- END STEP 2 (elapsed: {time.time() - t_step:.2f}s) -------------------------------------")

    except Exception as e:
        _logger.error(f"[ERROR] [ENDF] Failed to read file: {str(e)}", console=True)
        return

    # Step 3: Compute energy bins
    t_step = time.time()
    _logger.info("")
    _logger.info("#-- STEP 3: Compute energy bins --------------------------------------------")

    if energy_grid_source == "union":
        grid_energies_ev = build_union_energy_grid(
            exfor_cache=exfor_cache,
            subentries=union_grid_subentries,
            energy_min_mev=energy_min_mev,
            energy_max_mev=energy_max_mev,
        )
        _logger.info(f"  Using union grid: {len(grid_energies_ev)} points")
    else:
        grid_energies_ev = energies_ev
        _logger.info(f"  Using ENDF grid: {len(grid_energies_ev)} points")

    energy_bins = compute_energy_bins_with_tof_resolution(
        energies_ev=grid_energies_ev,
        energy_min_mev=energy_min_mev,
        energy_max_mev=energy_max_mev,
        delta_t_ns=DELTA_T_NS,
        flight_path_m=FLIGHT_PATH_M,
        reference_grid_ev=energies_ev if energy_grid_source == "union" else None,
        delta_t_is_fwhm=DELTA_T_IS_FWHM,
    )

    if not energy_bins:
        _logger.error(f"No energy points in range [{energy_min_mev}, {energy_max_mev}] MeV", console=True)
        return

    # Map each bin to nearest ENDF energy (for original coeffs and ENDF writing)
    for bin_info in energy_bins:
        idx = int(np.argmin(np.abs(energies_ev - bin_info.energy_ev)))
        bin_info.endf_index = idx
        if idx < len(original_coeffs):
            bin_info.original_coeffs = list(original_coeffs[idx])

    _logger.info(f">> energy_bins = {len(energy_bins)}")
    _logger.info(f"#-- END STEP 3 (elapsed: {time.time() - t_step:.2f}s) -------------------------------------")
    _logger.info(f"  [INFO] [ENDF] Processing {len(energy_bins)} energy bins", console=True)

    # Step 4: Nominal fits
    t_step = time.time()
    _logger.info("")
    _logger.info("#-- STEP 4: Nominal fits ---------------------------------------------------")

    nominal_results = perform_nominal_fits(
        energy_bins=energy_bins,
        exfor_cache=exfor_cache,
        sorted_energies=sorted_exfor_energies,
        max_degree=max_degree,
        select_degree=select_degree,
        ridge_lambda=ridge_lambda,
        m_proj_u=m_proj_u,
        m_targ_u=m_targ_u,
        use_band_discrepancy=use_band_discrepancy,
        min_points_per_band=min_points_per_band,
        max_band_scale=max_band_scale,
        tau_smoothing_window=tau_smoothing_window,
        n_eff_warning_threshold=n_eff_warning_threshold,
        min_degree_for_averaging=min_degree_for_averaging,
        exclude_experiments=exclude_experiments,
        min_relative_uncertainty=min_relative_uncertainty,
        tau_prior_floor=tau_prior_floor,
        tau_prior_neff_threshold=tau_prior_neff_threshold,
        tau_prior_percentile=tau_prior_percentile,
        angular_quality_gate=ANGULAR_QUALITY_GATE,
        min_angular_points=MIN_ANGULAR_POINTS,
        min_bands_covered=MIN_BANDS_COVERED,
        max_bin_expansion=MAX_BIN_EXPANSION,
        rerun_aicc_post_tau=RERUN_AICC_POST_TAU,
        use_gls_kernel=USE_GLS_KERNEL,
        tau_refit_use_gls=TAU_REFIT_USE_GLS,
        sigma_norm_systematic=sigma_norm_systematic,
        membership_k_sigma=MEMBERSHIP_K_SIGMA,
        logger=_logger,
    )

    n_with_data = sum(1 for nr in nominal_results if nr.has_data)
    _logger.info(f"  [INFO] [FIT] Bins with EXFOR data: {n_with_data}/{len(nominal_results)}")
    _logger.info(f">> bins_with_data = {n_with_data}")
    _logger.info(f">> bins_total = {len(nominal_results)}")

    # Step 4b: Interpolate missing bins (NEVER use original ENDF coefficients)
    n_missing = len(nominal_results) - n_with_data
    if n_missing > 0:
        _logger.info(f"  [INFO] [FIT] Interpolating {n_missing} missing energy bins")

        nominal_results = interpolate_missing_nominal_fits(
            nominal_results=nominal_results,
            logger=_logger,
        )

        n_with_data_after = sum(1 for nr in nominal_results if nr.has_data)
        n_interpolated = sum(1 for nr in nominal_results if nr.interpolated)
        _logger.info(f"  [INFO] [FIT] After interpolation: {n_with_data_after}/{len(nominal_results)} bins have coefficients")
        _logger.info(f"  [INFO] [FIT] ({n_interpolated} interpolated, {n_with_data_after - n_interpolated} from EXFOR)")
        _logger.info(f">> interpolated_bins = {n_interpolated}")
        _warning_counts['interpolated_bins'] = n_interpolated
    else:
        n_interpolated = 0

    # Log experiment summary
    log_experiments_summary(nominal_results, logger=_logger)

    # IC solver path: GLS for multi-experiment bins, WLS fallback for
    # single-experiment bins (see resample_AD.py:1745-1760 for rationale).
    # Skipped bins (has_data=False) and interpolated bins are excluded.
    if USE_GLS_KERNEL:
        n_gls = sum(
            1 for nr in nominal_results
            if nr.has_data and not nr.interpolated and len(nr.experiments_info) >= 2
        )
        n_wls = sum(
            1 for nr in nominal_results
            if nr.has_data and not nr.interpolated and len(nr.experiments_info) == 1
        )
        _logger.info(
            f"  [INFO] [FIT] IC solver: GLS={n_gls} bins, "
            f"WLS-fallback (1-experiment)={n_wls} bins"
        )
    else:
        n_fitted = sum(1 for nr in nominal_results if nr.has_data and not nr.interpolated)
        _logger.info(f"  [INFO] [FIT] IC solver: WLS={n_fitted} bins (USE_GLS_KERNEL=False)")

    if save_nominal_fits:
        rows = []
        # Bin metadata by index, so the per-bin TOF resolution and edges travel
        # with the fits. rebuild_mf33.py needs them to reconstruct the folding
        # kernel offline without re-deriving the grid from the EXFOR database.
        _bin_by_idx_nom = {b.index: b for b in energy_bins}
        for r in nominal_results:
            coeffs = np.full(max_degree + 1, np.nan)
            if r.nominal_coeffs is not None and len(r.nominal_coeffs) > 0:
                n_c = min(len(r.nominal_coeffs), max_degree + 1)
                coeffs[:n_c] = r.nominal_coeffs[:n_c]
            tau = r.tau_info or {}
            _eb = _bin_by_idx_nom.get(r.energy_index)
            row = {
                'energy_mev': r.energy_mev,
                'sigma_E_mev': getattr(_eb, 'sigma_E_mev', np.nan),
                'bin_lower_mev': getattr(_eb, 'bin_lower_mev', np.nan),
                'bin_upper_mev': getattr(_eb, 'bin_upper_mev', np.nan),
                'energy_index': r.energy_index,
                'endf_index': r.endf_index,
                'has_data': r.has_data,
                'interpolated': r.interpolated,
                'expanded_bins': r.expanded_bins,
                'frozen_degree': r.frozen_degree,
                'chi2_red': r.chi2_red,
                'tau_F': tau.get('tau_F', 1.0),
                'tau_M': tau.get('tau_M', 1.0),
                'tau_B': tau.get('tau_B', 1.0),
                'mc_order_cap': r.mc_order_cap,
                'n_pts': len(r.exfor_df) if r.exfor_df is not None else 0,
                'n_eff': (r.kernel_diagnostics.n_eff
                          if r.kernel_diagnostics is not None else np.nan),
                'aicc_weights_json': (
                    json.dumps({int(k): float(v) for k, v in r.degree_weights.items()})
                    if r.degree_weights else None
                ),
            }
            row.update({f'c_{l}': float(coeffs[l]) for l in range(max_degree + 1)})
            # PHASE 3: under SHIP_MIXTURE_MEAN the c_* above are the MIXTURE
            # MEAN and are what the ENDF ships. win_c_* carries the winning
            # degree's coefficients so this run can still be compared
            # column-for-column against every run before it.
            _wc = r.winner_coeffs
            row.update({
                f'win_c_{l}': (float(_wc[l]) if _wc is not None and l < len(_wc)
                               else float('nan'))
                for l in range(max_degree + 1)
            })
            # RESEARCH FORK: the model-averaged central and inclusion
            # probabilities, as ADDITIONAL columns.
            if WRITE_AVERAGING_DIAGNOSTICS:
                if r.averaging_diag is not None:
                    row.update(r.averaging_diag)
                else:
                    # Interpolated bins never went through the fitting loop, so
                    # they carry no candidate set and no data to score against.
                    row.update(compute_averaging_diagnostics(
                        None, None, max_degree, DEGREE_WEIGHT_FLOOR))
                    row.update({c: float('nan') for c in CHI2_COLUMNS})
            rows.append(row)
        df_nom = pd.DataFrame(rows)
        out_nom = output_path / 'nominal_fits.parquet'
        df_nom.to_parquet(out_nom, index=False)
        _logger.info(f"  Saved per-bin nominal fits ({len(df_nom)} rows) to {out_nom.name}")
        if WRITE_AVERAGING_DIAGNOSTICS and 'avg_a_1' in df_nom.columns:
            _n_avg = int(df_nom['avg_a_1'].notna().sum())
            _n_int = int(df_nom['interpolated'].sum()) if 'interpolated' in df_nom else 0
            _logger.info(
                f"  [AVG] model-averaged central written for {_n_avg}/{len(df_nom)} bins "
                f"(weight floor {DEGREE_WEIGHT_FLOOR}); {len(df_nom) - _n_avg} NaN "
                f"({_n_int} interpolated). Median effective models: "
                f"{df_nom['n_eff_models'].median():.2f}",
                console=True,
            )
            # THE Phase-2 question: does the averaged central still describe the
            # data? The winner minimises AIC = chi2 + 2k, so no other combination
            # can beat it in-sample — a ratio slightly above 1 is expected and is
            # not a defect. The magnitude is what decides whether averaging is
            # affordable.
            if 'chi2_pp_ratio' in df_nom.columns:
                _r = df_nom['chi2_pp_ratio'].to_numpy(dtype=float)
                _r = _r[np.isfinite(_r)]
                if _r.size:
                    _worse = float(np.mean(_r > 1.0)) * 100.0
                    _logger.info(
                        f"  [AVG] data-space chi2 per point, averaged/winner: "
                        f"median {np.median(_r):.4f}, p90 {np.percentile(_r, 90):.4f}, "
                        f"max {_r.max():.4f}; worse in {_worse:.1f}% of bins. "
                        f"(>1 expected: the winner is the penalised chi2 minimiser.)",
                        console=True,
                    )

    _logger.info(f"#-- END STEP 4 (elapsed: {time.time() - t_step:.2f}s) -------------------------------------")
    _logger.info(f"  [INFO] [FIT] Nominal fits: {n_with_data}/{len(nominal_results)} with data, {n_interpolated} interpolated", console=True)

    if STOP_AFTER_NOMINAL_FITS:
        # Steps 1-4 cost ~2 minutes against ~5 h for the MC, so this is the
        # cheap way to see how a config change lands on the fits (n_eff, tau,
        # degree selection) before committing to a production run.
        _logger.info(
            "  STOP_AFTER_NOMINAL_FITS=True — stopping before MC sampling. "
            f"nominal_fits.parquet is in {output_dir}",
            console=True,
        )
        return

    # Step 5: MC sampling
    t_step = time.time()
    _logger.info("")
    _logger.info("#-- STEP 5: MC sampling ----------------------------------------------------")
    _logger.info(f"  [INFO] [MC] Generating {n_samples} samples, method={CORRELATION_METHOD}")

    _prebuilt_gaussian_cov = None
    _prebuilt_mc_mean = None

    # Phase-2 magnitude channel: defined here so the MF33 write block downstream
    # is safe under every CORRELATION_METHOD (only the KW path can turn it on).
    record_c0_channel = False
    mf33_rel_cov_fine = None
    mf33_c0_nom = None
    mf33_energy_grid_ev = None
    mf33_sigma_host_bin = None
    _mf33_products = None
    _mf33_pendf_path = None

    if CORRELATION_METHOD == "gaussian":
        # Per-bin stochastic MC → Gaussian parametric correlations → Cholesky samples
        _logger.info("  " + "=" * 60)
        _logger.info("  Method: Gaussian decay correlation + Cholesky resampling")
        _logger.info("  " + "=" * 60)
        if N_PROCS > 1:
            _logger.info(f"  Using {N_PROCS} parallel processes over bins")

        bin_args_list = []
        for nr in nominal_results:
            if not nr.has_data:
                continue
            bin_args_list.append((
                nr.energy_index,
                nr.frozen_degree,
                nr.nominal_coeffs,
                nr.interpolated,
                nr.exfor_df,
                nr.kernel_weights,
                nr.degree_weights,
                n_samples,
                base_seed,
                max_degree,
                ridge_lambda,
                RIDGE_POWER,
                DF_METHOD,
                use_band_discrepancy,
                min_points_per_band,
                max_band_scale,
                USE_DEGREE_SAMPLING_IN_MC,
                RESCALE_UNC_BY_CHI2,
                ALLOW_SHRINK_UNC,
                FREEZE_C0,
                sigma_norm_systematic,
                sigma_norm_common_mode,
                NORM_DIST,
                MAX_SAMPLE_ORDER,
                apply_positivity_projection,
                positivity_check_points,
                nr.tau_info,
                nr.mc_order_cap,
                # 29th: Phase-2 magnitude channel — off on this path.
                False,
                # 30th: Phase-3 mixture config, or None for the legacy
                # pooled degree draw (Gate A).
                _MIXTURE_CFG,
            ))

        if N_PROCS > 1:
            with Pool(N_PROCS) as pool:
                bin_results = pool.map(_mc_one_bin, bin_args_list)
        else:
            bin_results = [_mc_one_bin(a) for a in bin_args_list]

        _log_mc_bin_failures(bin_results, nominal_results, _warning_counts, _logger)
        all_samples_stochastic = {s_idx: {} for s_idx in range(n_samples)}
        mixture_by_bin = {}
        for rec in bin_results:
            energy_idx, is_interpolated, results_by_sample, success, error_msg, _c0 = rec[:6]
            _mix = rec[6] if len(rec) > 6 else {}
            if _mix:
                mixture_by_bin[energy_idx] = _mix
            for s_idx, endf_coeffs in results_by_sample.items():
                all_samples_stochastic[s_idx][energy_idx] = endf_coeffs

        _logger.info(f"  Per-bin stochastic pass completed: {len(bin_args_list)} bins")

        # Build Gaussian correlation covariance and generate Cholesky samples
        _logger.info("  Building Gaussian correlation covariance from stochastic pass")
        energy_indices_for_gauss = [nr.energy_index for nr in nominal_results if nr.has_data]

        # Build valid-parameter mask: only parameters actually fitted are valid
        nr_by_idx = {nr.energy_index: nr for nr in nominal_results}
        _valid_mask = np.zeros(len(energy_indices_for_gauss) * max_degree, dtype=bool)
        for ie, e_idx in enumerate(energy_indices_for_gauss):
            nr = nr_by_idx[e_idx]
            n_valid = bin_valid_orders(nr, max_degree)
            if MAX_SAMPLE_ORDER is not None:
                n_valid = min(n_valid, MAX_SAMPLE_ORDER)
            for l in range(n_valid):
                _valid_mask[ie * max_degree + l] = True
        n_invalid = int(np.sum(~_valid_mask))
        _logger.info(f"  Valid-parameter mask: {int(np.sum(_valid_mask))}/{len(_valid_mask)} valid "
                     f"({n_invalid} zeroed for absent higher-order coefficients)")

        # Compute stochastic covariance (for per-bin variances and cross-order correlations)
        _snr_thr = near_zero_snr_threshold if regularize_near_zero else 0.0
        cov_stochastic_pass2, _, _, mc_mean_stochastic = compute_covariance_from_samples(
            all_samples=all_samples_stochastic,
            energy_indices=energy_indices_for_gauss,
            max_order=max_degree,
            valid_mask=_valid_mask,
            snr_threshold=_snr_thr,
            n_neighbors=near_zero_n_neighbors,
            logger=_logger,
        )

        # Build full covariance with Gaussian energy correlations
        _gaussian_cov_full = build_gaussian_correlation_covariance(
            cov_stochastic=cov_stochastic_pass2,
            energy_bins=energy_bins,
            energy_indices=energy_indices_for_gauss,
            max_order=max_degree,
            logger=_logger,
            valid_mask=_valid_mask,
        )

        # Generate properly correlated samples via Cholesky
        _logger.info(f"  Generating {n_samples} Cholesky samples")
        all_samples = generate_cholesky_samples(
            cov_full=_gaussian_cov_full,
            mean_params=mc_mean_stochastic,
            energy_indices=energy_indices_for_gauss,
            max_order=max_degree,
            n_samples=n_samples,
            seed=base_seed,
            logger=_logger,
        )

        # Store pre-built covariance for Step 7
        _prebuilt_gaussian_cov = _gaussian_cov_full
        _prebuilt_mc_mean = mc_mean_stochastic

    elif CORRELATION_METHOD in ("kernel_weight_mc", "hybrid"):
        # Kernel-weighted multi-bin MC → correlations from shared perturbed datasets
        _is_hybrid = (CORRELATION_METHOD == "hybrid")
        _logger.info("  " + "=" * 60)
        _logger.info(f"  Method: {'Hybrid KW+Gaussian blend' if _is_hybrid else 'Kernel-weight MC correlations'}")
        _logger.info(f"  Two-pass mode: {KW_MC_TWO_PASS}")
        if KW_MC_TWO_PASS:
            _mode = "inject + Higham repair (legacy)" if KW_MC_INJECT else "congruence transform (PSD by construction)"
            _logger.info(f"  Pass-2 combine: {_mode}")
        _logger.info(f"  Min overlap weight: {KW_MC_MIN_WEIGHT}")
        _logger.info("  " + "=" * 60)

        # Load per-experiment TOF parameters for overlap weight computation
        tof_params_cache = {}
        if TOF_PARAMETERS_FILE:
            try:
                tof_params_cache = load_tof_parameters_file(TOF_PARAMETERS_FILE)
                _logger.info(f"  Loaded TOF parameters for {len(tof_params_cache)} experiments")
            except FileNotFoundError:
                _logger.warning(f"[WARN] [MC] TOF parameters file not found: {TOF_PARAMETERS_FILE}")

        # Precompute overlap weights for all datasets across all bins
        overlap_weights = precompute_overlap_weights(
            nominal_results=nominal_results,
            energy_bins=energy_bins,
            min_weight=KW_MC_MIN_WEIGHT,
            tof_params_cache=tof_params_cache if tof_params_cache else None,
            default_flight_path_m=FLIGHT_PATH_M,
            default_time_resolution_ns=DELTA_T_NS,
            default_delta_t_is_fwhm=DELTA_T_IS_FWHM,
            logger=_logger,
        )

        n_datasets_total = sum(len(dsets) for dsets in overlap_weights.values())
        _logger.info(f"  Overlap weights computed: {n_datasets_total} (dataset, bin) pairs")

        # Run kernel-weight MC (all bins coupled via shared perturbations)
        # Phase-2 magnitude channel needs both passes (Pass-1 correlations,
        # Pass-2 variances). If the knob is on but two-pass is off, warn and
        # skip rather than emit a half-built covariance.
        record_c0_channel = (GENERATE_MF3_MF33 > 0) and KW_MC_TWO_PASS
        if GENERATE_MF3_MF33 > 0 and not KW_MC_TWO_PASS:
            _logger.warning(
                "  [MF33] GENERATE_MF3_MF33 is on but KW_MC_TWO_PASS=False; "
                "the fixed-shape c0 channel needs both passes — skipping.",
                console=True,
            )
        _kw_out = run_mc_with_kernel_weights(
            nominal_results=nominal_results,
            energy_bins=energy_bins,
            overlap_weights=overlap_weights,
            n_samples=n_samples,
            n_workers=N_PROCS,
            sigma_norm=sigma_norm_systematic,
            sigma_norm_common_mode=sigma_norm_common_mode,
            norm_dist=NORM_DIST,
            max_degree=max_degree,
            ridge_lambda=ridge_lambda,
            base_seed=base_seed,
            use_band_discrepancy=use_band_discrepancy,
            min_points_per_band=min_points_per_band,
            max_band_scale=max_band_scale,
            freeze_c0=FREEZE_C0,
            # Pin c0 at the per-bin nominal across all KW samples — matches
            # _mc_one_bin (Pass 2), which gets the same effect implicitly because
            # SLC fits c0 from the unperturbed df before drawing MC samples. With
            # c0 floating per sample in Pass 1 and pinned in Pass 2, the
            # congruence-merge in the combine mixed correlations and stds taken
            # under different noise models. Aligns the two passes.
            fix_c0_at_nominal=True,
            # Use σ_total² = (τ·σ_stat)² + σ_sys² for MC fit weights — matches
            # _mc_one_bin (Pass 2), which already passes sys_unc_col='_sigma_sys_abs'
            # to SLC. Without this, Pass 1 fit weights were stat-only and points
            # with small σ_stat but large σ_sys were over-weighted in the
            # cross-bin correlations.
            sys_aware_mc_fit=True,
            max_sample_order=MAX_SAMPLE_ORDER,
            apply_positivity_projection=apply_positivity_projection,
            positivity_check_points=positivity_check_points,
            max_experiment_weight_fraction=MAX_EXP_WEIGHT_FRAC_BIN,
            min_relative_uncertainty=MIN_RELATIVE_UNCERTAINTY,
            band_aware_ess=BAND_AWARE_ESS,
            record_c0_channel=record_c0_channel,
            logger=_logger,
        )
        if record_c0_channel:
            kw_samples, c0_samples_kw = _kw_out
        else:
            kw_samples, c0_samples_kw = _kw_out, None

        if KW_MC_TWO_PASS:
            _logger.info("  Two-pass: running per-bin MC for variance")

            # Pass 2: per-bin MC for variance
            bin_args_list = []
            for nr in nominal_results:
                if not nr.has_data:
                    continue
                bin_args_list.append((
                    nr.energy_index,
                    nr.frozen_degree,
                    nr.nominal_coeffs,
                    nr.interpolated,
                    nr.exfor_df,
                    nr.kernel_weights,
                    nr.degree_weights,
                    n_samples,
                    base_seed,
                    max_degree,
                    ridge_lambda,
                    RIDGE_POWER,
                    DF_METHOD,
                    use_band_discrepancy,
                    min_points_per_band,
                    max_band_scale,
                    USE_DEGREE_SAMPLING_IN_MC,
                    RESCALE_UNC_BY_CHI2,
                    ALLOW_SHRINK_UNC,
                    FREEZE_C0,
                    sigma_norm_systematic,
                    sigma_norm_common_mode,
                    NORM_DIST,
                    MAX_SAMPLE_ORDER,
                    apply_positivity_projection,
                    positivity_check_points,
                    nr.tau_info,
                    nr.mc_order_cap,
                    # 29th field: Phase-2 magnitude-channel recording flag.
                    record_c0_channel,
                    # 30th field: Phase-3 mixture config, or None for the
                    # legacy pooled degree draw (Gate A).
                    _MIXTURE_CFG,
                ))

            if N_PROCS > 1:
                with Pool(N_PROCS) as pool:
                    bin_results = pool.map(_mc_one_bin, bin_args_list)
            else:
                bin_results = [_mc_one_bin(a) for a in bin_args_list]

            _log_mc_bin_failures(bin_results, nominal_results, _warning_counts, _logger)
            all_samples_perbin = {s_idx: {} for s_idx in range(n_samples)}
            c0_samples_perbin = {s_idx: {} for s_idx in range(n_samples)} if record_c0_channel else None
            mixture_by_bin = {}
            for rec in bin_results:
                (energy_idx, is_interpolated, results_by_sample, success,
                 error_msg, c0_by_sample) = rec[:6]
                _mix = rec[6] if len(rec) > 6 else {}
                if _mix:
                    mixture_by_bin[energy_idx] = _mix
                for s_idx, endf_coeffs in results_by_sample.items():
                    all_samples_perbin[s_idx][energy_idx] = endf_coeffs
                if record_c0_channel:
                    for s_idx, c0_val in c0_by_sample.items():
                        c0_samples_perbin[s_idx][energy_idx] = c0_val

            _logger.info(f"  Per-bin stochastic pass completed: {len(bin_args_list)} bins")

            # Persist raw KW multi-bin MC samples (Pass-1 marginals + full
            # non-Gaussian joint distribution).  Research/diagnostic only —
            # downstream TMC consumers should prefer the TMC parquet.
            if save_raw_kw_parquet:
                try:
                    raw_kw_path = save_all_legendre_coefficients(
                        nominal_results=nominal_results,
                        all_samples=kw_samples,
                        output_dir=str(output_path),
                        max_degree=max_degree,
                        filename='legendre_samples_raw_kw.parquet',
                    )
                    _logger.info(f"  [INFO] [MC] Raw KW samples saved to: {raw_kw_path}")
                except Exception as e:
                    _logger.error(
                        f"[ERROR] [MC] Failed to save raw KW samples: {str(e)}",
                        console=True,
                    )

            # Combine: correlations from kw_samples, variance from per-bin
            energy_indices_kw = [nr.energy_index for nr in nominal_results if nr.has_data]
            nr_by_idx_kw = {nr.energy_index: nr for nr in nominal_results}

            # --- Phase-2 elastic magnitude channel: fixed-shape c0 → MF33 ----
            # Pass-1 shared draws give cross-bin correlations, Pass-2 per-bin
            # draws give marginal variances — the same congruence combine as the
            # Legendre-vector channel, on the scalar c0 per bin. Fine-grid
            # relative covariance is stashed for the MF33 write + sidecars, and
            # the channel gets its own adaptive multigroup grid (independent of
            # the MF34 one) built alongside it.
            mf33_rel_cov_fine = None
            mf33_c0_nom = None
            mf33_energy_grid_ev = None
            mf33_sigma_host_bin = None
            _mf33_products = None
            _mf33_pendf_path = None
            if record_c0_channel:
                try:
                    _bin_by_idx_mf33 = {b.index: b for b in energy_bins}
                    mf33_c0_nom = np.array([
                        float(nr_by_idx_kw[e].nominal_coeffs[0]) for e in energy_indices_kw
                    ])
                    _mf33_rel_dcs, _mf33_cov_abs, _df_c0, _mf33_diag = build_mf33_channel(
                        c0_samples_kw, c0_samples_perbin,
                        energy_indices_kw, mf33_c0_nom, n_samples,
                    )
                    # Fine energy grid (eV): lower edges of the has-data bins
                    # plus the last upper edge — hard-asserts bin adjacency (a
                    # gapped grid would be a semantically wrong ENDF grid).
                    _vb = [_bin_by_idx_mf33[e] for e in energy_indices_kw]
                    mf33_energy_grid_ev = contiguous_grid_from_bins(_vb)

                    # Phase-2 magnitude<->shape correlation (roadmap §9.4.1).
                    # Diagnostic sidecar only — nothing downstream reads it, so
                    # the shipped ENDF is bit-identical with this on or off.
                    if COMPUTE_MF33_MF34_CROSS and c0_samples_perbin is not None:
                        try:
                            _xc = compute_mf33_mf34_cross(
                                all_samples_perbin, c0_samples_perbin,
                                energy_indices_kw, max_degree,
                            )
                            np.save(output_path / "mf33_mf34_cross_covariance.npy", _xc["cov"])
                            np.save(output_path / "mf33_mf34_cross_correlation.npy", _xc["rho"])
                            np.save(output_path / "mf33_mf34_cross_n_pairs.npy", _xc["n_pairs"])
                            # The COMPLETE block, Cov(c0(E_i), a_l(E_j)) — roadmap
                            # §10.1.6. This is the one the chi^2 must consume; the
                            # within-bin sidecar above stays for continuity with
                            # runs 86/87 and for the rho diagnostics.
                            _xf = _xc.get("cov_full")
                            if _xf is not None:
                                np.save(
                                    output_path / "mf33_mf34_cross_covariance_full.npy", _xf
                                )
                                _wb = np.abs(np.einsum("iil->il", _xf)).mean()
                                _tot = np.abs(_xf).mean()
                                _logger.info(
                                    f"  [XCORR] FULL cross block written: {_xf.shape} "
                                    f"over {_xc['n_common']} replicas complete in both "
                                    f"channels across all bins "
                                    f"({100.0 * _xc['n_common'] / max(len(c0_samples_perbin), 1):.1f} %). "
                                    f"mean|within-bin|={_wb:.3e}, mean|all|={_tot:.3e}"
                                )
                            else:
                                _logger.warning(
                                    "  [XCORR] FULL cross block NOT written: only "
                                    f"{_xc.get('n_common', 0)} replicas are complete in "
                                    "both channels across every bin. A partial block is "
                                    "what §10.1.5 showed is not a valid covariance, so "
                                    "it is withheld rather than filled."
                                )
                            _nb = _xc["rho"].shape[0]
                            _logger.info(
                                f"  [XCORR] Cov(c0, a_l) over shared Pass-2 replicas: "
                                f"{_nb} bins, paired draws min/median "
                                f"{int(np.min(_xc['n_pairs']))}/{int(np.median(_xc['n_pairs']))}"
                            )
                            # THE number this whole exercise is for: the sign.
                            # §9.4.1 bounded |rho| <= 1 but could not sign it.
                            _logger.info(
                                "  [XCORR] median rho by order: "
                                + ", ".join(
                                    f"a_{l+1} {np.nanmedian(_xc['rho'][:, l]):+.3f}"
                                    for l in range(max_degree)
                                )
                            )
                            _finite = np.isfinite(_xc["rho"])
                            _logger.info(
                                f"  [XCORR] defined in {int(_finite.sum())}/{_finite.size} "
                                "(bin, order) slots; the rest are orders restored "
                                "from nominal, where rho is undefined, not zero."
                            )
                        except Exception as _e_xc:
                            _logger.warning(f"  [XCORR] cross-covariance failed: {_e_xc}")

                    # Completeness + Pass-1 correlation inspection (warn-only).
                    _p1c = _mf33_diag["p1_finite_per_bin"]
                    _p2c = _mf33_diag["p2_finite_per_bin"]
                    _logger.info(
                        "  [MF33] Sample completeness per bin: "
                        f"Pass-1 min/median {int(np.min(_p1c))}/{int(np.median(_p1c))}, "
                        f"Pass-2 min/median {int(np.min(_p2c))}/{int(np.median(_p2c))} "
                        f"(of {n_samples})"
                    )
                    _corr_eig = _mf33_diag["corr_pass1_min_eig"]
                    if _corr_eig < -1e-8:
                        _logger.warning(
                            f"  [MF33] Pass-1 pairwise-complete correlation not PSD "
                            f"(min eig {_corr_eig:.2e}) — warn only, not repaired.",
                            console=True,
                        )

                    # Recentre the relative covariance on the HOST central: the
                    # DCS analysis infers the absolute covariance C_abs; dividing
                    # by the host means keeps the absolute uncertainty claim
                    # intact for users who multiply the relative MF33 by MF3.
                    #
                    # The denominator is the RECONR-reconstructed cross section
                    # folded through each bin's TOF kernel, NOT a box average of
                    # File 3 — File 3 is zero inside the resolved resonance range
                    # and a 1 keV box is not what the fit measured. See the
                    # MF33_PENDF_* config block.
                    _mf33_den = build_mf33_denominator(
                        endf_file,
                        _vb,
                        mt=mt_number,
                        njoy_exe=ACE_NJOY_EXE,
                        pendf_tolerance=MF33_PENDF_TOLERANCE,
                        pendf_cache_dir=MF33_PENDF_CACHE_DIR,
                        grid_ev=mf33_energy_grid_ev,
                        logger=_logger,
                    )
                    mf33_sigma_host_bin = _mf33_den.sigma_host_bin
                    # Reused by the folded-comparison diagnostic below, so it
                    # sees the same reconstructed cross section.
                    _host_e_ev, _host_xs_b = _mf33_den.e_ev, _mf33_den.xs_b
                    # The MT1 rebuild needs every partial's cross section, so it
                    # reads the PENDF again rather than just MT2's fold.
                    _mf33_pendf_path = _mf33_den.pendf_path
                    _c0_host = mf33_sigma_host_bin / _MF33_FOUR_PI

                    # Build the fine relative covariance and its own adaptive
                    # multigroup collapse in one go — the same call the offline
                    # rebuild uses, so both paths emit identical sections.
                    _mf33_products = build_mf33_matrices(
                        cov_abs_fine=_mf33_cov_abs,
                        sigma_host_bin=mf33_sigma_host_bin,
                        energy_bins=_vb,
                        grid_fine_ev=mf33_energy_grid_ev,
                        c0_dcs=mf33_c0_nom,
                        regularize_near_zero=regularize_near_zero,
                        snr_threshold=near_zero_snr_threshold,
                        n_neighbors=near_zero_n_neighbors,
                        rho_min=MF33_MULTIGROUP_RHO_MIN,
                        sigma_ratio_max=MF33_MULTIGROUP_SIGMA_RATIO_MAX,
                        diagnostics_file=(
                            output_path / "mf33_boundary_decisions.csv"
                            if SAVE_MF33_MULTIGROUP_DIAGNOSTICS_CSV else None
                        ),
                        logger=_logger,
                    )
                    mf33_rel_cov_fine = _mf33_products.rel_fine
                    _logger.info(
                        f"  [MF33] Fixed-shape c0 channel: {len(energy_indices_kw)} "
                        f"fine bins -> {_mf33_products.multigroup.cov_rel_grouped.shape[0]} "
                        f"MF33 groups (the MF34 grid is collapsed separately)"
                    )
                    _min_eig = float(np.min(np.linalg.eigvalsh(mf33_rel_cov_fine)))
                    if _min_eig < -1e-8:
                        _logger.warning(
                            f"  [MF33] Fine-grid relative covariance not PSD "
                            f"(min eig {_min_eig:.2e}) — warn only, not repaired.",
                            console=True,
                        )

                    # Sidecar outputs. mf33_absolute_covariance.npy is the
                    # primary object — the relative one is derived from it and
                    # the folded host central, and is the lossy one (rows with a
                    # non-positive central are zeroed). Together with the grid
                    # and nominal_fits.parquet these are exactly what
                    # scripts/rebuild_mf33.py needs to rebuild MF33 offline.
                    _mf33_sidecars = [
                        "mf33_relative_covariance.npy", "mf33_absolute_covariance.npy",
                        "mf33_c0_nominal.npy", "mf33_c0_host.npy",
                        "mf33_energy_grid_ev.npy", "mf33_multigroup_grid_ev.npy",
                        "mf33_multigroup_relative_covariance.npy",
                    ]
                    np.save(output_path / "mf33_relative_covariance.npy", mf33_rel_cov_fine)
                    np.save(output_path / "mf33_absolute_covariance.npy", _mf33_cov_abs)
                    np.save(output_path / "mf33_c0_nominal.npy", mf33_c0_nom)
                    np.save(output_path / "mf33_c0_host.npy", _c0_host)
                    np.save(output_path / "mf33_energy_grid_ev.npy", mf33_energy_grid_ev)
                    np.save(
                        output_path / "mf33_multigroup_grid_ev.npy",
                        _mf33_products.multigroup.group_boundaries_ev,
                    )
                    np.save(
                        output_path / "mf33_multigroup_relative_covariance.npy",
                        _mf33_products.multigroup.cov_rel_grouped,
                    )
                    if SAVE_MF33_C0_SAMPLES:
                        _df_c0.to_parquet(
                            output_path / "mf33_c0_samples.parquet",
                            engine="pyarrow", index=False,
                        )
                        _mf33_sidecars.append("mf33_c0_samples.parquet")
                    _logger.info(
                        f"  [MF33] Sidecars written: {', '.join(_mf33_sidecars)}"
                    )

                    # Folded-host comparison (warn-only, never gates): fold the
                    # host MF3 through each contributing experiment's TOF
                    # kernel at the bin energy and compare against 4*pi*c0.
                    try:
                        _cmp_sub, _cmp_e, _cmp_dcs, _cmp_rel, _cmp_camp = [], [], [], [], []
                        _rel_dcs_diag = np.sqrt(
                            np.clip(np.diag(_mf33_rel_dcs), 0.0, None)
                        )
                        for _ib, _eidx in enumerate(energy_indices_kw):
                            _nr_b = nr_by_idx_kw[_eidx]
                            for _exp in (_nr_b.experiments_info or []):
                                _cmp_sub.append(
                                    f"{_exp.get('entry', '')}{_exp.get('subentry', '')}"
                                )
                                _cmp_e.append(float(_nr_b.energy_mev))
                                _cmp_dcs.append(_MF33_FOUR_PI * mf33_c0_nom[_ib])
                                _cmp_rel.append(float(_rel_dcs_diag[_ib]))
                                _cmp_camp.append(str(_exp.get('entry', '')))
                        if _cmp_sub:
                            _folded_b = fold_host_mf3_at_points(
                                _host_e_ev, _host_xs_b, _cmp_sub, _cmp_e,
                                tof_params_cache,
                                default_flight_path_m=FLIGHT_PATH_M,
                                default_time_resolution_ns=DELTA_T_NS,
                                default_delta_t_is_fwhm=DELTA_T_IS_FWHM,
                            )
                            _cmp_df = pd.DataFrame({
                                "campaign": _cmp_camp,
                                "energy_mev": _cmp_e,
                                "sigma_folded_b": _folded_b,
                                "sigma_el_dcs_b": _cmp_dcs,
                                "rel_sigma_dcs": _cmp_rel,
                            })
                            _folded_stats = folded_c0_comparison_stats(_cmp_df)
                            log_folded_comparison(
                                _folded_stats, _logger,
                                verbose=bool(VERBOSE_DIAGNOSTICS),
                            )
                    except Exception as _e_fold:
                        _logger.warning(
                            f"  [MF33] Folded-host comparison failed "
                            f"(diagnostic only): {_e_fold}",
                            console=True,
                        )
                except Exception as _e_mf33:
                    _logger.error(
                        f"[ERROR] [MF33] Fixed-shape c0 channel failed: {_e_mf33}",
                        console=True,
                    )
                    mf33_rel_cov_fine = None
                    mf33_sigma_host_bin = None

            _valid_mask_kw = np.zeros(len(energy_indices_kw) * max_degree, dtype=bool)
            for ie, e_idx in enumerate(energy_indices_kw):
                nr = nr_by_idx_kw[e_idx]
                n_valid = bin_valid_orders(nr, max_degree)
                if MAX_SAMPLE_ORDER is not None:
                    n_valid = min(n_valid, MAX_SAMPLE_ORDER)
                for l in range(n_valid):
                    _valid_mask_kw[ie * max_degree + l] = True
            _n_legacy_kw = sum(
                min(nr_by_idx_kw[e].frozen_degree, max_degree)
                if MAX_SAMPLE_ORDER is None
                else min(nr_by_idx_kw[e].frozen_degree, max_degree, MAX_SAMPLE_ORDER)
                for e in energy_indices_kw
            )
            _n_valid_kw = int(np.sum(_valid_mask_kw))
            _logger.info(
                f"  Valid-parameter mask: {_n_valid_kw}/{len(_valid_mask_kw)} valid "
                f"({_n_valid_kw / max(len(_valid_mask_kw), 1) * 100:.1f}%)"
                + (f"  [MIX] q_l > {MIXTURE_Q_MASK_THRESHOLD} vs {_n_legacy_kw} "
                   f"under the frozen_degree rule "
                   f"(+{_n_valid_kw - _n_legacy_kw} slots, "
                   f"MF34 grows ~{_n_valid_kw / max(_n_legacy_kw, 1):.2f}x)"
                   if USE_MIXTURE_COVARIANCE else "")
            )

            _snr_thr = near_zero_snr_threshold if regularize_near_zero else 0.0
            cov_kw, corr_kw, _, mc_mean_kw, _ = compute_covariance_from_samples(
                all_samples=kw_samples, energy_indices=energy_indices_kw,
                max_order=max_degree, valid_mask=_valid_mask_kw,
                snr_threshold=_snr_thr, n_neighbors=near_zero_n_neighbors,
                logger=_logger,
            )
            log_psd_diagnostics(corr_kw, "corr_kw (Pass 1)", _logger)
            # Phase 3: the per-bin pass is where the mixture belongs — it is the
            # pass that supplies the variances (std_perbin) and the within-bin
            # correlations. Pass 1 fits at the frozen winner degree only
            # (run_mc_with_kernel_weights receives no degree weights), so it
            # cannot supply high-order structure and is left alone.
            _mix_blocks, _mix_diag = ({}, {})
            if USE_MIXTURE_COVARIANCE and mixture_by_bin:
                _mix_blocks, _mix_diag = build_mixture_blocks(
                    mixture_by_bin, nominal_results, max_degree, logger=_logger)
            cov_perbin, corr_perbin, _, mc_mean_perbin, _ = compute_covariance_from_samples(
                all_samples=all_samples_perbin, energy_indices=energy_indices_kw,
                max_order=max_degree, valid_mask=_valid_mask_kw,
                snr_threshold=_snr_thr, n_neighbors=near_zero_n_neighbors,
                logger=_logger,
                mixture_blocks=_mix_blocks or None,
            )
            log_psd_diagnostics(corr_perbin, "corr_perbin (Pass 2)", _logger)

            # Use correlation from absolute covariance (returned by
            # compute_covariance_from_samples), NOT re-extracted from relative
            # covariance.  Extracting from relative cov flips the sign of
            # off-diagonal entries when mean_i and mean_j have opposite signs
            # (e.g. a_1 > 0 and a_2 < 0), corrupting cross-order correlations.
            std_perbin = np.sqrt(np.maximum(np.diag(cov_perbin), 0.0))

            # Sign-aware outer product of mean signs.  cov_combined will be
            # multiplied by outer(mean, mean) inside generate_cholesky_samples
            # to produce absolute covariance.  Without this factor the round
            # trip would flip the sign of off-diagonal entries for parameter
            # pairs whose means have opposite signs (corr_kw is the sign-
            # correct *absolute* correlation; std_perbin is *relative* std,
            # always positive).  Treat zero means as +1 to preserve diagonals;
            # those entries are zeroed downstream by the outer(mean, mean) step.
            _mean_signs = np.sign(mc_mean_perbin)
            _mean_signs[_mean_signs == 0] = 1.0
            _sign_outer = np.outer(_mean_signs, _mean_signs)

            # TMC parquet — Pass-1 correlation structure (raw, non-Gaussian)
            # with Pass-2 calibrated marginals.  Per-parameter affine rescaling
            # preserves Pearson and rank correlations exactly.  Recommended
            # TMC input for downstream consumers.
            if save_tmc_parquet:
                try:
                    n_params = len(energy_indices_kw) * max_degree
                    mat_kw_pass1 = stack_samples_to_matrix(
                        kw_samples, energy_indices_kw, n_samples, max_degree,
                    )
                    mat_perbin_pass2 = stack_samples_to_matrix(
                        all_samples_perbin, energy_indices_kw, n_samples, max_degree,
                    )
                    mean_kw_pass1 = mat_kw_pass1.mean(axis=0)
                    std_kw_pass1 = mat_kw_pass1.std(axis=0, ddof=0)
                    mean_perbin_pass2 = mat_perbin_pass2.mean(axis=0)
                    std_perbin_pass2 = mat_perbin_pass2.std(axis=0, ddof=0)
                    eps = 1e-30
                    n_zero_std = int(np.sum(std_kw_pass1 < eps))
                    if n_zero_std > 0:
                        _logger.info(
                            f"  [TMC] {n_zero_std}/{n_params} parameters had std_kw < {eps:.0e}; "
                            f"TMC values set deterministically to Pass-2 mean"
                        )
                    scale_factors = np.where(
                        std_kw_pass1 >= eps,
                        std_perbin_pass2 / np.maximum(std_kw_pass1, eps),
                        0.0,
                    )
                    mat_tmc = mean_perbin_pass2[None, :] + (mat_kw_pass1 - mean_kw_pass1[None, :]) * scale_factors[None, :]
                    tmc_samples_dict = {s: {} for s in range(n_samples)}
                    for s in range(n_samples):
                        for k, e_idx in enumerate(energy_indices_kw):
                            start = k * max_degree
                            tmc_samples_dict[s][e_idx] = mat_tmc[s, start:start + max_degree].copy()
                    tmc_path = save_all_legendre_coefficients(
                        nominal_results=nominal_results,
                        all_samples=tmc_samples_dict,
                        output_dir=str(output_path),
                        max_degree=max_degree,
                        filename='legendre_samples_tmc.parquet',
                    )
                    _logger.info(f"  [INFO] [TMC] TMC parquet: {tmc_path}")
                    _logger.info(
                        "  [INFO] [TMC] Pearson correlations preserved exactly "
                        "across MC rows (filter df[~df.is_nominal] before computing "
                        "sample statistics — the nominal row is the deterministic "
                        "fit, not an affine-mapped sample); marginals match Pass-2 "
                        "mean/std (recommended TMC input)"
                    )
                except Exception as exc:
                    _logger.error(
                        f"[ERROR] [TMC] Failed to save TMC parquet: {exc}",
                        console=True,
                    )

            # Phase 4: Loss-of-correlation diagnostic — flag parameter pairs
            # with significant non-Gaussian dependence that MF34 (Pearson
            # covariance only) cannot preserve. Uses the same Pass-1 KW samples.
            try:
                from scripts.correlation_loss_diagnostic import (
                    compute_correlation_loss_diagnostic,
                )
                _n_p4 = len(energy_indices_kw) * max_degree
                _sm_kw_p4 = np.zeros((n_samples, _n_p4))
                for _s in range(n_samples):
                    for _k, _e in enumerate(energy_indices_kw):
                        _kw_c = kw_samples[_s].get(_e, np.zeros(max_degree))
                        _st = _k * max_degree
                        _sm_kw_p4[_s, _st:_st + min(len(_kw_c), max_degree)] = _kw_c[:max_degree]
                _param_labels_p4 = [
                    (e_idx, l + 1)
                    for e_idx in energy_indices_kw
                    for l in range(max_degree)
                ]
                _e_mev_lookup = {
                    nr.energy_index: nr.energy_mev
                    for nr in nominal_results
                }
                # Active mask: valid + non-zero variance + above-SNR
                _abs_std_p4 = std_perbin * np.abs(mc_mean_perbin)
                _active_p4 = (
                    _valid_mask_kw
                    & (_abs_std_p4 > 0)
                    & (np.abs(mc_mean_perbin) >= near_zero_snr_threshold * _abs_std_p4)
                )
                compute_correlation_loss_diagnostic(
                    sample_matrix=_sm_kw_p4,
                    param_labels=_param_labels_p4,
                    energy_mev_lookup=_e_mev_lookup,
                    active_mask=_active_p4,
                    logger=_logger,
                )
            except Exception as _exc:
                _logger.error(
                    f"[ERROR] [Phase 4] Loss-of-correlation diagnostic failed: {_exc}",
                    console=True,
                )

            if _is_hybrid:
                # Build Gaussian parametric correlation as fallback
                cov_gauss = build_gaussian_correlation_covariance(
                    cov_stochastic=cov_perbin,
                    energy_bins=energy_bins,
                    energy_indices=energy_indices_kw,
                    max_order=max_degree,
                    logger=None,
                    valid_mask=_valid_mask_kw,
                    corr_stochastic_in=corr_perbin,
                )
                corr_gauss = _extract_correlation_matrix(cov_gauss)

                # Compute per-energy alpha from KW overlap diagnostics
                alpha_per_energy = []
                for e_idx in energy_indices_kw:
                    nr = nr_by_idx_kw[e_idx]
                    if nr.interpolated:
                        alpha_per_energy.append(0.0)
                        continue

                    bin_overlap = overlap_weights.get(e_idx, [])
                    kw_diag = compute_kw_diagnostics(bin_overlap)
                    a = compute_kw_reliability_alpha(
                        kw_diag=kw_diag,
                        interpolated=nr.interpolated,
                        min_points_ref=max_degree + 1,
                    )
                    alpha_per_energy.append(a)

                    if kw_diag is not None:
                        _logger.debug(
                            f"    Bin {e_idx}: n_eff_kw={kw_diag.n_eff_kw:.2f}, "
                            f"n_exp_kw={kw_diag.n_experiments_kw}, "
                            f"max_frac_kw={kw_diag.max_experiment_weight_frac_kw:.3f}, "
                            f"med_pts={kw_diag.weighted_median_n_points:.1f}, "
                            f"alpha={a:.3f}"
                        )
                alpha_per_energy = np.array(alpha_per_energy)

                nan_mask = ~np.isfinite(alpha_per_energy)
                if np.any(nan_mask):
                    n_nan = int(np.sum(nan_mask))
                    _logger.warning(
                        f"  Hybrid blend: {n_nan} bins had non-finite alpha — "
                        f"using 0.0 (pure Gaussian fallback)"
                    )
                    alpha_per_energy[nan_mask] = 0.0

                # Expand to parameter space (energy x order): same alpha for all orders
                n_params = len(energy_indices_kw) * max_degree
                alpha_param = np.zeros(n_params)
                for ie, a in enumerate(alpha_per_energy):
                    alpha_param[ie * max_degree:(ie + 1) * max_degree] = a

                # Pairwise alpha: min(alpha_i, alpha_j) — conservative
                alpha_ij = np.minimum(alpha_param[:, None], alpha_param[None, :])

                # Range-aware blend: only apply Gaussian where it has an
                # opinion (short-range, within a few sigma_E).  At long range
                # the Gaussian decay → 0, so blending would just dilute the
                # KW correlations for no reason.
                #
                # g_ij = Gaussian relevance ∈ [0, 1]:
                #   g ≈ 1 for |dE| << sigma_E  (Gaussian model informative)
                #   g → 0 for |dE| >> sigma_E  (Gaussian model contributes nothing)
                #
                # Effective blend weight for Gaussian:
                #   w_gauss_ij = (1 - alpha_ij) * g_ij
                #
                # So: corr_hyb = (1 - w_gauss) * corr_kw + w_gauss * corr_gauss
                #   Short-range: g≈1 → standard alpha blend
                #   Long-range:  g≈0 → corr_hyb ≈ corr_kw (KW passes through)
                g_ij = build_gaussian_relevance_matrix(
                    energy_bins=energy_bins,
                    energy_indices=energy_indices_kw,
                    max_order=max_degree,
                )
                w_gauss_ij = (1.0 - alpha_ij) * g_ij

                # Hybrid blend (cross-bin correlations)
                corr_hyb = (1.0 - w_gauss_ij) * corr_kw + w_gauss_ij * corr_gauss
                corr_hyb = (corr_hyb + corr_hyb.T) / 2.0
                np.fill_diagonal(corr_hyb, 1.0)
                corr_hyb = np.clip(corr_hyb, -1.0, 1.0)

                log_psd_diagnostics(corr_hyb, "corr_hyb (post-blend)", _logger)

                if KW_MC_INJECT:
                    # Legacy path: splice Pass-2 within-bin blocks into Pass-1
                    # cross-bin scaffold, then snap to nearest PSD via Higham.
                    # Not PSD-preserving by construction.
                    corr_hyb = inject_within_bin_correlations(
                        corr_hyb, corr_perbin, len(energy_indices_kw), max_degree,
                    )
                    log_psd_diagnostics(corr_hyb, "corr_hyb (post-inject)", _logger)

                    _abs_std_perbin = std_perbin * np.abs(mc_mean_perbin)
                    _dead_mask = (
                        (~_valid_mask_kw)
                        | (_abs_std_perbin <= 0)
                        | (np.abs(mc_mean_perbin) < near_zero_snr_threshold * _abs_std_perbin)
                    )
                    corr_hyb = psd_repair_correlation_active(
                        corr_hyb, _dead_mask,
                        label="corr_hyb (hybrid)", logger=_logger,
                    )
                    log_psd_diagnostics(corr_hyb, "corr_hyb (post-PSD-repair)", _logger)

                    cov_combined = corr_hyb * np.outer(std_perbin, std_perbin) * _sign_outer
                    log_psd_diagnostics(cov_combined, "cov_combined (post-rescale, hybrid)", _logger)
                else:
                    # Default: congruence transform Cov = D * Corr * D with
                    # D = diag(std_perbin). Corr is the convex blend of two PSD
                    # matrices with unit diagonal, so the result is PSD.
                    cov_combined = corr_hyb * np.outer(std_perbin, std_perbin) * _sign_outer
                    log_psd_diagnostics(cov_combined, "cov_combined (congruence, hybrid)", _logger)
                    _diag_diff = float(np.max(np.abs(np.diag(cov_combined) - std_perbin**2)))
                    _logger.info(f"  [Congruence check, hybrid] max |diag(cov) - std_perbin^2| = {_diag_diff:.3e}")

                # Log alpha and range-aware blend statistics
                n_interp = int(np.sum(alpha_per_energy == 0.0))
                n_data = len(alpha_per_energy) - n_interp
                if n_data > 0:
                    data_alphas = alpha_per_energy[alpha_per_energy > 0.0]
                    # Effective KW weight: fraction of matrix where KW dominates
                    off_diag = ~np.eye(n_params, dtype=bool)
                    mean_w_gauss = float(np.mean(w_gauss_ij[off_diag]))
                    mean_w_kw = 1.0 - mean_w_gauss
                    _logger.info(f"  Hybrid blend (range-aware): {n_data} data bins "
                                 f"(alpha mean={np.mean(data_alphas):.3f}, "
                                 f"median={np.median(data_alphas):.3f}, "
                                 f"min={np.min(data_alphas):.3f}, max={np.max(data_alphas):.3f}), "
                                 f"{n_interp} interpolated bins (alpha=0)")
                    _logger.info(f"  Range-aware weights: mean KW weight={mean_w_kw:.3f}, "
                                 f"mean Gaussian weight={mean_w_gauss:.3f} "
                                 f"(long-range KW correlations preserved)")
                else:
                    _logger.info(f"  Hybrid blend: all {n_interp} bins interpolated (pure Gaussian)")
            else:  # pure kernel_weight_mc
                if KW_MC_INJECT:
                    # Legacy path: splice Pass-2 within-bin blocks into Pass-1
                    # cross-bin scaffold, then snap to nearest PSD via Higham.
                    # Not PSD-preserving by construction.
                    corr_kw = inject_within_bin_correlations(
                        corr_kw, corr_perbin, len(energy_indices_kw), max_degree,
                    )
                    log_psd_diagnostics(corr_kw, "corr_kw (post-inject, pure-KW)", _logger)

                    _abs_std_perbin_pk = std_perbin * np.abs(mc_mean_perbin)
                    _dead_mask_pk = (
                        (~_valid_mask_kw)
                        | (_abs_std_perbin_pk <= 0)
                        | (np.abs(mc_mean_perbin) < near_zero_snr_threshold * _abs_std_perbin_pk)
                    )
                    corr_kw = psd_repair_correlation_active(
                        corr_kw, _dead_mask_pk,
                        label="corr_kw (pure-KW)", logger=_logger,
                    )
                    log_psd_diagnostics(corr_kw, "corr_kw (post-PSD-repair, pure-KW)", _logger)

                    cov_combined = corr_kw * np.outer(std_perbin, std_perbin) * _sign_outer
                    log_psd_diagnostics(cov_combined, "cov_combined (post-rescale, pure-KW)", _logger)
                else:
                    # Default: congruence transform Cov = D * Corr_pass1 * D with
                    # D = diag(std_perbin). compute_covariance_from_samples sets
                    # corr diagonals to 0 for zero-variance slots (exfor_utils.py
                    # ~line 2350); restore unit diagonal so Pass-2 variances are
                    # not silently zeroed for those slots.
                    np.fill_diagonal(corr_kw, 1.0)
                    corr_kw = np.clip(corr_kw, -1.0, 1.0)
                    if (USE_MIXTURE_COVARIANCE and _mix_blocks
                            and MIXTURE_SPLICE_WITHIN_BIN_CORR):
                        # OFF BY DEFAULT — see MIXTURE_SPLICE_WITHIN_BIN_CORR.
                        # The argument below is why it is tempting; the measured
                        # PSD cost is why it is not the default.
                        # run_mc_with_kernel_weights fits at the frozen winner
                        # degree and is given no degree weights, so Pass-1 has
                        # no information at all about orders above the winner:
                        # those slots are constant across its samples, their
                        # variance is 0 and their cross-order correlations come
                        # back as 0. Taking std from the mixture while leaving
                        # corr at Pass-1 would ship the new a_5/a_6 variance as
                        # UNCORRELATED with a_1..a_4, which is wrong — within a
                        # bin those orders covary strongly through the shared
                        # model choice. corr_perbin now carries the mixture, so
                        # splice its within-bin blocks in and repair.
                        corr_kw = inject_within_bin_correlations(
                            corr_kw, corr_perbin, len(energy_indices_kw), max_degree,
                        )
                        log_psd_diagnostics(corr_kw, "corr_kw (post-mixture-inject)", _logger)
                        _abs_std_mix = std_perbin * np.abs(mc_mean_perbin)
                        _dead_mask_mix = (
                            (~_valid_mask_kw)
                            | (_abs_std_mix <= 0)
                            | (np.abs(mc_mean_perbin) < near_zero_snr_threshold * _abs_std_mix)
                        )
                        corr_kw = psd_repair_correlation_active(
                            corr_kw, _dead_mask_mix,
                            label="corr_kw (mixture)", logger=_logger,
                        )
                        log_psd_diagnostics(corr_kw, "corr_kw (post-PSD-repair, mixture)", _logger)
                    cov_combined = corr_kw * np.outer(std_perbin, std_perbin) * _sign_outer
                    log_psd_diagnostics(cov_combined, "cov_combined (congruence, pure-KW)", _logger)
                    _diag_diff = float(np.max(np.abs(np.diag(cov_combined) - std_perbin**2)))
                    _logger.info(f"  [Congruence check, pure-KW] max |diag(cov) - std_perbin^2| = {_diag_diff:.3e}")

            # Generate Cholesky samples from combined covariance
            _logger.info(f"  Generating {n_samples} Cholesky samples from combined covariance")
            all_samples = generate_cholesky_samples(
                cov_full=cov_combined,
                mean_params=mc_mean_perbin,
                energy_indices=energy_indices_kw,
                max_order=max_degree,
                n_samples=n_samples,
                seed=base_seed,
                logger=_logger,
            )
        else:
            # Single pass: kernel-weight MC → full covariance directly
            all_samples = kw_samples
            _logger.info("  Single-pass mode: using KW MC samples directly")

        n_sampled = sum(1 for nr in nominal_results if nr.has_data and not nr.interpolated)
        n_interpolated_used = sum(1 for nr in nominal_results if nr.has_data and nr.interpolated)
        _logger.info(f"  [INFO] [MC] Complete: {n_sampled} bins with data, {n_interpolated_used} interpolated")

    else:
        raise ValueError(f"Unknown CORRELATION_METHOD: {CORRELATION_METHOD!r}. Use 'gaussian', 'kernel_weight_mc', or 'hybrid'.")

    _logger.info(f">> samples_generated = {n_samples}")
    _logger.info(f"#-- END STEP 5 (elapsed: {time.time() - t_step:.2f}s) -------------------------------------")

    # Step 6: Save coefficients
    # Step 7: Covariance
    cov_matrix = None
    energy_indices = [nr.energy_index for nr in nominal_results if nr.has_data]

    # Build valid-parameter mask for Step 7 covariance paths
    #
    # `_mg_valid_orders` is the SAME rule as a callable, so the multigroup
    # collapse (Step 7b) and the merge path cannot drift from this mask. They
    # used to: both rebuilt `min(frozen_degree, max_order)` internally, which is
    # what kept Phase 3 out of the shipped multigroup file (roadmap §8.4).
    def _mg_valid_orders(nr, mo):
        n = bin_valid_orders(nr, mo)
        if MAX_SAMPLE_ORDER is not None:
            n = min(n, MAX_SAMPLE_ORDER)
        return min(n, mo)

    nr_by_idx_s7 = {nr.energy_index: nr for nr in nominal_results}
    valid_mask_s7 = np.zeros(len(energy_indices) * max_degree, dtype=bool)
    for ie, e_idx in enumerate(energy_indices):
        nr = nr_by_idx_s7[e_idx]
        for l in range(_mg_valid_orders(nr, max_degree)):
            valid_mask_s7[ie * max_degree + l] = True

    if True:  # Covariance always computed (needed for MF34)
        t_step = time.time()
        _logger.info("")
        _logger.info("#-- STEP 7: Compute covariance matrix --------------------------------------")

        if _prebuilt_gaussian_cov is not None:
            # Gaussian correlation mode: use pre-built covariance from Step 5
            _logger.info("  Using pre-built Gaussian correlation covariance")
            cov_matrix = _prebuilt_gaussian_cov
            mc_mean_params = _prebuilt_mc_mean
            param_labels = [(e_idx, l + 1) for e_idx in energy_indices for l in range(max_degree)]

            # Compute correlation from covariance
            std_combined = np.sqrt(np.maximum(np.diag(cov_matrix), 0.0))
            std_combined[std_combined == 0] = 1.0
            corr_matrix = cov_matrix / np.outer(std_combined, std_combined)

            # Recover absolute covariance from MC-relative
            cov_abs = cov_matrix * np.outer(mc_mean_params, mc_mean_params)

            var_total = np.diag(cov_matrix)
            mask = var_total > 0
            if np.any(mask):
                _logger.info(f"  Gaussian cov variance: mean={np.mean(var_total[mask]):.2e}")
        else:
            # Standard covariance from MC samples
            _snr_thr = near_zero_snr_threshold if regularize_near_zero else 0.0

            _use_raw_mc_corr = (
                MULTIGROUP_USE_RAW_MC_CORR
                and KW_MC_TWO_PASS
                and CORRELATION_METHOD in ("kernel_weight_mc", "hybrid")
            )

            if _use_raw_mc_corr:
                # Fix 5: build covariance directly from raw KW correlations
                # (PSD by construction, no inject step) combined with Pass-2
                # calibrated per-bin std. This bypasses the inject + Higham
                # smear path that produces spurious deep-negative correlations
                # at remote off-diagonals. Trade-off: within-bin order
                # correlations come from KW samples, not from per-bin
                # stochastic samples.
                _logger.info(
                    "  [Fix 5] Multigroup input: raw KW correlations + "
                    "Pass-2 std (bypassing inject + Higham smear)"
                )

                cov_matrix = corr_kw * np.outer(std_perbin, std_perbin) * _sign_outer
                corr_matrix = corr_kw.copy()
                mc_mean_params = mc_mean_perbin.copy()
                param_labels = [
                    (e_idx, l + 1)
                    for e_idx in energy_indices
                    for l in range(max_degree)
                ]
                cov_abs = cov_matrix * np.outer(mc_mean_params, mc_mean_params)

                # Log within-bin (corr_kw - corr_perbin) discrepancy: this is
                # the cost of skipping the inject step. Frobenius norm per bin.
                _n_bins = len(energy_indices)
                _within_diffs = np.zeros(_n_bins)
                for _ie in range(_n_bins):
                    _s = _ie * max_degree
                    _e = _s + max_degree
                    _within_diffs[_ie] = np.linalg.norm(
                        corr_kw[_s:_e, _s:_e] - corr_perbin[_s:_e, _s:_e],
                        'fro',
                    )
                _logger.info(
                    f"  [Fix 5] Within-bin (corr_kw - corr_perbin) Frobenius: "
                    f"mean={np.mean(_within_diffs):.3e}, "
                    f"median={np.median(_within_diffs):.3e}, "
                    f"max={np.max(_within_diffs):.3e}"
                )
                log_psd_diagnostics(corr_matrix, "corr_for_multigroup (Fix 5)", _logger)
            else:
                cov_matrix, corr_matrix, param_labels, mc_mean_params, cov_abs = compute_covariance_from_samples(
                    all_samples=all_samples,
                    energy_indices=energy_indices,
                    max_order=max_degree,
                    valid_mask=valid_mask_s7,
                    snr_threshold=_snr_thr,
                    n_neighbors=near_zero_n_neighbors,
                    logger=_logger,
                )

        # Safety net: sanitize non-finite entries before multigroup collapse
        n_nonfinite = int(np.sum(~np.isfinite(cov_matrix)))
        if n_nonfinite > 0:
            _logger.warning(f"[WARN] [COV] Covariance matrix has {n_nonfinite} non-finite entries -- replacing with 0")
            cov_matrix = np.where(np.isfinite(cov_matrix), cov_matrix, 0.0)
            cov_matrix = (cov_matrix + cov_matrix.T) / 2.0  # re-symmetrize
            # Also fix cov_abs and corr_matrix
            cov_abs = np.where(np.isfinite(cov_abs), cov_abs, 0.0)
            cov_abs = (cov_abs + cov_abs.T) / 2.0
            std_fix = np.sqrt(np.maximum(np.diag(cov_abs), 0.0))
            std_fix[std_fix == 0] = 1.0
            corr_matrix = cov_abs / np.outer(std_fix, std_fix)

        # Validate covariance: check that diagonal values are non-trivial
        diag = np.diag(cov_matrix)
        diag_nonzero = diag[diag > 0]
        if len(diag_nonzero) > 0:
            min_diag = np.min(diag_nonzero)
            max_diag = np.max(diag_nonzero)
            mean_diag = np.mean(diag_nonzero)
            # Covariance is now relative (fractional): diag = Var(a)/mean(a)^2
            mean_rel_std = np.sqrt(mean_diag)
            _logger.info(f"  Relative variance stats: min={min_diag:.2e}, max={max_diag:.2e}, mean={mean_diag:.2e}")
            _logger.info(f"  Mean relative std: {mean_rel_std:.4f} ({mean_rel_std*100:.2f}%)")

            if np.all(diag < 1e-20):
                _logger.error(
                    "  WARNING: All diagonal covariance values are essentially zero!",
                    console=True
                )
                _logger.error(
                    "  This indicates MC sampling failed - all samples may be identical.",
                    console=True
                )
        else:
            _logger.warning("[WARN] [COV] No positive diagonal elements in covariance matrix!")

        _logger.info(f"  Covariance matrix shape: {cov_matrix.shape}")

        # Detailed parameter diagnostics
        n_total = len(valid_mask_s7)
        n_valid = int(np.sum(valid_mask_s7))
        n_unfitted = n_total - n_valid
        near_zero_thresh = 1e-6
        near_zero_mask = valid_mask_s7 & (np.abs(mc_mean_params) < near_zero_thresh)
        well_constrained_mask = valid_mask_s7 & (np.abs(mc_mean_params) >= near_zero_thresh)
        n_near_zero = int(np.sum(near_zero_mask))
        n_well = int(np.sum(well_constrained_mask))
        _logger.info(f"  Parameter breakdown: {n_valid} valid ({n_well} well-constrained, "
                     f"{n_near_zero} near-zero-mean floored), {n_unfitted} unfitted")

        diag_wc = diag[well_constrained_mask]
        diag_wc_pos = diag_wc[diag_wc > 0]
        if len(diag_wc_pos) > 0:
            mean_rel_std_wc = np.sqrt(np.mean(diag_wc_pos))
            max_rel_std_wc = np.sqrt(np.max(diag_wc_pos))
            _logger.info(f"  Well-constrained relative std: mean={mean_rel_std_wc*100:.2f}%, "
                         f"max={max_rel_std_wc*100:.2f}%")

        if len(diag) > 0:
            idx_max = np.argmax(diag)
            label = param_labels[idx_max]
            mean_val = mc_mean_params[idx_max]
            _logger.info(f"  Max relative variance at E_idx={label[0]}, L={label[1]}: "
                         f"var_rel={diag[idx_max]:.2e}, mean={mean_val:.2e}")

        # l-l' correlation diagnostics
        if max_degree > 1:
            extract_ll_prime_correlations(
                cov_matrix=cov_matrix,
                energy_indices=energy_indices,
                max_order=max_degree,
                logger=_logger,
                valid_mask=valid_mask_s7,
            )

        _logger.info(f">> covariance_shape = {cov_matrix.shape}")
        _logger.info(f"#-- END STEP 7 (elapsed: {time.time() - t_step:.2f}s) -------------------------------------")

        # Step 7a: Apply covariance cap (Layer 1) if enabled
        if apply_covariance_cap:
            t_step = time.time()
            _logger.info("")
            _logger.info("#-- STEP 7a: Covariance diagonal cap (Layer 1) ----------------------------")

            # Build energy MeV lookup for logging
            energy_mev_lookup = {
                nr.energy_index: nr.energy_mev
                for nr in nominal_results if nr.has_data
            }

            cov_matrix, cap_diagnostics = cap_covariance_relative_uncertainty(
                cov_matrix=cov_matrix,
                max_relative_std=max_relative_std_cap,
                param_labels=param_labels,
                energy_mev_lookup=energy_mev_lookup,
                logger=_logger,
            )

            n_capped = cap_diagnostics['n_capped']
            if n_capped > 0:
                _logger.warning(
                    f"[WARN] [COV] Capping applied to {n_capped} entries. "
                    f"Set APPLY_COVARIANCE_CAP=False for uncapped covariance.",
                    console=True,
                )
                _warning_counts['covariance_capped'] = n_capped
            else:
                _logger.info("  [INFO] [COV] No entries exceeded the cap -- no capping applied.")

            # Keep cov_abs consistent with capped MC-relative covariance
            cov_abs = cov_matrix * np.outer(mc_mean_params, mc_mean_params)

            # Recompute correlation from capped covariance
            std_capped = np.sqrt(np.maximum(np.diag(cov_matrix), 0.0))
            std_capped[std_capped == 0] = 1.0
            corr_matrix = cov_matrix / np.outer(std_capped, std_capped)

        # Save final covariance (capped if capping was applied, raw otherwise)
        if save_covariance_files:
            np.save(output_path / "legendre_covariance.npy", cov_matrix)
            if save_correlation_matrices:
                np.save(output_path / "legendre_correlation.npy", corr_matrix)

    # Trim covariance to effective order when MAX_SAMPLE_ORDER restricts it
    if MAX_SAMPLE_ORDER is not None and MAX_SAMPLE_ORDER < max_degree and cov_matrix is not None:
        _eff = MAX_SAMPLE_ORDER
        n_bins_cov = len(energy_indices)
        keep = []
        for ib in range(n_bins_cov):
            for l in range(_eff):
                keep.append(ib * max_degree + l)
        keep = np.array(keep)
        cov_matrix = cov_matrix[np.ix_(keep, keep)]
        corr_matrix = corr_matrix[np.ix_(keep, keep)]
        mc_mean_params = mc_mean_params[keep]
        cov_abs = cov_abs[np.ix_(keep, keep)]
        param_labels = [param_labels[i] for i in keep]
        valid_mask_s7 = valid_mask_s7[keep]
        _logger.info(f"  Trimmed covariance from {max_degree} to {_eff} orders/bin "
                     f"(MAX_SAMPLE_ORDER={MAX_SAMPLE_ORDER}): {cov_matrix.shape[0]} params")
        max_degree = _eff

    # Step 7b: Multigroup covariance (optional)
    multigroup_result = None
    multigroup_failure_reason = None
    if generate_multigroup_covariance:
        if cov_matrix is None:
            multigroup_failure_reason = "covariance matrix is None (computation may have failed)"
        else:
            t_step = time.time()
            _logger.info("")
            _logger.info("#-- STEP 7b: Adaptive multigroup covariance --------------------------------")
            _logger.info(f"  [INFO] [MG] Using l=1 correlation for grouping (same grid for all orders)")

            # Extract forced MF34 grid if requested
            forced_grid = None
            if use_original_mf34_grid:
                _logger.info(f"  Extracting original MF34 energy grid for forced grouping")
                forced_grid = _extract_mf34_energy_grid(
                    endf_file=endf_file,
                    mf34_source_file=mf34_source_file,
                    mt_number=mt_number,
                    logger=_logger,
                )
                if forced_grid is None:
                    _logger.warning("[WARN] [MG] MF34 grid extraction failed -- falling back to adaptive grouping")

            try:
                multigroup_result = perform_adaptive_multigroup_collapse(
                    cov_matrix=cov_matrix,
                    corr_matrix=corr_matrix,
                    nominal_results=nominal_results,
                    energy_bins=energy_bins,
                    max_order=max_degree,
                    rho_min=multigroup_rho_min,
                    sigma_ratio_max=multigroup_sigma_ratio_max,
                    variance_percentile_min=multigroup_variance_pct_min,
                    variance_percentile_max=multigroup_variance_pct_max,
                    variance_ratio_ref=multigroup_variance_ratio_ref,
                    logger=_logger,
                    apply_covariance_cap=apply_covariance_cap,
                    max_relative_std_cap=max_relative_std_cap,
                    forced_group_boundaries_mev=forced_grid,
                    diagnostics_file=(output_path / "multigroup_boundary_decisions.csv"
                                      if save_multigroup_diagnostics_csv else None),
                    # PHASE 3 FIX (2026-08-03). Without this the collapse rebuilt
                    # the legacy winner-take-all mask from `frozen_degree`
                    # internally, so the mixture reached the fine MF34 and never
                    # the multigroup one that actually gets scored (roadmap §8.4:
                    # dead parameters at l=6 went 92.3% -> 1.0% fine, but
                    # 81.6% -> 81.7% multigroup). Same rule as `valid_mask_s7`
                    # above, MAX_SAMPLE_ORDER clamp included.
                    valid_orders_fn=_mg_valid_orders,
                )

                # Log and save results
                n_fine = len([nr for nr in nominal_results if not nr.interpolated and nr.has_data])
                n_groups = len(multigroup_result.groups)
                _logger.info(f"  Fine bins: {n_fine} -> Multigroups: {n_groups}")
                _logger.info(f"  Compression: {n_fine/n_groups:.1f}x")

                if save_covariance_files:
                    np.save(output_path / "legendre_covariance_multigroup.npy",
                            multigroup_result.cov_grouped)
                    if save_correlation_matrices:
                        np.save(output_path / "legendre_correlation_multigroup.npy",
                                multigroup_result.corr_grouped)
                    np.save(output_path / "multigroup_boundaries_ev.npy",
                            multigroup_result.group_boundaries_ev)
                    np.save(output_path / "multigroup_mean_coeffs.npy",
                            multigroup_result.mean_grouped)
                    _logger.info(f"  Saved multigroup .npy artifacts (cov, mean, boundaries)")
                else:
                    _logger.info(f"  SAVE_COVARIANCE_FILES=False -- skipping multigroup .npy artifacts")

            except Exception as e:
                multigroup_failure_reason = f"{str(e)}\n{traceback.format_exc()}"
                _logger.error(f"[ERROR] [MG] Failed to compute multigroup covariance: {str(e)}", console=True)
                _logger.error(f"  Traceback:\n{traceback.format_exc()}", console=False)
                multigroup_result = None
                _warning_counts['multigroup_failed'] = 1

    # Prepare samples for ENDF writing:
    # With splice mode, samples use pipeline energy_index directly (no remapping needed).
    # Without splice, remap to ENDF grid indices for in-place coefficient replacement.
    use_splice = energy_bins is not None and len(energy_bins) > 0
    splice_range = (energy_min_mev, energy_max_mev) if use_splice else None

    if use_splice:
        all_samples_endf = all_samples
        _logger.info(f"  Using splice mode: pipeline grid will replace ENDF grid in "
                      f"[{energy_min_mev:.3f}, {energy_max_mev:.3f}] MeV")
    else:
        idx_to_endf = {nr.energy_index: nr.endf_index
                       for nr in nominal_results if nr.endf_index is not None}
        if idx_to_endf:
            all_samples_endf = remap_samples_to_endf_indices(all_samples, idx_to_endf)
        else:
            all_samples_endf = all_samples

    # Step 8: Write ENDF files
    t_step = time.time()
    average_file = None
    if generate_mc_mean_endf:
        _logger.info("")
        _logger.info("#-- STEP 8: Write ENDF files -----------------------------------------------")

        try:
            average_file = write_average_endf(
                original_endf_file=endf_file,
                mt_number=mt_number,
                nominal_results=nominal_results,
                all_samples=all_samples_endf,
                output_dir=str(output_path),
                energy_bins=energy_bins if use_splice else None,
                energy_range_mev=splice_range,
            )
            _logger.info(f"  [INFO] [ENDF] Average ENDF: {average_file}")
        except Exception as e:
            _logger.error(f"[ERROR] [ENDF] Failed to write average ENDF: {str(e)}", console=True)

    nominal_file = None
    if generate_nominal_endf:
        _logger.info(f"  [INFO] [ENDF] Writing nominal ENDF file")

        try:
            nominal_file = write_nominal_endf(
                original_endf_file=endf_file,
                mt_number=mt_number,
                nominal_results=nominal_results,
                output_dir=str(output_path),
                energy_bins=energy_bins if use_splice else None,
                energy_range_mev=splice_range,
            )
            _logger.info(f"  [INFO] [ENDF] Nominal ENDF: {nominal_file}")
        except Exception as e:
            _logger.error(f"[ERROR] [ENDF] Failed to write nominal ENDF: {str(e)}", console=True)

    _logger.info(f"#-- END STEP 8 (elapsed: {time.time() - t_step:.2f}s) -------------------------------------")

    # Step 9: Write fitting sample files (Pipeline A)
    output_files = []
    if generate_fitting_samples:
        t_step = time.time()
        _logger.info("")
        _logger.info("#-- STEP 9: Write fitting ENDF samples (Pipeline A) -------------------------")

        output_files = write_endf_samples_batch(
            original_endf_file=endf_file,
            mt_number=mt_number,
            all_samples=all_samples_endf,
            output_dir=str(output_path),
            n_procs=N_PROCS,
            energy_bins=energy_bins if use_splice else None,
            energy_range_mev=splice_range,
        )
        _logger.info(f"  [INFO] [ENDF] Written {len(output_files)} sample files")
        _logger.info(f">> fitting_samples_written = {len(output_files)}")
        _logger.info(f"#-- END STEP 9 (elapsed: {time.time() - t_step:.2f}s) -------------------------------------")

    # Step 9b: ACE generation for fitting samples (Pipeline A)
    if generate_fitting_ace:
        t_step = time.time()
        _logger.info("")
        _logger.info("#-- STEP 9b: ACE generation (Pipeline A) ------------------------------------")

        # Bridge logger: let kika sampling functions log to our pipeline log
        _set_kika_logger(_logger)

        # Get ENDF file list: use output_files from Step 9 if available,
        # otherwise discover existing files on disk
        if output_files:
            endf_sample_files = output_files
            _logger.info(f"  Using {len(endf_sample_files)} ENDF files from Step 9")
        else:
            _logger.info("  Discovering existing ENDF sample files...")
            endf_sample_files = _discover_exfor_endf_samples(
                output_dir=str(output_path),
                n_samples=n_samples,
                endf_file=endf_file,
            )

        # Filter out missing files (empty strings)
        valid_files = [(f, i) for i, f in enumerate(endf_sample_files) if f]

        if not valid_files:
            _logger.warning("[WARN] [ACE] No ENDF sample files found -- skipping ACE generation", console=True)
        else:
            _logger.info(f"  Processing {len(valid_files)} ENDF samples at {len(ace_temperatures)} temperature(s)")
            _logger.info(f"  Temperatures: {ace_temperatures} K")

            # Read ZAID from original ENDF file for skip-existing checks
            ace_zaid = None
            if ace_skip_existing:
                try:
                    _endf_for_zaid = read_endf(endf_file)
                    ace_zaid = _endf_for_zaid.zaid
                    _logger.info(f"  ZAID for skip-existing check: {ace_zaid}")
                except Exception:
                    _logger.warning("[WARN] [ACE] Could not read ZAID -- skip-existing disabled", console=False)

            # Build worker args
            ace_args_list = [
                (f, idx, ace_temperatures, str(output_path), ace_njoy_exe,
                 ace_library_name, ace_njoy_version, ace_xsdir_file,
                 ace_skip_existing, ace_zaid)
                for f, idx in valid_files
            ]

            # Run NJOY processing
            t_ace_start = time.time()
            ace_results = []

            if n_procs > 1 and len(ace_args_list) > 1:
                _logger.info(f"  Running NJOY in parallel ({n_procs} processes)")
                with Pool(n_procs) as pool:
                    ace_results = pool.map(_ace_worker, ace_args_list)
            else:
                _logger.info("  Running NJOY sequentially")
                for ace_args in ace_args_list:
                    ace_results.append(_ace_worker(ace_args))

            # Summarize results
            t_ace_elapsed = time.time() - t_ace_start
            n_success = sum(1 for _, r in ace_results if r.get("success", False))
            n_failed = sum(1 for _, r in ace_results if not r.get("success", False))
            n_skipped = sum(1 for _, r in ace_results
                           if r.get("skipped") and len(r["skipped"]) == len(ace_temperatures))
            all_errors = []
            for _, r in ace_results:
                all_errors.extend(r.get("errors", []))

            _logger.info(f"  ACE generation completed in {t_ace_elapsed:.1f}s")
            _logger.info(f"  Results: {n_success} succeeded, {n_failed} failed, {n_skipped} fully skipped")

            # Per-temperature breakdown
            for temp in ace_temperatures:
                temp_processed = sum(
                    1 for _, r in ace_results
                    if temp in r.get("temperatures_processed", [])
                )
                temp_skipped = sum(
                    1 for _, r in ace_results
                    if temp in r.get("skipped", [])
                )
                _logger.info(f"    {temp} K: {temp_processed} processed, {temp_skipped} skipped")

            if all_errors:
                _logger.warning(f"[WARN] [ACE] {len(all_errors)} error(s) during ACE generation:", console=True)
                for err in all_errors[:10]:
                    _logger.warning(f"    {err}", console=False)
                if len(all_errors) > 10:
                    _logger.warning(f"    ... and {len(all_errors) - 10} more", console=False)
                _warning_counts['ace_errors'] = len(all_errors)

            _logger.info(f">> ace_success = {n_success}")
            _logger.info(f">> ace_failed = {n_failed}")
            _logger.info(f"#-- END STEP 9b (elapsed: {time.time() - t_step:.2f}s) ------------------------------------")
            _logger.info(f"  [INFO] [ACE] {n_success}/{len(valid_files)} samples processed in {t_ace_elapsed:.1f}s", console=True)

    # Step 10: MF34 (using library functions)
    mg_nom_file = None
    if cov_matrix is not None:
        t_step = time.time()
        _logger.info("")
        _logger.info("#-- STEP 10: Write MF34 covariance -----------------------------------------")
        _logger.info(f"  [INFO] [MF34] Covariance type: {mf34_covariance_type}")

        try:
            endf_orig = read_endf(endf_file)
            mf1 = endf_orig.get_file(1)
            mt451 = mf1.sections.get(451) if mf1 else None

            if mt451:
                za = mt451._za
                awr = mt451._awr
                mat = mt451._mat
            else:
                za = 26056.0
                awr = 55.845
                mat = 2631

            mf4 = endf_orig.get_file(4)
            mt_data = mf4.sections.get(mt_number)
            all_energies_ev = np.array(mt_data.legendre_energies)

            # Build nominal parameter vector
            nominal_params = np.zeros_like(mc_mean_params)
            for k, e_idx in enumerate(energy_indices):
                nr = next((r for r in nominal_results if r.energy_index == e_idx), None)
                if nr is not None and nr.has_data:
                    endf_nom = endf_normalize_legendre_coeffs(nr.nominal_coeffs, include_a0=False)
                    n = min(len(endf_nom), max_degree)
                    nominal_params[k * max_degree: k * max_degree + n] = endf_nom[:n]

            # Direct absolute-to-nominal conversion (no intermediate rescaling)
            cov_matrix_nominal = absolute_to_nominal_relative(cov_abs, nominal_params)
            _nom_rel_std = np.sqrt(np.maximum(np.diag(cov_matrix_nominal), 0.0))
            _logger.info(f"  FG abs→nominal conversion: max rel_std = {np.max(_nom_rel_std)*100:.1f}%")
            if verbose_diagnostics:
                log_rel_std_profile(cov_matrix_nominal, max_degree, "FG post-convert", _logger, verbose=True)

            # Near-zero regularization on nominal-relative covariance
            if regularize_near_zero:
                cov_matrix_nominal, _nz_nom_diag = regularize_near_zero_relative_covariance(
                    cov_rel=cov_matrix_nominal,
                    mean_params=nominal_params,
                    cov_abs=cov_abs,
                    max_order=max_degree,
                    snr_threshold=near_zero_snr_threshold,
                    n_neighbors=near_zero_n_neighbors,
                    logger=_logger,
                )
                if verbose_diagnostics:
                    log_rel_std_profile(cov_matrix_nominal, max_degree, "FG post-nz-nom", _logger, verbose=True)

            # Between-experiment scatter floor (independent of postprocessing pipeline)
            cov_matrix_nominal, _bexp_nom = apply_between_experiment_floor(
                cov_rel=cov_matrix_nominal,
                nominal_results=nominal_results,
                energy_indices=energy_indices,
                max_order=max_degree,
                logger=_logger,
                apply=apply_between_exp_floor,
            )
            # avg branch only consumed when average_file is set; the
            # diagnostic also runs on cov_matrix_nominal, so skipping here
            # only loses a duplicate log line.
            if average_file:
                cov_matrix, _bexp_avg = apply_between_experiment_floor(
                    cov_rel=cov_matrix,
                    nominal_results=nominal_results,
                    energy_indices=energy_indices,
                    max_order=max_degree,
                    logger=_logger,
                    apply=apply_between_exp_floor,
                )
            if verbose_diagnostics:
                log_rel_std_profile(cov_matrix_nominal, max_degree, "FG post-between-exp", _logger, verbose=True)

            if apply_cov_postprocessing:
                # Step 2: Dip/spike smoothing & absent-order interpolation
                cov_matrix_nominal, _smooth_nom = smooth_absent_order_uncertainties(
                    cov_rel=cov_matrix_nominal,
                    valid_mask=valid_mask_s7,
                    max_order=max_degree,
                    min_rel_std=SMOOTH_MIN_REL_STD,
                    dip_fraction=SMOOTH_DIP_FRACTION,
                    spike_factor=SMOOTH_SPIKE_FACTOR,
                    dip_n_neighbors=SMOOTH_DIP_N_NEIGHBORS,
                    median_fill_threshold=SMOOTH_MEDIAN_FILL_THRESHOLD,
                    logger=_logger,
                )
                if average_file:
                    cov_matrix, _smooth_avg = smooth_absent_order_uncertainties(
                        cov_rel=cov_matrix,
                        valid_mask=valid_mask_s7,
                        max_order=max_degree,
                        min_rel_std=SMOOTH_MIN_REL_STD,
                        dip_fraction=SMOOTH_DIP_FRACTION,
                        spike_factor=SMOOTH_SPIKE_FACTOR,
                        dip_n_neighbors=SMOOTH_DIP_N_NEIGHBORS,
                        median_fill_threshold=SMOOTH_MEDIAN_FILL_THRESHOLD,
                        logger=_logger,
                    )
                if verbose_diagnostics:
                    log_rel_std_profile(cov_matrix_nominal, max_degree, "FG post-smooth", _logger, verbose=True)

                # Step 3: Spatial Gaussian smoothing (before hard cap)
                if SMOOTH_DIAGONAL_WINDOW >= 3:
                    cov_matrix_nominal = smooth_diagonal_median(
                        cov_rel=cov_matrix_nominal,
                        max_order=max_degree,
                        window=SMOOTH_DIAGONAL_WINDOW,
                        logger=_logger,
                    )
                    if average_file:
                        cov_matrix = smooth_diagonal_median(
                            cov_rel=cov_matrix,
                            max_order=max_degree,
                            window=SMOOTH_DIAGONAL_WINDOW,
                            logger=_logger,
                        )
                    if verbose_diagnostics:
                        log_rel_std_profile(cov_matrix_nominal, max_degree, "FG post-median", _logger, verbose=True)

                # Step 4: Order-dependent hard cap — final safety net
                if ORDER_REL_STD_CAPS is not None:
                    cov_matrix_nominal, _cap_nom = cap_order_relative_uncertainty(
                        cov_rel=cov_matrix_nominal,
                        max_order=max_degree,
                        order_caps=ORDER_REL_STD_CAPS,
                        logger=_logger,
                    )
                    if average_file:
                        cov_matrix, _cap_avg = cap_order_relative_uncertainty(
                            cov_rel=cov_matrix,
                            max_order=max_degree,
                            order_caps=ORDER_REL_STD_CAPS,
                            logger=_logger,
                        )
                if verbose_diagnostics:
                    log_rel_std_profile(cov_matrix_nominal, max_degree, "FG post-cap", _logger, verbose=True)

                # Step 5: Forward-fill absent-order rel_std
                if FORWARD_FILL_REL_STD_ENABLED:
                    _logger.info("  Applying forward-fill to FG rel_std for absent orders...")
                    cov_matrix_nominal = forward_fill_rel_std(cov_matrix_nominal, max_degree, logger=_logger)
                    if average_file:
                        cov_matrix = forward_fill_rel_std(cov_matrix, max_degree, logger=_logger)
                else:
                    _logger.info("  Forward-fill DISABLED (FORWARD_FILL_REL_STD_ENABLED=False)")
                if verbose_diagnostics:
                    log_rel_std_profile(cov_matrix_nominal, max_degree, "FG post-ffill", _logger, verbose=True)
            else:
                _logger.info("  Post-processing DISABLED (APPLY_COV_POSTPROCESSING=False)")

            # Read original MF34 from reference file (if present)
            # Use mf34_source_file if provided, otherwise fall back to endf_file
            mf34_ref = mf34_source_file if mf34_source_file else endf_file
            original_mf34_mt = None
            try:
                # Reuse endf_orig (already loaded at start of Step 10 from
                # endf_file) when mf34_ref points to the same file — the
                # source ENDF read takes ~30s for typical evaluations.
                if mf34_ref == endf_file:
                    endf_for_mf34 = endf_orig
                else:
                    endf_for_mf34 = read_endf(mf34_ref, mf_numbers=34)
                mf34_file = endf_for_mf34.get_file(34)
                if mf34_file is not None:
                    original_mf34_mt = mf34_file.sections.get(mt_number)
                    if original_mf34_mt is not None:
                        _logger.info(f"  Original MF34 found for MT{mt_number} in {mf34_ref}")
                        _logger.info(f"  Will merge with pipeline MF34")
                    else:
                        _logger.info(f"  MF34 file found but no MT{mt_number} section in {mf34_ref}")
                else:
                    _logger.info(f"  No MF34 section in {mf34_ref}")
            except Exception as exc:
                _logger.warning(
                    f"  Could not read MF34 from {mf34_ref}: {exc}",
                    console=True,
                )
                _logger.warning(
                    "  MF34 merge will be skipped — pipeline MF34 will only cover "
                    "the pipeline energy range.",
                    console=True,
                )

            # Helper to merge pipeline MF34 with original if available
            def _check_cov_psd(cov_matrix, label, max_order, logger):
                """Log PSD status of a covariance matrix before MF34 write."""
                eigvals = np.linalg.eigvalsh(cov_matrix)
                min_eig = float(eigvals[0])
                n_neg = int(np.sum(eigvals < -1e-10))
                n_params = cov_matrix.shape[0]
                n_energies = n_params // max_order

                if n_neg == 0:
                    logger.info(
                        f"  [PSD CHECK] {label}: OK "
                        f"(min_eig={min_eig:.3e}, {n_params} params = "
                        f"{n_energies} bins × {max_order} orders)"
                    )
                else:
                    logger.warning(
                        f"  [PSD CHECK] {label}: NOT PSD — "
                        f"{n_neg} negative eigenvalues "
                        f"(min_eig={min_eig:.3e}, {n_params} params = "
                        f"{n_energies} bins × {max_order} orders). "
                        f"Sampling from this matrix will be unreliable.",
                        console=True,
                    )

            def _maybe_merge(pipeline_mf34_obj, pipe_grid_ev):
                if original_mf34_mt is not None:
                    pipe_emin = float(pipe_grid_ev[0])
                    pipe_emax = float(pipe_grid_ev[-1])
                    return merge_mf34(
                        base_mf34=original_mf34_mt,
                        overlay_mf34=pipeline_mf34_obj,
                        overlay_energy_min_ev=pipe_emin,
                        overlay_energy_max_ev=pipe_emax,
                    )
                return pipeline_mf34_obj

            # Compute fine energy grid from midpoint bin boundaries.
            # Each energy point's fitting bin is [bin_lower, bin_upper] (midpoints
            # to its neighbours).  These boundaries must be contiguous
            # (bin_upper[i] == bin_lower[i+1]) to form a proper group structure —
            # contiguous_grid_from_bins hard-asserts it (a gapped grid would be a
            # semantically wrong ENDF grid).
            bin_by_idx = {b.index: b for b in energy_bins}
            valid_bins = [bin_by_idx[i] for i in energy_indices]
            energy_grid_ev = contiguous_grid_from_bins(valid_bins)

            # Write fine-grid MF34 if requested
            if mf34_covariance_type in ("fine", "both"):
                if average_file:
                    _logger.info(f"  Pre-MF34 check (fine avg): cov shape={cov_matrix.shape}, "
                                 f"inf={np.sum(np.isinf(cov_matrix))}, "
                                 f"nan={np.sum(np.isnan(cov_matrix))}, "
                                 f"grid len={len(energy_grid_ev)}, "
                                 f"grid finite={np.all(np.isfinite(energy_grid_ev))}")
                    mf34_fine_avg = create_mf34_from_covariance(
                        cov_matrix=cov_matrix,
                        energy_grid_ev=energy_grid_ev,
                        max_order=max_degree,
                        za=za,
                        awr=awr,
                        mat=mat,
                        mt=mt_number,
                    )
                    mf34_fine_avg = _maybe_merge(mf34_fine_avg, energy_grid_ev)
                    write_mf34_to_file(average_file, mf34_fine_avg, average_file)
                    _logger.info(f"  Fine MF34 added to average: {average_file}")

                if nominal_file:
                    _check_cov_psd(cov_matrix_nominal, "Fine-grid nominal covariance (Step 10)", max_degree, _logger)
                    _logger.info(f"  Pre-MF34 check (fine nom): cov shape={cov_matrix_nominal.shape}, "
                                 f"inf={np.sum(np.isinf(cov_matrix_nominal))}, "
                                 f"nan={np.sum(np.isnan(cov_matrix_nominal))}, "
                                 f"grid len={len(energy_grid_ev)}, "
                                 f"grid finite={np.all(np.isfinite(energy_grid_ev))}")
                    mf34_fine_nom = create_mf34_from_covariance(
                        cov_matrix=cov_matrix_nominal,
                        energy_grid_ev=energy_grid_ev,
                        max_order=max_degree,
                        za=za,
                        awr=awr,
                        mat=mat,
                        mt=mt_number,
                    )
                    mf34_fine_nom = _maybe_merge(mf34_fine_nom, energy_grid_ev)
                    write_mf34_to_file(nominal_file, mf34_fine_nom, nominal_file)
                    _logger.info(f"  Fine MF34 added to nominal: {nominal_file}")

                    # Fine MF33 goes in beside the fine MF34, on the same MF4
                    # grid. Written here rather than with the multigroup MF33 so
                    # it does not depend on the multigroup branch running.
                    if record_c0_channel and _mf33_products is not None:
                        try:
                            write_mf33_products(
                                _mf33_products,
                                fine_endf=nominal_file,
                                mg_endf=None,
                                mt=mt_number,
                                za=za, awr=awr, mat=mat,
                                rebuild_mt1=MF33_REBUILD_MT1,
                                pendf_path=_mf33_pendf_path,
                                logger=_logger,
                            )
                        except Exception as _e_mf33f:
                            _logger.error(
                                f"[ERROR] [MF33] Fine MF33 write failed: {_e_mf33f}",
                                console=True,
                            )

            # Compute nominal-relative grouped covariance (needed for MG MF34 and/or Step 11)
            cov_grouped_nominal = None
            if multigroup_result is not None:
                A = multigroup_result.aggregation_matrix
                # A operates on non-interpolated bins only; slice mc_mean_params
                # to match (same filtering as perform_adaptive_multigroup_collapse)
                valid_indices = [
                    i for i, nr in enumerate(nominal_results)
                    if not nr.interpolated and nr.has_data
                ]
                all_data_indices = [i for i, nr in enumerate(nominal_results) if nr.has_data]
                if len(all_data_indices) != len(valid_indices):
                    all_data_set = {v: pos for pos, v in enumerate(all_data_indices)}
                    valid_positions = [all_data_set[vi] for vi in valid_indices]
                    flat_indices = []
                    for pos in valid_positions:
                        for l in range(max_degree):
                            flat_indices.append(pos * max_degree + l)
                    flat_indices = np.array(flat_indices)
                    mc_mean_for_mg = mc_mean_params[flat_indices]
                    cov_abs_for_mg = cov_abs[np.ix_(flat_indices, flat_indices)]
                else:
                    mc_mean_for_mg = mc_mean_params
                    cov_abs_for_mg = cov_abs

                mc_mean_grouped = A @ mc_mean_for_mg

                # Build nominal parameter vector for non-interpolated bins
                nom_params_for_mg = np.zeros_like(mc_mean_for_mg)
                for k, vi in enumerate(valid_indices):
                    nr = nominal_results[vi]
                    if nr.has_data:
                        endf_nom = endf_normalize_legendre_coeffs(nr.nominal_coeffs, include_a0=False)
                        n = min(len(endf_nom), max_degree)
                        nom_params_for_mg[k * max_degree: k * max_degree + n] = endf_nom[:n]
                nom_mean_grouped = A @ nom_params_for_mg

                # Collapse absolute covariance through aggregation matrix
                cov_abs_grouped = A @ cov_abs_for_mg @ A.T

                # Direct absolute-to-nominal conversion
                cov_grouped_nominal = absolute_to_nominal_relative(cov_abs_grouped, nom_mean_grouped)
                _mg_nom_rel_std = np.sqrt(np.maximum(np.diag(cov_grouped_nominal), 0.0))
                _logger.info(f"  MG abs→nominal conversion: max rel_std = {np.max(_mg_nom_rel_std)*100:.1f}%")
                if verbose_diagnostics:
                    log_rel_std_profile(cov_grouped_nominal, max_degree, "MG post-convert", _logger, verbose=True)

                # Near-zero regularization on nominal-relative covariance (MG)
                if regularize_near_zero:
                    cov_grouped_nominal, _ = regularize_near_zero_relative_covariance(
                        cov_rel=cov_grouped_nominal,
                        mean_params=nom_mean_grouped,
                        cov_abs=cov_abs_grouped,
                        max_order=max_degree,
                        snr_threshold=near_zero_snr_threshold,
                        n_neighbors=near_zero_n_neighbors,
                        logger=_logger,
                    )
                    if verbose_diagnostics:
                        log_rel_std_profile(cov_grouped_nominal, max_degree, "MG post-nz-nom", _logger, verbose=True)

                # Between-experiment scatter floor at the multigroup level
                # (mirror of the FG floor — without this the published MF34
                # ignores APPLY_BETWEEN_EXP_FLOOR entirely).
                cov_grouped_nominal, _bexp_mg_nom = apply_between_experiment_floor_mg(
                    cov_rel=cov_grouped_nominal,
                    nominal_results=nominal_results,
                    valid_indices=valid_indices,
                    groups=multigroup_result.groups,
                    max_order=max_degree,
                    logger=_logger,
                    apply=apply_between_exp_floor,
                )
                if average_file:
                    multigroup_result.cov_grouped, _bexp_mg_avg = apply_between_experiment_floor_mg(
                        cov_rel=multigroup_result.cov_grouped,
                        nominal_results=nominal_results,
                        valid_indices=valid_indices,
                        groups=multigroup_result.groups,
                        max_order=max_degree,
                        logger=_logger,
                        apply=apply_between_exp_floor,
                    )
                if verbose_diagnostics:
                    log_rel_std_profile(cov_grouped_nominal, max_degree, "MG post-between-exp", _logger, verbose=True)

                if apply_cov_postprocessing:
                    # Step 2: Dip/spike smoothing & absent-order interpolation (MG)
                    _mg_valid = multigroup_result.valid_mask_grouped
                    cov_grouped_nominal, _mg_smooth_nom = smooth_absent_order_uncertainties(
                        cov_rel=cov_grouped_nominal,
                        valid_mask=_mg_valid,
                        max_order=max_degree,
                        min_rel_std=SMOOTH_MIN_REL_STD,
                        dip_fraction=SMOOTH_DIP_FRACTION,
                        spike_factor=MG_SMOOTH_SPIKE_FACTOR,
                        dip_n_neighbors=MG_SMOOTH_DIP_N_NEIGHBORS,
                        median_fill_threshold=SMOOTH_MEDIAN_FILL_THRESHOLD,
                        logger=_logger,
                    )
                    if average_file:
                        multigroup_result.cov_grouped, _mg_smooth_avg = smooth_absent_order_uncertainties(
                            cov_rel=multigroup_result.cov_grouped,
                            valid_mask=_mg_valid,
                            max_order=max_degree,
                            min_rel_std=SMOOTH_MIN_REL_STD,
                            dip_fraction=SMOOTH_DIP_FRACTION,
                            spike_factor=MG_SMOOTH_SPIKE_FACTOR,
                            dip_n_neighbors=MG_SMOOTH_DIP_N_NEIGHBORS,
                            median_fill_threshold=SMOOTH_MEDIAN_FILL_THRESHOLD,
                            logger=_logger,
                        )
                    if verbose_diagnostics:
                        log_rel_std_profile(cov_grouped_nominal, max_degree, "MG post-smooth", _logger, verbose=True)

                    # Spatial smoothing at MG level (before cap)
                    if SMOOTH_DIAGONAL_WINDOW >= 3:
                        cov_grouped_nominal = smooth_diagonal_median(
                            cov_rel=cov_grouped_nominal,
                            max_order=max_degree,
                            window=SMOOTH_DIAGONAL_WINDOW,
                            logger=_logger,
                        )
                        if average_file:
                            multigroup_result.cov_grouped = smooth_diagonal_median(
                                cov_rel=multigroup_result.cov_grouped,
                                max_order=max_degree,
                                window=SMOOTH_DIAGONAL_WINDOW,
                                logger=_logger,
                            )
                        if verbose_diagnostics:
                            log_rel_std_profile(cov_grouped_nominal, max_degree, "MG post-median", _logger, verbose=True)

                    # Step 4: Hard cap (MG)
                    if ORDER_REL_STD_CAPS is not None:
                        cov_grouped_nominal, _mg_cap_nom = cap_order_relative_uncertainty(
                            cov_rel=cov_grouped_nominal,
                            max_order=max_degree,
                            order_caps=ORDER_REL_STD_CAPS,
                            logger=_logger,
                        )
                        if average_file:
                            multigroup_result.cov_grouped, _mg_cap_avg = cap_order_relative_uncertainty(
                                cov_rel=multigroup_result.cov_grouped,
                                max_order=max_degree,
                                order_caps=ORDER_REL_STD_CAPS,
                                logger=_logger,
                            )
                    if verbose_diagnostics:
                        log_rel_std_profile(cov_grouped_nominal, max_degree, "MG post-cap", _logger, verbose=True)

                    # Forward-fill absent-order rel_std at MG level
                    if FORWARD_FILL_REL_STD_ENABLED:
                        _logger.info("  Applying forward-fill to MG rel_std for absent orders...")
                        cov_grouped_nominal = forward_fill_rel_std(cov_grouped_nominal, max_degree, logger=_logger)
                        if average_file:
                            multigroup_result.cov_grouped = forward_fill_rel_std(multigroup_result.cov_grouped, max_degree, logger=_logger)
                    else:
                        _logger.info("  Forward-fill DISABLED at MG level")
                    if verbose_diagnostics:
                        log_rel_std_profile(cov_grouped_nominal, max_degree, "MG post-ffill", _logger, verbose=True)
                else:
                    _logger.info("  MG post-processing DISABLED (APPLY_COV_POSTPROCESSING=False)")

            # --- Second-pass regrouping after smoothing ---
            if (multigroup_regroup_after_smooth
                    and multigroup_result is not None
                    and cov_grouped_nominal is not None):
                from scripts.multigroup_collapse import idx as _mg_idx

                # Re-extract fine-grid arrays (same filtering as perform_adaptive_multigroup_collapse)
                _rg_valid_idx = [
                    i for i, nr in enumerate(nominal_results)
                    if not nr.interpolated and nr.has_data
                ]
                _rg_n_fine = len(_rg_valid_idx)
                _rg_valid_bins = [energy_bins[i] for i in _rg_valid_idx]
                _rg_valid_nominal = [nominal_results[i] for i in _rg_valid_idx]
                _rg_bin_lower = np.array([eb.bin_lower_mev for eb in _rg_valid_bins])
                _rg_bin_upper = np.array([eb.bin_upper_mev for eb in _rg_valid_bins])
                _rg_bin_widths = _rg_bin_upper - _rg_bin_lower

                # Build fine mean vector
                _rg_mean_fine = np.zeros(_rg_n_fine * max_degree)
                for _i, _nr in enumerate(_rg_valid_nominal):
                    _c = _nr.nominal_coeffs
                    for _l in range(1, min(max_degree + 1, len(_c) + 1)):
                        _rg_mean_fine[_mg_idx(_i, _l, max_degree)] = _c[_l - 1] if _l - 1 < len(_c) else 0.0

                # Build valid_mask — same rule as `valid_mask_s7` and the Step-7b
                # collapse. This site was the second copy of the legacy
                # winner-take-all rule; dormant because
                # MULTIGROUP_REGROUP_AFTER_SMOOTH is False, but it would have
                # silently undone Phase 3 the moment that flag was turned on.
                _rg_valid_mask = np.zeros(_rg_n_fine * max_degree, dtype=bool)
                for _i, _nr in enumerate(_rg_valid_nominal):
                    for _l in range(1, _mg_valid_orders(_nr, max_degree) + 1):
                        _rg_valid_mask[_mg_idx(_i, _l, max_degree)] = True

                # Use absolute covariance sliced to non-interpolated bins for re-collapse
                _rg_cov_fine = cov_abs_for_mg

                merged = try_merge_adjacent_multigroups(
                    multigroup_result=multigroup_result,
                    cov_mg_smoothed=cov_grouped_nominal,
                    cov_fine=_rg_cov_fine,
                    mean_fine=_rg_mean_fine,
                    fine_bin_widths_mev=_rg_bin_widths,
                    fine_bin_lower_mev=_rg_bin_lower,
                    fine_bin_upper_mev=_rg_bin_upper,
                    n_fine=_rg_n_fine,
                    max_order=max_degree,
                    rho_min=multigroup_rho_min,
                    valid_mask=_rg_valid_mask,
                    logger=_logger,
                )

                if merged is not None:
                    n_old = len(multigroup_result.groups)
                    multigroup_result = merged

                    # Recompute nominal-relative grouped cov with new A
                    A = multigroup_result.aggregation_matrix
                    mc_mean_grouped = A @ mc_mean_for_mg
                    nom_mean_grouped = A @ nom_params_for_mg

                    # Re-collapse absolute covariance through new aggregation matrix
                    cov_abs_grouped = A @ cov_abs_for_mg @ A.T

                    # Direct absolute-to-nominal conversion
                    cov_grouped_nominal = absolute_to_nominal_relative(cov_abs_grouped, nom_mean_grouped)
                    _rg_nom_rel_std = np.sqrt(np.maximum(np.diag(cov_grouped_nominal), 0.0))
                    _logger.info(f"  RG abs→nominal conversion: max rel_std = {np.max(_rg_nom_rel_std)*100:.1f}%")
                    if verbose_diagnostics:
                        log_rel_std_profile(cov_grouped_nominal, max_degree, "RG post-convert", _logger, verbose=True)

                    # Near-zero regularization on nominal-relative covariance (RG)
                    if regularize_near_zero:
                        cov_grouped_nominal, _ = regularize_near_zero_relative_covariance(
                            cov_rel=cov_grouped_nominal,
                            mean_params=nom_mean_grouped,
                            cov_abs=cov_abs_grouped,
                            max_order=max_degree,
                            snr_threshold=near_zero_snr_threshold,
                            n_neighbors=near_zero_n_neighbors,
                            logger=_logger,
                        )
                        if verbose_diagnostics:
                            log_rel_std_profile(cov_grouped_nominal, max_degree, "RG post-nz-nom", _logger, verbose=True)

                    # Re-apply between-experiment floor on the regrouped MG cov.
                    cov_grouped_nominal, _bexp_rg_nom = apply_between_experiment_floor_mg(
                        cov_rel=cov_grouped_nominal,
                        nominal_results=nominal_results,
                        valid_indices=valid_indices,
                        groups=multigroup_result.groups,
                        max_order=max_degree,
                        logger=_logger,
                        apply=apply_between_exp_floor,
                    )
                    if average_file:
                        multigroup_result.cov_grouped, _bexp_rg_avg = apply_between_experiment_floor_mg(
                            cov_rel=multigroup_result.cov_grouped,
                            nominal_results=nominal_results,
                            valid_indices=valid_indices,
                            groups=multigroup_result.groups,
                            max_order=max_degree,
                            logger=_logger,
                            apply=apply_between_exp_floor,
                        )
                    if verbose_diagnostics:
                        log_rel_std_profile(cov_grouped_nominal, max_degree, "RG post-between-exp", _logger, verbose=True)

                    if apply_cov_postprocessing:
                        # Step 2: Dip/spike smoothing & absent-order interpolation (RG)
                        _mg_valid = multigroup_result.valid_mask_grouped
                        cov_grouped_nominal, _ = smooth_absent_order_uncertainties(
                            cov_rel=cov_grouped_nominal,
                            valid_mask=_mg_valid,
                            max_order=max_degree,
                            min_rel_std=SMOOTH_MIN_REL_STD,
                            dip_fraction=SMOOTH_DIP_FRACTION,
                            spike_factor=MG_SMOOTH_SPIKE_FACTOR,
                            dip_n_neighbors=MG_SMOOTH_DIP_N_NEIGHBORS,
                            median_fill_threshold=SMOOTH_MEDIAN_FILL_THRESHOLD,
                            logger=_logger,
                        )
                        if average_file:
                            multigroup_result.cov_grouped, _ = smooth_absent_order_uncertainties(
                                cov_rel=multigroup_result.cov_grouped,
                                valid_mask=_mg_valid,
                                max_order=max_degree,
                                min_rel_std=SMOOTH_MIN_REL_STD,
                                dip_fraction=SMOOTH_DIP_FRACTION,
                                spike_factor=MG_SMOOTH_SPIKE_FACTOR,
                                dip_n_neighbors=MG_SMOOTH_DIP_N_NEIGHBORS,
                                median_fill_threshold=SMOOTH_MEDIAN_FILL_THRESHOLD,
                                logger=_logger,
                            )
                        if verbose_diagnostics:
                            log_rel_std_profile(cov_grouped_nominal, max_degree, "RG post-smooth", _logger, verbose=True)

                        # Spatial smoothing at RG level (before cap)
                        if SMOOTH_DIAGONAL_WINDOW >= 3:
                            cov_grouped_nominal = smooth_diagonal_median(
                                cov_rel=cov_grouped_nominal,
                                max_order=max_degree,
                                window=SMOOTH_DIAGONAL_WINDOW,
                                logger=_logger,
                            )
                            if average_file:
                                multigroup_result.cov_grouped = smooth_diagonal_median(
                                    cov_rel=multigroup_result.cov_grouped,
                                    max_order=max_degree,
                                    window=SMOOTH_DIAGONAL_WINDOW,
                                    logger=_logger,
                                )
                            if verbose_diagnostics:
                                log_rel_std_profile(cov_grouped_nominal, max_degree, "RG post-median", _logger, verbose=True)

                        # Step 4: Hard cap (RG)
                        if ORDER_REL_STD_CAPS is not None:
                            cov_grouped_nominal, _ = cap_order_relative_uncertainty(
                                cov_rel=cov_grouped_nominal,
                                max_order=max_degree,
                                order_caps=ORDER_REL_STD_CAPS,
                                logger=_logger,
                            )
                            if average_file:
                                multigroup_result.cov_grouped, _ = cap_order_relative_uncertainty(
                                    cov_rel=multigroup_result.cov_grouped,
                                    max_order=max_degree,
                                    order_caps=ORDER_REL_STD_CAPS,
                                    logger=_logger,
                                )
                        if verbose_diagnostics:
                            log_rel_std_profile(cov_grouped_nominal, max_degree, "RG post-cap", _logger, verbose=True)

                        # Forward-fill absent-order rel_std at RG level
                        if FORWARD_FILL_REL_STD_ENABLED:
                            _logger.info("  Applying forward-fill to RG rel_std for absent orders...")
                            cov_grouped_nominal = forward_fill_rel_std(cov_grouped_nominal, max_degree, logger=_logger)
                            if average_file:
                                multigroup_result.cov_grouped = forward_fill_rel_std(multigroup_result.cov_grouped, max_degree, logger=_logger)
                        else:
                            _logger.info("  Forward-fill DISABLED at RG level")
                        if verbose_diagnostics:
                            log_rel_std_profile(cov_grouped_nominal, max_degree, "RG post-ffill", _logger, verbose=True)

                    _logger.info(f"  Regrouped: {n_old} -> {len(multigroup_result.groups)} groups")

            # Phase 5: Optional hard-zero of small multigroup correlations.
            # Applied to both avg and nominal multigroup matrices before MF34
            # write. ENDF-6 MF34 LB=5 is dense, so this does not save file
            # size; it removes sub-noise correlations from the published file.
            if isinstance(MULTIGROUP_CORRELATION_THRESHOLD, str):
                if MULTIGROUP_CORRELATION_THRESHOLD.lower() == "auto":
                    _mg_thr = 1.0 / np.sqrt(max(n_samples, 1))
                    _logger.info(
                        f"  MULTIGROUP_CORRELATION_THRESHOLD='auto' -> "
                        f"1/sqrt({n_samples}) = {_mg_thr:.4f}"
                    )
                else:
                    raise ValueError(
                        f"MULTIGROUP_CORRELATION_THRESHOLD={MULTIGROUP_CORRELATION_THRESHOLD!r} "
                        f"is not a recognized string (use 'auto' or a numeric value)."
                    )
            else:
                _mg_thr = float(MULTIGROUP_CORRELATION_THRESHOLD)

            if _mg_thr > 0 and multigroup_result is not None:
                # avg branch only consumed when average_file is set; the
                # post-threshold Higham PSD repair is the dominant cost in
                # Step 10 (~10 min on a 2946² MG matrix), so skip when unused.
                if average_file:
                    multigroup_result.cov_grouped, _ = threshold_small_correlations(
                        multigroup_result.cov_grouped,
                        threshold=_mg_thr,
                        label="MG cov_grouped (avg)",
                        logger=_logger,
                    )
                if cov_grouped_nominal is not None:
                    cov_grouped_nominal, _ = threshold_small_correlations(
                        cov_grouped_nominal,
                        threshold=_mg_thr,
                        label="MG cov_grouped (nominal)",
                        logger=_logger,
                    )

            # Write multigroup MF34 if requested and available
            if mf34_covariance_type in ("multigroup", "both") and cov_grouped_nominal is not None:
                # For multigroup, write to separate files with _mg suffix
                if average_file:
                    _logger.info(f"  Pre-MF34 check (MG avg): cov shape={multigroup_result.cov_grouped.shape}, "
                                 f"inf={np.sum(np.isinf(multigroup_result.cov_grouped))}, "
                                 f"nan={np.sum(np.isnan(multigroup_result.cov_grouped))}, "
                                 f"grid len={len(multigroup_result.group_boundaries_ev)}, "
                                 f"grid finite={np.all(np.isfinite(multigroup_result.group_boundaries_ev))}")
                    mf34_mg_avg = create_mf34_from_covariance(
                        cov_matrix=multigroup_result.cov_grouped,
                        energy_grid_ev=multigroup_result.group_boundaries_ev,
                        max_order=max_degree,
                        za=za,
                        awr=awr,
                        mat=mat,
                        mt=mt_number,
                    )
                    mf34_mg_avg = _maybe_merge(mf34_mg_avg, multigroup_result.group_boundaries_ev)
                    mg_avg_file = average_file.replace('.txt', '_mg.endf').replace('.endf', '_mg.endf')
                    if mg_avg_file == average_file:
                        mg_avg_file = average_file + '_mg'
                    # Copy average file and add multigroup MF34
                    import shutil
                    shutil.copy(average_file, mg_avg_file)
                    write_mf34_to_file(mg_avg_file, mf34_mg_avg, mg_avg_file)
                    _logger.info(f"  Multigroup MF34 written to: {mg_avg_file}")

                if nominal_file:
                    _check_cov_psd(cov_grouped_nominal, "MG nominal covariance (Step 10)", max_degree, _logger)
                    _logger.info(f"  Pre-MF34 check (MG nom): cov shape={cov_grouped_nominal.shape}, "
                                 f"inf={np.sum(np.isinf(cov_grouped_nominal))}, "
                                 f"nan={np.sum(np.isnan(cov_grouped_nominal))}, "
                                 f"grid len={len(multigroup_result.group_boundaries_ev)}, "
                                 f"grid finite={np.all(np.isfinite(multigroup_result.group_boundaries_ev))}")
                    mf34_mg_nom = create_mf34_from_covariance(
                        cov_matrix=cov_grouped_nominal,
                        energy_grid_ev=multigroup_result.group_boundaries_ev,
                        max_order=max_degree,
                        za=za,
                        awr=awr,
                        mat=mat,
                        mt=mt_number,
                    )
                    mf34_mg_nom = _maybe_merge(mf34_mg_nom, multigroup_result.group_boundaries_ev)
                    mg_nom_file = nominal_file.replace('.txt', '_mg.endf').replace('.endf', '_mg.endf')
                    if mg_nom_file == nominal_file:
                        mg_nom_file = nominal_file + '_mg'
                    import shutil
                    shutil.copy(nominal_file, mg_nom_file)
                    write_mf34_to_file(mg_nom_file, mf34_mg_nom, mg_nom_file)
                    _logger.info(f"  Multigroup MF34 written to: {mg_nom_file}")

                    # --- Phase-2: MF33 MT2 (elastic magnitude covariance) ----
                    # The _mg product gets MF33 on the MF33 channel's OWN
                    # adaptive grid — not the MF34 one. The shape and magnitude
                    # channels have different correlation lengths, and ENDF lets
                    # each MF/MT section carry its own boundaries. (The fine
                    # MF33 was written beside the fine MF34 further up, and the
                    # _mg file inherits it from the copy — this replaces it.)
                    #
                    # RANGE-MERGED into the host MF33 MT2: our matrix replaces
                    # the host inside the working range, the host survives
                    # outside, and in/out cross terms are zeroed (a documented
                    # factorization assumption that Phase 3 measures, never a
                    # format limit). Sibling MF33 MT sections (1, 4, 5, 16, 102,
                    # 103) are preserved by the per-MT section writer.
                    #
                    # The MF3 central is intentionally left at the host value;
                    # the DCS magnitude is 4*pi*c0, reconstructable from
                    # mf33_c0_nominal.npy (no File-3 rewrite required).
                    if record_c0_channel and _mf33_products is not None:
                        try:
                            write_mf33_products(
                                _mf33_products,
                                fine_endf=None,
                                mg_endf=mg_nom_file,
                                mt=mt_number,
                                za=za, awr=awr, mat=mat,
                                mg_representation=MF33_MG_REPRESENTATION,
                                rebuild_mt1=MF33_REBUILD_MT1,
                                pendf_path=_mf33_pendf_path,
                                logger=_logger,
                            )
                        except Exception as _e_mf33w:
                            _logger.error(
                                f"[ERROR] [MF33] MF33 write failed: {_e_mf33w}",
                                console=True,
                            )

            elif mf34_covariance_type in ("multigroup", "both") and cov_grouped_nominal is None:
                if multigroup_failure_reason:
                    _logger.warning(f"  Multigroup covariance requested but failed: {multigroup_failure_reason}", console=True)
                elif not generate_multigroup_covariance:
                    _logger.warning("  Multigroup covariance requested but not computed (enable GENERATE_MULTIGROUP_COVARIANCE)")
                else:
                    _logger.warning("  Multigroup covariance requested but not computed (unknown reason)")

            # --- Resolve MF34 source file for Step 11 ---
            mf34_sample_source = None
            if generate_mf34_samples and nominal_file:
                sr = sampling_resolution

                if sr == "fine":
                    if merge_original_mf34 and mf34_covariance_type in ("fine", "both"):
                        # Already written as merged fine MF34 in nominal_file
                        mf34_sample_source = nominal_file
                    else:
                        # Write fine MF34 for sampling (merged or pipeline-only)
                        import shutil
                        suffix = "_pipeline" if not merge_original_mf34 else "_fine_sampling"
                        base_stem = Path(nominal_file).stem
                        base_dir = Path(nominal_file).parent
                        sampling_file = str(base_dir / f"{base_stem}{suffix}.endf")
                        shutil.copy(nominal_file, sampling_file)
                        _check_cov_psd(cov_matrix_nominal, "Fine-grid nominal covariance", max_degree, _logger)
                        _logger.info(f"  Pre-MF34 check (fine nom): cov shape={cov_matrix_nominal.shape}, "
                                     f"inf={np.sum(np.isinf(cov_matrix_nominal))}, "
                                     f"nan={np.sum(np.isnan(cov_matrix_nominal))}, "
                                     f"grid len={len(energy_grid_ev)}, "
                                     f"grid finite={np.all(np.isfinite(energy_grid_ev))}")
                        mf34_fine_obj = create_mf34_from_covariance(
                            cov_matrix=cov_matrix_nominal,
                            energy_grid_ev=energy_grid_ev,
                            max_order=max_degree,
                            za=za, awr=awr, mat=mat, mt=mt_number,
                        )
                        if merge_original_mf34:
                            mf34_fine_obj = _maybe_merge(mf34_fine_obj, energy_grid_ev)
                        write_mf34_to_file(sampling_file, mf34_fine_obj, sampling_file)
                        mf34_sample_source = sampling_file
                        _logger.info(f"  Fine MF34 ({'merged' if merge_original_mf34 else 'pipeline-only'}) for sampling: {sampling_file}")

                elif sr == "multigroup" and cov_grouped_nominal is not None:
                    if merge_original_mf34 and mg_nom_file:
                        # Already written as merged MG MF34
                        mf34_sample_source = mg_nom_file
                    else:
                        # Write multigroup MF34 for sampling (merged or pipeline-only)
                        import shutil
                        suffix = "_mg_pipeline" if not merge_original_mf34 else "_mg_sampling"
                        base_stem = Path(nominal_file).stem
                        base_dir = Path(nominal_file).parent
                        sampling_file = str(base_dir / f"{base_stem}{suffix}.endf")
                        shutil.copy(nominal_file, sampling_file)
                        _check_cov_psd(cov_grouped_nominal, "Multigroup nominal covariance", max_degree, _logger)
                        _logger.info(f"  Pre-MF34 check (MG sampling): cov shape={cov_grouped_nominal.shape}, "
                                     f"inf={np.sum(np.isinf(cov_grouped_nominal))}, "
                                     f"nan={np.sum(np.isnan(cov_grouped_nominal))}, "
                                     f"grid len={len(multigroup_result.group_boundaries_ev)}, "
                                     f"grid finite={np.all(np.isfinite(multigroup_result.group_boundaries_ev))}")
                        mf34_mg_obj = create_mf34_from_covariance(
                            cov_matrix=cov_grouped_nominal,
                            energy_grid_ev=multigroup_result.group_boundaries_ev,
                            max_order=max_degree,
                            za=za, awr=awr, mat=mat, mt=mt_number,
                        )
                        if merge_original_mf34:
                            mf34_mg_obj = _maybe_merge(mf34_mg_obj, multigroup_result.group_boundaries_ev)
                        write_mf34_to_file(sampling_file, mf34_mg_obj, sampling_file)
                        mf34_sample_source = sampling_file
                        _logger.info(f"  Multigroup MF34 ({'merged' if merge_original_mf34 else 'pipeline-only'}) for sampling: {sampling_file}")

                if mf34_sample_source:
                    _logger.info(f"  Step 11 MF34 source: {mf34_sample_source}")
                else:
                    _logger.warning(
                        f"  Could not resolve MF34 for ({sr}, merge={merge_original_mf34}) — "
                        f"multigroup_result={'available' if multigroup_result is not None else 'None'}",
                        console=True,
                    )

        except Exception as e:
            _logger.error(f"[ERROR] [MF34] Failed to write MF34: {str(e)}", console=True)
            _logger.error(f"  Traceback:\n{traceback.format_exc()}", console=False)
            mf34_sample_source = None
            _warning_counts['mf34_write_failed'] = 1

        _logger.info(f"#-- END STEP 10 (elapsed: {time.time() - t_step:.2f}s) ------------------------------------")

    else:
        mf34_sample_source = None

    # Step 11: Generate perturbed ENDF samples from MF34 covariance (Pipeline B)
    if generate_mf34_samples:
        _logger.info("")
        _logger.info("#-- STEP 11: MF34 sampling (Pipeline B) ------------------------------------")
        _logger.info(f"  Resolution: {sampling_resolution}, Merge original: {merge_original_mf34}")
        _logger.info(f"  Space: {sampling_space}, Decomposition: {sampling_decomposition}")
        _logger.info(f"  Sampling method: {sampling_method}, N={n_samples}")

        if mf34_sample_source and os.path.exists(mf34_sample_source):
            _logger.info(f"  MF34 source file: {mf34_sample_source}")
            try:
                perturb_ENDF_files(
                    endf_files=nominal_file,
                    mt_list=[mt_number],
                    legendre_coeffs=list(range(1, max_degree + 1)),
                    num_samples=n_samples,
                    mf34_cov_files=mf34_sample_source,
                    space=sampling_space,
                    decomposition_method=sampling_decomposition,
                    sampling_method=sampling_method,
                    output_dir=str(output_path),
                    seed=base_seed,
                    nprocs=n_procs,
                    generate_ace=generate_mf34_ace,
                    njoy_exe=ace_njoy_exe if generate_mf34_ace else None,
                    temperatures=ace_temperatures if generate_mf34_ace else None,
                    library_name=ace_library_name if generate_mf34_ace else None,
                    njoy_version=ace_njoy_version,
                    xsdir_file=ace_xsdir_file if generate_mf34_ace else None,
                )
                _logger.info(f"  [INFO] [MF34] Pipeline B samples written to: {output_path / 'endf'}")
            except Exception as e:
                _logger.error(f"[ERROR] [MF34] Pipeline B sampling failed: {str(e)}", console=True)
                _logger.error(f"  Traceback:\n{traceback.format_exc()}", console=False)
        else:
            _logger.warning(
                f"[WARN] [MF34] Source for ({sampling_resolution}, merge={merge_original_mf34}) "
                f"not available -- skipping Step 11",
                console=True,
            )

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    total_time = time.time() - t_exfor_start
    _logger.info("")
    _logger.info("#== SUMMARY ================================================================")
    _logger.info(f">> total_elapsed = {total_time:.2f}s")
    _logger.info(f">> api_version = kika.exfor (v2)")
    _logger.info(f">> bins_total = {len(nominal_results)}")
    _logger.info(f">> bins_with_data = {n_with_data}")
    _logger.info(f">> bins_interpolated = {n_interpolated}")
    _logger.info(f">> samples_generated = {n_samples}")
    if cov_matrix is not None:
        _logger.info(f">> covariance_shape = {cov_matrix.shape}")
    if multigroup_result is not None:
        _logger.info(f">> multigroup_bins = {len(multigroup_result.groups)}")
    _logger.info(f"  Pipeline A: nominal={'written' if nominal_file else 'skipped'}, "
                 f"samples={len(output_files) if output_files else 'skipped'}")
    _logger.info(f"  Pipeline B: {'enabled' if generate_mf34_samples else 'disabled'}")
    _logger.info("#== END SUMMARY ============================================================")

    # ── WARNINGS ─────────────────────────────────────────────────────────────
    if _warning_counts:
        _logger.info("")
        _logger.info("#== WARNINGS ===============================================================")
        for wkey, wcount in _warning_counts.items():
            _logger.info(f"  {wkey} -- {wcount}")
        _logger.info("#== END WARNINGS ===========================================================")

    # NOTE: the post-pipeline chi²/N diagnostics block has been removed.
    # The chi² statistic computed against EXFOR data inside the pipeline was
    # misleading: it relied on the fitted (nominal) coefficients evaluated at
    # the same data used to fit them, and combined per-point and per-experiment
    # uncertainties in a way that did not match how the manifest now decomposes
    # σ_stat / σ_sys. External chi² analysis (see scripts/chi2_analysis.ipynb)
    # is the right place for goodness-of-fit comparisons across libraries.

    _logger.info(f"  [INFO] Completed! Output: {output_path}", console=True)
    _logger.info(f"  [INFO] Total time: {total_time:.1f}s", console=True)

    return nominal_results, all_samples, output_files


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    run_exfor_to_endf_sampling_v2(
        endf_file=ENDF_FILE,
        exfor_directory=EXFOR_DIRECTORY,
        output_dir=OUTPUT_DIR,
        n_samples=N_SAMPLES,
        energy_min_mev=ENERGY_MIN_MEV,
        energy_max_mev=ENERGY_MAX_MEV,
        mt_number=MT_NUMBER,
        max_degree=MAX_LEGENDRE_DEGREE,
        select_degree=SELECT_DEGREE,
        ridge_lambda=RIDGE_LAMBDA,
        m_proj_u=M_PROJ_U,
        m_targ_u=M_TARG_U,
        use_band_discrepancy=USE_BAND_DISCREPANCY,
        min_points_per_band=MIN_POINTS_PER_BAND,
        max_band_scale=MAX_BAND_SCALE_FACTOR,
        tau_smoothing_window=TAU_SMOOTHING_WINDOW,
        tau_prior_floor=TAU_PRIOR_FLOOR,
        tau_prior_neff_threshold=TAU_PRIOR_NEFF_THRESHOLD,
        tau_prior_percentile=TAU_PRIOR_PERCENTILE,
        sigma_norm_systematic=NORM_SYSTEMATIC_SIGMA,
        sigma_norm_common_mode=NORM_COMMON_MODE_SIGMA,
        use_model_averaging=USE_MODEL_AVERAGING,
        min_degree_for_averaging=MIN_DEGREE_FOR_AVERAGING,
        n_eff_warning_threshold=N_EFF_WARNING_THRESHOLD,
        n_procs=N_PROCS,
        base_seed=BASE_SEED,
        generate_nominal_endf=GENERATE_NOMINAL_ENDF,
        generate_mc_mean_endf=GENERATE_MC_MEAN_ENDF,
        generate_fitting_samples=GENERATE_FITTING_SAMPLES,
        generate_fitting_ace=GENERATE_FITTING_ACE,
        save_covariance_files=SAVE_COVARIANCE_FILES,
        # Multigroup covariance options
        generate_multigroup_covariance=GENERATE_MULTIGROUP_COVARIANCE,
        multigroup_rho_min=MULTIGROUP_RHO_MIN,
        multigroup_sigma_ratio_max=MULTIGROUP_SIGMA_RATIO_MAX,
        multigroup_variance_pct_min=MULTIGROUP_VARIANCE_PCT_MIN,
        multigroup_variance_pct_max=MULTIGROUP_VARIANCE_PCT_MAX,
        multigroup_variance_ratio_ref=MULTIGROUP_VARIANCE_RATIO_REF,
        multigroup_regroup_after_smooth=MULTIGROUP_REGROUP_AFTER_SMOOTH,
        mf34_covariance_type=MF34_COVARIANCE_TYPE,
        use_original_mf34_grid=USE_ORIGINAL_MF34_GRID,
        # Database configuration
        exfor_db_path=EXFOR_DB_PATH,
        exfor_source=EXFOR_SOURCE,
        target_zaid=TARGET_ZAIDS,
        target_projectile=TARGET_PROJECTILE,
        supplementary_json_files=SUPPLEMENTARY_JSON_FILES,
        # Experiment exclusion and uncertainty floor
        exclude_experiments=EXCLUDE_EXPERIMENTS,
        min_relative_uncertainty=MIN_RELATIVE_UNCERTAINTY,
        # Energy grid source
        energy_grid_source=ENERGY_GRID_SOURCE,
        union_grid_subentries=UNION_GRID_SUBENTRIES,
        # Covariance cap (Layer 1) and positivity projection (Layer 2)
        apply_covariance_cap=APPLY_COVARIANCE_CAP,
        max_relative_std_cap=MAX_RELATIVE_STD_CAP,
        regularize_near_zero=REGULARIZE_NEAR_ZERO_REL_UNC,
        near_zero_snr_threshold=NEAR_ZERO_SNR_THRESHOLD,
        near_zero_n_neighbors=NEAR_ZERO_N_NEIGHBORS,
        apply_between_exp_floor=APPLY_BETWEEN_EXP_FLOOR,
        apply_cov_postprocessing=APPLY_COV_POSTPROCESSING,
        apply_positivity_projection=APPLY_POSITIVITY_PROJECTION,
        positivity_check_points=POSITIVITY_CHECK_POINTS,
        # File output options
        save_correlation_matrices=SAVE_CORRELATION_MATRICES,
        save_tmc_parquet=SAVE_TMC_PARQUET,
        save_raw_kw_parquet=SAVE_RAW_KW_PARQUET,
        save_multigroup_diagnostics_csv=SAVE_MULTIGROUP_DIAGNOSTICS_CSV,
        save_nominal_fits=SAVE_NOMINAL_FITS,
        # ACE common options
        ace_temperatures=ACE_TEMPERATURES,
        ace_njoy_exe=ACE_NJOY_EXE,
        ace_library_name=ACE_LIBRARY_NAME,
        ace_njoy_version=ACE_NJOY_VERSION,
        ace_xsdir_file=ACE_XSDIR_FILE,
        ace_skip_existing=ACE_SKIP_EXISTING,
        mf34_source_file=MF34_SOURCE_FILE,
        # MF34 sampling (Pipeline B)
        generate_mf34_samples=GENERATE_MF34_SAMPLES,
        generate_mf34_ace=GENERATE_MF34_ACE,
        sampling_resolution=SAMPLING_RESOLUTION,
        merge_original_mf34=MERGE_ORIGINAL_MF34,
        sampling_space=SAMPLING_SPACE,
        sampling_decomposition=SAMPLING_DECOMPOSITION,
        sampling_method=SAMPLING_METHOD,
        # Diagnostic output
        verbose_diagnostics=VERBOSE_DIAGNOSTICS,
    )
