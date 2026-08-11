"""Fe-56 elastic evaluation: EXFOR angular distributions -> ENDF MF4/MF34/MF33.

Research fork of ``exfor_to_endf_sampling_v2.py``, which is the frozen thesis
pipeline and is never edited for research.

Fits Legendre coefficients to EXFOR angular distributions in each energy bin,
propagates the experimental uncertainty budget by Monte Carlo, and writes the
result as a consistent ENDF tape: the MF4 central, the MF34 shape covariance,
the MF33 elastic magnitude covariance, and the MF33<->MF34 cross term.

Products, in the order they are built (see WHAT THIS RUN PRODUCES below):

    nominal ENDF  ->  _mg ENDF  ->  _a0cross ENDF        <-- the deliverable
    fine MF34         MF34 grouped   cross term in the MF34 a0 blocks

Everything downstream of the evaluation -- chi2 scoring, library comparison --
is post-processing and lives outside this script.

Run:
    KIKA_STOP_AFTER_NOMINAL_FITS=0 KIKA_OUTPUT_DIR=<dir> python exfor_to_endf_research.py

Per-point sigma_stat and per-experiment sigma_sys come from
``scripts/uncertainty_manifest.py``, applied to each dataset as the EXFOR cache
is built. Where a manifest entry declares ``derive_stat_only``, the named column
is a TOTAL that already contains the correlated systematic, so the resolver
returns sigma_stat = sqrt(max(sigma_total^2 - sigma_sys^2, 0)) per point and the
per-experiment MC factor reproduces sigma_total^2 without double counting.
"""
from __future__ import annotations

import os
import sys

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

_kika_path = Path(__file__).parent.parent
if str(_kika_path) not in sys.path:
    sys.path.insert(0, str(_kika_path))

from kika.endf.read_endf import read_endf
from kika.endf.classes.mf4.polynomial import MF4MTLegendre
from kika.endf.classes.mf4.mixed import MF4MTMixed

from kika.exfor import read_all_exfor

from kika.endf.writers import create_mf34_from_covariance, write_mf34_to_file, merge_mf34
from kika.endf.writers import (
    create_mf33_from_covariance,
    merge_mf33_covariance_into_host,
    write_mf33_to_file,
)

from scripts.exfor_utils import (
    DualLogger,
    _get_logger,
    _set_logger,
    _format_condensed_experiments,
    EnergyBinInfo,
    SamplingResult,
    KernelDiagnostics,
    KWDiagnostics,
    compute_kw_diagnostics,
    compute_kw_reliability_alpha,
    compute_energy_bins_with_tof_resolution,
    build_union_energy_grid,
    remap_samples_to_endf_indices,
    build_exfor_cache_from_objects,
    filter_exfor_with_energy_bin,
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
    precompute_overlap_weights,
    run_mc_with_kernel_weights,
    stack_samples_to_matrix,
    build_mf33_channel,
    contiguous_grid_from_bins,
    recentre_relative_covariance,
    write_nominal_endf,
    write_average_endf,
    write_endf_samples_batch,
    write_endf_sample,
    _write_sample_wrapper,
)

from kika.sampling.endf_perturbation import _process_njoy_for_sample, perturb_ENDF_files
from kika.sampling.reprocess_endf_to_ace import _ace_file_exists
from kika.sampling.utils import _set_logger as _set_kika_logger

from scripts.multigroup_collapse import (
    perform_adaptive_multigroup_collapse,
    try_merge_adjacent_multigroups,
    MultigroupResult,
)

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


def _env_flag(name: str, default: bool) -> bool:
    """Boolean from the environment: 1/0, true/false, yes/no, on/off (any case).

    Unset gives `default`. An unrecognised value raises rather than falling back
    silently — a full run costs ~5 h and a misread flag invalidates all of it.
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


# =============================================================================
# WHAT THIS RUN PRODUCES
# =============================================================================
# The switches that decide which artefacts land in OUTPUT_DIR. Everything below
# this block tunes HOW they are built; this block decides WHETHER they are.
#
# The full deliverable chain is:
#   nominal ENDF (fine MF34 + MF33)
#     -> _mg ENDF        (MF34 collapsed to the adaptive group grid)
#       -> _a0cross ENDF (the same _mg tape with the MF33<->MF34 cross term
#                         written into MF34's a0 blocks)   <-- THE DELIVERABLE
# Each step consumes the previous one, so turning an earlier one off turns off
# everything downstream. `_preflight_products()` enforces that at startup.

GENERATE_NOMINAL_ENDF = True             # Fine-grid tape: MF4 central + fine MF34 (+ MF33 if enabled)
GENERATE_MULTIGROUP_COVARIANCE = True    # Collapse MF34 onto an adaptive group grid
GENERATE_MF3_MF33 = 1                    # 0/1. Write the elastic magnitude covariance (MF33 MT2).
                                         # MF3 itself is NOT rewritten: the central stays at the host.
GENERATE_CROSS_TERM_ENDF = True          # Write the final _a0cross tape (needs all three above)
MF34_COVARIANCE_TYPE = "both"            # Which MF34 to write: "fine" | "multigroup" | "both"
MF33_MG_REPRESENTATION = "fine"          # MF33 in the _mg tape: "fine" (full MF4 grid) | "multigroup".
                                         # Fine is shipped: grouping MF33 is irreversible and damps
                                         # the peak elastic sigma; MF34 is what makes the file large.

GENERATE_MC_MEAN_ENDF = False            # Extra tape built from the MC mean instead of the fit
GENERATE_FITTING_SAMPLES = False         # Per-sample ENDFs from the MC draws -> endf_direct/
GENERATE_FITTING_ACE = False             # ACE for those samples (needs NJOY)
GENERATE_MF34_SAMPLES = False            # Pipeline B: perturbed ENDFs drawn from MF34 -> endf/
GENERATE_MF34_ACE = False                # ACE for the Pipeline B samples

STOP_AFTER_NOMINAL_FITS = _env_flag("KIKA_STOP_AFTER_NOMINAL_FITS", True)
                                         # Exit after the nominal fits (~2 min) instead of running the
                                         # MC (~5 h). Writes nominal_fits.parquet and nothing else, so
                                         # no covariance, no ENDF, nothing that can be shipped.
                                         # Set KIKA_STOP_AFTER_NOMINAL_FITS=0 for a full run.

# --- CROSS TERM (MF33 <-> MF34) -------------------------------------------- #
# Cov(c0, a_l) shipped inside MF34's (L=0, L1) blocks, (0,0) self-block null.
# Built by scripts/build_group_cross.py from this run's own MC replicas, so it
# is Cauchy-Schwarz-compatible with the marginals it ships beside. It cannot be
# a sidecar and it cannot be bolted onto a foreign MF34.
COMPUTE_MF33_MF34_CROSS = True           # Measure Cov(c0, a_l) and save the .npy sidecars.
                                         # Diagnostic; the shipped tape is built from the replicas.
CROSS_ENDF_SUFFIX = "_a0cross"           # Filename suffix for the cross tape
CROSS_MAG_GRID = "fine"                  # Magnitude axis: "fine" (the analysis mesh, shipped) or
                                         # "group" (the adaptive MF33 grid). Only "fine" makes the
                                         # fold a congruence, so only "fine" certifies PSD.
CROSS_NULL_FILL = "zero"                 # What MF34 carries where the grouped nominal a_l is exactly
                                         # zero: "zero" (dead parameter, no covariance) or "ship"
                                         # (keep the host's declared value). "ship" reproduced the
                                         # run-89 failure; "fine" requires "zero".

# --- SIDECARS THE CROSS TERM NEEDS ----------------------------------------- #
# Not optional when GENERATE_CROSS_TERM_ENDF is on: the cross build reads all
# three back off disk to pair c0 with a_l replica by replica.
SAVE_NOMINAL_FITS = True                 # nominal_fits.parquet — per-bin c_0..c_L, tau, IC weights
SAVE_TMC_PARQUET = True                  # legendre_samples_tmc.parquet — the a_l replicas (~690 MB)
SAVE_MF33_C0_SAMPLES = True              # mf33_c0_samples.parquet — the c0 replicas (~300 MB)

# --- OTHER DIAGNOSTIC OUTPUT ----------------------------------------------- #
SAVE_COVARIANCE_FILES = False            # Fine + multigroup .npy (cov, mean, boundaries)
SAVE_CORRELATION_MATRICES = False        # Correlation beside covariance (needs the line above)
SAVE_RAW_KW_PARQUET = False              # legendre_samples_raw_kw.parquet — Pass-1 marginals
SAVE_MULTIGROUP_DIAGNOSTICS_CSV = True   # multigroup_boundary_decisions.csv
SAVE_MF33_MULTIGROUP_DIAGNOSTICS_CSV = True   # mf33_boundary_decisions.csv


# =============================================================================
# PATHS & I/O
# =============================================================================
ENDF_FILE = "/share_snc/snc/JuanMonleon/jeff40_with_MF4_from_jeff33/26-Fe-56g.txt"
MF34_SOURCE_FILE = None                          # Separate MF34 source (None = use ENDF_FILE)
EXFOR_DIRECTORY = "/share_snc/snc/JuanMonleon/EXFOR/data_v1/"
EXFOR_DB_PATH = '/share_snc/snc/JuanMonleon/EXFOR/x4_iron_angular.db'
TOF_PARAMETERS_FILE = "/share_snc/snc/JuanMonleon/EXFOR/exfor_tof_parameters.json"

# A full run writes ~1.7 GB (843 MB fine ENDF + 690 MB TMC parquet + 300 MB c0
# parquet + ~350 MB cross tape). The share sits near 91 % use — point
# KIKA_OUTPUT_DIR at /SCRATCH for production runs and copy back afterwards.
OUTPUT_DIR = os.environ.get(
    "KIKA_OUTPUT_DIR",
    "/share_snc/snc/JuanMonleon/ENDF_samples/NEW_FIT_RESEARCH/",
)

# =============================================================================
# DATA SOURCE
# =============================================================================
EXFOR_SOURCE = "database"                        # "json" | "database" | "auto" | "both"
TARGET_ZAIDS = [26056, 26000]                    # Fe-56 + natural iron
TARGET_PROJECTILE = "N"
SUPPLEMENTARY_JSON_FILES = [                     # Loaded alongside the main source
    '/share_snc/snc/JuanMonleon/EXFOR/data_v1/27673002.json',
]
EXCLUDE_EXPERIMENTS = ["32246002", "400750022"]  # Tostkii 1957; Morozov 1972 pointer 2
                                                 # (SPA-fitted sister of 400750021, double-counts it)

# =============================================================================
# ENERGY RANGE & PHYSICS
# =============================================================================
ENERGY_MIN_MEV = 0.847
ENERGY_MAX_MEV = 4
MT_NUMBER = 2                                    # 2 = elastic scattering
M_PROJ_U = 1.008665                              # Neutron mass, u
M_TARG_U = 55.93494                              # Fe-56 mass, u

# =============================================================================
# LEGENDRE FITTING
# =============================================================================
MAX_LEGENDRE_DEGREE = 6                          # Max Legendre order (hard cap 8)
SELECT_DEGREE = "aic"                            # "aic" | "aicc" | "bic" | None (always use max).
                                                 # AIC (penalty 2k); AICc over-penalises high orders
                                                 # in low-n bins. The parquet column is named `aicc`
                                                 # for legacy reasons whatever this says.
RIDGE_LAMBDA = 1e-4
RIDGE_POWER = 4                                  # Ridge penalty scales as l**RIDGE_POWER
DF_METHOD = "hat"                                # Degrees of freedom: "hat" | "naive"
MAX_SAMPLE_ORDER = 6                             # Publish covariance for l = 1..this only
FREEZE_C0 = True                                 # MC refits a_l = c_l/c0 with c0 pinned at the nominal
USE_GLS_KERNEL = True                            # Block-correlated GLS (D + u u^T + v v^T) for the IC
                                                 # scan, the nominal fit and the post-tau rescan
TAU_REFIT_USE_GLS = True                         # Also use it inside the tau-IRLS refit. False (the
                                                 # library default) ships a WLS central whose degree
                                                 # was chosen by GLS scores.
RERUN_AICC_POST_TAU = True                       # Rescore degrees against tau-inflated sigma before MC

# =============================================================================
# UNCERTAINTY & DISCREPANCY
# =============================================================================
USE_BAND_DISCREPANCY = True                      # Per-band (F/M/B) tau instead of a global Birge factor
MIN_POINTS_PER_BAND = 4                          # Min points to estimate s_b for a band
MAX_BAND_SCALE_FACTOR = 5.0                      # Max multiplicative scale per band
TAU_SMOOTHING_WINDOW = 1                         # Moving-median window for s_b(E); 1 = off
TAU_PRIOR_FLOOR = True                           # Floor tau from well-supported bins
TAU_PRIOR_NEFF_THRESHOLD = 5.0                   # Min N_eff to count as well-supported
TAU_PRIOR_PERCENTILE = 50                        # Percentile of those taus used as the floor
TAU_IRLS_MAX_ITERS = 20
TAU_IRLS_TOL = 1e-2                              # Converge when max |delta tau_b| < tol
TAU_IRLS_DAMPING = 0.5                           # Half-step in log-tau; breaks MAD limit cycles
BAND_SCALE_METHOD = "mad"                        # "mad" | "rms" | "hybrid" (= max of the two)
SIGMA_SYS_AWARE_FIT = True                       # Fit weights use sigma_total^2 = stat^2 + sys^2, and
                                                 # tau is measured against it, so tau only absorbs
                                                 # scatter beyond what both predict
RESCALE_UNC_BY_CHI2 = True                       # Birge scaling when band discrepancy is off
ALLOW_SHRINK_UNC = False                         # Allow uncertainties to shrink at chi2_red < 1
NORM_COMMON_MODE_SIGMA = 0.0                     # ONE global normalization factor shared by every
                                                 # experiment (a common monitor lineage). 0 = off.
NORM_SYSTEMATIC_SIGMA = 0.05                     # Per-experiment normalization fallback; the manifest's
                                                 # own sigma_sys supersedes it wherever present
NORM_DIST = "lognormal"                          # "lognormal" (always positive) | "normal"
MIN_STAT_RELATIVE_UNCERTAINTY = 1e-4             # Guard against literal-zero sigma_stat (infinite GLS
                                                 # weights). Set below the smallest credible reported
                                                 # value so real ones pass through untouched. The
                                                 # physical floor lives in uncertainty_manifest.py.
MIN_RELATIVE_UNCERTAINTY = MIN_STAT_RELATIVE_UNCERTAINTY  # backwards-compat alias
UNCERTAINTY_FLOOR_STRATEGY = 'bin_median'        # 'fixed' | 'bin_median' | 'band_median'

# =============================================================================
# MODEL AVERAGING AND THE PER-BIN MIXTURE
# =============================================================================
# The fit scores candidate Legendre degrees by AIC. Winner-take-all ships the
# winner's coefficients and declares exactly zero uncertainty above its degree.
# The mixture replaces that with the law of total covariance over the IC weights,
#   V_mix = sum_L w_L V_L + sum_L w_L (mu_L - mu_bar)(mu_L - mu_bar)^T,
# so every candidate contributes to every order and the between term (model-
# selection uncertainty) enters the file. Per-bin blocks only; cross-bin
# correlations still come from the Pass-1 MC.
USE_MODEL_AVERAGING = True                       # Feeds the MC degree draws. NOTE: does not by itself
                                                 # average the central value — SHIP_MIXTURE_MEAN does.
MIN_DEGREE_FOR_AVERAGING = 1
USE_DEGREE_SAMPLING_IN_MC = True                 # Draw the degree per MC sample from the IC weights
DEGREE_WEIGHT_FLOOR = float(os.environ.get("KIKA_DEGREE_WEIGHT_FLOOR", "0.01"))
                                                 # Min IC weight for a candidate to survive. 0.01 is
                                                 # v2's value and is what every shipped run (84-88, and
                                                 # so the 91 deliverable) used. 0.0 keeps every feasible
                                                 # candidate and is the open research option -- it is
                                                 # NOT inert, it feeds the MC degree draws and hence
                                                 # the covariance.
MC_CAP_FROM_SUPPORT_ONLY = _env_flag("KIKA_MC_CAP_FROM_SUPPORT_ONLY", True)
                                                 # True: the MC order cap comes from angular support.
                                                 # False: v2's min(winner degree, support).
WRITE_AVERAGING_DIAGNOSTICS = True               # Extra averaged-central columns in nominal_fits.parquet
USE_MIXTURE_COVARIANCE = _env_flag("KIKA_USE_MIXTURE_COVARIANCE", True)
SHIP_MIXTURE_MEAN = _env_flag("KIKA_SHIP_MIXTURE_MEAN", True)
                                                 # MF4 central = the mixture mean, so MF4 and MF34
                                                 # describe one distribution. The winner's coefficients
                                                 # survive in nominal_fits.parquet as win_c_0..win_c_6.
MIXTURE_MIN_SAMPLES_PER_MODEL = int(os.environ.get(
    "KIKA_MIXTURE_MIN_SAMPLES_PER_MODEL", "500"))
                                                 # Non-singularity floor on the per-candidate batch,
                                                 # not a precision one. 0 = pure proportional allocation.
MIXTURE_Q_MASK_THRESHOLD = float(os.environ.get(
    "KIKA_MIXTURE_Q_MASK_THRESHOLD", "0.01"))
                                                 # An order is a valid parameter when its inclusion
                                                 # probability exceeds this. Deliberately low: the
                                                 # near-zero guard already drops empty parameters, and
                                                 # two competing mechanisms would be unattributable.
MIXTURE_SPLICE_WITHIN_BIN_CORR = _env_flag(
    "KIKA_MIXTURE_SPLICE_WITHIN_BIN_CORR", False)
                                                 # Force the mixture's within-bin CORRELATIONS into the
                                                 # Pass-1 scaffold as well as its variances. False:
                                                 # splicing full-rank blocks into a rank-limited
                                                 # estimate makes it badly indefinite, and repairing
                                                 # that much negative mass does not preserve the
                                                 # spliced blocks. The plain congruence is PSD by
                                                 # construction. The cost is that within-bin cross-order
                                                 # correlations above the winner's degree ship as ~0.

# _mc_one_bin switches on the PRESENCE of this object, not on a flag read inside
# the worker, so a worker that never sees the config cannot take the new branch.
_MIXTURE_CFG = (
    {'min_samples_per_model': MIXTURE_MIN_SAMPLES_PER_MODEL}
    if USE_MIXTURE_COVARIANCE else None
)

# =============================================================================
# ENERGY BINNING & CORRELATION
# =============================================================================
NORMALIZE_BY_N_POINTS = True                     # Study-level GLS-ESS weighting
BAND_AWARE_ESS = False                           # Kish ESS from per-band counts
MAX_EXP_WEIGHT_FRAC_BIN = 0.8                    # Cap on one experiment's weight share in a bin
ANGULAR_QUALITY_GATE = True                      # Expand/interpolate bins with thin angular coverage.
                                                 # A no-op on this database — every bin already passes.
MIN_ANGULAR_POINTS = 4
MIN_BANDS_COVERED = 3                            # Need data in all three of F/M/B
MAX_BIN_EXPANSION = 3                            # Max expansion steps when the gate fires
MEMBERSHIP_K_SIGMA = 0.0                         # Widen WHICH datasets may constrain a bin to
                                                 # +- k*sigma_E. 0.0 = hard bin edges. Membership only:
                                                 # the selected point still carries weight 1.0.
ENERGY_GRID_SOURCE = "union"                     # "endf" (the MF4 grid) | "union" (from the subentries)
UNION_GRID_SUBENTRIES = [                        # (subentry, min_MeV, max_MeV)
    ("10571002", 0.847, 2.5),                    # Kinney
    ("23365005", 2.5, 4),                        # Pirovano
]
CORRELATION_METHOD = "kernel_weight_mc"          # "gaussian" | "kernel_weight_mc" | "hybrid"
KW_MC_TWO_PASS = True                            # Per-bin variance from Pass 2, correlations from Pass 1
KW_MC_INJECT = False                             # False: congruence Cov = D * Corr_pass1 * D, PSD by
                                                 # construction. True: legacy splice + Higham repair.
KW_MC_MIN_WEIGHT = 1e-3                          # Overlap-weight threshold
KW_MIN_POINTS_REF = None                         # Quality-penalty threshold (set to max_order+1 at runtime)
DELTA_T_NS = 5.0                                 # Default TOF time resolution
FLIGHT_PATH_M = 27.037                           # Default flight path
DELTA_T_IS_FWHM = True                           # Treat DELTA_T_NS and the per-subentry values in
                                                 # TOF_PARAMETERS_FILE as FWHM, so sigma_E = FWHM/2.3548.
                                                 # CHANGES RESULTS: sigma_E sets how far each point
                                                 # spreads across bins, hence n_eff, tau, degree
                                                 # selection and every covariance. Runs <= 81 used False.
N_SIGMA_CUTOFF = 3.0                             # Gaussian kernel cutoff, +- n sigma

# =============================================================================
# COVARIANCE PIPELINE
# =============================================================================
APPLY_COVARIANCE_CAP = False                     # Global relative-std cap
MAX_RELATIVE_STD_CAP = 1.0
REGULARIZE_NEAR_ZERO_REL_UNC = True              # Tame relative sigma where the mean passes through
                                                 # zero. With the cap and post-processing both off this
                                                 # is the ONLY active guard against rel-sigma blow-up,
                                                 # and the mixture makes the exposure worse by
                                                 # construction (a_l glides through zero).
NEAR_ZERO_SNR_THRESHOLD = 1.0                    # Flag when |mean|/sigma_abs falls below this
NEAR_ZERO_N_NEIGHBORS = 3                        # Valid neighbours sought each side
APPLY_BETWEEN_EXP_FLOOR = False                  # Between-experiment scatter floor (redundant with tau)
APPLY_POSITIVITY_PROJECTION = True               # Project MC samples onto non-negative distributions
                                                 # (min ||c*-c||^2 s.t. sum a_l P_l(mu) >= 0, SLSQP).
                                                 # Nothing is discarded; a passing sample is untouched.
POSITIVITY_CHECK_POINTS = 101                    # mu points tested in [-1, 1]

# --- Post-processing smoothing (all inert while APPLY_COV_POSTPROCESSING=False)
APPLY_COV_POSTPROCESSING = False
SMOOTH_MIN_REL_STD = 0.005                       # Below this an entry counts as absent
SMOOTH_DIP_FRACTION = 0.50                       # Flag dips under fraction*median (None = off)
SMOOTH_SPIKE_FACTOR = 3.0                        # Flag spikes over factor*median (None = off)
SMOOTH_DIP_N_NEIGHBORS = 3
SMOOTH_MEDIAN_FILL_THRESHOLD = 0.50              # Above this absent fraction, use a flat median fill
MG_SMOOTH_SPIKE_FACTOR = 2.0                     # Tighter spike threshold at multigroup level
MG_SMOOTH_DIP_N_NEIGHBORS = 5
SMOOTH_DIAGONAL_WINDOW = 0                       # Gaussian kernel window (0 = off, >= 3 to enable)
ORDER_REL_STD_CAPS = {1: 0.50, 2: 0.50, 3: 0.40, 4: 0.35, 5: 0.25, 6: 0.20}
FORWARD_FILL_REL_STD_ENABLED = False             # Propagate the last valid rel_std into absent bins

# =============================================================================
# MULTIGROUP COLLAPSE (the MF34 shape grid)
# =============================================================================
# Collapse the REPLICAS, then rescale in group space as one congruence — that is
# what keeps the joint PSD. Costs +1.56 % of chi2 on `all`, measured
# single-variable. Irreversible: a consumer can regroup a fine file, nobody can
# ungroup a collapsed one.
MULTIGROUP_RHO_MIN = 0.85                        # Min l=1 adjacent correlation to merge two bins
MULTIGROUP_SIGMA_RATIO_MAX = 5.0                 # Max running max(sigma_l1)/min(sigma_l1) in a group.
                                                 # Stops heterogeneous merges forcing the percentile
                                                 # compensation to over-inflate. None disables.
MULTIGROUP_USE_RAW_MC_CORR = True                # Feed the collapse raw KW correlations + Pass-2 std,
                                                 # bypassing the inject + Higham path
MULTIGROUP_CORRELATION_THRESHOLD = 0             # Hard-zero |rho| below this. 0 = off, "auto" =
                                                 # 1/sqrt(N_SAMPLES), or an explicit float.
MULTIGROUP_VARIANCE_PCT_MIN = 67                 # Base percentile for homogeneous groups
MULTIGROUP_VARIANCE_PCT_MAX = 85                 # Max percentile for heterogeneous groups
MULTIGROUP_VARIANCE_RATIO_REF = 5.0              # Sigma ratio at which the percentile saturates
MULTIGROUP_REGROUP_AFTER_SMOOTH = False          # Second regrouping pass after smoothing
USE_ORIGINAL_MF34_GRID = False                   # Force the grid from the host MF34
MERGE_ORIGINAL_MF34 = True                       # Merge our MF34 with the host's over its full range

# =============================================================================
# MF33 ELASTIC MAGNITUDE CHANNEL
# =============================================================================
# The relative MF33 is C_abs / sigma_host^2, so it needs a host central per bin.
# Two things make that non-trivial:
#   1. The fitted c0 is what a detector with sigma_E = 4-41 keV measured, not a
#      box average over the bin — so the denominator is folded through the same
#      kernel.
#   2. File 3 carries only the smooth background inside a resolved resonance
#      range, and MF3 MT2 is identically zero below 850 keV while this grid
#      starts at 846.8 keV. The denominator therefore comes from the RECONR
#      PENDF, not File 3, which costs one cached NJOY run.
MF33_PENDF_TOLERANCE = 0.001                     # RECONR linearization tolerance
MF33_PENDF_CACHE_DIR = None                      # None -> kika's temp cache, keyed on ENDF sha256 +
                                                 # tolerance, so repeat runs are free
MF33_MULTIGROUP_RHO_MIN = 0.85                   # MF33's own adaptive grid — independent of MF34's,
MF33_MULTIGROUP_SIGMA_RATIO_MAX = 5.0            # because the two channels have different correlation
                                                 # lengths and ENDF lets each section carry its own grid
MF33_REBUILD_MT1 = True                          # Rebuild MT1 over the analysis window as the sandwich
                                                 # over the partials with cross terms zeroed, so the
                                                 # file's total stops contradicting its own elastic

# =============================================================================
# MONTE CARLO
# =============================================================================
N_SAMPLES = int(os.environ.get("KIKA_N_SAMPLES", "10000"))
                                                 # A low value is NOT a valid evaluation — the
                                                 # covariance is badly under-sampled. It only proves
                                                 # the chain runs. Anything published uses the default.
BASE_SEED = 42

# =============================================================================
# PIPELINE B: SAMPLING FROM MF34
# =============================================================================
SAMPLING_RESOLUTION = "multigroup"               # "fine" | "multigroup"
SAMPLING_SPACE = "linear"                        # "linear" | "log"
SAMPLING_DECOMPOSITION = "svd"                   # "svd" | "cholesky" | "eigen" | "pca"
SAMPLING_METHOD = "random"                       # "sobol" | "lhs" | "random"

# =============================================================================
# ACE / NJOY
# =============================================================================
ACE_TEMPERATURES = [293.6]                       # Kelvin
ACE_NJOY_EXE = "/soft_snc/NJOY/2016.78/bin/njoy"
ACE_LIBRARY_NAME = "jeff40"
ACE_NJOY_VERSION = "NJOY 2016.78"
ACE_XSDIR_FILE = "/share_snc/snc/JuanMonleon/xsdir_MCNPy/xsdir40-irdff2"
ACE_SKIP_EXISTING = False

# =============================================================================
# RUNTIME
# =============================================================================
N_PROCS = 24                                     # Keep in step with --cpus-per-task in the sbatch
                                                 # runner or the pool oversubscribes the allocation
N_EFF_WARNING_THRESHOLD = 5.0                    # Warn when the effective sample size falls below this
VERBOSE_DIAGNOSTICS = True                       # Per-order percentile stats at every stage


def _preflight_products():
    """Refuse a run whose product switches cannot produce what they ask for.

    Runs before any work. The chain nominal -> _mg -> _a0cross means an earlier
    switch being off silently starves a later one, and the failure would
    otherwise surface ~5 h in, at the last step.
    """
    problems = []
    if GENERATE_CROSS_TERM_ENDF:
        if STOP_AFTER_NOMINAL_FITS:
            problems.append(
                "GENERATE_CROSS_TERM_ENDF needs the MC; set "
                "KIKA_STOP_AFTER_NOMINAL_FITS=0")
        if not GENERATE_NOMINAL_ENDF:
            problems.append("GENERATE_CROSS_TERM_ENDF needs GENERATE_NOMINAL_ENDF")
        if not GENERATE_MULTIGROUP_COVARIANCE:
            problems.append(
                "GENERATE_CROSS_TERM_ENDF rewrites the _mg tape; it needs "
                "GENERATE_MULTIGROUP_COVARIANCE")
        if MF34_COVARIANCE_TYPE not in ("multigroup", "both"):
            problems.append(
                f"GENERATE_CROSS_TERM_ENDF needs a multigroup MF34; "
                f"MF34_COVARIANCE_TYPE is {MF34_COVARIANCE_TYPE!r}")
        if not GENERATE_MF3_MF33:
            problems.append("GENERATE_CROSS_TERM_ENDF needs GENERATE_MF3_MF33")
        for flag, name, why in (
            (SAVE_NOMINAL_FITS, "SAVE_NOMINAL_FITS", "the grouped nominal a_l"),
            (SAVE_TMC_PARQUET, "SAVE_TMC_PARQUET", "the a_l replicas"),
            (SAVE_MF33_C0_SAMPLES, "SAVE_MF33_C0_SAMPLES", "the c0 replicas"),
        ):
            if not flag:
                problems.append(
                    f"GENERATE_CROSS_TERM_ENDF reads {why} back from disk; "
                    f"it needs {name}")
        if CROSS_MAG_GRID not in ("fine", "group"):
            problems.append(f"CROSS_MAG_GRID must be 'fine' or 'group', not {CROSS_MAG_GRID!r}")
        if CROSS_NULL_FILL not in ("zero", "ship"):
            problems.append(f"CROSS_NULL_FILL must be 'zero' or 'ship', not {CROSS_NULL_FILL!r}")
        if CROSS_MAG_GRID == "fine" and CROSS_NULL_FILL != "zero":
            problems.append("CROSS_MAG_GRID='fine' requires CROSS_NULL_FILL='zero'")
    if problems:
        raise SystemExit(
            "Product configuration cannot produce what it asks for:\n  - "
            + "\n  - ".join(problems))


# =============================================================================
# END OF CONFIGURATION
# =============================================================================


_logger = None


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

    manifest_path: Optional[str] = None
    manifest_sha256: Optional[str] = None
    manifest_path_reachable: Optional[bool] = None
    try:
        from scripts.uncertainty_manifest import manifest_path as _resolve_manifest_path
        manifest_path = str(_resolve_manifest_path())
    except Exception:
        try:
            from uncertainty_manifest import manifest_path as _resolve_manifest_path
            manifest_path = str(_resolve_manifest_path())
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
    """Per-bin and cross-energy Cov(c0, a_l) over the shared Pass-2 MC replicas.

    The magnitude and shape channels are one Monte Carlo, not two: a single
    perturbed-data matrix is built per batch, and row i becomes both replica i's
    a_l and replica i's c0. So `all_samples_perbin[s]` and `c0_samples_perbin[s]`
    are the same replica and their covariance across s measures the correlation
    the shared data induces. Freezing c0 inside the shape refit does not break
    that -- a_l does not depend on the c0 drawn, but both respond to the same
    perturbed data.

    Returns ABSOLUTE covariance and correlation, never the relative form: that
    would divide by a_l_nom, which passes through zero.

    `cov_full` (n_bins, n_bins, max_order) is the cross-ENERGY block and is not
    optional. Shipping only the within-bin entries against complete MF33/MF34
    diagonals is not a valid covariance -- zeroing the cross-energy entries makes
    the joint non-PSD in every energy window, and the discarded entries are the
    same size as the kept ones and ~159x more numerous. It is NOT symmetric:
    row = magnitude bin, column = shape bin. Do not transpose it into place.

    `cov_full` uses only replicas complete in BOTH channels across ALL bins
    (reported as `n_common`). A pairwise-complete covariance carries no PSD
    guarantee. The per-bin `cov`/`rho` keep their own per-bin intersection.

    Parameters
    ----------
    all_samples_perbin : {s_idx: {energy_idx: a_l array}}
        Pass-2 shape draws, post freeze/positivity/normalise.
    c0_samples_perbin : {s_idx: {energy_idx: float}}
        Pass-2 magnitude draws, same replica index.
    energy_indices : sequence of int
        Bin order for the returned rows.
    max_order : int
        Legendre orders 1..max_order.
    min_samples : int
        Bins with fewer paired replicas return NaN.

    Returns
    -------
    dict with 'cov', 'rho' (n_bins, max_order), 'std_c0', 'std_a', 'n_pairs',
    and when `compute_full` also 'cov_full' (n_bins, n_bins, max_order) and the
    scalar 'n_common'.
    """
    energy_indices = list(energy_indices)
    n_bins = len(energy_indices)
    cov = np.full((n_bins, max_order), np.nan, dtype=float)
    rho = np.full((n_bins, max_order), np.nan, dtype=float)
    std_a = np.full((n_bins, max_order), np.nan, dtype=float)
    std_c0 = np.full(n_bins, np.nan, dtype=float)
    n_pairs = np.zeros(n_bins, dtype=int)

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
        cov[row] = cc @ Ac / (n - 1)
        s_c = float(np.sqrt(cc @ cc / (n - 1)))
        s_a = np.sqrt(np.einsum("ij,ij->j", Ac, Ac) / (n - 1))
        std_c0[row] = s_c
        std_a[row] = s_a
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
    record_c0 = bool(args[28]) if len(args) > 28 else False
    mixture_cfg = args[29] if len(args) > 29 else None
    mixture_by_degree: dict = {}
    c0_by_sample: dict = {}

    energy_idx = nr_energy_idx

    if nr_interpolated:
        endf_coeffs = endf_normalize_legendre_coeffs(nr_nominal_coeffs, include_a0=False)
        results = {s_idx: endf_coeffs for s_idx in range(n_samples)}
        if record_c0 and len(nr_nominal_coeffs) > 0:
            c0_by_sample = {s_idx: float(nr_nominal_coeffs[0]) for s_idx in range(n_samples)}
        return (energy_idx, True, results, True, None, c0_by_sample, {})

    bin_seed = base_seed + energy_idx
    rng = np.random.default_rng(bin_seed)
    mc_weights = nr_mc_weights

    if frozen_tau_info and use_band_discrepancy:
        from scripts.resample_AD import sigma_eff_from_tau
        mu_arr = nr_mc_df['mu'].to_numpy()
        sigma_arr = nr_mc_df['unc'].to_numpy()
        inflated_unc = sigma_eff_from_tau(mu_arr, sigma_arr, frozen_tau_info)
        nr_mc_df = nr_mc_df.copy()
        nr_mc_df['unc'] = inflated_unc
        use_band_discrepancy = False

    use_degree_sampling = (
        use_degree_sampling_in_mc and
        nr_degree_weights is not None and
        len(nr_degree_weights) > 1
    )

    effective_sample_order = max_sample_order
    if mc_order_cap is not None:
        if effective_sample_order is not None:
            effective_sample_order = min(effective_sample_order, mc_order_cap)
        else:
            effective_sample_order = mc_order_cap

    sys_unc_col_arg = (
        '_sigma_sys_abs' if '_sigma_sys_abs' in nr_mc_df.columns else None
    )

    results = {}
    n_positivity_projected = 0
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
                sampled_degrees = rng.choice(degrees, size=n_samples, p=probs)
                degree_groups = {}
                for s_idx, d in enumerate(sampled_degrees):
                    degree_groups.setdefault(int(d), []).append(s_idx)
                fit_counts = {deg: len(idx) for deg, idx in degree_groups.items()}
            else:
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
                _batch_a = [] if mixture_cfg is not None else None
                for local_i in range(n_batch):
                    s_idx = s_indices[local_i] if local_i < len(s_indices) else None
                    sample_coeffs = coef_df_batch.iloc[local_i].to_numpy()
                    if len(sample_coeffs) < max_degree + 1:
                        sample_coeffs = np.pad(sample_coeffs, (0, max_degree + 1 - len(sample_coeffs)))
                    if effective_sample_order is not None:
                        for l in range(effective_sample_order + 1, len(sample_coeffs)):
                            if l < len(nr_nominal_coeffs):
                                sample_coeffs[l] = nr_nominal_coeffs[l]
                    if _apply_positivity_projection:
                        if not check_angular_distribution_positivity(sample_coeffs, _positivity_check_points):
                            n_positivity_projected += 1
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
                        n_positivity_projected += 1
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
                mixture_by_degree, n_positivity_projected)

    except Exception as exc:
        endf_coeffs = endf_normalize_legendre_coeffs(nr_nominal_coeffs, include_a0=False)
        results = {s_idx: endf_coeffs for s_idx in range(n_samples)}
        c0_fallback = (
            {s_idx: float(nr_nominal_coeffs[0]) for s_idx in range(n_samples)}
            if record_c0 and len(nr_nominal_coeffs) > 0 else {}
        )
        return (energy_idx, False, results, False,
                f"{type(exc).__name__}: {exc}", c0_fallback, {})


def _log_positivity_projections(bin_results, nominal_results, n_samples, logger):
    """Report how often the positivity projection fired, and in WHICH bins.

    The projection (`project_to_positive_distribution`) silently modifies any MC
    sample whose Legendre expansion goes negative somewhere in mu. It is the
    right thing to do — a negative angular distribution is unphysical — but it
    means the covariance we publish is that of a *projected* sample set, and
    ENDF-6 MF34 has no syntax for a positivity constraint, so a consumer drawing
    a Gaussian from it is not protected the same way.

    How often that happens was never recorded, which left the size of the effect
    unknown in both directions. This logs it: the overall rate, and the bins
    where it concentrates so they can be located in energy. Roadmap §6.5.
    """
    energy_by_idx = {nr.energy_index: nr.energy_mev for nr in nominal_results}
    per_bin = {}
    total = 0
    for rec in bin_results:
        energy_idx = rec[0]
        n_proj = rec[7] if len(rec) > 7 else 0
        if n_proj:
            per_bin[energy_idx] = n_proj
            total += n_proj

    n_bins = len(bin_results)
    denom = max(n_bins * max(n_samples, 1), 1)
    if total == 0:
        logger.info(
            f"  [POSITIVITY] projection never fired "
            f"({n_bins} bins x {n_samples} samples)"
        )
        return per_bin

    logger.info(
        f"  [POSITIVITY] projection fired on {total}/{denom} samples "
        f"({100.0 * total / denom:.3f} %) in {len(per_bin)}/{n_bins} bins"
    )
    ranked = sorted(per_bin.items(), key=lambda kv: kv[1], reverse=True)
    logger.info("  [POSITIVITY] bins where it fires, worst first "
                "(energy MeV: samples projected of %d):" % n_samples)
    for energy_idx, n_proj in ranked[:40]:
        e = energy_by_idx.get(energy_idx)
        e_str = f"{e:.6f}" if e is not None else f"idx {energy_idx}"
        logger.info(f"  [POSITIVITY]    {e_str} : {n_proj} "
                    f"({100.0 * n_proj / max(n_samples, 1):.1f} %)")
    if len(ranked) > 40:
        logger.info(f"  [POSITIVITY]    ... and {len(ranked) - 40} further bins")
    return per_bin


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


def _write_cross_term_endf(mg_endf, output_path, logger):
    """Rewrite the _mg tape with the MF33<->MF34 cross term in MF34's a0 blocks.

    Reads this run's own c0 and a_l replicas back off disk, collapses them onto
    the grids the _mg tape already carries, and rescales the whole joint as one
    congruence so the shipped marginals are reproduced and the result is PSD by
    construction. The cross term cannot ship as a sidecar and cannot be attached
    to marginals it was not built with.

    Returns the path written, or None on failure.
    """
    from scripts.build_group_cross import build_cross_and_write_endf

    mg_path = Path(mg_endf)
    stem = mg_path.name
    out_name = (stem.replace("_mg.endf", f"{CROSS_ENDF_SUFFIX}_mg.endf")
                if stem.endswith("_mg.endf") else
                f"{mg_path.stem}{CROSS_ENDF_SUFFIX}{mg_path.suffix}")
    out_path = mg_path.parent / out_name

    logger.info(f"  source _mg tape : {mg_path}")
    logger.info(f"  magnitude grid  : {CROSS_MAG_GRID}")
    logger.info(f"  null fill       : {CROSS_NULL_FILL}")
    logger.info(f"  output          : {out_path}")

    import contextlib
    import io
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            build_cross_and_write_endf(
                run_dir=output_path,
                source_endf=mg_path,
                out_endf=out_path,
                mag_grid=CROSS_MAG_GRID,
                null_fill=CROSS_NULL_FILL,
                cache=Path(output_path) / ".group_cross_cache",
            )
    except Exception as e:
        for line in buf.getvalue().splitlines():
            logger.info(f"  | {line}")
        logger.error(f"[ERROR] [CROSS] cross-term ENDF failed: {e}", console=True)
        logger.error(f"  Traceback:\n{traceback.format_exc()}", console=False)
        return None

    for line in buf.getvalue().splitlines():
        logger.info(f"  | {line}")
    logger.info(f"  [INFO] [CROSS] cross-term ENDF written: {out_path}", console=True)
    return out_path


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
    interpolated: bool = False
    kernel_diagnostics: Optional[KernelDiagnostics] = None
    degree_weights: Optional[Dict[int, float]] = None
    all_degrees_info: Optional[Dict[int, Dict]] = None
    averaging_diag: Optional[Dict[str, float]] = None
    winner_coeffs: Optional[np.ndarray] = None
    mc_order_cap: Optional[int] = None
    between_exp_scatter: Optional[np.ndarray] = None
    between_exp_L_common: int = 0
    skip_reason: Optional[str] = None
    expanded_bins: int = 0
    endf_index: Optional[int] = None


AVERAGING_COLUMNS = (
    [f'avg_a_{l}' for l in range(1, 7)]
    + [f'q_a_{l}' for l in range(1, 7)]
    + [f'cond_a_{l}' for l in range(1, 7)]
    + ['n_eff_models', 'weight_floor_applied']
)

CHI2_COLUMNS = ['chi2_pp_win', 'chi2_pp_avg', 'chi2_pp_ratio']


def data_space_chi2(
    mu: np.ndarray,
    y: np.ndarray,
    sigma_eff: np.ndarray,
    weights: Optional[np.ndarray],
    a_vec: np.ndarray,
    c0: float,
) -> float:
    r"""Kernel-weighted mean squared standardized residual for one central value.

    .. math::
        \chi^2_{pp} = \frac{\sum_i w_i\,[(y_i-\hat y_i)/\sigma_{{\rm eff},i}]^2}
                            {\sum_i w_i}

    One explicit statistic, evaluated for the winner and the average with the
    SAME sigma_eff, weights and points, so only the predicted curve differs.
    The fit's own `chi2_red` uses a different quadratic form and dof convention;
    this is not comparable to it and must not be reported beside it.

    The curve is rebuilt at a shared c0 (the winner's) so the comparison isolates
    the angular shape, which is what the order treatment changes.
    """
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

    degrees = sorted(d for d in degree_weights if d in all_degrees_info)
    if not degrees:
        return nan_row

    weights = np.array([degree_weights[d] for d in degrees], dtype=float)
    if not np.isfinite(weights).any() or weights.sum() <= 0:
        return nan_row

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
    for l in range(max_degree + 1, 7):
        out[f'avg_a_{l}'] = float('nan')
        out[f'q_a_{l}'] = float('nan')
        out[f'cond_a_{l}'] = float('nan')

    out['n_eff_models'] = float(effective_n_models(weights))
    out['weight_floor_applied'] = float(weight_floor)
    return out


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
        for miss_idx in missing_indices:
            nominal_results[miss_idx].nominal_coeffs = np.array([1.0])
            nominal_results[miss_idx].frozen_degree = 0
        return nominal_results

    valid_energies = np.array([nominal_results[i].energy_mev for i in valid_indices])

    max_n_coeffs = max(len(nominal_results[i].nominal_coeffs) for i in valid_indices)

    valid_coeffs = []
    for i in valid_indices:
        coeffs = nominal_results[i].nominal_coeffs
        if len(coeffs) < max_n_coeffs:
            padded = np.zeros(max_n_coeffs, dtype=float)
            padded[:len(coeffs)] = coeffs
            valid_coeffs.append(padded)
        else:
            valid_coeffs.append(coeffs)
    valid_coeffs = np.array(valid_coeffs)

    valid_degrees = np.array([nominal_results[i].frozen_degree for i in valid_indices])

    n_interpolated = 0
    n_extrapolated = 0

    for miss_idx in missing_indices:
        miss_energy = nominal_results[miss_idx].energy_mev

        if miss_energy < valid_energies.min():
            if logger:
                logger.warning(
                    f"E={miss_energy:.4f} MeV: Below EXFOR data range [{valid_energies.min():.4f} MeV] "
                    f"- extrapolating from lowest valid bin"
                )
            nearest_idx = valid_indices[0]
            nominal_results[miss_idx].nominal_coeffs = nominal_results[nearest_idx].nominal_coeffs.copy()
            nominal_results[miss_idx].frozen_degree = nominal_results[nearest_idx].frozen_degree
            nominal_results[miss_idx].interpolated = True
            nominal_results[miss_idx].has_data = True
            n_extrapolated += 1
            continue

        if miss_energy > valid_energies.max():
            if logger:
                logger.warning(
                    f"E={miss_energy:.4f} MeV: Above EXFOR data range [{valid_energies.max():.4f} MeV] "
                    f"- extrapolating from highest valid bin"
                )
            nearest_idx = valid_indices[-1]
            nominal_results[miss_idx].nominal_coeffs = nominal_results[nearest_idx].nominal_coeffs.copy()
            nominal_results[miss_idx].frozen_degree = nominal_results[nearest_idx].frozen_degree
            nominal_results[miss_idx].interpolated = True
            nominal_results[miss_idx].has_data = True
            n_extrapolated += 1
            continue

        interp_coeffs = np.zeros(max_n_coeffs, dtype=float)
        for coeff_idx in range(max_n_coeffs):
            y_vals = valid_coeffs[:, coeff_idx]
            interp_coeffs[coeff_idx] = np.interp(miss_energy, valid_energies, y_vals)

        interp_degree = int(round(np.interp(miss_energy, valid_energies, valid_degrees.astype(float))))

        nominal_results[miss_idx].nominal_coeffs = interp_coeffs
        nominal_results[miss_idx].frozen_degree = interp_degree
        nominal_results[miss_idx].interpolated = True
        nominal_results[miss_idx].has_data = True
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
            continue

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

    sorted_experiments = sorted(
        experiment_totals.items(),
        key=lambda x: x[1]['total_points'],
        reverse=True
    )

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
    gate_expanded_1 = 0
    gate_expanded_2 = 0
    gate_failed = 0

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

        winner_coeffs = np.asarray(nominal_coeffs, dtype=float).copy()
        if SHIP_MIXTURE_MEAN and _avg_diag is not None:
            _avg_a = [_avg_diag.get(f'avg_a_{l}') for l in range(1, max_degree + 1)]
            if all(a is not None and np.isfinite(a) for a in _avg_a):
                _c0v = float(winner_coeffs[0])
                _mix_c = np.zeros(max_degree + 1, dtype=float)
                _mix_c[0] = _c0v
                for _l in range(1, max_degree + 1):
                    _mix_c[_l] = float(_avg_a[_l - 1]) * (2 * _l + 1) * _c0v
                nominal_coeffs = _mix_c

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

        support_diag = compute_angular_support_diagnostics(mu, kernel_weights, max_degree)
        mc_cap = (
            support_diag['recommended_mc_order'] if MC_CAP_FROM_SUPPORT_ONLY
            else min(frozen_degree, support_diag['recommended_mc_order'])
        )
        results[-1].mc_order_cap = mc_cap

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

            condensed_lines = _format_condensed_experiments(experiments_info)
            for line in condensed_lines:
                logger.info(line)

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
            _exp_diag = tau_info.get('exp_diag')
            if _exp_diag:
                band_names = {'F': 'Forward', 'M': 'Mid', 'B': 'Backward'}
                for bnd, entries in _exp_diag.items():
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
            logger.info("")

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

    if (tau_smoothing_window > 1 or tau_prior_floor) and use_band_discrepancy:
        for r in results:
            if not r.has_data or r.interpolated:
                continue
            mu = r.exfor_df['mu'].to_numpy()
            sigma = r.exfor_df['unc'].to_numpy()
            r.sigma_eff = sigma_eff_from_tau(mu, sigma, r.tau_info)
            r.kernel_diagnostics.n_eff = compute_n_eff(r.kernel_weights, r.sigma_eff)

    if total_floor_floored > 0 and logger:
        valid_repls = [v for v in floor_replacement_vals if not np.isnan(v)]
        median_repl = float(np.median(valid_repls)) * 100 if valid_repls else 0.0
        logger.info(f"  [Unc floor] Summary: {total_floor_floored}/{total_floor_pts} total pts "
                    f"floored across {total_floor_bins} bins, "
                    f"median replacement={median_repl:.1f}% (strategy={UNCERTAINTY_FLOOR_STRATEGY})")

    total_expanded = gate_expanded_1 + gate_expanded_2
    if angular_quality_gate and logger and (total_expanded > 0 or gate_failed > 0):
        logger.info(
            f"  [Quality gate] {total_expanded} bins expanded "
            f"({gate_expanded_1} by ±1, {gate_expanded_2} by ±2), "
            f"{gate_failed} bins failed → interpolate"
        )

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
    n_capped = {'F': 0, 'M': 0, 'B': 0}
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
        ranked = sorted(exp_stats.items(),
                        key=lambda x: len(x[1]['abs_r']), reverse=True)
        logger.info("  Experiments most frequently in capped bands:")
        for exp_id, stats in ranked[:10]:
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

    l1_scatter = [(r.energy_mev, r.between_exp_scatter[0]) for r in scatter_bins
                  if len(r.between_exp_scatter) > 0]
    if not l1_scatter:
        return

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

    exp_residuals = defaultdict(list)

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
    bias_data = []
    for exp_id, resids in exp_residuals.items():
        n = len(resids)
        mean_r = float(np.mean(resids))
        std_r = float(np.std(resids))
        bias_data.append((exp_id, n, mean_r, std_r))

    bias_data.sort(key=lambda x: abs(x[2]), reverse=True)
    logger.info(f"  {'Experiment':<12} {'N_pts':>7} {'mean_r':>8} {'std_r':>7}  note")
    for exp_id, n, mean_r, std_r in bias_data:
        threshold = 2.0 / np.sqrt(max(n, 1))
        flag = ""
        if abs(mean_r) > threshold and n >= 10:
            direction = "high" if mean_r > 0 else "low"
            flag = f"  ← systematic {direction} ({abs(mean_r)/threshold:.1f}σ)"
        logger.info(f"  {str(exp_id):<12} {n:>7} {mean_r:>+8.3f} {std_r:>7.2f}{flag}")


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

    exfor_objects = list(exfor_dict.values())

    if logger:
        logger.info(f"  Loaded {len(exfor_objects)} EXFOR datasets")

        if supplementary_json_files:
            logger.info("  Supplementary file load results:")
            for item in load_status.get('loaded', []):
                logger.info(f"    LOADED: {item['id']} ({item.get('label', 'unknown')}, {item['n_energies']} energies)")
            for item in load_status.get('skipped', []):
                logger.warning(f"    SKIPPED: {item['id']} - {item['reason']}")
            for item in load_status.get('failed', []):
                logger.warning(f"    FAILED: {item['file']} - {item['error']}")

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

    if ang_covmat.has_uniform_energy_grid():
        if ang_covmat.energy_grids:
            grid_ev = np.array(ang_covmat.energy_grids[0], dtype=float)
        else:
            if logger:
                logger.warning(f"  MF34/MT{mt_number} has no energy grids")
            return None
    else:
        union_grids = ang_covmat.compute_union_energy_grids()
        all_points = set()
        for grid in union_grids.values():
            all_points.update(np.asarray(grid, dtype=float).tolist())
        if not all_points:
            if logger:
                logger.warning(f"  MF34/MT{mt_number} union grids are empty")
            return None
        grid_ev = np.array(sorted(all_points))

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
    generate_multigroup_covariance: bool = False,
    multigroup_rho_min: float = 0.90,
    multigroup_sigma_ratio_max: Optional[float] = None,
    multigroup_variance_pct_min: float = 67.0,
    multigroup_variance_pct_max: float = 85.0,
    multigroup_variance_ratio_ref: float = 5.0,
    multigroup_regroup_after_smooth: bool = False,
    mf34_covariance_type: str = "fine",
    use_original_mf34_grid: bool = False,
    exfor_db_path: str = None,
    exfor_source: str = "auto",
    target_zaid: Union[int, List[int]] = None,
    target_projectile: str = "N",
    supplementary_json_files: List[str] = None,
    exclude_experiments: Optional[List[str]] = None,
    min_relative_uncertainty: float = 0.0,
    energy_grid_source: str = ENERGY_GRID_SOURCE,
    union_grid_subentries: Optional[List[Tuple[str, Optional[float], Optional[float]]]] = None,
    apply_covariance_cap: bool = False,
    max_relative_std_cap: float = 1.0,
    regularize_near_zero: bool = True,
    near_zero_snr_threshold: float = 1.0,
    near_zero_n_neighbors: int = 3,
    apply_between_exp_floor: bool = True,
    apply_cov_postprocessing: bool = True,
    apply_positivity_projection: bool = False,
    positivity_check_points: int = 50,
    save_correlation_matrices: bool = False,
    save_tmc_parquet: bool = True,
    save_raw_kw_parquet: bool = False,
    save_multigroup_diagnostics_csv: bool = True,
    save_nominal_fits: bool = True,
    ace_temperatures: Optional[List[float]] = None,
    ace_njoy_exe: Optional[str] = None,
    ace_library_name: Optional[str] = None,
    ace_njoy_version: str = "NJOY 2016.78",
    ace_xsdir_file: Optional[str] = None,
    ace_skip_existing: bool = False,
    mf34_source_file: Optional[str] = None,
    generate_mf34_samples: bool = False,
    generate_mf34_ace: bool = False,
    sampling_resolution: str = "fine",
    merge_original_mf34: bool = True,
    sampling_space: str = "linear",
    sampling_decomposition: str = "svd",
    sampling_method: str = "sobol",
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

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = output_path / f'exfor_to_endf_{timestamp}.log'
    _logger = DualLogger(str(log_file))
    _set_logger(_logger)

    _logger.info("EXFOR-to-ENDF Angular Distribution Sampling (v2)")
    _logger.info(f"Timestamp: {datetime.now().isoformat()}")
    _logger.info("")

    print(f"[INFO] Starting EXFOR-to-ENDF sampling (v2)")
    print(f"[INFO] Log file: {log_file}")

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

    _logger.info("#== MODEL-ORDER POLICY ====================================================")
    _logger.info("  Nominal MF4 order = AICc winner per bin.")
    _logger.info("  MC samples may draw alternate AICc-supported orders (mixture).")
    _logger.info("  MF34 covariance is published only for orders present in nominal MF4")
    _logger.info("    (n_valid = min(frozen_degree, MAX_SAMPLE_ORDER)).")
    _logger.info("  Higher sampled orders affect retained-order variance but are not published.")
    _logger.info("")

    _warning_counts = {}

    _logger.info("#== CONFIG ================================================================")
    _logger.info("")

    _logger.info("  # Paths & I/O")
    _logger.info(f"  ENDF_FILE = {endf_file}")
    _logger.info(f"  MF34_SOURCE_FILE = {mf34_source_file or '(same as ENDF_FILE)'}")
    _logger.info(f"  EXFOR_DIRECTORY = {exfor_directory}")
    _logger.info(f"  EXFOR_DB_PATH = {exfor_db_path}")
    _logger.info(f"  OUTPUT_DIR = {output_dir}")
    _logger.info(f"  TOF_PARAMETERS_FILE = {TOF_PARAMETERS_FILE}")
    _logger.info("")

    _logger.info("  # Data Source")
    _logger.info(f"  EXFOR_SOURCE = {exfor_source}")
    _logger.info(f"  TARGET_ZAIDS = {target_zaid}")
    _logger.info(f"  TARGET_PROJECTILE = {target_projectile}")
    _logger.info(f"  SUPPLEMENTARY_JSON_FILES = {supplementary_json_files}")
    _logger.info("")

    _logger.info("  # Energy Range & Physics")
    _logger.info(f"  ENERGY_MIN_MEV = {energy_min_mev}")
    _logger.info(f"  ENERGY_MAX_MEV = {energy_max_mev}")
    _logger.info(f"  MT_NUMBER = {mt_number}")
    _logger.info(f"  M_PROJ_U = {m_proj_u}")
    _logger.info(f"  M_TARG_U = {m_targ_u}")
    _logger.info("")

    _logger.info("  # Legendre Fitting")
    _logger.info(f"  MAX_LEGENDRE_DEGREE = {max_degree}")
    _logger.info(f"  SELECT_DEGREE = {select_degree if select_degree else 'None (use max)'}")
    _logger.info(f"  RIDGE_LAMBDA = {ridge_lambda}")
    _logger.info(f"  RIDGE_POWER = {RIDGE_POWER}")
    _logger.info(f"  DF_METHOD = {DF_METHOD}")
    _logger.info("")

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

    _logger.info("  # Model Averaging")
    _logger.info(f"  USE_MODEL_AVERAGING = {use_model_averaging}")
    _logger.info(f"  MIN_DEGREE_FOR_AVERAGING = {min_degree_for_averaging}")
    _logger.info(f"  USE_DEGREE_SAMPLING_IN_MC = {USE_DEGREE_SAMPLING_IN_MC}")
    _logger.info(f"  RERUN_AICC_POST_TAU = {RERUN_AICC_POST_TAU}")
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

    _any_ace = generate_fitting_ace or generate_mf34_ace
    if _any_ace:
        if ace_temperatures is None:
            ace_temperatures = [293.6]
        elif isinstance(ace_temperatures, (int, float)):
            ace_temperatures = [float(ace_temperatures)]

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

    _logger.info("  # Runtime")
    _logger.info(f"  N_PROCS = {n_procs}")
    _logger.info(f"  N_EFF_WARNING_THRESHOLD = {n_eff_warning_threshold}")
    _logger.info(f"  VERBOSE_DIAGNOSTICS = {verbose_diagnostics}")
    _logger.info("")

    _logger.info("#== END CONFIG ============================================================")
    _logger.info("")

    if not os.path.exists(endf_file):
        _logger.error(f"[ERROR] [ENDF] File not found: {endf_file}", console=True)
        return

    if not os.path.isdir(exfor_directory):
        _logger.error(f"[ERROR] [EXFOR] Directory not found: {exfor_directory}", console=True)
        return

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

        try:
            from scripts.uncertainty_manifest import load_manifest as _load_manifest
            from scripts.exfor_utils import _parse_exclusion_list, _is_experiment_excluded
            _m = _load_manifest()
            _excl_patterns = _parse_exclusion_list(exclude_experiments)
            _desync = []
            for _dsid, _entry in (_m.get('datasets') or {}).items():
                if _entry.get('flag') != 'excluded':
                    continue
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

    for bin_info in energy_bins:
        idx = int(np.argmin(np.abs(energies_ev - bin_info.energy_ev)))
        bin_info.endf_index = idx
        if idx < len(original_coeffs):
            bin_info.original_coeffs = list(original_coeffs[idx])

    _logger.info(f">> energy_bins = {len(energy_bins)}")
    _logger.info(f"#-- END STEP 3 (elapsed: {time.time() - t_step:.2f}s) -------------------------------------")
    _logger.info(f"  [INFO] [ENDF] Processing {len(energy_bins)} energy bins", console=True)

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

    log_experiments_summary(nominal_results, logger=_logger)

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
            _wc = r.winner_coeffs
            row.update({
                f'win_c_{l}': (float(_wc[l]) if _wc is not None and l < len(_wc)
                               else float('nan'))
                for l in range(max_degree + 1)
            })
            if WRITE_AVERAGING_DIAGNOSTICS:
                if r.averaging_diag is not None:
                    row.update(r.averaging_diag)
                else:
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
        _logger.info(
            "  STOP_AFTER_NOMINAL_FITS=True — stopping before MC sampling. "
            f"nominal_fits.parquet is in {output_dir}",
            console=True,
        )
        return

    t_step = time.time()
    _logger.info("")
    _logger.info("#-- STEP 5: MC sampling ----------------------------------------------------")
    _logger.info(f"  [INFO] [MC] Generating {n_samples} samples, method={CORRELATION_METHOD}")

    _prebuilt_gaussian_cov = None
    _prebuilt_mc_mean = None

    record_c0_channel = False
    mf33_rel_cov_fine = None
    mf33_c0_nom = None
    mf33_energy_grid_ev = None
    mf33_sigma_host_bin = None
    _mf33_products = None
    _mf33_pendf_path = None

    if CORRELATION_METHOD == "gaussian":
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
                False,
                _MIXTURE_CFG,
            ))

        if N_PROCS > 1:
            with Pool(N_PROCS) as pool:
                bin_results = pool.map(_mc_one_bin, bin_args_list)
        else:
            bin_results = [_mc_one_bin(a) for a in bin_args_list]

        _log_mc_bin_failures(bin_results, nominal_results, _warning_counts, _logger)
        _log_positivity_projections(bin_results, nominal_results, n_samples, _logger)
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

        _logger.info("  Building Gaussian correlation covariance from stochastic pass")
        energy_indices_for_gauss = [nr.energy_index for nr in nominal_results if nr.has_data]

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

        _gaussian_cov_full = build_gaussian_correlation_covariance(
            cov_stochastic=cov_stochastic_pass2,
            energy_bins=energy_bins,
            energy_indices=energy_indices_for_gauss,
            max_order=max_degree,
            logger=_logger,
            valid_mask=_valid_mask,
        )

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

        _prebuilt_gaussian_cov = _gaussian_cov_full
        _prebuilt_mc_mean = mc_mean_stochastic

    elif CORRELATION_METHOD in ("kernel_weight_mc", "hybrid"):
        _is_hybrid = (CORRELATION_METHOD == "hybrid")
        _logger.info("  " + "=" * 60)
        _logger.info(f"  Method: {'Hybrid KW+Gaussian blend' if _is_hybrid else 'Kernel-weight MC correlations'}")
        _logger.info(f"  Two-pass mode: {KW_MC_TWO_PASS}")
        if KW_MC_TWO_PASS:
            _mode = "inject + Higham repair (legacy)" if KW_MC_INJECT else "congruence transform (PSD by construction)"
            _logger.info(f"  Pass-2 combine: {_mode}")
        _logger.info(f"  Min overlap weight: {KW_MC_MIN_WEIGHT}")
        _logger.info("  " + "=" * 60)

        tof_params_cache = {}
        if TOF_PARAMETERS_FILE:
            try:
                tof_params_cache = load_tof_parameters_file(TOF_PARAMETERS_FILE)
                _logger.info(f"  Loaded TOF parameters for {len(tof_params_cache)} experiments")
            except FileNotFoundError:
                _logger.warning(f"[WARN] [MC] TOF parameters file not found: {TOF_PARAMETERS_FILE}")

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
            fix_c0_at_nominal=True,
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
                    record_c0_channel,
                    _MIXTURE_CFG,
                ))

            if N_PROCS > 1:
                with Pool(N_PROCS) as pool:
                    bin_results = pool.map(_mc_one_bin, bin_args_list)
            else:
                bin_results = [_mc_one_bin(a) for a in bin_args_list]

            _log_mc_bin_failures(bin_results, nominal_results, _warning_counts, _logger)
            _log_positivity_projections(bin_results, nominal_results, n_samples, _logger)
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

            energy_indices_kw = [nr.energy_index for nr in nominal_results if nr.has_data]
            nr_by_idx_kw = {nr.energy_index: nr for nr in nominal_results}

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
                    _vb = [_bin_by_idx_mf33[e] for e in energy_indices_kw]
                    mf33_energy_grid_ev = contiguous_grid_from_bins(_vb)

                    if COMPUTE_MF33_MF34_CROSS and c0_samples_perbin is not None:
                        try:
                            _xc = compute_mf33_mf34_cross(
                                all_samples_perbin, c0_samples_perbin,
                                energy_indices_kw, max_degree,
                            )
                            np.save(output_path / "mf33_mf34_cross_covariance.npy", _xc["cov"])
                            np.save(output_path / "mf33_mf34_cross_correlation.npy", _xc["rho"])
                            np.save(output_path / "mf33_mf34_cross_n_pairs.npy", _xc["n_pairs"])
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
                    _host_e_ev, _host_xs_b = _mf33_den.e_ev, _mf33_den.xs_b
                    _mf33_pendf_path = _mf33_den.pendf_path
                    _c0_host = mf33_sigma_host_bin / _MF33_FOUR_PI

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

            std_perbin = np.sqrt(np.maximum(np.diag(cov_perbin), 0.0))

            _mean_signs = np.sign(mc_mean_perbin)
            _mean_signs[_mean_signs == 0] = 1.0
            _sign_outer = np.outer(_mean_signs, _mean_signs)

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

                n_params = len(energy_indices_kw) * max_degree
                alpha_param = np.zeros(n_params)
                for ie, a in enumerate(alpha_per_energy):
                    alpha_param[ie * max_degree:(ie + 1) * max_degree] = a

                alpha_ij = np.minimum(alpha_param[:, None], alpha_param[None, :])

                g_ij = build_gaussian_relevance_matrix(
                    energy_bins=energy_bins,
                    energy_indices=energy_indices_kw,
                    max_order=max_degree,
                )
                w_gauss_ij = (1.0 - alpha_ij) * g_ij

                corr_hyb = (1.0 - w_gauss_ij) * corr_kw + w_gauss_ij * corr_gauss
                corr_hyb = (corr_hyb + corr_hyb.T) / 2.0
                np.fill_diagonal(corr_hyb, 1.0)
                corr_hyb = np.clip(corr_hyb, -1.0, 1.0)

                log_psd_diagnostics(corr_hyb, "corr_hyb (post-blend)", _logger)

                if KW_MC_INJECT:
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
                    cov_combined = corr_hyb * np.outer(std_perbin, std_perbin) * _sign_outer
                    log_psd_diagnostics(cov_combined, "cov_combined (congruence, hybrid)", _logger)
                    _diag_diff = float(np.max(np.abs(np.diag(cov_combined) - std_perbin**2)))
                    _logger.info(f"  [Congruence check, hybrid] max |diag(cov) - std_perbin^2| = {_diag_diff:.3e}")

                n_interp = int(np.sum(alpha_per_energy == 0.0))
                n_data = len(alpha_per_energy) - n_interp
                if n_data > 0:
                    data_alphas = alpha_per_energy[alpha_per_energy > 0.0]
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
            else:
                if KW_MC_INJECT:
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
                    np.fill_diagonal(corr_kw, 1.0)
                    corr_kw = np.clip(corr_kw, -1.0, 1.0)
                    if (USE_MIXTURE_COVARIANCE and _mix_blocks
                            and MIXTURE_SPLICE_WITHIN_BIN_CORR):
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
            all_samples = kw_samples
            _logger.info("  Single-pass mode: using KW MC samples directly")

        n_sampled = sum(1 for nr in nominal_results if nr.has_data and not nr.interpolated)
        n_interpolated_used = sum(1 for nr in nominal_results if nr.has_data and nr.interpolated)
        _logger.info(f"  [INFO] [MC] Complete: {n_sampled} bins with data, {n_interpolated_used} interpolated")

    else:
        raise ValueError(f"Unknown CORRELATION_METHOD: {CORRELATION_METHOD!r}. Use 'gaussian', 'kernel_weight_mc', or 'hybrid'.")

    _logger.info(f">> samples_generated = {n_samples}")
    _logger.info(f"#-- END STEP 5 (elapsed: {time.time() - t_step:.2f}s) -------------------------------------")

    cov_matrix = None
    energy_indices = [nr.energy_index for nr in nominal_results if nr.has_data]

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

    if True:
        t_step = time.time()
        _logger.info("")
        _logger.info("#-- STEP 7: Compute covariance matrix --------------------------------------")

        if _prebuilt_gaussian_cov is not None:
            _logger.info("  Using pre-built Gaussian correlation covariance")
            cov_matrix = _prebuilt_gaussian_cov
            mc_mean_params = _prebuilt_mc_mean
            param_labels = [(e_idx, l + 1) for e_idx in energy_indices for l in range(max_degree)]

            std_combined = np.sqrt(np.maximum(np.diag(cov_matrix), 0.0))
            std_combined[std_combined == 0] = 1.0
            corr_matrix = cov_matrix / np.outer(std_combined, std_combined)

            cov_abs = cov_matrix * np.outer(mc_mean_params, mc_mean_params)

            var_total = np.diag(cov_matrix)
            mask = var_total > 0
            if np.any(mask):
                _logger.info(f"  Gaussian cov variance: mean={np.mean(var_total[mask]):.2e}")
        else:
            _snr_thr = near_zero_snr_threshold if regularize_near_zero else 0.0

            _use_raw_mc_corr = (
                MULTIGROUP_USE_RAW_MC_CORR
                and KW_MC_TWO_PASS
                and CORRELATION_METHOD in ("kernel_weight_mc", "hybrid")
            )

            if _use_raw_mc_corr:
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

        n_nonfinite = int(np.sum(~np.isfinite(cov_matrix)))
        if n_nonfinite > 0:
            _logger.warning(f"[WARN] [COV] Covariance matrix has {n_nonfinite} non-finite entries -- replacing with 0")
            cov_matrix = np.where(np.isfinite(cov_matrix), cov_matrix, 0.0)
            cov_matrix = (cov_matrix + cov_matrix.T) / 2.0
            cov_abs = np.where(np.isfinite(cov_abs), cov_abs, 0.0)
            cov_abs = (cov_abs + cov_abs.T) / 2.0
            std_fix = np.sqrt(np.maximum(np.diag(cov_abs), 0.0))
            std_fix[std_fix == 0] = 1.0
            corr_matrix = cov_abs / np.outer(std_fix, std_fix)

        diag = np.diag(cov_matrix)
        diag_nonzero = diag[diag > 0]
        if len(diag_nonzero) > 0:
            min_diag = np.min(diag_nonzero)
            max_diag = np.max(diag_nonzero)
            mean_diag = np.mean(diag_nonzero)
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

        if apply_covariance_cap:
            t_step = time.time()
            _logger.info("")
            _logger.info("#-- STEP 7a: Covariance diagonal cap (Layer 1) ----------------------------")

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

            cov_abs = cov_matrix * np.outer(mc_mean_params, mc_mean_params)

            std_capped = np.sqrt(np.maximum(np.diag(cov_matrix), 0.0))
            std_capped[std_capped == 0] = 1.0
            corr_matrix = cov_matrix / np.outer(std_capped, std_capped)

        if save_covariance_files:
            np.save(output_path / "legendre_covariance.npy", cov_matrix)
            if save_correlation_matrices:
                np.save(output_path / "legendre_correlation.npy", corr_matrix)

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
                    valid_orders_fn=_mg_valid_orders,
                )

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

    if generate_fitting_ace:
        t_step = time.time()
        _logger.info("")
        _logger.info("#-- STEP 9b: ACE generation (Pipeline A) ------------------------------------")

        _set_kika_logger(_logger)

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

        valid_files = [(f, i) for i, f in enumerate(endf_sample_files) if f]

        if not valid_files:
            _logger.warning("[WARN] [ACE] No ENDF sample files found -- skipping ACE generation", console=True)
        else:
            _logger.info(f"  Processing {len(valid_files)} ENDF samples at {len(ace_temperatures)} temperature(s)")
            _logger.info(f"  Temperatures: {ace_temperatures} K")

            ace_zaid = None
            if ace_skip_existing:
                try:
                    _endf_for_zaid = read_endf(endf_file)
                    ace_zaid = _endf_for_zaid.zaid
                    _logger.info(f"  ZAID for skip-existing check: {ace_zaid}")
                except Exception:
                    _logger.warning("[WARN] [ACE] Could not read ZAID -- skip-existing disabled", console=False)

            ace_args_list = [
                (f, idx, ace_temperatures, str(output_path), ace_njoy_exe,
                 ace_library_name, ace_njoy_version, ace_xsdir_file,
                 ace_skip_existing, ace_zaid)
                for f, idx in valid_files
            ]

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

            nominal_params = np.zeros_like(mc_mean_params)
            for k, e_idx in enumerate(energy_indices):
                nr = next((r for r in nominal_results if r.energy_index == e_idx), None)
                if nr is not None and nr.has_data:
                    endf_nom = endf_normalize_legendre_coeffs(nr.nominal_coeffs, include_a0=False)
                    n = min(len(endf_nom), max_degree)
                    nominal_params[k * max_degree: k * max_degree + n] = endf_nom[:n]

            cov_matrix_nominal = absolute_to_nominal_relative(cov_abs, nominal_params)
            _nom_rel_std = np.sqrt(np.maximum(np.diag(cov_matrix_nominal), 0.0))
            _logger.info(f"  FG abs→nominal conversion: max rel_std = {np.max(_nom_rel_std)*100:.1f}%")
            if verbose_diagnostics:
                log_rel_std_profile(cov_matrix_nominal, max_degree, "FG post-convert", _logger, verbose=True)

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

            cov_matrix_nominal, _bexp_nom = apply_between_experiment_floor(
                cov_rel=cov_matrix_nominal,
                nominal_results=nominal_results,
                energy_indices=energy_indices,
                max_order=max_degree,
                logger=_logger,
                apply=apply_between_exp_floor,
            )
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

            mf34_ref = mf34_source_file if mf34_source_file else endf_file
            original_mf34_mt = None
            try:
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

            bin_by_idx = {b.index: b for b in energy_bins}
            valid_bins = [bin_by_idx[i] for i in energy_indices]
            energy_grid_ev = contiguous_grid_from_bins(valid_bins)

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

            cov_grouped_nominal = None
            if multigroup_result is not None:
                A = multigroup_result.aggregation_matrix
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

                nom_params_for_mg = np.zeros_like(mc_mean_for_mg)
                for k, vi in enumerate(valid_indices):
                    nr = nominal_results[vi]
                    if nr.has_data:
                        endf_nom = endf_normalize_legendre_coeffs(nr.nominal_coeffs, include_a0=False)
                        n = min(len(endf_nom), max_degree)
                        nom_params_for_mg[k * max_degree: k * max_degree + n] = endf_nom[:n]
                nom_mean_grouped = A @ nom_params_for_mg

                cov_abs_grouped = A @ cov_abs_for_mg @ A.T

                cov_grouped_nominal = absolute_to_nominal_relative(cov_abs_grouped, nom_mean_grouped)
                _mg_nom_rel_std = np.sqrt(np.maximum(np.diag(cov_grouped_nominal), 0.0))
                _logger.info(f"  MG abs→nominal conversion: max rel_std = {np.max(_mg_nom_rel_std)*100:.1f}%")
                if verbose_diagnostics:
                    log_rel_std_profile(cov_grouped_nominal, max_degree, "MG post-convert", _logger, verbose=True)

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

            if (multigroup_regroup_after_smooth
                    and multigroup_result is not None
                    and cov_grouped_nominal is not None):
                from scripts.multigroup_collapse import idx as _mg_idx

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

                _rg_mean_fine = np.zeros(_rg_n_fine * max_degree)
                for _i, _nr in enumerate(_rg_valid_nominal):
                    _c = _nr.nominal_coeffs
                    for _l in range(1, min(max_degree + 1, len(_c) + 1)):
                        _rg_mean_fine[_mg_idx(_i, _l, max_degree)] = _c[_l - 1] if _l - 1 < len(_c) else 0.0

                _rg_valid_mask = np.zeros(_rg_n_fine * max_degree, dtype=bool)
                for _i, _nr in enumerate(_rg_valid_nominal):
                    for _l in range(1, _mg_valid_orders(_nr, max_degree) + 1):
                        _rg_valid_mask[_mg_idx(_i, _l, max_degree)] = True

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

                    A = multigroup_result.aggregation_matrix
                    mc_mean_grouped = A @ mc_mean_for_mg
                    nom_mean_grouped = A @ nom_params_for_mg

                    cov_abs_grouped = A @ cov_abs_for_mg @ A.T

                    cov_grouped_nominal = absolute_to_nominal_relative(cov_abs_grouped, nom_mean_grouped)
                    _rg_nom_rel_std = np.sqrt(np.maximum(np.diag(cov_grouped_nominal), 0.0))
                    _logger.info(f"  RG abs→nominal conversion: max rel_std = {np.max(_rg_nom_rel_std)*100:.1f}%")
                    if verbose_diagnostics:
                        log_rel_std_profile(cov_grouped_nominal, max_degree, "RG post-convert", _logger, verbose=True)

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

            if mf34_covariance_type in ("multigroup", "both") and cov_grouped_nominal is not None:
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

            mf34_sample_source = None
            if generate_mf34_samples and nominal_file:
                sr = sampling_resolution

                if sr == "fine":
                    if merge_original_mf34 and mf34_covariance_type in ("fine", "both"):
                        mf34_sample_source = nominal_file
                    else:
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
                        mf34_sample_source = mg_nom_file
                    else:
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

        if GENERATE_CROSS_TERM_ENDF and mg_nom_file:
            t_step = time.time()
            _logger.info("")
            _logger.info("#-- STEP 10b: MF33<->MF34 cross term ---------------------------------------")
            cross_file = _write_cross_term_endf(mg_nom_file, output_path, _logger)
            if cross_file is None:
                _warning_counts['cross_term_endf_failed'] = 1
            _logger.info(f"#-- END STEP 10b (elapsed: {time.time() - t_step:.2f}s) -----------------------------------")
        elif GENERATE_CROSS_TERM_ENDF:
            _logger.warning(
                "[WARN] [CROSS] no multigroup ENDF was written -- skipping the cross term",
                console=True,
            )

    else:
        mf34_sample_source = None

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

    if _warning_counts:
        _logger.info("")
        _logger.info("#== WARNINGS ===============================================================")
        for wkey, wcount in _warning_counts.items():
            _logger.info(f"  {wkey} -- {wcount}")
        _logger.info("#== END WARNINGS ===========================================================")


    _logger.info(f"  [INFO] Completed! Output: {output_path}", console=True)
    _logger.info(f"  [INFO] Total time: {total_time:.1f}s", console=True)

    return nominal_results, all_samples, output_files


if __name__ == "__main__":
    _preflight_products()
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
        generate_multigroup_covariance=GENERATE_MULTIGROUP_COVARIANCE,
        multigroup_rho_min=MULTIGROUP_RHO_MIN,
        multigroup_sigma_ratio_max=MULTIGROUP_SIGMA_RATIO_MAX,
        multigroup_variance_pct_min=MULTIGROUP_VARIANCE_PCT_MIN,
        multigroup_variance_pct_max=MULTIGROUP_VARIANCE_PCT_MAX,
        multigroup_variance_ratio_ref=MULTIGROUP_VARIANCE_RATIO_REF,
        multigroup_regroup_after_smooth=MULTIGROUP_REGROUP_AFTER_SMOOTH,
        mf34_covariance_type=MF34_COVARIANCE_TYPE,
        use_original_mf34_grid=USE_ORIGINAL_MF34_GRID,
        exfor_db_path=EXFOR_DB_PATH,
        exfor_source=EXFOR_SOURCE,
        target_zaid=TARGET_ZAIDS,
        target_projectile=TARGET_PROJECTILE,
        supplementary_json_files=SUPPLEMENTARY_JSON_FILES,
        exclude_experiments=EXCLUDE_EXPERIMENTS,
        min_relative_uncertainty=MIN_RELATIVE_UNCERTAINTY,
        energy_grid_source=ENERGY_GRID_SOURCE,
        union_grid_subentries=UNION_GRID_SUBENTRIES,
        apply_covariance_cap=APPLY_COVARIANCE_CAP,
        max_relative_std_cap=MAX_RELATIVE_STD_CAP,
        regularize_near_zero=REGULARIZE_NEAR_ZERO_REL_UNC,
        near_zero_snr_threshold=NEAR_ZERO_SNR_THRESHOLD,
        near_zero_n_neighbors=NEAR_ZERO_N_NEIGHBORS,
        apply_between_exp_floor=APPLY_BETWEEN_EXP_FLOOR,
        apply_cov_postprocessing=APPLY_COV_POSTPROCESSING,
        apply_positivity_projection=APPLY_POSITIVITY_PROJECTION,
        positivity_check_points=POSITIVITY_CHECK_POINTS,
        save_correlation_matrices=SAVE_CORRELATION_MATRICES,
        save_tmc_parquet=SAVE_TMC_PARQUET,
        save_raw_kw_parquet=SAVE_RAW_KW_PARQUET,
        save_multigroup_diagnostics_csv=SAVE_MULTIGROUP_DIAGNOSTICS_CSV,
        save_nominal_fits=SAVE_NOMINAL_FITS,
        ace_temperatures=ACE_TEMPERATURES,
        ace_njoy_exe=ACE_NJOY_EXE,
        ace_library_name=ACE_LIBRARY_NAME,
        ace_njoy_version=ACE_NJOY_VERSION,
        ace_xsdir_file=ACE_XSDIR_FILE,
        ace_skip_existing=ACE_SKIP_EXISTING,
        mf34_source_file=MF34_SOURCE_FILE,
        generate_mf34_samples=GENERATE_MF34_SAMPLES,
        generate_mf34_ace=GENERATE_MF34_ACE,
        sampling_resolution=SAMPLING_RESOLUTION,
        merge_original_mf34=MERGE_ORIGINAL_MF34,
        sampling_space=SAMPLING_SPACE,
        sampling_decomposition=SAMPLING_DECOMPOSITION,
        sampling_method=SAMPLING_METHOD,
        verbose_diagnostics=VERBOSE_DIAGNOSTICS,
    )
