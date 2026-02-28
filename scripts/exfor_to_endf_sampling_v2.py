"""
EXFOR-to-ENDF Angular Distribution Sampling Script (v2 - Using kika.exfor module).

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

Author: Generated for kika project
"""
from __future__ import annotations

import os
import sys
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
    DatasetEnergyInfo,
    BinJumpDiagnostics,
    # Energy binning
    compute_energy_bins_with_tof_resolution,
    # EXFOR data conversion (new API -> legacy format)
    build_exfor_cache_from_objects,
    # EXFOR filtering (per-energy methods)
    filter_exfor_with_kernel_weights,
    filter_exfor_with_energy_bin,
    # Energy jitter MC (Improvement 1.4)
    precompute_dataset_energy_info,
    run_mc_with_energy_jitter,
    log_bin_jump_diagnostics,
    # Covariance
    compute_covariance_from_samples,
    cap_covariance_relative_uncertainty,
    save_all_legendre_coefficients,
    # ENDF writing
    write_nominal_endf,
    write_average_endf,
    write_endf_samples_batch,
    write_endf_sample,
    _write_sample_wrapper,
)

# Import multigroup collapse module
from scripts.multigroup_collapse import (
    perform_adaptive_multigroup_collapse,
    MultigroupResult,
)

# Import TOF parameters module (Improvement 1.4)
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
    smooth_tau_in_energy,
    apply_tau_prior_floor,
    compute_n_eff,
    fit_legendre_global_convolution,
    build_global_convolution_system,
    solve_global_convolution,
    solve_global_convolution_shape_only,
    sample_global_convolution_mc,
    sample_global_convolution_mc_shape_only,
    GlobalFitDiagnostics,
    GlobalConvolutionSystem,
)

import time


# =============================================================================
# CONFIGURATION PARAMETERS - MODIFY THESE BEFORE RUNNING
# =============================================================================

# -----------------------------------------------------------------------------
# 1. INPUT/OUTPUT PATHS
# -----------------------------------------------------------------------------
# Reference ENDF file (source of energy grid and original Legendre coefficients)
ENDF_FILE = "/soft_snc/lib/endf/jeff40/neutrons/26-Fe-56g.txt"

# EXFOR JSON directory (for source="json" or "auto")
EXFOR_DIRECTORY = "/share_snc/snc/JuanMonleon/EXFOR/data_v1/"

# X4Pro SQLite database path (for source="database" or "auto")
# Set to None to use KIKA_X4PRO_DB_PATH env variable or builtin default
EXFOR_DB_PATH = '/share_snc/snc/JuanMonleon/EXFOR/x4_iron_angular.db'

# Output directory (all generated files go here)
OUTPUT_DIR = "/SCRATCH/users/monleon-de-la-jan/MCNPy_LIB/EXFOR_FIT_JEFF_V14_ENERGYBIN/"

# -----------------------------------------------------------------------------
# 2. DATA SOURCE CONFIGURATION
# -----------------------------------------------------------------------------
# Data source: "json", "database", "auto" (database + JSON fallback), or "both"
EXFOR_SOURCE = "database"

# Filter options for database queries
# Use list of ZAIDs to include both Fe-56 and natural iron (Fe-0) experiments
# Natural iron is ~92% Fe-56 and has much more experimental coverage in 1-3 MeV range
TARGET_ZAIDS = [26056, 26000]                    # Target ZAIDs: Fe-56 + natural iron
TARGET_PROJECTILE = "N"                          # Projectile (N for neutrons)

# Supplementary JSON files (for experiments not in database)
# These files will be loaded in addition to the main data source
SUPPLEMENTARY_JSON_FILES = [
    '/share_snc/snc/JuanMonleon/EXFOR/data_v1/27673002.json',
    # "C:/Users/Usuario/BaradDur/EXFOR/data_v1/data_v1/27673002.json",  # Gkatis (2025)
]

# -----------------------------------------------------------------------------
# 3. OUTPUT GENERATION OPTIONS
# -----------------------------------------------------------------------------
GENERATE_NOMINAL_ENDF = True                     # Best-fit coefficients ENDF
GENERATE_MC_MEAN_ENDF = True                     # MC mean coefficients ENDF
GENERATE_SAMPLES_ENDF = True                    # Individual MC sample ENDFs
GENERATE_COVARIANCE = True                      # Covariance matrix (.npy)
GENERATE_MF34 = True                            # MF34 covariance section in ENDF
N_SAMPLES = 25                                   # Number of MC samples

# -----------------------------------------------------------------------------
# 3b. MULTIGROUP COVARIANCE OPTIONS
# -----------------------------------------------------------------------------
GENERATE_MULTIGROUP_COVARIANCE = True           # Enable adaptive multigroup collapse
MULTIGROUP_RHO_MIN = 0.90                        # Min correlation to merge (0.85-0.95)
MULTIGROUP_SIGMA_RATIO_MAX = 2.0                 # Max sigma ratio within group (1.5-2.0)
MULTIGROUP_MIN_WIDTH_FACTOR = 2.0                # Group width >= k * median(sigma_E)
MF34_COVARIANCE_TYPE = "both"                    # "fine", "multigroup", or "both"

# Variance percentile for multigroup collapse
# Controls how diagonal variances are scaled after averaging:
# - 50 = median of fine variances in group (typical)
# - 80-90 = conservative but not extreme
# - 100 = maximum fine variance in group (most conservative)
MULTIGROUP_VARIANCE_PERCENTILE = 90.0

# --- Layer 1: Covariance diagonal cap ---
# Caps excessive relative uncertainties in the covariance matrix.
# Set APPLY_COVARIANCE_CAP = False to disable (preserves existing behavior).
APPLY_COVARIANCE_CAP = True
MAX_RELATIVE_STD_CAP = 1.0  # 100% relative uncertainty cap

# --- File output options ---
SAVE_CORRELATION_MATRICES = False       # Save correlation alongside covariance

# --- Layer 2: Positivity-constrained projection ---
# Projects MC samples to ensure non-negative angular distributions.
# Set APPLY_POSITIVITY_PROJECTION = False to disable (preserves existing behavior).
APPLY_POSITIVITY_PROJECTION = True
POSITIVITY_CHECK_POINTS = 50  # Number of mu points in [-1, 1]

# -----------------------------------------------------------------------------
# 4. GENERAL PARAMETERS (Apply to ALL methods)
# -----------------------------------------------------------------------------
# Energy range to process (in MeV)
ENERGY_MIN_MEV = 0.847
ENERGY_MAX_MEV = 4

# MT reaction number (2 = elastic scattering)
MT_NUMBER = 2

# Target isotope masses (for LAB->CM frame conversion)
M_PROJ_U = 1.008665                              # Projectile mass in u (neutron)
M_TARG_U = 55.93494                              # Target mass in u (Fe-56)

# Legendre fitting parameters
MAX_LEGENDRE_DEGREE = 6                          # Maximum Legendre order (capped at 8)
SELECT_DEGREE = "aicc"                           # "aicc", "bic", or None (use max)
RIDGE_LAMBDA = 1e-6                              # Ridge regularization parameter
RIDGE_POWER = 4                                  # Power for ridge penalty (l^ridge_power)
DF_METHOD = "hat"                                # Degrees of freedom method: "hat" or "naive"

# Processing options
N_PROCS = 12                                      # Parallel processes (1 = sequential)
BASE_SEED = 42                                   # Random seed for reproducibility

# -----------------------------------------------------------------------------
# 5. EXPERIMENT SELECTION METHOD
# -----------------------------------------------------------------------------
# Available methods:
#
# "global_convolution"
#     Fits ALL energy points simultaneously using Tikhonov regularization.
#     Properly accounts for energy resolution smearing across energy bins.
#     Each EXFOR measurement contributes to multiple ENDF energies according
#     to its resolution-weighted probability. Enforces smooth energy dependence.
#     --> Uses parameters from section 6a (TOF Resolution) and 6b (Global Convolution)
#
# "kernel_weights"
#     Fits each ENDF energy point independently using Gaussian kernel weighting.
#     EXFOR data are weighted by g_ij = exp(-0.5 * ((E_i - E_j)/σE)²)
#     --> Uses parameters from sections 6a, 6c, 6d, 6e, 6f
#
# "energy_bin"
#     Simple energy binning without resolution-based weighting.
#     Uses hard bin boundaries (no Gaussian weighting).
#     Fastest but ignores energy resolution effects.
#     --> Uses parameters from sections 6d, 6e, 6f
#
EXPERIMENT_SELECTION_METHOD = "energy_bin"

# --- Experiment Exclusion and Uncertainty Floor ---
# Experiments to exclude from fitting (e.g., experiments with known issues)
# Accepts formats: "20743" (all subentries), "20743002", or "20743/002"
EXCLUDE_EXPERIMENTS = ["20743002", "32246002"]  
# - "20743002" - Cierjacks (1978)
# - "32246002" - Tostkii (1957)


# Minimum relative uncertainty floor (prevents unrealistically small errors from dominating)
# Set to 0.0 to disable. e.g., 0.05 for 5% minimum uncertainty
MIN_RELATIVE_UNCERTAINTY = 0.05

# -----------------------------------------------------------------------------
# 6. METHOD-SPECIFIC PARAMETERS
# -----------------------------------------------------------------------------

# --- 6a. TOF Energy Resolution (kernel_weights, global_convolution) ---
DELTA_T_NS = 5.0                                 # Time resolution in nanoseconds
FLIGHT_PATH_M = 27.037                           # Flight path in meters
N_SIGMA_CUTOFF = 3.0                             # Gaussian kernel cutoff (±n_sigma * σE)

# --- 6b. Global Convolution (EXPERIMENT_SELECTION_METHOD = "global_convolution") ---
GLOBAL_CONV_LAMBDA = 0.001                       # Tikhonov regularization strength
L_DEPENDENT_POWER = 2.0                          # ℓ-scaling exponent (2-4 recommended)
SKIP_C0_REGULARIZATION = True                    # Don't apply smoothing penalty to c0
MIN_WEIGHT_SUM_THRESHOLD = 0.95                  # Warn if weight_sum < this (skip if < 0.5)
GLOBAL_CONV_SHAPE_ONLY = True                   # Two-pass shape-only fit (Improvement 3.4)

# --- 6c. Kernel Weight Control (EXPERIMENT_SELECTION_METHOD = "kernel_weights") ---
MIN_KERNEL_WEIGHT_FRACTION = 1e-3                # Minimum weight threshold
MAX_EXPERIMENT_WEIGHT_FRACTION = 0.5             # Weight cap per experiment
N_EFF_WARNING_THRESHOLD = 5.0                    # Warning if N_eff < threshold
WEIGHT_SPAN_WARNING_RATIO = 3.0                  # Warning if span > ratio * σE
DEDUPE_NOMINAL = True                            # Dedupe for nominal fits (stability)
DEDUPE_MC = False                                # Dedupe for MC sampling (False enables energy correlations)
USE_OVERLAP_WEIGHTS = True                       # Use overlap weights (True) vs Gaussian kernel (False)

# --- 6d. Angular-Band Discrepancy (kernel_weights, energy_bin) ---
USE_BAND_DISCREPANCY = True                      # Use band-based uncertainty (vs global Birge)
MIN_POINTS_PER_BAND = 3                          # Minimum points to estimate τ_b per band
MAX_TAU_FRACTION = 0.25                          # Cap τ_b at 25% of cross section
TAU_SMOOTHING_WINDOW = 3                         # Moving median window for τ_b(E) smoothing
TAU_PRIOR_FLOOR = True                           # Apply tau prior floor from multi-experiment bins
TAU_PRIOR_MIN_EXPERIMENTS = 2                    # Min experiments to count as "well-estimated"
TAU_PRIOR_PERCENTILE = 50                        # Percentile of well-estimated tau for baseline
RESCALE_UNC_BY_CHI2 = True                       # Apply Birge scaling when band discrepancy disabled
ALLOW_SHRINK_UNC = False                         # Allow uncertainties to shrink (chi2_red < 1)

# --- 6e. Per-Experiment Normalization (kernel_weights, energy_bin) ---
NORMALIZATION_SIGMA = 0.05                       # Per-experiment normalization uncertainty (5%)
NORM_DIST = "lognormal"                          # Distribution: "lognormal" (always positive) or "normal"

# --- 6f. Model Averaging (kernel_weights, energy_bin) ---
USE_MODEL_AVERAGING = True                       # Enable model averaging over Legendre orders
MIN_DEGREE_FOR_AVERAGING = 1                     # Minimum degree to consider (1 = include all)
USE_DEGREE_SAMPLING_IN_MC = True                 # Sample degree from degree_weights distribution

# --- 6g. Energy Bin Method Specific (Improvements 1.1-1.2) ---
NORMALIZE_BY_N_POINTS = True                     # Equal weight per experiment (1/n_points weighting)
MAX_EXP_WEIGHT_FRAC_BIN = 0.5                    # Cap per-experiment dominance (1.0 = disabled)
FREEZE_C0 = True                                # Fix c0 for shape-only refits

# --- 6h. Energy Jitter for Cross-Bin Coupling (Improvement 1.4) ---
USE_ENERGY_JITTER = True                         # Enable energy jitter for cross-bin correlation
TOF_PARAMETERS_FILE = "/share_snc/snc/JuanMonleon/EXFOR/exfor_tof_parameters.json"
JITTER_N_SIGMA_CLIP = 3.0                        # Clip jitter at ±n_sigma
TRACK_BIN_JUMPS = True                           # Track bin jump statistics for diagnostics

# =============================================================================
# END OF CONFIGURATION
# =============================================================================


# Global logger reference (set by _set_logger from exfor_utils)
_logger = None


# =============================================================================
# PARALLEL MC HELPER (top-level for pickling)
# =============================================================================

def _mc_one_bin(args):
    """
    Run MC sampling for a single energy bin (top-level for Pool.map pickling).

    Returns
    -------
    tuple
        (energy_idx, is_interpolated, results_by_sample, success)
        where results_by_sample is Dict[s_idx, np.ndarray] of ENDF coefficients.
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
        max_tau_fraction,
        use_degree_sampling_in_mc,
        rescale_unc_by_chi2,
        allow_shrink_unc,
        freeze_c0,
        normalization_sigma,
        norm_dist,
        _apply_positivity_projection,
        _positivity_check_points,
    ) = args

    energy_idx = nr_energy_idx

    if nr_interpolated:
        endf_coeffs = endf_normalize_legendre_coeffs(nr_nominal_coeffs, include_a0=False)
        results = {s_idx: endf_coeffs for s_idx in range(n_samples)}
        return (energy_idx, True, results, True)

    bin_seed = base_seed + energy_idx
    rng = np.random.default_rng(bin_seed)
    mc_weights = nr_mc_weights

    use_degree_sampling = (
        use_degree_sampling_in_mc and
        nr_degree_weights is not None and
        len(nr_degree_weights) > 1
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

            for s_idx in range(n_samples):
                sample_degree = rng.choice(degrees, p=probs)
                coef_df_single, _ = sample_legendre_coefficients(
                    nr_mc_df,
                    value_col="value",
                    unc_col="unc",
                    degree=sample_degree,
                    max_degree=max_degree,
                    select_degree=None,
                    ridge_lambda=ridge_lambda,
                    ridge_power=ridge_power,
                    df_method=df_method,
                    external_weights=mc_weights if len(mc_weights) > 0 else None,
                    n_samples=1,
                    stochastic=True,
                    rescale_unc_by_chi2=rescale_unc_by_chi2,
                    allow_shrink_unc=allow_shrink_unc,
                    random_state=bin_seed + s_idx,
                    use_band_discrepancy=use_band_discrepancy,
                    min_points_per_band=min_points_per_band,
                    max_tau_fraction=max_tau_fraction,
                    freeze_c0=freeze_c0,
                    sigma_norm=normalization_sigma,
                    norm_dist=norm_dist,
                )
                sample_coeffs = coef_df_single.iloc[0].to_numpy()
                if len(sample_coeffs) < max_degree + 1:
                    sample_coeffs = np.pad(sample_coeffs, (0, max_degree + 1 - len(sample_coeffs)))
                if _apply_positivity_projection:
                    if not check_angular_distribution_positivity(sample_coeffs, _positivity_check_points):
                        sample_coeffs = project_to_positive_distribution(sample_coeffs, _positivity_check_points)
                endf_coeffs = endf_normalize_legendre_coeffs(sample_coeffs, include_a0=False)
                results[s_idx] = endf_coeffs
        else:
            from scripts.resample_AD import (
                check_angular_distribution_positivity,
                project_to_positive_distribution,
            )

            coef_df, _ = sample_legendre_coefficients(
                nr_mc_df,
                value_col="value",
                unc_col="unc",
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
                max_tau_fraction=max_tau_fraction,
                freeze_c0=freeze_c0,
                sigma_norm=normalization_sigma,
                norm_dist=norm_dist,
            )
            for s_idx in range(n_samples):
                sample_coeffs = coef_df.iloc[s_idx].to_numpy()
                if _apply_positivity_projection:
                    if not check_angular_distribution_positivity(sample_coeffs, _positivity_check_points):
                        sample_coeffs = project_to_positive_distribution(sample_coeffs, _positivity_check_points)
                endf_coeffs = endf_normalize_legendre_coeffs(sample_coeffs, include_a0=False)
                results[s_idx] = endf_coeffs

        return (energy_idx, False, results, True)

    except Exception:
        endf_coeffs = endf_normalize_legendre_coeffs(nr_nominal_coeffs, include_a0=False)
        results = {s_idx: endf_coeffs for s_idx in range(n_samples)}
        return (energy_idx, False, results, False)


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
    # Two-pass dedupe: store non-dedupe data for MC if needed
    exfor_df_mc: Optional[pd.DataFrame] = None
    kernel_weights_mc: Optional[np.ndarray] = None


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


def perform_nominal_fits(
    energy_bins: List[EnergyBinInfo],
    exfor_cache: Dict[float, List[Tuple[pd.DataFrame, Dict]]],
    sorted_energies: List[float],
    n_sigma: float,
    max_degree: int,
    select_degree: Optional[str],
    ridge_lambda: float,
    m_proj_u: float,
    m_targ_u: float,
    use_band_discrepancy: bool,
    min_points_per_band: int,
    max_tau_fraction: float,
    tau_smoothing_window: int,
    min_kernel_weight_fraction: float = 1e-3,
    max_experiment_weight_fraction: float = 0.5,
    n_eff_warning_threshold: float = 5.0,
    weight_span_warning_ratio: float = 3.0,
    experiment_selection_method: str = "global_convolution",
    min_degree_for_averaging: int = 3,
    delta_t_ns: float = 5.0,
    flight_path_m: float = 27.037,
    tikhonov_lambda: float = 0.001,
    l_dependent_power: float = 2.0,
    skip_c0_regularization: bool = True,
    min_weight_sum_threshold: float = 0.95,
    shape_only: bool = False,
    exclude_experiments: Optional[List[str]] = None,
    min_relative_uncertainty: float = 0.0,
    tau_prior_floor: bool = True,
    tau_prior_min_experiments: int = 2,
    tau_prior_percentile: float = 50.0,
    logger = None,
) -> Tuple[List[NominalFitResult], Optional[GlobalConvolutionSystem], Optional[Dict[int, float]]]:
    """Phase 1: Perform nominal fits to determine frozen orders and band discrepancies.

    Returns
    -------
    Tuple[List[NominalFitResult], Optional[GlobalConvolutionSystem], Optional[Dict[int, float]]]
        - List of nominal fit results for each energy bin
        - GlobalConvolutionSystem for MC sampling (only for global_convolution method)
        - c0_frozen dict (energy_index -> c0 value) if shape_only=True and global_convolution method, else None
    """
    from numpy.polynomial.legendre import legvander, legval

    logger = _get_logger()
    results = []
    global_system = None  # Will be set only for global_convolution method
    c0_frozen = None  # Will be set only for global_convolution with shape_only=True

    # GLOBAL CONVOLUTION METHOD
    if experiment_selection_method == "global_convolution":
        if logger:
            logger.info("")
            logger.info("[GLOBAL CONVOLUTION FIT]")
            logger.info(f"  Method: global_convolution (all energies fitted simultaneously)")
            logger.info(f"  Tikhonov λ: {tikhonov_lambda}")
            logger.info(f"  ℓ-dependent power: {l_dependent_power}")
            logger.info(f"  Skip c0 regularization: {skip_c0_regularization}")
            logger.info(f"  Shape-only mode (Improvement 3.4): {shape_only}")
            logger.info(f"  Max Legendre degree: {max_degree}")
            logger.info(f"  Energy kernel cutoff: ±{n_sigma}σ")
            logger.info(f"  Min weight fraction: {min_kernel_weight_fraction}")
            logger.info(f"  Min weight sum threshold: {min_weight_sum_threshold}")
            logger.info("")

        coeffs_by_energy, global_diag, global_system, c0_frozen = fit_legendre_global_convolution(
            exfor_cache=exfor_cache,
            sorted_energies=sorted_energies,
            energy_bins=energy_bins,
            max_degree=max_degree,
            n_sigma=n_sigma,
            tikhonov_lambda=tikhonov_lambda,
            min_kernel_weight_fraction=min_kernel_weight_fraction,
            min_weight_sum_threshold=min_weight_sum_threshold,
            m_proj_u=m_proj_u,
            m_targ_u=m_targ_u,
            delta_t_ns=delta_t_ns,
            flight_path_m=flight_path_m,
            l_dependent_power=l_dependent_power,
            skip_c0_regularization=skip_c0_regularization,
            shape_only=shape_only,
            logger=logger,
        )

        for bin_info in energy_bins:
            if bin_info.index in coeffs_by_energy:
                coeffs = coeffs_by_energy[bin_info.index]
                chi2_red = global_diag.chi2_per_energy.get(bin_info.index, 0.0)
                n_eff = global_diag.n_eff_per_energy.get(bin_info.index, 0.0)

                kernel_diag = KernelDiagnostics(
                    n_eff=n_eff,
                    weight_span_95=0.0,
                    weight_span_ratio=0.0,
                    n_experiments=0,
                    max_experiment_weight_frac=0.0,
                    experiment_weights={},
                    n_points_dropped=0,
                    capping_applied=False,
                )

                results.append(NominalFitResult(
                    energy_mev=bin_info.energy_mev,
                    energy_index=bin_info.index,
                    exfor_df=pd.DataFrame(),
                    experiments_info=[],
                    kernel_weights=np.array([]),
                    frozen_degree=max_degree,
                    nominal_coeffs=coeffs,
                    sigma_eff=np.array([]),
                    tau_info={'tau_F': 0.0, 'tau_M': 0.0, 'tau_B': 0.0},
                    chi2_red=chi2_red,
                    has_data=True,
                    kernel_diagnostics=kernel_diag,
                    degree_weights=None,
                    all_degrees_info=None,
                ))
            else:
                results.append(NominalFitResult(
                    energy_mev=bin_info.energy_mev,
                    energy_index=bin_info.index,
                    exfor_df=pd.DataFrame(),
                    experiments_info=[],
                    kernel_weights=np.array([]),
                    frozen_degree=0,
                    nominal_coeffs=np.array([1.0]),
                    sigma_eff=np.array([]),
                    tau_info={'tau_F': 0.0, 'tau_M': 0.0, 'tau_B': 0.0},
                    chi2_red=0.0,
                    has_data=False,
                    kernel_diagnostics=None,
                    degree_weights=None,
                    all_degrees_info=None,
                ))

        if logger:
            n_with_data = len(global_diag.energies_with_data)
            logger.info("[GLOBAL FIT SUMMARY]")
            logger.info(f"  Energies with data: {n_with_data}/{len(energy_bins)}")
            logger.info(f"  Total χ² = {global_diag.chi2:.2f}")
            logger.info(f"  Total data points = {global_diag.n_data_points}")
            # Log weight guard diagnostics (Improvement 3.2)
            if global_diag.weight_sum_min < 1.0:
                logger.info(f"  Min weight sum: {global_diag.weight_sum_min:.3f}")
            if global_diag.n_datasets_skipped > 0:
                logger.warning(f"  Datasets skipped (severe truncation): {global_diag.n_datasets_skipped}")
            if global_diag.truncated_datasets:
                logger.info(f"  Datasets with moderate truncation: {len(global_diag.truncated_datasets)}")
            logger.info("")
            # Log per-energy results in condensed form
            logger.info("  Per-energy results:")
            for bin_info in energy_bins:
                if bin_info.index in coeffs_by_energy:
                    chi2_e = global_diag.chi2_per_energy.get(bin_info.index, 0.0)
                    n_eff_e = global_diag.n_eff_per_energy.get(bin_info.index, 0.0)
                    logger.info(
                        f"    E={bin_info.energy_mev:.4f} MeV: χ²/dof={chi2_e:.2f}, N_eff={n_eff_e:.1f}"
                    )
                else:
                    logger.info(f"    E={bin_info.energy_mev:.4f} MeV: No data")
            logger.info("")

        return results, global_system, c0_frozen

    # PER-ENERGY METHODS
    for bin_info in energy_bins:
        # Initialize MC-specific data (only used by kernel_weights method with two-pass dedupe)
        exfor_df_mc = None
        kernel_weights_mc = None

        if experiment_selection_method == "energy_bin":
            exfor_df, experiments_info, kernel_weights, diagnostics = filter_exfor_with_energy_bin(
                exfor_cache=exfor_cache,
                sorted_energies=sorted_energies,
                bin_lower_mev=bin_info.bin_lower_mev,
                bin_upper_mev=bin_info.bin_upper_mev,
                target_energy_mev=bin_info.energy_mev,
                m_proj_u=m_proj_u,
                m_targ_u=m_targ_u,
                dedupe_per_experiment=DEDUPE_NOMINAL,
                exclude_experiments=exclude_experiments,
                min_relative_uncertainty=min_relative_uncertainty,
                # Per-experiment weighting (Improvement 1.1)
                normalize_by_n_points=NORMALIZE_BY_N_POINTS,
                max_experiment_weight_fraction=MAX_EXP_WEIGHT_FRAC_BIN,
            )
        else:
            # Nominal fit filter (always with DEDUPE_NOMINAL)
            exfor_df, experiments_info, kernel_weights, diagnostics = filter_exfor_with_kernel_weights(
                exfor_cache=exfor_cache,
                sorted_energies=sorted_energies,
                energy_mev=bin_info.energy_mev,
                sigma_E_mev=bin_info.sigma_E_mev,
                n_sigma=n_sigma,
                m_proj_u=m_proj_u,
                m_targ_u=m_targ_u,
                bin_lower_mev=bin_info.bin_lower_mev,
                bin_upper_mev=bin_info.bin_upper_mev,
                min_kernel_weight_fraction=min_kernel_weight_fraction,
                max_experiment_weight_fraction=max_experiment_weight_fraction,
                default_delta_t_ns=delta_t_ns,
                default_flight_path_m=flight_path_m,
                use_overlap_weights=USE_OVERLAP_WEIGHTS,
                normalize_by_n_points=NORMALIZE_BY_N_POINTS,
                dedupe_per_experiment=DEDUPE_NOMINAL,
                exclude_experiments=exclude_experiments,
                min_relative_uncertainty=min_relative_uncertainty,
                logger=logger,
            )

            # MC filter (if different from nominal - two-pass dedupe)
            if DEDUPE_MC != DEDUPE_NOMINAL:
                exfor_df_mc, _, kernel_weights_mc, _ = filter_exfor_with_kernel_weights(
                    exfor_cache=exfor_cache,
                    sorted_energies=sorted_energies,
                    energy_mev=bin_info.energy_mev,
                    sigma_E_mev=bin_info.sigma_E_mev,
                    n_sigma=n_sigma,
                    m_proj_u=m_proj_u,
                    m_targ_u=m_targ_u,
                    bin_lower_mev=bin_info.bin_lower_mev,
                    bin_upper_mev=bin_info.bin_upper_mev,
                    min_kernel_weight_fraction=min_kernel_weight_fraction,
                    max_experiment_weight_fraction=max_experiment_weight_fraction,
                    default_delta_t_ns=delta_t_ns,
                    default_flight_path_m=flight_path_m,
                    use_overlap_weights=USE_OVERLAP_WEIGHTS,
                    normalize_by_n_points=NORMALIZE_BY_N_POINTS,
                    dedupe_per_experiment=DEDUPE_MC,
                    exclude_experiments=exclude_experiments,
                    min_relative_uncertainty=min_relative_uncertainty,
                    logger=None,  # Don't log MC filter details
                )

        if exfor_df.empty or len(exfor_df) < 3:
            results.append(NominalFitResult(
                energy_mev=bin_info.energy_mev,
                energy_index=bin_info.index,
                exfor_df=pd.DataFrame(),
                experiments_info=[],
                kernel_weights=np.array([]),
                frozen_degree=0,
                nominal_coeffs=np.array([1.0]),
                sigma_eff=np.array([]),
                tau_info={'tau_F': 0.0, 'tau_M': 0.0, 'tau_B': 0.0},
                chi2_red=0.0,
                has_data=False,
                kernel_diagnostics=None,
            ))
            if logger:
                logger.warning(f"E={bin_info.energy_mev:.4f} MeV: No EXFOR data (σE={bin_info.sigma_E_mev:.4f} MeV)")
            continue

        bin_info.has_exfor_data = True
        bin_info.exfor_n_points = len(exfor_df)
        bin_info.exfor_n_experiments = len(experiments_info)
        bin_info.experiments_used = experiments_info

        mu = exfor_df['mu'].to_numpy()
        y = exfor_df['value'].to_numpy()
        sigma = exfor_df['unc'].to_numpy()

        coef_df, fit_info = sample_legendre_coefficients(
            exfor_df,
            value_col="value",
            unc_col="unc",
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
            max_tau_fraction=max_tau_fraction,
        )

        frozen_degree = fit_info['degree']
        nominal_coeffs = coef_df.iloc[0].to_numpy()
        chi2_red = fit_info['chi2_red']
        tau_info = fit_info.get('tau_info', {'tau_F': 0.0, 'tau_M': 0.0, 'tau_B': 0.0})

        all_degrees_info = fit_info.get('all_degrees_info', None)
        degree_weights = None

        if all_degrees_info and len(all_degrees_info) > 1:
            aicc_values = {d: info['aicc'] for d, info in all_degrees_info.items()}
            min_aicc = min(aicc_values.values())
            raw_weights = {d: np.exp(-0.5 * (aicc - min_aicc)) for d, aicc in aicc_values.items()}
            total = sum(raw_weights.values())
            degree_weights = {d: w / total for d, w in raw_weights.items()}
            degree_weights = {d: w for d, w in degree_weights.items() if w > 0.01 and d >= min_degree_for_averaging}
            if degree_weights:
                total = sum(degree_weights.values())
                degree_weights = {d: w / total for d, w in degree_weights.items()}
            else:
                degree_weights = {frozen_degree: 1.0}

        y_fit = legval(mu, nominal_coeffs)

        if use_band_discrepancy and tau_info:
            sigma_eff, _ = compute_angular_band_discrepancy(
                mu=mu, y=y, sigma=sigma, y_fit=y_fit,
                min_points_per_band=min_points_per_band,
                max_tau_fraction=max_tau_fraction,
            )
            tau_F = tau_info.get('tau_F', 0.0)
            tau_M = tau_info.get('tau_M', 0.0)
            tau_B = tau_info.get('tau_B', 0.0)
        else:
            scale = max(1.0, np.sqrt(chi2_red))
            sigma_eff = sigma * scale
            tau_F = tau_M = tau_B = 0.0
            tau_info = {'tau_F': 0.0, 'tau_M': 0.0, 'tau_B': 0.0}

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
            if diagnostics.weight_span_ratio > weight_span_warning_ratio:
                logger.warning(
                    f"E={bin_info.energy_mev:.4f} MeV: Wide weight span "
                    f"({diagnostics.weight_span_95:.4f} MeV = {diagnostics.weight_span_ratio:.1f}×σE)"
                )

        results.append(NominalFitResult(
            energy_mev=bin_info.energy_mev,
            energy_index=bin_info.index,
            exfor_df=exfor_df,
            experiments_info=experiments_info,
            kernel_weights=kernel_weights,
            frozen_degree=frozen_degree,
            nominal_coeffs=nominal_coeffs,
            sigma_eff=sigma_eff,
            tau_info=tau_info if tau_info else {'tau_F': 0.0, 'tau_M': 0.0, 'tau_B': 0.0},
            chi2_red=chi2_red,
            has_data=True,
            kernel_diagnostics=diagnostics,
            degree_weights=degree_weights,
            all_degrees_info=all_degrees_info,
            # Two-pass dedupe: MC-specific data (only used by kernel_weights method)
            exfor_df_mc=exfor_df_mc,
            kernel_weights_mc=kernel_weights_mc,
        ))

        # Log comprehensive energy bin information
        if logger:
            # Energy header with bin boundaries or σE depending on method
            if experiment_selection_method == "energy_bin":
                logger.info(
                    f"E = {bin_info.energy_mev:.4f} MeV (bin: [{bin_info.bin_lower_mev:.4f}, {bin_info.bin_upper_mev:.4f}] MeV):"
                )
            else:
                logger.info(
                    f"E = {bin_info.energy_mev:.4f} MeV (σE = {bin_info.sigma_E_mev:.4f} MeV):"
                )

            # Experiments used (condensed - one line per experiment with ranges)
            condensed_lines = _format_condensed_experiments(experiments_info)
            for line in condensed_lines:
                logger.info(line)

            # Fit results
            logger.info(
                f"  Fit: L={frozen_degree}, χ²/dof={chi2_red:.2f}, {len(exfor_df)} pts, N_eff={final_n_eff:.1f}"
            )
            logger.info(
                f"  τ values: τ_F={tau_F:.4f}, τ_M={tau_M:.4f}, τ_B={tau_B:.4f}"
            )
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
            min_experiments=tau_prior_min_experiments,
            percentile=tau_prior_percentile,
        )
        if logger:
            logger.info(f"  Tau prior floor baselines: τ_F={baselines['tau_F']:.4f}, "
                        f"τ_M={baselines['tau_M']:.4f}, τ_B={baselines['tau_B']:.4f}")

    return results, None, None  # No global_system or c0_frozen for per-energy methods


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
    delta_t_ns: float = 5.0,
    flight_path_m: float = 27.037,
    n_sigma_cutoff: float = 3.0,
    use_band_discrepancy: bool = True,
    min_points_per_band: int = 3,
    max_tau_fraction: float = 0.25,
    tau_smoothing_window: int = 3,
    tau_prior_floor: bool = True,
    tau_prior_min_experiments: int = 2,
    tau_prior_percentile: float = 50.0,
    sigma_norm: float = 0.05,
    use_model_averaging: bool = True,
    min_degree_for_averaging: int = 3,
    min_kernel_weight_fraction: float = 1e-3,
    max_experiment_weight_fraction: float = 0.5,
    n_eff_warning_threshold: float = 5.0,
    weight_span_warning_ratio: float = 3.0,
    experiment_selection_method: str = "global_convolution",
    tikhonov_lambda: float = 0.001,
    n_procs: int = 1,
    base_seed: int = 42,
    generate_nominal_endf: bool = True,
    generate_mc_mean_endf: bool = True,
    generate_samples_endf: bool = True,
    generate_covariance: bool = True,
    generate_mf34: bool = False,
    # Multigroup covariance options
    generate_multigroup_covariance: bool = False,
    multigroup_rho_min: float = 0.90,
    multigroup_sigma_ratio_max: float = 1.7,
    multigroup_min_width_factor: float = 2.0,
    multigroup_variance_percentile: float = 50.0,
    mf34_covariance_type: str = "fine",
    # Database configuration (new parameters)
    exfor_db_path: str = None,
    exfor_source: str = "auto",
    target_zaid: Union[int, List[int]] = None,
    target_projectile: str = "N",
    supplementary_json_files: List[str] = None,
    # Experiment exclusion and uncertainty floor
    exclude_experiments: Optional[List[str]] = None,
    min_relative_uncertainty: float = 0.0,
    # Covariance cap (Layer 1) and positivity projection (Layer 2)
    apply_covariance_cap: bool = False,
    max_relative_std_cap: float = 1.0,
    apply_positivity_projection: bool = False,
    positivity_check_points: int = 50,
    # File output options
    save_correlation_matrices: bool = False,
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

    separator = "=" * 80
    _logger.info(separator)
    _logger.info("EXFOR-to-ENDF Angular Distribution Sampling (v2 - using kika.exfor)")
    _logger.info(separator)
    _logger.info(f"Timestamp: {datetime.now().isoformat()}")
    _logger.info("")

    print(f"[INFO] Starting EXFOR-to-ENDF sampling (v2)")
    print(f"[INFO] Log file: {log_file}")

    # ── [RUN PARAMETERS] ─────────────────────────────────────────────────────
    _logger.info("[RUN PARAMETERS]")
    _logger.info("")

    # -- General: Paths --
    _logger.info("  Paths:")
    _logger.info(f"    ENDF_FILE              = {endf_file}")
    _logger.info(f"    EXFOR_DIRECTORY         = {exfor_directory}")
    _logger.info(f"    EXFOR_DB_PATH           = {exfor_db_path}")
    _logger.info(f"    OUTPUT_DIR              = {output_dir}")
    _logger.info("")

    # -- General: Data source --
    _logger.info("  Data Source:")
    _logger.info(f"    EXFOR_SOURCE            = {exfor_source}")
    _logger.info(f"    TARGET_ZAIDS            = {target_zaid}")
    _logger.info(f"    TARGET_PROJECTILE       = {target_projectile}")
    _logger.info(f"    SUPPLEMENTARY_JSON_FILES = {supplementary_json_files}")
    _logger.info("")

    # -- General: Physics --
    _logger.info("  Energy Range & Physics:")
    _logger.info(f"    ENERGY_MIN_MEV          = {energy_min_mev}")
    _logger.info(f"    ENERGY_MAX_MEV          = {energy_max_mev}")
    _logger.info(f"    MT_NUMBER               = {mt_number}")
    _logger.info(f"    M_PROJ_U                = {m_proj_u}")
    _logger.info(f"    M_TARG_U                = {m_targ_u}")
    _logger.info("")

    # -- General: Legendre fitting --
    _logger.info("  Legendre Fitting:")
    _logger.info(f"    MAX_LEGENDRE_DEGREE     = {max_degree}")
    _logger.info(f"    SELECT_DEGREE           = {select_degree if select_degree else 'None (use max)'}")
    _logger.info(f"    RIDGE_LAMBDA            = {ridge_lambda}")
    _logger.info(f"    RIDGE_POWER             = {RIDGE_POWER}")
    _logger.info(f"    DF_METHOD               = {DF_METHOD}")
    _logger.info("")

    # -- General: Output flags --
    _logger.info("  Output Flags:")
    _logger.info(f"    GENERATE_NOMINAL_ENDF   = {generate_nominal_endf}")
    _logger.info(f"    GENERATE_MC_MEAN_ENDF   = {generate_mc_mean_endf}")
    _logger.info(f"    GENERATE_SAMPLES_ENDF   = {generate_samples_endf}")
    _logger.info(f"    GENERATE_COVARIANCE     = {generate_covariance}")
    _logger.info(f"    GENERATE_MF34           = {generate_mf34}")
    _logger.info(f"    N_SAMPLES               = {n_samples}")
    _logger.info("")

    # -- General: Processing --
    _logger.info("  Processing:")
    _logger.info(f"    N_PROCS                 = {n_procs}")
    _logger.info(f"    BASE_SEED               = {base_seed}")
    _logger.info("")

    # -- General: Exclusions --
    _logger.info("  Exclusions & Uncertainty:")
    _logger.info(f"    EXCLUDE_EXPERIMENTS     = {exclude_experiments if exclude_experiments else 'None'}")
    _logger.info(f"    MIN_RELATIVE_UNCERTAINTY = {min_relative_uncertainty} ({min_relative_uncertainty*100:.1f}%)")
    _logger.info("")

    # -- Multigroup Covariance (only if enabled) --
    if generate_multigroup_covariance:
        _logger.info("  Multigroup Covariance:")
        _logger.info(f"    GENERATE_MULTIGROUP_COVARIANCE = {generate_multigroup_covariance}")
        _logger.info(f"    MULTIGROUP_RHO_MIN             = {multigroup_rho_min}")
        _logger.info(f"    MULTIGROUP_SIGMA_RATIO_MAX     = {multigroup_sigma_ratio_max}")
        _logger.info(f"    MULTIGROUP_MIN_WIDTH_FACTOR    = {multigroup_min_width_factor}")
        _logger.info(f"    MF34_COVARIANCE_TYPE           = {mf34_covariance_type}")
        _logger.info(f"    MULTIGROUP_VARIANCE_PERCENTILE = {multigroup_variance_percentile}")
        _logger.info("")

    # -- Experiment Selection Method --
    _logger.info(f"  EXPERIMENT_SELECTION_METHOD = {experiment_selection_method}")
    _logger.info("")

    # -- 6a. TOF Resolution (kernel_weights, global_convolution) --
    if experiment_selection_method in ("kernel_weights", "global_convolution"):
        _logger.info("  TOF Energy Resolution (6a):")
        _logger.info(f"    DELTA_T_NS              = {delta_t_ns}")
        _logger.info(f"    FLIGHT_PATH_M           = {flight_path_m}")
        _logger.info(f"    N_SIGMA_CUTOFF          = {n_sigma_cutoff}")
        _logger.info("")

    # -- 6b. Global Convolution (global_convolution only) --
    if experiment_selection_method == "global_convolution":
        _logger.info("  Global Convolution (6b):")
        _logger.info(f"    GLOBAL_CONV_LAMBDA      = {tikhonov_lambda}")
        _logger.info(f"    L_DEPENDENT_POWER       = {L_DEPENDENT_POWER}")
        _logger.info(f"    SKIP_C0_REGULARIZATION  = {SKIP_C0_REGULARIZATION}")
        _logger.info(f"    MIN_WEIGHT_SUM_THRESHOLD = {MIN_WEIGHT_SUM_THRESHOLD}")
        _logger.info(f"    GLOBAL_CONV_SHAPE_ONLY  = {GLOBAL_CONV_SHAPE_ONLY}")
        _logger.info("")

    # -- 6c. Kernel Weight Control (kernel_weights only) --
    if experiment_selection_method == "kernel_weights":
        _logger.info("  Kernel Weight Control (6c):")
        _logger.info(f"    MIN_KERNEL_WEIGHT_FRACTION     = {min_kernel_weight_fraction}")
        _logger.info(f"    MAX_EXPERIMENT_WEIGHT_FRACTION  = {max_experiment_weight_fraction}")
        _logger.info(f"    N_EFF_WARNING_THRESHOLD        = {n_eff_warning_threshold}")
        _logger.info(f"    WEIGHT_SPAN_WARNING_RATIO      = {weight_span_warning_ratio}")
        _logger.info(f"    DEDUPE_NOMINAL                 = {DEDUPE_NOMINAL}")
        _logger.info(f"    DEDUPE_MC                      = {DEDUPE_MC}")
        _logger.info(f"    USE_OVERLAP_WEIGHTS            = {USE_OVERLAP_WEIGHTS}")
        _logger.info("")

    # -- 6d. Angular-Band Discrepancy (kernel_weights, energy_bin) --
    if experiment_selection_method in ("kernel_weights", "energy_bin"):
        _logger.info("  Angular-Band Discrepancy (6d):")
        _logger.info(f"    USE_BAND_DISCREPANCY           = {use_band_discrepancy}")
        _logger.info(f"    MIN_POINTS_PER_BAND            = {min_points_per_band}")
        _logger.info(f"    MAX_TAU_FRACTION               = {max_tau_fraction}")
        _logger.info(f"    TAU_SMOOTHING_WINDOW           = {tau_smoothing_window}")
        _logger.info(f"    TAU_PRIOR_FLOOR                = {tau_prior_floor}")
        if tau_prior_floor:
            _logger.info(f"    TAU_PRIOR_MIN_EXPERIMENTS      = {tau_prior_min_experiments}")
            _logger.info(f"    TAU_PRIOR_PERCENTILE           = {tau_prior_percentile}")
        _logger.info(f"    RESCALE_UNC_BY_CHI2            = {RESCALE_UNC_BY_CHI2}")
        _logger.info(f"    ALLOW_SHRINK_UNC               = {ALLOW_SHRINK_UNC}")
        _logger.info("")

    # -- 6e. Per-Experiment Normalization (kernel_weights, energy_bin) --
    if experiment_selection_method in ("kernel_weights", "energy_bin"):
        _logger.info("  Per-Experiment Normalization (6e):")
        _logger.info(f"    NORMALIZATION_SIGMA             = {sigma_norm}")
        _logger.info(f"    NORM_DIST                      = {NORM_DIST}")
        _logger.info("")

    # -- 6f. Model Averaging (kernel_weights, energy_bin) --
    if experiment_selection_method in ("kernel_weights", "energy_bin"):
        _logger.info("  Model Averaging (6f):")
        _logger.info(f"    USE_MODEL_AVERAGING            = {use_model_averaging}")
        _logger.info(f"    MIN_DEGREE_FOR_AVERAGING       = {min_degree_for_averaging}")
        _logger.info(f"    USE_DEGREE_SAMPLING_IN_MC      = {USE_DEGREE_SAMPLING_IN_MC}")
        _logger.info("")

    # -- 6g. Energy Bin Specific (energy_bin only) --
    if experiment_selection_method == "energy_bin":
        _logger.info("  Energy Bin Method (6g):")
        _logger.info(f"    NORMALIZE_BY_N_POINTS          = {NORMALIZE_BY_N_POINTS}")
        _logger.info(f"    MAX_EXP_WEIGHT_FRAC_BIN        = {MAX_EXP_WEIGHT_FRAC_BIN}")
        _logger.info(f"    FREEZE_C0                      = {FREEZE_C0}")
        _logger.info("")

    # -- 6h. Energy Jitter (energy_bin only) --
    if experiment_selection_method == "energy_bin":
        _logger.info("  Energy Jitter (6h):")
        _logger.info(f"    USE_ENERGY_JITTER              = {USE_ENERGY_JITTER}")
        if USE_ENERGY_JITTER:
            _logger.info(f"    TOF_PARAMETERS_FILE            = {TOF_PARAMETERS_FILE}")
        _logger.info(f"    JITTER_N_SIGMA_CLIP            = {JITTER_N_SIGMA_CLIP}")
        _logger.info(f"    TRACK_BIN_JUMPS                = {TRACK_BIN_JUMPS}")
        _logger.info("")

    _logger.info(separator)
    _logger.info("")

    # Warning legend
    _logger.info("[WARNING LEGEND]")
    _logger.info("  The following warnings may appear during fitting:")
    _logger.info("")
    _logger.info("  - 'No EXFOR data': No experimental points found within energy tolerance")
    _logger.info("      -> Effect: Coefficients will be INTERPOLATED from neighboring bins")
    _logger.info("      -> Action: Check data coverage for this energy region")
    _logger.info("")
    _logger.info("  - 'Low N_eff=X.X (threshold: Y)': Effective sample size is low")
    _logger.info("      -> Cause: Few independent data points or highly unequal weights")
    _logger.info("      -> Effect: Larger statistical uncertainty in fitted coefficients")
    _logger.info("      -> Action: Consider widening σE cutoff or checking data availability")
    _logger.info("")
    _logger.info("  - 'Wide weight span (X.XXXX MeV = Y.Yxσ_E)': Energy spread of contributing data")
    _logger.info("      -> Cause: EXFOR data energies span a wide range around target energy")
    _logger.info("      -> Effect: Potential bias from energy-dependent shape changes")
    _logger.info("      -> Action: Review if energy resolution assumptions are appropriate")
    _logger.info("")
    if experiment_selection_method == "kernel_weights":
        _logger.info("  - 'Experiment capping applied': One experiment dominated the weights")
        _logger.info(f"      -> Cause: Experiment exceeded {max_experiment_weight_fraction*100:.0f}% of total weight")
        _logger.info("      -> Effect: Weight redistributed to prevent single-experiment bias")
        _logger.info("      -> Action: This is protective; review if capping is frequent")
        _logger.info("")
    _logger.info("  - 'INTERPOLATED from neighboring bins': No EXFOR data at this energy")
    _logger.info("      -> Cause: No experimental data available within energy tolerance")
    _logger.info("      -> Effect: Coefficients linearly interpolated from neighboring bins with data")
    _logger.info("      -> Note: Original ENDF coefficients are NEVER used (ensures independence)")
    _logger.info("")
    _logger.info("  - 'Below/Above EXFOR data range - extrapolating': Energy outside data coverage")
    _logger.info("      -> Cause: Energy bin is below/above the range of available EXFOR data")
    _logger.info("      -> Effect: Uses nearest neighbor's coefficients (no interpolation possible)")
    _logger.info("      -> Action: Expand energy range coverage or accept extrapolation uncertainty")
    _logger.info("")
    _logger.info(separator)

    # Validate inputs
    if not os.path.exists(endf_file):
        _logger.error(f"ENDF file not found: {endf_file}", console=True)
        return

    if not os.path.isdir(exfor_directory):
        _logger.error(f"EXFOR directory not found: {exfor_directory}", console=True)
        return

    # Step 1: Pre-load EXFOR data (using NEW API with database support)
    _logger.info("")
    _logger.info("[STEP 1] Pre-loading EXFOR data using NEW kika.exfor module")
    _logger.info(f"  Source: {exfor_source}")
    if exfor_source in ("database", "auto", "both"):
        _logger.info(f"  Database: {exfor_db_path or 'default'}")
    if exfor_directory:
        _logger.info(f"  JSON directory: {exfor_directory}")

    print(f"[INFO] Pre-loading EXFOR data (source={exfor_source})")
    t_exfor_start = time.time()

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
        t_exfor_elapsed = time.time() - t_exfor_start

        n_exfor_files = sum(len(entries) for entries in exfor_cache.values())
        _logger.info(f"  Loaded {n_exfor_files} EXFOR experiments at {len(sorted_exfor_energies)} unique energies")
        _logger.info(f"  EXFOR energy range: [{min(sorted_exfor_energies):.4f}, {max(sorted_exfor_energies):.4f}] MeV")
        _logger.info(f"  Pre-loading completed in {t_exfor_elapsed:.2f} seconds")
        print(f"[INFO] Loaded {n_exfor_files} EXFOR experiments in {t_exfor_elapsed:.1f}s")
    except Exception as e:
        _logger.error(f"Failed to load EXFOR data: {str(e)}", console=True)
        return

    # Step 2: Read ENDF and extract energy grid
    _logger.info("")
    _logger.info("[STEP 2] Reading ENDF file and extracting energy grid")

    try:
        endf = read_endf(endf_file)
        mf4 = endf.get_file(4)

        if mf4 is None:
            _logger.error("MF4 section not found in ENDF file", console=True)
            return

        mt_data = mf4.sections.get(mt_number)
        if mt_data is None:
            _logger.error(f"MT{mt_number} not found in MF4", console=True)
            return

        if not isinstance(mt_data, (MF4MTLegendre, MF4MTMixed)):
            _logger.error(f"MT{mt_number} is not Legendre or Mixed type (LTT={mt_data._ltt})", console=True)
            return

        energies_ev = np.array(mt_data.legendre_energies)
        original_coeffs = mt_data.legendre_coefficients

        _logger.info(f"  Found {len(energies_ev)} energy points in MF4/MT{mt_number}")

    except Exception as e:
        _logger.error(f"Failed to read ENDF file: {str(e)}", console=True)
        return

    # Step 3: Compute energy bins
    _logger.info("")
    _logger.info("[STEP 3] Computing energy bins with TOF-based resolution")

    energy_bins = compute_energy_bins_with_tof_resolution(
        energies_ev=energies_ev,
        energy_min_mev=energy_min_mev,
        energy_max_mev=energy_max_mev,
        delta_t_ns=delta_t_ns,
        flight_path_m=flight_path_m,
    )

    if not energy_bins:
        _logger.error(f"No energy points in range [{energy_min_mev}, {energy_max_mev}] MeV", console=True)
        return

    for bin_info in energy_bins:
        if bin_info.index < len(original_coeffs):
            bin_info.original_coeffs = list(original_coeffs[bin_info.index])

    _logger.info(f"  Processing {len(energy_bins)} energy bins")
    print(f"[INFO] Processing {len(energy_bins)} energy bins")

    # Step 4: Nominal fits
    _logger.info("")
    _logger.info("[STEP 4] Phase 1: Nominal fits")

    t_fit_start = time.time()

    nominal_results, global_system, c0_frozen_from_nominal = perform_nominal_fits(
        energy_bins=energy_bins,
        exfor_cache=exfor_cache,
        sorted_energies=sorted_exfor_energies,
        n_sigma=n_sigma_cutoff,
        max_degree=max_degree,
        select_degree=select_degree,
        ridge_lambda=ridge_lambda,
        m_proj_u=m_proj_u,
        m_targ_u=m_targ_u,
        use_band_discrepancy=use_band_discrepancy,
        min_points_per_band=min_points_per_band,
        max_tau_fraction=max_tau_fraction,
        tau_smoothing_window=tau_smoothing_window,
        min_kernel_weight_fraction=min_kernel_weight_fraction,
        max_experiment_weight_fraction=max_experiment_weight_fraction,
        n_eff_warning_threshold=n_eff_warning_threshold,
        weight_span_warning_ratio=weight_span_warning_ratio,
        experiment_selection_method=experiment_selection_method,
        min_degree_for_averaging=min_degree_for_averaging,
        delta_t_ns=delta_t_ns,
        flight_path_m=flight_path_m,
        tikhonov_lambda=tikhonov_lambda,
        l_dependent_power=L_DEPENDENT_POWER,
        skip_c0_regularization=SKIP_C0_REGULARIZATION,
        min_weight_sum_threshold=MIN_WEIGHT_SUM_THRESHOLD,
        shape_only=GLOBAL_CONV_SHAPE_ONLY,
        exclude_experiments=exclude_experiments,
        min_relative_uncertainty=min_relative_uncertainty,
        tau_prior_floor=tau_prior_floor,
        tau_prior_min_experiments=tau_prior_min_experiments,
        tau_prior_percentile=tau_prior_percentile,
        logger=_logger,
    )

    t_nominal_elapsed = time.time() - t_fit_start
    n_with_data = sum(1 for nr in nominal_results if nr.has_data)
    _logger.info(f"  Nominal fits completed in {t_nominal_elapsed:.2f}s")
    _logger.info(f"  Bins with EXFOR data: {n_with_data}/{len(nominal_results)}")
    print(f"[INFO] Nominal fits completed ({n_with_data}/{len(nominal_results)} with data)")

    # Step 4b: Interpolate missing bins (NEVER use original ENDF coefficients)
    n_missing = len(nominal_results) - n_with_data
    if n_missing > 0:
        _logger.info("")
        _logger.info("[STEP 4b] Interpolating missing energy bins")
        _logger.info(f"  Bins needing interpolation: {n_missing}")

        nominal_results = interpolate_missing_nominal_fits(
            nominal_results=nominal_results,
            logger=_logger,
        )

        # Count results after interpolation
        n_with_data_after = sum(1 for nr in nominal_results if nr.has_data)
        n_interpolated = sum(1 for nr in nominal_results if nr.interpolated)
        _logger.info(f"  After interpolation: {n_with_data_after}/{len(nominal_results)} bins have coefficients")
        _logger.info(f"  ({n_interpolated} interpolated, {n_with_data_after - n_interpolated} from EXFOR)")
        _logger.info(f"  IMPORTANT: Original ENDF coefficients are NEVER used as fallback")
        _logger.info(f"  (This ensures an independent evaluation based solely on EXFOR data)")
        print(f"[INFO] Interpolated {n_interpolated} bins without EXFOR data")

    # Log experiment summary (quick overview of all data sources used)
    log_experiments_summary(nominal_results, logger=_logger)

    # Step 5: MC sampling
    # Three modes available:
    # 1. Global convolution MC (for global_convolution method): All energies sampled jointly
    # 2. Energy jitter MC (for energy_bin method): Cross-bin correlation via energy perturbation
    # 3. Per-bin independent MC: Original method with independent sampling per bin
    _logger.info("")
    _logger.info("[STEP 5] Phase 2: MC sampling")
    _logger.info(f"  Generating {n_samples} MC samples")

    # GLOBAL CONVOLUTION MC (Improvement 3.1 - Fix MC Sampling)
    if experiment_selection_method == "global_convolution" and global_system is not None:
        # Shape-only MC (Improvement 3.4): c0 frozen, only c1..cL sampled
        if GLOBAL_CONV_SHAPE_ONLY and c0_frozen_from_nominal is not None:
            _logger.info(f"  Method: Global convolution MC (shape-only, c0 frozen)")
            _logger.info(f"  Experiment normalization σ: {sigma_norm}")
            _logger.info(f"  Tikhonov λ: {tikhonov_lambda}")
            _logger.info(f"  ℓ-dependent power: {L_DEPENDENT_POWER}")

            all_samples_raw, mc_diagnostics = sample_global_convolution_mc_shape_only(
                system=global_system,
                c0_frozen=c0_frozen_from_nominal,
                n_samples=n_samples,
                sigma_norm=sigma_norm,
                tikhonov_lambda=tikhonov_lambda,
                l_dependent_power=L_DEPENDENT_POWER,
                seed=base_seed,
                logger=_logger,
                apply_positivity_projection=apply_positivity_projection,
                positivity_check_points=positivity_check_points,
            )
        else:
            _logger.info(f"  Method: Global convolution MC (all energies sampled jointly)")
            _logger.info(f"  Experiment normalization σ: {sigma_norm}")

            # Generate MC samples using the global system
            all_samples_raw, mc_diagnostics = sample_global_convolution_mc(
                system=global_system,
                n_samples=n_samples,
                sigma_norm=sigma_norm,
                seed=base_seed,
                logger=_logger,
                apply_positivity_projection=apply_positivity_projection,
                positivity_check_points=positivity_check_points,
            )

        # Convert to the expected format: Dict[sample_idx, Dict[energy_idx, coeffs]]
        all_samples = {s_idx: {} for s_idx in range(n_samples)}
        n_coeffs = global_system.n_coeffs

        for s_idx in range(n_samples):
            coeffs_vec = all_samples_raw[s_idx, :]
            for bin_info in energy_bins:
                param_start = global_system.energy_idx_to_param_start[bin_info.index]
                raw_coeffs = coeffs_vec[param_start:param_start + n_coeffs]
                endf_coeffs = endf_normalize_legendre_coeffs(raw_coeffs, include_a0=False)
                all_samples[s_idx][bin_info.index] = endf_coeffs

        n_sampled = len([nr for nr in nominal_results if nr.has_data and not nr.interpolated])
        n_interpolated_used = len([nr for nr in nominal_results if nr.has_data and nr.interpolated])

        _logger.info(f"  MC sampling complete: {n_sampled} energies with data, {n_interpolated_used} interpolated")
        _logger.info(f"  Mean coeff std/mean ratio: {mc_diagnostics.get('coeffs_mean_std_ratio', 0):.4f}")

        # Shape-only specific diagnostics
        if GLOBAL_CONV_SHAPE_ONLY and mc_diagnostics.get('shape_only'):
            _logger.info(f"  c0 max std across samples: {mc_diagnostics.get('c0_max_std', 0):.2e} (should be ~0)")

        # Validation: Check that samples differ (MF34 should be non-zero)
        sample_stds = []
        for bin_info in energy_bins:
            if bin_info.index in all_samples[0]:
                coeffs_across_samples = np.array([all_samples[s][bin_info.index] for s in range(n_samples)])
                sample_std = np.std(coeffs_across_samples, axis=0)
                sample_stds.extend(sample_std)
        mean_sample_std = np.mean(sample_stds) if sample_stds else 0
        if mean_sample_std < 1e-10:
            _logger.warning("  WARNING: MC samples appear identical (mean std ≈ 0)")
            _logger.warning("  This may result in zero covariance/MF34. Check system setup.")
        else:
            _logger.info(f"  Sample diversity check: mean coeff std = {mean_sample_std:.6f}")

    # Check if energy jitter is enabled and applicable
    elif USE_ENERGY_JITTER and experiment_selection_method == "energy_bin" and TOF_PARAMETERS_FILE is not None:
        use_jitter = True
        _logger.info(f"  Method: Energy jitter MC (Improvement 1.4)")
        _logger.info(f"  TOF parameters file: {TOF_PARAMETERS_FILE}")
        _logger.info(f"  Jitter clipping: ±{JITTER_N_SIGMA_CLIP}σ")

        # Load TOF parameters
        try:
            tof_params_cache = load_tof_parameters_file(TOF_PARAMETERS_FILE)
            _logger.info(f"  Loaded TOF parameters for {len(tof_params_cache)} experiments")
        except FileNotFoundError:
            _logger.warning(f"  TOF parameters file not found: {TOF_PARAMETERS_FILE}")
            _logger.warning(f"  Falling back to per-bin independent MC")
            use_jitter = False
            tof_params_cache = {}
        except Exception as e:
            _logger.warning(f"  Failed to load TOF parameters: {e}")
            _logger.warning(f"  Falling back to per-bin independent MC")
            use_jitter = False
            tof_params_cache = {}

        if use_jitter:
            # Precompute dataset energy info (σE for each dataset)
            dataset_info_by_bin = precompute_dataset_energy_info(
                nominal_results=nominal_results,
                tof_params_cache=tof_params_cache,
                energy_bins=energy_bins,
                default_flight_path_m=flight_path_m,
                default_time_resolution_ns=delta_t_ns,
            )

            # Log summary of TOF parameter sources
            all_subentries = []
            for datasets in dataset_info_by_bin.values():
                for ds in datasets:
                    all_subentries.append(f"{ds.entry}{ds.subentry}")

            if all_subentries:
                tof_summary = summarize_tof_parameters(
                    tof_params_cache, all_subentries,
                    default_flight_path_m=flight_path_m,
                    default_time_resolution_ns=delta_t_ns,
                )
                _logger.info(f"  TOF params from file: {tof_summary['n_from_file']}, using defaults: {tof_summary['n_default']}")

            # Run MC with energy jitter
            all_samples, bin_jump_diag = run_mc_with_energy_jitter(
                nominal_results=nominal_results,
                energy_bins=energy_bins,
                dataset_info_by_bin=dataset_info_by_bin,
                exfor_cache=exfor_cache,
                sorted_energies=sorted_exfor_energies,
                n_samples=n_samples,
                base_seed=base_seed,
                max_degree=max_degree,
                ridge_lambda=ridge_lambda,
                m_proj_u=m_proj_u,
                m_targ_u=m_targ_u,
                use_band_discrepancy=use_band_discrepancy,
                min_points_per_band=min_points_per_band,
                max_tau_fraction=max_tau_fraction,
                jitter_n_sigma_clip=JITTER_N_SIGMA_CLIP,
                track_bin_jumps=TRACK_BIN_JUMPS,
                min_relative_uncertainty=min_relative_uncertainty,
                freeze_c0=FREEZE_C0,
                sigma_norm=NORMALIZATION_SIGMA,
                norm_dist=NORM_DIST,
                normalize_by_n_points=NORMALIZE_BY_N_POINTS,
                max_experiment_weight_fraction=MAX_EXP_WEIGHT_FRAC_BIN,
                n_procs=N_PROCS,
                logger=_logger,
                apply_positivity_projection=apply_positivity_projection,
                positivity_check_points=positivity_check_points,
            )

            # Log bin jump diagnostics
            if TRACK_BIN_JUMPS:
                log_bin_jump_diagnostics(bin_jump_diag, energy_bins, logger=_logger)

            n_sampled = sum(1 for nr in nominal_results if nr.has_data and not nr.interpolated)
            n_interpolated_used = sum(1 for nr in nominal_results if nr.has_data and nr.interpolated)
            _logger.info(f"  MC sampled with jitter: {n_sampled} bins, interpolated: {n_interpolated_used} bins")

        # If use_jitter was set to False during loading, fall through to per-bin MC
        if not use_jitter:
            # Fallback to per-bin MC if jitter loading failed
            _logger.warning("  " + "=" * 60)
            _logger.warning(f"  Method: Per-bin independent MC (jitter fallback)")
            if N_PROCS > 1:
                _logger.info(f"  Using {N_PROCS} parallel processes over bins")

            # Build args list for _mc_one_bin (no degree sampling in fallback)
            bin_args_list = []
            for nr in nominal_results:
                if not nr.has_data:
                    continue
                mc_df = nr.exfor_df_mc if nr.exfor_df_mc is not None else nr.exfor_df
                mc_weights = nr.kernel_weights_mc if nr.kernel_weights_mc is not None else nr.kernel_weights
                bin_args_list.append((
                    nr.energy_index,
                    nr.frozen_degree,
                    nr.nominal_coeffs,
                    nr.interpolated,
                    mc_df,
                    mc_weights,
                    None,  # no degree_weights in fallback
                    n_samples,
                    base_seed,
                    max_degree,
                    ridge_lambda,
                    RIDGE_POWER,
                    DF_METHOD,
                    use_band_discrepancy,
                    min_points_per_band,
                    max_tau_fraction,
                    False,  # USE_DEGREE_SAMPLING_IN_MC disabled in fallback
                    RESCALE_UNC_BY_CHI2,
                    ALLOW_SHRINK_UNC,
                    FREEZE_C0,
                    NORMALIZATION_SIGMA,
                    NORM_DIST,
                    apply_positivity_projection,
                    positivity_check_points,
                ))

            if N_PROCS > 1:
                with Pool(N_PROCS) as pool:
                    bin_results = pool.map(_mc_one_bin, bin_args_list)
            else:
                bin_results = [_mc_one_bin(a) for a in bin_args_list]

            all_samples = {s_idx: {} for s_idx in range(n_samples)}
            n_sampled = 0
            n_interpolated_used = 0

            for energy_idx, is_interpolated, results_by_sample, success in bin_results:
                for s_idx, endf_coeffs in results_by_sample.items():
                    all_samples[s_idx][energy_idx] = endf_coeffs
                if is_interpolated:
                    n_interpolated_used += 1
                elif success:
                    n_sampled += 1

            _logger.info(f"  MC sampled: {n_sampled} bins, interpolated: {n_interpolated_used} bins")

    else:
        # ORIGINAL METHOD: Per-bin independent MC sampling
        _logger.info(f"  Method: Per-bin independent MC")
        if N_PROCS > 1:
            _logger.info(f"  Using {N_PROCS} parallel processes over bins")

        # Build args list for _mc_one_bin
        bin_args_list = []
        for nr in nominal_results:
            if not nr.has_data:
                continue
            mc_df = nr.exfor_df_mc if nr.exfor_df_mc is not None else nr.exfor_df
            mc_weights = nr.kernel_weights_mc if nr.kernel_weights_mc is not None else nr.kernel_weights
            bin_args_list.append((
                nr.energy_index,
                nr.frozen_degree,
                nr.nominal_coeffs,
                nr.interpolated,
                mc_df,
                mc_weights,
                nr.degree_weights,
                n_samples,
                base_seed,
                max_degree,
                ridge_lambda,
                RIDGE_POWER,
                DF_METHOD,
                use_band_discrepancy,
                min_points_per_band,
                max_tau_fraction,
                USE_DEGREE_SAMPLING_IN_MC,
                RESCALE_UNC_BY_CHI2,
                ALLOW_SHRINK_UNC,
                FREEZE_C0,
                NORMALIZATION_SIGMA,
                NORM_DIST,
                apply_positivity_projection,
                positivity_check_points,
            ))

        # Run per-bin MC (parallel or sequential)
        if N_PROCS > 1:
            with Pool(N_PROCS) as pool:
                bin_results = pool.map(_mc_one_bin, bin_args_list)
        else:
            bin_results = [_mc_one_bin(a) for a in bin_args_list]

        # Assemble results into all_samples
        all_samples = {s_idx: {} for s_idx in range(n_samples)}
        n_sampled = 0
        n_interpolated_used = 0

        for energy_idx, is_interpolated, results_by_sample, success in bin_results:
            for s_idx, endf_coeffs in results_by_sample.items():
                all_samples[s_idx][energy_idx] = endf_coeffs
            if is_interpolated:
                n_interpolated_used += 1
            elif success:
                n_sampled += 1

        _logger.info(f"  MC sampled: {n_sampled} bins, interpolated: {n_interpolated_used} bins")

    # Step 6: Save coefficients
    _logger.info("")
    _logger.info("[STEP 6] Saving Legendre coefficients")

    try:
        parquet_file = save_all_legendre_coefficients(
            nominal_results=nominal_results,
            all_samples=all_samples,
            output_dir=str(output_path),
            max_degree=max_degree,
        )
        _logger.info(f"  Saved to: {parquet_file}")
    except Exception as e:
        _logger.error(f"Failed to save coefficients: {str(e)}", console=True)
        parquet_file = None

    # Step 7: Covariance
    cov_matrix = None
    energy_indices = [nr.energy_index for nr in nominal_results if nr.has_data]

    if generate_covariance:
        _logger.info("")
        _logger.info("[STEP 7] Computing covariance matrix")

        cov_matrix, corr_matrix, param_labels = compute_covariance_from_samples(
            all_samples=all_samples,
            energy_indices=energy_indices,
            max_order=max_degree,
        )

        # Validate covariance: check that diagonal values are non-trivial
        diag = np.diag(cov_matrix)
        diag_nonzero = diag[diag > 0]
        if len(diag_nonzero) > 0:
            min_diag = np.min(diag_nonzero)
            max_diag = np.max(diag_nonzero)
            mean_diag = np.mean(diag_nonzero)
            # Compute standard deviation of normalized Legendre coefficients
            # Note: The coefficients a_l = (c_l/c0)/(2l+1) are already normalized
            # by the total cross section, making them dimensionless. Their std dev
            # is thus inherently a fractional quantity (not a "relative uncertainty"
            # in the traditional sense of std/mean).
            mean_coeff_std = np.sqrt(mean_diag)
            _logger.info(f"  Diagonal stats: min={min_diag:.2e}, max={max_diag:.2e}, mean={mean_diag:.2e}")
            _logger.info(f"  Mean Legendre coeff std: {mean_coeff_std:.4f}")
            _logger.info(f"  (As fraction of unity: {mean_coeff_std*100:.2f}% - coeffs are normalized)")

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
            _logger.warning("  WARNING: No positive diagonal elements in covariance matrix!")

        _logger.info(f"  Covariance matrix shape: {cov_matrix.shape}")

        # Step 7a: Apply covariance cap (Layer 1) if enabled
        if apply_covariance_cap:
            _logger.info("")
            _logger.info("[STEP 7a] Applying covariance diagonal cap (Layer 1)")

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
                    f"  Covariance capping was applied to {n_capped} entries. "
                    f"Set APPLY_COVARIANCE_CAP=False and re-run to obtain the uncapped covariance matrix.",
                    console=True,
                )
            else:
                _logger.info("  No entries exceeded the cap — no capping applied, output is unchanged.")

            # Recompute correlation from capped covariance
            std_capped = np.sqrt(np.maximum(np.diag(cov_matrix), 0.0))
            std_capped[std_capped == 0] = 1.0
            corr_matrix = cov_matrix / np.outer(std_capped, std_capped)

        # Save final covariance (capped if capping was applied, raw otherwise)
        np.save(output_path / "legendre_covariance.npy", cov_matrix)
        if save_correlation_matrices:
            np.save(output_path / "legendre_correlation.npy", corr_matrix)

    # Step 7b: Multigroup covariance (optional)
    multigroup_result = None
    multigroup_failure_reason = None
    if generate_multigroup_covariance:
        if not generate_covariance:
            multigroup_failure_reason = "generate_covariance is disabled"
        elif cov_matrix is None:
            multigroup_failure_reason = "covariance matrix is None (computation may have failed)"
        else:
            _logger.info("")
            _logger.info("[STEP 7b] Computing adaptive multigroup covariance")
            _logger.info(f"  Using l=1 correlation for grouping (same grid for all orders)")

            try:
                multigroup_result = perform_adaptive_multigroup_collapse(
                    cov_matrix=cov_matrix,
                    corr_matrix=corr_matrix,
                    nominal_results=nominal_results,
                    energy_bins=energy_bins,
                    max_order=max_degree,
                    rho_min=multigroup_rho_min,
                    sigma_ratio_max=multigroup_sigma_ratio_max,
                    min_width_factor=multigroup_min_width_factor,
                    variance_percentile=multigroup_variance_percentile,
                    logger=_logger,
                    apply_covariance_cap=apply_covariance_cap,
                    max_relative_std_cap=max_relative_std_cap,
                )

                # Log and save results
                n_fine = len([nr for nr in nominal_results if not nr.interpolated and nr.has_data])
                n_groups = len(multigroup_result.groups)
                _logger.info(f"  Fine bins: {n_fine} -> Multigroups: {n_groups}")
                _logger.info(f"  Compression: {n_fine/n_groups:.1f}x")

                np.save(output_path / "legendre_covariance_multigroup.npy",
                        multigroup_result.cov_grouped)
                if save_correlation_matrices:
                    np.save(output_path / "legendre_correlation_multigroup.npy",
                            multigroup_result.corr_grouped)
                np.save(output_path / "multigroup_boundaries_ev.npy",
                        multigroup_result.group_boundaries_ev)
                np.save(output_path / "multigroup_mean_coeffs.npy",
                        multigroup_result.mean_grouped)
                _logger.info(f"  Saved multigroup covariance and boundaries")

            except Exception as e:
                multigroup_failure_reason = f"{str(e)}\n{traceback.format_exc()}"
                _logger.error(f"Failed to compute multigroup covariance: {str(e)}", console=True)
                _logger.error(f"  Traceback:\n{traceback.format_exc()}")
                multigroup_result = None

    # Step 8: Write ENDF files
    average_file = None
    if generate_mc_mean_endf:
        _logger.info("")
        _logger.info("[STEP 8] Writing average ENDF file (MC mean)")

        try:
            average_file = write_average_endf(
                original_endf_file=endf_file,
                mt_number=mt_number,
                nominal_results=nominal_results,
                all_samples=all_samples,
                output_dir=str(output_path),
            )
            _logger.info(f"  Average ENDF: {average_file}")
        except Exception as e:
            _logger.error(f"Failed to write average ENDF: {str(e)}", console=True)

    nominal_file = None
    if generate_nominal_endf:
        _logger.info("")
        _logger.info("[STEP 8b] Writing nominal ENDF file")

        try:
            nominal_file = write_nominal_endf(
                original_endf_file=endf_file,
                mt_number=mt_number,
                nominal_results=nominal_results,
                output_dir=str(output_path),
            )
            _logger.info(f"  Nominal ENDF: {nominal_file}")
        except Exception as e:
            _logger.error(f"Failed to write nominal ENDF: {str(e)}", console=True)

    # Step 9: Write sample files
    output_files = []
    if generate_samples_endf:
        _logger.info("")
        _logger.info("[STEP 9] Writing ENDF sample files")

        output_files = write_endf_samples_batch(
            original_endf_file=endf_file,
            mt_number=mt_number,
            all_samples=all_samples,
            output_dir=str(output_path),
            n_procs=N_PROCS,
        )
        _logger.info(f"  Written {len(output_files)} sample files")

    # Step 10: MF34 (using library functions)
    if generate_mf34 and generate_covariance and cov_matrix is not None:
        _logger.info("")
        _logger.info("[STEP 10] Writing MF34 using kika.endf.writers")
        _logger.info(f"  Covariance type: {mf34_covariance_type}")

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

            # Read original MF34 from the reference ENDF file (if present)
            original_mf34_mt = None
            try:
                endf_for_mf34 = read_endf(endf_file, mf_numbers=34)
                mf34_file = endf_for_mf34.get_file(34)
                if mf34_file is not None:
                    original_mf34_mt = mf34_file.sections.get(mt_number)
                    if original_mf34_mt is not None:
                        _logger.info(f"  Original MF34 found for MT{mt_number} — will merge with pipeline MF34")
            except Exception:
                pass  # No original MF34 available

            # Helper to merge pipeline MF34 with original if available
            def _maybe_merge(pipeline_mf34_obj, pipe_grid_ev):
                if original_mf34_mt is not None:
                    pipe_emin = float(pipe_grid_ev[0])
                    pipe_emax = float(pipe_grid_ev[-1])
                    return merge_mf34(
                        original_mf34=original_mf34_mt,
                        pipeline_mf34=pipeline_mf34_obj,
                        pipeline_energy_min_ev=pipe_emin,
                        pipeline_energy_max_ev=pipe_emax,
                    )
                return pipeline_mf34_obj

            # Write fine-grid MF34 if requested
            if mf34_covariance_type in ("fine", "both"):
                processed_energies_ev = np.array([all_energies_ev[i] for i in energy_indices])
                if energy_indices[-1] + 1 < len(all_energies_ev):
                    energy_grid_ev = np.append(processed_energies_ev, all_energies_ev[energy_indices[-1] + 1])
                else:
                    if len(processed_energies_ev) > 1:
                        delta = processed_energies_ev[-1] - processed_energies_ev[-2]
                        energy_grid_ev = np.append(processed_energies_ev, processed_energies_ev[-1] + delta)
                    else:
                        energy_grid_ev = np.append(processed_energies_ev, processed_energies_ev[-1] * 1.1)

                mf34_fine = create_mf34_from_covariance(
                    cov_matrix=cov_matrix,
                    energy_grid_ev=energy_grid_ev,
                    max_order=max_degree,
                    za=za,
                    awr=awr,
                    mat=mat,
                    mt=mt_number,
                )

                mf34_fine = _maybe_merge(mf34_fine, energy_grid_ev)

                if average_file:
                    write_mf34_to_file(average_file, mf34_fine, average_file)
                    _logger.info(f"  Fine MF34 added to average: {average_file}")

                if nominal_file:
                    write_mf34_to_file(nominal_file, mf34_fine, nominal_file)
                    _logger.info(f"  Fine MF34 added to nominal: {nominal_file}")

            # Write multigroup MF34 if requested and available
            if mf34_covariance_type in ("multigroup", "both") and multigroup_result is not None:
                mf34_mg = create_mf34_from_covariance(
                    cov_matrix=multigroup_result.cov_grouped,
                    energy_grid_ev=multigroup_result.group_boundaries_ev,
                    max_order=max_degree,
                    za=za,
                    awr=awr,
                    mat=mat,
                    mt=mt_number,
                )

                mf34_mg = _maybe_merge(mf34_mg, multigroup_result.group_boundaries_ev)

                # For multigroup, write to separate files with _mg suffix
                if average_file:
                    mg_avg_file = average_file.replace('.txt', '_mg.endf').replace('.endf', '_mg.endf')
                    if mg_avg_file == average_file:
                        mg_avg_file = average_file + '_mg'
                    # Copy average file and add multigroup MF34
                    import shutil
                    shutil.copy(average_file, mg_avg_file)
                    write_mf34_to_file(mg_avg_file, mf34_mg, mg_avg_file)
                    _logger.info(f"  Multigroup MF34 written to: {mg_avg_file}")

                if nominal_file:
                    mg_nom_file = nominal_file.replace('.txt', '_mg.endf').replace('.endf', '_mg.endf')
                    if mg_nom_file == nominal_file:
                        mg_nom_file = nominal_file + '_mg'
                    import shutil
                    shutil.copy(nominal_file, mg_nom_file)
                    write_mf34_to_file(mg_nom_file, mf34_mg, mg_nom_file)
                    _logger.info(f"  Multigroup MF34 written to: {mg_nom_file}")

            elif mf34_covariance_type in ("multigroup", "both") and multigroup_result is None:
                if multigroup_failure_reason:
                    _logger.warning(f"  Multigroup covariance requested but failed: {multigroup_failure_reason}", console=True)
                elif not generate_multigroup_covariance:
                    _logger.warning("  Multigroup covariance requested but not computed (enable GENERATE_MULTIGROUP_COVARIANCE)")
                else:
                    _logger.warning("  Multigroup covariance requested but not computed (unknown reason)")

        except Exception as e:
            _logger.error(f"Failed to write MF34: {str(e)}", console=True)

    # Summary
    total_time = time.time() - t_exfor_start
    _logger.info("")
    _logger.info(separator)
    _logger.info("[SUMMARY]")
    _logger.info(f"  Total execution time: {total_time:.2f}s")
    _logger.info(f"  API Version: kika.exfor (v2)")
    _logger.info(separator)

    print(f"\n[INFO] Completed! Output directory: {output_path}")
    print(f"[INFO] Total time: {total_time:.1f}s")

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
        delta_t_ns=DELTA_T_NS,
        flight_path_m=FLIGHT_PATH_M,
        n_sigma_cutoff=N_SIGMA_CUTOFF,
        use_band_discrepancy=USE_BAND_DISCREPANCY,
        min_points_per_band=MIN_POINTS_PER_BAND,
        max_tau_fraction=MAX_TAU_FRACTION,
        tau_smoothing_window=TAU_SMOOTHING_WINDOW,
        sigma_norm=NORMALIZATION_SIGMA,
        use_model_averaging=USE_MODEL_AVERAGING,
        min_degree_for_averaging=MIN_DEGREE_FOR_AVERAGING,
        min_kernel_weight_fraction=MIN_KERNEL_WEIGHT_FRACTION,
        max_experiment_weight_fraction=MAX_EXPERIMENT_WEIGHT_FRACTION,
        n_eff_warning_threshold=N_EFF_WARNING_THRESHOLD,
        weight_span_warning_ratio=WEIGHT_SPAN_WARNING_RATIO,
        experiment_selection_method=EXPERIMENT_SELECTION_METHOD,
        n_procs=N_PROCS,
        base_seed=BASE_SEED,
        generate_nominal_endf=GENERATE_NOMINAL_ENDF,
        generate_mc_mean_endf=GENERATE_MC_MEAN_ENDF,
        generate_samples_endf=GENERATE_SAMPLES_ENDF,
        generate_covariance=GENERATE_COVARIANCE,
        generate_mf34=GENERATE_MF34,
        # Multigroup covariance options
        generate_multigroup_covariance=GENERATE_MULTIGROUP_COVARIANCE,
        multigroup_rho_min=MULTIGROUP_RHO_MIN,
        multigroup_sigma_ratio_max=MULTIGROUP_SIGMA_RATIO_MAX,
        multigroup_min_width_factor=MULTIGROUP_MIN_WIDTH_FACTOR,
        multigroup_variance_percentile=MULTIGROUP_VARIANCE_PERCENTILE,
        mf34_covariance_type=MF34_COVARIANCE_TYPE,
        # Database configuration
        exfor_db_path=EXFOR_DB_PATH,
        exfor_source=EXFOR_SOURCE,
        target_zaid=TARGET_ZAIDS,
        target_projectile=TARGET_PROJECTILE,
        supplementary_json_files=SUPPLEMENTARY_JSON_FILES,
        # Experiment exclusion and uncertainty floor
        exclude_experiments=EXCLUDE_EXPERIMENTS,
        min_relative_uncertainty=MIN_RELATIVE_UNCERTAINTY,
        # Covariance cap (Layer 1) and positivity projection (Layer 2)
        apply_covariance_cap=APPLY_COVARIANCE_CAP,
        max_relative_std_cap=MAX_RELATIVE_STD_CAP,
        apply_positivity_projection=APPLY_POSITIVITY_PROJECTION,
        positivity_check_points=POSITIVITY_CHECK_POINTS,
        # File output options
        save_correlation_matrices=SAVE_CORRELATION_MATRICES,
    )
