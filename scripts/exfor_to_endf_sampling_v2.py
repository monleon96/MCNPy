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
    # Energy binning
    compute_energy_bins_with_tof_resolution,
    # EXFOR data conversion (new API -> legacy format)
    build_exfor_cache_from_objects,
    # EXFOR filtering
    filter_exfor_with_energy_bin,
    # Covariance
    compute_covariance_from_samples,
    extract_ll_prime_correlations,
    build_gaussian_correlation_covariance,
    generate_cholesky_samples,
    cap_covariance_relative_uncertainty,
    save_all_legendre_coefficients,
    # Kernel-weight MC (new)
    precompute_overlap_weights,
    run_mc_with_kernel_weights,
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
)

import time


# =============================================================================
# CONFIGURATION PARAMETERS - MODIFY THESE BEFORE RUNNING
# =============================================================================

# -----------------------------------------------------------------------------
# 1. INPUT/OUTPUT PATHS
# -----------------------------------------------------------------------------
# Reference ENDF file (source of energy grid and original Legendre coefficients)
ENDF_FILE = "/share_snc/snc/JuanMonleon/jeff40_with_MF4_from_jeff33/26-Fe-56g.txt"

# Optional: separate ENDF file for MF34 covariance data to merge with pipeline MF34.
# Set to None to use ENDF_FILE. Useful when ENDF_FILE lacks MF34 or has a
# parser-incompatible MF34 (e.g. mixed LB types).
MF34_SOURCE_FILE = None

# EXFOR JSON directory (for source="json" or "auto")
EXFOR_DIRECTORY = "/share_snc/snc/JuanMonleon/EXFOR/data_v1/"

# X4Pro SQLite database path (for source="database" or "auto")
# Set to None to use KIKA_X4PRO_DB_PATH env variable or builtin default
EXFOR_DB_PATH = '/share_snc/snc/JuanMonleon/EXFOR/x4_iron_angular.db'

# Output directory (all generated files go here)
OUTPUT_DIR = "/SCRATCH/users/monleon-de-la-jan/MCNPy_LIB/EXFOR_FIT_GAUSSCORR/"

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
GENERATE_MC_MEAN_ENDF = False                     # MC mean coefficients ENDF
GENERATE_SAMPLES_ENDF = False                    # Individual MC sample ENDFs (Pipeline B generates final samples)
SAVE_COVARIANCE_FILES = True                    # Save covariance/correlation .npy files
N_SAMPLES = 100                                   # Number of MC samples

# -----------------------------------------------------------------------------
# 3b. MULTIGROUP COVARIANCE OPTIONS
# -----------------------------------------------------------------------------
GENERATE_MULTIGROUP_COVARIANCE = True           # Enable adaptive multigroup collapse
MULTIGROUP_RHO_MIN = 0.90                        # Min correlation to merge (0.85-0.95)
MULTIGROUP_SIGMA_RATIO_MAX = 2.0                 # Max sigma ratio within group (1.5-2.0)
MULTIGROUP_MIN_WIDTH_FACTOR = 2.0                # Group width >= k * median(sigma_E)
MF34_COVARIANCE_TYPE = "both"                    # "fine", "multigroup", or "both"
USE_ORIGINAL_MF34_GRID = False                   # Force multigroup grid from original MF34
MERGE_ORIGINAL_MF34 = True                      # Merge pipeline MF34 with original (full range) or pipeline-only

# Variance percentile for multigroup collapse
# Controls how diagonal variances are scaled after averaging:
# - 50 = median of fine variances in group (typical)
# - 80-90 = conservative but not extreme
# - 100 = maximum fine variance in group (most conservative)
MULTIGROUP_VARIANCE_PERCENTILE = 66.67

# --- Layer 1: Covariance diagonal cap ---
# Caps excessive relative uncertainties in the covariance matrix.
# Set APPLY_COVARIANCE_CAP = False to disable (preserves existing behavior).
APPLY_COVARIANCE_CAP = False
MAX_RELATIVE_STD_CAP = 1.0  # 100% relative uncertainty cap

# --- File output options ---
SAVE_CORRELATION_MATRICES = False       # Save correlation alongside covariance

# --- Layer 2: Positivity-constrained projection ---
# Projects MC samples to ensure non-negative angular distributions.
# Set APPLY_POSITIVITY_PROJECTION = False to disable (preserves existing behavior).
APPLY_POSITIVITY_PROJECTION = True
POSITIVITY_CHECK_POINTS = 101  # Number of mu points in [-1, 1]

# -----------------------------------------------------------------------------
# 3c. ACE GENERATION OPTIONS
# -----------------------------------------------------------------------------
GENERATE_ACE = False                              # Process ENDF samples → ACE via NJOY
ACE_TEMPERATURES = [293.6]                         # Temperature(s) in Kelvin
ACE_NJOY_EXE = "/soft_snc/NJOY/2016.78/bin/njoy"
ACE_LIBRARY_NAME = "jeff40"                        # Library name (e.g., 'endfb81', 'jeff40')
ACE_NJOY_VERSION = "NJOY 2016.78"                 # NJOY version string
ACE_XSDIR_FILE = "/share_snc/snc/JuanMonleon/xsdir_MCNPy/xsdir40-irdff2"      # Master xsdir to update (None = per-sample only)
ACE_SKIP_EXISTING = False                          # Skip samples with existing ACE files

# -----------------------------------------------------------------------------
# 3d. UNIFIED MF34 SAMPLING (Pipeline B)
# -----------------------------------------------------------------------------
GENERATE_SAMPLES_FROM_MF34 = True                  # Generate samples via perturb_ENDF_files
SAMPLING_RESOLUTION = "multigroup"                 # "fine" | "multigroup" (grid controlled by USE_ORIGINAL_MF34_GRID)
SAMPLING_SPACE = "linear"                          # "linear" or "log"
SAMPLING_DECOMPOSITION = "svd"                     # "svd", "cholesky", "eigen", "pca"
SAMPLING_METHOD = "random"                          # "sobol", "lhs", "random"

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
RIDGE_LAMBDA = 1e-4                              # Ridge regularization parameter
RIDGE_POWER = 4                                  # Power for ridge penalty (l^ridge_power)
DF_METHOD = "hat"                                # Degrees of freedom method: "hat" or "naive"

# Processing options
N_PROCS = 24                                      # Parallel processes (1 = sequential)
BASE_SEED = 42                                   # Random seed for reproducibility
N_EFF_WARNING_THRESHOLD = 5.0                    # Warning if effective sample size < threshold

# --- Experiment Exclusion and Uncertainty Floor ---
# Experiments to exclude from fitting (e.g., experiments with known issues)
# Accepts formats: "20743" (all subentries), "20743002", or "20743/002"
EXCLUDE_EXPERIMENTS = ["20743002", "32246002"]  
# - "20743002" - Cierjacks (1978)
# - "32246002" - Tostkii (1957)


# Minimum relative uncertainty floor (prevents unrealistically small errors from dominating)
# Set to 0.0 to disable. e.g., 0.05 for 5% minimum uncertainty
MIN_RELATIVE_UNCERTAINTY = 0.02

# -----------------------------------------------------------------------------
# 6. METHOD-SPECIFIC PARAMETERS
# -----------------------------------------------------------------------------

# --- 6a. TOF Energy Resolution ---
DELTA_T_NS = 5.0                                 # Time resolution in nanoseconds
FLIGHT_PATH_M = 27.037                           # Flight path in meters
N_SIGMA_CUTOFF = 3.0                             # Gaussian kernel cutoff (±n_sigma * σE)

# --- 6d. Angular-Band Discrepancy ---
USE_BAND_DISCREPANCY = True                      # Use band-based uncertainty (vs global Birge)
MIN_POINTS_PER_BAND = 3                          # Minimum points to estimate τ_b per band
MAX_TAU_FRACTION = 0.05                          # Cap τ_b at 25% of cross section
TAU_SMOOTHING_WINDOW = 3                         # Moving median window for τ_b(E) smoothing
TAU_PRIOR_FLOOR = True                           # Apply tau prior floor from multi-experiment bins
TAU_PRIOR_MIN_EXPERIMENTS = 2                    # Min experiments to count as "well-estimated"
TAU_PRIOR_PERCENTILE = 50                        # Percentile of well-estimated tau for baseline
RESCALE_UNC_BY_CHI2 = True                       # Apply Birge scaling when band discrepancy disabled
ALLOW_SHRINK_UNC = True                          # Allow uncertainties to shrink (chi2_red < 1)

# --- 6e. Per-Experiment Normalization ---
NORMALIZATION_SIGMA = 0.05                       # Per-experiment normalization uncertainty (5%)
NORM_DIST = "lognormal"                          # Distribution: "lognormal" (always positive) or "normal"

# --- 6f. Model Averaging ---
USE_MODEL_AVERAGING = True                       # Enable model averaging over Legendre orders
MIN_DEGREE_FOR_AVERAGING = 1                     # Minimum degree to consider (1 = include all)
USE_DEGREE_SAMPLING_IN_MC = True                 # Sample degree from degree_weights distribution

# --- 6g. Energy Bin Method Specific  ---
NORMALIZE_BY_N_POINTS = True                     # Equal weight per experiment (1/n_points weighting)
MAX_EXP_WEIGHT_FRAC_BIN = 0.5                    # Cap per-experiment dominance (1.0 = disabled)
FREEZE_C0 = True                                # Fix c0 for shape-only refits
MAX_SAMPLE_ORDER = None                            # Max Legendre order to sample (None for no freeze)

# --- 6h. Correlation Method ---
# "gaussian"        — Per-bin stochastic MC → Gaussian parametric energy correlations → Cholesky samples
# "kernel_weight_mc" — Kernel-weighted multi-bin MC → correlations from shared perturbed datasets
CORRELATION_METHOD = "gaussian"
KW_MC_TWO_PASS = True                            # True: per-bin variance + KW correlations. False: single-pass.
KW_MC_MIN_WEIGHT = 1e-3                          # Overlap weight threshold for kernel-weight MC

# --- 6i. TOF Parameters (for energy resolution) ---
TOF_PARAMETERS_FILE = "/share_snc/snc/JuanMonleon/EXFOR/exfor_tof_parameters.json"

# =============================================================================
# END OF CONFIGURATION
# =============================================================================


# Global logger reference (set by _set_logger from exfor_utils)
_logger = None


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

    Used when generate_samples_endf=False but generate_ace=True (reprocessing
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
        max_sample_order,
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
                    max_sample_order=max_sample_order,
                )
                sample_coeffs = coef_df_single.iloc[0].to_numpy()
                if len(sample_coeffs) < max_degree + 1:
                    sample_coeffs = np.pad(sample_coeffs, (0, max_degree + 1 - len(sample_coeffs)))
                if _apply_positivity_projection:
                    if not check_angular_distribution_positivity(sample_coeffs, _positivity_check_points):
                        frozen = {}
                        if freeze_c0:
                            frozen[0] = sample_coeffs[0]
                        if max_sample_order is not None and max_sample_order + 1 < len(sample_coeffs):
                            frozen.update({i: sample_coeffs[i] for i in range(max_sample_order + 1, len(sample_coeffs))})
                        sample_coeffs = project_to_positive_distribution(sample_coeffs, _positivity_check_points, frozen_indices=frozen or None)
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
                max_sample_order=max_sample_order,
            )
            for s_idx in range(n_samples):
                sample_coeffs = coef_df.iloc[s_idx].to_numpy()
                if _apply_positivity_projection:
                    if not check_angular_distribution_positivity(sample_coeffs, _positivity_check_points):
                        frozen = {}
                        if freeze_c0:
                            frozen[0] = sample_coeffs[0]
                        if max_sample_order is not None and max_sample_order + 1 < len(sample_coeffs):
                            frozen.update({i: sample_coeffs[i] for i in range(max_sample_order + 1, len(sample_coeffs))})
                        sample_coeffs = project_to_positive_distribution(sample_coeffs, _positivity_check_points, frozen_indices=frozen or None)
                endf_coeffs = endf_normalize_legendre_coeffs(sample_coeffs, include_a0=False)
                if len(endf_coeffs) < max_degree:
                    padded = np.zeros(max_degree, dtype=float)
                    padded[:len(endf_coeffs)] = endf_coeffs
                    endf_coeffs = padded
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
    max_degree: int,
    select_degree: Optional[str],
    ridge_lambda: float,
    m_proj_u: float,
    m_targ_u: float,
    use_band_discrepancy: bool,
    min_points_per_band: int,
    max_tau_fraction: float,
    tau_smoothing_window: int,
    n_eff_warning_threshold: float = 5.0,
    min_degree_for_averaging: int = 3,
    exclude_experiments: Optional[List[str]] = None,
    min_relative_uncertainty: float = 0.0,
    tau_prior_floor: bool = True,
    tau_prior_min_experiments: int = 2,
    tau_prior_percentile: float = 50.0,
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

    for bin_info in energy_bins:
        exfor_df, experiments_info, kernel_weights, diagnostics = filter_exfor_with_energy_bin(
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
            normalize_by_n_points=NORMALIZE_BY_N_POINTS,
            max_experiment_weight_fraction=MAX_EXP_WEIGHT_FRAC_BIN,
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
        ))

        if logger:
            logger.info(
                f"E = {bin_info.energy_mev:.4f} MeV (bin: [{bin_info.bin_lower_mev:.4f}, {bin_info.bin_upper_mev:.4f}] MeV):"
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

    return results


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
    max_tau_fraction: float = 0.25,
    tau_smoothing_window: int = 3,
    tau_prior_floor: bool = True,
    tau_prior_min_experiments: int = 2,
    tau_prior_percentile: float = 50.0,
    sigma_norm: float = 0.05,
    use_model_averaging: bool = True,
    min_degree_for_averaging: int = 3,
    n_eff_warning_threshold: float = 5.0,
    n_procs: int = 1,
    base_seed: int = 42,
    generate_nominal_endf: bool = True,
    generate_mc_mean_endf: bool = True,
    generate_samples_endf: bool = True,
    save_covariance_files: bool = True,
    # Multigroup covariance options
    generate_multigroup_covariance: bool = False,
    multigroup_rho_min: float = 0.90,
    multigroup_sigma_ratio_max: float = 1.7,
    multigroup_min_width_factor: float = 2.0,
    multigroup_variance_percentile: float = 50.0,
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
    # Covariance cap (Layer 1) and positivity projection (Layer 2)
    apply_covariance_cap: bool = False,
    max_relative_std_cap: float = 1.0,
    apply_positivity_projection: bool = False,
    positivity_check_points: int = 50,
    # File output options
    save_correlation_matrices: bool = False,
    # ACE generation options
    generate_ace: bool = False,
    ace_temperatures: Optional[List[float]] = None,
    ace_njoy_exe: Optional[str] = None,
    ace_library_name: Optional[str] = None,
    ace_njoy_version: str = "NJOY 2016.78",
    ace_xsdir_file: Optional[str] = None,
    ace_skip_existing: bool = False,
    # MF34 merge source
    mf34_source_file: Optional[str] = None,
    # Unified MF34 sampling (Pipeline B)
    generate_samples_from_mf34: bool = False,
    sampling_resolution: str = "fine",  # "fine" or "multigroup"
    merge_original_mf34: bool = True,
    sampling_space: str = "linear",
    sampling_decomposition: str = "svd",
    sampling_method: str = "sobol",
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
    _logger.info(f"    MF34_SOURCE_FILE       = {mf34_source_file or '(same as ENDF_FILE)'}")
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
    _logger.info(f"    SAVE_COVARIANCE_FILES   = {save_covariance_files}")
    _logger.info(f"    N_SAMPLES               = {n_samples}")
    _logger.info(f"    SAVE_CORRELATION_MATRICES = {save_correlation_matrices}")
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

    # -- ACE Generation --
    _logger.info("  ACE Generation:")
    _logger.info(f"    GENERATE_ACE            = {generate_ace}")
    if generate_ace:
        # Normalize temperatures: single float → list
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
                "ACE_LIBRARY_NAME must be provided when GENERATE_ACE=True "
                "(e.g., 'endfb81', 'jeff40')."
            )
        if not ace_temperatures:
            raise ValueError(
                "ACE_TEMPERATURES must be a non-empty list when GENERATE_ACE=True."
            )

        _logger.info(f"    ACE_TEMPERATURES        = {ace_temperatures}")
        _logger.info(f"    ACE_NJOY_EXE            = {ace_njoy_exe}")
        _logger.info(f"    ACE_LIBRARY_NAME        = {ace_library_name}")
        _logger.info(f"    ACE_NJOY_VERSION        = {ace_njoy_version}")
        _logger.info(f"    ACE_XSDIR_FILE          = {ace_xsdir_file}")
        _logger.info(f"    ACE_SKIP_EXISTING       = {ace_skip_existing}")
    _logger.info("")

    # -- Unified MF34 Sampling (Pipeline B) --
    _logger.info("  Unified MF34 Sampling (Pipeline B):")
    _logger.info(f"    GENERATE_SAMPLES_FROM_MF34    = {generate_samples_from_mf34}")
    if generate_samples_from_mf34:
        _logger.info(f"    SAMPLING_RESOLUTION           = {sampling_resolution}")
        _logger.info(f"    MERGE_ORIGINAL_MF34           = {merge_original_mf34}")
        _logger.info(f"    SAMPLING_SPACE                = {sampling_space}")
        _logger.info(f"    SAMPLING_DECOMPOSITION        = {sampling_decomposition}")
        _logger.info(f"    SAMPLING_METHOD               = {sampling_method}")
    _logger.info("")

    # -- Post-Processing Layers --
    _logger.info("  Post-Processing Layers:")
    _logger.info(f"    APPLY_COVARIANCE_CAP       = {apply_covariance_cap}")
    if apply_covariance_cap:
        _logger.info(f"    MAX_RELATIVE_STD_CAP       = {max_relative_std_cap} ({max_relative_std_cap*100:.0f}%)")
    _logger.info(f"    APPLY_POSITIVITY_PROJECTION = {apply_positivity_projection}")
    if apply_positivity_projection:
        _logger.info(f"    POSITIVITY_CHECK_POINTS    = {positivity_check_points}")
    _logger.info("")

    # -- Angular-Band Discrepancy (6d) --
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

    # -- Per-Experiment Normalization (6e) --
    _logger.info("  Per-Experiment Normalization (6e):")
    _logger.info(f"    NORMALIZATION_SIGMA             = {sigma_norm}")
    _logger.info(f"    NORM_DIST                      = {NORM_DIST}")
    _logger.info("")

    # -- Model Averaging (6f) --
    _logger.info("  Model Averaging (6f):")
    _logger.info(f"    USE_MODEL_AVERAGING            = {use_model_averaging}")
    _logger.info(f"    MIN_DEGREE_FOR_AVERAGING       = {min_degree_for_averaging}")
    _logger.info(f"    USE_DEGREE_SAMPLING_IN_MC      = {USE_DEGREE_SAMPLING_IN_MC}")
    _logger.info("")

    # -- Energy Bin Method (6g) --
    _logger.info("  Energy Bin Method (6g):")
    _logger.info(f"    NORMALIZE_BY_N_POINTS          = {NORMALIZE_BY_N_POINTS}")
    _logger.info(f"    MAX_EXP_WEIGHT_FRAC_BIN        = {MAX_EXP_WEIGHT_FRAC_BIN}")
    _logger.info(f"    FREEZE_C0                      = {FREEZE_C0}")
    _logger.info(f"    MAX_SAMPLE_ORDER               = {MAX_SAMPLE_ORDER}")
    _logger.info("")

    # -- Correlation Method (6h) --
    _logger.info("  Correlation Method (6h):")
    _logger.info(f"    CORRELATION_METHOD              = {CORRELATION_METHOD}")
    if CORRELATION_METHOD == "kernel_weight_mc":
        _logger.info(f"    KW_MC_TWO_PASS                 = {KW_MC_TWO_PASS}")
        _logger.info(f"    KW_MC_MIN_WEIGHT               = {KW_MC_MIN_WEIGHT}")
    _logger.info(f"    TOF_PARAMETERS_FILE            = {TOF_PARAMETERS_FILE}")
    _logger.info("")

    _logger.info(separator)
    _logger.info("")

    # Warning legend
    _logger.info("[WARNING LEGEND]")
    _logger.info("  - 'No EXFOR data': Coefficients will be INTERPOLATED from neighboring bins")
    _logger.info("  - 'Low N_eff': Few independent data points contributing to fit")
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
        delta_t_ns=DELTA_T_NS,
        flight_path_m=FLIGHT_PATH_M,
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
        max_tau_fraction=max_tau_fraction,
        tau_smoothing_window=tau_smoothing_window,
        n_eff_warning_threshold=n_eff_warning_threshold,
        min_degree_for_averaging=min_degree_for_averaging,
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
    _logger.info("")
    _logger.info("[STEP 5] Phase 2: MC sampling")
    _logger.info(f"  Generating {n_samples} MC samples")
    _logger.info(f"  Correlation method: {CORRELATION_METHOD}")

    _prebuilt_gaussian_cov = None
    _prebuilt_mc_mean = None

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
                max_tau_fraction,
                USE_DEGREE_SAMPLING_IN_MC,
                RESCALE_UNC_BY_CHI2,
                ALLOW_SHRINK_UNC,
                FREEZE_C0,
                0.0,  # normalization not applied in per-bin pass (Gaussian model handles correlations)
                NORM_DIST,
                MAX_SAMPLE_ORDER,
                apply_positivity_projection,
                positivity_check_points,
            ))

        if N_PROCS > 1:
            with Pool(N_PROCS) as pool:
                bin_results = pool.map(_mc_one_bin, bin_args_list)
        else:
            bin_results = [_mc_one_bin(a) for a in bin_args_list]

        all_samples_stochastic = {s_idx: {} for s_idx in range(n_samples)}
        for energy_idx, is_interpolated, results_by_sample, success in bin_results:
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
            n_valid = min(nr.frozen_degree, max_degree)
            for l in range(n_valid):
                _valid_mask[ie * max_degree + l] = True
        n_invalid = int(np.sum(~_valid_mask))
        _logger.info(f"  Valid-parameter mask: {int(np.sum(_valid_mask))}/{len(_valid_mask)} valid "
                     f"({n_invalid} zeroed for absent higher-order coefficients)")

        # Compute stochastic covariance (for per-bin variances and cross-order correlations)
        cov_stochastic_pass2, _, _, mc_mean_stochastic = compute_covariance_from_samples(
            all_samples=all_samples_stochastic,
            energy_indices=energy_indices_for_gauss,
            max_order=max_degree,
            valid_mask=_valid_mask,
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

    elif CORRELATION_METHOD == "kernel_weight_mc":
        # Kernel-weighted multi-bin MC → correlations from shared perturbed datasets
        _logger.info("  " + "=" * 60)
        _logger.info("  Method: Kernel-weight MC correlations")
        _logger.info(f"  Two-pass mode: {KW_MC_TWO_PASS}")
        _logger.info(f"  Min overlap weight: {KW_MC_MIN_WEIGHT}")
        _logger.info("  " + "=" * 60)

        # Precompute overlap weights for all datasets across all bins
        overlap_weights = precompute_overlap_weights(
            nominal_results=nominal_results,
            energy_bins=energy_bins,
            min_weight=KW_MC_MIN_WEIGHT,
        )

        n_datasets_total = sum(len(dsets) for dsets in overlap_weights.values())
        _logger.info(f"  Overlap weights computed: {n_datasets_total} (dataset, bin) pairs")

        # Run kernel-weight MC (all bins coupled via shared perturbations)
        kw_samples = run_mc_with_kernel_weights(
            nominal_results=nominal_results,
            energy_bins=energy_bins,
            overlap_weights=overlap_weights,
            n_samples=n_samples,
            n_workers=N_PROCS,
            sigma_norm=NORMALIZATION_SIGMA,
            norm_dist=NORM_DIST,
            max_degree=max_degree,
            ridge_lambda=ridge_lambda,
            base_seed=base_seed,
            use_band_discrepancy=use_band_discrepancy,
            min_points_per_band=min_points_per_band,
            max_tau_fraction=max_tau_fraction,
            freeze_c0=FREEZE_C0,
            max_sample_order=MAX_SAMPLE_ORDER,
            apply_positivity_projection=apply_positivity_projection,
            positivity_check_points=positivity_check_points,
            logger=_logger,
        )

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
                    max_tau_fraction,
                    USE_DEGREE_SAMPLING_IN_MC,
                    RESCALE_UNC_BY_CHI2,
                    ALLOW_SHRINK_UNC,
                    FREEZE_C0,
                    NORMALIZATION_SIGMA,
                    NORM_DIST,
                    MAX_SAMPLE_ORDER,
                    apply_positivity_projection,
                    positivity_check_points,
                ))

            if N_PROCS > 1:
                with Pool(N_PROCS) as pool:
                    bin_results = pool.map(_mc_one_bin, bin_args_list)
            else:
                bin_results = [_mc_one_bin(a) for a in bin_args_list]

            all_samples_perbin = {s_idx: {} for s_idx in range(n_samples)}
            for energy_idx, is_interpolated, results_by_sample, success in bin_results:
                for s_idx, endf_coeffs in results_by_sample.items():
                    all_samples_perbin[s_idx][energy_idx] = endf_coeffs

            _logger.info(f"  Per-bin stochastic pass completed: {len(bin_args_list)} bins")

            # Combine: correlations from kw_samples, variance from per-bin
            energy_indices_kw = [nr.energy_index for nr in nominal_results if nr.has_data]
            nr_by_idx_kw = {nr.energy_index: nr for nr in nominal_results}
            _valid_mask_kw = np.zeros(len(energy_indices_kw) * max_degree, dtype=bool)
            for ie, e_idx in enumerate(energy_indices_kw):
                nr = nr_by_idx_kw[e_idx]
                n_valid = min(nr.frozen_degree, max_degree)
                for l in range(n_valid):
                    _valid_mask_kw[ie * max_degree + l] = True

            cov_kw, _, _, _ = compute_covariance_from_samples(
                all_samples=kw_samples, energy_indices=energy_indices_kw,
                max_order=max_degree, valid_mask=_valid_mask_kw,
            )
            cov_perbin, _, _, mc_mean_perbin = compute_covariance_from_samples(
                all_samples=all_samples_perbin, energy_indices=energy_indices_kw,
                max_order=max_degree, valid_mask=_valid_mask_kw,
            )

            # Extract correlation from KW, variance from per-bin
            std_kw = np.sqrt(np.maximum(np.diag(cov_kw), 0.0))
            std_kw[std_kw == 0] = 1.0
            corr_kw = cov_kw / np.outer(std_kw, std_kw)

            std_perbin = np.sqrt(np.maximum(np.diag(cov_perbin), 0.0))
            cov_combined = corr_kw * np.outer(std_perbin, std_perbin)

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
        _logger.info(f"  MC complete: {n_sampled} bins with data, {n_interpolated_used} interpolated")

    else:
        raise ValueError(f"Unknown CORRELATION_METHOD: {CORRELATION_METHOD!r}. Use 'gaussian' or 'kernel_weight_mc'.")

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

    # Build valid-parameter mask for Step 7 covariance paths
    nr_by_idx_s7 = {nr.energy_index: nr for nr in nominal_results}
    valid_mask_s7 = np.zeros(len(energy_indices) * max_degree, dtype=bool)
    for ie, e_idx in enumerate(energy_indices):
        nr = nr_by_idx_s7[e_idx]
        n_valid = min(nr.frozen_degree, max_degree)
        for l in range(n_valid):
            valid_mask_s7[ie * max_degree + l] = True

    if True:  # Covariance always computed (needed for MF34)
        _logger.info("")
        _logger.info("[STEP 7] Computing covariance matrix")

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

            var_total = np.diag(cov_matrix)
            mask = var_total > 0
            if np.any(mask):
                _logger.info(f"  Gaussian cov variance: mean={np.mean(var_total[mask]):.2e}")
        else:
            # Standard covariance from MC samples
            cov_matrix, corr_matrix, param_labels, mc_mean_params = compute_covariance_from_samples(
                all_samples=all_samples,
                energy_indices=energy_indices,
                max_order=max_degree,
                valid_mask=valid_mask_s7,
            )

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
            _logger.warning("  WARNING: No positive diagonal elements in covariance matrix!")

        _logger.info(f"  Covariance matrix shape: {cov_matrix.shape}")

        # l-l' correlation diagnostics
        if max_degree > 1:
            extract_ll_prime_correlations(
                cov_matrix=cov_matrix,
                energy_indices=energy_indices,
                max_order=max_degree,
                logger=_logger,
                valid_mask=valid_mask_s7,
            )

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
        if save_covariance_files:
            np.save(output_path / "legendre_covariance.npy", cov_matrix)
            if save_correlation_matrices:
                np.save(output_path / "legendre_correlation.npy", corr_matrix)

    # Step 7b: Multigroup covariance (optional)
    multigroup_result = None
    multigroup_failure_reason = None
    if generate_multigroup_covariance:
        if cov_matrix is None:
            multigroup_failure_reason = "covariance matrix is None (computation may have failed)"
        else:
            _logger.info("")
            _logger.info("[STEP 7b] Computing adaptive multigroup covariance")
            _logger.info(f"  Using l=1 correlation for grouping (same grid for all orders)")

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
                    _logger.warning("  MF34 grid extraction failed — falling back to adaptive grouping")

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
                    forced_group_boundaries_mev=forced_grid,
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

    # Step 9b: ACE generation via NJOY
    if generate_ace:
        _logger.info("")
        _logger.info("[STEP 9b] Generating ACE files via NJOY")

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
            _logger.warning("[ACE] No ENDF sample files found — skipping ACE generation", console=True)
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
                    _logger.warning("  Could not read ZAID from ENDF file — skip-existing disabled")

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
                _logger.warning(f"  {len(all_errors)} error(s) during ACE generation:", console=True)
                for err in all_errors[:10]:
                    _logger.warning(f"    {err}")
                if len(all_errors) > 10:
                    _logger.warning(f"    ... and {len(all_errors) - 10} more")

            print(f"[INFO] ACE generation: {n_success}/{len(valid_files)} samples processed in {t_ace_elapsed:.1f}s")

    # Step 10: MF34 (using library functions)
    mg_nom_file = None
    if cov_matrix is not None:
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

            # Build nominal relative covariance for the nominal file.
            # The cov_matrix is relative w.r.t. MC means (correct for average file).
            # For the nominal file, we need relative w.r.t. nominal coefficients:
            #   Cov_rel_nom(i,j) = Cov_rel_avg(i,j) * (mean_i/nom_i) * (mean_j/nom_j)
            nominal_params = np.zeros_like(mc_mean_params)
            for k, e_idx in enumerate(energy_indices):
                nr = next((r for r in nominal_results if r.energy_index == e_idx), None)
                if nr is not None and nr.has_data:
                    endf_nom = endf_normalize_legendre_coeffs(nr.nominal_coeffs, include_a0=False)
                    n = min(len(endf_nom), max_degree)
                    nominal_params[k * max_degree: k * max_degree + n] = endf_nom[:n]
            # Compute rescaling factors
            scale = np.ones_like(mc_mean_params)
            safe = np.abs(nominal_params) > 1e-30
            scale[safe] = mc_mean_params[safe] / nominal_params[safe]
            cov_matrix_nominal = cov_matrix * np.outer(scale, scale)
            max_scale_dev = np.max(np.abs(scale[safe] - 1.0)) if np.any(safe) else 0.0
            _logger.info(f"  Nominal vs MC-mean rescaling: max |scale-1| = {max_scale_dev:.6f}")

            # Read original MF34 from reference file (if present)
            # Use mf34_source_file if provided, otherwise fall back to endf_file
            mf34_ref = mf34_source_file if mf34_source_file else endf_file
            original_mf34_mt = None
            try:
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

            # Compute fine energy grid (needed for fine MF34 and/or Step 11 sampling)
            processed_energies_ev = np.array([all_energies_ev[i] for i in energy_indices])
            if energy_indices[-1] + 1 < len(all_energies_ev):
                energy_grid_ev = np.append(processed_energies_ev, all_energies_ev[energy_indices[-1] + 1])
            else:
                if len(processed_energies_ev) > 1:
                    delta = processed_energies_ev[-1] - processed_energies_ev[-2]
                    energy_grid_ev = np.append(processed_energies_ev, processed_energies_ev[-1] + delta)
                else:
                    energy_grid_ev = np.append(processed_energies_ev, processed_energies_ev[-1] * 1.1)

            # Write fine-grid MF34 if requested
            if mf34_covariance_type in ("fine", "both"):
                if average_file:
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
                    mc_mean_for_mg = mc_mean_params[np.array(flat_indices)]
                else:
                    mc_mean_for_mg = mc_mean_params

                mc_mean_grouped = A @ mc_mean_for_mg
                nom_mean_grouped = multigroup_result.mean_grouped  # based on nominal coeffs
                mg_scale = np.ones_like(mc_mean_grouped)
                mg_safe = np.abs(nom_mean_grouped) > 1e-30
                mg_scale[mg_safe] = mc_mean_grouped[mg_safe] / nom_mean_grouped[mg_safe]
                cov_grouped_nominal = multigroup_result.cov_grouped * np.outer(mg_scale, mg_scale)

            # Write multigroup MF34 if requested and available
            if mf34_covariance_type in ("multigroup", "both") and cov_grouped_nominal is not None:
                # For multigroup, write to separate files with _mg suffix
                if average_file:
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

            elif mf34_covariance_type in ("multigroup", "both") and cov_grouped_nominal is None:
                if multigroup_failure_reason:
                    _logger.warning(f"  Multigroup covariance requested but failed: {multigroup_failure_reason}", console=True)
                elif not generate_multigroup_covariance:
                    _logger.warning("  Multigroup covariance requested but not computed (enable GENERATE_MULTIGROUP_COVARIANCE)")
                else:
                    _logger.warning("  Multigroup covariance requested but not computed (unknown reason)")

            # --- Resolve MF34 source file for Step 11 ---
            mf34_sample_source = None
            if generate_samples_from_mf34 and nominal_file:
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
            _logger.error(f"Failed to write MF34: {str(e)}", console=True)
            mf34_sample_source = None

    else:
        mf34_sample_source = None

    # Step 11: Generate perturbed ENDF samples from MF34 covariance (Pipeline B)
    if generate_samples_from_mf34:
        _logger.info("")
        _logger.info("[STEP 11] Generating perturbed ENDF samples from MF34 (Pipeline B)")
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
                    generate_ace=generate_ace,
                    njoy_exe=ace_njoy_exe if generate_ace else None,
                    temperatures=ace_temperatures if generate_ace else None,
                    library_name=ace_library_name if generate_ace else None,
                    njoy_version=ace_njoy_version,
                    xsdir_file=ace_xsdir_file if generate_ace else None,
                )
                _logger.info(f"  Pipeline B samples written to: {output_path / 'endf'}")
            except Exception as e:
                _logger.error(f"  Pipeline B sampling failed: {str(e)}", console=True)
                _logger.error(f"  Traceback:\n{traceback.format_exc()}")
        else:
            _logger.warning(
                f"  MF34 source for ({sampling_resolution}, merge={merge_original_mf34}) "
                f"not available — skipping Step 11",
                console=True,
            )

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
        use_band_discrepancy=USE_BAND_DISCREPANCY,
        min_points_per_band=MIN_POINTS_PER_BAND,
        max_tau_fraction=MAX_TAU_FRACTION,
        tau_smoothing_window=TAU_SMOOTHING_WINDOW,
        sigma_norm=NORMALIZATION_SIGMA,
        use_model_averaging=USE_MODEL_AVERAGING,
        min_degree_for_averaging=MIN_DEGREE_FOR_AVERAGING,
        n_eff_warning_threshold=N_EFF_WARNING_THRESHOLD,
        n_procs=N_PROCS,
        base_seed=BASE_SEED,
        generate_nominal_endf=GENERATE_NOMINAL_ENDF,
        generate_mc_mean_endf=GENERATE_MC_MEAN_ENDF,
        generate_samples_endf=GENERATE_SAMPLES_ENDF,
        save_covariance_files=SAVE_COVARIANCE_FILES,
        # Multigroup covariance options
        generate_multigroup_covariance=GENERATE_MULTIGROUP_COVARIANCE,
        multigroup_rho_min=MULTIGROUP_RHO_MIN,
        multigroup_sigma_ratio_max=MULTIGROUP_SIGMA_RATIO_MAX,
        multigroup_min_width_factor=MULTIGROUP_MIN_WIDTH_FACTOR,
        multigroup_variance_percentile=MULTIGROUP_VARIANCE_PERCENTILE,
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
        # Covariance cap (Layer 1) and positivity projection (Layer 2)
        apply_covariance_cap=APPLY_COVARIANCE_CAP,
        max_relative_std_cap=MAX_RELATIVE_STD_CAP,
        apply_positivity_projection=APPLY_POSITIVITY_PROJECTION,
        positivity_check_points=POSITIVITY_CHECK_POINTS,
        # File output options
        save_correlation_matrices=SAVE_CORRELATION_MATRICES,
        # ACE generation options
        generate_ace=GENERATE_ACE,
        ace_temperatures=ACE_TEMPERATURES,
        ace_njoy_exe=ACE_NJOY_EXE,
        ace_library_name=ACE_LIBRARY_NAME,
        ace_njoy_version=ACE_NJOY_VERSION,
        ace_xsdir_file=ACE_XSDIR_FILE,
        ace_skip_existing=ACE_SKIP_EXISTING,
        mf34_source_file=MF34_SOURCE_FILE,
        # Unified MF34 sampling (Pipeline B)
        generate_samples_from_mf34=GENERATE_SAMPLES_FROM_MF34,
        sampling_resolution=SAMPLING_RESOLUTION,
        merge_original_mf34=MERGE_ORIGINAL_MF34,
        sampling_space=SAMPLING_SPACE,
        sampling_decomposition=SAMPLING_DECOMPOSITION,
        sampling_method=SAMPLING_METHOD,
    )
