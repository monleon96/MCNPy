"""
Utility functions for EXFOR-to-ENDF angular distribution sampling.

This module contains reusable functions for:
- EXFOR data loading and filtering
- Kernel weighting and diagnostics
- Energy binning with TOF resolution
- Covariance computation
- ENDF file writing
- MF34 covariance generation

These functions are used by the main workflow in exfor_to_endf_sampling.py.
"""
from __future__ import annotations

import os
import sys
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import norm

# Add kika to path if needed
_kika_path = Path(__file__).parent.parent
if str(_kika_path) not in sys.path:
    sys.path.insert(0, str(_kika_path))

# Import kika modules
from kika.endf.read_endf import read_endf
from kika.endf.writers.endf_writer import ENDFWriter
from kika.endf.classes.mf4.polynomial import MF4MTLegendre
from kika.endf.classes.mf4.mixed import MF4MTMixed

# Use new kika.exfor module for transforms
from kika.exfor.transforms import transform_lab_to_cm, jacobian_cm_to_lab
from kika.exfor.angular_distribution import ExforAngularDistribution

# Import resample_AD functions (relative import for same directory)
from .resample_AD import (
    load_exfor_for_fitting,
    endf_normalize_legendre_coeffs,
    compute_energy_resolution_tof,
    compute_n_eff,
    compute_weight_span_95,
)


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

class DualLogger:
    """Logger that writes to both file and optionally to console."""

    def __init__(self, log_file: str):
        self.log_file = log_file

        # Create logger
        self.logger = logging.getLogger('exfor_to_endf')
        self.logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        self.logger.handlers.clear()

        # File handler
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def info(self, msg: str, console: bool = False):
        self.logger.info(msg)
        if console:
            print(f"[INFO] {msg}")

    def warning(self, msg: str, console: bool = False):
        self.logger.warning(f"[WARNING] {msg}")
        if console:
            print(f"[WARNING] {msg}")

    def error(self, msg: str, console: bool = False):
        self.logger.error(f"[ERROR] {msg}")
        if console:
            print(f"[ERROR] {msg}")

    def debug(self, msg: str):
        self.logger.debug(msg)


# Global logger instance
_logger: Optional[DualLogger] = None


def _get_logger() -> Optional[DualLogger]:
    """Get the global logger instance."""
    return _logger


def _set_logger(logger: Optional[DualLogger]) -> None:
    """Set the global logger instance."""
    global _logger
    _logger = logger


def _format_condensed_experiments(experiments_info: List[Dict]) -> List[str]:
    """
    Group experiments by (entry, subentry) and format as condensed log lines.

    Instead of listing each experiment occurrence separately, this groups
    multiple occurrences of the same experiment and summarizes:
    - Number of energies used (and if deduplication occurred)
    - Energy range
    - Total number of angular points
    - Weight range

    Parameters
    ----------
    experiments_info : List[Dict]
        List of experiment info dicts with keys:
        entry, subentry, author, year, exfor_energy_mev, kernel_weight, n_points
        Optionally includes 'selected_from_n_energies' for deduplication info.

    Returns
    -------
    List[str]
        Formatted log lines, one per unique experiment
    """
    if not experiments_info:
        return []

    # Group by (entry, subentry)
    grouped = defaultdict(lambda: {
        'author': '',
        'year': '',
        'energies': [],
        'weights': [],
        'total_points': 0,
        'selected_from_n_energies': 0,  # Track deduplication
    })

    for exp in experiments_info:
        key = (exp['entry'], exp['subentry'])
        grouped[key]['author'] = exp['author']
        grouped[key]['year'] = exp['year']
        grouped[key]['energies'].append(exp['exfor_energy_mev'])
        grouped[key]['weights'].append(exp['kernel_weight'])
        grouped[key]['total_points'] += exp['n_points']
        # Track deduplication info (use max in case there are multiple)
        n_in_bin = exp.get('selected_from_n_energies', 1)
        if n_in_bin > grouped[key]['selected_from_n_energies']:
            grouped[key]['selected_from_n_energies'] = n_in_bin

    # Format each group
    lines = []
    for (entry, subentry), data in sorted(grouped.items()):
        exp_id = f"{entry}.{subentry}"
        author = data['author']
        year = data['year']

        energies = sorted(data['energies'])
        n_energies = len(energies)
        e_min, e_max = energies[0], energies[-1]

        weights = data['weights']
        w_min, w_max = min(weights), max(weights)

        total_pts = data['total_points']

        # Track deduplication
        n_in_bin = data['selected_from_n_energies']

        # Format energy string with deduplication info
        if n_energies == 1:
            if n_in_bin > 1:
                # Single energy selected from multiple in bin
                energy_str = f"{e_min:.4f} MeV (closest of {n_in_bin} in bin)"
            else:
                energy_str = f"{e_min:.4f} MeV"
        else:
            # Multiple energies used (shouldn't happen with deduplication, but handle anyway)
            if n_in_bin > n_energies:
                energy_str = f"{n_energies} energies [{e_min:.4f}-{e_max:.4f} MeV] (from {n_in_bin} in bin)"
            else:
                energy_str = f"{n_energies} energies [{e_min:.4f}-{e_max:.4f} MeV]"

        # Format weight string
        if abs(w_max - w_min) < 0.001:
            weight_str = f"w={w_min:.3f}"
        else:
            weight_str = f"w=[{w_min:.3f}-{w_max:.3f}]"

        line = f"  - {exp_id} ({author}, {year}): {energy_str}, {total_pts} pts, {weight_str}"
        lines.append(line)

    return lines


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EnergyBinInfo:
    """Information about an energy bin for EXFOR fitting."""
    index: int                           # Index in the energy grid
    energy_ev: float                     # Energy in eV
    energy_mev: float                    # Energy in MeV
    sigma_E_mev: float = 0.0             # Energy resolution from TOF (MeV)
    bin_lower_mev: float = 0.0           # Lower bin boundary (MeV) for energy_bin method
    bin_upper_mev: float = float('inf')  # Upper bin boundary (MeV) for energy_bin method
    original_coeffs: List[float] = field(default_factory=list)  # Original Legendre coefficients
    has_exfor_data: bool = False         # Whether EXFOR data was found
    exfor_n_points: int = 0              # Number of EXFOR data points
    exfor_n_experiments: int = 0         # Number of matching experiments
    experiments_used: List[Dict] = field(default_factory=list)  # List of experiments used
    fitted_degree: int = 0               # Fitted Legendre degree
    chi2_red: float = 0.0                # Reduced chi-squared of fit
    tau_F: float = 0.0                   # Forward band discrepancy
    tau_M: float = 0.0                   # Mid band discrepancy
    tau_B: float = 0.0                   # Backward band discrepancy
    interpolated: bool = False           # Whether coefficients were interpolated


@dataclass
class SamplingResult:
    """Result from sampling at one energy bin."""
    bin_info: EnergyBinInfo
    sampled_coeffs: Optional[np.ndarray] = None  # Shape: (n_samples, n_coeffs) - ENDF format (a_1, a_2, ...)
    fit_info: Optional[Dict[str, Any]] = None


@dataclass
class KernelDiagnostics:
    """Diagnostics for kernel weighting at one energy point.

    These metrics help assess the quality of the Gaussian kernel weighting:
    - n_eff: Effective sample size (higher is better)
    - weight_span_95: Energy interval containing 95% of weight
    - weight_span_ratio: weight_span_95 / σE (should be close to 2-3)
    - max_experiment_weight_frac: Largest single experiment contribution
    - capping_applied: Whether any experiment was weight-capped
    """
    n_eff: float                                    # Effective sample size
    weight_span_95: float                           # 95% weight span in MeV
    weight_span_ratio: float                        # weight_span_95 / sigma_E
    n_experiments: int                              # Number of experiments contributing
    max_experiment_weight_frac: float               # Largest single experiment weight fraction
    experiment_weights: Dict[str, float]            # {exp_key: weight_frac} before capping
    n_points_dropped: int                           # Points dropped by min weight threshold
    capping_applied: bool                           # Whether experiment capping was applied


@dataclass
class DatasetEnergyInfo:
    """
    Information about a dataset (experiment at one energy) for energy jitter MC.

    Used in Improvement 1.4 (cross-bin correlation via energy jitter) to track
    which datasets contribute to which energy bins and their energy resolution.

    Attributes
    ----------
    entry : str
        EXFOR entry number
    subentry : str
        EXFOR subentry number
    nominal_energy_mev : float
        Original EXFOR measurement energy in MeV
    sigma_E_mev : float
        Energy resolution at this energy (from TOF parameters)
    nominal_bin_index : int
        Index of the bin containing the nominal energy
    exfor_df_indices : List[int]
        Row indices in the combined exfor_df for this dataset
    tof_source : str
        Source of TOF parameters: "file" or "default"
    n_points : int
        Number of angular data points for this dataset
    """
    entry: str
    subentry: str
    nominal_energy_mev: float
    sigma_E_mev: float
    nominal_bin_index: int
    exfor_df_indices: List[int]
    tof_source: str
    n_points: int = 0


@dataclass
class BinJumpDiagnostics:
    """
    Diagnostics for energy jitter bin jumping behavior.

    Tracks how often datasets jump between bins during MC sampling,
    which indicates the strength of cross-bin energy correlations.

    Interpretation:
    - Jump rate >30%: Grid finer than σE, strong energy correlations
    - Jump rate 10-30%: Grid comparable to σE, moderate correlations
    - Jump rate <10%: Grid coarser than σE, weak correlations
    """
    total_assignments: int              # Total dataset-to-bin assignments across all samples
    jumped_bins: int                    # Number of assignments where dataset jumped from nominal bin
    jump_rate: float                    # jumped_bins / total_assignments
    jump_counts: Dict[Tuple[int, int], int]  # {(from_bin, to_bin): count}
    datasets_outside_range: int         # Datasets that fell outside all bins (clipped or dropped)
    interpolated_bins: int = 0          # Empty bins filled by neighbor interpolation (summed over all samples)
    nominal_fallback_bins: int = 0      # Empty bins where interpolation failed, fell back to nominal (summed over all samples)

    def top_jumps(self, n: int = 5) -> List[Tuple[Tuple[int, int], int]]:
        """Get the n most common bin jumps."""
        sorted_jumps = sorted(self.jump_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_jumps[:n]


# =============================================================================
# RESOLUTION OVERLAP WEIGHTING
# =============================================================================

def compute_overlap_weight(
    exp_energy_mev: float,
    sigma_E_mev: float,
    bin_lower_mev: float,
    bin_upper_mev: float,
) -> float:
    """
    Compute probability that measurement's true energy lies within bin.

    Instead of Gaussian distance-based weighting, this computes the probability
    that an experiment's true energy falls within the ENDF bin, given the
    experiment's energy resolution.

    Formula:
        w = Φ((E_high - E_j)/σ_j) - Φ((E_low - E_j)/σ_j)

    Where:
        Φ = standard normal CDF
        E_j = experimental measurement energy
        σ_j = experiment-specific energy resolution
        [E_low, E_high] = ENDF bin boundaries

    Properties:
        - E inside bin + good resolution → w ≈ 1
        - E outside bin + good resolution → w ≈ 0 (no dragging in off-energy data)
        - Poor resolution → smears across bins (physics-consistent)

    Parameters
    ----------
    exp_energy_mev : float
        Experimental measurement energy in MeV
    sigma_E_mev : float
        Experiment-specific energy resolution in MeV
    bin_lower_mev : float
        Lower boundary of ENDF bin in MeV
    bin_upper_mev : float
        Upper boundary of ENDF bin in MeV

    Returns
    -------
    float
        Probability weight [0, 1] that the true energy is in the bin
    """
    if sigma_E_mev <= 0:
        # Perfect resolution: 1 if inside bin, 0 otherwise
        return 1.0 if bin_lower_mev <= exp_energy_mev <= bin_upper_mev else 0.0

    z_high = (bin_upper_mev - exp_energy_mev) / sigma_E_mev
    z_low = (bin_lower_mev - exp_energy_mev) / sigma_E_mev

    return norm.cdf(z_high) - norm.cdf(z_low)


# =============================================================================
# ENERGY BINNING WITH TOF RESOLUTION
# =============================================================================

def compute_energy_bins_with_tof_resolution(
    energies_ev: np.ndarray,
    energy_min_mev: float,
    energy_max_mev: float,
    delta_t_ns: float = 5.0,
    flight_path_m: float = 27.037,
) -> List[EnergyBinInfo]:
    """
    Compute energy bins with TOF-based energy resolution.

    Uses TOF (Time-of-Flight) parameters to compute energy resolution σE(E)
    at each energy point, which determines the Gaussian kernel width for
    including experimental data.

    Also computes bin boundaries for the energy_bin selection method:
    - Lower bound: midpoint to previous energy point
    - Upper bound: midpoint to next energy point

    Parameters
    ----------
    energies_ev : np.ndarray
        Energy grid in eV
    energy_min_mev : float
        Minimum energy to process (in MeV)
    energy_max_mev : float
        Maximum energy to process (in MeV)
    delta_t_ns : float
        TOF time resolution in nanoseconds
    flight_path_m : float
        TOF flight path in meters

    Returns
    -------
    List[EnergyBinInfo]
        List of energy bin info objects with computed σE and bin boundaries
    """
    logger = _get_logger()

    energies_mev = energies_ev / 1e6  # Convert to MeV

    # First pass: identify indices in range
    indices_in_range = []
    for i, e_mev in enumerate(energies_mev):
        if e_mev >= energy_min_mev and e_mev <= energy_max_mev:
            indices_in_range.append(i)

    bins = []
    n_bins = len(indices_in_range)

    for local_idx, global_idx in enumerate(indices_in_range):
        e_mev = energies_mev[global_idx]

        # Compute TOF-based energy resolution
        sigma_E = compute_energy_resolution_tof(
            E_mev=e_mev,
            delta_t_ns=delta_t_ns,
            flight_path_m=flight_path_m,
        )

        # Compute bin boundaries (midpoints to neighbors)
        # Lower boundary
        if local_idx == 0:
            # First bin in range: use midpoint to previous grid point if available
            if global_idx > 0:
                bin_lower = (energies_mev[global_idx - 1] + e_mev) / 2.0
            else:
                bin_lower = 0.0  # No previous point, extend down to 0
        else:
            prev_global_idx = indices_in_range[local_idx - 1]
            bin_lower = (energies_mev[prev_global_idx] + e_mev) / 2.0

        # Upper boundary
        if local_idx == n_bins - 1:
            # Last bin in range: use midpoint to next grid point if available
            if global_idx < len(energies_mev) - 1:
                bin_upper = (e_mev + energies_mev[global_idx + 1]) / 2.0
            else:
                bin_upper = float('inf')  # No next point, extend up
        else:
            next_global_idx = indices_in_range[local_idx + 1]
            bin_upper = (e_mev + energies_mev[next_global_idx]) / 2.0

        bin_info = EnergyBinInfo(
            index=global_idx,
            energy_ev=energies_ev[global_idx],
            energy_mev=e_mev,
            sigma_E_mev=sigma_E,
            bin_lower_mev=bin_lower,
            bin_upper_mev=bin_upper,
        )
        bins.append(bin_info)

    if logger:
        logger.info(f"Computed tolerances for {len(bins)} energy bins in range [{energy_min_mev:.3f}, {energy_max_mev:.3f}] MeV")

    return bins


# =============================================================================
# EXFOR CACHE BUILDING
# =============================================================================

def build_exfor_cache_from_objects(
    exfor_objects: List[ExforAngularDistribution],
    exclude_experiments: Optional[List[str]] = None,
) -> Tuple[Dict[float, List[Tuple[pd.DataFrame, Dict]]], List[float]]:
    """
    Build an EXFOR data cache from ExforAngularDistribution objects.

    This function converts a list of ExforAngularDistribution objects into the
    cache format expected by filter_exfor_with_kernel_weights. For each object,
    it uses to_dataframe() to extract data at each available energy.

    Parameters
    ----------
    exfor_objects : List[ExforAngularDistribution]
        List of EXFOR angular distribution objects to cache
    exclude_experiments : List[str], optional
        List of experiments to exclude from the cache. Accepts multiple formats:
        - "20743" - excludes all subentries starting with 20743
        - "20743002" - excludes specific dataset
        - "20743/002" - same as above

    Returns
    -------
    Tuple[Dict[float, List[Tuple[pd.DataFrame, Dict]]], List[float]]
        - exfor_cache: Dict mapping energy (MeV) to list of (DataFrame, metadata) tuples
        - sorted_energies: Sorted list of all available energies (MeV)

    Notes
    -----
    The returned DataFrame has columns compatible with filter_exfor_with_kernel_weights:
        - 'angle': Angle in degrees
        - 'dsig': Differential cross section in b/sr
        - 'error_stat': Statistical uncertainty in b/sr

    The metadata dict contains:
        - 'entry': EXFOR entry number
        - 'subentry': EXFOR subentry number
        - 'angle_frame': Reference frame ('CM' or 'LAB')
        - 'reaction': Reaction notation
        - 'citation': Citation dict with authors, year, etc.
        - 'energy_resolution_inputs': TOF parameters if available
    """
    exfor_cache: Dict[float, List[Tuple[pd.DataFrame, Dict]]] = {}
    all_energies = set()

    # Parse exclusion patterns
    exclusion_patterns = _parse_exclusion_list(exclude_experiments)

    for exfor in exfor_objects:
        # Check if experiment is excluded
        if _is_experiment_excluded(exfor.entry, exfor.subentry, exclusion_patterns):
            continue

        # Get all available energies in MeV
        energies_mev = exfor.energies(unit='MeV')

        for energy_mev in energies_mev:
            # Get data at this energy using to_dataframe
            # Use small tolerance (0.1 keV) to avoid mixing data from different energies
            # while allowing for floating point precision issues
            df = exfor.to_dataframe(
                energy=energy_mev,
                energy_unit='MeV',
                cross_section_unit='b/sr',
                angle_unit='deg',
                tolerance=1e-4,  # 0.1 keV tolerance
            )

            if df.empty:
                continue

            # Convert DataFrame to expected column names
            cache_df = pd.DataFrame({
                'angle': df['angle'].values,
                'dsig': df['value'].values,
                'error_stat': df['error'].values,
            })

            # Build metadata dict
            meta = {
                'entry': exfor.entry,
                'subentry': exfor.subentry,
                'angle_frame': exfor.angle_frame,
                'reaction': exfor.reaction.get('notation', ''),
                'citation': exfor.citation,
            }

            # Add energy resolution inputs if available
            energy_res = exfor.method.get('energy_resolution_input') or exfor.method.get('energy_resolution_inputs')
            if energy_res:
                distance = energy_res.get('distance', {})
                time_res = energy_res.get('time_resolution', {})
                meta['energy_resolution_inputs'] = {
                    'flight_path_m': distance.get('value'),
                    'time_resolution_ns': time_res.get('value'),
                }

            # Add to cache
            if energy_mev not in exfor_cache:
                exfor_cache[energy_mev] = []
            exfor_cache[energy_mev].append((cache_df, meta))
            all_energies.add(energy_mev)

    sorted_energies = sorted(all_energies)
    return exfor_cache, sorted_energies


# =============================================================================
# EXFOR FILTERING FUNCTIONS
# =============================================================================


def _parse_exclusion_list(exclude_list: Optional[List[str]]) -> set:
    """
    Parse exclusion list into a set of normalized patterns for matching.

    Accepts multiple formats:
    - "20743" - excludes all subentries starting with 20743
    - "20743002" - excludes specific dataset
    - "20743/002" - same as above

    Parameters
    ----------
    exclude_list : List[str], optional
        List of experiment IDs to exclude

    Returns
    -------
    set
        Set of (entry_prefix, full_id) tuples for matching.
        entry_prefix is for matching all subentries, full_id for exact match.
    """
    if not exclude_list:
        return set()

    patterns = set()
    for item in exclude_list:
        item = item.strip()
        if not item:
            continue

        # Handle "entry/subentry" format
        if "/" in item:
            parts = item.split("/")
            entry = parts[0].strip()
            subentry = parts[1].strip() if len(parts) > 1 else ""
            full_id = entry + subentry
            patterns.add(full_id)
        elif len(item) <= 5:
            # Short ID - treat as entry prefix (matches all subentries)
            patterns.add(("prefix", item))
        else:
            # Full dataset ID
            patterns.add(item)

    return patterns


def _is_experiment_excluded(
    entry: str,
    subentry: str,
    exclusion_patterns: set,
) -> bool:
    """
    Check if an experiment matches any exclusion pattern.

    Parameters
    ----------
    entry : str
        EXFOR entry number (e.g., "20743")
    subentry : str
        EXFOR subentry number (e.g., "002")
    exclusion_patterns : set
        Set of patterns from _parse_exclusion_list()

    Returns
    -------
    bool
        True if experiment should be excluded
    """
    if not exclusion_patterns:
        return False

    # Build full dataset ID
    full_id = entry + subentry

    for pattern in exclusion_patterns:
        if isinstance(pattern, tuple) and pattern[0] == "prefix":
            # Prefix match - exclude all subentries of this entry
            if full_id.startswith(pattern[1]):
                return True
        elif full_id == pattern:
            # Exact match
            return True

    return False


def filter_exfor_with_kernel_weights(
    exfor_cache: Dict[float, List[Tuple[pd.DataFrame, Dict]]],
    sorted_energies: List[float],
    energy_mev: float,
    sigma_E_mev: float,
    n_sigma: float,
    m_proj_u: float,
    m_targ_u: float,
    bin_lower_mev: float = 0.0,
    bin_upper_mev: float = float('inf'),
    min_kernel_weight_fraction: float = 1e-3,
    max_experiment_weight_fraction: float = 0.5,
    default_delta_t_ns: float = 5.0,
    default_flight_path_m: float = 27.037,
    use_overlap_weights: bool = True,
    normalize_by_n_points: bool = True,
    dedupe_per_experiment: bool = True,
    exclude_experiments: Optional[List[str]] = None,
    min_relative_uncertainty: float = 0.0,
    logger = None,
) -> Tuple[pd.DataFrame, List[Dict], np.ndarray, KernelDiagnostics]:
    """
    Filter EXFOR data using resolution-aware kernel weighting with diagnostics.

    Two weighting modes are available:

    1. Overlap weights (use_overlap_weights=True, recommended):
       w = Φ((E_high - E_j)/σ_j) - Φ((E_low - E_j)/σ_j)
       Computes probability that the true energy lies within the bin.

    2. Gaussian kernel (use_overlap_weights=False, legacy):
       g_ij = exp(-0.5 * ((E_i - E_j)/σE_j)²)
       Distance-based weighting.

    Per-energy normalization (normalize_by_n_points=True):
       Divides weight by number of angular points at each energy
       to prevent experiments with many angles from dominating.

    Per-experiment deduplication (dedupe_per_experiment=True):
       If an experiment has multiple energies within the kernel range,
       only the energy with the highest kernel weight is selected.
       This prevents experiments with dense energy sampling from dominating.

    Parameters
    ----------
    exfor_cache : Dict[float, List[Tuple[pd.DataFrame, Dict]]]
        Pre-loaded EXFOR data organized by energy
    sorted_energies : List[float]
        Sorted list of available energies in cache
    energy_mev : float
        Target ENDF grid energy in MeV
    sigma_E_mev : float
        Default energy resolution at target energy in MeV (fallback)
    n_sigma : float
        Cutoff in units of σE (typically 3.0)
    m_proj_u : float
        Projectile mass in atomic mass units
    m_targ_u : float
        Target mass in atomic mass units
    bin_lower_mev : float
        Lower boundary of ENDF bin in MeV (for overlap weights)
    bin_upper_mev : float
        Upper boundary of ENDF bin in MeV (for overlap weights)
    min_kernel_weight_fraction : float
        Minimum kernel weight as fraction of max (default: 1e-3)
    max_experiment_weight_fraction : float
        Maximum allowed weight fraction per experiment (default: 0.5)
    default_delta_t_ns : float
        Default time resolution in nanoseconds for fallback (default: 10.0)
    default_flight_path_m : float
        Default flight path in meters for fallback (default: 27.037)
    use_overlap_weights : bool
        If True, use resolution overlap weighting (recommended).
        If False, use legacy Gaussian kernel weighting.
    normalize_by_n_points : bool
        If True, divide weight by number of angular points at each energy.
        This prevents experiments with dense angular sampling from dominating.
    dedupe_per_experiment : bool
        If True (default), select only the highest-weighted energy for each
        experiment. This prevents experiments with many energies in range from
        dominating the fit.
    exclude_experiments : List[str], optional
        List of experiments to exclude from filtering. Accepts multiple formats:
        - "20743" - excludes all subentries starting with 20743
        - "20743002" - excludes specific dataset
        - "20743/002" - same as above
    min_relative_uncertainty : float, optional
        Minimum relative uncertainty as a fraction (default: 0.0 = disabled).
        For example, 0.03 means 3% minimum uncertainty. This prevents experiments
        with unrealistically small uncertainties from dominating the fit.
    logger : logging.Logger, optional
        Logger for reporting fallback usage

    Returns
    -------
    Tuple[pd.DataFrame, List[Dict], np.ndarray, KernelDiagnostics]
        - DataFrame with EXFOR data including 'kernel_weight' column
        - List of experiment metadata dicts (includes 'selected_from_n_energies')
        - Array of kernel weights for each data point
        - KernelDiagnostics object with N_eff, weight span, etc.
    """
    # Empty diagnostics for early returns
    empty_diag = KernelDiagnostics(
        n_eff=0.0, weight_span_95=0.0, weight_span_ratio=0.0,
        n_experiments=0, max_experiment_weight_frac=0.0,
        experiment_weights={}, n_points_dropped=0, capping_applied=False
    )

    if not exfor_cache or not sorted_energies:
        return pd.DataFrame(), [], np.array([]), empty_diag

    # Parse exclusion patterns
    exclusion_patterns = _parse_exclusion_list(exclude_experiments)

    all_frames = []
    experiments_info = []
    all_kernel_weights = []

    # Track experiments that used fallback (log once per experiment, not per energy)
    fallback_logged = set()

    # Step 1: Collect all candidate data with computed kernel weights, grouped by experiment
    # experiment_candidates: {(entry, subentry): [(energy, df, meta, kernel_weight, exp_sigma_E, used_fallback), ...]}
    experiment_candidates: Dict[Tuple[str, str], List[Tuple[float, pd.DataFrame, Dict, float, float, bool]]] = defaultdict(list)

    for available_energy in sorted_energies:
        entries = exfor_cache.get(available_energy, [])
        for df, meta in entries:
            # Extract metadata
            entry = meta.get('entry', 'unknown')
            subentry = meta.get('subentry', 'unknown')

            # Check if experiment is excluded
            if _is_experiment_excluded(entry, subentry, exclusion_patterns):
                continue

            # Get experiment-specific TOF parameters for energy resolution
            energy_res = meta.get('energy_resolution_inputs')

            if (energy_res and
                energy_res.get('flight_path_m') is not None and
                energy_res.get('time_resolution_ns') is not None):
                # Compute experiment-specific sigma_E at the MEASUREMENT energy
                # This is important: use available_energy, not energy_mev
                exp_sigma_E = compute_energy_resolution_tof(
                    E_mev=available_energy,  # Use measurement energy
                    delta_t_ns=energy_res['time_resolution_ns'],
                    flight_path_m=energy_res['flight_path_m'],
                )
                used_fallback = False
            else:
                # Fallback to default parameters at measurement energy
                exp_sigma_E = compute_energy_resolution_tof(
                    E_mev=available_energy,  # Use measurement energy
                    delta_t_ns=default_delta_t_ns,
                    flight_path_m=default_flight_path_m,
                )
                used_fallback = True

            # Compute weight based on method
            if use_overlap_weights:
                # Resolution overlap: probability that true energy is in bin
                kernel_weight = compute_overlap_weight(
                    exp_energy_mev=available_energy,
                    sigma_E_mev=exp_sigma_E,
                    bin_lower_mev=bin_lower_mev,
                    bin_upper_mev=bin_upper_mev,
                )
                # Skip if weight is negligible
                if kernel_weight < min_kernel_weight_fraction:
                    continue
            else:
                # Legacy Gaussian kernel weighting
                # Check cutoff using THIS experiment's sigma_E
                exp_cutoff = n_sigma * exp_sigma_E
                if available_energy < (energy_mev - exp_cutoff) or available_energy > (energy_mev + exp_cutoff):
                    continue
                # Compute Gaussian kernel weight
                delta_E = abs(available_energy - energy_mev)
                kernel_weight = np.exp(-0.5 * (delta_E / exp_sigma_E)**2)

            exp_key = (entry, subentry)
            experiment_candidates[exp_key].append((available_energy, df, meta, kernel_weight, exp_sigma_E, used_fallback))

    # Step 2: For each experiment, select highest-weighted energy (or all if dedupe disabled)
    selected_data: List[Tuple[float, pd.DataFrame, Dict, float, float, bool]] = []

    for exp_key, candidates in experiment_candidates.items():
        if dedupe_per_experiment and len(candidates) > 1:
            # Select the energy with highest kernel weight
            best = max(candidates, key=lambda x: x[3])  # x[3] = kernel_weight
            selected_data.append(best)
        else:
            selected_data.extend(candidates)

    # Step 3: Process selected data (transform and build DataFrames)
    for available_energy, df, meta, kernel_weight, exp_sigma_E, used_fallback in selected_data:
        # Extract metadata
        entry = meta.get('entry', 'unknown')
        subentry = meta.get('subentry', 'unknown')
        frame = meta.get('angle_frame', 'CM').upper()
        reaction = meta.get('reaction', '')

        # Log fallback only once per experiment (subentry)
        if used_fallback and logger and subentry not in fallback_logged:
            logger.info(f"  Using default TOF params for {subentry} "
                       f"(delta_t={default_delta_t_ns}ns, L={default_flight_path_m}m)")
            fallback_logged.add(subentry)

        citation = meta.get('citation', {})
        authors = citation.get('authors', [])
        author = authors[0] if authors else 'unknown'
        year = citation.get('year', 'unknown')

        # Extract columns
        angles_deg = df['angle'].to_numpy(dtype=float)
        dsig = df['dsig'].to_numpy(dtype=float)
        error_stat = df['error_stat'].to_numpy(dtype=float)

        n_points = len(angles_deg)

        # Count how many energies this experiment had in range
        n_energies_in_range = len(experiment_candidates[(entry, subentry)])

        # Per-energy normalization (Upgrade 2)
        # Divide weight by number of angular points to prevent
        # experiments with dense angular sampling from dominating
        if normalize_by_n_points and n_points > 0:
            point_weight = kernel_weight / n_points
        else:
            point_weight = kernel_weight

        # Transform to CM frame if needed
        if frame == 'LAB':
            mu_lab = np.cos(np.deg2rad(angles_deg))
            mu_cm, dsig_cm, error_cm = transform_lab_to_cm(
                mu_lab, dsig, error_stat, m_proj_u, m_targ_u
            )

            angles_cm_deg = np.rad2deg(np.arccos(mu_cm))

            transformed_df = pd.DataFrame({
                'theta_deg': angles_cm_deg,
                'value': dsig_cm,
                'unc': error_cm,
                'mu': mu_cm,
                'frame': 'CM',
                'entry': entry,
                'subentry': subentry,
                'author': author,
                'year': year,
                'reaction': reaction,
                'exfor_energy_mev': available_energy,
                'kernel_weight': point_weight,  # Use normalized point weight
            })
        else:
            mu_cm = np.cos(np.deg2rad(angles_deg))

            transformed_df = pd.DataFrame({
                'theta_deg': angles_deg,
                'value': dsig,
                'unc': error_stat,
                'mu': mu_cm,
                'frame': frame,
                'entry': entry,
                'subentry': subentry,
                'author': author,
                'year': year,
                'reaction': reaction,
                'exfor_energy_mev': available_energy,
                'kernel_weight': point_weight,  # Use normalized point weight
            })

        all_frames.append(transformed_df)
        all_kernel_weights.extend([point_weight] * n_points)

        # Track experiment info (store original kernel_weight for diagnostics)
        exp_info = {
            'entry': entry,
            'subentry': subentry,
            'author': author,
            'year': year,
            'exfor_energy_mev': available_energy,
            'kernel_weight': kernel_weight,  # Original weight before normalization
            'point_weight': point_weight,    # Weight after per-energy normalization
            'n_points': n_points,
            'sigma_E_mev': exp_sigma_E,
            'used_fallback_tof': used_fallback,
            'selected_from_n_energies': n_energies_in_range,  # NEW: track deduplication
        }
        experiments_info.append(exp_info)

    if not all_frames:
        return pd.DataFrame(), [], np.array([]), empty_diag

    # Concatenate all experiments
    result = pd.concat(all_frames, ignore_index=True)
    kernel_weights = np.array(all_kernel_weights)

    # Apply uncertainty floor if requested
    if min_relative_uncertainty > 0:
        result = apply_uncertainty_floor(result, min_relative_uncertainty, unc_column='unc', value_column='value')

    # Apply minimum weight threshold
    kernel_weights, n_dropped = apply_min_weight_threshold(
        kernel_weights, min_kernel_weight_fraction
    )

    # Apply per-experiment weight capping
    kernel_weights, exp_weight_fracs, capping_applied = apply_per_experiment_weight_cap(
        result, kernel_weights, max_experiment_weight_fraction
    )

    # Update kernel_weight column in DataFrame
    result['kernel_weight'] = kernel_weights

    # Compute diagnostics
    exfor_energies = result['exfor_energy_mev'].values
    weight_span_95 = compute_weight_span_95(kernel_weights, exfor_energies, energy_mev)
    n_eff_prelim = compute_n_eff(kernel_weights, np.ones_like(kernel_weights))
    max_exp_frac = max(exp_weight_fracs.values()) if exp_weight_fracs else 0.0

    diagnostics = KernelDiagnostics(
        n_eff=n_eff_prelim,
        weight_span_95=weight_span_95,
        weight_span_ratio=weight_span_95 / sigma_E_mev if sigma_E_mev > 0 else 0.0,
        n_experiments=len(set(exp_weight_fracs.keys())) if exp_weight_fracs else len(experiments_info),
        max_experiment_weight_frac=max_exp_frac,
        experiment_weights=exp_weight_fracs,
        n_points_dropped=n_dropped,
        capping_applied=capping_applied,
    )

    return result, experiments_info, kernel_weights, diagnostics


def filter_exfor_with_energy_bin(
    exfor_cache: Dict[float, List[Tuple[pd.DataFrame, Dict]]],
    sorted_energies: List[float],
    bin_lower_mev: float,
    bin_upper_mev: float,
    target_energy_mev: float,
    m_proj_u: float,
    m_targ_u: float,
    dedupe_per_experiment: bool = True,
    exclude_experiments: Optional[List[str]] = None,
    min_relative_uncertainty: float = 0.0,
    # NEW: per-experiment weighting options (Improvement 1.1)
    normalize_by_n_points: bool = False,
    max_experiment_weight_fraction: float = 1.0,  # 1.0 = disabled
) -> Tuple[pd.DataFrame, List[Dict], np.ndarray, KernelDiagnostics]:
    """
    Filter EXFOR data using exact energy bin matching.

    Unlike Gaussian kernel weighting, this method:
    - Selects all experiments whose energy falls within [bin_lower, bin_upper]
    - When dedupe_per_experiment=True, selects only the closest energy per experiment
    - Assigns uniform weight = 1.0 to all selected points
    - Does NOT apply per-experiment weight capping

    Parameters
    ----------
    exfor_cache : Dict[float, List[Tuple[pd.DataFrame, Dict]]]
        Pre-loaded EXFOR data organized by energy
    sorted_energies : List[float]
        Sorted list of available energies in cache (in MeV)
    bin_lower_mev : float
        Lower bin boundary in MeV
    bin_upper_mev : float
        Upper bin boundary in MeV
    target_energy_mev : float
        Target ENDF grid energy in MeV (for diagnostics)
    m_proj_u : float
        Projectile mass in atomic mass units
    m_targ_u : float
        Target mass in atomic mass units
    dedupe_per_experiment : bool
        If True (default), select only the closest energy to target_energy_mev
        for each experiment. This prevents experiments with many energies in
        the bin from dominating the fit.
    exclude_experiments : List[str], optional
        List of experiments to exclude from filtering. Accepts multiple formats:
        - "20743" - excludes all subentries starting with 20743
        - "20743002" - excludes specific dataset
        - "20743/002" - same as above
    min_relative_uncertainty : float, optional
        Minimum relative uncertainty as a fraction (default: 0.0 = disabled).
        For example, 0.03 means 3% minimum uncertainty.
    normalize_by_n_points : bool, optional
        If True, each point's weight is 1/n_points_for_this_experiment, so each
        experiment contributes equally regardless of how many points it has.
        Default: False (uniform weights).
    max_experiment_weight_fraction : float, optional
        Maximum allowed weight fraction per experiment (default: 1.0 = disabled).
        If < 1.0, experiments exceeding this fraction are scaled down.
        Applied AFTER normalize_by_n_points if both are enabled.

    Returns
    -------
    Tuple[pd.DataFrame, List[Dict], np.ndarray, KernelDiagnostics]
        - DataFrame with EXFOR data (kernel_weight = 1.0 for all)
        - List of experiment metadata dicts (includes 'selected_from_n_energies')
        - Array of kernel weights
        - KernelDiagnostics object
    """
    # Empty diagnostics for early returns
    empty_diag = KernelDiagnostics(
        n_eff=0.0, weight_span_95=0.0, weight_span_ratio=0.0,
        n_experiments=0, max_experiment_weight_frac=0.0,
        experiment_weights={}, n_points_dropped=0, capping_applied=False
    )

    if not exfor_cache or not sorted_energies:
        return pd.DataFrame(), [], np.array([]), empty_diag

    # Parse exclusion patterns
    exclusion_patterns = _parse_exclusion_list(exclude_experiments)

    all_frames = []
    experiments_info = []

    # Step 1: Collect all candidate data grouped by experiment
    # experiment_candidates: {(entry, subentry): [(energy, df, meta), ...]}
    experiment_candidates: Dict[Tuple[str, str], List[Tuple[float, pd.DataFrame, Dict]]] = defaultdict(list)

    for available_energy in sorted_energies:
        # Exact bin matching - include if within [lower, upper]
        if available_energy < bin_lower_mev or available_energy > bin_upper_mev:
            continue

        entries = exfor_cache.get(available_energy, [])
        for df, meta in entries:
            entry = meta.get('entry', 'unknown')
            subentry = meta.get('subentry', 'unknown')

            # Check if experiment is excluded
            if _is_experiment_excluded(entry, subentry, exclusion_patterns):
                continue

            exp_key = (entry, subentry)
            experiment_candidates[exp_key].append((available_energy, df, meta))

    # Step 2: For each experiment, select closest energy to target (or all if dedupe disabled)
    selected_data: List[Tuple[float, pd.DataFrame, Dict]] = []

    for exp_key, candidates in experiment_candidates.items():
        if dedupe_per_experiment and len(candidates) > 1:
            # Select the energy closest to target
            closest = min(candidates, key=lambda x: abs(x[0] - target_energy_mev))
            selected_data.append(closest)
        else:
            selected_data.extend(candidates)

    # Step 2b: Count total points per experiment for weighting (Improvement 1.1)
    exp_n_points_map: Dict[Tuple[str, str], int] = {}
    if normalize_by_n_points:
        for available_energy, df, meta in selected_data:
            entry = meta.get('entry', 'unknown')
            subentry = meta.get('subentry', 'unknown')
            exp_key = (entry, subentry)
            n_pts = len(df['angle'])
            exp_n_points_map[exp_key] = exp_n_points_map.get(exp_key, 0) + n_pts

    # Also track kernel weights per row for final assembly
    all_kernel_weights: List[float] = []

    # Step 3: Process selected data (transform and build DataFrames)
    for available_energy, df, meta in selected_data:
        # Extract metadata (same as Gaussian kernel method)
        entry = meta.get('entry', 'unknown')
        subentry = meta.get('subentry', 'unknown')
        exp_key = (entry, subentry)

        n_points = len(df['angle'])

        # Determine kernel weight per point (Improvement 1.1)
        if normalize_by_n_points and exp_key in exp_n_points_map:
            # Each point gets 1/n_points for this experiment
            kernel_weight = 1.0 / exp_n_points_map[exp_key]
        else:
            # Uniform weight for bin method (no Gaussian decay)
            kernel_weight = 1.0

        # Extract metadata (same as Gaussian kernel method)
        frame = meta.get('angle_frame', 'CM').upper()
        reaction = meta.get('reaction', '')

        citation = meta.get('citation', {})
        authors = citation.get('authors', [])
        author = authors[0] if authors else 'unknown'
        year = citation.get('year', 'unknown')

        # Extract columns
        angles_deg = df['angle'].to_numpy(dtype=float)
        dsig = df['dsig'].to_numpy(dtype=float)
        error_stat = df['error_stat'].to_numpy(dtype=float)

        # Count how many energies this experiment had in the bin
        n_energies_in_bin = len(experiment_candidates[(entry, subentry)])

        # Transform to CM frame if needed (same logic as Gaussian method)
        if frame == 'LAB':
            mu_lab = np.cos(np.deg2rad(angles_deg))
            mu_cm, dsig_cm, error_cm = transform_lab_to_cm(
                mu_lab, dsig, error_stat, m_proj_u, m_targ_u
            )

            angles_cm_deg = np.rad2deg(np.arccos(mu_cm))

            transformed_df = pd.DataFrame({
                'theta_deg': angles_cm_deg,
                'value': dsig_cm,
                'unc': error_cm,
                'mu': mu_cm,
                'frame': 'CM',
                'entry': entry,
                'subentry': subentry,
                'author': author,
                'year': year,
                'reaction': reaction,
                'exfor_energy_mev': available_energy,
                'kernel_weight': kernel_weight,
            })
        else:
            mu_cm = np.cos(np.deg2rad(angles_deg))

            transformed_df = pd.DataFrame({
                'theta_deg': angles_deg,
                'value': dsig,
                'unc': error_stat,
                'mu': mu_cm,
                'frame': frame,
                'entry': entry,
                'subentry': subentry,
                'author': author,
                'year': year,
                'reaction': reaction,
                'exfor_energy_mev': available_energy,
                'kernel_weight': kernel_weight,
            })

        all_frames.append(transformed_df)

        # Track kernel weights for each row in this dataframe
        all_kernel_weights.extend([kernel_weight] * n_points)

        # Track experiment info with deduplication info
        exp_info = {
            'entry': entry,
            'subentry': subentry,
            'author': author,
            'year': year,
            'exfor_energy_mev': available_energy,
            'kernel_weight': kernel_weight,
            'n_points': n_points,
            'selected_from_n_energies': n_energies_in_bin,  # NEW: track deduplication
        }
        experiments_info.append(exp_info)

    if not all_frames:
        return pd.DataFrame(), [], np.array([]), empty_diag

    # Concatenate all experiments
    result = pd.concat(all_frames, ignore_index=True)
    kernel_weights = np.array(all_kernel_weights, dtype=float)

    # Apply uncertainty floor if requested
    if min_relative_uncertainty > 0:
        result = apply_uncertainty_floor(result, min_relative_uncertainty, unc_column='unc', value_column='value')

    # Apply per-experiment weight capping if requested (Improvement 1.1)
    capping_applied = False
    if max_experiment_weight_fraction < 1.0:
        kernel_weights, exp_weight_fracs, capping_applied = apply_per_experiment_weight_cap(
            result, kernel_weights, max_experiment_weight_fraction
        )
    else:
        # Compute experiment weight fractions (for logging)
        exp_weight_fracs = {}
        total_weight = float(np.sum(kernel_weights))
        if total_weight > 1e-30:
            for exp in experiments_info:
                key = f"{exp['entry']}.{exp['subentry']}"
                # Sum weights for this experiment
                exp_mask = (result['entry'] == exp['entry']) & (result['subentry'] == exp['subentry'])
                exp_total = float(np.sum(kernel_weights[exp_mask.values]))
                exp_weight_fracs[key] = exp_total / total_weight

    # Compute diagnostics
    # N_eff = (sum w)^2 / sum(w^2)
    total_weight = float(np.sum(kernel_weights))
    sum_w_sq = float(np.sum(kernel_weights ** 2))
    n_eff = (total_weight ** 2) / sum_w_sq if sum_w_sq > 1e-30 else float(len(result))

    # Weight span is the bin width
    weight_span = bin_upper_mev - bin_lower_mev if bin_upper_mev < float('inf') else 0.0

    diagnostics = KernelDiagnostics(
        n_eff=n_eff,
        weight_span_95=weight_span,
        weight_span_ratio=0.0,  # Not applicable for bin method
        n_experiments=len(set(exp_weight_fracs.keys())),
        max_experiment_weight_frac=max(exp_weight_fracs.values()) if exp_weight_fracs else 0.0,
        experiment_weights=exp_weight_fracs,
        n_points_dropped=0,
        capping_applied=capping_applied,
    )

    return result, experiments_info, kernel_weights, diagnostics


def apply_min_weight_threshold(
    kernel_weights: np.ndarray,
    min_weight_fraction: float = 1e-3,
) -> Tuple[np.ndarray, int]:
    """
    Zero out kernel weights below threshold.

    Points with g_ij < min_weight_fraction * max(g_ij) are set to zero weight.

    Parameters
    ----------
    kernel_weights : np.ndarray
        Gaussian kernel weights
    min_weight_fraction : float
        Minimum weight as fraction of maximum (default: 1e-3)

    Returns
    -------
    Tuple[np.ndarray, int]
        - filtered_weights: Weights with below-threshold points zeroed
        - n_dropped: Number of points dropped
    """
    if len(kernel_weights) == 0:
        return kernel_weights.copy(), 0

    g_max = np.max(kernel_weights)
    if g_max < 1e-30:
        return kernel_weights.copy(), 0

    threshold = min_weight_fraction * g_max

    filtered = kernel_weights.copy()
    mask = kernel_weights < threshold
    filtered[mask] = 0.0

    return filtered, int(np.sum(mask))


def apply_uncertainty_floor(
    exfor_df: pd.DataFrame,
    min_relative_uncertainty: float = 0.0,
    unc_column: str = "unc",
    value_column: str = "value",
) -> pd.DataFrame:
    """
    Apply minimum relative uncertainty floor to prevent experiments with
    unrealistically small uncertainties from dominating fits.

    For each data point, enforces: unc >= min_relative_uncertainty * |value|

    This is a safety mechanism to handle cases where uncertainties may be
    incorrectly reported or processed in the database.

    Parameters
    ----------
    exfor_df : pd.DataFrame
        EXFOR data with uncertainty and value columns
    min_relative_uncertainty : float
        Minimum relative uncertainty as a fraction (default: 0.0 = disabled).
        For example, 0.03 means 3% minimum uncertainty.
    unc_column : str
        Column name for uncertainties (default: 'unc')
    value_column : str
        Column name for cross section values (default: 'value')

    Returns
    -------
    pd.DataFrame
        Copy of DataFrame with updated uncertainties (or original if disabled)

    Examples
    --------
    >>> df = apply_uncertainty_floor(exfor_df, min_relative_uncertainty=0.03)
    >>> # Now all points have at least 3% relative uncertainty
    """
    if min_relative_uncertainty <= 0:
        return exfor_df

    df = exfor_df.copy()
    if unc_column not in df.columns or value_column not in df.columns:
        return df

    floor = min_relative_uncertainty * np.abs(df[value_column])
    df[unc_column] = np.maximum(df[unc_column], floor)
    return df


def apply_per_experiment_weight_cap(
    exfor_df: pd.DataFrame,
    kernel_weights: np.ndarray,
    max_experiment_weight_fraction: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, float], bool]:
    """
    Cap per-experiment total kernel weight to prevent dense experiments from dominating.

    Parameters
    ----------
    exfor_df : pd.DataFrame
        EXFOR data with 'entry', 'subentry' columns
    kernel_weights : np.ndarray
        Gaussian kernel weights (one per data point)
    max_experiment_weight_fraction : float
        Maximum allowed weight fraction per experiment (default: 0.5)

    Returns
    -------
    Tuple[np.ndarray, Dict[str, float], bool]
        - capped_weights: Adjusted kernel weights
        - experiment_weight_fracs: {exp_key: weight_fraction} BEFORE capping
        - capping_applied: Whether any capping was done
    """
    if max_experiment_weight_fraction >= 1.0:
        # Capping disabled
        return kernel_weights.copy(), {}, False

    if len(kernel_weights) == 0:
        return kernel_weights.copy(), {}, False

    # Build experiment key for each point
    entries = exfor_df['entry'].values
    subentries = exfor_df['subentry'].values
    n_points = len(kernel_weights)
    exp_keys = [f"{entries[i]}.{subentries[i]}" for i in range(n_points)]

    # Compute total weight per experiment
    exp_weights: Dict[str, float] = {}
    for i, key in enumerate(exp_keys):
        exp_weights[key] = exp_weights.get(key, 0.0) + kernel_weights[i]

    total_weight = np.sum(kernel_weights)
    if total_weight < 1e-30:
        return kernel_weights.copy(), {}, False

    # Compute fractions BEFORE capping (for diagnostics)
    exp_weight_fracs = {k: v / total_weight for k, v in exp_weights.items()}

    # Edge case: only one experiment - cannot cap
    if len(exp_weights) == 1:
        return kernel_weights.copy(), exp_weight_fracs, False

    # Apply capping
    capped_weights = kernel_weights.copy()
    capping_applied = False
    cap = max_experiment_weight_fraction

    for exp_key, frac in exp_weight_fracs.items():
        if frac > cap:
            # Scale factor to bring this experiment down to the cap
            scale = cap / frac

            # Apply scale to all points from this experiment
            for i, key in enumerate(exp_keys):
                if key == exp_key:
                    capped_weights[i] *= scale

            capping_applied = True

    return capped_weights, exp_weight_fracs, capping_applied


def load_exfor_with_asymmetric_tolerance(
    exfor_directory: str,
    energy_mev: float,
    tolerance_lower_mev: float,
    tolerance_upper_mev: float,
    m_proj_u: float,
    m_targ_u: float,
) -> Tuple[pd.DataFrame, int]:
    """
    Load EXFOR data with asymmetric tolerance bounds.

    Parameters
    ----------
    exfor_directory : str
        Path to EXFOR data directory
    energy_mev : float
        Target energy in MeV
    tolerance_lower_mev : float
        Lower tolerance in MeV
    tolerance_upper_mev : float
        Upper tolerance in MeV
    m_proj_u : float
        Projectile mass in atomic mass units
    m_targ_u : float
        Target mass in atomic mass units

    Returns
    -------
    Tuple[pd.DataFrame, int]
        DataFrame with EXFOR data and count of unique energies found
    """
    # Use the maximum tolerance for initial search
    max_tolerance = max(tolerance_lower_mev, tolerance_upper_mev)

    # Suppress print statements from load_exfor_for_fitting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            exfor_df = load_exfor_for_fitting(
                exfor_directory=exfor_directory,
                energy_mev=energy_mev,
                tolerance=max_tolerance,
                m_proj_u=m_proj_u,
                m_targ_u=m_targ_u,
            )
        finally:
            sys.stdout = old_stdout

    if exfor_df.empty:
        return exfor_df, 0

    # Count unique experiments (entry, subentry pairs)
    if 'entry' in exfor_df.columns and 'subentry' in exfor_df.columns:
        unique_experiments = exfor_df.groupby(['entry', 'subentry']).size()
        n_experiments = len(unique_experiments)
    else:
        n_experiments = 1

    return exfor_df, n_experiments


# =============================================================================
# ENERGY JITTER MC (Improvement 1.4)
# =============================================================================

def precompute_dataset_energy_info(
    nominal_results: List,  # List[NominalFitResult]
    tof_params_cache: Dict[str, Dict],
    energy_bins: List[EnergyBinInfo],
    default_flight_path_m: float = 27.037,
    default_time_resolution_ns: float = 5.0,
) -> Dict[int, List[DatasetEnergyInfo]]:
    """
    Precompute σE for all datasets across all bins.

    This function prepares dataset information needed for the energy jitter MC:
    - Extracts all datasets from nominal fit results
    - Computes σE for each dataset using experiment-specific TOF parameters
    - Maps datasets to their nominal (unperturbed) energy bins

    Parameters
    ----------
    nominal_results : List[NominalFitResult]
        Nominal fit results from Phase 1
    tof_params_cache : Dict[str, Dict]
        Pre-loaded TOF parameters from load_tof_parameters_file()
    energy_bins : List[EnergyBinInfo]
        List of energy bin info objects
    default_flight_path_m : float
        Default flight path in meters for fallback
    default_time_resolution_ns : float
        Default time resolution in nanoseconds for fallback

    Returns
    -------
    Dict[int, List[DatasetEnergyInfo]]
        Dictionary mapping bin index to list of DatasetEnergyInfo for datasets
        whose nominal energy falls in that bin
    """
    from .tof_parameters import get_tof_parameters, compute_sigma_E, find_bin_for_energy

    dataset_info_by_bin: Dict[int, List[DatasetEnergyInfo]] = {
        bin_info.index: [] for bin_info in energy_bins
    }

    for nr in nominal_results:
        if not nr.has_data or nr.interpolated:
            continue

        # Each experiment_info entry represents one dataset (experiment at one energy)
        for exp_info in nr.experiments_info:
            entry = exp_info.get('entry', 'unknown')
            subentry = exp_info.get('subentry', 'unknown')
            exfor_energy = exp_info.get('exfor_energy_mev', nr.energy_mev)
            n_points = exp_info.get('n_points', 0)

            # Get TOF parameters for this experiment
            tof_params = get_tof_parameters(
                subentry=f"{entry}{subentry}",  # Combined identifier
                tof_params_cache=tof_params_cache,
                default_flight_path_m=default_flight_path_m,
                default_time_resolution_ns=default_time_resolution_ns,
            )

            # Compute σE at this dataset's energy
            sigma_E = compute_sigma_E(exfor_energy, tof_params)

            # Find nominal bin (the bin containing this dataset's energy)
            nominal_bin_idx = find_bin_for_energy(exfor_energy, energy_bins)
            if nominal_bin_idx is None:
                # Dataset energy outside all bins - skip
                continue

            # Create dataset info
            dataset_info = DatasetEnergyInfo(
                entry=entry,
                subentry=subentry,
                nominal_energy_mev=exfor_energy,
                sigma_E_mev=sigma_E,
                nominal_bin_index=nominal_bin_idx,
                exfor_df_indices=[],  # Populated if needed for building combined DataFrames
                tof_source=tof_params.source,
                n_points=n_points,
            )

            dataset_info_by_bin[nominal_bin_idx].append(dataset_info)

    return dataset_info_by_bin


def get_all_datasets_flat(
    dataset_info_by_bin: Dict[int, List[DatasetEnergyInfo]],
) -> List[DatasetEnergyInfo]:
    """
    Flatten dataset info from bin-grouped dict to single list.

    Parameters
    ----------
    dataset_info_by_bin : Dict[int, List[DatasetEnergyInfo]]
        Datasets grouped by nominal bin

    Returns
    -------
    List[DatasetEnergyInfo]
        All datasets as a flat list
    """
    all_datasets = []
    for datasets in dataset_info_by_bin.values():
        all_datasets.extend(datasets)
    return all_datasets


def _interpolate_from_neighbors(
    fitted_bins: Dict[int, np.ndarray],
    energy_bins: list,
    target_idx: int,
    max_degree: int,
) -> Optional[np.ndarray]:
    """
    Interpolate Legendre coefficients for an empty bin from same-sample
    fitted neighbors using linear interpolation in energy space.

    Parameters
    ----------
    fitted_bins : dict
        Mapping bin_index -> fitted coefficient array for bins that
        were successfully fit in this MC sample.
    energy_bins : list of EnergyBinInfo
        All energy bins (sorted by energy).
    target_idx : int
        The bin index to interpolate.
    max_degree : int
        Maximum Legendre degree; output is zero-padded to this length.

    Returns
    -------
    np.ndarray or None
        Interpolated coefficients of length *max_degree*, or None if no
        fitted neighbors exist at all.
    """
    # Build an energy lookup for the target bin
    idx_to_energy = {b.index: b.energy_mev for b in energy_bins}
    target_energy = idx_to_energy[target_idx]

    # Sorted fitted indices by energy
    sorted_fitted = sorted(fitted_bins.keys(), key=lambda i: idx_to_energy[i])

    if not sorted_fitted:
        return None

    # Find nearest lower and upper fitted bins
    lower_idx = None
    upper_idx = None
    for idx in sorted_fitted:
        if idx_to_energy[idx] < target_energy:
            lower_idx = idx
        elif idx_to_energy[idx] > target_energy:
            upper_idx = idx
            break

    def _pad(arr: np.ndarray) -> np.ndarray:
        out = np.zeros(max_degree, dtype=float)
        n = min(len(arr), max_degree)
        out[:n] = arr[:n]
        return out

    if lower_idx is not None and upper_idx is not None:
        # Linear interpolation
        e_lo = idx_to_energy[lower_idx]
        e_hi = idx_to_energy[upper_idx]
        t = (target_energy - e_lo) / (e_hi - e_lo)
        c_lo = _pad(fitted_bins[lower_idx])
        c_hi = _pad(fitted_bins[upper_idx])
        return (1.0 - t) * c_lo + t * c_hi

    if lower_idx is not None:
        # Nearest-neighbor extrapolation (upper edge)
        return _pad(fitted_bins[lower_idx])

    if upper_idx is not None:
        # Nearest-neighbor extrapolation (lower edge)
        return _pad(fitted_bins[upper_idx])

    return None


def _run_one_jitter_sample(args):
    """
    Run a single MC sample with energy jitter (top-level for pickling).

    Parameters
    ----------
    args : tuple
        All inputs packed for Pool.map compatibility.

    Returns
    -------
    tuple
        (s_idx, sample_coeffs, local_jump_stats)
    """
    from .tof_parameters import find_bin_for_energy
    from .resample_AD import (
        sample_legendre_coefficients,
        check_angular_distribution_positivity,
        project_to_positive_distribution,
    )

    (
        s_idx,
        all_datasets,
        energy_bins,
        nominal_by_idx,
        dataset_exfor_lookup,
        base_seed,
        max_degree,
        ridge_lambda,
        m_proj_u,
        m_targ_u,
        use_band_discrepancy,
        min_points_per_band,
        max_tau_fraction,
        jitter_n_sigma_clip,
        track_bin_jumps,
        min_relative_uncertainty,
        freeze_c0,
        sigma_norm,
        norm_dist,
        normalize_by_n_points,
        max_experiment_weight_fraction,
        _apply_positivity_projection,
        _positivity_check_points,
        _stochastic_flag,
    ) = args

    rng = np.random.default_rng(base_seed + s_idx * 1000)

    # Step 1: Jitter all dataset energies and assign to bins
    bin_assignments: Dict[int, List[Tuple[DatasetEnergyInfo, float]]] = {
        bin_info.index: [] for bin_info in energy_bins
    }

    local_total_assignments = 0
    local_jumped_bins = 0
    local_jump_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    local_datasets_outside_range = 0

    for dataset in all_datasets:
        z = np.clip(
            rng.normal(0, 1),
            -jitter_n_sigma_clip,
            jitter_n_sigma_clip
        )
        E_star = dataset.nominal_energy_mev + z * dataset.sigma_E_mev
        target_bin = find_bin_for_energy(E_star, energy_bins)

        if target_bin is None:
            local_datasets_outside_range += 1
            continue

        local_total_assignments += 1

        if track_bin_jumps and target_bin != dataset.nominal_bin_index:
            local_jumped_bins += 1
            jump_key = (dataset.nominal_bin_index, target_bin)
            local_jump_counts[jump_key] += 1

        bin_assignments[target_bin].append((dataset, E_star))

    # Step 1a: Draw shared normalization factor per experiment
    # Placed after jitter loop to preserve RNG sequence when sigma_norm=0
    experiment_norm_factors = {}
    if sigma_norm > 0:
        unique_exps = set((ds.entry, ds.subentry) for ds in all_datasets)
        for exp_key in unique_exps:
            if norm_dist == "lognormal":
                experiment_norm_factors[exp_key] = rng.lognormal(0.0, sigma_norm)
            else:
                experiment_norm_factors[exp_key] = 1.0 + rng.normal(0.0, sigma_norm)

    # Step 1b: Compute per-bin experiment point counts for weighting
    bin_exp_n_points: Dict[int, Dict[Tuple[str, str], int]] = {
        bin_info.index: {} for bin_info in energy_bins
    }
    if normalize_by_n_points:
        for bin_idx, assigned in bin_assignments.items():
            for dataset, E_star in assigned:
                exp_key = (dataset.entry, dataset.subentry)
                current = bin_exp_n_points[bin_idx].get(exp_key, 0)
                bin_exp_n_points[bin_idx][exp_key] = current + dataset.n_points

    # Step 2: Two-pass fitting
    #   Pass 1 – fit bins that have sufficient data
    #   Pass 2 – interpolate empty/failed bins from same-sample neighbors

    sample_coeffs: Dict[int, np.ndarray] = {}
    fitted_bins: Dict[int, np.ndarray] = {}   # bins successfully fitted
    empty_bins: List[int] = []                 # bins that need interpolation

    # --- Pass 1: attempt to fit every bin ---
    for bin_info in energy_bins:
        assigned = bin_assignments[bin_info.index]

        if not assigned:
            empty_bins.append(bin_info.index)
            continue

        # Build combined DataFrame from assigned datasets
        frames = []
        for dataset, E_star in assigned:
            key = (dataset.entry, dataset.subentry, dataset.nominal_energy_mev)
            if key not in dataset_exfor_lookup:
                continue

            df, meta = dataset_exfor_lookup[key]

            angles_deg = df['angle'].to_numpy(dtype=float)
            dsig = df['dsig'].to_numpy(dtype=float)
            error_stat = df['error_stat'].to_numpy(dtype=float)

            frame = meta.get('angle_frame', 'CM').upper()
            if frame == 'LAB':
                mu_lab = np.cos(np.deg2rad(angles_deg))
                mu_cm, dsig_cm, error_cm = transform_lab_to_cm(
                    mu_lab, dsig, error_stat, m_proj_u, m_targ_u
                )
                angles_cm_deg = np.rad2deg(np.arccos(mu_cm))
            else:
                mu_cm = np.cos(np.deg2rad(angles_deg))
                angles_cm_deg = angles_deg
                dsig_cm = dsig
                error_cm = error_stat

            # Apply shared experiment normalization
            exp_key_norm = (dataset.entry, dataset.subentry)
            N_g = experiment_norm_factors.get(exp_key_norm, 1.0)
            if N_g != 1.0:
                dsig_cm = dsig_cm * N_g
                error_cm = error_cm * N_g   # preserves relative uncertainty

            exp_key = (dataset.entry, dataset.subentry)
            if normalize_by_n_points and exp_key in bin_exp_n_points[bin_info.index]:
                total_pts = bin_exp_n_points[bin_info.index][exp_key]
                kw = 1.0 / total_pts if total_pts > 0 else 1.0
            else:
                kw = 1.0

            transformed_df = pd.DataFrame({
                'theta_deg': angles_cm_deg,
                'value': dsig_cm,
                'unc': error_cm,
                'mu': mu_cm,
                'entry': dataset.entry,
                'subentry': dataset.subentry,
                'exfor_energy_mev': E_star,
                'kernel_weight': kw,
            })
            frames.append(transformed_df)

        if not frames:
            empty_bins.append(bin_info.index)
            continue

        combined_df = pd.concat(frames, ignore_index=True)

        # Apply per-experiment weight capping
        if max_experiment_weight_fraction < 1.0 and len(combined_df) > 0:
            kernel_weights_arr = combined_df['kernel_weight'].to_numpy()
            kernel_weights_arr, _, _ = apply_per_experiment_weight_cap(
                combined_df, kernel_weights_arr, max_experiment_weight_fraction
            )
            combined_df['kernel_weight'] = kernel_weights_arr

        # Apply uncertainty floor
        if min_relative_uncertainty > 0:
            combined_df = apply_uncertainty_floor(
                combined_df, min_relative_uncertainty,
                unc_column='unc', value_column='value'
            )

        # Pre-inflate uncertainties with nominal tau
        if use_band_discrepancy and bin_info.index in nominal_by_idx:
            nr = nominal_by_idx[bin_info.index]
            tau_info = nr.tau_info
            mu_arr = combined_df['mu'].to_numpy()
            unc_arr = combined_df['unc'].to_numpy().copy()

            band_map = {
                'tau_F': mu_arr > 0.5,
                'tau_M': (mu_arr >= -0.5) & (mu_arr <= 0.5),
                'tau_B': mu_arr < -0.5,
            }
            for tau_key, mask in band_map.items():
                tau_b = tau_info.get(tau_key, 0.0)
                if tau_b > 0 and np.any(mask):
                    unc_arr[mask] = np.sqrt(unc_arr[mask]**2 + tau_b**2)

            combined_df['unc'] = unc_arr
            use_band_for_fit = False
        else:
            use_band_for_fit = use_band_discrepancy

        # Get frozen degree from nominal fit
        if bin_info.index in nominal_by_idx:
            frozen_degree = nominal_by_idx[bin_info.index].frozen_degree
        else:
            frozen_degree = min(max_degree, len(combined_df) // 3)

        if len(combined_df) < 3:
            empty_bins.append(bin_info.index)
            continue

        # Fit Legendre coefficients
        try:
            coef_df, _ = sample_legendre_coefficients(
                combined_df,
                value_col="value",
                unc_col="unc",
                degree=frozen_degree,
                max_degree=max_degree,
                select_degree=None,
                ridge_lambda=ridge_lambda,
                external_weights=combined_df['kernel_weight'].to_numpy(),
                n_samples=1,
                stochastic=_stochastic_flag,
                random_state=base_seed + s_idx * 1000 + bin_info.index,
                use_band_discrepancy=use_band_for_fit,
                min_points_per_band=min_points_per_band,
                max_tau_fraction=max_tau_fraction,
                freeze_c0=freeze_c0,
                sigma_norm=0.0,  # normalization handled above (shared across bins)
                norm_dist=norm_dist,
            )

            fit_coeffs = coef_df.iloc[0].to_numpy()
            if _apply_positivity_projection:
                if not check_angular_distribution_positivity(fit_coeffs, _positivity_check_points):
                    fit_coeffs = project_to_positive_distribution(fit_coeffs, _positivity_check_points)
            endf_coeffs = endf_normalize_legendre_coeffs(fit_coeffs, include_a0=False)

        except Exception:
            empty_bins.append(bin_info.index)
            continue

        # Zero-pad to consistent length
        if len(endf_coeffs) < max_degree:
            padded = np.zeros(max_degree, dtype=float)
            padded[:len(endf_coeffs)] = endf_coeffs
            endf_coeffs = padded

        fitted_bins[bin_info.index] = endf_coeffs
        sample_coeffs[bin_info.index] = endf_coeffs

    # --- Pass 2: interpolate empty/failed bins from same-sample neighbors ---
    local_interpolated_bins = 0
    local_nominal_fallback_bins = 0

    for empty_idx in empty_bins:
        interp = _interpolate_from_neighbors(
            fitted_bins, energy_bins, empty_idx, max_degree
        )
        if interp is not None:
            sample_coeffs[empty_idx] = interp
            local_interpolated_bins += 1
        else:
            # Last resort: use nominal coefficients (no fitted neighbors at all)
            local_nominal_fallback_bins += 1
            if empty_idx in nominal_by_idx:
                nr = nominal_by_idx[empty_idx]
                endf_coeffs = endf_normalize_legendre_coeffs(nr.nominal_coeffs, include_a0=False)
                if len(endf_coeffs) < max_degree:
                    padded = np.zeros(max_degree, dtype=float)
                    padded[:len(endf_coeffs)] = endf_coeffs
                    endf_coeffs = padded
            else:
                endf_coeffs = np.zeros(max_degree)
            sample_coeffs[empty_idx] = endf_coeffs

    local_jump_stats = (
        local_jumped_bins,
        local_total_assignments,
        dict(local_jump_counts),
        local_datasets_outside_range,
        local_interpolated_bins,
        local_nominal_fallback_bins,
    )

    return (s_idx, sample_coeffs, local_jump_stats)


def run_mc_with_energy_jitter(
    nominal_results: List,  # List[NominalFitResult]
    energy_bins: List[EnergyBinInfo],
    dataset_info_by_bin: Dict[int, List[DatasetEnergyInfo]],
    exfor_cache: Dict[float, List[Tuple[pd.DataFrame, Dict]]],
    sorted_energies: List[float],
    n_samples: int,
    base_seed: int,
    max_degree: int,
    ridge_lambda: float,
    m_proj_u: float,
    m_targ_u: float,
    use_band_discrepancy: bool = True,
    min_points_per_band: int = 3,
    max_tau_fraction: float = 0.25,
    jitter_n_sigma_clip: float = 3.0,
    track_bin_jumps: bool = True,
    min_relative_uncertainty: float = 0.0,
    freeze_c0: bool = False,
    sigma_norm: float = 0.0,
    norm_dist: str = "lognormal",
    normalize_by_n_points: bool = False,
    max_experiment_weight_fraction: float = 1.0,
    n_procs: int = 1,
    logger=None,
    apply_positivity_projection: bool = False,
    positivity_check_points: int = 50,
    stochastic: bool = True,
) -> Tuple[Dict[int, Dict[int, np.ndarray]], BinJumpDiagnostics]:
    """
    Run MC sampling with energy jitter for cross-bin correlation.

    This implements Improvement 1.4: Instead of independent per-bin sampling,
    this method samples E* ~ N(E_nom, σE) for each dataset and assigns it to
    the containing bin, creating cross-bin coupling in the covariance matrix.

    Key insight:
    - Nominal fit: Uses energy_bin method unchanged (preserves resonance structure)
    - MC sampling: Introduces energy jitter per dataset
    - Effect: Creates cross-bin correlations useful for MF34

    Algorithm:
    1. For each MC sample:
       a. For each dataset: sample E* ~ N(E_nom, σE), clip to ±n_sigma
       b. Assign dataset to bin containing E*
       c. For each bin: fit using jitter-assigned datasets

    Parameters
    ----------
    nominal_results : List[NominalFitResult]
        Nominal fit results from Phase 1 (provides frozen degrees and tau values)
    energy_bins : List[EnergyBinInfo]
        Energy bin information
    dataset_info_by_bin : Dict[int, List[DatasetEnergyInfo]]
        Pre-computed dataset information from precompute_dataset_energy_info()
    exfor_cache : Dict[float, List[Tuple[pd.DataFrame, Dict]]]
        Pre-loaded EXFOR data cache
    sorted_energies : List[float]
        Sorted list of available EXFOR energies
    n_samples : int
        Number of MC samples to generate
    base_seed : int
        Base random seed for reproducibility
    max_degree : int
        Maximum Legendre degree
    ridge_lambda : float
        Ridge regularization parameter
    m_proj_u : float
        Projectile mass in atomic mass units
    m_targ_u : float
        Target mass in atomic mass units
    use_band_discrepancy : bool
        Use angular-band discrepancy model
    min_points_per_band : int
        Minimum points per angular band
    max_tau_fraction : float
        Maximum tau as fraction of cross section
    jitter_n_sigma_clip : float
        Clip energy jitter at ±n_sigma (default: 3.0)
    track_bin_jumps : bool
        Track bin jump statistics for diagnostics
    min_relative_uncertainty : float
        Minimum relative uncertainty floor
    freeze_c0 : bool
        Fix c0 coefficient during fitting
    sigma_norm : float
        Per-experiment normalization uncertainty
    norm_dist : str
        Normalization distribution ("lognormal" or "normal")
    normalize_by_n_points : bool
        If True, weight each point by 1/n_points for its experiment (equal weight per
        experiment regardless of point count). Matches nominal fit weighting.
    max_experiment_weight_fraction : float
        Cap total weight fraction any single experiment can have (e.g., 0.5 = max 50%).
        Set to 1.0 to disable capping. Matches nominal fit weighting.
    n_procs : int
        Number of parallel processes (1 = sequential)
    logger : optional
        Logger instance

    Returns
    -------
    Tuple[Dict[int, Dict[int, np.ndarray]], BinJumpDiagnostics]
        - all_samples: {sample_idx: {energy_index: endf_coeffs}}
        - diagnostics: BinJumpDiagnostics with statistics
    """
    # Get all datasets as flat list
    all_datasets = get_all_datasets_flat(dataset_info_by_bin)

    if logger:
        logger.info(f"  Energy jitter MC: {n_samples} samples, {len(all_datasets)} datasets")
        logger.info(f"  Jitter clipping: ±{jitter_n_sigma_clip}σ")
        if n_procs > 1:
            logger.info(f"  Using {n_procs} parallel processes")

    # Map nominal results by energy index for quick lookup
    nominal_by_idx = {nr.energy_index: nr for nr in nominal_results if nr.has_data}

    # Build dataset -> EXFOR data lookup
    # Key: (entry, subentry, energy_mev) -> (df, meta)
    dataset_exfor_lookup = {}
    for energy_mev, entries in exfor_cache.items():
        for df, meta in entries:
            entry = meta.get('entry', 'unknown')
            subentry = meta.get('subentry', 'unknown')
            key = (entry, subentry, energy_mev)
            dataset_exfor_lookup[key] = (df, meta)

    # Build shared args tuple (read-only data shared across all workers)
    shared_args = (
        all_datasets,
        energy_bins,
        nominal_by_idx,
        dataset_exfor_lookup,
        base_seed,
        max_degree,
        ridge_lambda,
        m_proj_u,
        m_targ_u,
        use_band_discrepancy,
        min_points_per_band,
        max_tau_fraction,
        jitter_n_sigma_clip,
        track_bin_jumps,
        min_relative_uncertainty,
        freeze_c0,
        sigma_norm,
        norm_dist,
        normalize_by_n_points,
        max_experiment_weight_fraction,
        apply_positivity_projection,
        positivity_check_points,
        stochastic,
    )

    args_list = [(s_idx,) + shared_args for s_idx in range(n_samples)]

    # Run samples (parallel or sequential)
    if n_procs > 1:
        with Pool(n_procs) as pool:
            results = pool.map(_run_one_jitter_sample, args_list)
    else:
        results = []
        for i, args in enumerate(args_list):
            results.append(_run_one_jitter_sample(args))
            if logger and (i + 1) % 10 == 0:
                logger.info(f"    Sample {i + 1}/{n_samples} completed")

    # Collect results and merge diagnostics
    all_samples: Dict[int, Dict[int, np.ndarray]] = {}
    total_assignments = 0
    jumped_bins = 0
    jump_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    datasets_outside_range = 0
    interpolated_bins = 0
    nominal_fallback_bins = 0

    for s_idx, sample_coeffs, local_jump_stats in results:
        all_samples[s_idx] = sample_coeffs

        (local_jumped, local_total, local_jumps, local_outside,
         local_interp, local_nominal_fb) = local_jump_stats
        jumped_bins += local_jumped
        total_assignments += local_total
        datasets_outside_range += local_outside
        interpolated_bins += local_interp
        nominal_fallback_bins += local_nominal_fb
        for key, count in local_jumps.items():
            jump_counts[key] += count

    if logger and n_procs > 1:
        logger.info(f"    All {n_samples} samples completed")

    # Compute bin jump diagnostics
    jump_rate = jumped_bins / total_assignments if total_assignments > 0 else 0.0

    diagnostics = BinJumpDiagnostics(
        total_assignments=total_assignments,
        jumped_bins=jumped_bins,
        jump_rate=jump_rate,
        jump_counts=dict(jump_counts),
        datasets_outside_range=datasets_outside_range,
        interpolated_bins=interpolated_bins,
        nominal_fallback_bins=nominal_fallback_bins,
    )

    return all_samples, diagnostics


def log_bin_jump_diagnostics(
    diagnostics: BinJumpDiagnostics,
    energy_bins: List[EnergyBinInfo],
    logger=None,
) -> None:
    """
    Log bin jump diagnostics in a readable format.

    Parameters
    ----------
    diagnostics : BinJumpDiagnostics
        Bin jump statistics from run_mc_with_energy_jitter
    energy_bins : List[EnergyBinInfo]
        Energy bin information (for energy labels)
    logger : optional
        Logger instance
    """
    if logger is None:
        return

    # Create bin index to energy mapping
    idx_to_energy = {bin_info.index: bin_info.energy_mev for bin_info in energy_bins}

    logger.info("")
    logger.info("[BIN JUMP DIAGNOSTICS]")
    logger.info(f"  Total assignments: {diagnostics.total_assignments}")
    logger.info(f"  Jumped bins: {diagnostics.jumped_bins}")
    logger.info(f"  Jump rate: {diagnostics.jump_rate * 100:.2f}%")

    if diagnostics.datasets_outside_range > 0:
        logger.info(f"  Datasets outside range: {diagnostics.datasets_outside_range}")

    if diagnostics.interpolated_bins > 0 or diagnostics.nominal_fallback_bins > 0:
        logger.info(f"  Empty bins filled by neighbor interpolation: {diagnostics.interpolated_bins}")
        logger.info(f"  Empty bins fell back to nominal (no neighbors): {diagnostics.nominal_fallback_bins}")

    # Interpretation
    if diagnostics.jump_rate > 0.30:
        logger.info("  Interpretation: Grid finer than σE → strong energy correlations")
    elif diagnostics.jump_rate > 0.10:
        logger.info("  Interpretation: Grid comparable to σE → moderate correlations")
    else:
        logger.info("  Interpretation: Grid coarser than σE → weak correlations")

    # Top jumps
    top_jumps = diagnostics.top_jumps(5)
    if top_jumps:
        logger.info("  Top bin jumps:")
        for (from_idx, to_idx), count in top_jumps:
            from_e = idx_to_energy.get(from_idx, 0.0)
            to_e = idx_to_energy.get(to_idx, 0.0)
            logger.info(f"    {from_idx}→{to_idx} ({from_e:.4f}→{to_e:.4f} MeV): {count}")

    logger.info("")


# =============================================================================
# INTERPOLATION
# =============================================================================

def interpolate_missing_bins(
    results: List[SamplingResult],
    n_samples: int,
) -> List[SamplingResult]:
    """
    Interpolate coefficients for bins where EXFOR data was not available.

    Uses linear interpolation in energy space between neighboring bins
    that have valid data.

    Parameters
    ----------
    results : List[SamplingResult]
        List of sampling results (some may have missing data)
    n_samples : int
        Number of samples expected

    Returns
    -------
    List[SamplingResult]
        Updated results with interpolated coefficients
    """
    logger = _get_logger()

    # Find indices with and without data
    valid_indices = []
    missing_indices = []

    for i, result in enumerate(results):
        if result.sampled_coeffs is not None:
            valid_indices.append(i)
        else:
            missing_indices.append(i)

    if not missing_indices:
        return results

    if len(valid_indices) < 2:
        if logger:
            logger.warning("Not enough valid bins for interpolation - keeping original coefficients for missing bins")
        return results

    # Get energies and coefficient arrays for valid bins
    valid_energies = np.array([results[i].bin_info.energy_mev for i in valid_indices])

    # Determine max coefficient length across all valid bins
    max_n_coeffs = max(results[i].sampled_coeffs.shape[1] for i in valid_indices)

    # Pad coefficient arrays to same size
    valid_coeffs = []
    for i in valid_indices:
        coeffs = results[i].sampled_coeffs
        if coeffs.shape[1] < max_n_coeffs:
            padded = np.zeros((n_samples, max_n_coeffs), dtype=float)
            padded[:, :coeffs.shape[1]] = coeffs
            valid_coeffs.append(padded)
        else:
            valid_coeffs.append(coeffs)
    valid_coeffs = np.array(valid_coeffs)  # Shape: (n_valid, n_samples, n_coeffs)

    # Interpolate for each missing bin
    for miss_idx in missing_indices:
        miss_energy = results[miss_idx].bin_info.energy_mev

        # Check if energy is within interpolation range
        if miss_energy < valid_energies.min() or miss_energy > valid_energies.max():
            if logger:
                logger.warning(
                    f"E={miss_energy:.4f} MeV is outside valid range "
                    f"[{valid_energies.min():.4f}, {valid_energies.max():.4f}] - keeping original"
                )
            continue

        # Interpolate each coefficient for each sample
        interp_coeffs = np.zeros((n_samples, max_n_coeffs), dtype=float)

        for sample_idx in range(n_samples):
            for coeff_idx in range(max_n_coeffs):
                # Get values at valid energies for this sample and coefficient
                y_vals = valid_coeffs[:, sample_idx, coeff_idx]
                # Linear interpolation
                interp_coeffs[sample_idx, coeff_idx] = np.interp(
                    miss_energy, valid_energies, y_vals
                )

        results[miss_idx].sampled_coeffs = interp_coeffs
        results[miss_idx].bin_info.interpolated = True

        if logger:
            logger.info(f"E={miss_energy:.4f} MeV: Interpolated coefficients from neighboring bins")

    return results


# =============================================================================
# COVARIANCE COMPUTATION
# =============================================================================

def compute_covariance_from_samples(
    all_samples: Dict[int, Dict[int, np.ndarray]],
    energy_indices: List[int],
    max_order: int,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Compute relative (fractional) covariance and correlation matrices from MC samples.

    The parameter vector is organized as:
        [a_1(E_1), a_2(E_1), ..., a_L(E_1), a_1(E_2), ..., a_L(E_N)]

    Parameters
    ----------
    all_samples : Dict[int, Dict[int, np.ndarray]]
        {sample_idx: {energy_index: coeffs}}
    energy_indices : List[int]
        List of energy indices (sorted)
    max_order : int
        Maximum Legendre order to include

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]], np.ndarray]
        - cov_matrix: Full relative (fractional) covariance matrix
        - corr_matrix: Full correlation matrix
        - param_labels: List of (energy_index, order) tuples
        - mean_params: MC mean parameter vector used as denominator for
          the relative conversion (same layout as param_labels)

    Covariance Conversion
    ---------------------
    ``np.cov()`` computes absolute covariance: Cov(a_i, a_j).
    ENDF MF34 with LB=5 expects relative (fractional) covariance:
        Cov_rel(i, j) = Cov_abs(i, j) / (mean_i * mean_j)

    The conversion is performed here so that the returned matrix can be
    written directly to MF34 with LB=5 format. Where |mean_i * mean_j| < 1e-30
    (effectively zero coefficients), the relative covariance is set to zero.
    """
    n_samples = len(all_samples)
    n_energies = len(energy_indices)
    n_params = n_energies * max_order

    # Build sample matrix
    sample_matrix = np.zeros((n_samples, n_params))

    for s_idx in range(n_samples):
        sample_data = all_samples[s_idx]
        row = []
        for e_idx in energy_indices:
            coeffs = sample_data.get(e_idx, np.zeros(max_order))
            # Pad or truncate to max_order
            padded = np.zeros(max_order)
            padded[:min(len(coeffs), max_order)] = coeffs[:max_order]
            row.extend(padded)
        sample_matrix[s_idx] = row

    # Compute absolute covariance
    cov_abs = np.cov(sample_matrix, rowvar=False)

    # Zero out rows/columns for parameters that were not actually fitted
    if valid_mask is not None:
        invalid = ~valid_mask
        cov_abs[invalid, :] = 0.0
        cov_abs[:, invalid] = 0.0

    # Convert absolute covariance to relative (fractional) covariance
    # Cov_rel(i,j) = Cov_abs(i,j) / (mean_i * mean_j)
    mean_params = np.mean(sample_matrix, axis=0)
    denom = np.outer(mean_params, mean_params)
    safe_mask = np.abs(denom) > 1e-30
    cov_matrix = np.zeros_like(cov_abs)
    cov_matrix[safe_mask] = cov_abs[safe_mask] / denom[safe_mask]

    # Flush near-zero values to exactly zero (numerical noise from np.cov)
    cov_matrix[np.abs(cov_matrix) < 1e-15] = 0.0

    # Compute correlation (from absolute covariance — identical to
    # computing from relative, since the mean factors cancel)
    std = np.sqrt(np.maximum(np.diag(cov_abs), 0.0))
    std[std == 0] = 1.0  # Avoid division by zero
    corr_matrix = cov_abs / np.outer(std, std)
    corr_matrix[np.abs(corr_matrix) < 1e-15] = 0.0

    # Generate labels
    param_labels = [(e_idx, l + 1) for e_idx in energy_indices for l in range(max_order)]

    return cov_matrix, corr_matrix, param_labels, mean_params


def combine_jitter_stochastic_covariance(
    cov_jitter: np.ndarray,
    cov_stochastic: np.ndarray,
    energy_bins: Optional[List] = None,
    energy_indices: Optional[List[int]] = None,
    max_order: Optional[int] = None,
    logger=None,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Combine jitter and stochastic relative covariance matrices.

    Uses the jitter pass's *correlation structure* (which captures cross-bin
    coupling from shared datasets/normalization) with the *total variance*
    from both passes combined:

        std_total = sqrt(diag(cov_jitter + cov_stochastic))
        corr = corr_jitter
        Cov_final = corr * outer(std_total, std_total)

    This preserves the strong cross-energy correlations from Pass 1 (jitter)
    while using the full variance magnitude from both passes.

    For parameters with zero jitter variance (undefined correlation), a
    Gaussian decay fallback is used for same-Legendre-order entries:
        rho = exp(-|E_i - E_j|^2 / (2 * L^2))
    where L = 3.0 * median(sigma_E).

    If ``energy_bins`` is None, falls back to simple addition (backward compat).

    Parameters
    ----------
    cov_jitter : np.ndarray
        Relative covariance matrix from jitter-only MC pass (Pass 1).
    cov_stochastic : np.ndarray
        Relative covariance matrix from stochastic MC pass (Pass 2).
    energy_bins : list, optional
        List of EnergyBinInfo objects (for Gaussian fallback energies).
    energy_indices : list of int, optional
        Energy indices corresponding to rows of the parameter layout.
    max_order : int, optional
        Number of Legendre orders per energy bin.
    logger : optional
        Logger for diagnostics.

    Returns
    -------
    np.ndarray
        Combined relative covariance matrix.
    """
    # Backward compatibility: simple addition when no energy info provided
    if energy_bins is None or energy_indices is None or max_order is None:
        return cov_jitter + cov_stochastic

    n = cov_jitter.shape[0]

    # Total variance from both independent passes
    var_total = np.diag(cov_jitter) + np.diag(cov_stochastic)
    std_total = np.sqrt(np.maximum(var_total, 0.0))

    # Zero out invalid (unfitted) parameters
    if valid_mask is not None:
        var_total[~valid_mask] = 0.0
        std_total[~valid_mask] = 0.0

    # Jitter correlation matrix (safe division for zero-variance params)
    std_jitter = np.sqrt(np.maximum(np.diag(cov_jitter), 0.0))
    std_jitter_safe = std_jitter.copy()
    zero_jitter_mask = std_jitter_safe < 1e-30
    std_jitter_safe[zero_jitter_mask] = 1.0
    corr_jitter = cov_jitter / np.outer(std_jitter_safe, std_jitter_safe)

    # Identify parameters with zero jitter variance -> need Gaussian fallback
    n_zero = int(np.sum(zero_jitter_mask))
    if n_zero > 0 and logger:
        logger.info(f"  Correlation fix: {n_zero}/{n} parameters have zero jitter variance -> Gaussian fallback")

    # Gaussian fallback for zero-jitter-variance parameters
    if n_zero > 0:
        # Compute length scale L = 3 * median(sigma_E)
        sigma_E_values = []
        for e_idx in energy_indices:
            if e_idx < len(energy_bins):
                s = energy_bins[e_idx].sigma_E_mev
                if s > 0:
                    sigma_E_values.append(s)
        if sigma_E_values:
            L_scale = 3.0 * np.median(sigma_E_values)
        else:
            # Fallback: 1% of energy range
            energies = [energy_bins[e_idx].energy_mev for e_idx in energy_indices if e_idx < len(energy_bins)]
            L_scale = 0.03 * (max(energies) - min(energies)) if len(energies) > 1 else 0.1

        # Get energy for each parameter
        param_energies = np.zeros(n)
        param_orders = np.zeros(n, dtype=int)
        for p in range(n):
            e_pos = p // max_order
            order = p % max_order + 1
            param_orders[p] = order
            if e_pos < len(energy_indices) and energy_indices[e_pos] < len(energy_bins):
                param_energies[p] = energy_bins[energy_indices[e_pos]].energy_mev

        # Fill fallback correlations for rows/columns with zero jitter variance
        for i in range(n):
            if not zero_jitter_mask[i]:
                continue
            for j in range(n):
                if i == j:
                    corr_jitter[i, j] = 1.0
                    continue
                # Skip invalid (unfitted) parameters
                if valid_mask is not None and (not valid_mask[i] or not valid_mask[j]):
                    continue
                # Only apply Gaussian decay for same-order cross-energy entries
                if param_orders[i] == param_orders[j]:
                    dE = param_energies[i] - param_energies[j]
                    rho = np.exp(-dE**2 / (2.0 * L_scale**2))
                    # Use fallback if EITHER param has zero jitter variance
                    corr_jitter[i, j] = rho
                    corr_jitter[j, i] = rho
                # Cross-order entries: keep whatever corr_jitter has (typically ~0)

    # Ensure diagonal is exactly 1
    np.fill_diagonal(corr_jitter, 1.0)
    # Clip to [-1, 1]
    corr_jitter = np.clip(corr_jitter, -1.0, 1.0)

    # Build combined covariance: corr_jitter * outer(std_total, std_total)
    cov_combined = corr_jitter * np.outer(std_total, std_total)
    cov_combined[np.abs(cov_combined) < 1e-15] = 0.0

    # Log correlation diagnostics
    if logger:
        # Off-diagonal correlation stats
        mask_offdiag = ~np.eye(n, dtype=bool)
        offdiag = np.abs(corr_jitter[mask_offdiag])
        if len(offdiag) > 0:
            logger.info(f"  Correlation fix applied: using jitter structure with total variances")
            logger.info(f"    Off-diagonal |corr|: mean={np.mean(offdiag):.4f}, median={np.median(offdiag):.4f}")

        # Same-order l=1 adjacent correlations
        if max_order >= 1 and len(energy_indices) >= 2:
            adj_corrs = []
            for k in range(len(energy_indices) - 1):
                i_param = k * max_order  # l=1 for energy k
                j_param = (k + 1) * max_order  # l=1 for energy k+1
                if i_param < n and j_param < n:
                    adj_corrs.append(corr_jitter[i_param, j_param])
            if adj_corrs:
                adj_corrs = np.array(adj_corrs)
                logger.info(f"    l=1 adjacent corr: mean={np.mean(adj_corrs):.4f}, "
                            f"median={np.median(adj_corrs):.4f}, "
                            f"min={np.min(adj_corrs):.4f}, max={np.max(adj_corrs):.4f}")

    return cov_combined


def extract_ll_prime_correlations(
    cov_matrix: np.ndarray,
    energy_indices: List[int],
    max_order: int,
    logger=None,
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[int, np.ndarray]:
    """
    Extract per-energy L×L correlation blocks from the full covariance matrix.

    The full covariance is laid out as (energy_0/l_1, energy_0/l_2, ...,
    energy_1/l_1, ...) with ``max_order`` Legendre orders per energy.
    This function pulls out the L×L correlation sub-block for each energy
    and logs summary statistics.

    Parameters
    ----------
    cov_matrix : np.ndarray
        Full covariance matrix of shape (n_energies * max_order, ...).
    energy_indices : List[int]
        Energy bin indices present in the matrix.
    max_order : int
        Number of Legendre orders per energy bin.
    logger : optional
        Logger instance for diagnostic output.

    Returns
    -------
    Dict[int, np.ndarray]
        Mapping energy_index → L×L correlation matrix.
    """
    n_e = len(energy_indices)
    ll_corr = {}

    off_diag_abs_all = []
    for ie, e_idx in enumerate(energy_indices):
        start = ie * max_order
        end = start + max_order
        block = cov_matrix[start:end, start:end]

        # Convert to correlation
        std = np.sqrt(np.maximum(np.diag(block), 0.0))
        std[std == 0] = 1.0
        corr = block / np.outer(std, std)
        np.fill_diagonal(corr, 1.0)

        # Mark entries involving absent (unfitted) orders as NaN
        if valid_mask is not None:
            local_valid = valid_mask[start:end]
            for li in range(max_order):
                for lj in range(max_order):
                    if not local_valid[li] or not local_valid[lj]:
                        corr[li, lj] = np.nan

        ll_corr[e_idx] = corr

        # Off-diagonal stats
        if max_order > 1:
            mask = ~np.eye(max_order, dtype=bool)
            off_vals = np.abs(corr[mask])
            off_vals = off_vals[~np.isnan(off_vals)]
            off_diag_abs_all.extend(off_vals.tolist())

    if logger is not None and max_order > 1 and off_diag_abs_all:
        arr = np.array(off_diag_abs_all)
        logger.info(f"  l-l' correlation summary ({n_e} energies, L_max={max_order}):")
        logger.info(f"    |off-diag| — mean={np.mean(arr):.4f}, "
                     f"median={np.median(arr):.4f}, "
                     f"max={np.max(arr):.4f}, min={np.min(arr):.4f}")
        # Per-energy extremes
        strongest_e = max(ll_corr, key=lambda e: np.max(np.abs(
            ll_corr[e][~np.eye(max_order, dtype=bool)])) if max_order > 1 else 0)
        strongest_val = np.max(np.abs(
            ll_corr[strongest_e][~np.eye(max_order, dtype=bool)]))
        logger.info(f"    Strongest l-l' coupling at energy idx {strongest_e}: "
                     f"|corr|={strongest_val:.4f}")

    return ll_corr


def build_gaussian_correlation_covariance(
    cov_stochastic: np.ndarray,
    energy_bins: List,
    energy_indices: List[int],
    max_order: int,
    logger=None,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build a full relative covariance matrix with Gaussian-decay energy correlations.

    Uses per-bin variances from the stochastic pass (Pass 2) and replaces the
    weak cross-energy correlations with a parametric Gaussian decay model based
    on the TOF energy resolution of each bin.

    The correlation model is:
    - Same energy, different order: keep stochastic cross-order correlation
    - Different energy, same order: Gaussian decay exp(-dE^2 / (2*sigma_eff^2))
    - Different energy, different order: Gaussian decay * cross-order factor

    Parameters
    ----------
    cov_stochastic : np.ndarray
        Relative covariance matrix from per-bin stochastic MC (Pass 2).
    energy_bins : list
        List of EnergyBinInfo objects with energy_mev and sigma_E_mev attributes.
    energy_indices : list of int
        Energy indices corresponding to rows of the parameter layout.
    max_order : int
        Number of Legendre orders per energy bin.
    logger : optional
        Logger for diagnostics.

    Returns
    -------
    np.ndarray
        Full relative covariance matrix with Gaussian energy correlations.
    """
    n = cov_stochastic.shape[0]
    n_energies = len(energy_indices)

    if logger:
        logger.info(f"  Building Gaussian decay correlation matrix: "
                    f"{n_energies} energies x {max_order} orders = {n} parameters")
        logger.info(f"  Assumptions: Gaussian energy decay (symmetric), "
                    f"separable cross-order x cross-energy, "
                    f"multivariate normal samples")

    # 1. Extract per-parameter variance and std from stochastic pass
    var_total = np.maximum(np.diag(cov_stochastic), 0.0)
    std_total = np.sqrt(var_total)

    # Zero out invalid (unfitted) parameters
    if valid_mask is not None:
        std_total[~valid_mask] = 0.0

    n_zero_var = int(np.sum(var_total < 1e-30))
    if logger and n_zero_var > 0:
        logger.info(f"  {n_zero_var}/{n} parameters have near-zero stochastic variance")

    # 2. Extract stochastic correlation matrix
    std_safe = std_total.copy()
    std_safe[std_safe < 1e-30] = 1.0
    corr_stochastic = cov_stochastic / np.outer(std_safe, std_safe)
    np.fill_diagonal(corr_stochastic, 1.0)
    corr_stochastic = np.clip(corr_stochastic, -1.0, 1.0)

    # 3. Build energy and sigma_E arrays for each parameter
    param_energies = np.zeros(n)
    param_sigma_E = np.zeros(n)
    param_orders = np.zeros(n, dtype=int)
    param_e_pos = np.zeros(n, dtype=int)  # position in energy_indices list

    for p in range(n):
        e_pos = p // max_order
        order = p % max_order
        param_orders[p] = order
        param_e_pos[p] = e_pos
        if e_pos < n_energies and energy_indices[e_pos] < len(energy_bins):
            ebin = energy_bins[energy_indices[e_pos]]
            param_energies[p] = ebin.energy_mev
            param_sigma_E[p] = ebin.sigma_E_mev

    # Fallback sigma_E for bins with zero resolution
    n_zero_sigma = int(np.sum(param_sigma_E <= 0))
    if np.any(param_sigma_E > 0):
        median_sigma = np.median(param_sigma_E[param_sigma_E > 0])
    else:
        median_sigma = 0.01
    param_sigma_E[param_sigma_E <= 0] = median_sigma

    if logger:
        sigma_vals = param_sigma_E[::max_order]  # one per energy
        logger.info(f"  sigma_E (MeV): min={np.min(sigma_vals):.4f}, "
                    f"median={np.median(sigma_vals):.4f}, max={np.max(sigma_vals):.4f}")
        if n_zero_sigma > 0:
            logger.info(f"  {n_zero_sigma}/{n} params had zero sigma_E, "
                        f"using median fallback={median_sigma:.4f} MeV")

    # 4. Build full correlation matrix (pointwise)
    corr = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            # Skip pairs involving invalid (unfitted) parameters
            if valid_mask is not None and (not valid_mask[i] or not valid_mask[j]):
                continue

            e_pos_i = param_e_pos[i]
            e_pos_j = param_e_pos[j]
            l_i = param_orders[i]
            l_j = param_orders[j]

            if e_pos_i == e_pos_j:
                # Same energy bin, different order -> keep stochastic cross-order
                corr[i, j] = corr_stochastic[i, j]
                corr[j, i] = corr_stochastic[j, i]
            else:
                # Different energy -> Gaussian decay
                dE = param_energies[i] - param_energies[j]
                sigma_eff = (param_sigma_E[i] + param_sigma_E[j]) / 2.0
                rho_E = np.exp(-dE**2 / (2.0 * sigma_eff**2))

                if l_i == l_j:
                    # Same order, different energy -> pure Gaussian
                    corr[i, j] = rho_E
                    corr[j, i] = rho_E
                else:
                    # Different order, different energy -> Gaussian * cross-order factor
                    # Use same-energy cross-order correlation at energy i as the factor
                    idx_same_e_li = e_pos_i * max_order + l_i
                    idx_same_e_lj = e_pos_i * max_order + l_j
                    corr_ll = corr_stochastic[idx_same_e_li, idx_same_e_lj]
                    corr[i, j] = rho_E * corr_ll
                    corr[j, i] = rho_E * corr_ll

    # 5. Build covariance from correlation and stds
    cov_full = corr * np.outer(std_total, std_total)

    # 6. Ensure PSD via eigenvalue clipping
    eigenvalues = np.linalg.eigvalsh(cov_full)
    min_eig = np.min(eigenvalues)
    max_eig = np.max(eigenvalues)
    if logger:
        logger.info(f"  Eigenvalue range: [{min_eig:.2e}, {max_eig:.2e}]")

    if min_eig < -1e-10:
        n_neg = int(np.sum(eigenvalues < -1e-10))
        if logger:
            logger.warning(f"  WARNING: Gaussian correlation matrix is NOT PSD "
                           f"(min eig={min_eig:.2e}, {n_neg} negative eigenvalues)")
            logger.info(f"  Projecting to nearest PSD via eigenvalue clipping...")
        eigvals, eigvecs = np.linalg.eigh(cov_full)
        eigvals_clipped = np.maximum(eigvals, 0.0)
        cov_full = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        cov_full = (cov_full + cov_full.T) / 2.0
        new_min = np.min(np.linalg.eigvalsh(cov_full))
        if logger:
            logger.info(f"  After PSD projection: min eig={new_min:.2e}")
    elif logger:
        logger.info(f"  Correlation matrix is PSD — no projection needed")

    # Log diagnostics
    if logger:
        # Adjacent l=1 correlations
        if max_order >= 1 and n_energies >= 2:
            adj_corrs = []
            adj_dE = []
            for k in range(n_energies - 1):
                i_param = k * max_order
                j_param = (k + 1) * max_order
                if i_param < n and j_param < n:
                    adj_corrs.append(corr[i_param, j_param])
                    adj_dE.append(abs(param_energies[i_param] - param_energies[j_param]))
            if adj_corrs:
                adj_corrs = np.array(adj_corrs)
                adj_dE = np.array(adj_dE)
                logger.info(f"  Gaussian correlation model results:")
                logger.info(f"    l=1 adjacent corr: mean={np.mean(adj_corrs):.4f}, "
                            f"median={np.median(adj_corrs):.4f}, "
                            f"min={np.min(adj_corrs):.4f}, max={np.max(adj_corrs):.4f}")
                logger.info(f"    Adjacent dE (MeV): mean={np.mean(adj_dE):.4f}, "
                            f"median={np.median(adj_dE):.4f}")

        # Overall off-diagonal stats
        mask_offdiag = ~np.eye(n, dtype=bool)
        offdiag = np.abs(corr[mask_offdiag])
        if len(offdiag) > 0:
            logger.info(f"    Off-diagonal |corr|: mean={np.mean(offdiag):.4f}, "
                        f"median={np.median(offdiag):.4f}, max={np.max(offdiag):.4f}")

    # Flush near-zero values to exactly zero (numerical noise)
    cov_full[np.abs(cov_full) < 1e-15] = 0.0

    return cov_full


def generate_cholesky_samples(
    cov_full: np.ndarray,
    mean_params: np.ndarray,
    energy_indices: List[int],
    max_order: int,
    n_samples: int,
    seed: int = 42,
    logger=None,
) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Generate correlated MC samples via Cholesky decomposition.

    Works in absolute space: converts relative covariance to absolute using
    the mean parameter vector, then generates samples as mean + L @ z.

    Parameters
    ----------
    cov_full : np.ndarray
        Full relative (fractional) covariance matrix.
    mean_params : np.ndarray
        MC mean parameter vector (ENDF-normalized coefficients).
    energy_indices : list of int
        Energy indices corresponding to parameter layout.
    max_order : int
        Number of Legendre orders per energy bin.
    n_samples : int
        Number of samples to generate.
    seed : int
        Random seed.
    logger : optional
        Logger for diagnostics.

    Returns
    -------
    Dict[int, Dict[int, np.ndarray]]
        {sample_idx: {energy_index: coeffs}} with ENDF-normalized coefficients.
    """
    n_params = len(energy_indices) * max_order

    if logger:
        logger.info(f"  Generating {n_samples} correlated samples via Cholesky "
                    f"({n_params} parameters, seed={seed})")
        logger.info(f"  NOTE: Samples are drawn from a multivariate normal distribution")
        n_near_zero = int(np.sum(np.abs(mean_params) < 1e-30))
        if n_near_zero > 0:
            logger.info(f"  {n_near_zero}/{n_params} mean params are near-zero "
                        f"(absolute cov will be ~0 for those)")

    # Convert relative covariance to absolute: Cov_abs = Cov_rel * outer(mean, mean)
    cov_abs = cov_full * np.outer(mean_params, mean_params)
    # Ensure symmetry
    cov_abs = (cov_abs + cov_abs.T) / 2.0

    # Cholesky decomposition (use eigendecomposition fallback if not PSD)
    cholesky_succeeded = False
    try:
        L = np.linalg.cholesky(cov_abs)
        cholesky_succeeded = True
        if logger:
            logger.info(f"  Cholesky decomposition successful ({n_params}x{n_params})")
    except np.linalg.LinAlgError as e:
        # Cholesky failure: the absolute covariance is not numerically PSD.
        # This can happen when mean params near zero create a near-singular
        # outer product, or from floating-point accumulation in the PSD
        # projection. The eigendecomposition fallback produces L such that
        # L @ L.T approximates cov_abs with all negative eigenvalues zeroed.
        eigvals, eigvecs = np.linalg.eigh(cov_abs)
        n_neg = int(np.sum(eigvals < 0))
        min_eig = np.min(eigvals)
        if logger:
            logger.warning(f"  WARNING: Cholesky decomposition FAILED ({e})")
            logger.warning(f"  Absolute covariance has {n_neg} negative eigenvalues "
                           f"(min={min_eig:.2e})")
            logger.info(f"  Using eigendecomposition fallback (zeroing negative eigenvalues)")
        eigvals = np.maximum(eigvals, 0.0)
        # L such that L @ L.T = cov_abs (approximately)
        L = eigvecs @ np.diag(np.sqrt(eigvals))
        if logger:
            # Report how much variance was lost
            total_var = np.trace(cov_abs)
            kept_var = np.sum(eigvals)
            if total_var > 0:
                logger.info(f"  Variance retained after clipping: "
                            f"{kept_var/total_var*100:.2f}%")

    # Generate samples
    rng = np.random.default_rng(seed)
    all_samples = {}

    for s_idx in range(n_samples):
        z = rng.standard_normal(n_params)
        sample_vec = mean_params + L @ z

        # Reshape into {energy_idx: coeffs} dict
        sample_dict = {}
        for k, e_idx in enumerate(energy_indices):
            start = k * max_order
            end = start + max_order
            sample_dict[e_idx] = sample_vec[start:end].copy()

        all_samples[s_idx] = sample_dict

    if logger:
        # Verify sample covariance roughly matches target
        sample_matrix = np.zeros((n_samples, n_params))
        for s_idx in range(n_samples):
            row = []
            for e_idx in energy_indices:
                row.extend(all_samples[s_idx][e_idx])
            sample_matrix[s_idx] = row
        sample_std = np.std(sample_matrix, axis=0)
        target_std = np.sqrt(np.maximum(np.diag(cov_abs), 0.0))
        mask = target_std > 1e-30
        if np.any(mask):
            ratio = sample_std[mask] / target_std[mask]
            logger.info(f"  Sample validation — std ratio (sample/target): "
                        f"mean={np.mean(ratio):.3f}, std={np.std(ratio):.3f}")
            if abs(np.mean(ratio) - 1.0) > 0.15:
                logger.warning(f"  WARNING: Sample std deviates >15% from target — "
                               f"consider increasing n_samples (currently {n_samples})")

        # Check adjacent-bin sample correlations
        if max_order >= 1 and len(energy_indices) >= 2:
            sample_corr = np.corrcoef(sample_matrix, rowvar=False)
            adj_sample_corrs = []
            for k in range(len(energy_indices) - 1):
                i_p = k * max_order
                j_p = (k + 1) * max_order
                if i_p < n_params and j_p < n_params:
                    adj_sample_corrs.append(sample_corr[i_p, j_p])
            if adj_sample_corrs:
                adj_arr = np.array(adj_sample_corrs)
                logger.info(f"  Sample l=1 adjacent corr: mean={np.mean(adj_arr):.4f}, "
                            f"median={np.median(adj_arr):.4f}")

    return all_samples


def cap_covariance_relative_uncertainty(
    cov_matrix: np.ndarray,
    max_relative_std: float,
    param_labels: List[Tuple[int, int]],
    energy_mev_lookup: Optional[Dict[int, float]] = None,
    logger=None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Cap diagonal variances of the relative covariance matrix, preserving correlations.

    Applies a congruence transformation cov_capped = diag(s) @ cov @ diag(s)
    where s[i] = min(max_std, std[i]) / std[i]. This preserves the correlation
    structure exactly and maintains positive semi-definiteness.

    Parameters
    ----------
    cov_matrix : np.ndarray
        Relative (fractional) covariance matrix of Legendre coefficients,
        as returned by ``compute_covariance_from_samples``.
    max_relative_std : float
        Maximum allowed standard deviation (e.g. 1.0 for 100% cap).
    param_labels : List[Tuple[int, int]]
        List of (energy_index, order) tuples identifying each parameter.
    energy_mev_lookup : Dict[int, float], optional
        Mapping energy_index -> energy in MeV for logging.
    logger : optional
        Logger instance for diagnostic output.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        - capped covariance matrix
        - diagnostics dict with keys: 'n_capped', 'n_total', 'max_original_std',
          'capped_entries' (list of dicts with entry details)
    """
    diag = np.diag(cov_matrix).copy()
    std = np.sqrt(np.maximum(diag, 0.0))
    n_total = len(std)
    max_std_sq = max_relative_std ** 2

    # Compute scale factors: s[i] = min(max_relative_std, std[i]) / std[i]
    s = np.ones(n_total)
    capped_entries = []

    for i in range(n_total):
        if std[i] > max_relative_std:
            s[i] = max_relative_std / std[i]
            e_idx, order = param_labels[i]
            entry_info = {
                'param_index': i,
                'energy_index': e_idx,
                'order': order,
                'original_std': float(std[i]),
                'capped_std': float(max_relative_std),
            }
            if energy_mev_lookup and e_idx in energy_mev_lookup:
                entry_info['energy_mev'] = energy_mev_lookup[e_idx]
            capped_entries.append(entry_info)

    n_capped = len(capped_entries)

    # Apply congruence transformation: cov_capped = diag(s) @ cov @ diag(s)
    cov_capped = cov_matrix * np.outer(s, s)

    # Build diagnostics
    max_original_std = float(np.max(std)) if n_total > 0 else 0.0
    diagnostics = {
        'n_capped': n_capped,
        'n_total': n_total,
        'max_original_std': max_original_std,
        'capped_entries': capped_entries,
    }

    # Logging
    if logger:
        logger.info(f"[Covariance Cap] Applying max relative std cap = {max_relative_std*100:.1f}%")
        if n_capped > 0:
            logger.info(f"[Covariance Cap] Capped {n_capped}/{n_total} diagonal entries ({100*n_capped/n_total:.1f}%)")
            for entry in capped_entries:
                e_idx = entry['energy_index']
                order = entry['order']
                orig = entry['original_std'] * 100
                capped = entry['capped_std'] * 100
                if 'energy_mev' in entry:
                    logger.info(
                        f"[Covariance Cap]   E_idx={e_idx}  ({entry['energy_mev']:.2f} MeV), "
                        f"L={order}: {orig:.1f}% -> {capped:.1f}%"
                    )
                else:
                    logger.info(
                        f"[Covariance Cap]   E_idx={e_idx}, L={order}: {orig:.1f}% -> {capped:.1f}%"
                    )
            # Find max original entry
            max_entry = max(capped_entries, key=lambda e: e['original_std'])
            logger.info(
                f"[Covariance Cap] Max original relative std: {max_entry['original_std']*100:.1f}% "
                f"(E_idx={max_entry['energy_index']}, L={max_entry['order']})"
            )
        else:
            logger.info(f"[Covariance Cap] No entries exceed cap — covariance unchanged")

    return cov_capped, diagnostics


def save_all_legendre_coefficients(
    nominal_results: List,  # List[NominalFitResult] - avoid circular import
    all_samples: Dict[int, Dict[int, np.ndarray]],
    output_dir: str,
    max_degree: int,
) -> str:
    """
    Save all Legendre coefficients (nominal + all MC samples) to Parquet.

    Parameters
    ----------
    nominal_results : List[NominalFitResult]
        Nominal fit results from Phase 1
    all_samples : Dict[int, Dict[int, np.ndarray]]
        {sample_idx: {energy_index: endf_coeffs}} for all MC samples
    output_dir : str
        Output directory
    max_degree : int
        Maximum Legendre order

    Returns
    -------
    str
        Path to saved Parquet file
    """
    output_path = Path(output_dir)

    # Collect all data
    data_rows = []

    # Add nominal coefficients (sample_idx = 0)
    for nr in nominal_results:
        if nr.has_data:
            endf_coeffs = endf_normalize_legendre_coeffs(nr.nominal_coeffs, include_a0=False)
            # Pad to max_degree if needed
            padded_coeffs = np.zeros(max_degree)
            padded_coeffs[:len(endf_coeffs)] = endf_coeffs

            row = {
                'sample_idx': 0,  # 0 = nominal
                'energy_index': nr.energy_index,
                'energy_mev': nr.energy_mev,
            }
            for l in range(max_degree):
                row[f'a_{l+1}'] = padded_coeffs[l]
            data_rows.append(row)

    # Add MC sample coefficients (sample_idx = 1 to N)
    n_samples = len(all_samples)
    for sample_idx in range(n_samples):
        sample_coeffs = all_samples[sample_idx]
        for energy_idx, endf_coeffs in sample_coeffs.items():
            # Find corresponding energy in MeV
            energy_mev = None
            for nr in nominal_results:
                if nr.energy_index == energy_idx:
                    energy_mev = nr.energy_mev
                    break

            # Pad to max_degree if needed
            padded_coeffs = np.zeros(max_degree)
            padded_coeffs[:len(endf_coeffs)] = endf_coeffs

            row = {
                'sample_idx': sample_idx + 1,  # 1-based for MC samples
                'energy_index': energy_idx,
                'energy_mev': energy_mev if energy_mev is not None else 0.0,
            }
            for l in range(max_degree):
                row[f'a_{l+1}'] = padded_coeffs[l]
            data_rows.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(data_rows)

    # Sort by sample_idx, then energy_index
    df = df.sort_values(['sample_idx', 'energy_index']).reset_index(drop=True)

    # Save as Parquet (compact binary columnar format)
    parquet_file = output_path / 'legendre_coefficients_all_samples.parquet'
    df.to_parquet(parquet_file, engine='pyarrow', index=False)

    return str(parquet_file)


# =============================================================================
# ENDF WRITING FUNCTIONS
# =============================================================================

def write_nominal_endf(
    original_endf_file: str,
    mt_number: int,
    nominal_results: List,  # List[NominalFitResult]
    output_dir: str,
) -> str:
    """
    Write ENDF file with nominal fit coefficients.

    Parameters
    ----------
    original_endf_file : str
        Path to original ENDF file
    mt_number : int
        MT reaction number
    nominal_results : List[NominalFitResult]
        Nominal fit results from Phase 1
    output_dir : str
        Output directory

    Returns
    -------
    str
        Path to output file
    """
    # Parse original ENDF
    endf = read_endf(original_endf_file)
    mf4 = endf.get_file(4)

    if mf4 is None:
        raise ValueError(f"MF4 not found in {original_endf_file}")

    mt_data = mf4.sections.get(mt_number)
    if mt_data is None:
        raise ValueError(f"MT{mt_number} not found in MF4")

    # Check type
    if not isinstance(mt_data, (MF4MTLegendre, MF4MTMixed)):
        raise ValueError(f"MT{mt_number} is not Legendre or Mixed type")

    # Apply nominal coefficients for energies with EXFOR data
    for nr in nominal_results:
        if nr.has_data and nr.energy_index < len(mt_data._legendre_coeffs):
            # Convert nominal coefficients to ENDF format
            endf_coeffs = endf_normalize_legendre_coeffs(nr.nominal_coeffs, include_a0=False)
            mt_data._legendre_coeffs[nr.energy_index] = list(endf_coeffs)

    # Create output structure (nominal file lives at output_dir level, not inside endf/)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base = Path(original_endf_file).stem
    output_file = output_path / f"{base}_nominal.endf"

    # Use ENDFWriter
    writer = ENDFWriter(original_endf_file)
    success = writer.replace_mf_section(mf4, str(output_file))

    if not success:
        raise RuntimeError(f"Failed to write {output_file}")

    return str(output_file)


def compute_mc_mean_coefficients(
    all_samples: Dict[int, Dict[int, np.ndarray]],
    nominal_results: List,  # List[NominalFitResult]
) -> Dict[int, np.ndarray]:
    """
    Compute MC mean coefficients from all samples.

    Parameters
    ----------
    all_samples : Dict[int, Dict[int, np.ndarray]]
        {sample_idx: {energy_index: endf_coeffs}}
    nominal_results : List[NominalFitResult]
        Nominal fit results (for energy indices)

    Returns
    -------
    Dict[int, np.ndarray]
        {energy_index: mc_mean_coeffs} in ENDF format (a_1, a_2, ...)
    """
    # Get all energy indices with data
    energy_indices = [nr.energy_index for nr in nominal_results if nr.has_data]

    mc_mean_coeffs = {}
    n_samples = len(all_samples)

    for e_idx in energy_indices:
        # Collect coefficients from all samples for this energy
        sample_coeffs_list = []
        for s_idx in range(n_samples):
            if e_idx in all_samples[s_idx]:
                sample_coeffs_list.append(all_samples[s_idx][e_idx])

        if sample_coeffs_list:
            # Stack and compute mean
            stacked = np.vstack(sample_coeffs_list)
            mc_mean_coeffs[e_idx] = np.mean(stacked, axis=0)

    return mc_mean_coeffs


def write_average_endf(
    original_endf_file: str,
    mt_number: int,
    nominal_results: List,  # List[NominalFitResult]
    all_samples: Dict[int, Dict[int, np.ndarray]],
    output_dir: str,
) -> str:
    """
    Write ENDF file with MC mean coefficients (average file).

    Parameters
    ----------
    original_endf_file : str
        Path to original ENDF file
    mt_number : int
        MT reaction number
    nominal_results : List[NominalFitResult]
        Nominal fit results from Phase 1
    all_samples : Dict[int, Dict[int, np.ndarray]]
        {sample_idx: {energy_index: endf_coeffs}} for all MC samples
    output_dir : str
        Output directory

    Returns
    -------
    str
        Path to output file
    """
    # Compute MC mean coefficients
    mc_mean_coeffs = compute_mc_mean_coefficients(all_samples, nominal_results)

    # Parse original ENDF
    endf = read_endf(original_endf_file)
    mf4 = endf.get_file(4)

    if mf4 is None:
        raise ValueError(f"MF4 not found in {original_endf_file}")

    mt_data = mf4.sections.get(mt_number)
    if mt_data is None:
        raise ValueError(f"MT{mt_number} not found in MF4")

    # Check type
    if not isinstance(mt_data, (MF4MTLegendre, MF4MTMixed)):
        raise ValueError(f"MT{mt_number} is not Legendre or Mixed type")

    # Apply MC mean coefficients for energies with data
    for e_idx, mean_coeffs in mc_mean_coeffs.items():
        if e_idx < len(mt_data._legendre_coeffs):
            mt_data._legendre_coeffs[e_idx] = list(mean_coeffs)

    # Create output structure (average file lives at output_dir level, not inside endf/)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base = Path(original_endf_file).stem
    output_file = output_path / f"{base}_average.endf"

    # Use ENDFWriter
    writer = ENDFWriter(original_endf_file)
    success = writer.replace_mf_section(mf4, str(output_file))

    if not success:
        raise RuntimeError(f"Failed to write {output_file}")

    return str(output_file)


def write_endf_sample(
    sample_index: int,
    original_endf_file: str,
    mt_number: int,
    energy_indices: List[int],
    sampled_coeffs_by_energy: Dict[int, np.ndarray],
    output_dir: str,
    cached_original_coeffs: Optional[List[List[float]]] = None,
) -> str:
    """
    Write a single ENDF sample file.

    Parameters
    ----------
    sample_index : int
        Sample index (0-based)
    original_endf_file : str
        Path to original ENDF file
    mt_number : int
        MT reaction number
    energy_indices : List[int]
        Indices of energies to modify (unused, kept for compatibility)
    sampled_coeffs_by_energy : Dict[int, np.ndarray]
        Coefficients for each energy index (for this sample)
    output_dir : str
        Output directory
    cached_original_coeffs : Optional[List[List[float]]]
        If provided, use these as the original coefficients

    Returns
    -------
    str
        Path to output file
    """
    # Parse original ENDF
    endf = read_endf(original_endf_file)
    mf4 = endf.get_file(4)

    if mf4 is None:
        raise ValueError(f"MF4 not found in {original_endf_file}")

    mt_data = mf4.sections.get(mt_number)
    if mt_data is None:
        raise ValueError(f"MT{mt_number} not found in MF4")

    # Check type
    if not isinstance(mt_data, (MF4MTLegendre, MF4MTMixed)):
        raise ValueError(f"MT{mt_number} is not Legendre or Mixed type")

    # Modify coefficients at each energy
    for energy_idx, new_coeffs in sampled_coeffs_by_energy.items():
        if energy_idx < len(mt_data._legendre_coeffs):
            mt_data._legendre_coeffs[energy_idx] = list(new_coeffs)

    # Write output file
    sample_str = f"{sample_index + 1:04d}"
    base = Path(original_endf_file).stem

    # Create output structure (endf_direct/ to avoid collision with Pipeline B's endf/)
    sample_dir = Path(output_dir) / "endf_direct" / sample_str
    sample_dir.mkdir(parents=True, exist_ok=True)

    output_file = sample_dir / f"{base}_{sample_str}.endf"

    # Use ENDFWriter
    writer = ENDFWriter(original_endf_file)
    success = writer.replace_mf_section(mf4, str(output_file))

    if not success:
        raise RuntimeError(f"Failed to write {output_file}")

    # Strip MF34 from sample files — samples should not carry covariance data
    remove_mf34_from_file(str(output_file))

    return str(output_file)


def _write_sample_wrapper(args):
    """Wrapper for parallel writing of ENDF samples."""
    return write_endf_sample(*args)


def write_endf_samples_batch(
    original_endf_file: str,
    mt_number: int,
    all_samples: Dict[int, Dict[int, np.ndarray]],
    output_dir: str,
    n_procs: int = 1,
) -> List[str]:
    """
    Write multiple ENDF samples efficiently.

    When n_procs=1, reads the ENDF file ONCE and reuses the parsed structure
    for all samples (sequential mode). When n_procs>1, each worker reads the
    ENDF template independently to avoid sharing mutable objects.

    Parameters
    ----------
    original_endf_file : str
        Path to original ENDF file
    mt_number : int
        MT reaction number
    all_samples : Dict[int, Dict[int, np.ndarray]]
        {sample_idx: {energy_index: coefficients}}
    output_dir : str
        Output directory
    n_procs : int
        Number of parallel processes (1 = sequential)

    Returns
    -------
    List[str]
        Paths to output files
    """
    n_total = len(all_samples)

    if n_procs > 1:
        # Parallel mode: each worker reads the ENDF template independently
        args_list = [
            (
                sample_idx,
                original_endf_file,
                mt_number,
                [],  # energy_indices (unused by write_endf_sample)
                all_samples[sample_idx],
                output_dir,
                None,  # cached_original_coeffs (worker reads fresh)
            )
            for sample_idx in sorted(all_samples.keys())
        ]

        print(f"[INFO] Writing {n_total} sample files using {n_procs} processes")
        with Pool(n_procs) as pool:
            output_files = pool.map(_write_sample_wrapper, args_list)

        return output_files

    # Sequential mode: parse ENDF once and reuse
    output_path = Path(output_dir)
    base = Path(original_endf_file).stem

    endf_template = read_endf(original_endf_file)
    mf4_template = endf_template.get_file(4)

    if mf4_template is None:
        raise ValueError(f"MF4 not found in {original_endf_file}")

    mt_data_template = mf4_template.sections.get(mt_number)
    if mt_data_template is None:
        raise ValueError(f"MT{mt_number} not found in MF4")

    if not isinstance(mt_data_template, (MF4MTLegendre, MF4MTMixed)):
        raise ValueError(f"MT{mt_number} is not Legendre or Mixed type")

    # Store original coefficients for restoration
    original_coeffs = [list(c) for c in mt_data_template._legendre_coeffs]

    # Create writer once
    writer = ENDFWriter(original_endf_file)

    output_files = []

    for sample_idx in sorted(all_samples.keys()):
        # Restore original coefficients
        mt_data_template._legendre_coeffs = [list(c) for c in original_coeffs]

        # Apply sampled coefficients
        sampled_coeffs = all_samples[sample_idx]
        for energy_idx, new_coeffs in sampled_coeffs.items():
            if energy_idx < len(mt_data_template._legendre_coeffs):
                mt_data_template._legendre_coeffs[energy_idx] = list(new_coeffs)

        # Write output file
        sample_str = f"{sample_idx + 1:04d}"
        sample_dir = output_path / "endf" / sample_str
        sample_dir.mkdir(parents=True, exist_ok=True)
        output_file = sample_dir / f"{base}_{sample_str}.endf"

        success = writer.replace_mf_section(mf4_template, str(output_file))
        if not success:
            raise RuntimeError(f"Failed to write {output_file}")

        # Strip MF34 from sample files — samples should not carry covariance data
        remove_mf34_from_file(str(output_file))

        output_files.append(str(output_file))

        if (sample_idx + 1) % 50 == 0 or sample_idx == 0 or sample_idx == n_total - 1:
            print(f"[INFO] Writing sample {sample_idx + 1}/{n_total}")

    return output_files


# =============================================================================
# MF34 COVARIANCE FUNCTIONS
# =============================================================================
# These functions are now provided by kika.endf.writers module.
# Import and re-export for backward compatibility with existing scripts.

from kika.endf.writers import (
    create_mf34_from_covariance,
    write_mf34_to_file as write_mf34_to_endf,  # Alias for backward compatibility
    remove_mf34_from_file,
)
