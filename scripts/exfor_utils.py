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

# Import TOF parameters module
from scripts.tof_parameters import get_tof_parameters, compute_sigma_E

# Import resample_AD functions (relative import for same directory)
from .resample_AD import (
    load_exfor_for_fitting,
    endf_normalize_legendre_coeffs,
    sample_legendre_coefficients,
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


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted median of values.

    Parameters
    ----------
    values : array
        Values to compute the median of.
    weights : array
        Non-negative weights (need not sum to 1).

    Returns
    -------
    float
        Weighted median value.
    """
    order = np.argsort(values)
    sorted_vals = values[order]
    sorted_w = weights[order]
    cum_w = np.cumsum(sorted_w)
    half = cum_w[-1] / 2.0
    idx = np.searchsorted(cum_w, half)
    return float(sorted_vals[min(idx, len(sorted_vals) - 1)])


@dataclass
class KWDiagnostics:
    """Diagnostics for kernel-weight overlap at one energy bin.

    Experiment-level diagnostics computed from precomputed overlap weights.
    Used for hybrid correlation blend alpha (KW vs Gaussian).
    """
    n_eff_kw: float                                 # Experiment-level effective sample size
    n_experiments_kw: int                            # Number of contributing experiments
    max_experiment_weight_frac_kw: float             # Largest single experiment weight fraction
    weighted_median_n_points: float                  # Overlap-weighted median points per experiment


def compute_kw_diagnostics(
    bin_overlap: List[Tuple[Dict, float]],
) -> Optional['KWDiagnostics']:
    """Compute KW-level diagnostics from overlap weights for one bin.

    Multiple datasets from the same experiment are aggregated: their overlap
    weights are summed and their n_points are summed to give per-experiment
    totals.  n_eff_kw and related metrics are then computed at the experiment
    level so they reflect true experimental diversity.

    Parameters
    ----------
    bin_overlap : list of (dataset_dict, weight) tuples
        From overlap_weights[bin_idx]. May contain multiple entries per
        experiment (different energies).

    Returns
    -------
    KWDiagnostics or None if no overlapping experiments.
    """
    if not bin_overlap:
        return None

    # Aggregate per experiment: sum overlap weights and n_points
    exp_agg: Dict[str, Tuple[float, int]] = {}  # exp_id -> (total_w, total_n_pts)
    for ds, w in bin_overlap:
        exp_id = ds['experiment_id']
        prev_w, prev_n = exp_agg.get(exp_id, (0.0, 0))
        exp_agg[exp_id] = (prev_w + w, prev_n + ds.get('n_points', 0))

    exp_weights = np.array([v[0] for v in exp_agg.values()])
    exp_n_points = np.array([v[1] for v in exp_agg.values()])
    n_experiments_kw = len(exp_weights)

    w_sum = exp_weights.sum()
    if w_sum <= 0:
        return None

    w_frac = exp_weights / w_sum
    n_eff_kw = 1.0 / np.sum(w_frac ** 2)
    max_experiment_weight_frac_kw = float(np.max(w_frac))

    # Overlap-weighted median of total n_points per experiment
    wm_n_pts = weighted_median(exp_n_points.astype(float), w_frac)

    return KWDiagnostics(
        n_eff_kw=n_eff_kw,
        n_experiments_kw=n_experiments_kw,
        max_experiment_weight_frac_kw=max_experiment_weight_frac_kw,
        weighted_median_n_points=wm_n_pts,
    )


def compute_kw_reliability_alpha(
    kw_diag: Optional['KWDiagnostics'],
    interpolated: bool = False,
    alpha_min_data: float = 0.0,
    alpha_max: float = 1.0,
    n_eff_mid: float = 3.5,
    n_eff_scale: float = 1.5,
    min_experiments: int = 2,
    dominance_threshold: float = 0.40,
    min_points_ref: float = 5.0,
) -> float:
    """Compute per-bin reliability alpha from KW overlap diagnostics.

    Same functional form as compute_bin_reliability_alpha but with
    parameters tuned for experiment-level overlap weights.

    Parameters
    ----------
    min_points_ref : float
        Reference point count for quality penalty. Bins where the
        weighted-median n_points is below this threshold get alpha
        scaled by (median / min_points_ref).

    Returns alpha in [0, 1] where alpha=1 means pure KW, alpha=0 means pure Gaussian.
    """
    if interpolated or kw_diag is None:
        return 0.0

    if not np.isfinite(kw_diag.n_eff_kw):
        return 0.0

    # Base alpha: sigmoid on experiment-level n_eff
    x = (kw_diag.n_eff_kw - n_eff_mid) / n_eff_scale
    alpha = alpha_min_data + (alpha_max - alpha_min_data) / (1.0 + np.exp(-x))

    # Experiment count penalty
    if kw_diag.n_experiments_kw < min_experiments:
        alpha *= 0.25 + 0.25 * kw_diag.n_experiments_kw

    # Dominance penalty
    frac = kw_diag.max_experiment_weight_frac_kw
    if frac > dominance_threshold:
        alpha *= 1.0 - 0.5 * (frac - dominance_threshold) / (1.0 - dominance_threshold)

    # Quality penalty: penalize bins where median experiment is too sparse
    if min_points_ref > 0 and kw_diag.weighted_median_n_points < min_points_ref:
        alpha *= kw_diag.weighted_median_n_points / min_points_ref

    return float(np.clip(alpha, alpha_min_data, alpha_max))


def compute_bin_reliability_alpha(
    diagnostics: Optional[KernelDiagnostics],
    interpolated: bool = False,
    alpha_min_data: float = 0.05,
    alpha_max: float = 0.75,
    n_eff_mid: float = 7.0,
    n_eff_scale: float = 2.0,
    min_experiments: int = 3,
    dominance_threshold: float = 0.25,
) -> float:
    """Compute per-bin reliability weight for KW vs Gaussian correlation blend.

    Returns alpha in [0, 1] where alpha=1 means pure KW and alpha=0 means pure Gaussian.
    Interpolated bins or bins without diagnostics always return 0.0.
    Bins with data return values in [alpha_min_data, alpha_max].
    """
    if interpolated or diagnostics is None:
        return 0.0

    # Non-finite n_eff → pure Gaussian fallback
    if not np.isfinite(diagnostics.n_eff):
        return 0.0

    # Base alpha: sigmoid on n_eff mapped to [alpha_min_data, alpha_max]
    x = (diagnostics.n_eff - n_eff_mid) / n_eff_scale
    alpha = alpha_min_data + (alpha_max - alpha_min_data) / (1.0 + np.exp(-x))

    # Experiment penalty: fewer than min_experiments reduces alpha
    if diagnostics.n_experiments < min_experiments:
        alpha *= 0.25 + 0.25 * diagnostics.n_experiments

    # Dominance penalty: one experiment dominates
    frac = diagnostics.max_experiment_weight_frac
    if frac > dominance_threshold:
        alpha *= 1.0 - 0.5 * (frac - dominance_threshold) / (1.0 - dominance_threshold)

    # Clamp to valid range
    return float(np.clip(alpha, alpha_min_data, alpha_max))


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
    cache format expected by filter_exfor_with_energy_bin. For each object,
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
    The returned DataFrame has columns compatible with filter_exfor_with_energy_bin:
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
# KERNEL-WEIGHT MC FOR CROSS-ENERGY CORRELATIONS
# =============================================================================

def precompute_overlap_weights(
    nominal_results: List,  # List[NominalFitResult]
    energy_bins: List[EnergyBinInfo],
    min_weight: float = 1e-3,
    tof_params_cache: Optional[Dict] = None,
    default_flight_path_m: float = 27.037,
    default_time_resolution_ns: float = 5.0,
) -> Dict[int, List[Tuple[Dict, float]]]:
    """Compute overlap weights from ALL datasets across all bins.

    For each bin, collects all datasets from all nominal results and computes
    their CDF-based overlap weight to that bin using compute_overlap_weight().

    When tof_params_cache is provided, per-experiment energy resolution is
    computed from TOF parameters (flight path + time resolution). Otherwise
    falls back to the bin's sigma_E (uniform resolution for all experiments).

    Parameters
    ----------
    nominal_results : list
        Nominal fit results (NominalFitResult objects).
    energy_bins : List[EnergyBinInfo]
        Energy bin definitions with boundaries and sigma_E.
    min_weight : float
        Minimum overlap weight to keep (default 1e-3).
    tof_params_cache : dict, optional
        Pre-loaded TOF parameters from load_tof_parameters_file().
    default_flight_path_m : float
        Default flight path in meters for experiments not in the cache.
    default_time_resolution_ns : float
        Default time resolution in nanoseconds for experiments not in the cache.

    Returns
    -------
    Dict[int, List[Tuple[Dict, float]]]
        bin_index -> [(dataset_dict, weight), ...] where dataset_dict has keys:
        'entry', 'subentry', 'exfor_energy_mev', 'exfor_df', 'n_points',
        'experiment_id', 'sigma_E_mev', 'tof_source'.
    """
    # Collect all unique datasets across all bins
    all_datasets = []
    seen = set()
    for nr in nominal_results:
        if not nr.has_data or nr.interpolated:
            continue
        for exp_info in nr.experiments_info:
            entry = exp_info.get('entry', 'unknown')
            subentry = exp_info.get('subentry', 'unknown')
            exfor_energy = exp_info.get('exfor_energy_mev', nr.energy_mev)
            key = (entry, subentry, f"{exfor_energy:.6f}")
            if key in seen:
                continue
            seen.add(key)

            # Extract EXFOR data for this dataset from the bin's DataFrame
            # The dataset's angular data is in nr.exfor_df filtered by experiment
            df = nr.exfor_df
            if df is None or df.empty:
                continue

            # Filter to this experiment's data
            mask = pd.Series([False] * len(df), index=df.index)
            if 'entry' in df.columns and 'subentry' in df.columns:
                mask = (df['entry'].astype(str) == str(entry)) & (df['subentry'].astype(str) == str(subentry))
            elif 'experiment_id' in df.columns:
                exp_id = f"{entry}.{subentry}"
                mask = df['experiment_id'] == exp_id

            dataset_df = df[mask]
            if dataset_df.empty:
                # Try matching by energy proximity
                if 'exfor_energy_mev' in df.columns:
                    mask = np.abs(df['exfor_energy_mev'] - exfor_energy) < 1e-6
                    dataset_df = df[mask]
                if dataset_df.empty:
                    continue

            # Compute per-experiment sigma_E from TOF parameters
            subentry_id = f"{entry}{subentry}"  # format: "10571002"
            if tof_params_cache is not None:
                tof_params = get_tof_parameters(
                    subentry_id, tof_params_cache,
                    default_flight_path_m, default_time_resolution_ns,
                )
                ds_sigma_E = compute_sigma_E(exfor_energy, tof_params)
                ds_tof_source = tof_params.source
            else:
                ds_sigma_E = None  # will use bin sigma_E as fallback
                ds_tof_source = "bin_default"

            all_datasets.append({
                'entry': entry,
                'subentry': subentry,
                'exfor_energy_mev': exfor_energy,
                'exfor_df': dataset_df.copy(),
                'n_points': len(dataset_df),
                'experiment_id': f"{entry}.{subentry}",
                'sigma_E_mev': ds_sigma_E,
                'tof_source': ds_tof_source,
            })

    # For each bin, collect ALL datasets with non-trivial overlap weight.
    # Multiple energies from the same experiment are kept — each with its own
    # overlap weight — so the fit benefits from the full angular coverage and
    # cross-bin bridging.  Intra-experiment weights are renormalized so that
    # each experiment's total contribution equals its best single overlap
    # weight (preventing dense-grid experiments from inflating weight).
    overlap_weights: Dict[int, List[Tuple[Dict, float]]] = {}
    for bin_info in energy_bins:
        bin_datasets = []
        for ds in all_datasets:
            # Use per-experiment sigma_E if available, else fall back to bin sigma_E
            sigma_E = ds['sigma_E_mev'] if ds['sigma_E_mev'] is not None else bin_info.sigma_E_mev
            w = compute_overlap_weight(
                exp_energy_mev=ds['exfor_energy_mev'],
                sigma_E_mev=sigma_E,
                bin_lower_mev=bin_info.bin_lower_mev,
                bin_upper_mev=bin_info.bin_upper_mev,
            )
            if w >= min_weight:
                bin_datasets.append((ds, w))

        # Experiment-level renormalization: each experiment's total weight
        # equals its best single overlap (what closest-only dedup would give).
        # This uses all angular data but prevents dense grids from inflating
        # experiment-level weight.
        exp_groups: Dict[str, List[int]] = defaultdict(list)
        for i, (ds, w) in enumerate(bin_datasets):
            exp_groups[ds['experiment_id']].append(i)

        normalized = []
        for exp_id, indices in exp_groups.items():
            weights = [bin_datasets[i][1] for i in indices]
            max_w = max(weights)
            sum_w = sum(weights)
            scale = max_w / sum_w  # shrink so total = max_w
            for i in indices:
                ds, w = bin_datasets[i]
                normalized.append((ds, w * scale))

        overlap_weights[bin_info.index] = normalized

    return overlap_weights


def _run_one_kw_sample(args_tuple):
    """Single kernel-weighted multi-bin MC sample (top-level for Pool.map).

    1. Draw shared normalization factor per experiment (once per sample)
    2. Perturb all datasets (apply norm + pointwise noise)
    3. For each bin: collect datasets with overlap weight, build weighted DataFrame
    4. Fit Legendre coefficients
    5. Return coefficients for all bins

    The SAME perturbed dataset is used for ALL bins it contributes to,
    creating cross-bin correlations.
    """
    (
        s_idx,
        overlap_weights,
        energy_bins_data,
        sigma_norm,
        norm_dist,
        max_degree,
        ridge_lambda,
        base_seed,
        use_band_discrepancy,
        min_points_per_band,
        max_tau_fraction,
        freeze_c0,
        max_sample_order,
        apply_positivity_projection,
        positivity_check_points,
        nominal_coeffs_by_bin,
        frozen_degrees_by_bin,
        max_experiment_weight_fraction,
    ) = args_tuple

    rng = np.random.default_rng(base_seed + s_idx)

    # Step 1: Draw shared normalization factors per experiment
    experiment_norms = {}
    for bin_idx, datasets_and_weights in overlap_weights.items():
        for ds, w in datasets_and_weights:
            exp_id = ds['experiment_id']
            if exp_id not in experiment_norms:
                if norm_dist == "lognormal" and sigma_norm > 0:
                    experiment_norms[exp_id] = rng.lognormal(
                        mean=-0.5 * sigma_norm**2, sigma=sigma_norm
                    )
                elif sigma_norm > 0:
                    experiment_norms[exp_id] = rng.normal(1.0, sigma_norm)
                else:
                    experiment_norms[exp_id] = 1.0

    # Step 2: Perturb all datasets (shared across bins)
    perturbed_datasets = {}
    for bin_idx, datasets_and_weights in overlap_weights.items():
        for ds, w in datasets_and_weights:
            exp_id = ds['experiment_id']
            e_key = f"{exp_id}_{ds['exfor_energy_mev']:.6f}"
            if e_key in perturbed_datasets:
                continue
            df = ds['exfor_df']
            if df.empty:
                continue
            norm_factor = experiment_norms.get(exp_id, 1.0)
            values = df['value'].to_numpy() * norm_factor
            unc = df['unc'].to_numpy()
            noise = rng.normal(0, unc)
            perturbed_datasets[e_key] = {
                'mu': df['mu'].to_numpy(),
                'value': values + noise,
                'unc': unc,
            }

    # Step 3-4: For each bin, collect perturbed data, fit
    sample_coeffs = {}
    for bin_info_data in energy_bins_data:
        bin_idx = bin_info_data['index']
        datasets_and_weights = overlap_weights.get(bin_idx, [])

        if not datasets_and_weights:
            # Use nominal (interpolated)
            nom = nominal_coeffs_by_bin.get(bin_idx)
            if nom is not None:
                sample_coeffs[bin_idx] = endf_normalize_legendre_coeffs(nom, include_a0=False)
            continue

        # Build combined weighted DataFrame
        all_mu = []
        all_values = []
        all_unc = []
        all_weights = []

        for ds, w in datasets_and_weights:
            exp_id = ds['experiment_id']
            e_key = f"{exp_id}_{ds['exfor_energy_mev']:.6f}"
            pert = perturbed_datasets.get(e_key)
            if pert is None:
                continue
            n_pts = len(pert['mu'])
            all_mu.append(pert['mu'])
            all_values.append(pert['value'])
            all_unc.append(pert['unc'])
            all_weights.append(np.full(n_pts, w / n_pts))

        # Apply per-experiment weight capping (analogous to nominal fits)
        if all_mu and max_experiment_weight_fraction < 1.0:
            weights_arr = np.concatenate(all_weights)
            # Track experiment id per point
            exp_ids = []
            for ds, w in datasets_and_weights:
                e_key = f"{ds['experiment_id']}_{ds['exfor_energy_mev']:.6f}"
                pert = perturbed_datasets.get(e_key)
                if pert is not None:
                    exp_ids.extend([ds['experiment_id']] * len(pert['mu']))
            exp_ids = np.array(exp_ids)
            total_w = weights_arr.sum()
            if total_w > 0:
                unique_exps = np.unique(exp_ids)
                for _ in range(5):  # iterative capping
                    changed = False
                    total_w = weights_arr.sum()
                    if total_w <= 0:
                        break
                    for exp in unique_exps:
                        mask = exp_ids == exp
                        frac = weights_arr[mask].sum() / total_w
                        if frac > max_experiment_weight_fraction:
                            scale = max_experiment_weight_fraction / frac
                            weights_arr[mask] *= scale
                            changed = True
                    if not changed:
                        break
                # Rebuild all_weights from capped array
                all_weights = []
                offset = 0
                for mu_arr in all_mu:
                    n = len(mu_arr)
                    all_weights.append(weights_arr[offset:offset + n])
                    offset += n

        if not all_mu:
            nom = nominal_coeffs_by_bin.get(bin_idx)
            if nom is not None:
                sample_coeffs[bin_idx] = endf_normalize_legendre_coeffs(nom, include_a0=False)
            continue

        mu = np.concatenate(all_mu)
        values = np.concatenate(all_values)
        unc = np.concatenate(all_unc)
        weights = np.concatenate(all_weights)

        if len(mu) < 3:
            nom = nominal_coeffs_by_bin.get(bin_idx)
            if nom is not None:
                sample_coeffs[bin_idx] = endf_normalize_legendre_coeffs(nom, include_a0=False)
            continue

        fit_df = pd.DataFrame({'mu': mu, 'value': values, 'unc': unc})
        degree = frozen_degrees_by_bin.get(bin_idx, max_degree)

        try:
            coef_df, _ = sample_legendre_coefficients(
                fit_df,
                value_col="value",
                unc_col="unc",
                degree=degree,
                max_degree=max_degree,
                select_degree=None,
                ridge_lambda=ridge_lambda,
                external_weights=weights,
                n_samples=1,
                stochastic=False,
                use_band_discrepancy=use_band_discrepancy,
                min_points_per_band=min_points_per_band,
                max_tau_fraction=max_tau_fraction,
                freeze_c0=freeze_c0,
            )
            coeffs = coef_df.iloc[0].to_numpy()
            if len(coeffs) < max_degree + 1:
                coeffs = np.pad(coeffs, (0, max_degree + 1 - len(coeffs)))

            # Freeze higher-order coefficients at nominal values
            if max_sample_order is not None:
                nom = nominal_coeffs_by_bin.get(bin_idx)
                if nom is not None:
                    for l in range(max_sample_order + 1, len(coeffs)):
                        if l < len(nom):
                            coeffs[l] = nom[l]

            if apply_positivity_projection:
                from .resample_AD import (
                    check_angular_distribution_positivity,
                    project_to_positive_distribution,
                )
                if not check_angular_distribution_positivity(coeffs, positivity_check_points):
                    frozen = {}
                    if freeze_c0:
                        frozen[0] = coeffs[0]
                    if max_sample_order is not None and max_sample_order + 1 < len(coeffs):
                        frozen.update({i: coeffs[i] for i in range(max_sample_order + 1, len(coeffs))})
                    coeffs = project_to_positive_distribution(
                        coeffs, positivity_check_points, frozen_indices=frozen or None
                    )

            sample_coeffs[bin_idx] = endf_normalize_legendre_coeffs(coeffs, include_a0=False)

        except Exception:
            nom = nominal_coeffs_by_bin.get(bin_idx)
            if nom is not None:
                sample_coeffs[bin_idx] = endf_normalize_legendre_coeffs(nom, include_a0=False)

    return s_idx, sample_coeffs


def run_mc_with_kernel_weights(
    nominal_results: List,  # List[NominalFitResult]
    energy_bins: List[EnergyBinInfo],
    overlap_weights: Dict[int, List[Tuple[Dict, float]]],
    n_samples: int,
    n_workers: int,
    sigma_norm: float,
    norm_dist: str,
    max_degree: int,
    ridge_lambda: float,
    base_seed: int,
    use_band_discrepancy: bool = True,
    min_points_per_band: int = 3,
    max_tau_fraction: float = 0.05,
    freeze_c0: bool = True,
    max_sample_order: Optional[int] = None,
    apply_positivity_projection: bool = False,
    positivity_check_points: int = 101,
    max_experiment_weight_fraction: float = 1.0,
    logger=None,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Orchestrate kernel-weighted multi-bin MC sampling.

    Parameters
    ----------
    nominal_results : list
        Nominal fit results.
    energy_bins : List[EnergyBinInfo]
        Energy bin definitions.
    overlap_weights : dict
        From precompute_overlap_weights().
    n_samples : int
        Number of MC samples.
    n_workers : int
        Number of parallel workers.
    sigma_norm : float
        Per-experiment normalization uncertainty.
    norm_dist : str
        "lognormal" or "normal".
    max_degree : int
        Maximum Legendre degree.
    ridge_lambda : float
        Ridge regularization parameter.
    base_seed : int
        Random seed.
    max_experiment_weight_fraction : float
        Maximum allowed weight fraction per experiment (1.0 = disabled).
    logger : optional
        Logger instance.

    Returns
    -------
    Dict[int, Dict[int, np.ndarray]]
        {sample_idx: {energy_idx: endf_coeffs}} compatible with
        compute_covariance_from_samples().
    """
    logger = logger or _get_logger()

    # Prepare bin data for pickling
    energy_bins_data = [
        {'index': b.index, 'energy_mev': b.energy_mev,
         'bin_lower_mev': b.bin_lower_mev, 'bin_upper_mev': b.bin_upper_mev}
        for b in energy_bins
    ]

    nominal_coeffs_by_bin = {}
    frozen_degrees_by_bin = {}
    for nr in nominal_results:
        if nr.has_data:
            nominal_coeffs_by_bin[nr.energy_index] = nr.nominal_coeffs
            frozen_degrees_by_bin[nr.energy_index] = nr.frozen_degree

    args_list = [
        (
            s_idx,
            overlap_weights,
            energy_bins_data,
            sigma_norm,
            norm_dist,
            max_degree,
            ridge_lambda,
            base_seed,
            use_band_discrepancy,
            min_points_per_band,
            max_tau_fraction,
            freeze_c0,
            max_sample_order,
            apply_positivity_projection,
            positivity_check_points,
            nominal_coeffs_by_bin,
            frozen_degrees_by_bin,
            max_experiment_weight_fraction,
        )
        for s_idx in range(n_samples)
    ]

    if n_workers > 1:
        if logger:
            logger.info(f"  Running kernel-weight MC with {n_workers} workers, {n_samples} samples")
        with Pool(n_workers) as pool:
            results = pool.map(_run_one_kw_sample, args_list)
    else:
        if logger:
            logger.info(f"  Running kernel-weight MC sequentially, {n_samples} samples")
        results = [_run_one_kw_sample(a) for a in args_list]

    # Assemble into expected format
    all_samples: Dict[int, Dict[int, np.ndarray]] = {s_idx: {} for s_idx in range(n_samples)}
    for s_idx, sample_coeffs in results:
        all_samples[s_idx] = sample_coeffs

    if logger:
        n_bins_with_data = sum(1 for b in energy_bins if overlap_weights.get(b.index))
        logger.info(f"  Kernel-weight MC complete: {n_samples} samples, {n_bins_with_data} bins with data")

    return all_samples


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
    snr_threshold: float = 0.0,
    n_neighbors: int = 3,
    logger=None,
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
        for k, e_idx in enumerate(energy_indices):
            coeffs = sample_data.get(e_idx, np.zeros(max_order))
            start = k * max_order
            n_copy = min(len(coeffs), max_order)
            sample_matrix[s_idx, start:start + n_copy] = coeffs[:n_copy]

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

    # Regularize near-zero-mean parameters via neighbor interpolation
    if snr_threshold > 0:
        cov_matrix, reg_diag = regularize_near_zero_relative_covariance(
            cov_rel=cov_matrix,
            mean_params=mean_params,
            cov_abs=cov_abs,
            max_order=max_order,
            snr_threshold=snr_threshold,
            n_neighbors=n_neighbors,
            logger=logger,
        )

    # Compute correlation (from absolute covariance — identical to
    # computing from relative, since the mean factors cancel)
    std = np.sqrt(np.maximum(np.diag(cov_abs), 0.0))
    std[std == 0] = 1.0  # Avoid division by zero
    corr_matrix = cov_abs / np.outer(std, std)
    corr_matrix[np.abs(corr_matrix) < 1e-15] = 0.0

    # Generate labels
    param_labels = [(e_idx, l + 1) for e_idx in energy_indices for l in range(max_order)]

    return cov_matrix, corr_matrix, param_labels, mean_params


def regularize_near_zero_relative_covariance(
    cov_rel: np.ndarray,
    mean_params: np.ndarray,
    cov_abs: np.ndarray,
    max_order: int,
    snr_threshold: float = 1.0,
    n_neighbors: int = 3,
    logger=None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Regularize relative covariance for parameters with near-zero means.

    When |mean| / sigma_abs < snr_threshold, the relative std is an artifact
    of dividing by a near-zero mean, not a genuinely large uncertainty. This
    replaces those explosive relative stds with interpolated values from
    neighboring bins of the same Legendre order, applied via congruence
    transform to preserve correlations and PSD.

    Returns regularized cov_rel and diagnostics dict.
    """
    n_params = len(mean_params)
    n_energies = n_params // max_order

    sigma_abs = np.sqrt(np.maximum(np.diag(cov_abs), 0.0))
    rel_std = np.sqrt(np.maximum(np.diag(cov_rel), 0.0))

    # Compute SNR for each parameter
    snr = np.full(n_params, np.inf)
    nonzero_abs = sigma_abs > 0
    snr[nonzero_abs] = np.abs(mean_params[nonzero_abs]) / sigma_abs[nonzero_abs]

    # Flag parameters with low SNR (near-zero means)
    flagged = snr < snr_threshold

    n_flagged = int(np.sum(flagged))
    if n_flagged == 0:
        diagnostics = {"n_regularized": 0, "n_total": n_params}
        if logger:
            logger.info("  Near-zero regularization: no parameters flagged (all SNR >= threshold)")
        return cov_rel.copy(), diagnostics

    # Build scale vector
    scale = np.ones(n_params)

    for i in range(n_params):
        if not flagged[i]:
            continue
        if rel_std[i] <= 0:
            continue  # already zero relative std, nothing to scale

        # Determine energy bin index k and Legendre order l
        k = i // max_order
        l = i % max_order

        # Collect rel_std from neighboring bins of the same order l
        neighbor_stds = []
        # Walk left
        count = 0
        for kk in range(k - 1, -1, -1):
            j = kk * max_order + l
            if not flagged[j] and rel_std[j] > 0:
                neighbor_stds.append(rel_std[j])
                count += 1
                if count >= n_neighbors:
                    break
        # Walk right
        count = 0
        for kk in range(k + 1, n_energies):
            j = kk * max_order + l
            if not flagged[j] and rel_std[j] > 0:
                neighbor_stds.append(rel_std[j])
                count += 1
                if count >= n_neighbors:
                    break

        if neighbor_stds:
            target_rel_std = float(np.median(neighbor_stds))
        else:
            # Fallback: 100% relative uncertainty
            target_rel_std = 1.0

        scale[i] = target_rel_std / rel_std[i]

    # Apply congruence transform: cov_reg = diag(scale) @ cov_rel @ diag(scale)
    cov_reg = cov_rel * np.outer(scale, scale)

    diagnostics = {
        "n_regularized": n_flagged,
        "n_total": n_params,
        "max_scale_down": float(np.min(scale[flagged])) if n_flagged > 0 else 1.0,
    }

    if logger:
        logger.info(f"  Near-zero regularization: {n_flagged}/{n_params} parameters "
                    f"(SNR < {snr_threshold})")
        # Report before/after stats
        old_max = float(np.max(rel_std[flagged])) if n_flagged > 0 else 0.0
        new_rel_std = np.sqrt(np.maximum(np.diag(cov_reg), 0.0))
        new_max = float(np.max(new_rel_std[flagged])) if n_flagged > 0 else 0.0
        logger.info(f"  Max relative std (flagged): {old_max:.4f} -> {new_max:.4f}")

    return cov_reg, diagnostics


def regularize_post_rescaling(
    cov_rel: np.ndarray,
    max_order: int,
    max_rel_std: float = 1.0,
    n_neighbors: int = 3,
    mc_abs_std: Optional[np.ndarray] = None,
    nominal_means: Optional[np.ndarray] = None,
    deflation_threshold: float = 0.5,
    logger=None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Bidirectional regularization of relative covariance after MC-mean ->
    nominal rescaling.

    After rescaling ``cov_grouped * outer(mg_scale, mg_scale)`` where
    ``mg_scale = mc_mean / nominal``, relative stds can both explode
    (when |mc_mean| >> |nominal|) and collapse (when |mc_mean| << |nominal|).

    Both directions are fixed with the **same strategy**: neighbor-
    interpolated relative std from the same Legendre order, applied via
    congruence transform (preserves correlations and PSD).

    Detection criteria
    ------------------
    - **High-side** (inflation): ``rel_std > max_rel_std``
    - **Low-side** (deflation): ``abs_std < deflation_threshold * mc_abs_std``
      where ``abs_std = rel_std * |nominal_mean|``

    Fix strategy (both sides)
    -------------------------
    Walk left/right from the flagged bin collecting up to ``n_neighbors``
    unflagged rel_std values of the same order, take their median as
    target.  For deflated bins, also compute the MC-informed target
    (``deflation_threshold * mc_abs_std / |nominal|``) and use the
    larger of the two (conservative).

    Parameters
    ----------
    cov_rel : np.ndarray
        Relative covariance matrix (post-rescaling).
    max_order : int
        Number of Legendre orders per energy bin.
    max_rel_std : float
        Flag parameters with relative std above this.
    n_neighbors : int
        Neighbors to seek on each side for interpolation.
    mc_abs_std : np.ndarray, optional
        Original MC absolute standard deviations (before rescaling).
        Enables deflation detection.
    nominal_means : np.ndarray, optional
        Nominal mean values (denominator of relative covariance).
        Required together with mc_abs_std for deflation detection.
    deflation_threshold : float
        Flag if post-rescaling abs_std < this fraction of mc_abs_std.
    logger : optional
        Logger instance.

    Returns
    -------
    cov_reg : np.ndarray
        Regularized relative covariance.
    diagnostics : dict
        Counts and summary statistics.
    """
    n_params = cov_rel.shape[0]
    n_energies = n_params // max_order

    rel_std = np.sqrt(np.maximum(np.diag(cov_rel), 0.0))

    # --- High-side flags: relative std too large ---
    flagged_high = rel_std > max_rel_std

    # --- Low-side flags: rescaling crushed absolute uncertainty ---
    flagged_low = np.zeros(n_params, dtype=bool)
    deflation_available = mc_abs_std is not None and nominal_means is not None
    if deflation_available:
        current_abs_std = rel_std * np.abs(nominal_means)
        deflated = (
            (mc_abs_std > 1e-20)
            & (current_abs_std < deflation_threshold * mc_abs_std)
            & (np.abs(nominal_means) > 1e-20)
            & ~flagged_high
        )
        flagged_low = deflated

    flagged = flagged_high | flagged_low
    n_flagged = int(np.sum(flagged))

    if n_flagged == 0:
        diagnostics = {"n_regularized": 0, "n_total": n_params,
                       "n_capped": 0, "n_deflated": 0}
        if logger:
            logger.info("  Post-rescaling regularization: no parameters flagged")
        return cov_rel.copy(), diagnostics

    # Per-order global fallback: median of unflagged stds
    order_fallback = np.full(max_order, max_rel_std)
    for l in range(max_order):
        order_indices = np.arange(l, n_params, max_order)
        unflagged_stds = rel_std[order_indices][~flagged[order_indices]]
        unflagged_stds = unflagged_stds[unflagged_stds > 0]
        if len(unflagged_stds) > 0:
            order_fallback[l] = float(np.median(unflagged_stds))

    # Neighbor-interpolated target (same logic for both directions)
    def _neighbor_target(i: int) -> Tuple[float, int, bool]:
        k = i // max_order
        l = i % max_order
        neighbor_stds = []
        count = 0
        for kk in range(k - 1, -1, -1):
            j = kk * max_order + l
            if not flagged[j] and rel_std[j] > 0:
                neighbor_stds.append(rel_std[j])
                count += 1
                if count >= n_neighbors:
                    break
        count = 0
        for kk in range(k + 1, n_energies):
            j = kk * max_order + l
            if not flagged[j] and rel_std[j] > 0:
                neighbor_stds.append(rel_std[j])
                count += 1
                if count >= n_neighbors:
                    break
        if neighbor_stds:
            return float(np.median(neighbor_stds)), len(neighbor_stds), False
        return order_fallback[l], 0, True

    # Build scale vector
    scale = np.ones(n_params)
    targets = {}  # i -> (target, n_neighbors_used, used_fallback, flag_type)

    for i in range(n_params):
        if not flagged[i]:
            continue

        if flagged_high[i]:
            if rel_std[i] <= 0:
                continue
            target, n_nb, fb = _neighbor_target(i)
            scale[i] = target / rel_std[i]
            targets[i] = (target, n_nb, fb, "high")

        elif flagged_low[i]:
            target, n_nb, fb = _neighbor_target(i)
            # MC-informed target: what the relative std should be to preserve
            # deflation_threshold of the original MC absolute uncertainty
            mc_target_rel = (
                deflation_threshold * mc_abs_std[i] / np.abs(nominal_means[i])
                if np.abs(nominal_means[i]) > 1e-20 else target
            )
            # Conservative: use the larger of neighbor-interpolated and MC-informed
            target = max(target, mc_target_rel)
            if rel_std[i] > 0:
                scale[i] = target / rel_std[i]
            targets[i] = (target, n_nb, fb, "low")

    # Apply congruence transform: C' = S @ C @ S (PSD-preserving)
    cov_reg = cov_rel * np.outer(scale, scale)

    n_high = int(np.sum(flagged_high))
    n_low = int(np.sum(flagged_low))

    diagnostics = {
        "n_regularized": n_flagged,
        "n_total": n_params,
        "n_capped": n_high,
        "n_deflated": n_low,
    }

    if logger:
        parts = []
        if n_high > 0:
            parts.append(f"{n_high} capped (>{max_rel_std*100:.0f}%)")
        if n_low > 0:
            parts.append(f"{n_low} deflated (<{deflation_threshold*100:.0f}% of MC)")
        logger.info(f"  Post-rescaling regularization: {n_flagged}/{n_params} — "
                    + ", ".join(parts))
        new_rel_std = np.sqrt(np.maximum(np.diag(cov_reg), 0.0))
        for l in range(max_order):
            order_indices = np.arange(l, n_params, max_order)
            # High-side
            order_flagged_h = flagged_high[order_indices]
            n_oh = int(np.sum(order_flagged_h))
            if n_oh > 0:
                old_max = float(np.max(rel_std[order_indices][order_flagged_h]))
                new_max = float(np.max(new_rel_std[order_indices][order_flagged_h]))
                order_targets = [targets[idx][0] for idx in order_indices[order_flagged_h] if idx in targets]
                order_n_fb = sum(1 for idx in order_indices[order_flagged_h] if idx in targets and targets[idx][2])
                tgt_min = min(order_targets) if order_targets else 0
                tgt_max = max(order_targets) if order_targets else 0
                logger.info(f"    l={l+1}: {n_oh} capped "
                            f"(max {old_max*100:.1f}% -> {new_max*100:.1f}%), "
                            f"targets [{tgt_min*100:.1f}%-{tgt_max*100:.1f}%], "
                            f"{order_n_fb} used fallback")
            # Low-side
            order_flagged_l = flagged_low[order_indices]
            n_ol = int(np.sum(order_flagged_l))
            if n_ol > 0 and deflation_available:
                old_min_abs = float(np.min(
                    rel_std[order_indices][order_flagged_l]
                    * np.abs(nominal_means[order_indices][order_flagged_l])
                ))
                new_min_abs = float(np.min(
                    new_rel_std[order_indices][order_flagged_l]
                    * np.abs(nominal_means[order_indices][order_flagged_l])
                ))
                logger.info(f"    l={l+1}: {n_ol} deflated "
                            f"(min abs_std {old_min_abs:.6f} -> {new_min_abs:.6f})")

    return cov_reg, diagnostics


def log_rel_std_profile(
    cov_rel: np.ndarray,
    max_order: int,
    stage: str,
    logger=None,
    verbose: bool = False,
) -> None:
    """Log per-order percentile statistics of relative std for diagnostics.

    Parameters
    ----------
    cov_rel : np.ndarray
        Relative covariance matrix.
    max_order : int
        Number of Legendre orders per energy bin.
    stage : str
        Label for the pipeline stage (e.g. "FG post-rescale").
    logger : optional
        Logger instance.
    verbose : bool
        If True, also log per-bin rel_std values for each order, useful
        for tracing exactly where spikes/serrations originate.
    """
    if logger is None:
        return
    n_params = cov_rel.shape[0]
    n_energies = n_params // max_order
    rel_std = np.sqrt(np.maximum(np.diag(cov_rel), 0.0))
    for l in range(max_order):
        order_vals = rel_std[l::max_order]
        nz = order_vals[order_vals > 1e-15]
        n_zero = int(np.sum(order_vals <= 1e-15))
        if len(nz) == 0:
            logger.info(f"  [DIAG {stage}] l={l+1}: all zero (N={len(order_vals)})")
            continue
        p5, p25, p50, p75, p95 = np.percentile(nz, [5, 25, 50, 75, 95])
        mx = np.max(nz)
        logger.info(
            f"  [DIAG {stage}] l={l+1}: p5={p5:.1%} p25={p25:.1%} p50={p50:.1%} "
            f"p75={p75:.1%} p95={p95:.1%} max={mx:.1%} (N={len(nz)}, "
            f"zero={n_zero}/{len(order_vals)})"
        )
        if verbose:
            # Dump every bin value for full traceability
            vals_str = ", ".join(f"{v:.4f}" for v in order_vals)
            logger.info(f"  [DIAG {stage}] l={l+1} bins: [{vals_str}]")


def smooth_absent_order_uncertainties(
    cov_rel: np.ndarray,
    valid_mask: np.ndarray,
    max_order: int,
    min_rel_std: float = 0.005,
    dip_fraction: float = 0.50,
    spike_factor: float = 3.0,
    dip_n_neighbors: int = 3,
    median_fill_threshold: float = 0.50,
    logger=None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fill in relative uncertainties for Legendre orders absent or near-zero
    in some energy bins, and smooth isolated spikes.

    Four classes of entries are smoothed:

    1. **Absent** — order was not fitted (not in valid_mask) or its diagonal
       is effectively zero.  These are structurally zero from covariance
       construction.

    2. **Below floor** — order was fitted but its relative std is below
       ``min_rel_std``.  This happens when the Legendre coefficient is
       near zero and both the mean and variance are tiny, producing a
       misleadingly small relative uncertainty.

    3. **Isolated dips** — relative std is much lower than neighbors of
       the same order (``< dip_fraction * median(neighbors)``).  These
       are artifacts of individual bins where coefficients pass through
       zero.

    4. **Isolated spikes** — relative std is much higher than neighbors
       of the same order (``> spike_factor * median(neighbors)``).  These
       are artifacts of individual bins where coefficients are poorly
       constrained.

    All flagged entries are replaced by linear interpolation from
    neighboring unflagged bins of the same order.  Scale-down (spikes)
    is always PSD-preserving; scale-up (dips/absent) uses eigenvalue
    clipping afterward if needed.

    Parameters
    ----------
    cov_rel : np.ndarray
        Relative covariance matrix (modified in-place copy returned).
    valid_mask : np.ndarray of bool
        True for parameters that were fitted (from frozen_degree).
        Length = n_energies * max_order.
    max_order : int
        Number of Legendre orders per energy bin.
    min_rel_std : float
        Floor below which relative std is treated as absent (default 0.5%).
    dip_fraction : float
        Flag entries with rel_std < dip_fraction * median(neighbors)
        (default 0.50 = half of neighbor median).
    spike_factor : float
        Flag entries with rel_std > spike_factor * median(neighbors)
        (default 3.0 = three times neighbor median).
    dip_n_neighbors : int
        Number of neighbors on each side for dip/spike detection (default 3).
    median_fill_threshold : float
        If the fraction of flagged bins for an order exceeds this threshold,
        use flat median fill instead of linear interpolation (default 0.50).
    logger : optional
        Logger instance.

    Returns
    -------
    cov_smoothed : np.ndarray
        Covariance with absent/near-zero/dip/spike diagonals filled.
    diagnostics : dict
        Per-order counts of bins smoothed, broken down by category.
    """
    n_params = cov_rel.shape[0]
    n_energies = n_params // max_order

    rel_std = np.sqrt(np.maximum(np.diag(cov_rel), 0.0))

    # Build a global scale vector: scale[i] = target_std / old_std
    # Applied as a single congruence transform at the end.
    scale = np.ones(n_params)
    # For truly-zero entries, we'll inject diagonal values directly.
    inject_diag = {}  # i -> target_variance (for zero-row/col entries)

    total_smoothed = 0
    per_order = {}

    for l in range(max_order):
        order_indices = np.arange(l, n_params, max_order)  # length = n_energies
        order_rel_std = rel_std[order_indices].copy()
        order_valid = valid_mask[order_indices]

        # --- Category 1: absent (not fitted or truly zero) ---
        absent = ~order_valid | (order_rel_std < 1e-15)

        # --- Category 2: below floor (fitted but rel_std too small) ---
        below_floor = order_valid & (order_rel_std > 0) & (order_rel_std < min_rel_std)

        # --- Categories 3 & 4: isolated dips and spikes (iterative) ---
        # After interpolating the deepest outliers, shallower ones can emerge.
        # Iterate until no new outliers are found (max 5 passes).
        dip = np.zeros(len(order_rel_std), dtype=bool)
        spike = np.zeros(len(order_rel_std), dtype=bool)
        for _pass in range(5):
            new_outliers = 0
            for k in range(len(order_rel_std)):
                if absent[k] or below_floor[k] or dip[k] or spike[k]:
                    continue
                if order_rel_std[k] <= 0:
                    continue
                excluded = absent | below_floor | dip | spike
                neighbor_vals = []
                for kk in range(max(0, k - dip_n_neighbors), k):
                    if not excluded[kk] and order_rel_std[kk] > min_rel_std:
                        neighbor_vals.append(order_rel_std[kk])
                for kk in range(k + 1, min(len(order_rel_std), k + dip_n_neighbors + 1)):
                    if not excluded[kk] and order_rel_std[kk] > min_rel_std:
                        neighbor_vals.append(order_rel_std[kk])
                if len(neighbor_vals) >= 2:
                    med = float(np.median(neighbor_vals))
                    if order_rel_std[k] < dip_fraction * med:
                        dip[k] = True
                        new_outliers += 1
                    elif order_rel_std[k] > spike_factor * med:
                        spike[k] = True
                        new_outliers += 1
            if new_outliers == 0:
                break
            # Update working rel_std with interpolated values for next pass
            flagged_pass = absent | below_floor | dip | spike
            present_pass = ~flagged_pass & (order_rel_std > min_rel_std)
            if np.any(present_pass) and np.any(flagged_pass):
                pi = np.where(present_pass)[0]
                fi = np.where(flagged_pass)[0]
                order_rel_std[fi] = np.interp(fi, pi, order_rel_std[pi])

        # Combine all flags
        flagged = absent | below_floor | dip | spike
        present = ~flagged & (rel_std[order_indices] > min_rel_std)

        n_flagged = int(np.sum(flagged))
        if n_flagged == 0 or not np.any(present):
            per_order[l + 1] = (0, 0, 0, 0)
            continue

        # Interpolate using present bins as anchor points (from original values)
        present_idx = np.where(present)[0]
        present_vals = rel_std[order_indices[present_idx]]

        flagged_idx = np.where(flagged)[0]
        absent_fraction = int(np.sum(absent)) / len(order_rel_std)
        if absent_fraction > median_fill_threshold and len(present_vals) > 0:
            # High-absence order: flat median fill avoids zigzag from sparse anchors
            filled_vals = np.full(len(flagged_idx), np.median(present_vals))
            if logger:
                logger.info(f"    l={l+1}: median fill ({absent_fraction:.0%} absent, "
                            f"median={np.median(present_vals):.4f})")
        elif absent_fraction > 0.25 and len(present_vals) >= 5:
            # Moderate-absence order: running-median interpolation avoids zigzag
            # from sparse linear interpolation anchors.  First interpolate linearly,
            # then smooth with a running median to remove serration.
            filled_vals = np.interp(flagged_idx, present_idx, present_vals)
            # Build a full vector (present + filled) for running median
            full_std = rel_std[order_indices].copy()
            full_std[flagged_idx] = filled_vals
            # Running median over the filled entries only
            _rm_half = max(dip_n_neighbors, 3)
            for _fi, _fk in enumerate(flagged_idx):
                lo = max(0, _fk - _rm_half)
                hi = min(len(full_std), _fk + _rm_half + 1)
                filled_vals[_fi] = np.median(full_std[lo:hi])
            if logger:
                logger.info(f"    l={l+1}: running-median interp ({absent_fraction:.0%} absent, "
                            f"window={2*_rm_half+1})")
        else:
            filled_vals = np.interp(flagged_idx, present_idx, present_vals)

        # Record scale factors or diagonal injections
        for k_local, val in zip(flagged_idx, filled_vals):
            i = order_indices[k_local]
            old_std = rel_std[i]
            if old_std > 1e-15:
                scale[i] = val / old_std
            else:
                inject_diag[i] = val ** 2

        n_absent = int(np.sum(absent))
        n_floor = int(np.sum(below_floor))
        n_dip = int(np.sum(dip))
        n_spike = int(np.sum(spike))
        per_order[l + 1] = (n_absent, n_floor, n_dip, n_spike)
        total_smoothed += n_flagged

    # Apply single congruence transform: C' = S @ C @ S
    # Scale-down (scale < 1) is always PSD-preserving.
    # Scale-up (scale > 1) can create |corr| > 1 if the original off-diagonal
    # was already large relative to the new diagonal.  We clip eigenvalues
    # afterward to restore PSD if needed.
    cov_smoothed = cov_rel * np.outer(scale, scale)

    # Inject diagonal values for truly-zero entries
    for i, var in inject_diag.items():
        cov_smoothed[i, i] = var

    # PSD projection: clip negative eigenvalues (if any)
    if np.any(scale > 1.0) or inject_diag:
        eigvals, eigvecs = np.linalg.eigh(cov_smoothed)
        min_eig = np.min(eigvals)
        if min_eig < -1e-14:
            eigvals = np.maximum(eigvals, 0.0)
            cov_smoothed = eigvecs @ np.diag(eigvals) @ eigvecs.T
            cov_smoothed = (cov_smoothed + cov_smoothed.T) / 2.0
            if logger:
                logger.info(f"  Absent-order smoothing: PSD projection applied "
                            f"(min_eig={min_eig:.2e})")

    diagnostics = {
        "n_smoothed": total_smoothed,
        "n_total": n_params,
        "per_order": per_order,
    }

    if logger:
        if total_smoothed > 0:
            parts = []
            for l_idx, counts in sorted(per_order.items()):
                if isinstance(counts, tuple):
                    if len(counts) == 4:
                        n_abs, n_fl, n_dp, n_sp = counts
                    else:
                        n_abs, n_fl, n_dp = counts
                        n_sp = 0
                    total_l = n_abs + n_fl + n_dp + n_sp
                else:
                    total_l = counts
                    n_abs = n_fl = n_dp = n_sp = 0
                if total_l > 0:
                    detail = []
                    if n_abs: detail.append(f"{n_abs} absent")
                    if n_fl: detail.append(f"{n_fl} floor")
                    if n_dp: detail.append(f"{n_dp} dip")
                    if n_sp: detail.append(f"{n_sp} spike")
                    parts.append(f"l={l_idx}: {total_l} ({', '.join(detail)})")
            logger.info(f"  Absent-order smoothing: {total_smoothed}/{n_params} "
                        f"diagonal entries filled ({', '.join(parts)})")
        else:
            logger.info("  Absent-order smoothing: no absent bins to fill")

    return cov_smoothed, diagnostics


def cap_order_relative_uncertainty(
    cov_rel: np.ndarray,
    max_order: int,
    order_caps: Dict[int, float],
    logger=None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Cap relative uncertainties for specified Legendre orders via congruence transform.

    Higher orders (e.g. l=4,5,6) may be weakly constrained, producing large
    relative uncertainties.  This function scales down those rows/columns so
    that rel_std <= cap, preserving correlations and PSD.

    Parameters
    ----------
    cov_rel : np.ndarray
        Relative covariance matrix.
    max_order : int
        Number of Legendre orders per energy bin.
    order_caps : dict {int: float}
        1-indexed order -> maximum allowed relative std.
        E.g. {4: 0.20, 5: 0.15, 6: 0.10}.
    logger : optional
        Logger instance.

    Returns
    -------
    cov_capped : np.ndarray
        Capped covariance matrix.
    diagnostics : dict
        Per-order counts and before/after max rel_std.
    """
    n_params = cov_rel.shape[0]
    rel_std = np.sqrt(np.maximum(np.diag(cov_rel), 0.0))
    scale = np.ones(n_params)

    per_order_counts = {}
    per_order_before = {}
    per_order_after = {}
    total_capped = 0

    for i in range(n_params):
        order = (i % max_order) + 1  # 1-indexed
        if order not in order_caps:
            continue
        cap = order_caps[order]
        if rel_std[i] > cap:
            scale[i] = cap / rel_std[i]

    # Gather diagnostics per order
    for order, cap in order_caps.items():
        order_indices = np.arange(order - 1, n_params, max_order)
        order_stds = rel_std[order_indices]
        flagged = int(np.sum(order_stds > cap))
        per_order_counts[order] = flagged
        per_order_before[order] = float(np.max(order_stds)) if len(order_stds) > 0 else 0.0
        # After capping: std * scale
        capped_stds = order_stds * scale[order_indices]
        per_order_after[order] = float(np.max(capped_stds)) if len(capped_stds) > 0 else 0.0
        total_capped += flagged

    # Congruence transform: cov_capped = S @ cov @ S  where S = diag(scale)
    cov_capped = cov_rel * np.outer(scale, scale)

    diagnostics = {
        "n_capped": total_capped,
        "n_total": n_params,
        "per_order_counts": per_order_counts,
        "per_order_max_before": per_order_before,
        "per_order_max_after": per_order_after,
    }

    if logger:
        if total_capped > 0:
            parts = []
            for order in sorted(per_order_counts.keys()):
                n = per_order_counts[order]
                if n > 0:
                    parts.append(
                        f"l={order}: {n} capped "
                        f"(max {per_order_before[order]:.4f} -> {per_order_after[order]:.4f})"
                    )
            logger.info(f"  Order-dependent cap: {total_capped}/{n_params} entries capped "
                        f"({', '.join(parts)})")
        else:
            logger.info("  Order-dependent cap: no entries exceed caps")

    return cov_capped, diagnostics


def smooth_diagonal_median(
    cov_rel: np.ndarray,
    max_order: int,
    window: int = 5,
    logger=None,
) -> np.ndarray:
    """
    Apply Gaussian-weighted moving average to each order's diagonal of the
    relative covariance, then rescale via congruence transform.

    Unlike a running median (which selects a value), a weighted average
    genuinely blends nearby values.  This breaks serrated present/absent
    patterns where adjacent bins alternate between natural MC uncertainty
    and filled/capped values — even when the alternation is ~50%.

    The Gaussian kernel has sigma = window/4 (so the window spans ~2σ on
    each side).  Edge bins use a truncated, renormalized kernel.

    Parameters
    ----------
    cov_rel : np.ndarray
        Relative covariance matrix.
    max_order : int
        Number of Legendre orders per energy bin.
    window : int
        Full window width for the Gaussian kernel (>=3).
        The Gaussian sigma = window / 4.
        0 or 1 disables smoothing.
    logger : optional
        Logger instance.

    Returns
    -------
    cov_smoothed : np.ndarray
        Covariance with smoothed diagonal.
    """
    if window < 3:
        return cov_rel.copy()

    n_params = cov_rel.shape[0]
    n_energies = n_params // max_order
    rel_std = np.sqrt(np.maximum(np.diag(cov_rel), 0.0))

    # Build Gaussian kernel
    sigma = window / 4.0
    half = window // 2
    kernel_x = np.arange(-half, half + 1, dtype=float)
    kernel = np.exp(-0.5 * (kernel_x / sigma) ** 2)
    # kernel is normalized per-position to handle edges

    scale = np.ones(n_params)
    total_changed = 0

    for l in range(max_order):
        order_indices = np.arange(l, n_params, max_order)
        order_std = rel_std[order_indices].copy()
        n = len(order_std)

        # Gaussian-weighted moving average
        smoothed = np.empty(n)
        for k in range(n):
            lo = max(0, k - half)
            hi = min(n, k + half + 1)
            # Slice kernel to match the valid range
            k_lo = lo - (k - half)  # offset into kernel
            k_hi = k_lo + (hi - lo)
            w = kernel[k_lo:k_hi]
            w_sum = w.sum()
            if w_sum > 0:
                smoothed[k] = np.dot(w, order_std[lo:hi]) / w_sum
            else:
                smoothed[k] = order_std[k]

        # Compute scale factors
        for k in range(n):
            i = order_indices[k]
            if order_std[k] > 1e-15 and smoothed[k] > 1e-15:
                s = smoothed[k] / order_std[k]
                if abs(s - 1.0) > 1e-6:
                    scale[i] = s
                    total_changed += 1

    # Congruence transform
    cov_smoothed = cov_rel * np.outer(scale, scale)

    # PSD projection if needed (scale-up can break PSD)
    if np.any(scale > 1.0 + 1e-6):
        eigvals, eigvecs = np.linalg.eigh(cov_smoothed)
        min_eig = np.min(eigvals)
        if min_eig < -1e-14:
            eigvals = np.maximum(eigvals, 0.0)
            cov_smoothed = eigvecs @ np.diag(eigvals) @ eigvecs.T
            cov_smoothed = (cov_smoothed + cov_smoothed.T) / 2.0
            if logger:
                logger.info(f"  Diagonal Gaussian smoothing: PSD projection applied (min_eig={min_eig:.2e})")

    if logger:
        logger.info(f"  Diagonal Gaussian smoothing (window={window}, σ={sigma:.1f}): "
                    f"{total_changed}/{n_params} entries adjusted")

    return cov_smoothed


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


def _extract_correlation_matrix(cov: np.ndarray) -> np.ndarray:
    """Extract correlation matrix from covariance with safe zero-variance handling."""
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    std_safe = std.copy()
    std_safe[std_safe < 1e-30] = 1.0
    corr = cov / np.outer(std_safe, std_safe)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


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

    # 4. Build full correlation matrix (vectorized)
    corr = np.eye(n)

    # Precompute pairwise quantities
    E = param_energies
    S = param_sigma_E
    sigma_eff_mat = (S[:, None] + S[None, :]) / 2.0
    dE_mat = E[:, None] - E[None, :]
    rho_E_mat = np.exp(-dE_mat**2 / (2.0 * sigma_eff_mat**2))

    same_e = param_e_pos[:, None] == param_e_pos[None, :]
    same_l = param_orders[:, None] == param_orders[None, :]
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)

    if valid_mask is not None:
        valid_pair = valid_mask[:, None] & valid_mask[None, :]
    else:
        valid_pair = np.ones((n, n), dtype=bool)

    active = upper & valid_pair

    # Same energy bin -> stochastic cross-order correlation
    mask_same_e = same_e & active
    corr[mask_same_e] = corr_stochastic[mask_same_e]

    # Different energy, same order -> pure Gaussian decay
    mask_diff_e_same_l = ~same_e & same_l & active
    corr[mask_diff_e_same_l] = rho_E_mat[mask_diff_e_same_l]

    # Different energy, different order -> Gaussian * cross-order factor
    mask_diff_e_diff_l = ~same_e & ~same_l & active
    if np.any(mask_diff_e_diff_l):
        # Cross-order factor: stochastic correlation at energy_pos of row i
        idx_i = param_e_pos[:, None] * max_order + param_orders[:, None]
        idx_j = param_e_pos[:, None] * max_order + param_orders[None, :]
        # Clip indices to valid range
        max_idx = corr_stochastic.shape[0] - 1
        idx_i_clip = np.clip(idx_i, 0, max_idx)
        idx_j_clip = np.clip(idx_j, 0, max_idx)
        cross_order_factor = corr_stochastic[idx_i_clip, idx_j_clip]
        corr[mask_diff_e_diff_l] = (rho_E_mat * cross_order_factor)[mask_diff_e_diff_l]

    # Mirror upper triangle to lower
    corr = corr + corr.T - np.diag(np.diag(corr))

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
        n_near_zero = int(np.sum(np.abs(mean_params) < 1e-6))
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
            if abs(min_eig) < 1e-10:
                logger.info(f"  -> Eigenvalues at machine-epsilon level; numerical noise, not a real PSD violation")
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
        # Exclude near-zero mean parameters: cov_abs = cov_rel * outer(mean, mean)
        # makes target_std ≈ 0 when mean ≈ 0, but L @ z still couples noise into
        # those dimensions, causing ratio blow-up that is numerical, not physical.
        mask = (target_std > 1e-30) & (np.abs(mean_params) > 1e-6)
        if np.any(mask):
            ratio = sample_std[mask] / target_std[mask]
            n_skipped = int(np.sum(np.abs(mean_params) <= 1e-6))
            logger.info(f"  Sample validation — std ratio (sample/target): "
                        f"mean={np.mean(ratio):.3f}, std={np.std(ratio):.3f}"
                        f" ({int(np.sum(mask))}/{n_params} params, "
                        f"{n_skipped} near-zero-mean excluded)")
            if abs(np.mean(ratio) - 1.0) > 0.15:
                logger.warning(f"  WARNING: Sample std deviates >15% from target — "
                               f"consider increasing n_samples (currently {n_samples})")

        # Check adjacent-bin sample correlations (compute only needed pairs, not full corrcoef)
        if max_order >= 1 and len(energy_indices) >= 2:
            adj_sample_corrs = []
            for k in range(len(energy_indices) - 1):
                i_p = k * max_order
                j_p = (k + 1) * max_order
                if i_p < n_params and j_p < n_params:
                    r = np.corrcoef(sample_matrix[:, i_p], sample_matrix[:, j_p])[0, 1]
                    adj_sample_corrs.append(r)
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
# FORWARD-FILL RELATIVE UNCERTAINTY FOR ABSENT ORDERS
# =============================================================================

def forward_fill_rel_std(
    cov_rel: np.ndarray,
    max_order: int,
    floor: float = 0.005,
    logger=None,
) -> np.ndarray:
    """
    Forward-fill (sample-and-hold) relative-std for low-uncertainty positions.

    For each Legendre order, a bin is considered "absent" when its rel_std
    falls below *floor* (default 0.5%).  Absent-bin diagonal entries are
    replaced by the last value above the floor (forward fill).  Leading
    absent bins are backward-filled from the first above-floor value.

    Only the **diagonal** is modified — off-diagonal terms are left as-is.
    The nominal coefficients remain zero for absent orders, so
    ``abs_unc = nominal × rel_unc = 0`` is preserved during propagation.

    Parameters
    ----------
    cov_rel : np.ndarray, shape (N, N)
        Relative covariance matrix, laid out as
        ``[bin0_l1, bin0_l2, ..., bin0_lM, bin1_l1, ...]``
        where M = max_order.
    max_order : int
        Number of Legendre orders per energy bin (e.g. 6).
    floor : float
        Rel-std values below this threshold are treated as absent and
        forward-filled.  Default 0.005 (0.5%).
    logger : optional
        Logger for diagnostics.

    Returns
    -------
    np.ndarray
        Modified copy of cov_rel with forward-filled diagonal.
    """
    cov = cov_rel.copy()
    n = cov.shape[0]
    n_bins = n // max_order
    if n_bins == 0:
        return cov

    total_filled = 0

    for l_idx in range(max_order):
        indices = [k * max_order + l_idx for k in range(n_bins)]
        stds = np.array([np.sqrt(max(cov[i, i], 0.0)) for i in indices])

        # Present = above floor
        present_mask = stds >= floor

        n_present = int(np.sum(present_mask))
        if n_present == 0:
            continue  # nothing above floor for this order

        filled_stds = stds.copy()

        # Forward fill: carry last present value through below-floor gaps
        last_val = 0.0
        for k in range(n_bins):
            if present_mask[k]:
                last_val = filled_stds[k]
            elif last_val > 0.0:
                filled_stds[k] = last_val
                total_filled += 1

        # Backward fill leading absent bins from first present value
        first_present = int(np.argmax(present_mask))
        if first_present > 0:
            n_leading = int(np.sum(~present_mask[:first_present]))
            total_filled += n_leading
            filled_stds[:first_present] = filled_stds[first_present]

        # Write back to diagonal as variance (only absent bins)
        for k in range(n_bins):
            if not present_mask[k]:
                idx = indices[k]
                cov[idx, idx] = filled_stds[k] ** 2

    if logger:
        n_below = sum(
            int(np.sum(np.array([np.sqrt(max(cov_rel[k * max_order + l, k * max_order + l], 0.0))
                                  for k in range(n_bins)]) < floor))
            for l in range(max_order)
        )
        logger.info(
            f"  Forward-fill rel_std: filled {total_filled}/{n_below} entries "
            f"below {floor*100:.1f}% floor across {n_bins} bins × {max_order} orders"
        )

    return cov


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
