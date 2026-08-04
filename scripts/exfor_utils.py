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

    def warning(self, msg: str, console: bool = True):
        self.logger.warning(msg)
        if console:
            print(f"[WARNING] {msg}")

    def error(self, msg: str, console: bool = True):
        self.logger.error(msg)
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


# Manifest-application diagnostics for build_exfor_cache_from_objects.
# Reset at the start of each call; exposed via get_last_manifest_stats() so
# the caller can include the counts in the run audit log.
_last_manifest_stats: Dict[str, Any] = {
    'attempted': 0,
    'failed': 0,
    'failures': [],  # list of (entry, subentry, exception_repr)
}


def get_last_manifest_stats() -> Dict[str, Any]:
    """Return diagnostics from the most recent build_exfor_cache_from_objects call."""
    return dict(_last_manifest_stats)


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
    tau_F: float = 1.0                   # Forward band scale factor (≥1.0)
    tau_M: float = 1.0                   # Mid band scale factor (≥1.0)
    tau_B: float = 1.0                   # Backward band scale factor (≥1.0)
    interpolated: bool = False           # Whether coefficients were interpolated
    endf_index: Optional[int] = None     # Nearest index in original ENDF grid (for I/O)


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
    experiment_weights: Dict[str, float]            # {entry.subentry: weight_frac} before capping
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
    min_points_ref: float = 5.0,
) -> float:
    """Compute per-bin reliability alpha from KW overlap diagnostics.

    Returns alpha in [0, 1] where alpha=1 means pure KW, alpha=0 means
    pure Gaussian.

    Parameters
    ----------
    min_points_ref : float
        Minimum angular data points for full KW trust.  Bins where the
        overlap-weighted median n_points is below this get alpha scaled
        down linearly.  Should be set to (max_legendre_order + 1).
    """
    if interpolated or kw_diag is None:
        return 0.0

    if not np.isfinite(kw_diag.n_eff_kw):
        return 0.0

    # Hard zero: need at least 2 experiments for inter-experiment correlations
    if kw_diag.n_experiments_kw < min_experiments:
        return 0.0

    # Base alpha: sigmoid on experiment-level n_eff
    x = (kw_diag.n_eff_kw - n_eff_mid) / n_eff_scale
    alpha = alpha_min_data + (alpha_max - alpha_min_data) / (1.0 + np.exp(-x))

    # Quality penalty: penalize bins where experiments have too few angular
    # points to reliably fit Legendre coefficients
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

    # Hard zero: need at least min_experiments for inter-experiment correlations
    if diagnostics.n_experiments < min_experiments:
        return 0.0

    # Base alpha: sigmoid on n_eff mapped to [alpha_min_data, alpha_max]
    x = (diagnostics.n_eff - n_eff_mid) / n_eff_scale
    alpha = alpha_min_data + (alpha_max - alpha_min_data) / (1.0 + np.exp(-x))

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
    reference_grid_ev: Optional[np.ndarray] = None,
    delta_t_is_fwhm: bool = True,
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
    reference_grid_ev : np.ndarray, optional
        Reference energy grid (e.g. original ENDF grid) in eV, used as
        fallback when the primary grid has no neighbour beyond the range.

    Returns
    -------
    List[EnergyBinInfo]
        List of energy bin info objects with computed σE and bin boundaries
    """
    logger = _get_logger()

    energies_mev = energies_ev / 1e6  # Convert to MeV
    ref_grid_mev = reference_grid_ev / 1e6 if reference_grid_ev is not None else None

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
            delta_t_is_fwhm=delta_t_is_fwhm,
        )

        # Compute bin boundaries (midpoints to neighbors)
        # Lower boundary
        if local_idx == 0:
            # First bin in range: use midpoint to previous grid point if available
            if global_idx > 0:
                bin_lower = (energies_mev[global_idx - 1] + e_mev) / 2.0
            elif ref_grid_mev is not None:
                below = ref_grid_mev[ref_grid_mev < e_mev]
                if len(below) > 0:
                    bin_lower = (below[-1] + e_mev) / 2.0
                elif n_bins > 1:
                    next_global_idx = indices_in_range[1]
                    half_width = (energies_mev[next_global_idx] - e_mev) / 2.0
                    bin_lower = max(0.0, e_mev - half_width)
                else:
                    bin_lower = 0.0
            else:
                if n_bins > 1:
                    next_global_idx = indices_in_range[1]
                    half_width = (energies_mev[next_global_idx] - e_mev) / 2.0
                    bin_lower = max(0.0, e_mev - half_width)
                else:
                    bin_lower = 0.0
        else:
            prev_global_idx = indices_in_range[local_idx - 1]
            bin_lower = (energies_mev[prev_global_idx] + e_mev) / 2.0

        # Upper boundary
        if local_idx == n_bins - 1:
            # Last bin in range: use midpoint to next grid point if available
            if global_idx < len(energies_mev) - 1:
                bin_upper = (e_mev + energies_mev[global_idx + 1]) / 2.0
            elif ref_grid_mev is not None:
                above = ref_grid_mev[ref_grid_mev > e_mev]
                if len(above) > 0:
                    bin_upper = (e_mev + above[0]) / 2.0
                else:
                    bin_upper = e_mev + (e_mev - bin_lower)
            else:
                bin_upper = e_mev + (e_mev - bin_lower)
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


def build_union_energy_grid(
    exfor_cache: Dict[float, List[Tuple[pd.DataFrame, Dict]]],
    subentries: List[Tuple[str, Optional[float], Optional[float]]],
    energy_min_mev: float,
    energy_max_mev: float,
) -> np.ndarray:
    """
    Build a union energy grid from selected EXFOR subentries.

    Each subentry carries its own energy range so that different experiments
    can cover different parts of the spectrum without unwanted overlap.

    Parameters
    ----------
    exfor_cache : Dict[float, List[Tuple[pd.DataFrame, Dict]]]
        EXFOR data cache mapping energy (MeV) to list of (DataFrame, metadata)
    subentries : List[Tuple[str, Optional[float], Optional[float]]]
        Each element is (subentry_id, min_MeV, max_MeV).
        None means use the global energy_min_mev / energy_max_mev.
    energy_min_mev : float
        Global minimum energy (MeV) — always included as endpoint
    energy_max_mev : float
        Global maximum energy (MeV) — always included as endpoint

    Returns
    -------
    np.ndarray
        Sorted union energy grid in eV
    """
    logger = _get_logger()

    # Build per-subentry energy ranges: normalized_id → (lo, hi)
    sub_ranges: Dict[str, Tuple[float, float]] = {}
    for sub_id, lo, hi in subentries:
        normed = sub_id.replace("/", "")
        sub_ranges[normed] = (
            lo if lo is not None else energy_min_mev,
            hi if hi is not None else energy_max_mev,
        )

    # Collect energies with per-subentry range filtering
    union_energies = set()
    matched_subentries = set()

    for energy_mev, entries in exfor_cache.items():
        for _df, meta in entries:
            sid = f"{meta['entry']}{meta['subentry']}"
            if sid in sub_ranges:
                lo, hi = sub_ranges[sid]
                if lo <= energy_mev <= hi:
                    union_energies.add(energy_mev)
                    matched_subentries.add(sid)

    # Always include global endpoints
    union_energies.add(energy_min_mev)
    union_energies.add(energy_max_mev)

    # Sort and convert to eV
    sorted_mev = np.array(sorted(union_energies))
    grid_ev = sorted_mev * 1e6

    if logger:
        logger.info(f"  Union grid: {len(grid_ev)} points from subentries {sorted(matched_subentries)}")
        if matched_subentries != set(sub_ranges):
            missing = set(sub_ranges) - matched_subentries
            logger.warning(f"  Subentries not found in cache: {sorted(missing)}")

    return grid_ev


def remap_samples_to_endf_indices(
    all_samples: Dict[int, Dict[int, np.ndarray]],
    idx_to_endf: Dict[int, int],
) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Remap all_samples keys from local energy_index to ENDF grid index.

    Parameters
    ----------
    all_samples : Dict[int, Dict[int, np.ndarray]]
        {sample_idx: {energy_index: coefficients}}
    idx_to_endf : Dict[int, int]
        Mapping from energy_index to endf_index

    Returns
    -------
    Dict[int, Dict[int, np.ndarray]]
        Remapped samples with ENDF indices as keys
    """
    remapped = {}
    for s_idx, sample_dict in all_samples.items():
        new_dict = {}
        for e_idx, coeffs in sample_dict.items():
            endf_idx = idx_to_endf.get(e_idx, e_idx)
            new_dict[endf_idx] = coeffs
        remapped[s_idx] = new_dict
    return remapped


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

    # Reset manifest-application diagnostics for this call.
    global _last_manifest_stats
    _last_manifest_stats = {'attempted': 0, 'failed': 0, 'failures': []}

    # Apply the uncertainty manifest at the pipeline boundary. The kika
    # library returns raw ExforAngularDistribution objects; here we layer the
    # manifest-derived per-point σ_stat (with optional decomposition from a
    # total) and the per-experiment σ_sys (split into indep and dep parts).
    # The manifest is mandatory: silently disabling it would let bad imports
    # revert the run to raw EXFOR uncertainties, which directly affect GLS
    # weights, AICc, τ, MC perturbations, and covariance.
    _import_errors: List[ImportError] = []
    try:
        from scripts.uncertainty_manifest import apply_manifest_to_exfor
    except ImportError as e:
        _import_errors.append(e)
        try:
            from uncertainty_manifest import apply_manifest_to_exfor  # in-tree fallback
        except ImportError as e2:
            _import_errors.append(e2)
            raise ImportError(
                "Could not import apply_manifest_to_exfor from "
                "scripts.uncertainty_manifest or uncertainty_manifest. "
                "The uncertainty manifest is mandatory for build_exfor_cache_from_objects. "
                f"Underlying errors: {[str(e) for e in _import_errors]}"
            )

    logger = _get_logger()
    for exfor in exfor_objects:
        # Check if experiment is excluded
        if _is_experiment_excluded(exfor.entry, exfor.subentry, exclusion_patterns):
            continue

        _last_manifest_stats['attempted'] += 1
        try:
            apply_manifest_to_exfor(
                exfor,
                uncertainty_components=getattr(exfor, '_raw_uncertainty_components', None),
            )
        except Exception as e:
            _last_manifest_stats['failed'] += 1
            _last_manifest_stats['failures'].append(
                (exfor.entry, exfor.subentry, repr(e))
            )
            msg = (
                f"Manifest application failed for {exfor.entry}/{exfor.subentry}: "
                f"{type(e).__name__}: {e}"
            )
            if logger is not None:
                logger.warning(msg)
            else:
                warnings.warn(msg, RuntimeWarning)

        # Get all available energies in MeV
        energies_mev = exfor.energies(unit='MeV')

        # Per-experiment manifest-derived sys components (relative fractions):
        #   sigma_sys_indep_relative : energy-independent scalar (one per experiment)
        #   sigma_sys_scalar_relative: representative total (for diagnostics)
        # The energy-dependent portion is recovered per-row from the per-point
        # error_sys vs the indep scalar — see cache_df construction below.
        sigma_sys_relative = float(getattr(exfor, 'sigma_sys_scalar_relative', 0.0) or 0.0)
        sigma_sys_indep   = float(getattr(exfor, 'sigma_sys_indep_relative',   0.0) or 0.0)
        manifest_flag = getattr(exfor, 'uncertainty_manifest_flag', 'default')

        for energy_mev in energies_mev:
            # Get data at this energy using to_dataframe with decomposed uncertainties
            # so that error_stat carries only the per-point uncorrelated noise
            # (sigma_sys is tracked per-experiment in meta below).
            df = exfor.to_dataframe(
                energy=energy_mev,
                energy_unit='MeV',
                cross_section_unit='b/sr',
                angle_unit='deg',
                tolerance=1e-4,  # 0.1 keV tolerance
                decompose_uncertainty=True,
            )

            if df.empty:
                continue

            # Convert DataFrame to expected column names. error_stat is the
            # per-point uncorrelated noise (used for GLS weights / chi^2).
            # error_sys is per-point correlated systematic (kept for diagnostics).
            cache_df = pd.DataFrame({
                'angle': df['angle'].values,
                'dsig': df['value'].values,
                'error_stat': df['error_stat'].values,
                'error_sys':  df['error_sys'].values,
            })

            # Build metadata dict
            meta = {
                'entry': exfor.entry,
                'subentry': exfor.subentry,
                'angle_frame': exfor.angle_frame,
                'reaction': exfor.reaction.get('notation', ''),
                'citation': exfor.citation,
                'sigma_sys_relative':       sigma_sys_relative,    # representative total
                'sigma_sys_indep_relative': sigma_sys_indep,       # energy-independent
                'uncertainty_manifest_flag': manifest_flag,
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


def check_angular_quality(exfor_df, min_points: int, min_bands: int):
    """Check whether per-bin angular data meets the F/M/B coverage gate.

    Returns ``(passes: bool, reason: str)``. ``reason`` is empty when the
    check passes; otherwise it briefly describes why the gate failed.
    """
    n_pts = len(exfor_df)
    if n_pts < min_points:
        return False, f"n_pts={n_pts} < {min_points}"
    mu = exfor_df['mu'].to_numpy()
    has_F = bool(np.any(mu > 0.5))
    has_M = bool(np.any((mu >= -0.5) & (mu <= 0.5)))
    has_B = bool(np.any(mu < -0.5))
    n_bands = int(has_F) + int(has_M) + int(has_B)
    if n_bands < min_bands:
        bands_str = "/".join(b for b, h in [("F", has_F), ("M", has_M), ("B", has_B)] if h)
        return False, f"bands={bands_str} ({n_bands}/{min_bands})"
    return True, ""


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
    unc_floor_strategy: str = "bin_median",
    # Per-experiment weighting options
    normalize_by_n_points: bool = False,
    sigma_norm: float = 0.05,                     # Normalization uncertainty for GLS-ESS weighting
    band_aware_ess: bool = False,                 # Split Kish budget by F/M/B bands
    max_experiment_weight_fraction: float = 1.0,  # 1.0 = disabled
    # Membership window (see "Membership vs weighting" in the notes)
    membership_k_sigma: float = 0.0,              # 0.0 = hard bin edges (default)
    sigma_E_mev: Optional[float] = None,          # required when membership_k_sigma > 0
    logger=None,
) -> Tuple[pd.DataFrame, List[Dict], np.ndarray, KernelDiagnostics, Dict]:
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
    membership_k_sigma : float, optional
        Widens the window that decides WHICH datasets may constrain this bin,
        to ``target_energy_mev +- membership_k_sigma * sigma_E_mev`` (unioned
        with the bin edges, so data is never lost). Default 0.0 keeps the hard
        bin edges.

        This is deliberately a membership knob and not a weighting knob. The
        analysis grid here is ~5x finer than any experiment's TOF resolution,
        so a bin-width window renews almost the entire point set from one bin
        to the next; widening it to the resolution scale makes the composition
        vary slowly. The selected point per experiment is still the one nearest
        the target and still carries weight 1.0 — no Gaussian overlap weighting
        is applied. That distinction matters: every EXFOR datum is ALREADY
        folded by its own resolution, so weighting the fit by an overlap kernel
        of the same width would convolve a second time and hand back an
        effective resolution of sqrt(2)*sigma_E. Widening membership does not.
    sigma_E_mev : float, optional
        The bin's TOF energy resolution. Required when membership_k_sigma > 0;
        ignored otherwise.

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

    # Membership window. Defaults to the bin edges; when membership_k_sigma > 0
    # it is unioned with +-k*sigma_E about the target so an experiment whose
    # resolution spans many bins can constrain all of them. Only membership is
    # affected — dedupe still picks the point nearest target_energy_mev and the
    # weight stays 1.0.
    member_lower_mev, member_upper_mev = bin_lower_mev, bin_upper_mev
    if membership_k_sigma > 0 and sigma_E_mev and sigma_E_mev > 0:
        half_width = membership_k_sigma * sigma_E_mev
        member_lower_mev = min(bin_lower_mev, target_energy_mev - half_width)
        member_upper_mev = max(bin_upper_mev, target_energy_mev + half_width)

    for available_energy in sorted_energies:
        # Exact bin matching - include if within [lower, upper]
        if available_energy < member_lower_mev or available_energy > member_upper_mev:
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

    # Step 2b: Count total points per study for GLS-ESS weighting
    # Group by entry (study-level), not entry.subentry
    study_n_points_map: Dict[str, int] = {}
    if normalize_by_n_points:
        for available_energy, df, meta in selected_data:
            entry = meta.get('entry', 'unknown')
            n_pts = len(df['angle'])
            study_n_points_map[entry] = study_n_points_map.get(entry, 0) + n_pts

    # Also track kernel weights per row for final assembly
    all_kernel_weights: List[float] = []

    # Step 3: Process selected data (transform and build DataFrames)
    for available_energy, df, meta in selected_data:
        # Extract metadata (same as Gaussian kernel method)
        entry = meta.get('entry', 'unknown')
        subentry = meta.get('subentry', 'unknown')
        exp_key = (entry, subentry)

        n_points = len(df['angle'])

        # Placeholder weight — GLS-ESS weights are computed after the
        # uncertainty floor so that ρ_j uses post-floor σ_stat values.
        kernel_weight = 1.0

        # Extract metadata (same as Gaussian kernel method)
        frame = meta.get('angle_frame', 'CM').upper()
        reaction = meta.get('reaction', '')

        citation = meta.get('citation', {})
        authors = citation.get('authors', [])
        author = authors[0] if authors else 'unknown'
        year = citation.get('year', 'unknown')

        # Per-row systematic uncertainty (relative fraction). Derived from the
        # manifest's per-point sigma_sys (in cache_df['error_sys']) so that
        # energy-dependent sys (e.g. Kinney's gain_shift) is preserved point-
        # by-point. Constant-per-experiment systematics (e.g. Tomita's 5%) end
        # up as a constant column. Falls back to the per-experiment scalar
        # from meta when the cache_df doesn't carry error_sys (legacy paths).
        if 'error_sys' in df.columns:
            err_sys_arr = df['error_sys'].to_numpy(dtype=float)
            sigma_sys_relative_per_row = err_sys_arr / np.maximum(
                np.abs(df['dsig'].to_numpy(dtype=float)), 1e-30
            )
        else:
            scalar = float(meta.get('sigma_sys_relative', 0.0) or 0.0)
            sigma_sys_relative_per_row = np.full(len(df), scalar, dtype=float)

        # Energy-independent vs energy-dependent split. indep is a per-experiment
        # scalar (constant across points); dep is per-row, derived from total
        # minus indep in quadrature. For Kinney, indep = sqrt(7² + 4²) ≈ 8.06%
        # and dep varies with energy via gain_shift. For experiments with only
        # scalar sys (e.g. Tomita 5%), indep = total and dep = 0.
        sigma_sys_indep_scalar = float(meta.get('sigma_sys_indep_relative', 0.0) or 0.0)
        sigma_sys_indep_per_row = np.full(len(df), sigma_sys_indep_scalar, dtype=float)
        sigma_sys_dep_per_row = np.sqrt(
            np.maximum(sigma_sys_relative_per_row ** 2 - sigma_sys_indep_scalar ** 2, 0.0)
        )

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
                'sigma_sys_relative':       sigma_sys_relative_per_row,
                'sigma_sys_indep_relative': sigma_sys_indep_per_row,
                'sigma_sys_dep_relative':   sigma_sys_dep_per_row,
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
                'sigma_sys_relative':       sigma_sys_relative_per_row,
                'sigma_sys_indep_relative': sigma_sys_indep_per_row,
                'sigma_sys_dep_relative':   sigma_sys_dep_per_row,
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
        empty_floor = {'n_floored': 0, 'n_total': 0, 'replacement_rel_unc': 0.0, 'per_experiment': []}
        return pd.DataFrame(), [], np.array([]), empty_diag, empty_floor

    # Concatenate all experiments
    result = pd.concat(all_frames, ignore_index=True)
    kernel_weights = np.array(all_kernel_weights, dtype=float)

    # Apply uncertainty floor if requested
    floor_stats = {'n_floored': 0, 'n_total': 0, 'replacement_rel_unc': 0.0, 'per_experiment': []}
    if min_relative_uncertainty > 0:
        result, floor_stats = apply_uncertainty_floor(
            result, min_relative_uncertainty, unc_column='unc', value_column='value',
            strategy=unc_floor_strategy, logger=logger,
        )

    # Compute GLS-ESS per-point weights using post-floor uncertainties.
    # Group by (entry, subentry, exfor_energy_mev): each (experiment, energy)
    # cell is a distinct correlated unit because the manifest's sigma_sys may
    # vary with energy (e.g. Kinney gain_shift, Cierjacks piecewise). Within
    # a cell sigma_sys is constant; ρ is computed once per cell.
    if normalize_by_n_points and sigma_norm > 0 and len(result) > 0:
        group_cols = ['entry', 'subentry']
        if 'exfor_energy_mev' in result.columns:
            group_cols.append('exfor_energy_mev')
        unique_groups = result[group_cols].drop_duplicates()
        for _, row in unique_groups.iterrows():
            mask = np.ones(len(result), dtype=bool)
            for c in group_cols:
                mask &= (result[c] == row[c]).values
            vals = result.loc[mask, 'value'].to_numpy()
            uncs = result.loc[mask, 'unc'].to_numpy()
            rel_arr = uncs / np.maximum(np.abs(vals), 1e-30)
            finite = np.isfinite(rel_arr)
            rel_unc = np.sqrt(np.nanmean(rel_arr[finite]**2)) if finite.any() else 0.0
            if not np.isfinite(rel_unc) or rel_unc <= 0:
                rel_unc = min_relative_uncertainty if min_relative_uncertainty > 0 else sigma_norm
            # Per-cell sigma_sys (constant within (entry, subentry, energy);
            # may vary across energies of the same experiment).
            ex_sigma_sys = 0.0
            if 'sigma_sys_relative' in result.columns:
                vals_sys = result.loc[mask, 'sigma_sys_relative'].to_numpy()
                if vals_sys.size > 0:
                    ex_sigma_sys = float(vals_sys[0])
            sigma_eff = ex_sigma_sys if ex_sigma_sys > 0 else sigma_norm
            rho = sigma_eff**2 / (sigma_eff**2 + rel_unc**2)
            if band_aware_ess:
                # Per-band ESS: a forward-only fragment shouldn't be collapsed
                # against forward+mid+backward points it doesn't constrain.
                mu_arr = result.loc[mask, 'mu'].to_numpy()
                band_masks = {
                    'F': mu_arr > 0.5,
                    'M': (mu_arr >= -0.5) & (mu_arr <= 0.5),
                    'B': mu_arr < -0.5,
                }
                g_per_point = np.empty(int(mask.sum()), dtype=float)
                for bmask in band_masks.values():
                    n_band = int(bmask.sum())
                    if n_band == 0:
                        continue
                    g_band = 1.0 / (1.0 + max(n_band - 1, 0) * rho)
                    g_per_point[bmask] = g_band
                kernel_weights[mask] = g_per_point
                result.loc[mask, 'kernel_weight'] = g_per_point
            else:
                n_j = int(mask.sum())
                g = 1.0 / (1.0 + max(n_j - 1, 0) * rho)
                kernel_weights[mask] = g
                result.loc[mask, 'kernel_weight'] = g
        # Update experiments_info with new weights (match by entry AND subentry)
        for exp in experiments_info:
            emask = (result['entry'] == exp['entry']) & (result['subentry'] == exp['subentry'])
            if emask.any():
                exp['kernel_weight'] = float(kernel_weights[emask.values][0])

    # Apply per-experiment weight capping if requested (Improvement 1.1)
    capping_applied = False
    if max_experiment_weight_fraction < 1.0:
        kernel_weights, exp_weight_fracs, capping_applied = apply_per_experiment_weight_cap(
            result, kernel_weights, max_experiment_weight_fraction
        )
        # Propagate capped weights to DataFrame for downstream consistency
        # (e.g. compute_between_experiment_coeffs reads grp['kernel_weight'])
        result['kernel_weight'] = kernel_weights
        for exp in experiments_info:
            mask = (result['entry'] == exp['entry']) & (result['subentry'] == exp['subentry'])
            if mask.any():
                exp['kernel_weight'] = float(kernel_weights[mask.values][0])
    else:
        # Compute weight fractions per (entry, subentry) for logging
        exp_weight_fracs = {}
        total_weight = float(np.sum(kernel_weights))
        if total_weight > 1e-30:
            seen_keys = set()
            for exp in experiments_info:
                es_key = f"{exp['entry']}.{exp['subentry']}"
                if es_key in seen_keys:
                    continue
                seen_keys.add(es_key)
                exp_mask = (result['entry'] == exp['entry']) & (result['subentry'] == exp['subentry'])
                exp_total = float(np.sum(kernel_weights[exp_mask.values]))
                exp_weight_fracs[es_key] = exp_total / total_weight

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

    return result, experiments_info, kernel_weights, diagnostics, floor_stats


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
    strategy: str = "bin_median",
    logger=None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply minimum relative uncertainty floor to prevent experiments with
    unrealistically small uncertainties from dominating fits.

    Three strategies:
    - 'fixed': enforces unc >= min_relative_uncertainty * |value| (simple floor)
    - 'bin_median': replaces unreliable uncertainties with the median relative
      uncertainty of trustworthy points in the bin (data-driven)
    - 'band_median': per-angular-band median (F: μ>0.5, M: |μ|≤0.5, B: μ<-0.5)
      with fallback to bin-median when fewer than 3 trustworthy in-band points,
      and to the fixed floor when bin-median has fewer than 3 trustworthy points
      either. Honors the angle dependence of relative uncertainties in AD data.

    Parameters
    ----------
    exfor_df : pd.DataFrame
        EXFOR data with uncertainty, value, entry, and author columns. Requires
        a 'mu' column when strategy='band_median'.
    min_relative_uncertainty : float
        Threshold below which uncertainties are considered unreliable.
    unc_column : str
        Column name for uncertainties (default: 'unc')
    value_column : str
        Column name for cross section values (default: 'value')
    strategy : str
        'fixed', 'bin_median', or 'band_median' (default: 'bin_median')
    logger : optional
        Logger for diagnostic output

    Returns
    -------
    (pd.DataFrame, dict)
        Copy of DataFrame with updated uncertainties, and stats dict with
        keys: n_floored, n_total, replacement_rel_unc, per_experiment (list).
        For strategy='band_median' the dict also includes per-band replacements
        in 'replacement_rel_unc_band' (dict with 'F'/'M'/'B' keys).
    """
    empty_stats = {'n_floored': 0, 'n_total': 0, 'replacement_rel_unc': 0.0, 'per_experiment': []}

    if min_relative_uncertainty <= 0:
        return exfor_df, empty_stats

    df = exfor_df.copy()
    if unc_column not in df.columns or value_column not in df.columns:
        return df, empty_stats

    abs_val = np.abs(df[value_column].values).astype(float)
    abs_val_safe = np.maximum(abs_val, 1e-30)
    rel_unc = df[unc_column].values.astype(float) / abs_val_safe
    n_total = len(df)

    # Identify points below threshold
    below = rel_unc < min_relative_uncertainty
    n_floored = int(np.sum(below))

    if n_floored == 0:
        return df, {'n_floored': 0, 'n_total': n_total,
                     'replacement_rel_unc': 0.0, 'per_experiment': []}

    trustworthy = ~below
    n_trustworthy = int(np.sum(trustworthy))

    def _bin_median_replacement() -> float:
        if n_trustworthy >= 3:
            r = float(np.nanmedian(rel_unc[trustworthy]))
            if np.isfinite(r) and r > 0:
                return r
        return float(min_relative_uncertainty)

    # Determine replacement value(s)
    replacement_rel_band: Dict[str, float] = {}
    if strategy == 'band_median':
        if 'mu' not in df.columns:
            if logger:
                logger.warning("    [Unc floor] strategy='band_median' but no 'mu' column; "
                               "falling back to bin_median")
            strategy = 'bin_median'

    if strategy == 'band_median':
        mu_arr = df['mu'].to_numpy(dtype=float)
        bin_replacement = _bin_median_replacement()
        band_masks_local = {
            'F': mu_arr > 0.5,
            'M': (mu_arr >= -0.5) & (mu_arr <= 0.5),
            'B': mu_arr < -0.5,
        }
        for bn, bm in band_masks_local.items():
            in_band_trust = bm & trustworthy
            n_in_band = int(np.sum(in_band_trust))
            if n_in_band >= 3:
                r_band = float(np.nanmedian(rel_unc[in_band_trust]))
                if not np.isfinite(r_band) or r_band <= 0:
                    r_band = bin_replacement
            else:
                r_band = bin_replacement
                if logger:
                    logger.debug(f"    [Unc floor] band {bn}: <3 trustworthy pts "
                                 f"({n_in_band}); falling back to bin median "
                                 f"{bin_replacement*100:.1f}%")
            replacement_rel_band[bn] = r_band
        # For the summary scalar, report the weighted average over floored points
        n_floored_per_band = {
            bn: int(np.sum(below & bm)) for bn, bm in band_masks_local.items()
        }
        weighted_sum = sum(replacement_rel_band[bn] * n_floored_per_band[bn]
                           for bn in 'FMB')
        replacement_rel = weighted_sum / max(n_floored, 1)
    elif strategy == 'bin_median':
        replacement_rel = _bin_median_replacement()
        if replacement_rel == min_relative_uncertainty and n_trustworthy < 3 and logger:
            logger.debug(f"    [Unc floor] <3 trustworthy pts ({n_trustworthy}), "
                         f"falling back to fixed floor {min_relative_uncertainty*100:.1f}%")
    else:
        replacement_rel = float(min_relative_uncertainty)

    # Apply replacement
    new_unc = df[unc_column].values.astype(float).copy()
    if strategy == 'band_median':
        mu_arr = df['mu'].to_numpy(dtype=float)
        for bn, r_band in replacement_rel_band.items():
            bm = {
                'F': mu_arr > 0.5,
                'M': (mu_arr >= -0.5) & (mu_arr <= 0.5),
                'B': mu_arr < -0.5,
            }[bn]
            apply_mask = below & bm
            new_unc[apply_mask] = (r_band * abs_val)[apply_mask]
    else:
        replacement_abs = replacement_rel * abs_val
        new_unc[below] = replacement_abs[below]
    df[unc_column] = new_unc

    # Per-experiment diagnostics
    per_exp_stats = []
    if 'entry' in df.columns:
        entries = df['entry'].values
        unique_entries = np.unique(entries)
        for ent in unique_entries:
            mask_ent = entries == ent
            n_pts_ent = int(np.sum(mask_ent))
            n_floored_ent = int(np.sum(below & mask_ent))
            orig_mean_rel = float(np.mean(rel_unc[mask_ent])) * 100
            author = ''
            if 'author' in df.columns:
                author = df.loc[mask_ent, 'author'].iloc[0]
            per_exp_stats.append({
                'entry': str(ent), 'author': author,
                'n_pts': n_pts_ent, 'n_floored': n_floored_ent,
                'orig_mean_rel_pct': orig_mean_rel,
            })

    stats = {
        'n_floored': n_floored,
        'n_total': n_total,
        'replacement_rel_unc': replacement_rel,
        'per_experiment': per_exp_stats,
    }
    if replacement_rel_band:
        stats['replacement_rel_unc_band'] = replacement_rel_band

    # Log per-bin detail
    if logger and n_floored > 0:
        energy_str = ''
        if 'exfor_energy_mev' in df.columns:
            energy_str = f" E~{df['exfor_energy_mev'].iloc[0]:.4f} MeV:"
        logger.info(f"    [Unc floor]{energy_str} {strategy} replacement="
                    f"{replacement_rel*100:.1f}%, {n_floored}/{n_total} pts floored")
        for es in per_exp_stats:
            if es['n_floored'] > 0:
                logger.info(f"      {es['author']} ({es['entry']}): "
                            f"{es['n_floored']}/{es['n_pts']} pts floored "
                            f"(orig mean={es['orig_mean_rel_pct']:.1f}% → {replacement_rel*100:.1f}%)")

    return df, stats


def apply_per_experiment_weight_cap(
    exfor_df: pd.DataFrame,
    kernel_weights: np.ndarray,
    max_experiment_weight_fraction: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, float], bool]:
    """
    Cap per-study WLS effective weight to prevent experiments with small
    uncertainties from dominating the fit.

    The WLS effective weight per point is kernel_weight_i / sigma_i^2.  The cap
    is applied to the WLS effective fraction per study (grouped by entry.subentry).
    When a study exceeds the cap, its kernel weights are scaled down so the
    combined WLS fraction respects the limit.

    Parameters
    ----------
    exfor_df : pd.DataFrame
        EXFOR data with 'entry', 'subentry', and 'unc' columns.
    kernel_weights : np.ndarray
        Kernel weights (one per data point).
    max_experiment_weight_fraction : float
        Maximum allowed WLS effective fraction per study (default: 0.5).

    Returns
    -------
    Tuple[np.ndarray, Dict[str, float], bool]
        - capped_weights: Adjusted kernel weights
        - experiment_weight_fracs: {study_key: wls_fraction} AFTER capping (post-cap if capping was applied)
        - capping_applied: Whether any capping was done
    """
    if max_experiment_weight_fraction >= 1.0:
        return kernel_weights.copy(), {}, False

    if len(kernel_weights) == 0:
        return kernel_weights.copy(), {}, False

    # Build study key for each point (group by entry.subentry)
    entries = exfor_df['entry'].values
    subentries = exfor_df['subentry'].values
    n_points = len(kernel_weights)
    exp_keys = np.array([f"{entries[i]}.{subentries[i]}" for i in range(n_points)])

    sigma = exfor_df['unc'].values.astype(float)
    # Guard against zero/tiny sigma
    sigma = np.maximum(sigma, 1e-30)

    # WLS effective weight: kernel_weight / sigma^2
    wls_eff = kernel_weights / (sigma ** 2)
    total_wls = wls_eff.sum()
    if total_wls < 1e-30:
        return kernel_weights.copy(), {}, False

    # Compute WLS fractions per study BEFORE capping (for diagnostics)
    unique_studies = list(dict.fromkeys(exp_keys))
    exp_weight_fracs: Dict[str, float] = {}
    for study in unique_studies:
        mask = exp_keys == study
        exp_weight_fracs[study] = float(wls_eff[mask].sum() / total_wls)

    # Edge case: only one study - cannot cap
    if len(unique_studies) == 1:
        return kernel_weights.copy(), exp_weight_fracs, False

    # Iterative capping: scale kernel weights so WLS fraction <= cap
    capped_weights = kernel_weights.copy()
    capping_applied = False
    cap = max_experiment_weight_fraction

    for _ in range(5):
        changed = False
        cur_wls = capped_weights / (sigma ** 2)
        cur_total = cur_wls.sum()
        if cur_total < 1e-30:
            break
        for study in unique_studies:
            mask = exp_keys == study
            frac = cur_wls[mask].sum() / cur_total
            if frac > cap:
                scale = cap / frac
                capped_weights[mask] *= scale
                capping_applied = True
                changed = True
        if not changed:
            break

    # Recompute fractions post-cap for accurate diagnostics
    if capping_applied:
        cur_wls = capped_weights / (sigma ** 2)
        cur_total = cur_wls.sum()
        if cur_total > 1e-30:
            exp_weight_fracs = {
                study: float(cur_wls[exp_keys == study].sum() / cur_total)
                for study in unique_studies
            }

    return capped_weights, exp_weight_fracs, capping_applied


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
    default_delta_t_is_fwhm: bool = True,
    logger=None,
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
    # Which subentries resolved to which TOF convention, for the audit below.
    _conv_seen: Dict[str, Any] = {}
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
                    default_delta_t_is_fwhm=default_delta_t_is_fwhm,
                )
                ds_sigma_E = compute_sigma_E(exfor_energy, tof_params)
                ds_tof_source = tof_params.source
                _conv_seen.setdefault(subentry_id, tof_params)
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
    # raw CDF overlap weight — so the fit benefits from the full angular
    # coverage and cross-bin bridging.  The GLS-ESS per-point weighting
    # in _run_one_kw_sample handles study-level budgeting.
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

        overlap_weights[bin_info.index] = bin_datasets

    # Audit the TOF convention actually applied per subentry. sigma_E scales by
    # ~2.355 between the FWHM and sigma readings, and it decides how far each
    # experimental point spreads across bins — so a silent fallback to the
    # pipeline default is worth naming rather than assuming.
    if logger is not None and _conv_seen:
        from_file = sorted(
            s for s, p in _conv_seen.items() if p.source == "file"
        )
        defaulted = sorted(
            s for s, p in _conv_seen.items() if p.source != "file"
        )
        conv = "FWHM" if default_delta_t_is_fwhm else "sigma"
        logger.info(
            f"  TOF convention: delta_t read as {conv} by default; "
            f"{len(from_file)} subentry(ies) had file parameters, "
            f"{len(defaulted)} fell back to L={default_flight_path_m} m, "
            f"dt={default_time_resolution_ns} ns"
        )
        if defaulted:
            logger.warning(
                f"  [TOF] No per-experiment parameters for: "
                f"{', '.join(defaulted[:12])}"
                f"{' ...' if len(defaulted) > 12 else ''} — these inherit the "
                f"global delta_t and its {conv} reading."
            )

    return overlap_weights


# -- Worker-shared state for Pool initializer (avoids re-pickling per sample) --
_kw_shared: Optional[dict] = None


def _init_kw_worker(shared: dict) -> None:
    """Pool initializer: store shared data in module-level global."""
    global _kw_shared
    _kw_shared = shared


def _run_one_kw_sample(args_tuple):
    """Single kernel-weighted multi-bin MC sample (top-level for Pool.map).

    1. Draw shared normalization factor per experiment (once per sample)
    2. Perturb all datasets (apply norm + pointwise noise)
    3. For each bin: collect datasets with overlap weight, build weighted DataFrame
    4. Fit Legendre coefficients
    5. Return coefficients for all bins

    The SAME perturbed dataset is used for ALL bins it contributes to,
    creating cross-bin correlations.

    Accepts either:
      - a single int (sample index) when shared data was set via _init_kw_worker
      - a full args tuple (legacy path, sequential fallback)
    """
    if isinstance(args_tuple, int):
        # Fast path: shared data lives in _kw_shared (set by Pool initializer)
        s_idx = args_tuple
        sh = _kw_shared
        overlap_weights = sh['overlap_weights']
        energy_bins_data = sh['energy_bins_data']
        sigma_norm = sh['sigma_norm']
        sigma_norm_common_mode = sh.get('sigma_norm_common_mode', 0.0)
        norm_dist = sh['norm_dist']
        max_degree = sh['max_degree']
        ridge_lambda = sh['ridge_lambda']
        base_seed = sh['base_seed']
        use_band_discrepancy = sh['use_band_discrepancy']
        min_points_per_band = sh['min_points_per_band']
        max_band_scale = sh['max_band_scale']
        freeze_c0 = sh['freeze_c0']
        fix_c0_at_nominal = sh.get('fix_c0_at_nominal', False)
        sys_aware_mc_fit = sh.get('sys_aware_mc_fit', False)
        max_sample_order = sh['max_sample_order']
        apply_positivity_projection = sh['apply_positivity_projection']
        positivity_check_points = sh['positivity_check_points']
        nominal_coeffs_by_bin = sh['nominal_coeffs_by_bin']
        frozen_degrees_by_bin = sh['frozen_degrees_by_bin']
        max_experiment_weight_fraction = sh['max_experiment_weight_fraction']
        min_relative_uncertainty = sh['min_relative_uncertainty']
        tau_info_by_bin = sh['tau_info_by_bin']
        mc_order_cap_by_bin = sh['mc_order_cap_by_bin']
        band_aware_ess = sh.get('band_aware_ess', False)
        record_c0_channel = sh.get('record_c0_channel', False)
    else:
        # Legacy path: full args tuple (sequential mode or old callers)
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
            max_band_scale,
            freeze_c0,
            max_sample_order,
            apply_positivity_projection,
            positivity_check_points,
            nominal_coeffs_by_bin,
            frozen_degrees_by_bin,
            max_experiment_weight_fraction,
            min_relative_uncertainty,
            tau_info_by_bin,
            mc_order_cap_by_bin,
        ) = args_tuple
        sigma_norm_common_mode = 0.0
        band_aware_ess = False
        fix_c0_at_nominal = False
        sys_aware_mc_fit = False
        record_c0_channel = False

    rng = np.random.default_rng(base_seed + s_idx)

    # Step 0: Draw one global common-mode normalization factor per MC sample.
    # Models uncertainty in the shared reference / monitor that all
    # experiments rely on. Applied to every data point regardless of which
    # experiment produced it; introduces a common-mode correlation across
    # all bins and all entries for this sample.
    if sigma_norm_common_mode > 0:
        if norm_dist == "lognormal":
            common_mode_factor = float(rng.lognormal(
                mean=-0.5 * sigma_norm_common_mode ** 2, sigma=sigma_norm_common_mode
            ))
        else:
            common_mode_factor = float(rng.normal(1.0, sigma_norm_common_mode))
    else:
        common_mode_factor = 1.0

    # Step 1: Draw ONE shared standard-normal `z` per experiment per sample.
    # The same z is applied to every point of every dataset that experiment
    # contributes to (across all energies, all subentries, all bins) — so the
    # direction of the systematic shift is correlated across the experiment's
    # full energy range, with the per-point AMPLITUDE coming from the manifest.
    #
    # Per-point factor (lognormal): exp(z*sigma_i - 0.5*sigma_i^2)
    # Per-point factor (normal):    1 + z*sigma_i
    #
    # Marginally each point sees a Lognormal/Normal multiplier with parameter
    # sigma_i (so per-point variance reproduces the manifest); the shared z
    # makes any two points of the same experiment perfectly correlated, which
    # is what produces long-range covariance between bins fed by the same
    # experiment. Cross-experiment shifts are independent (different z).
    #
    # The per-point sigma comes from `df['sigma_sys_relative']` (manifest-
    # derived total sys, computed as |error_sys|/|value| at DataFrame build
    # time). For Cierjacks band B this is 0.07 at every point; for Kinney it
    # is 0.0806 everywhere; for uncurated experiments it's the manifest
    # default 5%. Falls back to the global `sigma_norm` for legacy callers
    # whose DataFrame lacks the column.
    entry_z_norms: Dict[str, float] = {}
    for bin_idx, datasets_and_weights in overlap_weights.items():
        for ds, w in datasets_and_weights:
            entry_id = ds['experiment_id'].split('.')[0]
            if entry_id not in entry_z_norms:
                entry_z_norms[entry_id] = float(rng.standard_normal())

    # Step 2: Perturb all datasets (shared across bins)
    # Build per-dataset tau from the bin with highest overlap weight so that
    # the noise amplitude matches the band-inflated sigma_eff used in Pass 1.
    dataset_primary_tau: Dict[str, Dict[str, float]] = {}
    if use_band_discrepancy and tau_info_by_bin:
        _best_w: Dict[str, float] = {}
        for bin_idx, datasets_and_weights in overlap_weights.items():
            bin_tau = tau_info_by_bin.get(bin_idx)
            if bin_tau is None:
                continue
            for ds, w in datasets_and_weights:
                e_key = f"{ds['experiment_id']}_{ds['exfor_energy_mev']:.6f}"
                if e_key not in _best_w or w > _best_w[e_key]:
                    _best_w[e_key] = w
                    dataset_primary_tau[e_key] = bin_tau

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
            entry_id = exp_id.split('.')[0]
            z = entry_z_norms.get(entry_id, 0.0)
            # Per-point sys magnitude (relative). Already aggregated across
            # all manifest components (e.g. Kinney's monitor⊕geometry → 8.06%;
            # Cierjacks's piecewise → 7% in band B; etc). When the column is
            # missing (legacy DataFrames), fall back to the global `sigma_norm`.
            if 'sigma_sys_relative' in df.columns:
                sigma_per_pt = df['sigma_sys_relative'].to_numpy(dtype=float)
            else:
                sigma_per_pt = np.full(len(df), sigma_norm, dtype=float)
            # Apply per-point shared-z factor: same direction, per-point amplitude.
            if norm_dist == "lognormal":
                norm_per_pt = np.exp(z * sigma_per_pt - 0.5 * sigma_per_pt ** 2)
            else:
                norm_per_pt = 1.0 + z * sigma_per_pt
            # Compose elastic (global, all experiments) with per-experiment
            # shared-direction per-point factor.
            values = df['value'].to_numpy() * (common_mode_factor * norm_per_pt)
            # Optional MF33 multiplicative factor (v3 hook). When the caller
            # passes mf33_dsigma_per_sample + mf33_c0_per_bin + a home-bin map,
            # apply (1 + δσ/c0_home) to this dataset on top of the per-experiment
            # systematic shifts. The δσ vector is shared across all datasets
            # within a sample (drawn once from MF33's full covariance), so
            # bins inherit MF33-driven cross-bin correlation through the
            # per-dataset home-bin lookup. Default-off → exact v2 behavior.
            #
            # WARNING (MF33/MF34 roadmap, Phase 1): do NOT repurpose this hook to
            # derive an MF33↔MF34 cross-covariance. It injects an *assumed* MF33
            # draw into the DCS while c0 is pinned (fix_c0_at_nominal), so the
            # resulting a_l = c_l/c0 correlation is manufactured from fit
            # residuals, not measured. A genuine sigma↔a_l cross block must be
            # estimated from a joint fit (roadmap Phase 3), never from this
            # convenience factor. See kika-workspace/docs/mf3_mf33_roadmap.md.
            mf33_dsigma_per_sample = sh.get('mf33_dsigma_per_sample') if isinstance(args_tuple, int) else None
            if mf33_dsigma_per_sample is not None:
                home_map = sh['mf33_home_bin_by_e_key']
                c0_per_bin = sh['mf33_c0_per_bin']
                home_bin = home_map.get(e_key)
                if home_bin is not None and c0_per_bin[home_bin] > 0:
                    values = values * (1.0 + mf33_dsigma_per_sample[s_idx, home_bin] / c0_per_bin[home_bin])
            unc = df['unc'].to_numpy()
            # Inflate noise amplitude with band scale factors (consistent
            # with Pass 1 which draws noise from sigma_eff, not raw unc)
            if e_key in dataset_primary_tau:
                from .resample_AD import sigma_eff_from_tau
                noise_sigma = sigma_eff_from_tau(
                    df['mu'].to_numpy(), unc, dataset_primary_tau[e_key],
                )
            else:
                noise_sigma = unc
            noise = rng.normal(0, noise_sigma)
            # σ_sys in absolute units, scaled to the perturbed value so the
            # downstream fit sees σ_total² = (τ·σ_stat)² + σ_sys² as marginal
            # point variance. Matches Pass 2 / nominal fit weights.
            if 'sigma_sys_relative' in df.columns:
                sys_rel = df['sigma_sys_relative'].to_numpy(dtype=float)
                sigma_sys_abs = sys_rel * np.abs(values + noise)
            else:
                sigma_sys_abs = np.zeros_like(unc)
            perturbed_datasets[e_key] = {
                'mu': df['mu'].to_numpy(),
                'value': values + noise,
                'unc': unc,
                'sigma_sys_abs': sigma_sys_abs,
            }

    # Step 3-4: For each bin, collect perturbed data, fit
    sample_coeffs = {}
    # Phase-2 magnitude channel: fixed-shape c0 per bin for this sample. Stays
    # empty (and is dropped from the return) unless recording is on.
    sample_c0 = {} if record_c0_channel else None
    # Phase D audit follow-up: count bins where the per-bin coeffs needed a
    # positivity projection (only when projection is enabled). Surfaced via the
    # worker's return tuple so the orchestrator can aggregate across samples.
    n_pos_violations = 0
    for bin_info_data in energy_bins_data:
        bin_idx = bin_info_data['index']
        datasets_and_weights = overlap_weights.get(bin_idx, [])

        if not datasets_and_weights:
            # Use nominal (interpolated)
            nom = nominal_coeffs_by_bin.get(bin_idx)
            if nom is not None:
                sample_coeffs[bin_idx] = endf_normalize_legendre_coeffs(nom, include_a0=False)
            continue

        # Build combined weighted DataFrame with study-level GLS-ESS budgeting
        # Find core dataset per study (entry.subentry): highest overlap weight in this bin.
        # Use only its angular point count for GLS-ESS, matching the nominal
        # path's dedup-to-one-energy behavior.
        study_core_n: Dict[str, int] = {}
        study_core_w: Dict[str, float] = {}
        # When band-aware ESS is on, also remember the core's per-band point counts
        # so each evaluation point's Kish collapse uses only the in-band count.
        study_core_n_by_band: Dict[str, Dict[str, int]] = {}
        for ds, w in datasets_and_weights:
            study_id = ds['experiment_id']  # full entry.subentry
            e_key = f"{ds['experiment_id']}_{ds['exfor_energy_mev']:.6f}"
            pert = perturbed_datasets.get(e_key)
            if pert is not None:
                if study_id not in study_core_w or w > study_core_w[study_id]:
                    study_core_n[study_id] = len(pert['mu'])
                    study_core_w[study_id] = w
                    if band_aware_ess:
                        mu_core = pert['mu']
                        study_core_n_by_band[study_id] = {
                            'F': int(np.sum(mu_core > 0.5)),
                            'M': int(np.sum((mu_core >= -0.5) & (mu_core <= 0.5))),
                            'B': int(np.sum(mu_core < -0.5)),
                        }

        all_mu = []
        all_values = []
        all_unc = []
        all_sigma_sys_abs = []
        all_weights = []

        for ds, w in datasets_and_weights:
            exp_id = ds['experiment_id']
            study_id = exp_id  # full entry.subentry
            e_key = f"{exp_id}_{ds['exfor_energy_mev']:.6f}"
            pert = perturbed_datasets.get(e_key)
            if pert is None:
                continue
            n_pts = len(pert['mu'])
            all_mu.append(pert['mu'])
            all_values.append(pert['value'])
            all_unc.append(pert['unc'])
            all_sigma_sys_abs.append(
                pert.get('sigma_sys_abs', np.zeros_like(pert['unc']))
            )
            # GLS-ESS per-point weight using core dataset's post-floor uncertainties.
            # Kish ρ uses the per-(entry, energy) sigma_sys from the manifest;
            # constant within a (entry, energy) cell but allowed to vary across
            # energies of the same experiment (Kinney gain_shift case). Falls
            # back to the global sigma_norm if absent. The elastic factor is
            # common to all data and doesn't differentiate within- from
            # between-experiment redundancy, so it doesn't enter ρ.
            n_study = study_core_n.get(study_id, n_pts)
            unc_arr = pert['unc']
            val_arr = np.maximum(np.abs(pert['value']), 1e-30)
            rel_arr = unc_arr / val_arr
            finite = np.isfinite(rel_arr)
            rel_unc = float(np.sqrt(np.nanmean(rel_arr[finite]**2))) if finite.any() else 0.0
            if not np.isfinite(rel_unc) or rel_unc <= 0:
                rel_unc = min_relative_uncertainty if min_relative_uncertainty > 0 else sigma_norm
            rel_unc = max(rel_unc, min_relative_uncertainty)  # floor for ρ consistency
            ex_sigma_sys = 0.0
            df_ds_ex = ds.get('exfor_df')
            if df_ds_ex is not None and 'sigma_sys_relative' in df_ds_ex.columns and len(df_ds_ex) > 0:
                # Per-row sigma_sys is constant within a single (entry, energy)
                # cell (`ds` corresponds to one experiment at one energy).
                ex_sigma_sys = float(df_ds_ex['sigma_sys_relative'].iloc[0])
            sigma_eff = ex_sigma_sys if ex_sigma_sys > 0 else sigma_norm
            rho = sigma_eff**2 / (sigma_eff**2 + rel_unc**2)
            if band_aware_ess and study_id in study_core_n_by_band:
                # Per-band Kish: the core's per-band point count drives collapse
                # for each evaluation point in the same band.
                mu_pts = pert['mu']
                core_by_band = study_core_n_by_band[study_id]
                n_band_arr = np.where(
                    mu_pts > 0.5, core_by_band['F'],
                    np.where(mu_pts < -0.5, core_by_band['B'], core_by_band['M']),
                )
                # Fall back to total core count if a band has zero core points
                # (shouldn't normally happen given the angular-quality gate).
                n_band_arr = np.where(n_band_arr > 0, n_band_arr, n_study)
                g_arr = 1.0 / (1.0 + np.maximum(n_band_arr - 1, 0) * rho)
                per_point_w_arr = w * g_arr
                all_weights.append(per_point_w_arr.astype(float))
            else:
                g = 1.0 / (1.0 + max(n_study - 1, 0) * rho)
                per_point_w = w * g
                all_weights.append(np.full(n_pts, per_point_w))

        # Apply per-study weight capping (group by entry.subentry)
        if all_mu and max_experiment_weight_fraction < 1.0:
            weights_arr = np.concatenate(all_weights)
            # Track study (entry.subentry) id per point
            exp_ids = []
            for ds, w in datasets_and_weights:
                e_key = f"{ds['experiment_id']}_{ds['exfor_energy_mev']:.6f}"
                pert = perturbed_datasets.get(e_key)
                if pert is not None:
                    study_id = ds['experiment_id']  # full entry.subentry
                    exp_ids.extend([study_id] * len(pert['mu']))
            exp_ids = np.array(exp_ids)
            total_w = weights_arr.sum()
            if total_w > 0:
                unc_arr = np.concatenate(all_unc)
                unc_arr = np.maximum(unc_arr, 1e-30)
                unique_exps = np.unique(exp_ids)
                for _ in range(5):  # iterative WLS-level capping
                    changed = False
                    wls_eff = weights_arr / (unc_arr ** 2)
                    total_wls = wls_eff.sum()
                    if total_wls <= 0:
                        break
                    for exp in unique_exps:
                        mask = exp_ids == exp
                        frac = wls_eff[mask].sum() / total_wls
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
        sigma_sys_abs = np.concatenate(all_sigma_sys_abs) if all_sigma_sys_abs else None
        weights = np.concatenate(all_weights)

        if len(mu) < 3:
            nom = nominal_coeffs_by_bin.get(bin_idx)
            if nom is not None:
                sample_coeffs[bin_idx] = endf_normalize_legendre_coeffs(nom, include_a0=False)
            continue

        fit_df = pd.DataFrame({'mu': mu, 'value': values, 'unc': unc})
        # Sys-aware fit weights: σ_total² = (τ·σ_stat)² + σ_sys². Mirrors the
        # nominal fit and Pass 2 — without this, Pass 1 over-weights points
        # with small σ_stat but large σ_sys, distorting cross-bin correlations.
        # Gated on sys_aware_mc_fit so v2 callers stay bit-identical.
        sys_unc_col_name: Optional[str] = None
        if (sys_aware_mc_fit and sigma_sys_abs is not None
                and np.any(sigma_sys_abs > 0)):
            fit_df['sigma_sys_abs'] = sigma_sys_abs
            sys_unc_col_name = 'sigma_sys_abs'

        # Per-sample AICc-weighted degree drawing (v3 hook). When the caller
        # passes ``sampled_degrees_per_bin_sample`` (full-grid bin_idx →
        # length-n_samples int array) plus ``nominal_coeffs_by_bin_by_degree``
        # (bin_idx → {degree: padded_nominal_coeffs}), each sample uses its
        # drawn degree and the freeze-high step pulls from THAT drawn-degree's
        # nominal. Defaults to None → exact v2 behavior (every sample fits at
        # the AICc winner).
        sampled_degrees_per_bin_sample = (
            sh.get('sampled_degrees_per_bin_sample')
            if isinstance(args_tuple, int) else None
        )
        nominal_coeffs_by_bin_by_degree = (
            sh.get('nominal_coeffs_by_bin_by_degree')
            if isinstance(args_tuple, int) else None
        )
        if (sampled_degrees_per_bin_sample is not None
                and bin_idx in sampled_degrees_per_bin_sample):
            degree = int(sampled_degrees_per_bin_sample[bin_idx][s_idx])
        else:
            degree = frozen_degrees_by_bin.get(bin_idx, max_degree)
        if (nominal_coeffs_by_bin_by_degree is not None
                and bin_idx in nominal_coeffs_by_bin_by_degree
                and degree in nominal_coeffs_by_bin_by_degree[bin_idx]):
            nom_for_freeze_high = nominal_coeffs_by_bin_by_degree[bin_idx][degree]
        else:
            nom_for_freeze_high = nominal_coeffs_by_bin.get(bin_idx)

        # Apply per-bin MC order cap (from angular support diagnostics)
        bin_mc_cap = mc_order_cap_by_bin.get(bin_idx) if mc_order_cap_by_bin else None
        effective_sample_order = max_sample_order
        if bin_mc_cap is not None:
            degree = min(degree, bin_mc_cap)
            if effective_sample_order is not None:
                effective_sample_order = min(effective_sample_order, bin_mc_cap)
            else:
                effective_sample_order = bin_mc_cap

        # Pre-inflate uncertainties with frozen tau (from smoothed/floored nominal)
        bin_tau = tau_info_by_bin.get(bin_idx) if tau_info_by_bin else None
        fit_use_band_discrepancy = use_band_discrepancy
        if bin_tau and use_band_discrepancy:
            from .resample_AD import sigma_eff_from_tau
            inflated = sigma_eff_from_tau(mu, unc, bin_tau)
            fit_df['unc'] = inflated
            fit_use_band_discrepancy = False  # already applied

        # When fix_c0_at_nominal=True, pin c0 at the per-degree nominal so MF33
        # multiplicative perturbations propagate to a_l = c_l/c_0 instead of
        # cancelling out (with c0 floating, c0 absorbs the uniform scale and
        # a_l ≈ a_l_nom — the cross block sees no MF33→a_l correlation).
        c0_fix_arg: Optional[float] = None
        if fix_c0_at_nominal and freeze_c0 and nom_for_freeze_high is not None:
            c0_fix_arg = float(nom_for_freeze_high[0])

        try:
            coef_df, fit_info = sample_legendre_coefficients(
                fit_df,
                value_col="value",
                unc_col="unc",
                sys_unc_col=sys_unc_col_name,
                degree=degree,
                max_degree=max_degree,
                select_degree=None,
                ridge_lambda=ridge_lambda,
                external_weights=weights,
                n_samples=1,
                stochastic=False,
                use_band_discrepancy=fit_use_band_discrepancy,
                min_points_per_band=min_points_per_band,
                max_band_scale=max_band_scale,
                freeze_c0=freeze_c0,
                fixed_c0_value=c0_fix_arg,
                record_c0_scale=record_c0_channel,
                c0_scale_ref_coeffs=nom_for_freeze_high if record_c0_channel else None,
            )
            # Fixed-shape c0 scale of this sample's (shared-draw) perturbed data
            # against the frozen nominal shape — read-only, doesn't touch coeffs.
            if record_c0_channel and fit_info.get("c0_samples") is not None:
                sample_c0[bin_idx] = float(np.asarray(fit_info["c0_samples"]).ravel()[0])
            coeffs = coef_df.iloc[0].to_numpy()
            if len(coeffs) < max_degree + 1:
                coeffs = np.pad(coeffs, (0, max_degree + 1 - len(coeffs)))

            # Freeze higher-order coefficients at nominal values. Source is
            # the drawn-degree's nominal when per-sample AICc sampling is
            # active; otherwise the bin's AICc winner. Orders that the source
            # nominal didn't fit (e.g. drawn deg=2 → no L≥3 in nom) stay zero,
            # which is correct: a sample that 'chose' L=2 has no L≥3 content.
            if effective_sample_order is not None and nom_for_freeze_high is not None:
                for l in range(effective_sample_order + 1, len(coeffs)):
                    if l < len(nom_for_freeze_high):
                        coeffs[l] = nom_for_freeze_high[l]

            if apply_positivity_projection and positivity_check_points > 0:
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
                    n_pos_violations += 1

            sample_coeffs[bin_idx] = endf_normalize_legendre_coeffs(coeffs, include_a0=False)

        except Exception:
            nom = nominal_coeffs_by_bin.get(bin_idx)
            if nom is not None:
                sample_coeffs[bin_idx] = endf_normalize_legendre_coeffs(nom, include_a0=False)
                if record_c0_channel and len(nom) > 0:
                    # Fall back to the unperturbed magnitude so the c0 sample
                    # matrix stays aligned with the coeff samples.
                    sample_c0[bin_idx] = float(nom[0])

    if record_c0_channel:
        return s_idx, sample_coeffs, n_pos_violations, sample_c0
    return s_idx, sample_coeffs, n_pos_violations


def run_mc_with_kernel_weights(
    nominal_results: List,  # List[NominalFitResult]
    energy_bins: List[EnergyBinInfo],
    overlap_weights: Dict[int, List[Tuple[Dict, float]]],
    n_samples: int,
    n_workers: int,
    sigma_norm: float,
    sigma_norm_common_mode: float,
    norm_dist: str,
    max_degree: int,
    ridge_lambda: float,
    base_seed: int,
    use_band_discrepancy: bool = True,
    min_points_per_band: int = 3,
    max_band_scale: float = 3.0,
    freeze_c0: bool = True,
    fix_c0_at_nominal: bool = False,
    sys_aware_mc_fit: bool = False,
    max_sample_order: Optional[int] = None,
    apply_positivity_projection: bool = False,
    positivity_check_points: int = 101,
    max_experiment_weight_fraction: float = 1.0,
    min_relative_uncertainty: float = 0.05,
    band_aware_ess: bool = False,
    mf33_dsigma_per_sample: Optional[np.ndarray] = None,
    mf33_c0_per_bin: Optional[np.ndarray] = None,
    mf33_home_bin_by_e_key: Optional[Dict[str, int]] = None,
    sampled_degrees_per_bin_sample: Optional[Dict[int, np.ndarray]] = None,
    nominal_coeffs_by_bin_by_degree: Optional[Dict[int, Dict[int, np.ndarray]]] = None,
    record_c0_channel: bool = False,
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
        Per-experiment systematic normalization uncertainty. Drives the MC
        perturbation amplitude per experiment AND the Kish ρ for ESS collapse
        (same physical parameter in both roles).
    sigma_norm_common_mode : float
        Global normalization uncertainty applied as a single multiplicative
        factor per MC sample to ALL data points across ALL experiments
        (e.g. uncertainty in a shared reference / monitor).
        Set to 0.0 to disable.
    norm_dist : str
        "lognormal" or "normal".
    max_degree : int
        Maximum Legendre degree.
    ridge_lambda : float
        Ridge regularization parameter.
    base_seed : int
        Random seed.
    max_experiment_weight_fraction : float
        Maximum allowed weight fraction per study (1.0 = disabled).
    min_relative_uncertainty : float
        Minimum relative uncertainty (floor) for GLS-ESS ρ computation.
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
    tau_info_by_bin = {}
    mc_order_cap_by_bin = {}
    for nr in nominal_results:
        if nr.has_data:
            nominal_coeffs_by_bin[nr.energy_index] = nr.nominal_coeffs
            frozen_degrees_by_bin[nr.energy_index] = nr.frozen_degree
            if hasattr(nr, 'tau_info') and nr.tau_info:
                tau_info_by_bin[nr.energy_index] = nr.tau_info
            if hasattr(nr, 'mc_order_cap') and nr.mc_order_cap is not None:
                mc_order_cap_by_bin[nr.energy_index] = nr.mc_order_cap

    # Shared data dict — pickled once per worker (not once per sample)
    shared_data = {
        'overlap_weights': overlap_weights,
        'energy_bins_data': energy_bins_data,
        'sigma_norm': sigma_norm,
        'sigma_norm_common_mode': sigma_norm_common_mode,
        'norm_dist': norm_dist,
        'max_degree': max_degree,
        'ridge_lambda': ridge_lambda,
        'base_seed': base_seed,
        'use_band_discrepancy': use_band_discrepancy,
        'min_points_per_band': min_points_per_band,
        'max_band_scale': max_band_scale,
        'freeze_c0': freeze_c0,
        # When True (and freeze_c0=True), the worker pins c0 at the bin's
        # *nominal* value (per-degree-aware when degree sampling is on),
        # rather than the c0 fitted on the perturbed data. v3 needs this so
        # MF33 perturbations propagate to a_l = c_l/c_0 — with c0 floating,
        # a uniform multiplicative MF33 factor cancels out of a_l. Default
        # False preserves v2 behavior bit-for-bit.
        'fix_c0_at_nominal': fix_c0_at_nominal,
        # When True, MC fit weights use σ_total² = (τ·σ_stat)² + σ_sys² (with
        # σ_sys propagated through perturbed_datasets and rescaled per sample).
        # Default False keeps v2's stat-only WLS fit weights bit-identical.
        'sys_aware_mc_fit': sys_aware_mc_fit,
        'max_sample_order': max_sample_order,
        'apply_positivity_projection': apply_positivity_projection,
        'positivity_check_points': positivity_check_points,
        'nominal_coeffs_by_bin': nominal_coeffs_by_bin,
        'frozen_degrees_by_bin': frozen_degrees_by_bin,
        'max_experiment_weight_fraction': max_experiment_weight_fraction,
        'min_relative_uncertainty': min_relative_uncertainty,
        'tau_info_by_bin': tau_info_by_bin,
        'mc_order_cap_by_bin': mc_order_cap_by_bin,
        'band_aware_ess': band_aware_ess,
        # Optional MF33 hooks (default None → no behavior change for v2 callers).
        # Used by v3's two-pass orchestrator to inject a shared MF33 multiplicative
        # factor into the data perturbation step. Worker reads via sh.get(...).
        'mf33_dsigma_per_sample': mf33_dsigma_per_sample,
        'mf33_c0_per_bin': mf33_c0_per_bin,
        'mf33_home_bin_by_e_key': mf33_home_bin_by_e_key,
        # Optional per-sample AICc-weighted degree drawing (v3 hook). When both
        # are provided, every (bin, sample) draws its degree from the bin's
        # AICc weights and freezes high orders against THAT degree's nominal.
        # Default-off → exact v2 behavior.
        'sampled_degrees_per_bin_sample': sampled_degrees_per_bin_sample,
        'nominal_coeffs_by_bin_by_degree': nominal_coeffs_by_bin_by_degree,
        # Phase-2 magnitude channel: when True the worker also records the
        # fixed-shape c0 of each sample's perturbed data (read-only; the shape
        # coeffs are untouched). Default False → v2/v3 return shape unchanged.
        'record_c0_channel': record_c0_channel,
    }

    if n_workers > 1:
        if logger:
            logger.info(f"  Running kernel-weight MC with {n_workers} workers, {n_samples} samples")
        with Pool(n_workers, initializer=_init_kw_worker, initargs=(shared_data,)) as pool:
            results = pool.map(_run_one_kw_sample, range(n_samples))
    else:
        if logger:
            logger.info(f"  Running kernel-weight MC sequentially, {n_samples} samples")
        # Sequential: set shared state directly, pass int indices
        _init_kw_worker(shared_data)
        results = [_run_one_kw_sample(s_idx) for s_idx in range(n_samples)]

    # Assemble into expected format
    all_samples: Dict[int, Dict[int, np.ndarray]] = {s_idx: {} for s_idx in range(n_samples)}
    c0_samples_kw: Dict[int, Dict[int, float]] = {} if record_c0_channel else None
    total_pos_violations = 0
    for res in results:
        # Worker returns a 4-tuple (…, sample_c0) only when recording is on.
        if record_c0_channel:
            s_idx, sample_coeffs, n_pos_violations, sample_c0 = res
            c0_samples_kw[s_idx] = sample_c0
        else:
            s_idx, sample_coeffs, n_pos_violations = res
        all_samples[s_idx] = sample_coeffs
        total_pos_violations += n_pos_violations

    if logger:
        n_bins_with_data = sum(1 for b in energy_bins if overlap_weights.get(b.index))
        logger.info(f"  Kernel-weight MC complete: {n_samples} samples, {n_bins_with_data} bins with data")
        if apply_positivity_projection and positivity_check_points > 0:
            denom = n_samples * max(n_bins_with_data, 1)
            pct = 100.0 * total_pos_violations / max(denom, 1)
            logger.info(
                f"  Positivity: {total_pos_violations}/{denom} (bin, sample) "
                f"distributions projected ({pct:.2f}%)"
            )

    if record_c0_channel:
        return all_samples, c0_samples_kw
    return all_samples


def save_legendre_matrix_to_parquet(
    nominal_results: List,
    sample_matrix: np.ndarray,
    energy_indices: List[int],
    output_dir: str,
    max_degree: int,
    filename: str,
) -> str:
    """Wrapper around ``save_all_legendre_coefficients`` for callers (v3) that
    already hold the samples as a flat ``(n_samples, n_bins * max_degree)``
    ndarray and don't want to materialize a ``Dict[int, Dict[int, np.ndarray]]``
    just to feed it to the parquet writer.

    The dict is built lazily, consumed by the writer, and dropped — so the
    1M-entry dict overhead at 10k samples × 100 bins is transient.
    """
    n_samples = sample_matrix.shape[0]
    samples_dict = {
        s: {
            int(e_idx): sample_matrix[s, k * max_degree:(k + 1) * max_degree].copy()
            for k, e_idx in enumerate(energy_indices)
        }
        for s in range(n_samples)
    }
    return save_all_legendre_coefficients(
        nominal_results=nominal_results,
        all_samples=samples_dict,
        output_dir=output_dir,
        max_degree=max_degree,
        filename=filename,
    )


def stack_samples_to_matrix(
    samples_by_idx: Dict[int, Dict[int, np.ndarray]],
    energy_indices: List[int],
    n_samples: int,
    max_degree: int,
) -> np.ndarray:
    """Flatten ``{sample_idx -> {energy_idx -> coeffs}}`` into a contiguous
    ``(n_samples, len(energy_indices) * max_degree)`` ndarray.

    Missing entries (None or absent keys) leave zero blocks. Replaces the
    nested ``for s in range(n_samples): for k in ...`` loops that v2 and v3
    used to build their TMC / KW matrices; lifting the row pointer outside
    the inner loop and de-duplicating the call sites also lets each caller
    ``del samples_by_idx`` immediately afterwards to free the dict-of-dicts
    overhead (~hundreds of MB at 10k samples × 100 bins).
    """
    n_bins = len(energy_indices)
    out = np.zeros((n_samples, n_bins * max_degree), dtype=float)
    for s in range(n_samples):
        sd = samples_by_idx.get(s)
        if not sd:
            continue
        row = out[s]
        for k, e_idx in enumerate(energy_indices):
            arr = sd.get(int(e_idx))
            if arr is None:
                continue
            n = min(arr.shape[0], max_degree)
            row[k * max_degree:k * max_degree + n] = arr[:n]
    return out


def stack_c0_samples(
    c0_samples_by_idx: Dict[int, Dict[int, float]],
    energy_indices: List[int],
    n_samples: int,
) -> np.ndarray:
    """Flatten the fixed-shape c0 side channel ``{sample_idx -> {energy_idx ->
    c0}}`` into an ``(n_samples, len(energy_indices))`` ndarray.

    Missing (sample, bin) entries become NaN so the two-pass combine can treat
    them as absent (pairwise-complete correlations, per-column variances) rather
    than as a spurious zero magnitude.
    """
    n_bins = len(energy_indices)
    out = np.full((n_samples, n_bins), np.nan, dtype=float)
    for s in range(n_samples):
        sd = c0_samples_by_idx.get(s)
        if not sd:
            continue
        for k, e_idx in enumerate(energy_indices):
            val = sd.get(int(e_idx))
            if val is not None:
                out[s, k] = float(val)
    return out


def build_mf33_channel(
    c0_samples_pass1: Dict[int, Dict[int, float]],
    c0_samples_pass2: Dict[int, Dict[int, float]],
    energy_indices: List[int],
    c0_nominal: np.ndarray,
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray, "pd.DataFrame"]:
    """Assemble the fixed-shape c0 (MF33) channel from the two-pass samples.

    Pure (no I/O): stacks the two per-sample c0 dicts into matrices, runs the
    two-pass congruence combine, and builds a long-format sample frame for the
    sidecar.  The pipeline wraps this with the ``np.save`` / ``to_parquet``
    calls; keeping it separate makes the numeric path unit-testable without a
    full run.

    Returns
    -------
    (rel_cov, cov_abs, samples_df, diag)
        ``rel_cov`` / ``cov_abs`` are ``(n_bins, n_bins)`` relative / absolute
        c0 covariances aligned to ``energy_indices``; ``samples_df`` has columns
        ``sample_idx, energy_index, pass, c0`` for both passes.  ``diag`` holds
        completeness/PSD inspection numbers (warn-only material):
        ``p1_finite_per_bin`` / ``p2_finite_per_bin`` (finite-sample counts per
        bin) and ``corr_pass1_min_eig`` (min eigenvalue of the Pass-1
        pairwise-complete correlation matrix, which is not guaranteed PSD).
    """
    from .resample_AD import combine_c0_covariance

    c0_nominal = np.asarray(c0_nominal, dtype=float)
    p1 = stack_c0_samples(c0_samples_pass1, energy_indices, n_samples)
    p2 = stack_c0_samples(c0_samples_pass2, energy_indices, n_samples)
    rel_cov, cov_abs = combine_c0_covariance(p1, p2, c0_nominal)

    # Completeness + Pass-1 correlation PSD inspection (pairwise-complete
    # np.ma.corrcoef can go indefinite when different sample pairs are missing
    # in different bins).
    p1_counts = np.sum(np.isfinite(p1), axis=0).astype(int)
    p2_counts = np.sum(np.isfinite(p2), axis=0).astype(int)
    corr_p1 = np.ma.corrcoef(np.ma.masked_invalid(p1), rowvar=False)
    corr_p1 = np.asarray(np.ma.filled(corr_p1, 0.0), dtype=float)
    np.fill_diagonal(corr_p1, 1.0)
    corr_p1 = 0.5 * (corr_p1 + corr_p1.T)
    diag = {
        "p1_finite_per_bin": p1_counts,
        "p2_finite_per_bin": p2_counts,
        "corr_pass1_min_eig": float(np.min(np.linalg.eigvalsh(corr_p1))),
    }

    n_bins = len(energy_indices)
    s_col = np.repeat(np.arange(n_samples), n_bins)
    e_col = np.tile(np.asarray(energy_indices), n_samples)
    samples_df = pd.concat([
        pd.DataFrame({"sample_idx": s_col, "energy_index": e_col,
                      "pass": "pass1", "c0": p1.ravel()}),
        pd.DataFrame({"sample_idx": s_col, "energy_index": e_col,
                      "pass": "pass2", "c0": p2.ravel()}),
    ], ignore_index=True)
    return rel_cov, cov_abs, samples_df, diag


def recentre_relative_covariance(
    cov_abs: np.ndarray, ref_means: np.ndarray
) -> np.ndarray:
    """Convert an absolute covariance to relative against reference means.

    ``rel[i, j] = cov_abs[i, j] / (ref_means[i] * ref_means[j])`` — used to
    recentre the DCS-derived absolute c0 covariance on the HOST MF3 bin means
    (the shipped File-3 central), so the relative MF33 preserves the absolute
    uncertainty claim when users multiply it by the host cross section.  Rows
    and columns with a non-positive reference are zeroed.
    """
    cov_abs = np.asarray(cov_abs, dtype=float)
    ref = np.asarray(ref_means, dtype=float)
    rel = np.zeros_like(cov_abs)
    pos = ref > 0
    outer = np.outer(ref, ref)
    rel[np.ix_(pos, pos)] = cov_abs[np.ix_(pos, pos)] / outer[np.ix_(pos, pos)]
    return rel


def contiguous_grid_from_bins(valid_bins, rtol: float = 1e-9) -> np.ndarray:
    """Build the fine energy grid (eV) from has-data bins, asserting adjacency.

    The MF33/MF34 fine writes represent the bins as one contiguous grid
    (lower edges + last upper edge).  That is only correct when the bins are
    adjacent — a quality gate leaving an internal gap would silently produce a
    semantically wrong ENDF grid.  A gap is a structural bug, so this raises
    (hard error, not the warn-only policy reserved for PSD-type judgement
    calls).

    Parameters
    ----------
    valid_bins : sequence
        Bin objects exposing ``bin_lower_mev`` / ``bin_upper_mev``, in
        ascending energy order.
    rtol : float, default 1e-9
        Relative tolerance on ``upper[i] == lower[i+1]``.

    Returns
    -------
    np.ndarray
        Energy boundaries in eV, ``len(valid_bins) + 1`` values.
    """
    if not valid_bins:
        raise ValueError("contiguous_grid_from_bins: no bins given")
    for i in range(len(valid_bins) - 1):
        upper = float(valid_bins[i].bin_upper_mev)
        lower_next = float(valid_bins[i + 1].bin_lower_mev)
        if not np.isclose(upper, lower_next, rtol=rtol, atol=0.0):
            raise ValueError(
                f"contiguous_grid_from_bins: gap between bin {i} "
                f"(upper {upper:.9g} MeV) and bin {i + 1} "
                f"(lower {lower_next:.9g} MeV) — the fine-grid write assumes "
                f"adjacent bins; refusing to build a wrong contiguous grid."
            )
    grid_ev = np.empty(len(valid_bins) + 1, dtype=float)
    for k, b in enumerate(valid_bins):
        grid_ev[k] = float(b.bin_lower_mev) * 1e6
    grid_ev[-1] = float(valid_bins[-1].bin_upper_mev) * 1e6
    return grid_ev


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
    mixture_blocks: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]], np.ndarray, np.ndarray]:
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
    mixture_blocks : Dict[int, Dict[str, np.ndarray]], optional
        Phase-3 per-bin mixture moments, ``{energy_index: {'mean': (max_order,),
        'cov': (max_order, max_order)}}``, in the same ENDF a-space and the same
        absolute units as the samples.

        When given, each listed bin's diagonal block of the ABSOLUTE covariance
        and its slice of the mean vector are replaced before anything else
        happens. Everything downstream — the ``valid_mask`` zeroing, the
        relative conversion, the near-zero regularisation and the correlation
        extraction — then runs on the mixture exactly as it would on a sample
        covariance. That is deliberate: the near-zero guard is the only active
        protection against relative-sigma blow-up, and routing the mixture
        through this function is what keeps it applied rather than requiring a
        second copy of it at the call site.

        Bins not listed keep their sample estimates, so interpolated bins and
        bins whose MC failed fall through untouched. **Cross-bin blocks are
        never replaced** — they stay as estimated from the pooled samples.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]], np.ndarray, np.ndarray]
        - cov_matrix: Full relative (fractional) covariance matrix (w.r.t. MC means)
        - corr_matrix: Full correlation matrix
        - param_labels: List of (energy_index, order) tuples
        - mean_params: MC mean parameter vector used as denominator for
          the relative conversion (same layout as param_labels)
        - cov_abs: Absolute covariance matrix (before relative conversion)

    Covariance Conversion
    ---------------------
    ``np.cov()`` computes absolute covariance: Cov(a_i, a_j).
    ENDF MF34 with LB=5 expects relative (fractional) covariance:
        Cov_rel(i, j) = Cov_abs(i, j) / (mean_i * mean_j)

    The conversion is performed here so that the returned matrix can be
    written directly to MF34 with LB=5 format. Rows/columns of parameters
    with |mean| < 1e-6 (effectively zero coefficients at ENDF precision)
    are set to zero in the relative covariance.
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
    mean_params = np.mean(sample_matrix, axis=0)

    # Phase 3: substitute the analytic per-bin mixture moments for the sample
    # estimates. Done here, before the mask and the relative conversion, so the
    # mixture is subject to every guard the sampled path is subject to.
    if mixture_blocks:
        n_sub = 0
        for k, e_idx in enumerate(energy_indices):
            blk = mixture_blocks.get(e_idx)
            if not blk:
                continue
            m = np.asarray(blk['mean'], dtype=float)
            c = np.asarray(blk['cov'], dtype=float)
            if m.shape != (max_order,) or c.shape != (max_order, max_order):
                raise ValueError(
                    f"mixture block for energy_index {e_idx} has mean{m.shape} "
                    f"cov{c.shape}, expected ({max_order},) and "
                    f"({max_order}, {max_order})"
                )
            s = k * max_order
            e = s + max_order
            cov_abs[s:e, s:e] = 0.5 * (c + c.T)
            mean_params[s:e] = m
            n_sub += 1
        if logger:
            logger.info(
                f"  [MIX] per-bin mixture blocks substituted for {n_sub}/"
                f"{n_energies} bins (cross-bin blocks left as sampled)"
            )

    # Zero out rows/columns for parameters that were not actually fitted
    if valid_mask is not None:
        invalid = ~valid_mask
        cov_abs[invalid, :] = 0.0
        cov_abs[:, invalid] = 0.0

    # Convert absolute covariance to relative (fractional) covariance
    # Cov_rel(i,j) = Cov_abs(i,j) / (mean_i * mean_j)
    # Per-parameter safe test (|mean| > 1e-6, ~ENDF 6-sig-digit precision),
    # mirroring absolute_to_nominal_relative: the pairwise-product test
    # (|mean_i*mean_j| > threshold²) could zero a diagonal variance while
    # keeping its cross-terms, breaking PSD.
    param_safe = np.abs(mean_params) > 1e-6
    safe_mask = np.outer(param_safe, param_safe)
    denom = np.outer(mean_params, mean_params)
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

    return cov_matrix, corr_matrix, param_labels, mean_params, cov_abs


def absolute_to_nominal_relative(
    cov_abs: np.ndarray,
    nominal_params: np.ndarray,
    zero_threshold: float = 1e-6,
) -> np.ndarray:
    """Convert absolute covariance to relative-to-nominal.

    Cov_rel(i,j) = Cov_abs(i,j) / (nom_i * nom_j)

    Parameters with ``|nom_i| < zero_threshold`` are treated as
    undefined: their entire row **and** column are set to zero.  This
    preserves positive semi-definiteness (equivalent to removing those
    parameters from the matrix).

    The default threshold (1e-6) corresponds to the ENDF 11-character
    field precision (~6 significant digits).  Coefficients below this
    are indistinguishable from zero at ENDF precision.

    .. versionchanged:: 0.x
       Previous versions tested the *pairwise product*
       ``|nom_i * nom_j| < threshold²``, which could zero the diagonal
       (variance) while keeping cross-terms, breaking PSD.  The new
       per-parameter test avoids this.
    """
    param_safe = np.abs(nominal_params) > zero_threshold
    # Both row AND column parameter must be safe
    pair_safe = np.outer(param_safe, param_safe)
    denom = np.outer(nominal_params, nominal_params)

    cov_rel = np.zeros_like(cov_abs)
    cov_rel[pair_safe] = cov_abs[pair_safe] / denom[pair_safe]
    return cov_rel


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


def regularize_high_relative_std(
    cov_rel: np.ndarray,
    max_order: int,
    max_rel_std: float = 1.0,
    n_neighbors: int = 3,
    logger=None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Regularize explosive relative standard deviations via neighbor
    interpolation.

    When converting absolute covariance to nominal-relative via
    ``cov_abs / outer(nom, nom)``, near-zero nominal coefficients produce
    large relative stds.  This function detects entries with
    ``rel_std > max_rel_std`` and replaces them with the median of
    neighboring unflagged bins (same Legendre order), applied via
    congruence transform (preserves correlations and PSD).

    Parameters
    ----------
    cov_rel : np.ndarray
        Relative covariance matrix (nominal-relative).
    max_order : int
        Number of Legendre orders per energy bin.
    max_rel_std : float
        Flag parameters with relative std above this.
    n_neighbors : int
        Neighbors to seek on each side for interpolation.
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

    # Flag entries with relative std too large
    flagged = rel_std > max_rel_std
    n_flagged = int(np.sum(flagged))

    if n_flagged == 0:
        diagnostics = {"n_regularized": 0, "n_total": n_params}
        if logger:
            logger.info("  Rel-std regularization: no parameters flagged")
        return cov_rel.copy(), diagnostics

    # Per-order global fallback: median of unflagged stds
    order_fallback = np.full(max_order, max_rel_std)
    for l in range(max_order):
        order_indices = np.arange(l, n_params, max_order)
        unflagged_stds = rel_std[order_indices][~flagged[order_indices]]
        unflagged_stds = unflagged_stds[unflagged_stds > 0]
        if len(unflagged_stds) > 0:
            order_fallback[l] = float(np.median(unflagged_stds))

    # Neighbor-interpolated target
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
    targets = {}  # i -> (target, n_neighbors_used, used_fallback)

    for i in range(n_params):
        if not flagged[i] or rel_std[i] <= 0:
            continue
        target, n_nb, fb = _neighbor_target(i)
        scale[i] = target / rel_std[i]
        targets[i] = (target, n_nb, fb)

    # Apply congruence transform: C' = S @ C @ S (PSD-preserving)
    cov_reg = cov_rel * np.outer(scale, scale)

    diagnostics = {
        "n_regularized": n_flagged,
        "n_total": n_params,
    }

    if logger:
        logger.info(f"  Rel-std regularization: {n_flagged}/{n_params} "
                    f"capped (>{max_rel_std*100:.0f}%)")
        new_rel_std = np.sqrt(np.maximum(np.diag(cov_reg), 0.0))
        for l in range(max_order):
            order_indices = np.arange(l, n_params, max_order)
            order_flagged = flagged[order_indices]
            n_oh = int(np.sum(order_flagged))
            if n_oh > 0:
                old_max = float(np.max(rel_std[order_indices][order_flagged]))
                new_max = float(np.max(new_rel_std[order_indices][order_flagged]))
                order_targets = [targets[idx][0] for idx in order_indices[order_flagged] if idx in targets]
                order_n_fb = sum(1 for idx in order_indices[order_flagged] if idx in targets and targets[idx][2])
                tgt_min = min(order_targets) if order_targets else 0
                tgt_max = max(order_targets) if order_targets else 0
                logger.info(f"    l={l+1}: {n_oh} capped "
                            f"(max {old_max*100:.1f}% -> {new_max*100:.1f}%), "
                            f"targets [{tgt_min*100:.1f}%-{tgt_max*100:.1f}%], "
                            f"{order_n_fb} used fallback")

    return cov_reg, diagnostics


def apply_between_experiment_floor(
    cov_rel: np.ndarray,
    nominal_results: List,
    energy_indices: List[int],
    max_order: int,
    logger=None,
    apply: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply between-experiment scatter as an uncertainty floor.

    For each energy bin with ``between_exp_scatter`` available, compare
    the scatter-implied relative std to the current covariance diagonal.
    Where the scatter exceeds the MC-estimated uncertainty, inflate via
    a congruence scale (preserves correlations and PSD).

    When ``apply=False`` the diagnostics are still computed and logged
    (marked as *not applied*) so the user can assess their potential
    impact.  A warning is emitted when the floor *would have* inflated
    a significant number of entries.

    Parameters
    ----------
    cov_rel : np.ndarray
        Relative covariance matrix (nominal-relative).
    nominal_results : list of NominalFitResult
        Results from ``perform_nominal_fits()``.
    energy_indices : list of int
        Energy indices corresponding to the covariance matrix rows/cols.
    max_order : int
        Number of Legendre orders per energy bin.
    logger : optional
        Logger instance.
    apply : bool
        If True (default), return the floored covariance.  If False,
        return the original covariance unchanged but still log
        diagnostics.

    Returns
    -------
    cov_floored : np.ndarray
        Covariance matrix (floored if ``apply=True``, unchanged otherwise).
    diagnostics : dict
        Summary statistics.
    """
    from scripts.resample_AD import endf_normalize_legendre_coeffs

    n_params = cov_rel.shape[0]
    rel_std = np.sqrt(np.maximum(np.diag(cov_rel), 0.0))
    scale = np.ones(n_params)

    # Build lookup: energy_index -> NominalFitResult
    nr_lookup = {}
    for nr in nominal_results:
        if nr.has_data and not nr.interpolated:
            nr_lookup[nr.energy_index] = nr

    n_floored = 0
    per_order_stats = {l: {'n_floored': 0, 'inflation_factors': []} for l in range(max_order)}

    for k, e_idx in enumerate(energy_indices):
        nr = nr_lookup.get(e_idx)
        if nr is None or nr.between_exp_scatter is None:
            continue

        scatter = nr.between_exp_scatter
        L_common = nr.between_exp_L_common

        # Get nominal ENDF a_l coefficients for this bin
        endf_a_l = endf_normalize_legendre_coeffs(nr.nominal_coeffs, include_a0=False)

        for l_idx in range(min(L_common, max_order)):
            param_idx = k * max_order + l_idx

            # scatter is in absolute a_l units; convert to relative
            if l_idx < len(endf_a_l) and abs(endf_a_l[l_idx]) > 1e-15:
                scatter_rel = scatter[l_idx] / abs(endf_a_l[l_idx])
            else:
                continue

            current_rel_std = rel_std[param_idx]
            if current_rel_std > 1e-15 and scatter_rel > current_rel_std:
                s = scatter_rel / current_rel_std
                scale[param_idx] = s
                n_floored += 1
                per_order_stats[l_idx]['n_floored'] += 1
                per_order_stats[l_idx]['inflation_factors'].append(s)

    # Apply congruence transform: C' = S @ C @ S
    cov_floored = cov_rel * np.outer(scale, scale)

    n_bins_with_scatter = sum(
        1 for e_idx in energy_indices
        if nr_lookup.get(e_idx) is not None and nr_lookup[e_idx].between_exp_scatter is not None
    )

    diagnostics = {
        'n_bins_with_scatter': n_bins_with_scatter,
        'n_floored': n_floored,
        'per_order': per_order_stats,
    }

    status = "APPLIED" if apply else "NOT APPLIED (diagnostic only)"
    if logger:
        logger.info(f"  [Between-exp floor] {status} — "
                     f"{n_bins_with_scatter}/{len(energy_indices)} bins had "
                     f"scatter computed, {n_floored} (E,l) entries would be floored")
        for l in range(max_order):
            stats = per_order_stats[l]
            if stats['n_floored'] > 0:
                mean_infl = float(np.mean(stats['inflation_factors']))
                max_infl = float(np.max(stats['inflation_factors']))
                logger.info(f"    l={l+1}: {stats['n_floored']} floored "
                            f"(mean inflation {mean_infl:.1f}x, max {max_infl:.1f}x)")

        # Warning when disabled but would have significant impact
        if not apply and n_floored > 0:
            frac = n_floored / max(1, n_bins_with_scatter * max_order) * 100
            all_factors = []
            for l in range(max_order):
                all_factors.extend(per_order_stats[l]['inflation_factors'])
            median_infl = float(np.median(all_factors)) if all_factors else 1.0
            logger.warning(
                f"  [Between-exp floor WARNING] Floor is DISABLED but would inflate "
                f"{n_floored} entries ({frac:.0f}% of eligible, median {median_infl:.1f}x). "
                f"Consider setting APPLY_BETWEEN_EXP_FLOOR=True."
            )

    if apply:
        return cov_floored, diagnostics
    return cov_rel, diagnostics


def apply_between_experiment_floor_mg(
    cov_rel: np.ndarray,
    nominal_results: List,
    valid_indices: List[int],
    groups: List[List[int]],
    max_order: int,
    logger=None,
    apply: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Multigroup analogue of ``apply_between_experiment_floor``.

    The fine-grid floor compares per-bin between-experiment scatter to the
    fine-grid covariance diagonal.  At the multigroup level each group spans
    several fine bins, so the floor for ``(group g, order l)`` is the *maximum*
    scatter_rel across constituent fine bins (most conservative choice; a
    weaker scatter floor in one bin should not undo a stronger one in
    another).

    Parameters
    ----------
    cov_rel : np.ndarray
        Multigroup nominal-relative covariance matrix, shape
        ``(n_groups * max_order, n_groups * max_order)``.
    nominal_results : list of NominalFitResult
        Per-fine-bin nominal fits (used for ``between_exp_scatter`` and
        ``nominal_coeffs``).
    valid_indices : list of int
        Indices into ``nominal_results`` for the non-interpolated fine bins
        that participate in the multigroup aggregation.  ``valid_indices[k]``
        is the ``nominal_results`` index for the k-th fine bin used in the MG
        layout.
    groups : list of list of int
        ``multigroup_result.groups``.  ``groups[g]`` is a list of positions
        into ``valid_indices`` (i.e. into the non-interpolated fine subset).
    max_order : int
        Number of Legendre orders per bin.
    logger : optional
    apply : bool
        Same semantics as the FG version: when False, diagnostics are still
        computed and logged but the covariance is returned unchanged.

    Returns
    -------
    cov_floored : np.ndarray
    diagnostics : dict
    """
    from scripts.resample_AD import endf_normalize_legendre_coeffs

    n_groups = len(groups)
    if cov_rel.shape[0] != n_groups * max_order:
        raise ValueError(
            f"apply_between_experiment_floor_mg: cov_rel has {cov_rel.shape[0]} "
            f"rows, expected n_groups*max_order = {n_groups * max_order}"
        )

    rel_std = np.sqrt(np.maximum(np.diag(cov_rel), 0.0))
    scale = np.ones(cov_rel.shape[0])

    n_floored = 0
    n_groups_with_scatter = 0
    per_order_stats = {l: {"n_floored": 0, "inflation_factors": []}
                        for l in range(max_order)}

    for g, fine_positions in enumerate(groups):
        # Collect each constituent fine bin's scatter_rel per order
        scatter_rel_per_order: List[List[float]] = [[] for _ in range(max_order)]
        for fp in fine_positions:
            nr_idx = valid_indices[fp]
            nr = nominal_results[nr_idx]
            if nr is None or nr.between_exp_scatter is None:
                continue
            scatter = nr.between_exp_scatter
            L_common = nr.between_exp_L_common
            endf_a_l = endf_normalize_legendre_coeffs(
                nr.nominal_coeffs, include_a0=False
            )
            for l_idx in range(min(L_common, max_order)):
                if l_idx < len(endf_a_l) and abs(endf_a_l[l_idx]) > 1e-15:
                    scatter_rel_per_order[l_idx].append(
                        float(scatter[l_idx]) / abs(float(endf_a_l[l_idx]))
                    )

        if any(scatter_rel_per_order):
            n_groups_with_scatter += 1

        for l_idx in range(max_order):
            if not scatter_rel_per_order[l_idx]:
                continue
            scatter_rel_g = max(scatter_rel_per_order[l_idx])
            param_idx = g * max_order + l_idx
            current_rel_std = rel_std[param_idx]
            if current_rel_std > 1e-15 and scatter_rel_g > current_rel_std:
                s = scatter_rel_g / current_rel_std
                scale[param_idx] = s
                n_floored += 1
                per_order_stats[l_idx]["n_floored"] += 1
                per_order_stats[l_idx]["inflation_factors"].append(s)

    cov_floored = cov_rel * np.outer(scale, scale)

    diagnostics = {
        "n_groups_with_scatter": int(n_groups_with_scatter),
        "n_floored": int(n_floored),
        "per_order": per_order_stats,
    }

    status = "APPLIED" if apply else "NOT APPLIED (diagnostic only)"
    if logger:
        logger.info(
            f"  [Between-exp floor MG] {status} — "
            f"{n_groups_with_scatter}/{n_groups} groups had scatter computed, "
            f"{n_floored} (group,l) entries would be floored"
        )
        for l in range(max_order):
            stats = per_order_stats[l]
            if stats["n_floored"] > 0:
                mean_infl = float(np.mean(stats["inflation_factors"]))
                max_infl = float(np.max(stats["inflation_factors"]))
                logger.info(
                    f"    l={l + 1}: {stats['n_floored']} floored "
                    f"(mean inflation {mean_infl:.1f}x, max {max_infl:.1f}x)"
                )
        if not apply and n_floored > 0:
            frac = n_floored / max(1, n_groups_with_scatter * max_order) * 100
            all_factors = []
            for l in range(max_order):
                all_factors.extend(per_order_stats[l]["inflation_factors"])
            median_infl = float(np.median(all_factors)) if all_factors else 1.0
            logger.warning(
                f"  [Between-exp floor MG WARNING] Floor is DISABLED but would "
                f"inflate {n_floored} entries ({frac:.0f}% of eligible, "
                f"median {median_infl:.1f}x)."
            )

    if apply:
        return cov_floored, diagnostics
    return cov_rel, diagnostics


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
    dip_fraction: Optional[float] = 0.50,
    spike_factor: Optional[float] = 3.0,
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
                    if dip_fraction is not None and order_rel_std[k] < dip_fraction * med:
                        dip[k] = True
                        new_outliers += 1
                    elif spike_factor is not None and order_rel_std[k] > spike_factor * med:
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


def inject_within_bin_correlations(
    corr_cross: np.ndarray,
    corr_perbin: np.ndarray,
    n_energies: int,
    block_size: int,
) -> np.ndarray:
    """Replace within-bin diagonal blocks of a correlation matrix with per-bin values.

    Parameters
    ----------
    corr_cross : np.ndarray
        Correlation matrix whose off-diagonal (cross-bin) blocks are kept.
    corr_perbin : np.ndarray
        Correlation matrix from the per-bin MC pass, used to inject
        within-bin cross-order correlations.
    n_energies : int
        Number of energy bins.
    block_size : int
        Number of Legendre orders per bin (max_degree).

    Returns
    -------
    np.ndarray
        Correlation matrix with within-bin blocks from *corr_perbin* and
        cross-bin blocks from *corr_cross*.
    """
    corr_out = corr_cross.copy()
    for ie in range(n_energies):
        s = ie * block_size
        e = s + block_size
        corr_out[s:e, s:e] = corr_perbin[s:e, s:e]
    return corr_out


def psd_repair_correlation_active(
    corr: np.ndarray,
    dead_mask: np.ndarray,
    eigenvalue_threshold: float = -1e-10,
    eigval_floor: float = 1e-14,
    label: str = "",
    logger=None,
) -> np.ndarray:
    """Repair PSD of a correlation matrix, excluding dead parameters via Schur complement.

    Dead parameters are reinserted as block-diagonal sentinels (off-diag=0,
    diag=1), making the full matrix PSD by construction (block-diagonal sum
    of two PSD pieces). Active parameters are repaired via Higham only if
    the active-block min eigenvalue is below ``eigenvalue_threshold``.

    Operating on a correlation matrix (unit diagonal) keeps Higham's repair
    scale-free, avoiding the smearing that occurs when the same eigenvalue
    clip is applied in covariance space (where damage scales with std).

    Parameters
    ----------
    corr : np.ndarray (n, n)
        Correlation matrix to repair (unit diagonal expected).
    dead_mask : np.ndarray (n,) bool
        True where parameter is "dead" (low SNR, unfitted, or zero variance)
        and should be excluded from the eigendecomposition.
    eigenvalue_threshold : float
        Min eigenvalue below which Higham is triggered on the active subset.
    eigval_floor : float
        Floor passed to ``nearest_psd_higham``.
    label : str
        Short label for log messages.
    logger : logger or None
        Logger instance (optional).

    Returns
    -------
    np.ndarray
        Repaired correlation matrix. PSD by construction.
    """
    n = corr.shape[0]
    active = ~dead_mask
    n_active = int(np.sum(active))
    n_dead = n - n_active

    out = np.zeros_like(corr)
    np.fill_diagonal(out, 1.0)

    if n_active == 0:
        if logger is not None:
            logger.info(
                f"  [PSD repair / {label}] all {n} parameters dead — "
                f"returning identity sentinel"
            )
        return out

    active_idx = np.where(active)[0]
    corr_active = corr[np.ix_(active_idx, active_idx)]
    eigs = np.linalg.eigvalsh(corr_active)
    min_eig = float(np.min(eigs))

    if min_eig < eigenvalue_threshold:
        from kika.cov.decomposition import nearest_psd_higham
        corr_active_repaired, _psd_info = nearest_psd_higham(
            corr_active, preserve_diagonal=True, eigval_floor=eigval_floor,
            verbose=False, logger=logger,
        )
        max_dc = _psd_info.get('max_diagonal_change', 0.0)
        if logger is not None:
            logger.info(
                f"  [PSD repair / {label}] Higham on active subset: "
                f"n_active={n_active}, n_dead={n_dead}, "
                f"min_eig={min_eig:.3e}, max_diag_change={max_dc:.3e}"
            )
    else:
        corr_active_repaired = corr_active
        if logger is not None:
            logger.info(
                f"  [PSD repair / {label}] active subset already PSD: "
                f"n_active={n_active}, n_dead={n_dead}, min_eig={min_eig:.3e}"
            )

    out[np.ix_(active_idx, active_idx)] = corr_active_repaired
    np.fill_diagonal(out, 1.0)
    return out


def threshold_small_correlations(
    cov_rel: np.ndarray,
    threshold: float,
    eigenvalue_threshold: float = -1e-10,
    eigval_floor: float = 1e-14,
    label: str = "",
    logger=None,
) -> Tuple[np.ndarray, dict]:
    """Hard-zero off-diagonal correlations whose magnitude is below ``threshold``.

    Operates on a covariance matrix (relative or absolute) by extracting
    std, zeroing small correlations, and rebuilding cov with the same std.
    PSD is validated via ``nearest_psd_higham`` as a fallback if hard
    zeroing creates negative eigenvalues.

    Note: ENDF-6 MF34 LB=5 stores the upper triangle dense, so this does
    not save file size. The natural floor for ``threshold`` is sampling
    noise (~ 1/sqrt(N) for N samples).

    Returns
    -------
    Tuple of (thresholded covariance, info dict with n_zeroed/min_eig/higham_applied).
    """
    if threshold <= 0:
        return cov_rel, {
            'n_zeroed': 0, 'n_total_offdiag': 0,
            'min_eig': None, 'higham_applied': False,
        }

    n = cov_rel.shape[0]
    std = np.sqrt(np.maximum(np.diag(cov_rel), 0.0))
    safe_std = std.copy()
    safe_std[safe_std == 0] = 1.0
    corr = cov_rel / np.outer(safe_std, safe_std)
    np.fill_diagonal(corr, 1.0)

    mask = np.abs(corr) < threshold
    np.fill_diagonal(mask, False)
    n_total_offdiag = n * (n - 1) // 2
    n_zeroed = int(np.sum(mask)) // 2  # symmetric
    corr[mask] = 0.0

    cov_out = corr * np.outer(std, std)
    eigs = np.linalg.eigvalsh(cov_out)
    min_eig = float(np.min(eigs))
    higham_applied = False

    if min_eig < eigenvalue_threshold:
        from kika.cov.decomposition import nearest_psd_higham
        cov_out, _info = nearest_psd_higham(
            cov_out, preserve_diagonal=True, eigval_floor=eigval_floor,
            verbose=False, logger=logger,
        )
        higham_applied = True
        eigs2 = np.linalg.eigvalsh(cov_out)
        if not np.all(np.isfinite(eigs2)) or np.min(eigs2) < eigenvalue_threshold * 100:
            raise RuntimeError(
                f"Threshold step ({label}): Higham failed to restore PSD after "
                f"hard-zero (final min_eig={float(np.min(eigs2)):.3e}). "
                f"Consider lowering MULTIGROUP_CORRELATION_THRESHOLD."
            )

    info = {
        'n_zeroed': n_zeroed,
        'n_total_offdiag': n_total_offdiag,
        'min_eig': min_eig,
        'higham_applied': higham_applied,
    }
    if logger is not None:
        frac = 100.0 * n_zeroed / max(n_total_offdiag, 1)
        logger.info(
            f"  [Threshold / {label}] zeroed {n_zeroed}/{n_total_offdiag} "
            f"off-diagonals ({frac:.1f}%) at |rho|<{threshold:.2e}; "
            f"min_eig={min_eig:.3e}, "
            f"higham={'applied' if higham_applied else 'not needed'}"
        )
    return cov_out, info


def log_psd_diagnostics(matrix: np.ndarray, label: str, logger) -> dict:
    """Log eigenvalue spectrum diagnostics for a (cov or corr) matrix.

    Reports min eigenvalue, count of negative eigenvalues, and total
    negative mass (sum of negative eigenvalues). The negative mass is the
    scalar most relevant to assessing PSD-repair smearing damage.

    Parameters
    ----------
    matrix : np.ndarray
        Symmetric matrix (covariance or correlation).
    label : str
        Short label identifying the checkpoint (e.g. "corr_kw").
    logger : logger
        Logger instance for the message.

    Returns
    -------
    dict with keys 'min_eig', 'n_neg', 'neg_mass', 'n_total'.
    """
    eigs = np.linalg.eigvalsh(matrix)
    n_total = int(len(eigs))
    n_neg = int(np.sum(eigs < 0))
    neg_mass = float(np.sum(eigs[eigs < 0])) if n_neg > 0 else 0.0
    min_eig = float(np.min(eigs))
    logger.info(
        f"  [PSD diag] {label}: min_eig={min_eig:.3e}, "
        f"n_neg={n_neg}/{n_total}, neg_mass={neg_mass:.3e}"
    )
    return {
        'min_eig': min_eig,
        'n_neg': n_neg,
        'neg_mass': neg_mass,
        'n_total': n_total,
    }


def build_gaussian_relevance_matrix(
    energy_bins: List,
    energy_indices: List[int],
    max_order: int,
) -> np.ndarray:
    """Compute pairwise Gaussian-model relevance for range-aware hybrid blend.

    Returns a matrix g_ij in [0, 1] indicating how much the Gaussian energy-
    correlation model "has an opinion" about the (i, j) parameter pair.
    g_ij ≈ 1 for nearby bins (|dE| << sigma_E) where the Gaussian decay
    provides a meaningful alternative to KW correlations, and g_ij → 0 for
    distant bins where the Gaussian model contributes nothing (rho_E → 0).

    The relevance is defined as rho_E(dE, sigma_eff) — the same Gaussian
    decay kernel used inside ``build_gaussian_correlation_covariance``.
    Same-energy pairs get relevance = 1 (the Gaussian model keeps the
    stochastic cross-order correlation there).

    Parameters
    ----------
    energy_bins : list
        EnergyBinInfo objects with energy_mev and sigma_E_mev.
    energy_indices : list of int
        Energy indices for the parameter layout.
    max_order : int
        Legendre orders per energy bin.

    Returns
    -------
    np.ndarray, shape (n_params, n_params)
        Symmetric matrix with values in [0, 1].
    """
    n_energies = len(energy_indices)
    n = n_energies * max_order
    bin_by_idx = {b.index: b for b in energy_bins}

    # Build per-parameter energy and sigma_E arrays
    param_energies = np.zeros(n)
    param_sigma_E = np.zeros(n)
    param_e_pos = np.zeros(n, dtype=int)

    for p in range(n):
        e_pos = p // max_order
        param_e_pos[p] = e_pos
        if e_pos < n_energies and energy_indices[e_pos] in bin_by_idx:
            ebin = bin_by_idx[energy_indices[e_pos]]
            param_energies[p] = ebin.energy_mev
            param_sigma_E[p] = ebin.sigma_E_mev

    # Fallback for zero sigma_E
    if np.any(param_sigma_E > 0):
        median_sigma = np.median(param_sigma_E[param_sigma_E > 0])
    else:
        median_sigma = 0.01
    param_sigma_E[param_sigma_E <= 0] = median_sigma

    # Gaussian decay: same kernel as build_gaussian_correlation_covariance
    sigma_eff_mat = (param_sigma_E[:, None] + param_sigma_E[None, :]) / 2.0
    dE_mat = param_energies[:, None] - param_energies[None, :]
    relevance = np.exp(-dE_mat**2 / (2.0 * sigma_eff_mat**2))

    # Same-energy pairs: Gaussian model fully applies (keeps stochastic corr)
    same_e = param_e_pos[:, None] == param_e_pos[None, :]
    relevance[same_e] = 1.0

    return relevance


def build_gaussian_correlation_covariance(
    cov_stochastic: np.ndarray,
    energy_bins: List,
    energy_indices: List[int],
    max_order: int,
    logger=None,
    valid_mask: Optional[np.ndarray] = None,
    corr_stochastic_in: Optional[np.ndarray] = None,
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
    corr_stochastic_in : np.ndarray, optional
        Pre-computed correlation matrix from absolute covariance.  When
        provided, this is used instead of extracting correlation from
        *cov_stochastic* (which is relative and can flip signs for
        parameters with negative means).

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
    if corr_stochastic_in is not None:
        corr_stochastic = corr_stochastic_in
    else:
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

    # Build index→bin lookup (handles non-sequential indices from union grids)
    bin_by_idx = {b.index: b for b in energy_bins}

    for p in range(n):
        e_pos = p // max_order
        order = p % max_order
        param_orders[p] = order
        param_e_pos[p] = e_pos
        if e_pos < n_energies and energy_indices[e_pos] in bin_by_idx:
            ebin = bin_by_idx[energy_indices[e_pos]]
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

    # Ridge regularization: add small nugget to diagonal for numerical stability.
    # Near-zero-mean params create near-singular cov_abs (zero rows/columns).
    # The nugget makes the matrix strictly PD, bounds the condition number,
    # and adds negligible variance (~1e-10 relative to max diagonal).
    diag_abs = np.diag(cov_abs)
    nugget = np.max(diag_abs[diag_abs > 0]) * 1e-10 if np.any(diag_abs > 0) else 1e-30
    cov_abs[np.diag_indices_from(cov_abs)] += nugget
    if logger:
        logger.info(f"  Ridge regularization: nugget={nugget:.2e}")

    # Cholesky decomposition (Higham PSD projection fallback if not PD)
    cholesky_succeeded = False
    try:
        L = np.linalg.cholesky(cov_abs)
        cholesky_succeeded = True
        if logger:
            logger.info(f"  Cholesky decomposition successful ({n_params}x{n_params})")
    except np.linalg.LinAlgError as e:
        # Cholesky failure: apply Higham nearest-PSD projection with diagonal
        # preservation (keeps variances exact, minimal off-diagonal distortion).
        from kika.cov.decomposition import nearest_psd_higham
        if logger:
            logger.warning(f"  WARNING: Cholesky decomposition FAILED ({e})")
            logger.info(f"  Applying Higham nearest-PSD projection (single-step eigenvalue clip)")
        cov_psd, psd_info = nearest_psd_higham(
            cov_abs, preserve_diagonal=False, eigval_floor=1e-14,
            verbose=True, logger=logger,
        )
        max_diag_change = psd_info.get('max_diagonal_change', 0.0)
        max_diag = np.max(np.diag(cov_abs))
        if logger and max_diag > 0:
            logger.info(f"  Max diagonal change: {max_diag_change:.2e} "
                        f"({max_diag_change / max_diag * 100:.4f}% of max variance)")
        try:
            L = np.linalg.cholesky(cov_psd)
            cholesky_succeeded = True
            if logger:
                logger.info(f"  Cholesky successful after PSD projection "
                            f"(frob_err={psd_info.get('relative_frobenius_error', 0):.2e})")
        except np.linalg.LinAlgError:
            # Extremely rare: clip didn't suffice, add small jitter
            jitter = np.max(np.diag(cov_psd)) * 1e-10
            cov_psd[np.diag_indices_from(cov_psd)] += jitter
            L = np.linalg.cholesky(cov_psd)
            if logger:
                logger.warning(f"  Cholesky still failed after PSD clip — applied jitter={jitter:.2e}")

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
    filename: str = 'legendre_samples.parquet',
) -> str:
    """
    Save all Legendre coefficients (nominal + all MC samples) to Parquet.

    Schema
    ------
    Each row carries ``is_nominal`` (bool):
      - ``is_nominal == True``  → deterministic nominal fit (also ``sample_idx == 0``)
      - ``is_nominal == False`` → MC draw (``sample_idx == 1 … N``)

    Consumers computing sample statistics (mean, std, Pearson correlations)
    MUST filter on ``df[~df.is_nominal]``. The nominal row is appended
    as-is and is **not** transformed by downstream calibration steps, so
    including it in correlation calculations breaks the per-column affine
    invariance that preserves Pearson between the raw KW and the
    calibrated parquets.

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

    # Nominal coefficients: sample_idx=0, is_nominal=True
    for nr in nominal_results:
        if nr.has_data:
            endf_coeffs = endf_normalize_legendre_coeffs(nr.nominal_coeffs, include_a0=False)
            padded_coeffs = np.zeros(max_degree)
            padded_coeffs[:len(endf_coeffs)] = endf_coeffs

            row = {
                'sample_idx': 0,
                'is_nominal': True,
                'energy_index': nr.energy_index,
                'energy_mev': nr.energy_mev,
            }
            for l in range(max_degree):
                row[f'a_{l+1}'] = padded_coeffs[l]
            data_rows.append(row)

    # MC samples: sample_idx=1..N, is_nominal=False
    n_samples = len(all_samples)
    for sample_idx in range(n_samples):
        sample_coeffs = all_samples[sample_idx]
        for energy_idx, endf_coeffs in sample_coeffs.items():
            energy_mev = None
            for nr in nominal_results:
                if nr.energy_index == energy_idx:
                    energy_mev = nr.energy_mev
                    break

            padded_coeffs = np.zeros(max_degree)
            padded_coeffs[:len(endf_coeffs)] = endf_coeffs

            row = {
                'sample_idx': sample_idx + 1,
                'is_nominal': False,
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
    parquet_file = output_path / filename
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
# SPLICE HELPERS FOR PIPELINE ENERGY GRID
# =============================================================================

def build_pipeline_coeffs_for_splice(
    energy_bins: List,  # List[EnergyBinInfo]
    coeffs_by_index: Dict[int, np.ndarray],
) -> Tuple[List[float], List[List[float]]]:
    """
    Build sorted (energies, coefficients) lists for splicing into MF4.

    Parameters
    ----------
    energy_bins : List[EnergyBinInfo]
        Energy bins from the pipeline, with .index, .energy_ev, .original_coeffs
    coeffs_by_index : Dict[int, ndarray]
        ENDF-format coefficients keyed by energy_bin.index

    Returns
    -------
    (energies_ev, coeffs_lists) : Tuple[List[float], List[List[float]]]
        Sorted by energy
    """
    sorted_bins = sorted(energy_bins, key=lambda b: b.energy_ev)
    energies = []
    coeffs = []
    for b in sorted_bins:
        energies.append(b.energy_ev)
        if b.index in coeffs_by_index:
            coeffs.append(list(coeffs_by_index[b.index]))
        else:
            coeffs.append(list(b.original_coeffs))
    return energies, coeffs


#: Half-width, in eV, of the band around the splice boundaries within which an
#: original ENDF point counts as coincident with the requested range and is
#: dropped in favour of the pipeline grid. Was an unnamed literal 1.0 in two
#: places.
SPLICE_KEEP_TOLERANCE_EV = 1.0


def split_original_grid(
    orig_energies,
    orig_coeffs,
    energy_min_ev: float,
    energy_max_ev: float,
) -> Tuple[List[float], List[List[float]], List[float], List[List[float]]]:
    """The original points kept either side of the splice window.

    Returns ``(left_e, left_c, right_e, right_c)``.
    """
    orig_e = np.asarray(orig_energies)
    tol = SPLICE_KEEP_TOLERANCE_EV
    keep = (orig_e < energy_min_ev - tol) | (orig_e > energy_max_ev + tol)

    left_idx = np.where(keep & (orig_e < energy_min_ev))[0]
    right_idx = np.where(keep & (orig_e > energy_max_ev))[0]

    return (
        [orig_e[i] for i in left_idx],
        [list(orig_coeffs[i]) for i in left_idx],
        [orig_e[i] for i in right_idx],
        [list(orig_coeffs[i]) for i in right_idx],
    )


def merge_spliced_grid(
    left_energies: List[float],
    left_coeffs: List[List[float]],
    pipeline_energies: List[float],
    pipeline_coeffs: List[List[float]],
    right_energies: List[float],
    right_coeffs: List[List[float]],
    logger=None,
) -> Tuple[List[float], List[List[float]]]:
    """Concatenate left + pipeline + right, and police the result.

    Evaluated files may legitimately repeat an abscissa. JEFF-4.0 Fe-56 does:
    MF4/MT2 carries two consecutive LIST subsections at 3.905 MeV with
    byte-identical coefficients. Points outside the splice window are kept
    verbatim, so both copies survived here and the old strict ``>`` check
    rejected a grid that had come straight out of the input file.

    Policy, in order:

    * **Exact duplicate** — same energy *and* same coefficients. A redundant
      record, carrying no information: the second copy is dropped and the
      event logged.
    * **Same energy, different coefficients** — a genuine ENDF discontinuity.
      Both are kept and a warning names the energy. Note that the single
      lin-lin region assigned by the callers cannot represent a repeated
      abscissa; that is a separate hazard and is deliberately not papered over
      here.
    * **Out of order** — still an error. This is the check that has to survive,
      and relaxing it to "non-decreasing" is what makes it meaningful again.
    """
    energies = list(left_energies) + list(pipeline_energies) + list(right_energies)
    coeffs = (
        [list(c) for c in left_coeffs]
        + [list(c) for c in pipeline_coeffs]
        + [list(c) for c in right_coeffs]
    )

    merged_e: List[float] = []
    merged_c: List[List[float]] = []
    n_dropped = 0
    for energy, coefficient in zip(energies, coeffs):
        if merged_e and energy < merged_e[-1]:
            raise AssertionError(
                f"Spliced grid out of order at index {len(merged_e)}: "
                f"{merged_e[-1]:.1f} > {energy:.1f}"
            )
        if merged_e and energy == merged_e[-1]:
            if coefficient == merged_c[-1]:
                n_dropped += 1
                if logger:
                    logger.info(
                        f"  [SPLICE] dropped a duplicate MF4 record at "
                        f"E = {energy:.1f} eV (identical coefficients)"
                    )
                continue
            if logger:
                logger.warning(
                    f"  [SPLICE] E = {energy:.1f} eV appears twice with "
                    f"different coefficients; keeping both as a discontinuity"
                )
        merged_e.append(energy)
        merged_c.append(coefficient)

    if n_dropped and logger:
        logger.info(f"  [SPLICE] dropped {n_dropped} duplicate MF4 record(s) in total")

    return merged_e, merged_c


def splice_legendre_grid(
    mt_data,                        # MF4MTLegendre or MF4MTMixed
    pipeline_energies_ev: List[float],
    pipeline_coeffs: List[List[float]],
    energy_min_ev: float,
    energy_max_ev: float,
    logger=None,
) -> None:
    """
    Remove original ENDF energy points in [E_min, E_max] and insert pipeline
    points instead. Points outside that range are kept.

    Modifies mt_data in-place (_energies, _legendre_coeffs, _interpolation, _nr).
    """
    left_e, left_c, right_e, right_c = split_original_grid(
        mt_data._energies, mt_data._legendre_coeffs, energy_min_ev, energy_max_ev
    )

    new_energies, new_coeffs = merge_spliced_grid(
        left_e, left_c,
        list(pipeline_energies_ev), [list(c) for c in pipeline_coeffs],
        right_e, right_c,
        logger=logger,
    )

    # Assign back
    mt_data._energies = new_energies
    mt_data._legendre_coeffs = new_coeffs
    new_ne = len(new_energies)
    mt_data._interpolation = [(new_ne, 2)]
    mt_data._nr = 1


# =============================================================================
# ENDF WRITING FUNCTIONS
# =============================================================================

def write_nominal_endf(
    original_endf_file: str,
    mt_number: int,
    nominal_results: List,  # List[NominalFitResult]
    output_dir: str,
    energy_bins: Optional[List] = None,
    energy_range_mev: Optional[Tuple[float, float]] = None,
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
    energy_bins : Optional[List[EnergyBinInfo]]
        Pipeline energy bins (enables splice mode)
    energy_range_mev : Optional[Tuple[float, float]]
        (E_min, E_max) in MeV for the splice range

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

    if energy_bins is not None and energy_range_mev is not None:
        # Splice mode: replace energy grid in [E_min, E_max] with pipeline grid
        coeffs_by_index = {}
        for nr in nominal_results:
            if nr.has_data:
                endf_coeffs = endf_normalize_legendre_coeffs(nr.nominal_coeffs, include_a0=False)
                coeffs_by_index[nr.energy_index] = endf_coeffs
        energies, coeffs = build_pipeline_coeffs_for_splice(energy_bins, coeffs_by_index)
        e_min_ev = energy_range_mev[0] * 1e6
        e_max_ev = energy_range_mev[1] * 1e6
        splice_legendre_grid(mt_data, energies, coeffs, e_min_ev, e_max_ev,
                             logger=_logger)
    else:
        # Legacy in-place mode
        for nr in nominal_results:
            endf_idx = getattr(nr, 'endf_index', None)
            if endf_idx is None:
                endf_idx = nr.energy_index
            if nr.has_data and endf_idx < len(mt_data._legendre_coeffs):
                endf_coeffs = endf_normalize_legendre_coeffs(nr.nominal_coeffs, include_a0=False)
                mt_data._legendre_coeffs[endf_idx] = list(endf_coeffs)

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
    energy_bins: Optional[List] = None,
    energy_range_mev: Optional[Tuple[float, float]] = None,
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
    energy_bins : Optional[List[EnergyBinInfo]]
        Pipeline energy bins (enables splice mode)
    energy_range_mev : Optional[Tuple[float, float]]
        (E_min, E_max) in MeV for the splice range

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

    if energy_bins is not None and energy_range_mev is not None:
        # Splice mode
        energies, coeffs = build_pipeline_coeffs_for_splice(energy_bins, mc_mean_coeffs)
        e_min_ev = energy_range_mev[0] * 1e6
        e_max_ev = energy_range_mev[1] * 1e6
        splice_legendre_grid(mt_data, energies, coeffs, e_min_ev, e_max_ev,
                             logger=_logger)
    else:
        # Legacy in-place mode
        idx_to_endf = {}
        for nr in nominal_results:
            endf_idx = getattr(nr, 'endf_index', None)
            if endf_idx is not None:
                idx_to_endf[nr.energy_index] = endf_idx

        for e_idx, mean_coeffs in mc_mean_coeffs.items():
            endf_idx = idx_to_endf.get(e_idx, e_idx)
            if endf_idx < len(mt_data._legendre_coeffs):
                mt_data._legendre_coeffs[endf_idx] = list(mean_coeffs)

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
    energy_bins: Optional[List] = None,
    energy_range_mev: Optional[Tuple[float, float]] = None,
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
    energy_bins : Optional[List[EnergyBinInfo]]
        Pipeline energy bins (enables splice mode)
    energy_range_mev : Optional[Tuple[float, float]]
        (E_min, E_max) in MeV for the splice range

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

    if energy_bins is not None and energy_range_mev is not None:
        # Splice mode
        energies, coeffs = build_pipeline_coeffs_for_splice(energy_bins, sampled_coeffs_by_energy)
        e_min_ev = energy_range_mev[0] * 1e6
        e_max_ev = energy_range_mev[1] * 1e6
        splice_legendre_grid(mt_data, energies, coeffs, e_min_ev, e_max_ev,
                             logger=_logger)
    else:
        # Legacy in-place mode
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
    energy_bins: Optional[List] = None,
    energy_range_mev: Optional[Tuple[float, float]] = None,
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
    energy_bins : Optional[List[EnergyBinInfo]]
        Pipeline energy bins (enables splice mode)
    energy_range_mev : Optional[Tuple[float, float]]
        (E_min, E_max) in MeV for the splice range

    Returns
    -------
    List[str]
        Paths to output files
    """
    n_total = len(all_samples)
    use_splice = energy_bins is not None and energy_range_mev is not None

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
                energy_bins,
                energy_range_mev,
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

    # Create writer once
    writer = ENDFWriter(original_endf_file)
    output_files = []

    if use_splice:
        # Splice mode: establish grid structure once, then swap coefficients per sample
        e_min_ev = energy_range_mev[0] * 1e6
        e_max_ev = energy_range_mev[1] * 1e6

        # Compute left/right portions once (same for every sample)
        left_energies, left_coeffs, right_energies, right_coeffs = split_original_grid(
            mt_data_template._energies, mt_data_template._legendre_coeffs,
            e_min_ev, e_max_ev,
        )

        # Build pipeline energies once (same for every sample)
        sorted_bins = sorted(energy_bins, key=lambda b: b.energy_ev)
        pipeline_energies = [b.energy_ev for b in sorted_bins]

        for sample_idx in sorted(all_samples.keys()):
            # Build pipeline coefficients for this sample
            sampled = all_samples[sample_idx]
            pipeline_coeffs = []
            for b in sorted_bins:
                if b.index in sampled:
                    pipeline_coeffs.append(list(sampled[b.index]))
                else:
                    pipeline_coeffs.append(list(b.original_coeffs))

            # Assemble through the shared merge. This branch used to
            # concatenate left + pipeline + right by hand, with no sortedness
            # check at all — so where the parallel branch raised on a bad grid,
            # this one wrote it to disk in silence. The grid it produces is the
            # same for every sample (duplicates come from the constant
            # left/right portions), so re-merging per sample costs nothing and
            # buys one implementation instead of two.
            merged_e, merged_c = merge_spliced_grid(
                left_energies, left_coeffs,
                pipeline_energies, pipeline_coeffs,
                right_energies, right_coeffs,
                logger=_logger if sample_idx == min(all_samples) else None,
            )
            mt_data_template._energies = merged_e
            mt_data_template._legendre_coeffs = merged_c
            mt_data_template._interpolation = [(len(merged_e), 2)]
            mt_data_template._nr = 1

            # Write output file
            sample_str = f"{sample_idx + 1:04d}"
            sample_dir = output_path / "endf" / sample_str
            sample_dir.mkdir(parents=True, exist_ok=True)
            output_file = sample_dir / f"{base}_{sample_str}.endf"

            success = writer.replace_mf_section(mf4_template, str(output_file))
            if not success:
                raise RuntimeError(f"Failed to write {output_file}")

            remove_mf34_from_file(str(output_file))
            output_files.append(str(output_file))

            if (sample_idx + 1) % 50 == 0 or sample_idx == 0 or sample_idx == n_total - 1:
                print(f"[INFO] Writing sample {sample_idx + 1}/{n_total}")
    else:
        # Legacy in-place mode
        original_coeffs = [list(c) for c in mt_data_template._legendre_coeffs]

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


# =============================================================================
# PIPELINE DIAGNOSTICS (sigma-coverage, outlier bins, chi2/N) — DEPRECATED
# -----------------------------------------------------------------------------
# This block previously emitted a post-pipeline chi²/N summary against EXFOR
# data using the nominal coefficients. After the manifest refactor (per-dataset
# σ_stat / σ_sys decomposition), the in-pipeline chi² statistic was no longer
# a faithful goodness-of-fit measure: it would combine post-floor σ_stat and
# manifest-derived σ_sys in a way that didn't reflect how the MC variance and
# covariance are constructed. The functions below are KEPT FOR REFERENCE but
# are no longer called from the pipeline. External chi² analysis lives in
# `scripts/precompute_chi2_data.ipynb` + `scripts/chi2_analysis.ipynb`, which
# pulls per-point σ_stat ⊕ σ_sys from the manifest and adds the integral
# cross-section uncertainty from the JEFF / JENDL GENDF (MF33) covariance
# files.
# =============================================================================

import traceback as _traceback
from numpy.polynomial.legendre import legvander as _legvander, legval as _legval


def _legendre_jacobian(mu: float, max_degree: int) -> np.ndarray:
    """Row vector J_l = (2l+1) P_l(mu) for l = 1..max_degree."""
    pl = _legvander(np.asarray(mu, dtype=float), max_degree)[0, 1:]
    weights = 2.0 * np.arange(1, max_degree + 1) + 1.0
    return pl * weights


def _eval_xs_at_mu(c0: float, a_l_endf: np.ndarray, mu: float) -> float:
    """sigma(E,mu) = c0 * (1 + sum_{l>=1} a_l (2l+1) P_l(mu)) for ENDF-normalized a_l."""
    L = len(a_l_endf)
    weights = 2.0 * np.arange(1, L + 1) + 1.0
    raw_c = np.concatenate(([c0], c0 * a_l_endf * weights))
    return float(_legval(float(mu), raw_c))


def build_residuals_dataframe(
    nominal_results: List[Any],
    cov_abs: Optional[np.ndarray],
    energy_indices: List[int],
    max_degree: int,
) -> Tuple[pd.DataFrame, int]:
    """Build per-EXFOR-point residuals DataFrame for end-of-pipeline diagnostics.

    Returns (df, n_bins_skipped). cov_abs may be None: sigma_eval/sigma_total/
    n_sigma_total columns will be NaN in that case.
    """
    has_cov = cov_abs is not None
    n_idx_to_pos = {e_idx: i for i, e_idx in enumerate(energy_indices)}

    rows = []
    n_skipped = 0
    for nr in nominal_results:
        if not getattr(nr, 'has_data', False):
            continue
        if nr.exfor_df is None or len(nr.exfor_df) == 0:
            n_skipped += 1
            continue

        c0 = float(nr.nominal_coeffs[0])
        a_l_endf = endf_normalize_legendre_coeffs(nr.nominal_coeffs, include_a0=False)
        a_l_endf = np.asarray(a_l_endf, dtype=float)
        L_eff = min(max_degree, len(a_l_endf))
        a_l_endf = a_l_endf[:L_eff]
        if L_eff < max_degree:
            a_l_endf = np.concatenate([a_l_endf, np.zeros(max_degree - L_eff)])

        cov_block = None
        if has_cov and nr.energy_index in n_idx_to_pos:
            i = n_idx_to_pos[nr.energy_index]
            s, e = i * max_degree, (i + 1) * max_degree
            if e <= cov_abs.shape[0]:
                cov_block = cov_abs[s:e, s:e]

        for _, ex in nr.exfor_df.iterrows():
            mu = float(ex['mu'])
            y_exp = float(ex['value'])
            sigma_exp = float(ex['unc'])

            y_eval = _eval_xs_at_mu(c0, a_l_endf, mu)

            if cov_block is not None and sigma_exp > 0:
                J = _legendre_jacobian(mu, max_degree)
                var_shape = float(c0 * c0 * (J @ cov_block @ J))
                sigma_eval = float(np.sqrt(max(var_shape, 0.0)))
                sigma_total = float(np.sqrt(sigma_exp ** 2 + sigma_eval ** 2))
                n_sigma_total = abs(y_exp - y_eval) / sigma_total if sigma_total > 0 else np.nan
            else:
                sigma_eval = np.nan
                sigma_total = np.nan
                n_sigma_total = np.nan

            n_sigma_exp = abs(y_exp - y_eval) / sigma_exp if sigma_exp > 0 else np.nan

            rows.append({
                'energy_mev': float(nr.energy_mev),
                'energy_index': int(nr.energy_index),
                'mu': mu,
                'y_exp': y_exp,
                'y_eval': y_eval,
                'sigma_exp': sigma_exp,
                'sigma_eval': sigma_eval,
                'sigma_total': sigma_total,
                'n_sigma_exp': n_sigma_exp,
                'n_sigma_total': n_sigma_total,
            })

    df = pd.DataFrame(rows)
    return df, n_skipped


def compute_sigma_coverage(df: pd.DataFrame, sigma_col: str) -> Optional[Dict[str, float]]:
    """Fractions of points in each n_sigma band. Returns None if column all NaN."""
    if sigma_col not in df.columns:
        return None
    s = df[sigma_col].dropna()
    n = len(s)
    if n == 0:
        return None
    le1 = (s <= 1).sum()
    b12 = ((s > 1) & (s <= 2)).sum()
    b23 = ((s > 2) & (s <= 3)).sum()
    gt3 = (s > 3).sum()
    return {
        'N': int(n),
        'frac_le_1': 100.0 * le1 / n,
        'frac_1_2': 100.0 * b12 / n,
        'frac_2_3': 100.0 * b23 / n,
        'frac_gt_3': 100.0 * gt3 / n,
    }


def compute_outlier_bins(df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    """Per-bin outlier counts using n_sigma_total (falls back to n_sigma_exp)."""
    col = 'n_sigma_total' if 'n_sigma_total' in df.columns and df['n_sigma_total'].notna().any() else 'n_sigma_exp'
    grouped = df.groupby(['energy_index', 'energy_mev'], dropna=False)
    rows = []
    for (e_idx, e_mev), grp in grouped:
        n_total = len(grp)
        n_outlier = int((grp[col].dropna() > threshold).sum())
        frac = 100.0 * n_outlier / n_total if n_total > 0 else 0.0
        rows.append({
            'energy_index': int(e_idx),
            'energy_mev': float(e_mev),
            'n_total': int(n_total),
            'n_outlier': int(n_outlier),
            'frac_outlier': frac,
        })
    return pd.DataFrame(rows)


def compute_chi2_summary(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Scenarios A (sigma_exp only) and B (sigma_total). chi2/N is sum of n_sigma**2 / N."""
    out: Dict[str, Dict[str, Any]] = {}

    sa = df['n_sigma_exp'].dropna() if 'n_sigma_exp' in df.columns else pd.Series([], dtype=float)
    if len(sa) > 0:
        out['A'] = {'N': int(len(sa)), 'chi2_red': float((sa ** 2).sum() / len(sa))}
    else:
        out['A'] = {'N': 0, 'chi2_red': float('nan')}

    sb = df['n_sigma_total'].dropna() if 'n_sigma_total' in df.columns else pd.Series([], dtype=float)
    if len(sb) > 0:
        out['B'] = {'N': int(len(sb)), 'chi2_red': float((sb ** 2).sum() / len(sb))}
    else:
        out['B'] = {'N': 0, 'chi2_red': float('nan')}

    return out


def log_diagnostics_block(
    nominal_results: List[Any],
    cov_abs: Optional[np.ndarray],
    energy_indices: List[int],
    max_degree: int,
    logger: Any,
    outlier_threshold: float = 2.0,
    top_n_worst: int = 5,
) -> None:
    """Emit sigma-coverage, outlier-bin, and chi2/N summary to the run log.

    Never raises. On any failure, logs a single skip line.
    """
    try:
        if not nominal_results or not any(getattr(nr, 'has_data', False) for nr in nominal_results):
            logger.info("")
            logger.info(">> diagnostics skipped: no EXFOR-fitted bins")
            return

        cov_for_calc = cov_abs
        cov_reason = None
        if cov_abs is None:
            cov_reason = "cov_abs is None"
        else:
            expected = len(energy_indices) * max_degree
            if cov_abs.shape[0] != expected:
                cov_reason = f"cov shape {cov_abs.shape[0]} != {expected}"
                cov_for_calc = None

        df, n_skipped = build_residuals_dataframe(
            nominal_results=nominal_results,
            cov_abs=cov_for_calc,
            energy_indices=energy_indices,
            max_degree=max_degree,
        )

        if len(df) == 0:
            logger.info("")
            logger.info(">> diagnostics skipped: no EXFOR points after filtering")
            return

        n_bins_with_data = int(df['energy_index'].nunique())
        n_total = int(len(df))

        # ── SIGMA COVERAGE ─────────────────────────────────────────────────
        cov_total = compute_sigma_coverage(df, 'n_sigma_total')
        cov_exp = compute_sigma_coverage(df, 'n_sigma_exp')

        logger.info("")
        logger.info("#== SIGMA COVERAGE =========================================================")
        logger.info(">> note: sigma_eval is angular-shape only (MF34-equivalent); MF33/c0 not folded")
        if cov_reason:
            logger.info(f">> sigma_total unavailable: {cov_reason}")
        logger.info(f">> N points = {n_total}   bins with data = {n_bins_with_data}")
        if n_skipped:
            logger.info(f">> bins skipped (no exfor data) = {n_skipped}")
        logger.info("                       <=1s    1-2s    2-3s     >3s")
        if cov_total is not None:
            logger.info(
                f"   sigma_total      :  {cov_total['frac_le_1']:5.1f}%  "
                f"{cov_total['frac_1_2']:5.1f}%   {cov_total['frac_2_3']:5.1f}%   {cov_total['frac_gt_3']:5.1f}%"
            )
        else:
            logger.info("   sigma_total      :    n/a     n/a     n/a     n/a")
        if cov_exp is not None:
            logger.info(
                f"   sigma_exp only   :  {cov_exp['frac_le_1']:5.1f}%  "
                f"{cov_exp['frac_1_2']:5.1f}%   {cov_exp['frac_2_3']:5.1f}%   {cov_exp['frac_gt_3']:5.1f}%"
            )
        else:
            logger.info("   sigma_exp only   :    n/a     n/a     n/a     n/a")
        logger.info("   Gaussian ref     :  68.27%  27.18%   4.28%   0.27%")
        logger.info("#== END SIGMA COVERAGE =====================================================")

        # ── OUTLIER BINS ───────────────────────────────────────────────────
        bin_df = compute_outlier_bins(df, threshold=outlier_threshold)
        ref_col = 'n_sigma_total' if cov_for_calc is not None else 'n_sigma_exp'

        logger.info("")
        logger.info("#== OUTLIER BINS ===========================================================")
        logger.info(
            f">> per-bin fraction of EXFOR points with {ref_col} > {outlier_threshold} (top {top_n_worst} worst)"
        )
        logger.info("   E [MeV]     n_total   n_outlier    frac")
        worst = bin_df.sort_values('frac_outlier', ascending=False).head(top_n_worst)
        for _, r in worst.iterrows():
            logger.info(
                f"   {r['energy_mev']:7.3f}    {int(r['n_total']):6d}     {int(r['n_outlier']):6d}    {r['frac_outlier']:5.1f}%"
            )
        logger.info(">> all bins (ascending energy):")
        for _, r in bin_df.sort_values('energy_mev').iterrows():
            logger.info(
                f"   {r['energy_mev']:7.3f}    {int(r['n_total']):6d}     {int(r['n_outlier']):6d}    {r['frac_outlier']:5.1f}%"
            )
        logger.info("#== END OUTLIER BINS =======================================================")

        # ── CHI2 SUMMARY ───────────────────────────────────────────────────
        chi2 = compute_chi2_summary(df)
        a_n, a_v = chi2['A']['N'], chi2['A']['chi2_red']
        b_n, b_v = chi2['B']['N'], chi2['B']['chi2_red']

        logger.info("")
        logger.info("#== CHI2 SUMMARY ===========================================================")
        if a_n > 0:
            logger.info(f">> Scenario A (sigma_exp only) :  chi2/N = {a_v:.3f}   (N={a_n})")
        else:
            logger.info(">> Scenario A (sigma_exp only) :  n/a")
        if b_n > 0 and np.isfinite(b_v):
            logger.info(f">> Scenario B (sigma_total)    :  chi2/N = {b_v:.3f}   (N={b_n})")
        else:
            logger.info(">> Scenario B (sigma_total)    :  n/a (covariance unavailable)")
        logger.info("#== END CHI2 SUMMARY =======================================================")

        # Console echo (single line)
        if b_n > 0 and np.isfinite(b_v):
            logger.info(
                f"  [INFO] chi2/N: A(exp)={a_v:.3f}  B(total)={b_v:.3f}  (N={a_n})",
                console=True,
            )
        elif a_n > 0:
            logger.info(
                f"  [INFO] chi2/N: A(exp)={a_v:.3f}  (B unavailable)  (N={a_n})",
                console=True,
            )

    except Exception as e:
        try:
            logger.info(f">> diagnostics skipped: internal error ({type(e).__name__})")
            logger.debug(_traceback.format_exc())
        except Exception:
            pass
