"""
TOF (Time-of-Flight) Parameters Module for Energy Resolution Calculation.

This module provides functions for:
- Loading TOF parameters from JSON files (per-experiment flight path and time resolution)
- Computing energy resolution σE from TOF parameters
- Fallback to default values when experiment-specific parameters unavailable

The TOF energy resolution formula:
    σE = E × 2 × (δt / t)

where:
    E = neutron energy
    t = L / v = flight path / velocity
    v = c × √(2E/m_n)  (relativistically corrected for high energies)
    δt = time resolution (ns)
    L = flight path (m)

Author: Generated for kika project (Improvement 1.4)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np


# Physical constants
NEUTRON_MASS_MEV = 939.565378  # MeV/c²
SPEED_OF_LIGHT_M_NS = 0.299792458  # m/ns


@dataclass
class TOFParameters:
    """
    TOF parameters for a single experiment/subentry.

    Attributes
    ----------
    flight_path_m : float
        Total flight path in meters (source to detector)
    time_resolution_ns : float
        Time resolution in nanoseconds
    source : str
        Source of parameters: "file" (from JSON) or "default" (fallback values)
    """
    flight_path_m: float
    time_resolution_ns: float
    source: str  # "file" or "default"

    def __repr__(self) -> str:
        return (
            f"TOFParameters(L={self.flight_path_m:.2f}m, "
            f"δt={self.time_resolution_ns:.1f}ns, source={self.source})"
        )


def load_tof_parameters_file(filepath: str) -> Dict[str, Dict[str, Any]]:
    """
    Load TOF parameters from a JSON file.

    The JSON file is expected to have the structure:
    {
        "subentry_id": {
            "energy_resolution_input": {
                "distance": {"value": <float>, "unit": "m"},
                "time_resolution": {"value": <float>, "unit": "ns"}
            },
            ...
        },
        ...
    }

    Parameters
    ----------
    filepath : str
        Path to the TOF parameters JSON file

    Returns
    -------
    Dict[str, Dict]
        Dictionary keyed by subentry ID with TOF parameter data

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    json.JSONDecodeError
        If the file is not valid JSON
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"TOF parameters file not found: {filepath}")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def get_tof_parameters(
    subentry: str,
    tof_params_cache: Dict[str, Dict[str, Any]],
    default_flight_path_m: float = 27.037,
    default_time_resolution_ns: float = 5.0,
) -> TOFParameters:
    """
    Get TOF parameters for a subentry with fallback to defaults.

    Attempts to find TOF parameters in the cache for the given subentry.
    If not found or values are null, falls back to default values.

    Parameters
    ----------
    subentry : str
        EXFOR subentry identifier (e.g., "10037024")
    tof_params_cache : Dict[str, Dict]
        Pre-loaded TOF parameters from load_tof_parameters_file()
    default_flight_path_m : float
        Default flight path in meters (default: 27.037m, typical ORELA)
    default_time_resolution_ns : float
        Default time resolution in nanoseconds (default: 5.0ns)

    Returns
    -------
    TOFParameters
        TOF parameters with source indicator
    """
    # Try to find the subentry in cache
    if subentry in tof_params_cache:
        entry_data = tof_params_cache[subentry]

        # Try to extract energy_resolution_input
        eri = entry_data.get('energy_resolution_input') or entry_data.get('energy_resolution_inputs')

        if eri:
            distance_data = eri.get('distance', {})
            time_res_data = eri.get('time_resolution', {})

            flight_path = distance_data.get('value')
            time_res = time_res_data.get('value')

            # If both values are present and not None, use them
            if flight_path is not None and time_res is not None:
                return TOFParameters(
                    flight_path_m=float(flight_path),
                    time_resolution_ns=float(time_res),
                    source="file"
                )

    # Fallback to defaults
    return TOFParameters(
        flight_path_m=default_flight_path_m,
        time_resolution_ns=default_time_resolution_ns,
        source="default"
    )


def compute_sigma_E(
    energy_mev: float,
    tof_params: TOFParameters,
    min_sigma_E_kev: float = 1.0,
) -> float:
    """
    Compute energy resolution σE from TOF parameters.

    Uses the formula:
        σE = E × 2 × (δt / t)

    where:
        t = L / v  (flight time)
        v = c × √(2E / m_n)  (neutron velocity, non-relativistic)

    For non-relativistic neutrons (E << m_n c²):
        v = √(2E / m_n) in natural units

    In practical units:
        v [m/ns] = sqrt(2 * E_MeV / 939.565) * c [m/ns]
                 = sqrt(2 * E_MeV / 939.565) * 0.2998 m/ns

    Parameters
    ----------
    energy_mev : float
        Neutron energy in MeV
    tof_params : TOFParameters
        TOF parameters (flight path and time resolution)
    min_sigma_E_kev : float
        Minimum σE floor in keV (default: 1.0 keV)

    Returns
    -------
    float
        Energy resolution σE in MeV
    """
    if energy_mev <= 0:
        return min_sigma_E_kev / 1000.0  # Return minimum in MeV

    # Neutron velocity in m/ns (non-relativistic)
    # v = sqrt(2*E/m) where E and m are in consistent units
    # E_MeV / m_MeV gives dimensionless ratio
    # Then multiply by c to get velocity in m/ns
    velocity_m_ns = SPEED_OF_LIGHT_M_NS * np.sqrt(2.0 * energy_mev / NEUTRON_MASS_MEV)

    # Flight time in ns
    flight_time_ns = tof_params.flight_path_m / velocity_m_ns

    # Energy resolution: σE/E = 2 × δt/t
    # σE = E × 2 × δt/t
    sigma_E_mev = energy_mev * 2.0 * (tof_params.time_resolution_ns / flight_time_ns)

    # Apply minimum floor
    min_sigma_E_mev = min_sigma_E_kev / 1000.0
    return max(sigma_E_mev, min_sigma_E_mev)


def compute_sigma_E_direct(
    energy_mev: float,
    flight_path_m: float,
    time_resolution_ns: float,
    min_sigma_E_kev: float = 1.0,
) -> float:
    """
    Compute energy resolution σE directly from flight path and time resolution.

    Convenience wrapper that doesn't require creating a TOFParameters object.

    Parameters
    ----------
    energy_mev : float
        Neutron energy in MeV
    flight_path_m : float
        Flight path in meters
    time_resolution_ns : float
        Time resolution in nanoseconds
    min_sigma_E_kev : float
        Minimum σE floor in keV (default: 1.0 keV)

    Returns
    -------
    float
        Energy resolution σE in MeV
    """
    tof_params = TOFParameters(
        flight_path_m=flight_path_m,
        time_resolution_ns=time_resolution_ns,
        source="direct"
    )
    return compute_sigma_E(energy_mev, tof_params, min_sigma_E_kev)


def fold_xs_over_resolution(
    e_grid_ev: np.ndarray,
    xs: np.ndarray,
    energy_mev: float,
    sigma_E_mev: float,
    n_nodes: int = 12,
) -> float:
    """Average a tabulated cross section σ(E') over a Gaussian energy-resolution
    kernel N(energy_mev, sigma_E_mev²) via Gauss–Hermite quadrature.

    σ(E') is obtained by linear interpolation on the (e_grid_ev, xs) table and
    clamped to the table endpoints outside its coverage (numpy.interp default).
    With sigma_E_mev <= 0 the kernel collapses to a delta and the unfolded value
    σ(energy_mev) is returned. Units follow `xs` (barns for MF3 elastic).

    Gauss–Hermite is exact for the Gaussian weight: with nodes xᵢ and weights wᵢ
    for the weight exp(-x²),
        ⟨σ⟩ = (1/√π) · Σᵢ wᵢ · σ(E₀ + √2·σ_E·xᵢ).

    Parameters
    ----------
    e_grid_ev : np.ndarray
        Cross-section energy grid in eV (ascending), e.g. MF3 `energies`.
    xs : np.ndarray
        Cross section on `e_grid_ev` (same length).
    energy_mev : float
        Kernel centroid (the datapoint's nominal incident energy) in MeV.
    sigma_E_mev : float
        Gaussian energy-resolution width in MeV (e.g. from `compute_sigma_E`).
    n_nodes : int
        Number of Gauss–Hermite nodes (default 12).

    Returns
    -------
    float
        Resolution-averaged cross section, same units as `xs`.
    """
    e0_ev = energy_mev * 1e6
    if sigma_E_mev <= 0.0 or n_nodes < 1:
        return float(np.interp(e0_ev, e_grid_ev, xs))
    nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)
    sample_e_ev = e0_ev + np.sqrt(2.0) * (sigma_E_mev * 1e6) * nodes
    sample_xs = np.interp(sample_e_ev, e_grid_ev, xs)
    return float(np.sum(weights * sample_xs) / np.sqrt(np.pi))


def find_bin_for_energy(
    energy_mev: float,
    energy_bins: list,  # List[EnergyBinInfo]
) -> Optional[int]:
    """
    Find the bin index containing a given energy.

    Searches through energy bins to find which bin contains the given energy.
    Returns None if the energy is outside all bins.

    Parameters
    ----------
    energy_mev : float
        Energy to look up (in MeV)
    energy_bins : List[EnergyBinInfo]
        List of energy bin objects with bin_lower_mev and bin_upper_mev attributes

    Returns
    -------
    Optional[int]
        Index of the containing bin (bin_info.index), or None if outside range
    """
    for bin_info in energy_bins:
        if bin_info.bin_lower_mev <= energy_mev <= bin_info.bin_upper_mev:
            return bin_info.index
    return None


def summarize_tof_parameters(
    tof_params_cache: Dict[str, Dict[str, Any]],
    subentries: list,
    default_flight_path_m: float = 27.037,
    default_time_resolution_ns: float = 5.0,
) -> Dict[str, Any]:
    """
    Generate summary statistics for TOF parameters across experiments.

    Parameters
    ----------
    tof_params_cache : Dict[str, Dict]
        Pre-loaded TOF parameters
    subentries : list
        List of subentry IDs to summarize
    default_flight_path_m : float
        Default flight path in meters
    default_time_resolution_ns : float
        Default time resolution in nanoseconds

    Returns
    -------
    Dict[str, Any]
        Summary statistics including:
        - n_from_file: Number with file-based parameters
        - n_default: Number using defaults
        - flight_paths: List of flight paths (m)
        - time_resolutions: List of time resolutions (ns)
    """
    n_from_file = 0
    n_default = 0
    flight_paths = []
    time_resolutions = []

    for subentry in subentries:
        params = get_tof_parameters(
            subentry=subentry,
            tof_params_cache=tof_params_cache,
            default_flight_path_m=default_flight_path_m,
            default_time_resolution_ns=default_time_resolution_ns,
        )

        if params.source == "file":
            n_from_file += 1
        else:
            n_default += 1

        flight_paths.append(params.flight_path_m)
        time_resolutions.append(params.time_resolution_ns)

    return {
        'n_from_file': n_from_file,
        'n_default': n_default,
        'flight_paths': flight_paths,
        'time_resolutions': time_resolutions,
        'mean_flight_path_m': np.mean(flight_paths) if flight_paths else 0.0,
        'mean_time_resolution_ns': np.mean(time_resolutions) if time_resolutions else 0.0,
    }
