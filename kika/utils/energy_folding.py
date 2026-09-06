"""
Energy folding utilities for differential cross section calculations.

This module provides tools for computing energy-folded differential cross sections,
accounting for TOF (Time-of-Flight) energy resolution effects in experimental data.

The folding can be applied to:
- Cross sections only (xs_only): Default mode, folds σ(E) with Gaussian kernel
- Angular distributions only (angular_only): Folds f(μ,E) shape with Gaussian kernel
- Both (both): Folds both XS and angular distribution

Theory:
    The measured differential cross section at energy E₀ with resolution σE is:

    dσ/dΩ(μ, E₀) = σ_folded(E₀) × f_folded(μ, E₀)

    where:
    - σ_folded = ∫ σ(E) × G(E; E₀, σE) dE / ∫ G(E; E₀, σE) dE
    - f_folded = ∫ f(μ, E) × G(E; E₀, σE) dE / ∫ G(E; E₀, σE) dE
    - G(E; E₀, σE) = exp(-0.5 × ((E - E₀) / σE)²) is the Gaussian kernel

TOF Energy Resolution:
    The energy resolution σE is computed from TOF parameters:
    1. FWHM_E = E × 2 × Δt / t  (TOF differential relation)
    2. σE = FWHM_E / 2.35482    (FWHM to standard deviation)

    IMPORTANT: delta_t_ns is interpreted as the FWHM of the time resolution,
    following the standard experimental convention used by GELINA, ORELA, n_TOF, etc.

Usage:
    >>> from kika.utils.energy_folding import (
    ...     EnergyFoldingConfig,
    ...     compute_energy_resolution_tof,
    ...     compute_folded_differential_xs,
    ... )
    >>>
    >>> # Create config with GELINA defaults (delta_t_ns is FWHM)
    >>> config = EnergyFoldingConfig(delta_t_ns=10.0, flight_path_m=27.037)
    >>>
    >>> # Compute σE at 1.406 MeV
    >>> sigma_E = compute_energy_resolution_tof(1.406, config)
    >>> print(f"σE = {sigma_E * 1000:.2f} keV")  # ~3.3 keV
    >>>
    >>> # Compute folded differential cross section
    >>> mu_grid, dsigma_domega, info = compute_folded_differential_xs(
    ...     ace_data=ace,
    ...     endf_coeffs=coeffs,  # ENDF format [a1, a2, ...]
    ...     target_energy_mev=1.406,
    ...     mt=2,
    ...     config=config,
    ...     mode="xs_only",  # or "angular_only" or "both"
    ... )
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import warnings

import numpy as np
from scipy.special import legendre

from kika._constants import (
    FWHM_TO_SIGMA as _FWHM_TO_SIGMA,
    NEUTRON_MASS_MEV,
    SPEED_OF_LIGHT_M_NS,
)
from kika.utils.numerics import fold_tabulated


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EnergyFoldingConfig:
    """
    Configuration for energy folding calculations.

    Attributes:
        flight_path_m: Flight path length in meters (default: GELINA 27.037 m)
        delta_t_ns: Time resolution FWHM in nanoseconds (default: 5.0 ns)
                    Note: This is the Full Width at Half Maximum of the time response,
                    following standard experimental convention (GELINA, ORELA, n_TOF).
        n_sigma: Number of sigma for integration window (default: 4.0)
        n_samples: Number of energy samples for folding integration (default: 21)
    """
    flight_path_m: float = 27.037  # GELINA default
    delta_t_ns: float = 5.0        # Time resolution FWHM (not sigma!)
    n_sigma: float = 4.0           # Integration window in sigma units
    n_samples: int = 21            # Energy samples for angular folding


# =============================================================================
# TOF Energy Resolution
# =============================================================================

# Re-exported from kika._constants so existing `from kika.utils import
# FWHM_TO_SIGMA` keeps working.  The value is now the exact 2*sqrt(2*ln 2)
# rather than the 6-digit 2.35482 this module used to carry.
FWHM_TO_SIGMA = _FWHM_TO_SIGMA


def tof_energy_resolution(
    energy_mev: float,
    *,
    flight_path_m: float,
    delta_t_ns: float,
    delta_t_is_fwhm: bool = True,
) -> float:
    r"""TOF energy resolution :math:`\sigma_E` in MeV.

    Time-of-flight relates energy spread to timing spread by
    :math:`\Delta E / E = 2\,\Delta t / t`, with
    :math:`t = L / v` and :math:`v = c\sqrt{2E/m_n}` (non-relativistic; the
    correction is below 1% up to ~10 MeV).

    **This function is the single definition of that formula.**  It previously
    existed five times across the repo in two mutually incompatible conventions
    differing by the factor :data:`FWHM_TO_SIGMA` — see ``delta_t_is_fwhm``.

    Parameters
    ----------
    energy_mev : float
        Neutron energy in MeV.
    flight_path_m : float
        Flight path in metres.
    delta_t_ns : float
        Timing spread in nanoseconds.  See ``delta_t_is_fwhm`` for what it means.
    delta_t_is_fwhm : bool, default True
        Whether ``delta_t_ns`` is a **FWHM** (the usual experimental convention:
        pulse widths and detector timing are normally quoted that way) or
        already a standard deviation.  When True the result is divided by
        :data:`FWHM_TO_SIGMA`.  There is no safe default that suits both
        readings, so callers that care should pass this explicitly.

    Returns
    -------
    float
        :math:`\sigma_E` in MeV (a standard deviation, either way).
    """
    if energy_mev <= 0.0:
        return 0.0
    velocity_m_per_ns = SPEED_OF_LIGHT_M_NS * np.sqrt(2.0 * energy_mev / NEUTRON_MASS_MEV)
    t_ns = flight_path_m / velocity_m_per_ns
    width_mev = energy_mev * 2.0 * delta_t_ns / t_ns
    return width_mev / FWHM_TO_SIGMA if delta_t_is_fwhm else width_mev


def compute_energy_resolution_tof(
    energy_mev: float,
    config: Optional[EnergyFoldingConfig] = None,
    *,
    flight_path_m: Optional[float] = None,
    delta_t_ns: Optional[float] = None,
) -> float:
    """
    Calculate TOF energy resolution σE in MeV.

    The energy resolution for time-of-flight measurements is derived from:
        ΔE/E = 2 × Δt/t  (TOF differential relation)

    where Δt is the FWHM of the time resolution (standard experimental convention).

    This function:
    1. Computes the FWHM of energy resolution: FWHM_E = E × 2 × Δt / t
    2. Converts to standard deviation: σE = FWHM_E / 2.35482

    Parameters:
        energy_mev: Neutron energy in MeV
        config: EnergyFoldingConfig with TOF parameters (optional)
        flight_path_m: Flight path in meters (overrides config if provided)
        delta_t_ns: Time resolution FWHM in nanoseconds (overrides config if provided)
                    Note: This should be the FWHM of the time response, NOT sigma.

    Returns:
        Energy resolution σE (standard deviation) in MeV

    Examples:
        >>> # Using config
        >>> config = EnergyFoldingConfig(flight_path_m=27.037, delta_t_ns=10.0)
        >>> sigma_E = compute_energy_resolution_tof(1.406, config)
        >>> print(f"σE = {sigma_E * 1000:.2f} keV")  # ~3.3 keV

        >>> # Using explicit parameters
        >>> sigma_E = compute_energy_resolution_tof(1.0, flight_path_m=27.037, delta_t_ns=5.0)
    """
    # Deprecated shim over tof_energy_resolution, which is now the single
    # definition of this formula.  Kept because six workspace notebooks import
    # this name directly.  Behaviour is unchanged (delta_t_ns is a FWHM here).
    if config is None:
        config = EnergyFoldingConfig()

    L = flight_path_m if flight_path_m is not None else config.flight_path_m
    dt = delta_t_ns if delta_t_ns is not None else config.delta_t_ns

    return tof_energy_resolution(
        energy_mev, flight_path_m=L, delta_t_ns=dt, delta_t_is_fwhm=True,
    )


# =============================================================================
# Cross Section Folding
# =============================================================================

def fold_cross_section(
    ace_data,
    target_energy_mev: float,
    mt: int,
    sigma_E_mev: float,
    n_sigma: float = 4.0,
) -> Tuple[float, float]:
    """
    Compute energy-folded cross section using ACE data.

    Thin ACE adapter over :func:`kika.utils.numerics.fold_tabulated`:

        σ_folded = ∫ σ(E) × G(E; E₀, σE) dE / ∫ G(E; E₀, σE) dE

    .. versionchanged::
        This used to average the *tabulated points* weighted by the Gaussian,
        with no dE measure.  An ACE energy grid is adaptively refined — densest
        exactly at resonance peaks — so that over-weighted the peaks: on the
        JEFF-4.0 Fe-56 elastic it differed from the integral by 8.6% on average
        and 21.4% at worst.  It now integrates the interpolant by Gauss-Hermite
        quadrature, which is insensitive to how the input is sampled.  **Folded
        values from this function change accordingly.**

    Parameters:
        ace_data: ACE data object from kika.read_ace()
        target_energy_mev: Central energy for folding (MeV)
        mt: Reaction MT number (e.g., 2 for elastic)
        sigma_E_mev: Energy resolution σE (MeV)
        n_sigma: Ignored, kept for backward compatibility.  Gauss-Hermite
            quadrature has no truncation window to size.

    Returns:
        Tuple of (folded_xs, unfolded_xs) in barns
    """
    # Get cross section data from ACE
    xs_data = ace_data.cross_section.to_plot_data(mt=mt)
    energies_mev = np.asarray(xs_data.x)
    xs_values = np.asarray(xs_data.y)

    folded_xs = fold_tabulated(energies_mev, xs_values, target_energy_mev, sigma_E_mev)
    unfolded_xs = float(np.interp(target_energy_mev, energies_mev, xs_values))

    return folded_xs, unfolded_xs


# =============================================================================
# Angular Distribution Functions
# =============================================================================

def endf_angular_distribution(
    mu: np.ndarray,
    a_coeffs: np.ndarray,
) -> np.ndarray:
    """
    Compute normalized angular distribution from ENDF Legendre coefficients.
    
    ENDF format:
        f(μ) = (1/2) × Σ_{l=0}^{L} (2l+1) × a_l × P_l(μ)
    
    where a_0 = 1 (implicit) and a_coeffs = [a_1, a_2, ..., a_L]
    
    The result integrates to 1 over μ ∈ [-1, 1].
    
    Parameters:
        mu: Cosine of scattering angle (array)
        a_coeffs: ENDF coefficients [a_1, a_2, ..., a_L] (a_0=1 implicit)
    
    Returns:
        Angular distribution f(μ) (normalized PDF)
    """
    mu = np.atleast_1d(mu)
    
    # P_0(μ) = 1, a_0 = 1 by convention
    result = 0.5 * np.ones_like(mu, dtype=float)  # (2×0+1)/2 × 1 × P_0 = 0.5
    
    # Add higher orders: l = 1, 2, ..., L
    for l, a_l in enumerate(a_coeffs, start=1):
        P_l = legendre(l)(mu)
        result += 0.5 * (2 * l + 1) * a_l * P_l
    
    return result


def fold_angular_distribution(
    ace_data,
    target_energy_mev: float,
    mt: int,
    sigma_E_mev: float,
    n_sigma: float = 4.0,
    n_samples: int = 21,
    num_mu_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute energy-folded angular distribution shape using ACE data.
    
    This folds the angular distribution PDF (shape only, not cross section).
    
    Parameters:
        ace_data: ACE data object from kika.read_ace()
        target_energy_mev: Central energy for folding (MeV)
        mt: Reaction MT number
        sigma_E_mev: Energy resolution σE (MeV)
        n_sigma: Integration window in sigma units
        n_samples: Number of energy samples for integration
        num_mu_points: Number of μ grid points
    
    Returns:
        Tuple of (mu_grid, folded_pdf, unfolded_pdf)
    """
    # Get energy bounds from ACE
    e_min_ace = np.min(ace_data.energies)
    e_max_ace = np.max(ace_data.energies)
    
    # Define window
    E_lo = max(target_energy_mev - n_sigma * sigma_E_mev, e_min_ace)
    E_hi = min(target_energy_mev + n_sigma * sigma_E_mev, e_max_ace)
    
    # Sample energies
    sample_energies = np.linspace(E_lo, E_hi, n_samples)
    
    # Gaussian weights
    weights = np.exp(-0.5 * ((sample_energies - target_energy_mev) / sigma_E_mev) ** 2)
    weights /= weights.sum()
    
    # Accumulate weighted angular distributions
    mu_grid = np.linspace(-1, 1, num_mu_points)
    folded_pdf = np.zeros(num_mu_points)
    
    for E_sample, w in zip(sample_energies, weights):
        # Get angular distribution at this energy
        plot_data = ace_data.angular_distributions.to_plot_data(
            mt=mt,
            energy=E_sample,
            ace=ace_data,
            interpolate=True,
            num_points=num_mu_points,
            normalize_to_xs=False,  # PDF only
        )
        # Ensure proper sorting for interpolation (np.interp requires increasing xp)
        ace_mu = np.asarray(plot_data.x)
        ace_pdf = np.asarray(plot_data.y)
        sort_idx = np.argsort(ace_mu)
        folded_pdf += w * np.interp(mu_grid, ace_mu[sort_idx], ace_pdf[sort_idx])
    
    # Unfolded PDF at target energy
    unfolded_data = ace_data.angular_distributions.to_plot_data(
        mt=mt,
        energy=target_energy_mev,
        ace=ace_data,
        interpolate=True,
        num_points=num_mu_points,
        normalize_to_xs=False,
    )
    # Ensure proper sorting for interpolation
    ace_mu_unf = np.asarray(unfolded_data.x)
    ace_pdf_unf = np.asarray(unfolded_data.y)
    sort_idx_unf = np.argsort(ace_mu_unf)
    unfolded_pdf = np.interp(mu_grid, ace_mu_unf[sort_idx_unf], ace_pdf_unf[sort_idx_unf])
    
    return mu_grid, folded_pdf, unfolded_pdf


# =============================================================================
# Main Entry Point
# =============================================================================

FoldingMode = Literal["xs_only", "angular_only", "both"]


def compute_folded_differential_xs(
    ace_data,
    target_energy_mev: float,
    mt: int,
    config: Optional[EnergyFoldingConfig] = None,
    *,
    mode: FoldingMode = "xs_only",
    endf_coeffs: Optional[np.ndarray] = None,
    sigma_E_mev: Optional[float] = None,
    num_mu_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute energy-folded differential cross section dσ/dΩ(μ).
    
    This is the main entry point for computing differential cross sections
    with TOF energy resolution effects.
    
    Modes:
        - "xs_only" (default): Fold cross section only, use unfolded angular shape
        - "angular_only": Fold angular distribution only, use unfolded XS
        - "both": Fold both cross section and angular distribution
    
    Parameters:
        ace_data: ACE data object from kika.read_ace()
        target_energy_mev: Central energy for folding (MeV)
        mt: Reaction MT number (e.g., 2 for elastic scattering)
        config: EnergyFoldingConfig with TOF parameters (uses defaults if None)
        mode: Folding mode - "xs_only", "angular_only", or "both"
        endf_coeffs: ENDF Legendre coefficients [a_1, ..., a_L] for custom angular
                     distribution. If None, uses ACE angular distributions.
        sigma_E_mev: Override energy resolution (computed from TOF if None)
        num_mu_points: Number of μ grid points for output
    
    Returns:
        Tuple of:
        - mu_grid: cos(θ) values, shape (num_mu_points,)
        - dsigma_domega: dσ/dΩ in b/sr, shape (num_mu_points,)
        - info: Dict with diagnostic information:
            - sigma_E_mev: Energy resolution used
            - folded_xs: Folded cross section (barns)
            - unfolded_xs: Unfolded cross section (barns)
            - xs_ratio: folded_xs / unfolded_xs
            - mode: Folding mode used
    
    Examples:
        >>> import kika
        >>> ace = kika.read_ace("path/to/file.ace")
        >>> 
        >>> # XS-only folding (default, recommended)
        >>> mu, dsigma, info = compute_folded_differential_xs(
        ...     ace, target_energy_mev=1.0, mt=2
        ... )
        >>> 
        >>> # Full folding with custom ENDF coefficients
        >>> mu, dsigma, info = compute_folded_differential_xs(
        ...     ace, target_energy_mev=1.0, mt=2,
        ...     mode="both",
        ...     endf_coeffs=[0.1, 0.05, 0.02],  # a1, a2, a3
        ... )
    """
    if config is None:
        config = EnergyFoldingConfig()
    
    # Compute energy resolution if not provided
    if sigma_E_mev is None:
        sigma_E_mev = compute_energy_resolution_tof(target_energy_mev, config)
    
    # Create μ grid
    mu_grid = np.linspace(-1, 1, num_mu_points)
    
    # Get cross sections (always needed for normalization)
    if mode in ("xs_only", "both"):
        folded_xs, unfolded_xs = fold_cross_section(
            ace_data, target_energy_mev, mt, sigma_E_mev, config.n_sigma
        )
    else:
        # angular_only: no XS folding
        _, unfolded_xs = fold_cross_section(
            ace_data, target_energy_mev, mt, sigma_E_mev, config.n_sigma
        )
        folded_xs = unfolded_xs
    
    # Select which XS to use for normalization
    xs_for_norm = folded_xs if mode in ("xs_only", "both") else unfolded_xs
    
    # Get angular distribution
    if mode in ("angular_only", "both"):
        # Fold angular distribution
        _, angular_pdf, _ = fold_angular_distribution(
            ace_data, target_energy_mev, mt, sigma_E_mev,
            config.n_sigma, config.n_samples, num_mu_points
        )
    else:
        # xs_only: use unfolded angular distribution
        if endf_coeffs is not None:
            # Use provided ENDF coefficients
            angular_pdf = endf_angular_distribution(mu_grid, endf_coeffs)
        else:
            # Use ACE angular distribution at target energy
            plot_data = ace_data.angular_distributions.to_plot_data(
                mt=mt,
                energy=target_energy_mev,
                ace=ace_data,
                interpolate=True,
                num_points=num_mu_points,
                normalize_to_xs=False,
            )
            # Ensure proper sorting for interpolation (np.interp requires increasing xp)
            ace_mu = np.asarray(plot_data.x)
            ace_pdf = np.asarray(plot_data.y)
            sort_idx = np.argsort(ace_mu)
            angular_pdf = np.interp(mu_grid, ace_mu[sort_idx], ace_pdf[sort_idx])
    
    # Compute differential cross section: dσ/dΩ = σ × f(μ) / (4π)
    # Since f(μ) integrates to 1 over μ ∈ [-1,1], and solid angle integral is 4π,
    # we have: dσ/dΩ = σ × f(μ) / (2) to get b/sr
    # Actually, f(μ) is per unit μ, so dσ/dΩ = σ × f(μ) / (2π)
    # Note: The factor depends on the normalization convention.
    # For ENDF: ∫ f(μ) dμ = 1, and dσ/dΩ = σ × f(μ) / (2π) gives b/sr
    dsigma_domega = xs_for_norm * angular_pdf / (2 * np.pi)
    
    # Build info dict
    info = {
        "sigma_E_mev": sigma_E_mev,
        "sigma_E_kev": sigma_E_mev * 1000,
        "folded_xs": folded_xs,
        "unfolded_xs": unfolded_xs,
        "xs_ratio": folded_xs / unfolded_xs if unfolded_xs > 0 else 1.0,
        "mode": mode,
        "config": config,
    }
    
    return mu_grid, dsigma_domega, info


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_unfolded_differential_xs(
    ace_data,
    target_energy_mev: float,
    mt: int,
    endf_coeffs: Optional[np.ndarray] = None,
    num_mu_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute unfolded (raw) differential cross section dσ/dΩ(μ).
    
    This is equivalent to calling compute_folded_differential_xs with
    σE = 0, but more efficient as it skips the folding calculations.
    
    Parameters:
        ace_data: ACE data object from kika.read_ace()
        target_energy_mev: Energy in MeV
        mt: Reaction MT number
        endf_coeffs: Optional ENDF coefficients [a_1, ..., a_L]
        num_mu_points: Number of μ grid points
    
    Returns:
        Tuple of (mu_grid, dsigma_domega, info)
    """
    mu_grid = np.linspace(-1, 1, num_mu_points)
    
    # Get cross section at target energy
    xs_data = ace_data.cross_section.to_plot_data(mt=mt)
    xs = np.interp(target_energy_mev, xs_data.x, xs_data.y)
    
    # Get angular distribution
    if endf_coeffs is not None:
        angular_pdf = endf_angular_distribution(mu_grid, endf_coeffs)
    else:
        plot_data = ace_data.angular_distributions.to_plot_data(
            mt=mt,
            energy=target_energy_mev,
            ace=ace_data,
            interpolate=True,
            num_points=num_mu_points,
            normalize_to_xs=False,
        )
        # Ensure proper sorting for interpolation (np.interp requires increasing xp)
        ace_mu = np.asarray(plot_data.x)
        ace_pdf = np.asarray(plot_data.y)
        sort_idx = np.argsort(ace_mu)
        angular_pdf = np.interp(mu_grid, ace_mu[sort_idx], ace_pdf[sort_idx])
    
    dsigma_domega = xs * angular_pdf / (2 * np.pi)
    
    info = {
        "sigma_E_mev": 0.0,
        "sigma_E_kev": 0.0,
        "folded_xs": xs,
        "unfolded_xs": xs,
        "xs_ratio": 1.0,
        "mode": "unfolded",
    }
    
    return mu_grid, dsigma_domega, info
