"""Generic multigroup collapse operations for covariance matrices.

Format-independent utilities for energy-group rebinning, covariance
collapse, and relative/absolute conversion.  These are used by
Legendre-specific orchestration in ``kika.cov.multigroup.collapse``
and can be reused for any multigroup covariance collapse.
"""

import numpy as np
from typing import Union, Callable
from scipy import special


__all__ = [
    "WeightingFunction",
    "compute_rebin_operator",
    "collapse_covariance",
    "relative_to_absolute",
    "absolute_to_relative",
    # Backward compatibility aliases
    "compute_energy_rebin_operator",
    "map_covariance_matrix",
    "convert_relative_to_absolute_covariance",
    "convert_absolute_to_relative_covariance",
]


class WeightingFunction:
    """
    Class to represent various weighting functions phi(E) for multigroup collapse.
    """

    @staticmethod
    def constant(energy: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Constant weighting function phi(E) = 1."""
        if isinstance(energy, np.ndarray):
            return np.ones_like(energy)
        return 1.0

    @staticmethod
    def constant_antiderivative(energy: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Antiderivative of constant weighting: Phi(E) = E."""
        return energy

    @staticmethod
    def maxwellian(energy: Union[float, np.ndarray], temperature: float = 2.53e-2) -> Union[float, np.ndarray]:
        """Maxwellian spectrum phi(E) = sqrt(E) * exp(-E/kT)."""
        kT = temperature  # in eV, default = 0.0253 eV (room temperature)
        return np.sqrt(energy) * np.exp(-energy / kT)

    @staticmethod
    def maxwellian_antiderivative(energy: Union[float, np.ndarray], temperature: float = 2.53e-2) -> Union[float, np.ndarray]:
        """Antiderivative of Maxwellian: ∫ sqrt(E) e^{-E/kT} dE = (kT)^{3/2} Γ(3/2) * gammainc(3/2, E/kT)."""
        kT = temperature
        x = np.asarray(energy) / kT
        return (kT ** 1.5) * special.gamma(1.5) * special.gammainc(1.5, x)

    @staticmethod
    def fission_spectrum(energy: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Simplified fission spectrum phi(E) = sqrt(E) * exp(-E/1.29e6)."""
        return np.sqrt(energy) * np.exp(-energy / 1.29e6)  # 1.29 MeV average

    @staticmethod
    def fission_spectrum_antiderivative(energy: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Antiderivative of fission spectrum with kT≈1.29 MeV."""
        kT = 1.29e6  # 1.29 MeV in eV
        x = np.asarray(energy) / kT
        return (kT ** 1.5) * special.gamma(1.5) * special.gammainc(1.5, x)

    # --- New lethargy (phi=1/E) weighting for log-energy (flat in lethargy) averaging ---
    @staticmethod
    def lethargy(energy: Union[float, np.ndarray], epsilon: float = 1e-30) -> Union[float, np.ndarray]:
        """Lethargy weighting function phi(E)=1/E (flat per unit lethargy).

        This produces group averages consistent with a uniform distribution in lethargy
        u = ln(E). A tiny epsilon prevents division by zero for any accidental
        non-positive energies.
        """
        if isinstance(energy, np.ndarray):
            return 1.0 / np.maximum(energy, epsilon)
        return 1.0 / max(energy, epsilon)

    @staticmethod
    def lethargy_antiderivative(energy: Union[float, np.ndarray], epsilon: float = 1e-30) -> Union[float, np.ndarray]:
        """Antiderivative of 1/E: Phi(E)=ln(E). Safe for vector/scalar inputs."""
        if isinstance(energy, np.ndarray):
            return np.log(np.maximum(energy, epsilon))
        return float(np.log(max(energy, epsilon)))


def compute_rebin_operator(coarse_energy_grid: np.ndarray,
                           fine_energy_grid: np.ndarray,
                           phi_func: Callable = WeightingFunction.constant,
                           phi_antiderivative: Callable = WeightingFunction.constant_antiderivative) -> np.ndarray:
    """
    Build the energy-rebin operator M mapping coarse bins to fine bins.

    This creates a row-stochastic matrix M where M[g,h] represents the contribution
    of coarse energy bin h to fine energy bin g, weighted by the spectrum.

    M_{g,h} = ∫(E∈(g∩h)) w(E) dE / ∫(E∈g) w(E) dE

    Parameters
    ----------
    coarse_energy_grid : np.ndarray
        Coarse energy grid boundaries (N_coarse + 1 edges)
    fine_energy_grid : np.ndarray
        Fine target energy grid boundaries (N_fine + 1 edges)
    phi_func : Callable
        Weighting function w(E)
    phi_antiderivative : Callable
        Antiderivative of weighting function

    Returns
    -------
    np.ndarray
        Row-stochastic rebin matrix M of shape (N_fine, N_coarse)
    """
    # Vectorized overlap computation using broadcasting
    E_g_lo = fine_energy_grid[:-1, None]      # (N_fine, 1)
    E_g_hi = fine_energy_grid[1:, None]       # (N_fine, 1)
    E_h_lo = coarse_energy_grid[None, :-1]    # (1, N_coarse)
    E_h_hi = coarse_energy_grid[None, 1:]     # (1, N_coarse)

    lower = np.maximum(E_g_lo, E_h_lo)       # (N_fine, N_coarse)
    upper = np.minimum(E_g_hi, E_h_hi)       # (N_fine, N_coarse)
    has_overlap = upper > lower

    # Compute numerator: ∫(E∈(g∩h)) w(E) dE
    numer = np.zeros_like(lower)
    if np.any(has_overlap):
        numer[has_overlap] = (phi_antiderivative(upper[has_overlap])
                              - phi_antiderivative(lower[has_overlap]))

    # Compute denominator: ∫(E∈g) w(E) dE, shape (N_fine,)
    denom = phi_antiderivative(fine_energy_grid[1:]) - phi_antiderivative(fine_energy_grid[:-1])
    valid_denom = np.abs(denom) >= 1e-15

    M = np.zeros_like(numer)
    M[valid_denom] = numer[valid_denom] / denom[valid_denom, None]

    return M


def collapse_covariance(coarse_matrix: np.ndarray,
                        rebin_operator: np.ndarray) -> np.ndarray:
    """
    Collapse a covariance matrix from a coarse grid to a fine grid using a rebin operator.

    Performs the congruence transform: C_fine = M @ C_coarse @ M^T

    Parameters
    ----------
    coarse_matrix : np.ndarray
        Covariance matrix on the coarse energy grid
    rebin_operator : np.ndarray
        Energy rebin operator M of shape (N_fine, N_coarse)

    Returns
    -------
    np.ndarray
        Covariance matrix on the fine energy grid
    """
    return rebin_operator @ coarse_matrix @ rebin_operator.T


def relative_to_absolute(relative_matrix: np.ndarray,
                         means_row: np.ndarray,
                         means_col: np.ndarray) -> np.ndarray:
    """
    Convert a relative covariance matrix to absolute covariance.

    C_abs = diag(means_row) @ C_rel @ diag(means_col)

    Parameters
    ----------
    relative_matrix : np.ndarray
        Relative covariance matrix
    means_row : np.ndarray
        Mean values for the row dimension
    means_col : np.ndarray
        Mean values for the column dimension

    Returns
    -------
    np.ndarray
        Absolute covariance matrix
    """
    return np.diag(means_row) @ relative_matrix @ np.diag(means_col)


def absolute_to_relative(absolute_matrix: np.ndarray,
                         means_row: np.ndarray,
                         means_col: np.ndarray,
                         epsilon: float = 1e-15) -> np.ndarray:
    """
    Convert an absolute covariance matrix to relative covariance.

    C_rel = diag(means_row)^{-1} @ C_abs @ diag(means_col)^{-1}

    Elements where the corresponding mean is smaller than *epsilon* are
    set to NaN to flag the ill-defined division.

    Parameters
    ----------
    absolute_matrix : np.ndarray
        Absolute covariance matrix
    means_row : np.ndarray
        Mean values for the row dimension
    means_col : np.ndarray
        Mean values for the column dimension
    epsilon : float
        Small value to prevent division by zero

    Returns
    -------
    np.ndarray
        Relative covariance matrix
    """
    # Create inverse diagonal matrices with epsilon protection
    means_row_safe = np.where(np.abs(means_row) > epsilon, means_row, epsilon)
    means_col_safe = np.where(np.abs(means_col) > epsilon, means_col, epsilon)

    inv_diag_row = np.diag(1.0 / means_row_safe)
    inv_diag_col = np.diag(1.0 / means_col_safe)

    relative = inv_diag_row @ absolute_matrix @ inv_diag_col

    # Set elements to NaN where original means were too small
    mask_row = np.abs(means_row) <= epsilon
    mask_col = np.abs(means_col) <= epsilon

    bad_mask = mask_row[:, None] | mask_col[None, :]
    relative[bad_mask] = np.nan

    return relative


# Backward compatibility aliases
compute_energy_rebin_operator = compute_rebin_operator
map_covariance_matrix = collapse_covariance
convert_relative_to_absolute_covariance = relative_to_absolute
convert_absolute_to_relative_covariance = absolute_to_relative
