"""
Adaptive Multigroup Covariance Collapse Module.

This module provides functions for collapsing fine-grid Legendre coefficient
covariance matrices into coarser multigroup representations while preserving
the correlation structure.

Key features:
- Uses l=1 (first Legendre order) correlation structure for grouping decisions
- Applies the same energy grid to all Legendre orders (avoids MF34 complications)
- Linear transformation: C_y = A @ C_x @ A.T

Author: Generated for kika project
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GroupInfo:
    """Diagnostics for a single multigroup energy bin."""
    group_index: int
    fine_indices: List[int]
    energy_lower_mev: float
    energy_upper_mev: float
    width_mev: float
    min_correlation_l1: float  # Min adjacent correlation for l=1 within group
    sigma_ratio: float         # max(sigma_l1)/min(sigma_l1) within group
    n_fine_bins: int


@dataclass
class MultigroupResult:
    """Result from adaptive multigroup collapse."""
    group_boundaries_mev: np.ndarray      # (N_groups + 1,) boundaries in MeV
    group_boundaries_ev: np.ndarray       # Same in eV
    aggregation_matrix: np.ndarray        # (N_groups * L, N_fine * L)
    cov_grouped: np.ndarray               # Grouped covariance matrix
    corr_grouped: np.ndarray              # Grouped correlation matrix
    mean_grouped: np.ndarray              # Grouped mean coefficients
    groups: List[List[int]]               # Fine bin indices per group
    group_info: List[GroupInfo]           # Diagnostics per group


# =============================================================================
# INDEX FUNCTIONS
# =============================================================================

def idx(energy_idx: int, order: int, max_order: int) -> int:
    """
    Parameter index in the flattened coefficient/covariance array.

    Layout: [a1(E0), a2(E0), ..., aL(E0), a1(E1), a2(E1), ..., aL(E1), ...]

    Parameters
    ----------
    energy_idx : int
        Index of the energy bin (0-indexed)
    order : int
        Legendre order (1-indexed, from 1 to max_order)
    max_order : int
        Maximum Legendre order (L)

    Returns
    -------
    int
        Index in the flattened array
    """
    return energy_idx * max_order + (order - 1)


def idx_to_energy_order(flat_idx: int, max_order: int) -> Tuple[int, int]:
    """
    Inverse of idx(): get (energy_idx, order) from flattened index.

    Parameters
    ----------
    flat_idx : int
        Index in the flattened array
    max_order : int
        Maximum Legendre order (L)

    Returns
    -------
    Tuple[int, int]
        (energy_idx, order) where order is 1-indexed
    """
    energy_idx = flat_idx // max_order
    order = (flat_idx % max_order) + 1
    return energy_idx, order


# =============================================================================
# GROUPING ALGORITHM
# =============================================================================

def find_adaptive_group_boundaries(
    corr_matrix: np.ndarray,
    cov_matrix: np.ndarray,
    fine_energies_mev: np.ndarray,
    sigma_E_mev: np.ndarray,
    fine_bin_widths_mev: np.ndarray,
    max_order: int,
    rho_min: float = 0.90,
    sigma_ratio_max: float = 1.7,
    min_width_factor: float = 2.0,
    logger=None,
) -> Tuple[List[List[int]], List[GroupInfo]]:
    """
    Find group boundaries using l=1 correlation structure only.

    Algorithm (greedy merging):
    1. Start at i=0
    2. Extend group [i0, i1] while:
       - Adjacent l=1 correlation: corr[idx(i1,1), idx(i1+1,1)] >= rho_min
       - Sigma ratio for l=1 in [i0..i1+1]: max/min <= sigma_ratio_max
       - OR Width < min_width_factor * median(sigma_E) (force merge for resolution)
    3. Finalize group, start next at i1+1

    Parameters
    ----------
    corr_matrix : np.ndarray
        Full correlation matrix, shape (n_fine * max_order, n_fine * max_order)
    cov_matrix : np.ndarray
        Full covariance matrix, same shape
    fine_energies_mev : np.ndarray
        Energy of each fine bin in MeV, shape (n_fine,)
    sigma_E_mev : np.ndarray
        Energy resolution for each fine bin in MeV, shape (n_fine,)
    fine_bin_widths_mev : np.ndarray
        Width of each fine bin in MeV, shape (n_fine,)
    max_order : int
        Maximum Legendre order (L)
    rho_min : float
        Minimum correlation to allow merging (default 0.90)
    sigma_ratio_max : float
        Maximum sigma ratio within group (default 1.7)
    min_width_factor : float
        Minimum group width as multiple of median sigma_E (default 2.0)
    logger : optional
        Logger for diagnostics

    Returns
    -------
    Tuple[List[List[int]], List[GroupInfo]]
        groups: List of lists, each containing fine bin indices for a group
        group_info: Diagnostic information for each group
    """
    n_fine = len(fine_energies_mev)

    # Handle edge case: sigma_E might have zeros
    valid_sigma_E = sigma_E_mev[sigma_E_mev > 0]
    if len(valid_sigma_E) > 0:
        median_sigma_E = np.median(valid_sigma_E)
    else:
        # Fallback: use 1% of mean energy
        median_sigma_E = 0.01 * np.mean(fine_energies_mev)

    min_width = min_width_factor * median_sigma_E

    if logger:
        logger.info(f"  Grouping parameters:")
        logger.info(f"    rho_min = {rho_min}")
        logger.info(f"    sigma_ratio_max = {sigma_ratio_max}")
        logger.info(f"    min_width_factor = {min_width_factor}")
        logger.info(f"    median(sigma_E) = {median_sigma_E:.4f} MeV")
        logger.info(f"    min_width = {min_width:.4f} MeV")

    groups = []
    group_info_list = []

    i0 = 0
    while i0 < n_fine:
        i1 = i0

        # Track group properties
        min_corr_in_group = 1.0

        while i1 + 1 < n_fine:
            # Check l=1 adjacent correlation
            idx_i1 = idx(i1, 1, max_order)
            idx_i1_next = idx(i1 + 1, 1, max_order)
            rho = corr_matrix[idx_i1, idx_i1_next]

            # Check l=1 sigma ratio in [i0, i1+1]
            sigmas_l1 = []
            for i in range(i0, i1 + 2):
                var_i = cov_matrix[idx(i, 1, max_order), idx(i, 1, max_order)]
                if var_i > 0:
                    sigmas_l1.append(np.sqrt(var_i))

            if len(sigmas_l1) >= 2:
                ratio = max(sigmas_l1) / min(sigmas_l1)
            else:
                ratio = 1.0

            # Check current group width
            current_width = sum(fine_bin_widths_mev[i0:i1 + 2])

            # Decision logic
            should_extend = False

            if rho >= rho_min and ratio <= sigma_ratio_max:
                # Both criteria met - extend
                should_extend = True
                min_corr_in_group = min(min_corr_in_group, rho)
            elif current_width < min_width:
                # Force merge for minimum resolution
                should_extend = True
                min_corr_in_group = min(min_corr_in_group, rho)
            else:
                # Stop extending
                should_extend = False

            if should_extend:
                i1 += 1
            else:
                break

        # Finalize this group
        group_indices = list(range(i0, i1 + 1))
        groups.append(group_indices)

        # Compute final group diagnostics
        group_width = sum(fine_bin_widths_mev[i0:i1 + 1])

        # Final sigma ratio for the group
        sigmas_l1_final = []
        for i in group_indices:
            var_i = cov_matrix[idx(i, 1, max_order), idx(i, 1, max_order)]
            if var_i > 0:
                sigmas_l1_final.append(np.sqrt(var_i))

        if len(sigmas_l1_final) >= 2:
            final_ratio = max(sigmas_l1_final) / min(sigmas_l1_final)
        else:
            final_ratio = 1.0

        # For single-bin groups, min_corr is undefined (set to 1.0)
        if len(group_indices) == 1:
            min_corr_in_group = 1.0

        info = GroupInfo(
            group_index=len(groups) - 1,
            fine_indices=group_indices,
            energy_lower_mev=fine_energies_mev[i0],
            energy_upper_mev=fine_energies_mev[i1],
            width_mev=group_width,
            min_correlation_l1=min_corr_in_group,
            sigma_ratio=final_ratio,
            n_fine_bins=len(group_indices),
        )
        group_info_list.append(info)

        # Move to next group
        i0 = i1 + 1

    if logger:
        logger.info(f"  Found {len(groups)} multigroups from {n_fine} fine bins")
        for info in group_info_list:
            logger.info(
                f"    Group {info.group_index}: bins {info.fine_indices[0]}-{info.fine_indices[-1]} "
                f"({info.n_fine_bins} bins), "
                f"E=[{info.energy_lower_mev:.4f}, {info.energy_upper_mev:.4f}] MeV, "
                f"min_rho_l1={info.min_correlation_l1:.3f}, "
                f"sigma_ratio={info.sigma_ratio:.2f}"
            )

    return groups, group_info_list


# =============================================================================
# AGGREGATION MATRIX
# =============================================================================

def build_aggregation_matrix(
    groups: List[List[int]],
    fine_bin_widths_mev: np.ndarray,
    n_fine: int,
    max_order: int,
) -> np.ndarray:
    """
    Build aggregation matrix A for y = A @ x, C_y = A @ C_x @ A.T.

    A is block-diagonal: same grouping applied to each Legendre order.
    Weights: w_i = bin_width_i (energy interval average definition).

    Layout matches fine covariance: [a1(E1), a2(E1), ..., aL(E1), a1(E2), ...]

    For group g containing fine bins {i1, i2, ...}:
        A[g*L + (l-1), i*L + (l-1)] = w_i / sum(w_j for j in group)

    Parameters
    ----------
    groups : List[List[int]]
        List of groups, each containing fine bin indices
    fine_bin_widths_mev : np.ndarray
        Width of each fine bin in MeV, shape (n_fine,)
    n_fine : int
        Number of fine energy bins
    max_order : int
        Maximum Legendre order (L)

    Returns
    -------
    np.ndarray
        Aggregation matrix A, shape (n_groups * max_order, n_fine * max_order)
    """
    n_groups = len(groups)
    n_fine_params = n_fine * max_order
    n_group_params = n_groups * max_order

    A = np.zeros((n_group_params, n_fine_params))

    for g_idx, fine_indices in enumerate(groups):
        # Compute normalization (sum of bin widths in this group)
        total_width = sum(fine_bin_widths_mev[i] for i in fine_indices)

        if total_width <= 0:
            # Fallback: equal weighting
            total_width = len(fine_indices)
            weights = {i: 1.0 / total_width for i in fine_indices}
        else:
            weights = {i: fine_bin_widths_mev[i] / total_width for i in fine_indices}

        # Fill in weights for each Legendre order
        for order in range(1, max_order + 1):
            row_idx = idx(g_idx, order, max_order)

            for fine_i in fine_indices:
                col_idx = idx(fine_i, order, max_order)
                A[row_idx, col_idx] = weights[fine_i]

    return A


# =============================================================================
# ENERGY BOUNDARIES
# =============================================================================

def construct_group_energy_boundaries(
    groups: List[List[int]],
    fine_bin_lower_mev: np.ndarray,
    fine_bin_upper_mev: np.ndarray,
) -> np.ndarray:
    """
    Construct N_groups + 1 boundary points for MF34.

    boundary[0] = fine_bin_lower[groups[0][0]]
    boundary[g] = fine_bin_lower[groups[g][0]]  for g > 0
    boundary[N] = fine_bin_upper[groups[-1][-1]]

    Parameters
    ----------
    groups : List[List[int]]
        List of groups, each containing fine bin indices
    fine_bin_lower_mev : np.ndarray
        Lower boundary of each fine bin in MeV, shape (n_fine,)
    fine_bin_upper_mev : np.ndarray
        Upper boundary of each fine bin in MeV, shape (n_fine,)

    Returns
    -------
    np.ndarray
        Group boundaries in MeV, shape (n_groups + 1,)
    """
    n_groups = len(groups)
    boundaries = np.zeros(n_groups + 1)

    for g_idx, fine_indices in enumerate(groups):
        boundaries[g_idx] = fine_bin_lower_mev[fine_indices[0]]

    # Last boundary is upper edge of last group
    boundaries[n_groups] = fine_bin_upper_mev[groups[-1][-1]]

    return boundaries


# =============================================================================
# COVARIANCE COLLAPSE
# =============================================================================

def collapse_covariance(
    cov_matrix: np.ndarray,
    aggregation_matrix: np.ndarray,
) -> np.ndarray:
    """
    Collapse covariance matrix: C_y = A @ C_x @ A.T

    Parameters
    ----------
    cov_matrix : np.ndarray
        Fine covariance matrix, shape (n_fine * L, n_fine * L)
    aggregation_matrix : np.ndarray
        Aggregation matrix A, shape (n_groups * L, n_fine * L)

    Returns
    -------
    np.ndarray
        Grouped covariance matrix, shape (n_groups * L, n_groups * L)
    """
    return aggregation_matrix @ cov_matrix @ aggregation_matrix.T


def apply_percentile_variance_scaling(
    cov_grouped: np.ndarray,
    cov_fine: np.ndarray,
    groups: List[List[int]],
    max_order: int,
    variance_percentile: float = 50.0,
    logger=None,
) -> np.ndarray:
    """
    Apply PSD-safe diagonal scaling to preserve variance at target percentile.

    The standard covariance collapse C_y = A @ C_x @ A.T reduces variance due to
    averaging. This function rescales the diagonal to match a target percentile
    of the fine variances within each group, while preserving correlation structure.

    Steps:
    1. For each group and order, compute percentile of fine variances
    2. Build diagonal scaling matrix S where S[i,i] = sqrt(target/current)
    3. Return C_final = S @ C_grouped @ S

    Parameters
    ----------
    cov_grouped : np.ndarray
        Grouped covariance from A @ C_fine @ A.T
    cov_fine : np.ndarray
        Original fine covariance matrix
    groups : List[List[int]]
        Fine bin indices per group
    max_order : int
        Maximum Legendre order
    variance_percentile : float
        Percentile of fine variances to use as target (0-100)
        - 50 = median (typical)
        - 80-90 = conservative
        - 100 = maximum (most conservative)
    logger : optional
        Logger for diagnostics

    Returns
    -------
    np.ndarray
        Scaled covariance matrix (PSD-safe)
    """
    n_groups = len(groups)
    n_params = n_groups * max_order

    # Build scaling factors
    scale_factors = np.ones(n_params)

    for g_idx, fine_indices in enumerate(groups):
        for order in range(1, max_order + 1):
            g_param = idx(g_idx, order, max_order)

            # Collect fine variances for this order in this group
            fine_vars = []
            for fine_i in fine_indices:
                fine_param = idx(fine_i, order, max_order)
                fine_vars.append(cov_fine[fine_param, fine_param])
            fine_vars = np.array(fine_vars)

            # Compute target variance (percentile of fine variances)
            if len(fine_vars) > 0 and np.any(fine_vars > 0):
                target_var = np.percentile(fine_vars[fine_vars > 0], variance_percentile)
            else:
                target_var = 0.0

            # Compute scaling factor
            current_var = cov_grouped[g_param, g_param]
            if current_var > 1e-20 and target_var > 0:
                scale_factors[g_param] = np.sqrt(target_var / current_var)
            else:
                scale_factors[g_param] = 1.0

    # Apply scaling: C_final = S @ C_grouped @ S (where S is diagonal)
    # This is equivalent to C_final[i,j] = scale[i] * C_grouped[i,j] * scale[j]
    S = np.diag(scale_factors)
    cov_scaled = S @ cov_grouped @ S

    # Log diagnostics
    if logger:
        avg_scale = np.mean(scale_factors)
        min_scale = np.min(scale_factors)
        max_scale = np.max(scale_factors)
        logger.info(f"  Variance scaling (percentile={variance_percentile}%):")
        logger.info(f"    Scale factors: mean={avg_scale:.2f}, min={min_scale:.2f}, max={max_scale:.2f}")

    return cov_scaled


def collapse_mean(
    mean_vector: np.ndarray,
    aggregation_matrix: np.ndarray,
) -> np.ndarray:
    """
    Collapse mean vector: mean_y = A @ mean_x

    Parameters
    ----------
    mean_vector : np.ndarray
        Fine mean coefficients, shape (n_fine * L,)
    aggregation_matrix : np.ndarray
        Aggregation matrix A, shape (n_groups * L, n_fine * L)

    Returns
    -------
    np.ndarray
        Grouped mean coefficients, shape (n_groups * L,)
    """
    return aggregation_matrix @ mean_vector


def cov_to_corr(cov_matrix: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    Convert covariance matrix to correlation matrix.

    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix
    eps : float
        Small value to avoid division by zero

    Returns
    -------
    np.ndarray
        Correlation matrix
    """
    diag = np.diag(cov_matrix)
    # Handle zero or negative diagonal (set to eps)
    diag = np.maximum(diag, eps)
    std = np.sqrt(diag)
    outer_std = np.outer(std, std)
    corr = cov_matrix / outer_std
    # Ensure diagonal is exactly 1
    np.fill_diagonal(corr, 1.0)
    # Clip to [-1, 1] for numerical stability
    corr = np.clip(corr, -1.0, 1.0)
    return corr


def check_positive_semidefinite(
    matrix: np.ndarray,
    name: str = "matrix",
    logger=None,
    fix_if_needed: bool = True,
) -> Tuple[np.ndarray, bool]:
    """
    Check if matrix is positive semidefinite and optionally fix it.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix to check
    name : str
        Name for logging
    logger : optional
        Logger
    fix_if_needed : bool
        If True, project to nearest PSD matrix if needed

    Returns
    -------
    Tuple[np.ndarray, bool]
        (possibly fixed matrix, was_psd)
    """
    eigenvalues = np.linalg.eigvalsh(matrix)
    min_eig = np.min(eigenvalues)

    if min_eig >= -1e-10:
        if logger:
            logger.info(f"  {name} is PSD (min eigenvalue: {min_eig:.2e})")
        return matrix, True

    if logger:
        logger.warning(f"  {name} has negative eigenvalues (min: {min_eig:.2e})")

    if fix_if_needed:
        # Project to nearest PSD matrix
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.maximum(eigvals, 0)
        fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Ensure symmetry
        fixed = (fixed + fixed.T) / 2

        if logger:
            new_min = np.min(np.linalg.eigvalsh(fixed))
            logger.info(f"  Projected {name} to PSD (new min eigenvalue: {new_min:.2e})")

        return fixed, False

    return matrix, False


# =============================================================================
# FORCED GRID GROUPING
# =============================================================================

def build_groups_from_forced_boundaries(
    forced_boundaries_mev: np.ndarray,
    fine_energies_mev: np.ndarray,
    fine_bin_lower_mev: np.ndarray,
    fine_bin_upper_mev: np.ndarray,
    logger=None,
) -> Tuple[List[List[int]], List[GroupInfo]]:
    """
    Build groups from forced energy boundaries (e.g., from original MF34 grid).

    Each forced group is defined by adjacent boundary pairs. Fine bins are assigned
    to the forced group whose interval contains their center energy.

    Parameters
    ----------
    forced_boundaries_mev : np.ndarray
        Sorted energy boundary points in MeV (N points -> N-1 groups).
    fine_energies_mev : np.ndarray
        Center energy of each fine bin in MeV.
    fine_bin_lower_mev : np.ndarray
        Lower boundary of each fine bin in MeV.
    fine_bin_upper_mev : np.ndarray
        Upper boundary of each fine bin in MeV.
    logger : optional
        Logger for diagnostics.

    Returns
    -------
    Tuple[List[List[int]], List[GroupInfo]]
        (groups, group_info) — fine bin indices per group and diagnostics.
    """
    n_forced = len(forced_boundaries_mev) - 1
    groups = []
    group_info_list = []

    for g in range(n_forced):
        lo = forced_boundaries_mev[g]
        hi = forced_boundaries_mev[g + 1]

        # Assign fine bins whose center energy falls within [lo, hi)
        # (last group uses <= for the upper bound)
        if g == n_forced - 1:
            mask = (fine_energies_mev >= lo) & (fine_energies_mev <= hi)
        else:
            mask = (fine_energies_mev >= lo) & (fine_energies_mev < hi)

        indices = list(np.where(mask)[0])
        if not indices:
            if logger:
                logger.warning(f"  Forced group {g}: [{lo:.4f}, {hi:.4f}] MeV — no fine bins, skipping")
            continue

        groups.append(indices)
        width = fine_bin_upper_mev[indices[-1]] - fine_bin_lower_mev[indices[0]]
        info = GroupInfo(
            group_index=len(groups) - 1,
            fine_indices=indices,
            energy_lower_mev=fine_energies_mev[indices[0]],
            energy_upper_mev=fine_energies_mev[indices[-1]],
            width_mev=width,
            min_correlation_l1=np.nan,
            sigma_ratio=np.nan,
            n_fine_bins=len(indices),
        )
        group_info_list.append(info)

    if logger:
        logger.info(f"  Forced grouping: {n_forced} intervals -> {len(groups)} non-empty groups")

    return groups, group_info_list


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def perform_adaptive_multigroup_collapse(
    cov_matrix: np.ndarray,
    corr_matrix: np.ndarray,
    nominal_results: List,  # List[NominalFitResult]
    energy_bins: List,      # List[EnergyBinInfo]
    max_order: int,
    rho_min: float = 0.90,
    sigma_ratio_max: float = 1.7,
    min_width_factor: float = 2.0,
    variance_percentile: float = 50.0,
    logger=None,
    apply_covariance_cap: bool = False,
    max_relative_std_cap: float = 1.0,
    forced_group_boundaries_mev: Optional[np.ndarray] = None,
) -> MultigroupResult:
    """
    Main entry point for adaptive multigroup covariance collapse.

    Steps:
    1. Extract energies, sigma_E, bin boundaries from nominal_results/energy_bins
    2. Find groups using l=1 correlation
    3. Build aggregation matrix (same groups for all orders)
    4. Collapse: mean_g = A @ mean_fine, cov_g = A @ cov @ A.T
    5. Compute correlation from collapsed covariance
    6. Return MultigroupResult

    Parameters
    ----------
    cov_matrix : np.ndarray
        Fine covariance matrix, shape (n_fine * max_order, n_fine * max_order)
    corr_matrix : np.ndarray
        Fine correlation matrix, same shape
    nominal_results : List[NominalFitResult]
        Nominal fit results (used for mean coefficients)
    energy_bins : List[EnergyBinInfo]
        Energy bin information (used for energies, sigma_E, bin boundaries)
    max_order : int
        Maximum Legendre order
    rho_min : float
        Minimum correlation to merge (default 0.90)
    sigma_ratio_max : float
        Maximum sigma ratio within group (default 1.7)
    min_width_factor : float
        Group width >= k * median(sigma_E) (default 2.0)
    variance_percentile : float
        Percentile of fine variances to use as target for diagonal scaling (0-100).
        - 50 = median (typical, default)
        - 80-90 = conservative but not extreme
        - 100 = maximum (most conservative)
        This rescales the grouped covariance diagonal to preserve uncertainty
        magnitudes while maintaining the correlation structure from averaging.
    logger : optional
        Logger for diagnostics

    Returns
    -------
    MultigroupResult
        Complete result of the multigroup collapse
    """
    # Step 1: Extract data from nominal_results and energy_bins
    # Only use non-interpolated results (actual EXFOR data)
    valid_indices = [
        i for i, nr in enumerate(nominal_results)
        if not nr.interpolated and nr.has_data
    ]

    if len(valid_indices) == 0:
        raise ValueError("No valid (non-interpolated) nominal results found")

    n_fine = len(valid_indices)

    # The covariance matrix may include interpolated bins (all has_data bins).
    # We need to slice it down to only the non-interpolated bins.
    all_data_indices = [i for i, nr in enumerate(nominal_results) if nr.has_data]
    if len(all_data_indices) != n_fine:
        # Build index mapping: position of each valid bin in the full cov matrix
        all_data_set = {v: pos for pos, v in enumerate(all_data_indices)}
        valid_positions = [all_data_set[vi] for vi in valid_indices]

        # Build flat parameter indices to extract from cov/corr
        flat_indices = []
        for pos in valid_positions:
            for l in range(max_order):
                flat_indices.append(pos * max_order + l)
        flat_indices = np.array(flat_indices)

        if logger:
            n_interp = len(all_data_indices) - n_fine
            logger.info(f"  Slicing covariance: {len(all_data_indices)} total bins -> {n_fine} non-interpolated ({n_interp} interpolated excluded)")

        cov_matrix = cov_matrix[np.ix_(flat_indices, flat_indices)]
        corr_matrix = corr_matrix[np.ix_(flat_indices, flat_indices)]

    # Map from valid index to energy bin
    valid_energy_bins = [energy_bins[i] for i in valid_indices]
    valid_nominal = [nominal_results[i] for i in valid_indices]

    # Extract arrays
    fine_energies_mev = np.array([eb.energy_mev for eb in valid_energy_bins])
    sigma_E_mev = np.array([eb.sigma_E_mev for eb in valid_energy_bins])
    fine_bin_lower_mev = np.array([eb.bin_lower_mev for eb in valid_energy_bins])
    fine_bin_upper_mev = np.array([eb.bin_upper_mev for eb in valid_energy_bins])
    fine_bin_widths_mev = fine_bin_upper_mev - fine_bin_lower_mev

    # Build mean coefficient vector
    mean_fine = np.zeros(n_fine * max_order)
    for i, nr in enumerate(valid_nominal):
        coeffs = nr.nominal_coeffs
        for l in range(1, min(max_order + 1, len(coeffs) + 1)):
            mean_fine[idx(i, l, max_order)] = coeffs[l - 1] if l - 1 < len(coeffs) else 0.0

    if logger:
        logger.info(f"  Fine grid: {n_fine} energy bins")
        logger.info(f"  Energy range: [{fine_energies_mev[0]:.4f}, {fine_energies_mev[-1]:.4f}] MeV")
        logger.info(f"  Covariance shape: {cov_matrix.shape}")

        # Diagnostic: l=1 adjacent correlation statistics
        if n_fine >= 2 and max_order >= 1:
            adjacent_corrs = []
            for i in range(n_fine - 1):
                idx_i = idx(i, 1, max_order)
                idx_next = idx(i + 1, 1, max_order)
                if idx_i < corr_matrix.shape[0] and idx_next < corr_matrix.shape[0]:
                    adjacent_corrs.append(corr_matrix[idx_i, idx_next])
            if adjacent_corrs:
                adjacent_corrs = np.array(adjacent_corrs)
                logger.info(f"  l=1 adjacent corr: mean={np.mean(adjacent_corrs):.4f}, "
                            f"median={np.median(adjacent_corrs):.4f}, "
                            f"max={np.max(adjacent_corrs):.4f}, "
                            f"min={np.min(adjacent_corrs):.4f}")

    # Step 2: Find groups
    if forced_group_boundaries_mev is not None and len(forced_group_boundaries_mev) >= 2:
        # Hybrid grouping: forced grid where available, adaptive elsewhere
        forced_lo = forced_group_boundaries_mev[0]
        forced_hi = forced_group_boundaries_mev[-1]

        if logger:
            logger.info(f"  Forced MF34 grid: {len(forced_group_boundaries_mev)} boundaries, "
                        f"[{forced_lo:.4f}, {forced_hi:.4f}] MeV")

        # Identify fine bins inside vs outside the forced grid range
        inside_mask = (fine_energies_mev >= forced_lo) & (fine_energies_mev <= forced_hi)
        inside_indices = np.where(inside_mask)[0]
        below_indices = np.where(fine_energies_mev < forced_lo)[0]
        above_indices = np.where(fine_energies_mev > forced_hi)[0]

        all_groups = []
        all_info = []

        # Adaptive grouping for bins BELOW forced range
        if len(below_indices) > 0:
            if logger:
                logger.info(f"  Adaptive grouping for {len(below_indices)} bins below forced range")
            sub_groups, sub_info = find_adaptive_group_boundaries(
                corr_matrix=corr_matrix,
                cov_matrix=cov_matrix,
                fine_energies_mev=fine_energies_mev,
                sigma_E_mev=sigma_E_mev,
                fine_bin_widths_mev=fine_bin_widths_mev,
                max_order=max_order,
                rho_min=rho_min,
                sigma_ratio_max=sigma_ratio_max,
                min_width_factor=min_width_factor,
                logger=logger,
            )
            # Filter to only groups that contain bins below forced range
            for g, gi in zip(sub_groups, sub_info):
                if any(i in below_indices for i in g):
                    filtered = [i for i in g if i in below_indices]
                    if filtered:
                        all_groups.append(filtered)
                        all_info.append(gi)

        # Forced grouping for bins INSIDE the forced range
        if len(inside_indices) > 0:
            # Clip forced boundaries to fine grid range for inside bins
            forced_groups, forced_info = build_groups_from_forced_boundaries(
                forced_boundaries_mev=forced_group_boundaries_mev,
                fine_energies_mev=fine_energies_mev,
                fine_bin_lower_mev=fine_bin_lower_mev,
                fine_bin_upper_mev=fine_bin_upper_mev,
                logger=logger,
            )
            all_groups.extend(forced_groups)
            all_info.extend(forced_info)

        # Adaptive grouping for bins ABOVE forced range
        if len(above_indices) > 0:
            if logger:
                logger.info(f"  Adaptive grouping for {len(above_indices)} bins above forced range")
            sub_groups, sub_info = find_adaptive_group_boundaries(
                corr_matrix=corr_matrix,
                cov_matrix=cov_matrix,
                fine_energies_mev=fine_energies_mev,
                sigma_E_mev=sigma_E_mev,
                fine_bin_widths_mev=fine_bin_widths_mev,
                max_order=max_order,
                rho_min=rho_min,
                sigma_ratio_max=sigma_ratio_max,
                min_width_factor=min_width_factor,
                logger=logger,
            )
            # Filter to only groups that contain bins above forced range
            for g, gi in zip(sub_groups, sub_info):
                if any(i in above_indices for i in g):
                    filtered = [i for i in g if i in above_indices]
                    if filtered:
                        all_groups.append(filtered)
                        all_info.append(gi)

        groups = all_groups
        # Re-index group_info
        group_info = []
        for new_idx, gi in enumerate(all_info):
            gi.group_index = new_idx
            group_info.append(gi)
    else:
        # Standard adaptive grouping
        groups, group_info = find_adaptive_group_boundaries(
            corr_matrix=corr_matrix,
            cov_matrix=cov_matrix,
            fine_energies_mev=fine_energies_mev,
            sigma_E_mev=sigma_E_mev,
            fine_bin_widths_mev=fine_bin_widths_mev,
            max_order=max_order,
            rho_min=rho_min,
            sigma_ratio_max=sigma_ratio_max,
            min_width_factor=min_width_factor,
            logger=logger,
        )

    n_groups = len(groups)

    # Step 3: Build aggregation matrix
    A = build_aggregation_matrix(
        groups=groups,
        fine_bin_widths_mev=fine_bin_widths_mev,
        n_fine=n_fine,
        max_order=max_order,
    )

    if logger:
        logger.info(f"  Aggregation matrix shape: {A.shape}")

    # Step 4: Collapse covariance and mean
    # Standard averaging collapse
    cov_grouped = collapse_covariance(cov_matrix, A)
    mean_grouped = collapse_mean(mean_fine, A)

    # Apply percentile-based variance scaling to preserve uncertainty magnitudes
    cov_grouped = apply_percentile_variance_scaling(
        cov_grouped=cov_grouped,
        cov_fine=cov_matrix,
        groups=groups,
        max_order=max_order,
        variance_percentile=variance_percentile,
        logger=logger,
    )

    # Apply covariance cap to grouped matrix if enabled
    if apply_covariance_cap:
        from scripts.exfor_utils import cap_covariance_relative_uncertainty

        n_groups = len(groups)
        mg_param_labels = [
            (g_idx, l + 1) for g_idx in range(n_groups) for l in range(max_order)
        ]

        # Build energy MeV lookup: group index -> midpoint energy
        energy_mev_lookup = {
            gi.group_index: 0.5 * (gi.energy_lower_mev + gi.energy_upper_mev)
            for gi in group_info
        }

        if logger:
            logger.info(f"  Applying covariance cap to multigroup matrix (max std = {max_relative_std_cap*100:.0f}%)")

        cov_grouped, cap_diag = cap_covariance_relative_uncertainty(
            cov_matrix=cov_grouped,
            max_relative_std=max_relative_std_cap,
            param_labels=mg_param_labels,
            energy_mev_lookup=energy_mev_lookup,
            logger=logger,
        )

    # Check and fix PSD if needed
    cov_grouped, was_psd = check_positive_semidefinite(
        cov_grouped, "Grouped covariance", logger, fix_if_needed=True
    )

    # Step 5: Compute correlation from collapsed covariance
    corr_grouped = cov_to_corr(cov_grouped)

    # Step 6: Construct energy boundaries
    group_boundaries_mev = construct_group_energy_boundaries(
        groups=groups,
        fine_bin_lower_mev=fine_bin_lower_mev,
        fine_bin_upper_mev=fine_bin_upper_mev,
    )
    group_boundaries_ev = group_boundaries_mev * 1e6

    if logger:
        logger.info(f"  Grouped covariance shape: {cov_grouped.shape}")
        logger.info(f"  Group boundaries (MeV): {group_boundaries_mev}")
        compression = n_fine / n_groups
        logger.info(f"  Compression ratio: {compression:.1f}x ({n_fine} -> {n_groups})")

    return MultigroupResult(
        group_boundaries_mev=group_boundaries_mev,
        group_boundaries_ev=group_boundaries_ev,
        aggregation_matrix=A,
        cov_grouped=cov_grouped,
        corr_grouped=corr_grouped,
        mean_grouped=mean_grouped,
        groups=groups,
        group_info=group_info,
    )
