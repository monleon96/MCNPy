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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
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
    valid_mask_grouped: np.ndarray = None # (N_groups * L,) bool – which MG params had valid fine data
    scale_factors: Optional[np.ndarray] = None  # (N_groups * L,) — diagonal of S from
                                                # apply_percentile_variance_scaling. v3 uses it
                                                # to apply the same compensation to the a_l axis
                                                # of the (σ, a_l) cross block; without it, the
                                                # cross block ends up under-scaled relative to
                                                # the variance-compensated a_l block.


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
    sigma_ratio_max: Optional[float] = None,
    sigma_ratio_per_order: bool = False,
    logger=None,
    diagnostics_file: Optional[Path] = None,
) -> Tuple[List[List[int]], List[GroupInfo]]:
    """
    Find group boundaries using l=1 correlation structure only.

    Algorithm (greedy merging):
    1. Start at i=0
    2. Extend group [i0, i1]: merge if rho >= rho_min AND
       (if sigma_ratio_max is set) the running group's max(σ)/min(σ) remains
       <= sigma_ratio_max -- at l=1 only, or in EVERY order when
       ``sigma_ratio_per_order`` is set
    3. Finalize group, start next at i1+1

    The optional sigma_ratio cap prevents merging strongly correlated bins
    whose l=1 std differ by more than the configured factor — these are
    physically heterogeneous and force the percentile-based variance
    compensation to over-inflate the group variance.  When ``sigma_ratio_max``
    is None, only the correlation criterion applies.

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
        Minimum l=1 adjacent correlation to allow merging (default 0.90)
    sigma_ratio_max : float, optional
        Maximum running max(σ)/min(σ) across the group being grown.
        ``None`` (default) disables the gate, restoring the correlation-only
        behavior.

        ⛔ THE VALUE HAS NO DERIVATION.  It entered on 2026-01-30 (67f65ce) with
        no recorded justification, and the "recommended range 3-10" this
        docstring used to assert had no source either.  What it actually bounds
        is the WORST MEMBER's under-declaration: the group ships one number, so
        a member whose sigma is ``c`` times the group's smallest is declared up
        to ``c`` short in sigma (``c^2`` in variance).  That is the reading to
        argue from, and it makes ``c`` a conservatism budget, not a tuning knob.
    sigma_ratio_per_order : bool
        Read the ratio in EVERY order instead of only l=1 (default ``False``,
        the historical behaviour).

        ⛔ AT l=1 THE CAP PROTECTS NOBODY, and this is measured, not argued.
        The sigma ratio INSIDE the groups the pipeline ships, per order
        (run 99, `docs/chi2-mf4/mf34_mesh_frontier.md` §2-bis):

            order     a_1    a_2    a_3    a_4      a_5      a_6
            p95       4.4    4.7    6.1     46      101       26
            max       5.0   12.9   17.5   1.9e3   2.0e3    1.2e3

        a_1's maximum is EXACTLY the cap: the gate binds on the one order that
        satisfies it anyway, while a_4-a_6 run to 2000x -- i.e. those orders are
        under-declared by up to three orders of magnitude at specific energies,
        which is the 711x folded error the mesh frontier reports.

        Measured on run 99's emitted covariance, keeping the SAME cap value and
        changing only where it is read (destroyed = sum of tr(B_S) - u^T B_S u,
        the DP's own functional):

            l=1 only, cap 5     677 groups   100.0 % destroyed
            per order, cap 5   1245 groups    43.2 %
            per order, cap 3   1424 groups    27.8 %
            per order, cap 2   1575 groups    15.1 %
            no grouping        1738 groups     0.0 %

        So reading the same number in the right place recovers 57 % of the
        destroyed variance.  The defect was the PLACE, not the value.

        ⚑ MONOTONE BY CONSTRUCTION, which is what makes it safe: requiring the
        bound in more orders can only refuse merges, so the mesh can only get
        FINER and the destroyed variance can only go DOWN.  It cannot
        under-declare anything the l=1 version declared.
    logger : optional
        Logger for diagnostics
    diagnostics_file : Path, optional
        If provided, write per-boundary merge decisions to this CSV file.
        Each row is an adjacent pair (i, i+1) with correlation, sigma_ratio,
        individual sigmas, and the merge decision. Useful for post-processing
        to evaluate the effect of different thresholds without re-running.

    Returns
    -------
    Tuple[List[List[int]], List[GroupInfo]]
        groups: List of lists, each containing fine bin indices for a group
        group_info: Diagnostic information for each group
    """
    n_fine = len(fine_energies_mev)

    if logger:
        logger.info(f"  Grouping parameters:")
        logger.info(f"    rho_min = {rho_min}")
        if sigma_ratio_max is not None:
            logger.info(f"    sigma_ratio_max = {sigma_ratio_max}"
                        + (f" (leido en los {max_order} ordenes)"
                           if sigma_ratio_per_order else " (leido SOLO en l=1)"))
        else:
            logger.info(f"    sigma_ratio_max = (disabled)")

    # Pre-compute per-bin sigma for every order.  Row 0 is l=1, which is both
    # the diagnostics column and the historical gate.
    sigma_per_bin = np.array([
        [np.sqrt(max(cov_matrix[idx(i, l, max_order), idx(i, l, max_order)], 0.0))
         for i in range(n_fine)]
        for l in range(1, max_order + 1)
    ])
    sigma_l1_per_bin = sigma_per_bin[0]
    gate_orders = range(max_order) if sigma_ratio_per_order else range(1)

    # Pre-compute all adjacent l=1 correlations for diagnostics
    adj_rho = np.array([
        corr_matrix[idx(i, 1, max_order), idx(i + 1, 1, max_order)]
        for i in range(n_fine - 1)
    ])

    # Write diagnostics CSV header
    diag_fh = None
    if diagnostics_file is not None:
        diag_fh = open(diagnostics_file, "w")
        diag_fh.write(
            "bin_i,bin_j,E_i_MeV,E_j_MeV,rho_adj,sigma_i,sigma_j,"
            "sigma_ratio_pair,sigma_ratio_group,merged,group_id\n"
        )

    groups = []
    group_info_list = []

    i0 = 0
    current_group_id = 0
    while i0 < n_fine:
        i1 = i0

        # Track group properties
        min_corr_in_group = 1.0

        while i1 + 1 < n_fine:
            rho = adj_rho[i1]

            # Running group sigma ratio: max/min of all positive sigmas in the
            # candidate-extended group [i0 .. i1+1].  `ratio` stays the l=1
            # number so the diagnostics CSV keeps meaning what it meant; the
            # gate uses the worst order among `gate_orders`.
            sigmas_l1 = [s for s in sigma_l1_per_bin[i0:i1 + 2] if s > 0]
            ratio = max(sigmas_l1) / min(sigmas_l1) if len(sigmas_l1) >= 2 else 1.0
            gate_ratio = 0.0
            for _l in gate_orders:
                sl = sigma_per_bin[_l, i0:i1 + 2]
                sl = sl[sl > 0]
                if len(sl) >= 2:
                    gate_ratio = max(gate_ratio, float(sl.max()/sl.min()))

            s_i = sigma_l1_per_bin[i1]
            s_j = sigma_l1_per_bin[i1 + 1]
            if s_i > 0 and s_j > 0:
                pair_ratio = max(s_i, s_j) / min(s_i, s_j)
            else:
                pair_ratio = 1.0

            # Merge if correlation criterion met AND (optionally) the
            # candidate-extended group keeps σ-heterogeneity bounded.
            merged = rho >= rho_min
            if sigma_ratio_max is not None:
                merged = merged and (max(gate_ratio, 1.0) <= sigma_ratio_max)

            if diag_fh is not None:
                diag_fh.write(
                    f"{i1},{i1+1},{fine_energies_mev[i1]:.6f},"
                    f"{fine_energies_mev[i1+1]:.6f},{rho:.6f},"
                    f"{s_i:.6e},{s_j:.6e},{pair_ratio:.4f},"
                    f"{ratio:.4f},{int(merged)},{current_group_id}\n"
                )

            if merged:
                min_corr_in_group = min(min_corr_in_group, rho)
                i1 += 1
            else:
                break

        # Finalize this group
        group_indices = list(range(i0, i1 + 1))
        groups.append(group_indices)

        # Compute final group diagnostics
        group_width = sum(fine_bin_widths_mev[i0:i1 + 1])

        # Final sigma ratio for the group
        sigmas_pos = [s for s in sigma_l1_per_bin[i0:i1 + 1] if s > 0]
        if len(sigmas_pos) >= 2:
            final_ratio = max(sigmas_pos) / min(sigmas_pos)
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
        current_group_id += 1
        i0 = i1 + 1

    if diag_fh is not None:
        diag_fh.close()
        if logger:
            logger.info(f"  Boundary decisions written to: {diagnostics_file}")

    if logger:
        logger.info(f"  Found {len(groups)} multigroups from {n_fine} fine bins")
        # Summary statistics
        multi_bin = [g for g in group_info_list if g.n_fine_bins > 1]
        single_bin = len(group_info_list) - len(multi_bin)
        if multi_bin:
            rhos = [g.min_correlation_l1 for g in multi_bin]
            srs = [g.sigma_ratio for g in multi_bin]
            logger.info(f"    Single-bin groups: {single_bin}, multi-bin groups: {len(multi_bin)}")
            logger.info(f"    Multi-bin min_rho: min={min(rhos):.3f}, mean={np.mean(rhos):.3f}")
            logger.info(f"    Multi-bin sigma_ratio: max={max(srs):.2f}, mean={np.mean(srs):.2f}")
        else:
            logger.info(f"    All {single_bin} groups are single-bin (no merges)")
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
# SECOND-PASS MERGE AFTER SMOOTHING
# =============================================================================

def try_merge_adjacent_multigroups(
    multigroup_result: MultigroupResult,
    cov_mg_smoothed: np.ndarray,
    cov_fine: np.ndarray,
    mean_fine: np.ndarray,
    fine_bin_widths_mev: np.ndarray,
    fine_bin_lower_mev: np.ndarray,
    fine_bin_upper_mev: np.ndarray,
    n_fine: int,
    max_order: int,
    rho_min: float = 0.90,
    sigma_ratio_max: float = 2.5,
    variance_percentile: float = 50.0,
    valid_mask: Optional[np.ndarray] = None,
    logger=None,
) -> Optional[MultigroupResult]:
    """
    Attempt to merge adjacent multigroups whose smoothed covariance now
    satisfies the merge criteria (correlation and sigma-ratio).

    After smoothing fixes dip/spike artifacts, adjacent groups that were
    previously split by sigma-ratio violations may now be mergeable. This
    function performs a greedy left-to-right scan and, if any merges are
    found, re-collapses from the fine grid with the merged group structure.

    Parameters
    ----------
    multigroup_result : MultigroupResult
        Result from the first-pass collapse.
    cov_mg_smoothed : np.ndarray
        Smoothed multigroup covariance (used for merge decisions).
    cov_fine : np.ndarray
        Original fine-grid covariance (used for re-collapse).
    mean_fine : np.ndarray
        Fine-grid mean parameter vector.
    fine_bin_widths_mev : np.ndarray
        Width of each fine bin in MeV.
    fine_bin_lower_mev, fine_bin_upper_mev : np.ndarray
        Lower/upper boundaries of fine bins in MeV.
    n_fine : int
        Number of fine energy bins.
    max_order : int
        Maximum Legendre order.
    rho_min : float
        Minimum l=1 adjacent correlation to allow merge.
    sigma_ratio_max : float
        Maximum l=1 sigma ratio within candidate merged group.
    variance_percentile : float
        Maximum percentile of fine-bin variances used as target (50-95).
    valid_mask : np.ndarray, optional
        Boolean mask for fitted parameters (n_fine * max_order).
    logger : optional
        Logger for diagnostics.

    Returns
    -------
    Optional[MultigroupResult]
        New result with merged groups, or None if no merges were possible.
    """
    groups = multigroup_result.groups
    n_groups = len(groups)
    if n_groups < 2:
        return None

    # Correlation matrix from smoothed MG covariance
    corr_mg = cov_to_corr(cov_mg_smoothed)

    # Extract l=1 sigma for each multigroup from smoothed diagonal
    mg_sigma_l1 = np.array([
        np.sqrt(max(cov_mg_smoothed[idx(g, 1, max_order), idx(g, 1, max_order)], 0.0))
        for g in range(n_groups)
    ])

    # Greedy left-to-right merge scan
    merged_groups = [list(groups[0])]
    merged_sigmas = [mg_sigma_l1[0]]  # track l=1 sigmas contributing to current group

    n_merges = 0
    regroup_decisions = []  # (g_prev, g, rho, sr, merged)
    for g in range(1, n_groups):
        # l=1 adjacent correlation between original group g-1 and g
        rho_adj = corr_mg[idx(g - 1, 1, max_order), idx(g, 1, max_order)]

        # Sigma ratio of candidate merged group: collect all l=1 sigmas
        candidate_sigmas = merged_sigmas + [mg_sigma_l1[g]]
        pos_sigmas = [s for s in candidate_sigmas if s > 0]

        if len(pos_sigmas) >= 2:
            sr = max(pos_sigmas) / min(pos_sigmas)
        else:
            sr = 1.0

        merged = rho_adj >= rho_min and sr <= sigma_ratio_max
        regroup_decisions.append((g - 1, g, rho_adj, sr,
                                  mg_sigma_l1[g - 1], mg_sigma_l1[g], merged))

        if merged:
            # Merge: extend current group with fine bins from group g
            merged_groups[-1].extend(groups[g])
            merged_sigmas.append(mg_sigma_l1[g])
            n_merges += 1
        else:
            # Start new group
            merged_groups.append(list(groups[g]))
            merged_sigmas = [mg_sigma_l1[g]]

    if logger:
        logger.info(f"  Regroup scan ({n_groups - 1} boundaries, rho_min={rho_min}, "
                     f"sigma_ratio_max={sigma_ratio_max}):")
        for g_prev, g_curr, rho, sr, s_prev, s_curr, did_merge in regroup_decisions:
            tag = "MERGE" if did_merge else "SPLIT"
            logger.info(f"    [{tag}] G{g_prev}-G{g_curr}: rho={rho:.3f}, "
                         f"sr={sr:.2f}, sigma=({s_prev:.3e},{s_curr:.3e})")

    if n_merges == 0:
        if logger:
            logger.info("  Regroup: no adjacent groups eligible for merge")
        return None

    if logger:
        logger.info(f"  Regroup: {n_groups} -> {len(merged_groups)} groups ({n_merges} merges)")

    # Re-collapse from fine grid with merged group structure
    n_new = len(merged_groups)

    A = build_aggregation_matrix(
        groups=merged_groups,
        fine_bin_widths_mev=fine_bin_widths_mev,
        n_fine=n_fine,
        max_order=max_order,
        valid_mask=valid_mask,
    )

    cov_grouped = collapse_covariance(cov_fine, A)
    mean_grouped = collapse_mean(mean_fine, A)

    # Variance scaling
    cov_grouped, scale_factors = apply_percentile_variance_scaling(
        cov_grouped=cov_grouped,
        cov_fine=cov_fine,
        groups=merged_groups,
        max_order=max_order,
        variance_percentile_min=67.0,
        variance_percentile_max=85.0,
        variance_ratio_ref=5.0,
        fine_bin_widths_mev=fine_bin_widths_mev,
        valid_mask=valid_mask,
        logger=logger,
    )

    # PSD check
    cov_grouped, _ = check_positive_semidefinite(
        cov_grouped, "Regrouped covariance", logger, fix_if_needed=True
    )

    corr_grouped = cov_to_corr(cov_grouped)

    group_boundaries_mev = construct_group_energy_boundaries(
        groups=merged_groups,
        fine_bin_lower_mev=fine_bin_lower_mev,
        fine_bin_upper_mev=fine_bin_upper_mev,
    )
    group_boundaries_ev = group_boundaries_mev * 1e6

    # Build GroupInfo for merged groups
    group_info = []
    for g_idx, fine_indices in enumerate(merged_groups):
        e_lo = fine_bin_lower_mev[fine_indices[0]]
        e_hi = fine_bin_upper_mev[fine_indices[-1]]
        group_info.append(GroupInfo(
            group_index=g_idx,
            fine_indices=fine_indices,
            energy_lower_mev=e_lo,
            energy_upper_mev=e_hi,
            width_mev=e_hi - e_lo,
            min_correlation_l1=np.nan,
            sigma_ratio=np.nan,
            n_fine_bins=len(fine_indices),
        ))

    # Compute valid_mask_grouped: True where at least one fine bin contributed
    _vmg = np.any(A != 0, axis=1)

    return MultigroupResult(
        group_boundaries_mev=group_boundaries_mev,
        group_boundaries_ev=group_boundaries_ev,
        aggregation_matrix=A,
        cov_grouped=cov_grouped,
        corr_grouped=corr_grouped,
        mean_grouped=mean_grouped,
        groups=merged_groups,
        group_info=group_info,
        valid_mask_grouped=_vmg,
        scale_factors=scale_factors,
    )


# =============================================================================
# AGGREGATION MATRIX
# =============================================================================

def build_aggregation_matrix(
    groups: List[List[int]],
    fine_bin_widths_mev: np.ndarray,
    n_fine: int,
    max_order: int,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build aggregation matrix A for y = A @ x, C_y = A @ C_x @ A.T.

    A is block-diagonal: same grouping applied to each Legendre order.
    Weights: w_i = bin_width_i (energy interval average definition).

    Layout matches fine covariance: [a1(E1), a2(E1), ..., aL(E1), a1(E2), ...]

    For group g containing fine bins {i1, i2, ...}:
        A[g*L + (l-1), i*L + (l-1)] = w_i / sum(w_j for j in group)

    When ``valid_mask`` is provided, only fitted (valid) bins contribute to
    each order's weights.  Unfitted bins get weight 0, so their zero
    coefficients don't dilute the group mean.

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
    valid_mask : np.ndarray, optional
        Boolean mask of shape (n_fine * max_order,). True = parameter was
        fitted (frozen_degree >= order). When None, all bins contribute.

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
        for order in range(1, max_order + 1):
            row_idx = idx(g_idx, order, max_order)

            # Filter to fitted bins for this order
            if valid_mask is not None:
                fitted_indices = [i for i in fine_indices
                                  if valid_mask[idx(i, order, max_order)]]
            else:
                fitted_indices = fine_indices

            if not fitted_indices:
                continue  # all weights remain 0

            total_width = sum(fine_bin_widths_mev[i] for i in fitted_indices)
            if total_width <= 0:
                total_width = len(fitted_indices)
                weights = {i: 1.0 / total_width for i in fitted_indices}
            else:
                weights = {i: fine_bin_widths_mev[i] / total_width
                           for i in fitted_indices}

            for fine_i in fitted_indices:
                col_idx = idx(fine_i, order, max_order)
                A[row_idx, col_idx] = weights[fine_i]

    return A


def build_l0_row_aggregator(
    groups: List[List[int]],
    fine_bin_widths_mev: np.ndarray,
    n_fine: int,
) -> np.ndarray:
    """Width-weighted row aggregator for the v3 L=0 cross block.

    Returns A_0 of shape ``(n_groups, n_fine)`` such that
    ``A_0[g, i] = w_i / sum_{j in group_g} w_j`` for ``i in group_g``, else 0.

    This is the scalar (one-row-per-bin) analogue of the per-order pattern in
    ``build_aggregation_matrix``. Use it to collapse the v3 (a_0, a_l) cross
    block: ``cov_c0_al_grouped = A_0 @ cov_c0_al_fine @ A.T`` with shape
    ``(n_groups, n_groups * max_order)``. Width-weighting matches MF33's own
    piecewise-constant projection onto the fine bins (xs_cov.project_to_grid),
    so this is the linear extension of that projection onto the coarse grid.
    """
    n_groups = len(groups)
    A_0 = np.zeros((n_groups, n_fine))
    for g_idx, fine_indices in enumerate(groups):
        if not fine_indices:
            continue
        widths = np.array([fine_bin_widths_mev[i] for i in fine_indices], dtype=float)
        total = widths.sum()
        if total <= 0:
            weights = np.ones_like(widths) / len(fine_indices)
        else:
            weights = widths / total
        for i, fine_i in enumerate(fine_indices):
            A_0[g_idx, fine_i] = weights[i]
    return A_0


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


def _compute_group_quality_score(
    fine_indices: List[int],
    cov_fine: np.ndarray,
    fine_bin_widths_mev: np.ndarray,
    max_order: int,
) -> float:
    """
    Compute retention-based quality score q for a group of fine bins.

    q = rho_w + (1 - rho_w) / n_eff

    where rho_w is the weight-aware mean off-diagonal l=1 correlation and
    n_eff = 1 / sum(w_i^2) is the effective group size.

    Returns 1.0 for single-bin groups or all-zero variance groups.
    """
    n = len(fine_indices)
    if n == 1:
        return 1.0

    # Extract l=1 variances
    l1_vars = np.array([
        cov_fine[idx(i, 1, max_order), idx(i, 1, max_order)]
        for i in fine_indices
    ])

    # All-zero variance → no information to lose
    if not np.any(l1_vars > 0):
        return 1.0

    # Build l=1 correlation sub-matrix
    l1_std = np.sqrt(np.maximum(l1_vars, 0.0))
    n_bins = len(fine_indices)
    corr_sub = np.eye(n_bins)
    for a in range(n_bins):
        for b in range(a + 1, n_bins):
            if l1_std[a] > 0 and l1_std[b] > 0:
                cov_ab = cov_fine[
                    idx(fine_indices[a], 1, max_order),
                    idx(fine_indices[b], 1, max_order),
                ]
                rho_ab = cov_ab / (l1_std[a] * l1_std[b])
                rho_ab = np.clip(rho_ab, -1.0, 1.0)
                corr_sub[a, b] = rho_ab
                corr_sub[b, a] = rho_ab

    # Compute width-based weights (same as aggregation matrix)
    widths = fine_bin_widths_mev[fine_indices]
    total_w = widths.sum()
    if total_w <= 0:
        w = np.ones(n_bins) / n_bins
    else:
        w = widths / total_w

    # n_eff = 1 / sum(w_i^2)
    n_eff = 1.0 / np.sum(w ** 2)

    # rho_w = weighted mean off-diagonal correlation
    num = 0.0
    den = 0.0
    for a in range(n_bins):
        for b in range(a + 1, n_bins):
            ww = w[a] * w[b]
            num += ww * corr_sub[a, b]
            den += ww

    if den > 0:
        rho_w = num / den
    else:
        rho_w = 0.0

    q = rho_w + (1.0 - rho_w) / n_eff
    return q


def apply_percentile_variance_scaling(
    cov_grouped: np.ndarray,
    cov_fine: np.ndarray,
    groups: List[List[int]],
    max_order: int,
    variance_percentile_min: float = 67.0,
    variance_percentile_max: float = 85.0,
    variance_ratio_ref: float = 5.0,
    fine_bin_widths_mev: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    logger=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply PSD-safe diagonal scaling to compensate averaging-induced variance loss.

    The standard covariance collapse C_y = A @ C_x @ A.T reduces variance due to
    averaging. This function rescales the diagonal to match a target percentile
    of the fine variances within each group, while preserving correlation structure.

    The percentile is adaptive per group, driven by the intra-group variance
    heterogeneity (l=1 sigma ratio):

        sigma_ratio = max(sigma_l1) / min(sigma_l1)   within the group
        t = min((sigma_ratio - 1) / (variance_ratio_ref - 1), 1)
        group_pct = variance_percentile_min + t * (variance_percentile_max - variance_percentile_min)

    Homogeneous groups (ratio ~ 1) receive ``variance_percentile_min``.
    Heterogeneous groups (ratio >= ``variance_ratio_ref``) receive
    ``variance_percentile_max``.

    A no-deflation guard ensures scale factors are always >= 1.0.

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
    variance_percentile_min : float
        Base percentile for groups with homogeneous variance (default 67).
    variance_percentile_max : float
        Maximum percentile for groups with highly heterogeneous variance
        (default 85).
    variance_ratio_ref : float
        Intra-group sigma ratio at which the percentile saturates at
        ``variance_percentile_max`` (default 5.0).
    fine_bin_widths_mev : np.ndarray, optional
        Width of each fine bin in MeV (unused, kept for API compatibility).
    valid_mask : np.ndarray, optional
        Boolean mask for fitted parameters (n_fine * max_order).
    logger : optional
        Logger for diagnostics

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(cov_scaled, scale_factors)`` where ``cov_scaled = S @ cov_grouped @ S``
        and ``scale_factors`` is the length-``n_groups·max_order`` diagonal of S.
        Callers that propagate the same compensation to a coupled block (e.g.
        the v3 σ–a_l cross block) need ``scale_factors`` to apply S on the a_l
        axis of that block.
    """
    n_groups = len(groups)
    n_params = n_groups * max_order

    p_min = np.clip(variance_percentile_min, 50.0, 95.0)
    p_max = np.clip(variance_percentile_max, p_min, 95.0)

    # Build scaling factors
    scale_factors = np.ones(n_params)
    group_percentiles = np.full(n_groups, p_min)
    group_sigma_ratios = np.ones(n_groups)

    for g_idx, fine_indices in enumerate(groups):
        # Compute intra-group l=1 sigma ratio
        l1_sigmas = np.array([
            np.sqrt(max(cov_fine[idx(i, 1, max_order), idx(i, 1, max_order)], 0.0))
            for i in fine_indices
        ])
        pos_sigmas = l1_sigmas[l1_sigmas > 0]
        if len(pos_sigmas) >= 2:
            sigma_ratio = float(pos_sigmas.max() / pos_sigmas.min())
        else:
            sigma_ratio = 1.0
        group_sigma_ratios[g_idx] = sigma_ratio

        # Adaptive percentile: linear ramp from p_min to p_max
        if variance_ratio_ref > 1.0:
            t = min((sigma_ratio - 1.0) / (variance_ratio_ref - 1.0), 1.0)
        else:
            t = 0.0
        group_pct = p_min + t * (p_max - p_min)
        group_percentiles[g_idx] = group_pct

        for order in range(1, max_order + 1):
            g_param = idx(g_idx, order, max_order)

            # Collect fine variances for this order in this group
            # Skip unfitted bins (valid_mask == False) so zeros don't dilute
            fine_vars = []
            for fine_i in fine_indices:
                fine_param = idx(fine_i, order, max_order)
                if valid_mask is not None and not valid_mask[fine_param]:
                    continue
                fine_vars.append(cov_fine[fine_param, fine_param])
            fine_vars = np.array(fine_vars)

            # Compute target variance (percentile of fine variances)
            if len(fine_vars) > 0 and np.any(fine_vars > 0):
                target_var = np.percentile(fine_vars[fine_vars > 0], group_pct)
            else:
                target_var = 0.0

            # Compute scaling factor with no-deflation guard
            current_var = cov_grouped[g_param, g_param]
            if current_var > 1e-20 and target_var > 0:
                scale_factors[g_param] = max(1.0, np.sqrt(target_var / current_var))
            else:
                scale_factors[g_param] = 1.0

    # Apply scaling: C_final = S @ C_grouped @ S (where S is diagonal)
    # This is equivalent to C_final[i,j] = scale[i] * C_grouped[i,j] * scale[j]
    S = np.diag(scale_factors)
    cov_scaled = S @ cov_grouped @ S

    # Compute variance retention diagnostics
    # Compare grouped diagonal variances against mean of fine variances per group
    retention_before = []
    retention_after = []
    for g_idx, fine_indices in enumerate(groups):
        for order in range(1, max_order + 1):
            g_param = idx(g_idx, order, max_order)
            fine_vars = []
            for fine_i in fine_indices:
                fine_param = idx(fine_i, order, max_order)
                if valid_mask is not None and not valid_mask[fine_param]:
                    continue
                fine_vars.append(cov_fine[fine_param, fine_param])
            fine_vars = np.array(fine_vars)
            mean_fine_var = np.mean(fine_vars[fine_vars > 0]) if np.any(fine_vars > 0) else 0.0
            if mean_fine_var > 1e-20:
                retention_before.append(cov_grouped[g_param, g_param] / mean_fine_var)
                retention_after.append(cov_scaled[g_param, g_param] / mean_fine_var)
    retention_before = np.array(retention_before) * 100
    retention_after = np.array(retention_after) * 100

    # Log diagnostics
    if logger:
        avg_scale = np.mean(scale_factors)
        min_scale = np.min(scale_factors)
        max_scale = np.max(scale_factors)
        logger.info(f"  Adaptive variance compensation (p_min={p_min:.0f}%, "
                     f"p_max={p_max:.0f}%, ratio_ref={variance_ratio_ref:.1f}):")
        logger.info(f"    Sigma ratios: mean={np.mean(group_sigma_ratios):.2f}, "
                    f"min={np.min(group_sigma_ratios):.2f}, max={np.max(group_sigma_ratios):.2f}")
        logger.info(f"    Per-group percentiles: mean={np.mean(group_percentiles):.1f}, "
                    f"min={np.min(group_percentiles):.1f}, max={np.max(group_percentiles):.1f}")
        logger.info(f"    Scale factors: mean={avg_scale:.2f}, min={min_scale:.2f}, max={max_scale:.2f}")
        if len(retention_before) > 0:
            logger.info(f"    Variance retention before: {np.mean(retention_before):.0f}% (mean), "
                        f"{np.min(retention_before):.0f}%-{np.max(retention_before):.0f}% (range)")
            logger.info(f"    Variance retention after:  {np.mean(retention_after):.0f}% (mean), "
                        f"{np.min(retention_after):.0f}%-{np.max(retention_after):.0f}% (range)")

    return cov_scaled, scale_factors


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
    # Sanitize NaN/Inf before eigendecomposition
    if np.any(~np.isfinite(matrix)):
        n_bad = int(np.sum(~np.isfinite(matrix)))
        if logger:
            logger.warning(f"  {name}: {n_bad} non-finite entries found, replacing with 0")
        matrix = matrix.copy()
        matrix[~np.isfinite(matrix)] = 0.0
        matrix = (matrix + matrix.T) / 2  # re-symmetrize

    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
    except np.linalg.LinAlgError:
        if logger:
            logger.warning(f"  {name}: eigvalsh failed to converge, falling back to eigh with clipping")
        if fix_if_needed:
            try:
                eigvals, eigvecs = np.linalg.eigh(matrix)
            except np.linalg.LinAlgError:
                if logger:
                    logger.warning(f"  {name}: eigh also failed, returning diagonal-only matrix")
                fixed = np.diag(np.maximum(np.diag(matrix), 0.0))
                return fixed, False
            eigvals = np.maximum(eigvals, 0)
            fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
            fixed = (fixed + fixed.T) / 2
            return fixed, False
        return matrix, False

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
            try:
                new_min = np.min(np.linalg.eigvalsh(fixed))
                logger.info(f"  Projected {name} to PSD (new min eigenvalue: {new_min:.2e})")
            except np.linalg.LinAlgError:
                logger.info(f"  Projected {name} to PSD (verification eigvalsh skipped)")

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
    sigma_ratio_max: Optional[float] = None,
    sigma_ratio_per_order: bool = False,
    variance_percentile_min: float = 67.0,
    variance_percentile_max: float = 85.0,
    variance_ratio_ref: float = 5.0,
    logger=None,
    apply_covariance_cap: bool = False,
    max_relative_std_cap: float = 1.0,
    forced_group_boundaries_mev: Optional[np.ndarray] = None,
    diagnostics_file: Optional[Path] = None,
    valid_orders_fn: Optional[Callable[[Any, int], int]] = None,
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
        Minimum l=1 adjacent correlation to merge (default 0.90)
    sigma_ratio_max : float, optional
        Optional cap on the running max(σ_l1)/min(σ_l1) within a group.
        ``None`` (default) disables the cap.  When set, prevents merging
        bins whose l=1 std differ by more than this factor — keeps groups
        physically homogeneous and avoids over-inflation by the percentile
        variance compensation downstream.
    variance_percentile_min : float
        Base percentile for groups with homogeneous variance (default 67).
    variance_percentile_max : float
        Maximum percentile for groups with highly heterogeneous variance
        (default 85).
    variance_ratio_ref : float
        Intra-group sigma ratio at which the percentile saturates at
        ``variance_percentile_max`` (default 5.0).
    logger : optional
        Logger for diagnostics
    valid_orders_fn : callable, optional
        ``f(nominal_result, max_order) -> int`` returning how many leading
        Legendre orders count as real parameters for that bin. Overrides the
        legacy winner-take-all rule ``min(frozen_degree, max_order)``.

        Defaults to ``None`` (legacy), so ``exfor_to_endf_sampling_v2.py`` is
        bit-unchanged. The research fork passes ``bin_valid_orders``, which keys
        on the inclusion probability ``q_l`` instead of the winning degree.

        **This parameter exists because its absence silently confined Phase 3 to
        the fine grid** — the fork had replaced the mask everywhere except here,
        so the shipped multigroup MF34 kept the winner-take-all zero structure
        while the fine one was fully populated. See the roadmap §8.4.

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

    # Build valid_mask for the non-interpolated fine grid.
    #
    # Legacy rule: a parameter (bin i, order l) is valid iff frozen_degree >= l,
    # i.e. winner-take-all. `valid_orders_fn` overrides it with the caller's own
    # rule (the research fork passes `bin_valid_orders`, keyed on the inclusion
    # probability q_l). It defaults to None so v2 stays bit-identical.
    #
    # ⚠ THIS BLOCK IS WHY PHASE 3 NEVER REACHED THE MULTIGROUP PRODUCT.
    # The fork replaced the mask on the *fine* path but this function rebuilt the
    # legacy one from scratch, so run 85's mixture populated the fine MF34
    # (dead parameters at l=6: 92.3% -> 1.0%) and left the multigroup MF34
    # untouched (81.6% -> 81.7%). Measured 2026-08-03; see the roadmap §8.4.
    _orders_of = valid_orders_fn or (lambda nr, mo: min(nr.frozen_degree, mo))
    valid_mask_mg = np.zeros(n_fine * max_order, dtype=bool)
    for i, nr in enumerate(valid_nominal):
        for l in range(1, min(int(_orders_of(nr, max_order)), max_order) + 1):
            valid_mask_mg[idx(i, l, max_order)] = True

    n_fitted = int(valid_mask_mg.sum())
    n_total = n_fine * max_order
    if logger:
        _rule = "caller-supplied" if valid_orders_fn else "legacy frozen_degree"
        logger.info(f"  Valid mask: {n_fitted}/{n_total} parameters fitted "
                    f"({n_total - n_fitted} unfitted, {100*(n_total-n_fitted)/n_total:.1f}%)"
                    f"  [rule: {_rule}]")

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
            _diag_below = (diagnostics_file.parent / (diagnostics_file.stem + "_below.csv")
                           if diagnostics_file else None)
            sub_groups, sub_info = find_adaptive_group_boundaries(
                corr_matrix=corr_matrix,
                cov_matrix=cov_matrix,
                fine_energies_mev=fine_energies_mev,
                sigma_E_mev=sigma_E_mev,
                fine_bin_widths_mev=fine_bin_widths_mev,
                max_order=max_order,
                rho_min=rho_min,
                sigma_ratio_max=sigma_ratio_max,
                sigma_ratio_per_order=sigma_ratio_per_order,
                logger=logger,
                diagnostics_file=_diag_below,
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
            _diag_above = (diagnostics_file.parent / (diagnostics_file.stem + "_above.csv")
                           if diagnostics_file else None)
            sub_groups, sub_info = find_adaptive_group_boundaries(
                corr_matrix=corr_matrix,
                cov_matrix=cov_matrix,
                fine_energies_mev=fine_energies_mev,
                sigma_E_mev=sigma_E_mev,
                fine_bin_widths_mev=fine_bin_widths_mev,
                max_order=max_order,
                rho_min=rho_min,
                sigma_ratio_max=sigma_ratio_max,
                sigma_ratio_per_order=sigma_ratio_per_order,
                logger=logger,
                diagnostics_file=_diag_above,
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
            sigma_ratio_per_order=sigma_ratio_per_order,
            logger=logger,
            diagnostics_file=diagnostics_file,
        )

    n_groups = len(groups)

    # Step 3: Build aggregation matrix
    A = build_aggregation_matrix(
        groups=groups,
        fine_bin_widths_mev=fine_bin_widths_mev,
        n_fine=n_fine,
        max_order=max_order,
        valid_mask=valid_mask_mg,
    )

    if logger:
        logger.info(f"  Aggregation matrix shape: {A.shape}")

    # Step 4: Collapse covariance and mean
    # Standard averaging collapse
    cov_grouped = collapse_covariance(cov_matrix, A)
    mean_grouped = collapse_mean(mean_fine, A)

    # Apply percentile-based variance scaling to preserve uncertainty magnitudes.
    # ``scale_factors`` (the diagonal of S) is exposed via MultigroupResult so
    # callers can apply the same compensation to coupled blocks (e.g. v3's
    # σ–a_l cross block, which is collapsed with the unscaled A and otherwise
    # ends up under-scaled relative to the variance-compensated a_l block).
    cov_grouped, scale_factors = apply_percentile_variance_scaling(
        cov_grouped=cov_grouped,
        cov_fine=cov_matrix,
        groups=groups,
        max_order=max_order,
        variance_percentile_min=variance_percentile_min,
        variance_percentile_max=variance_percentile_max,
        variance_ratio_ref=variance_ratio_ref,
        fine_bin_widths_mev=fine_bin_widths_mev,
        valid_mask=valid_mask_mg,
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

    # Compute valid_mask_grouped: True where at least one fine bin contributed
    _vmg = np.any(A != 0, axis=1)

    return MultigroupResult(
        group_boundaries_mev=group_boundaries_mev,
        group_boundaries_ev=group_boundaries_ev,
        aggregation_matrix=A,
        cov_grouped=cov_grouped,
        corr_grouped=corr_grouped,
        mean_grouped=mean_grouped,
        groups=groups,
        group_info=group_info,
        valid_mask_grouped=_vmg,
        scale_factors=scale_factors,
    )


# =============================================================================
# MF33 (cross-section magnitude) single-block collapse
# =============================================================================

@dataclass
class MF33MultigroupResult:
    """Result of the scalar (c0 / magnitude) adaptive collapse."""
    group_boundaries_mev: np.ndarray   # (N_groups + 1,)
    group_boundaries_ev: np.ndarray    # same in eV
    aggregation_matrix: np.ndarray     # A_0, (N_groups, N_fine), row-stochastic
    cov_rel_grouped: np.ndarray        # (N_groups, N_groups) relative
    cov_abs_grouped: np.ndarray        # (N_groups, N_groups) absolute
    means_grouped: np.ndarray          # (N_groups,) width-weighted host central
    groups: List[List[int]]
    group_info: List[GroupInfo]


def perform_mf33_multigroup_collapse(
    cov_abs_fine: np.ndarray,
    means_fine: np.ndarray,
    energy_bins: List,          # List[EnergyBinInfo], one per fine bin, in order
    *,
    rho_min: float = 0.85,
    sigma_ratio_max: Optional[float] = None,
    logger=None,
    diagnostics_file: Optional[Path] = None,
    label: str = "MF33 MT2",
) -> MF33MultigroupResult:
    """Adaptive multigroup collapse for the scalar MF33 magnitude channel.

    The MF34 collapse groups on the ``l=1`` correlation structure of a
    ``(n_fine * L)`` matrix.  MF33 is a single square block over one scalar per
    bin, so the very same machinery applies with ``max_order=1``: ``idx(i, 1, 1)``
    is just ``i``, which makes :func:`find_adaptive_group_boundaries` and the
    per-bin sigma/correlation diagnostics operate directly on the scalar matrix.

    The grid this produces is **independent of the MF34 grid** — the magnitude
    channel has its own correlation length, and ENDF lets every MF/MT section
    carry its own boundaries.  Use :func:`collapse_mf33_covariance_to_grid`
    instead when you deliberately want MF33 projected onto a *given* grid (the
    MF34 one, say).

    Collapse runs in **absolute** space and converts back with the collapsed
    means, which is the rigorous order when the central varies across merged
    bins:  ``C_abs_g = A0 C_abs A0^T``, ``m_g = A0 m``, ``C_rel_g = C_abs_g /
    outer(m_g)``.

    No percentile variance compensation is applied, matching what the pipeline
    actually publishes for MF34 (it rebuilds the multigroup covariance from
    ``A @ cov_abs @ A.T`` and never applies ``scale_factors``).

    Parameters
    ----------
    cov_abs_fine : np.ndarray
        ``(n_fine, n_fine)`` **absolute** covariance of the magnitude per bin.
    means_fine : np.ndarray
        ``(n_fine,)`` central value per bin, in the same units as the covariance
        (e.g. the TOF-folded host ``sigma_el``, or ``c0`` — the relative result
        is identical either way as long as both are consistent).
    energy_bins : List[EnergyBinInfo]
        One entry per row of ``cov_abs_fine``, in matching order.
    rho_min, sigma_ratio_max
        Merge gates, passed straight to :func:`find_adaptive_group_boundaries`.
    diagnostics_file : Path, optional
        Per-boundary merge decisions CSV (same schema as the MF34 one).

    Returns
    -------
    MF33MultigroupResult
    """
    C_abs = np.asarray(cov_abs_fine, dtype=float)
    means = np.asarray(means_fine, dtype=float)
    n_fine = len(energy_bins)

    if C_abs.shape != (n_fine, n_fine):
        raise ValueError(
            f"cov_abs_fine shape {C_abs.shape} doesn't match {n_fine} energy bins"
        )
    if means.shape != (n_fine,):
        raise ValueError(
            f"means_fine shape {means.shape} doesn't match {n_fine} energy bins"
        )

    fine_energies_mev = np.array([eb.energy_mev for eb in energy_bins], dtype=float)
    sigma_E_mev = np.array([eb.sigma_E_mev for eb in energy_bins], dtype=float)
    fine_bin_lower_mev = np.array([eb.bin_lower_mev for eb in energy_bins], dtype=float)
    fine_bin_upper_mev = np.array([eb.bin_upper_mev for eb in energy_bins], dtype=float)
    fine_bin_widths_mev = fine_bin_upper_mev - fine_bin_lower_mev

    corr_fine = cov_to_corr(C_abs)

    if logger:
        adj = np.array([corr_fine[i, i + 1] for i in range(n_fine - 1)])
        logger.info(
            f"  [{label}] Adaptive grid: {n_fine} fine bins, adjacent c0 corr "
            f"mean={np.mean(adj):.4f}, median={np.median(adj):.4f}, "
            f"min={np.min(adj):.4f}, max={np.max(adj):.4f}"
        )

    # max_order=1 -> idx(i, 1, 1) == i, so the MF34 grouper reads the scalar
    # matrix directly with no reshaping.
    groups, group_info = find_adaptive_group_boundaries(
        corr_matrix=corr_fine,
        cov_matrix=C_abs,
        fine_energies_mev=fine_energies_mev,
        sigma_E_mev=sigma_E_mev,
        fine_bin_widths_mev=fine_bin_widths_mev,
        max_order=1,
        rho_min=rho_min,
        sigma_ratio_max=sigma_ratio_max,
        logger=logger,
        diagnostics_file=diagnostics_file,
    )

    A0 = build_l0_row_aggregator(
        groups=groups,
        fine_bin_widths_mev=fine_bin_widths_mev,
        n_fine=n_fine,
    )

    cov_abs_grouped = collapse_covariance(C_abs, A0)
    means_grouped = A0 @ means

    good = np.isfinite(means_grouped) & (means_grouped > 0.0)
    cov_rel_grouped = np.zeros_like(cov_abs_grouped)
    if np.any(good):
        block = np.ix_(good, good)
        cov_rel_grouped[block] = (
            cov_abs_grouped[block] / np.outer(means_grouped[good], means_grouped[good])
        )
    if not np.all(good) and logger:
        logger.warning(
            f"  [{label}] {int(np.count_nonzero(~good))} group(s) have a "
            f"non-positive collapsed central; their relative rows are zeroed."
        )

    cov_rel_grouped = 0.5 * (cov_rel_grouped + cov_rel_grouped.T)
    cov_rel_grouped, _ = check_positive_semidefinite(
        cov_rel_grouped, name=f"{label} multigroup", logger=logger,
        fix_if_needed=False,
    )

    group_boundaries_mev = construct_group_energy_boundaries(
        groups=groups,
        fine_bin_lower_mev=fine_bin_lower_mev,
        fine_bin_upper_mev=fine_bin_upper_mev,
    )

    if logger:
        logger.info(
            f"  [{label}] Collapsed {n_fine} -> {len(groups)} groups "
            f"({n_fine / max(1, len(groups)):.1f}x), "
            f"[{group_boundaries_mev[0]:.4f}, {group_boundaries_mev[-1]:.4f}] MeV"
        )

    return MF33MultigroupResult(
        group_boundaries_mev=group_boundaries_mev,
        group_boundaries_ev=group_boundaries_mev * 1e6,
        aggregation_matrix=A0,
        cov_rel_grouped=cov_rel_grouped,
        cov_abs_grouped=cov_abs_grouped,
        means_grouped=means_grouped,
        groups=groups,
        group_info=group_info,
    )


def collapse_mf33_covariance_to_grid(
    native_grid_ev: np.ndarray,
    native_cov: np.ndarray,
    target_grid_ev: np.ndarray,
    *,
    native_means: Optional[np.ndarray] = None,
    is_relative: bool = True,
    weighting: str = "width",
    label: str = "MF33 MT2",
    logger=None,
) -> np.ndarray:
    """Collapse a single-block MF33 covariance onto target (multigroup) bin edges.

    MF33 is a single square self-block (one reaction, no Legendre triangle), so
    this reuses the width-weighted piecewise-constant overlap operator that
    ``kika.cov.cross_section_covariance.project_to_grid`` uses
    (``kika.processing.multigroup.compute_rebin_operator`` +
    ``collapse_covariance``) — the same MF34 multigroup boundaries can be passed
    as ``target_grid_ev`` so the MF33 and MF34 outputs share one grid.

    For a **relative** covariance the collapse is done in absolute space and
    converted back to relative on the target grid (rigorous); this needs the
    per-native-bin means (``native_means``, the σ per bin). Without means the
    relative matrix is collapsed directly, which is exact only in the flat-σ
    limit — documented, and logged when a logger is given.

    The final matrix gets a **warn-only** PSD check (never repaired), matching
    the pipeline's final-matrix PSD policy.

    Parameters
    ----------
    native_grid_ev : np.ndarray
        Native MF33 bin edges in eV (``N_native + 1``).
    native_cov : np.ndarray
        ``(N_native, N_native)`` covariance on the native grid.
    target_grid_ev : np.ndarray
        Target (multigroup) bin edges in eV (``N_target + 1``).
    native_means : np.ndarray, optional
        Per-native-bin σ (b). Required for a rigorous relative collapse.
    is_relative : bool, default True
        Whether ``native_cov`` is relative.
    weighting : str, default "width"
        Only ``"width"`` (piecewise-constant) is supported in Phase 1.
    label : str
        Name used in the PSD warning / log messages.
    logger : optional
        Logger for informational messages.

    Returns
    -------
    np.ndarray
        ``(N_target, N_target)`` collapsed covariance (relative if the input
        was relative), symmetric.
    """
    import warnings
    from kika.processing.multigroup import (
        compute_rebin_operator,
        collapse_covariance,
        relative_to_absolute,
        absolute_to_relative,
    )

    if weighting != "width":
        raise ValueError(
            f"weighting={weighting!r} not supported; only 'width' in Phase 1"
        )

    native_grid = np.asarray(native_grid_ev, dtype=float)
    target_grid = np.asarray(target_grid_ev, dtype=float)
    C = np.asarray(native_cov, dtype=float)

    n_native = native_grid.size - 1
    if C.shape != (n_native, n_native):
        raise ValueError(
            f"native_cov shape {C.shape} doesn't match native grid with "
            f"{n_native} intervals"
        )
    if target_grid.size < 2:
        raise ValueError("target_grid_ev must have at least 2 edges")

    # Row-stochastic width-weighted overlap operator, shape (N_target, N_native).
    M = compute_rebin_operator(native_grid, target_grid)

    if is_relative and native_means is not None:
        means = np.asarray(native_means, dtype=float)
        if means.shape != (n_native,):
            raise ValueError(
                f"native_means shape {means.shape} doesn't match {n_native} "
                f"native intervals"
            )
        C_abs = relative_to_absolute(C, means, means)
        C_tgt_abs = collapse_covariance(C_abs, M)
        means_tgt = M @ means  # width-weighted group σ

        # A target group whose native means are all non-positive has no central
        # value to be relative to.  ``absolute_to_relative`` flags that division
        # with NaN by design; letting those rows through is what fed NaNs to
        # ``np.linalg.eigvalsh`` in the ENDF writer and surfaced as the opaque
        # "Eigenvalues did not converge".  Zero the affected rows/columns here
        # and name the groups, instead of relying on a downstream sanitizer.
        good = np.isfinite(means_tgt) & (means_tgt > 0.0)
        C_tgt = np.zeros_like(C_tgt_abs)
        if np.any(good):
            block = np.ix_(good, good)
            C_tgt[block] = absolute_to_relative(
                C_tgt_abs[block], means_tgt[good], means_tgt[good],
            )
        n_dropped = int(np.count_nonzero(~good))
        if n_dropped and logger:
            spans = [
                f"[{target_grid[i]:.6g}, {target_grid[i + 1]:.6g}]"
                for i in np.flatnonzero(~good)
            ]
            shown = ", ".join(spans[:5]) + (" ..." if len(spans) > 5 else "")
            logger.warning(
                f"  {label}: {n_dropped}/{len(means_tgt)} target groups have a "
                f"non-positive mean and cannot carry a relative covariance; "
                f"their rows/columns are zeroed (eV): {shown}"
            )
    else:
        if is_relative and native_means is None and logger:
            logger.info(
                f"  {label}: collapsing relative covariance without means "
                f"(flat-σ limit; exact only if σ is constant across merged bins)"
            )
        C_tgt = collapse_covariance(C, M)

    C_tgt = 0.5 * (C_tgt + C_tgt.T)

    # Warn-only PSD check — never repair (final-matrix PSD policy).  Take the
    # returned matrix: check_positive_semidefinite sanitizes non-finite entries
    # on a copy, so discarding it (the historical ``_, was_psd = ...``) logged
    # "replacing with 0" while handing the caller the unsanitized matrix.
    C_tgt, was_psd = check_positive_semidefinite(
        C_tgt, name=label, logger=logger, fix_if_needed=False,
    )
    if not was_psd:
        warnings.warn(
            f"{label}: collapsed MF33 covariance is not PSD (negative "
            f"eigenvalue); left unrepaired per the warn-only policy.",
            RuntimeWarning,
            stacklevel=2,
        )

    return C_tgt
