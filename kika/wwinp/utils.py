"""Utility functions for WWINP processing.

Includes grid lookup helpers and pure-numpy ratio calculations.
"""

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Grid lookup functions
# ---------------------------------------------------------------------------

def get_closest_indices(
    grid: np.ndarray, value: float, atol: float = 1e-9
) -> np.ndarray:
    """Return bin indices that bracket *value* in a sorted *grid*.

    For a fine mesh grid with *N+1* boundary points, valid bin indices are
    ``0 .. N-1``.  If *value* falls exactly on a boundary that is not an
    edge, both adjacent bins are returned.
    """
    n = len(grid)
    if n < 2:
        return np.array([0])

    # Clamp to grid range
    if value <= grid[0]:
        return np.array([0])
    if value >= grid[-1]:
        return np.array([n - 2])

    idx = int(np.searchsorted(grid, value))

    # Exact hit on an interior boundary → return both adjacent bins
    if idx < n and np.isclose(grid[idx], value, atol=atol):
        if idx == 0:
            return np.array([0])
        if idx == n - 1:
            return np.array([n - 2])
        return np.array([idx - 1, idx])

    # Between two boundaries → single bin
    return np.array([max(idx - 1, 0)])


def get_range_indices(
    grid: np.ndarray, range_tuple: tuple[float, float]
) -> np.ndarray:
    """Return all bin indices whose interval overlaps ``[v_min, v_max]``."""
    v_min, v_max = range_tuple
    if v_min > v_max:
        raise ValueError(f"Invalid range: min {v_min} > max {v_max}")

    n = len(grid)
    if n < 2:
        return np.array([0])

    # Find first bin whose upper boundary > v_min
    lo = int(np.searchsorted(grid, v_min, side="right")) - 1
    lo = max(lo, 0)

    # Find last bin whose lower boundary < v_max
    hi = int(np.searchsorted(grid, v_max, side="left"))
    hi = min(hi, n - 2)

    if lo > hi:
        return np.array([], dtype=int)
    return np.arange(lo, hi + 1)


def get_bin_intervals(
    grid: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(starts, ends)`` boundary arrays for the given bin *indices*."""
    if indices.size == 0:
        return np.array([]), np.array([])
    starts = grid[indices]
    ends = grid[np.minimum(indices + 1, len(grid) - 1)]
    return starts, ends


# ---------------------------------------------------------------------------
# Header verification
# ---------------------------------------------------------------------------

def verify_and_correct(
    ni: int,
    nt: Optional[list[int]],
    ne: list[int],
    iv: int,
) -> tuple[int, Optional[list[int]], list[int]]:
    """Verify and correct header parameters.

    Removes particle types with zero energy or time bins and ensures
    array lengths are consistent with *ni*.
    """
    if iv == 2 and nt is not None:
        min_len = min(len(nt), len(ne), ni)
        nt = list(nt[:min_len])
        ne = list(ne[:min_len])
        ni = min_len
    else:
        min_len = min(len(ne), ni)
        ne = list(ne[:min_len])
        ni = min_len

    # Remove particles with zero energy bins
    keep = [i for i in range(ni) if ne[i] > 0]
    if iv == 2 and nt is not None:
        keep = [i for i in keep if nt[i] > 0]
        nt = [nt[i] for i in keep]
    ne = [ne[i] for i in keep]
    ni = len(ne)

    return ni, nt, ne


# ---------------------------------------------------------------------------
# Ratio calculations (pure numpy — no Numba)
# ---------------------------------------------------------------------------

def calculate_max_ratio_array(arr: np.ndarray) -> np.ndarray:
    """For each cell, compute the maximum neighbour / centre ratio.

    Parameters
    ----------
    arr : np.ndarray
        3-D array of shape ``(nz, ny, nx)``.

    Returns
    -------
    np.ndarray
        Same shape, where each element is ``max(neighbour_values) / centre``.
        Cells with zero centre value get ratio 1.0.
    """
    ratios = np.ones_like(arr, dtype=np.float64)
    nonzero = arr != 0

    for axis in range(3):
        # Forward neighbour
        fwd = np.empty_like(arr)
        fwd[:] = 0
        slc_src = [slice(None)] * 3
        slc_dst = [slice(None)] * 3
        slc_src[axis] = slice(1, None)
        slc_dst[axis] = slice(None, -1)
        fwd[tuple(slc_dst)] = arr[tuple(slc_src)]

        # Backward neighbour
        bwd = np.empty_like(arr)
        bwd[:] = 0
        slc_src2 = [slice(None)] * 3
        slc_dst2 = [slice(None)] * 3
        slc_src2[axis] = slice(None, -1)
        slc_dst2[axis] = slice(1, None)
        bwd[tuple(slc_dst2)] = arr[tuple(slc_src2)]

        for nb in (fwd, bwd):
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(nonzero, nb / arr, 1.0)
            np.maximum(ratios, r, out=ratios)

    return ratios


def calculate_ratios_stats(arr: np.ndarray) -> tuple[float, float]:
    """Compute average and max direction-invariant neighbour ratio.

    Each adjacent pair is counted once (forward neighbours only).
    Ratio = max(a, b) / min(a, b), so it is always >= 1.

    Returns ``(average_ratio, max_ratio)``.  If no valid pairs exist,
    returns ``(nan, nan)``.
    """
    all_ratios: list[np.ndarray] = []

    for axis in range(3):
        slc_lo = [slice(None)] * 3
        slc_hi = [slice(None)] * 3
        slc_lo[axis] = slice(None, -1)
        slc_hi[axis] = slice(1, None)

        a = arr[tuple(slc_lo)]
        b = arr[tuple(slc_hi)]

        both_pos = (a > 0) & (b > 0)
        if not np.any(both_pos):
            continue

        a_pos = a[both_pos]
        b_pos = b[both_pos]
        hi = np.maximum(a_pos, b_pos)
        lo = np.minimum(a_pos, b_pos)
        all_ratios.append(hi / lo)

    if not all_ratios:
        return float("nan"), float("nan")

    combined = np.concatenate(all_ratios)
    return float(np.mean(combined)), float(np.max(combined))
