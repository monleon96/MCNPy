"""
Adaptive energy grid generation (broken-stick linearization).

Generates an energy mesh where cross sections can be linearly interpolated
within a specified tolerance.
"""

import numpy as np
from typing import Callable, Optional


def linearize(sigma_func: Callable[[np.ndarray], np.ndarray],
              E_lo: float,
              E_hi: float,
              tol: float = 1e-3,
              initial_points: Optional[np.ndarray] = None,
              max_points: int = 500_000,
              min_spacing: float = 1e-6,
              errint: Optional[float] = None) -> np.ndarray:
    """Adaptive broken-stick linearization of sigma(E).

    Uses an iterative pass-based approach: on each pass, evaluate midpoints
    of all current segments in a single vectorized call, then insert those
    that exceed tolerance.  This is much faster than point-by-point evaluation.

    Parameters
    ----------
    sigma_func : callable
        Function E_array -> sigma_array (barns). Must accept 1-D numpy arrays.
    E_lo, E_hi : float
        Energy range (eV).
    tol : float
        Relative tolerance for linearization (default 0.1%).
    initial_points : array, optional
        Initial grid points to seed the algorithm (e.g. resonance energies).
    max_points : int
        Safety limit on total grid size.
    min_spacing : float
        Minimum spacing between adjacent points (eV).
    errint : float, optional
        Resonance integral tolerance.  After the main linearization loop,
        a Gauss-Legendre quadrature check catches narrow resonances that
        fall entirely between grid points.  Default: ``tol / 20000``.
        Set to 0 to disable.

    Returns
    -------
    ndarray
        Sorted energy grid where sigma is adequately linearized.
    """
    # Build initial grid
    pts = [E_lo, E_hi]
    if initial_points is not None:
        for p in initial_points:
            if E_lo < p < E_hi:
                pts.append(float(p))

    grid = np.array(sorted(set(pts)))
    sigma_vals = sigma_func(grid)

    # Iterative refinement: each pass checks all midpoints at once
    max_passes = 50
    for _ in range(max_passes):
        if len(grid) >= max_points:
            break

        # Compute all midpoints
        midpoints = 0.5 * (grid[:-1] + grid[1:])
        spacings = grid[1:] - grid[:-1]

        # Skip segments that are too narrow
        wide_enough = spacings >= min_spacing
        if not np.any(wide_enough):
            break

        # Evaluate sigma at all midpoints (single vectorized call)
        sigma_mid_exact = sigma_func(midpoints)

        # Linear interpolation estimates
        sigma_mid_linear = 0.5 * (sigma_vals[:-1] + sigma_vals[1:])

        # Relative error
        ref = np.maximum(np.abs(sigma_mid_exact), np.abs(sigma_mid_linear))
        ref = np.maximum(ref, 1e-30)
        rel_err = np.abs(sigma_mid_exact - sigma_mid_linear) / ref

        # Find segments that need refinement
        needs_refine = (rel_err > tol) & wide_enough

        if not np.any(needs_refine):
            break

        # Insert midpoints where needed — merge into existing grid
        insert_E = midpoints[needs_refine]
        insert_sigma = sigma_mid_exact[needs_refine]

        # Merge via sorted insertion (preserves cached sigma values)
        idx = np.searchsorted(grid, insert_E)
        grid = np.insert(grid, idx, insert_E)
        sigma_vals = np.insert(sigma_vals, idx, insert_sigma)

    # --- Resonance integral verification pass ---
    # Catches narrow resonances that fall entirely between grid points.
    # Compare trapezoidal integral vs 3-point Gauss-Legendre per segment.
    if errint is None:
        errint = tol / 20000.0  # NJOY default

    if errint > 0 and len(grid) > 1:
        gl_nodes = np.array([0.5 - np.sqrt(3.0 / 5.0) / 2.0,
                             0.5,
                             0.5 + np.sqrt(3.0 / 5.0) / 2.0])
        gl_weights = np.array([5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0])

        dE = grid[1:] - grid[:-1]
        trap_integral = 0.5 * (sigma_vals[:-1] + sigma_vals[1:]) * dE

        # Evaluate sigma at GL nodes for each segment
        gl_E = grid[:-1, None] + gl_nodes[None, :] * dE[:, None]  # (n_seg, 3)
        gl_sigma = np.array([sigma_func(gl_E[:, j]) for j in range(3)]).T
        gl_integral = np.sum(gl_sigma * gl_weights[None, :], axis=1) * dE

        int_err = np.abs(gl_integral - trap_integral)
        needs_split = int_err > errint * dE

        if np.any(needs_split):
            split_mids = 0.5 * (grid[:-1][needs_split] + grid[1:][needs_split])
            split_sigma = sigma_func(split_mids)
            idx = np.searchsorted(grid, split_mids)
            grid = np.insert(grid, idx, split_mids)
            sigma_vals = np.insert(sigma_vals, idx, split_sigma)

    return grid
