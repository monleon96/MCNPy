"""Flux-weighted energy-group averaging of pointwise cross sections.

Pointwise comparisons in the resonance region are dominated by narrow
peaks whose positions differ slightly between evaluations, producing
large pointwise discrepancies with small integral impact. Group
averaging with a 1/E spectrum is the standard way to compare
evaluations in that region.

For each group ``g`` with boundaries ``[E_lo, E_hi]`` the average is:

    sigma_bar_g = integral sigma(E) * phi(E) dE / integral phi(E) dE

evaluated on the overlap of the group with the pointwise energy range.
The pointwise cross section is treated as piecewise linear between
tabulated points. For ``phi(E) = 1/E`` the integrals are evaluated in
lethargy space (``u = ln E``) where the weight reduces to a constant,
keeping numerator and denominator on identical numerics so that a
constant cross section reproduces itself exactly. For ``phi = 1`` the
integrals are evaluated directly in energy.
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

__all__ = ["resonance_group_average"]


Weighting = Literal["lethargy", "constant"]


def resonance_group_average(
    energies: np.ndarray,
    cross_sections: np.ndarray,
    group_boundaries: np.ndarray,
    weighting: Weighting = "lethargy",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute flux-weighted group averages of a pointwise cross section.

    Parameters
    ----------
    energies : np.ndarray
        Pointwise energies in eV, strictly increasing.
    cross_sections : np.ndarray
        Pointwise cross section values (e.g., barns), same shape as
        ``energies``.
    group_boundaries : np.ndarray
        Group bin edges (``n_bins + 1`` values), strictly increasing.
    weighting : {"lethargy", "constant"}
        Spectrum ``phi(E)`` used for weighting. ``"lethargy"`` (default)
        uses ``phi(E) = 1/E``.

    Returns
    -------
    group_edges : np.ndarray
        The input ``group_boundaries`` (returned for convenience, so
        callers can treat the function as the source of both edges and
        values).
    group_averages : np.ndarray
        Group-averaged cross sections, length ``n_bins``. Groups that
        fall entirely outside the pointwise energy range, or that have
        a degenerate weight integral, are set to ``NaN``.
    """
    energies = np.asarray(energies, dtype=float)
    cross_sections = np.asarray(cross_sections, dtype=float)
    group_boundaries = np.asarray(group_boundaries, dtype=float)

    if energies.ndim != 1 or cross_sections.ndim != 1:
        raise ValueError("energies and cross_sections must be 1-D arrays")
    if energies.shape != cross_sections.shape:
        raise ValueError("energies and cross_sections must have the same shape")
    if energies.size < 2:
        raise ValueError("need at least 2 pointwise energies")
    if not np.all(np.diff(energies) > 0):
        raise ValueError("energies must be strictly increasing")
    if group_boundaries.ndim != 1 or group_boundaries.size < 2:
        raise ValueError("group_boundaries must be a 1-D array with >=2 entries")
    if not np.all(np.diff(group_boundaries) > 0):
        raise ValueError("group_boundaries must be strictly increasing")
    if weighting not in ("lethargy", "constant"):
        raise ValueError(f"unknown weighting: {weighting!r}")
    if weighting == "lethargy" and energies[0] <= 0.0:
        raise ValueError("lethargy weighting requires strictly positive energies")

    n_groups = group_boundaries.size - 1
    averages = np.full(n_groups, np.nan, dtype=float)

    e_min = energies[0]
    e_max = energies[-1]

    for g in range(n_groups):
        e_lo = max(group_boundaries[g], e_min)
        e_hi = min(group_boundaries[g + 1], e_max)
        if e_hi <= e_lo:
            continue

        # Build an integration grid by inserting the clipped group
        # boundaries into the pointwise grid. Cross-section values at
        # the boundaries come from linear-in-E interpolation, matching
        # the piecewise-linear tabulation contract used by ENDF MF3
        # TAB1 records reconstructed to a fine grid.
        inner_mask = (energies > e_lo) & (energies < e_hi)
        inner_e = energies[inner_mask]
        inner_xs = cross_sections[inner_mask]
        xs_lo = float(np.interp(e_lo, energies, cross_sections))
        xs_hi = float(np.interp(e_hi, energies, cross_sections))

        e_grid = np.concatenate(([e_lo], inner_e, [e_hi]))
        xs_grid = np.concatenate(([xs_lo], inner_xs, [xs_hi]))

        if weighting == "lethargy":
            # Integrate in lethargy u = ln E; phi·dE = du, so the
            # numerator is trap(sigma, u) and the denominator is
            # exactly u_hi - u_lo. This keeps both consistent and
            # exact for constant sigma.
            u_grid = np.log(e_grid)
            numerator = float(np.trapezoid(xs_grid, u_grid))
            denominator = float(u_grid[-1] - u_grid[0])
        else:  # constant
            numerator = float(np.trapezoid(xs_grid, e_grid))
            denominator = float(e_hi - e_lo)

        if abs(denominator) < 1e-30:
            continue
        averages[g] = numerator / denominator

    return group_boundaries.copy(), averages
