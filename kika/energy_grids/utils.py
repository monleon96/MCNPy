"""
Utility functions for working with energy grids.
"""


import numpy as np
from typing import Optional, Tuple
from .grids import (
    VITAMINJ174, VITAMINJ175, SCALE44, SCALE56, SCALE238, SCALE252,
    WIMS69, ECCO33, ONEGROUP20, CASMO12, CASMO40, CASMO70, LANL30, XMAS172
)

def _identify_energy_grid(energies, tolerance=1e-5):
    """
    Compare an energy grid with known predefined grids to identify which standard grid it matches.
    
    :param energies: List or array of energy bin boundaries
    :type energies: list or numpy.ndarray
    :param tolerance: Relative tolerance for floating point comparison, defaults to 1e-5
    :type tolerance: float, optional
    :returns: Name of the matching grid or None if no match is found
    :rtype: str or None
    """
    if energies is None or len(energies) < 2:
        return None
    
    # Dictionary mapping grid variables to their names
    grid_dict = {
        "VITAMINJ174": VITAMINJ174,
        "VITAMINJ175": VITAMINJ175,
        "SCALE44": SCALE44,
        "SCALE56": SCALE56,
        "SCALE238": SCALE238,
        "SCALE252": SCALE252,
        "WIMS69": WIMS69,
        "ECCO33": ECCO33,
        "ONEGROUP20": ONEGROUP20,
        "CASMO12": CASMO12,
        "CASMO40": CASMO40,
        "CASMO70": CASMO70,
        "LANL30": LANL30,
        "XMAS172": XMAS172,
    }
    
    # Convert input to numpy array for comparison
    energy_array = np.array(energies)
    
    # Check each grid for match
    for grid_name, grid_values in grid_dict.items():
        # Skip if length doesn't match
        if len(energy_array) != len(grid_values):
            continue
        
        # Convert grid to numpy array
        grid_array = np.array(grid_values)
        
        # Calculate relative differences
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = np.abs((energy_array - grid_array) / grid_array)
            
        # Check if all values are within tolerance (or exactly equal for zeros)
        is_match = np.all(
            np.logical_or(
                rel_diff <= tolerance,
                np.logical_and(energy_array == 0, grid_array == 0)
            )
        )
        
        if is_match:
            return grid_name

    return None


# Complete grid dictionary for reuse
_ALL_GRIDS = {
    "VITAMINJ174": VITAMINJ174,
    "VITAMINJ175": VITAMINJ175,
    "SCALE44": SCALE44,
    "SCALE56": SCALE56,
    "SCALE238": SCALE238,
    "SCALE252": SCALE252,
    "WIMS69": WIMS69,
    "ECCO33": ECCO33,
    "ONEGROUP20": ONEGROUP20,
    "CASMO12": CASMO12,
    "CASMO40": CASMO40,
    "CASMO70": CASMO70,
    "LANL30": LANL30,
    "XMAS172": XMAS172,
}


def _identify_energy_grid_or_subset(
    energies, tolerance=1e-5
) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """
    Try exact match first, then check if energies is a contiguous subset of a preset grid.

    :param energies: List or array of energy bin boundaries.
    :param tolerance: Relative tolerance for floating point comparison.
    :returns: Tuple of (grid_name, range_min, range_max).
        - Exact match: (grid_name, None, None)
        - Subset match: (grid_name, first_energy, last_energy)
        - No match: (None, None, None)
    """
    if energies is None or len(energies) < 2:
        return (None, None, None)

    # Try exact match first
    exact = _identify_energy_grid(energies, tolerance)
    if exact is not None:
        return (exact, None, None)

    energy_array = np.array(energies)
    n = len(energy_array)

    for grid_name, grid_values in _ALL_GRIDS.items():
        grid_array = np.array(grid_values)
        if n > len(grid_array):
            continue  # Can't be a subset of a smaller grid

        # Find where energy_array[0] matches in grid_array
        for start_idx in range(len(grid_array) - n + 1):
            candidate = grid_array[start_idx : start_idx + n]

            with np.errstate(divide='ignore', invalid='ignore'):
                rel_diff = np.abs((energy_array - candidate) / candidate)

            is_match = np.all(
                np.logical_or(
                    rel_diff <= tolerance,
                    np.logical_and(energy_array == 0, candidate == 0)
                )
            )

            if is_match:
                return (grid_name, float(energy_array[0]), float(energy_array[-1]))

    return (None, None, None)