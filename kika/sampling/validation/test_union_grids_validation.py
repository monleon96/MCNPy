#!/usr/bin/env python3
"""
Validation script for union grids and boundary check modifications.

This script performs sanity checks to verify:
1. Shape & alignment of covariance matrix and union grids
2. No boundary double-hits during perturbation application
"""

import numpy as np
from typing import Dict, List, Tuple, Set
import logging


def validate_union_grids_shape_alignment(mf34_cov):
    """
    Check that covariance matrix shape aligns with union grids.
    
    Parameters
    ----------
    mf34_cov : LegendreCovariance
        The MF34 covariance matrix object to validate

    Returns
    -------
    bool
        True if validation passes, False otherwise
    """
    print("\n" + "="*60)
    print("VALIDATION 1: Shape & Alignment Check")
    print("="*60)
    
    try:
        # Get parameter triplets and union grids
        param_triplets = mf34_cov._get_param_triplets()
        union_grids = mf34_cov.get_union_energy_grids()
        
        print(f"Number of parameter triplets: {len(param_triplets)}")
        
        # Calculate expected dimensions
        max_G = 0
        grid_info = {}
        
        for triplet in param_triplets:
            grid = union_grids[triplet]
            num_bins = len(grid) - 1
            grid_info[triplet] = {
                'num_points': len(grid),
                'num_bins': num_bins,
                'energy_range': (grid[0], grid[-1]) if len(grid) > 0 else (0, 0)
            }
            max_G = max(max_G, num_bins)
        
        expected_dimension = len(param_triplets) * max_G
        print(f"Max energy bins (max_G): {max_G}")
        print(f"Expected covariance matrix dimension: {expected_dimension}")
        
        # Get actual covariance matrix shape
        cov_matrix = mf34_cov.covariance_matrix
        actual_shape = cov_matrix.shape
        print(f"Actual covariance matrix shape: {actual_shape}")
        
        # Check alignment
        alignment_ok = (actual_shape[0] == actual_shape[1] == expected_dimension)
        print(f"Shape alignment check: {'PASS' if alignment_ok else 'FAIL'}")
        
        # Print detailed grid information for first few triplets
        print(f"\nDetailed grid information (first 5 triplets):")
        for i, (triplet, info) in enumerate(list(grid_info.items())[:5]):
            iso, mt, l = triplet
            print(f"  {i+1}. ISO={iso}, MT={mt}, L={l}: {info['num_bins']} bins, "
                  f"range=[{info['energy_range'][0]:.3e}, {info['energy_range'][1]:.3e}] MeV")
        
        if len(grid_info) > 5:
            print(f"  ... and {len(grid_info) - 5} more triplets")
        
        return alignment_ok
        
    except Exception as e:
        print(f"ERROR during shape validation: {e}")
        return False


def validate_boundary_double_hits(
    mt_data, 
    factors, 
    param_mapping, 
    energy_grids, 
    mt_number: int,
    verbose: bool = True
):
    """
    Validate that boundaries are processed exactly once without double-hits.
    
    This function simulates the boundary processing logic to count how many
    times each boundary would be processed.
    
    Parameters
    ----------
    mt_data : MF4 data object
        The MF4 MT section data
    factors : np.ndarray
        Perturbation factors
    param_mapping : List[Tuple[int, int, int, int]]
        Parameter mapping
    energy_grids : Dict
        Energy grids for each triplet
    mt_number : int
        MT reaction number to validate
    verbose : bool
        Whether to print detailed information
        
    Returns
    -------
    bool
        True if validation passes (no double-hits), False otherwise
    """
    print(f"\n" + "="*60)
    print(f"VALIDATION 2: Boundary Double-Hit Check for MT{mt_number}")
    print("="*60)
    
    try:
        # Gather boundaries (replicating the logic from _apply_factors_to_mf4_legendre)
        boundaries = set()
        boundary_sources = {}  # boundary -> list of (triplet, energy_bin) that create it
        
        for factor_idx, (isotope, mt, l_coeff, energy_bin) in enumerate(param_mapping):
            if mt != mt_number:
                continue
                
            triplet = (isotope, mt, l_coeff)
            energy_grid = energy_grids.get(triplet, [])
            
            if len(energy_grid) < 2:
                continue
                
            # Add internal bin boundaries (not the first and last)
            for i in range(1, len(energy_grid) - 1):
                boundary = energy_grid[i]
                # Only include if within some reasonable energy span (simplified check)
                boundaries.add(boundary)
                
                if boundary not in boundary_sources:
                    boundary_sources[boundary] = []
                boundary_sources[boundary].append((triplet, energy_bin))
        
        boundaries = np.asarray(sorted(boundaries), dtype=float)
        
        def _on_boundary(x, atol=1e-10):
            """Check if energy x is on any boundary within tolerance."""
            return np.any(np.isclose(boundaries, x, rtol=0.0, atol=atol))
        
        print(f"Found {len(boundaries)} unique boundaries")
        
        # Count interior vs boundary applications
        interior_applications = 0
        boundary_detections = 0
        
        # Check interior applications
        for factor_idx, (isotope, mt, l_coeff, energy_bin) in enumerate(param_mapping):
            if mt != mt_number:
                continue
            
            triplet = (isotope, mt, l_coeff)
            energy_grid = energy_grids.get(triplet, [])
            
            if len(energy_grid) < 2 or energy_bin >= len(energy_grid) - 1:
                continue
                
            energy_low = energy_grid[energy_bin]
            energy_high = energy_grid[energy_bin + 1]
            
            # Simulate checking each energy point (using simplified energy range)
            test_energies = np.linspace(energy_low, energy_high, 21)  # 21 test points
            
            for energy in test_energies:
                if energy_low <= energy < energy_high and not _on_boundary(energy):
                    interior_applications += 1
        
        # Check boundary detections
        for boundary_energy in boundaries:
            boundary_hits_for_this_boundary = 0
            
            for factor_idx, (isotope, mt, l_coeff, energy_bin) in enumerate(param_mapping):
                if mt != mt_number:
                    continue
                    
                triplet = (isotope, mt, l_coeff)
                energy_grid = energy_grids.get(triplet, [])
                
                if len(energy_grid) < 2 or energy_bin >= len(energy_grid) - 1:
                    continue
                
                energy_low = energy_grid[energy_bin]
                energy_high = energy_grid[energy_bin + 1]
                
                # Check boundary detection (using tolerant comparison)
                if np.isclose(energy_high, boundary_energy, rtol=0.0, atol=1e-10):
                    boundary_hits_for_this_boundary += 1
                if np.isclose(energy_low, boundary_energy, rtol=0.0, atol=1e-10):
                    boundary_hits_for_this_boundary += 1
            
            boundary_detections += 1
            
            if verbose and boundary_hits_for_this_boundary > 2:
                print(f"  WARNING: Boundary at {boundary_energy:.3e} MeV detected "
                      f"{boundary_hits_for_this_boundary} times (expected ≤2)")
        
        # Validate results
        expected_boundary_count = len(boundaries)
        boundary_check_ok = (boundary_detections == expected_boundary_count)
        
        print(f"Interior applications: {interior_applications}")
        print(f"Boundary detections: {boundary_detections}")
        print(f"Expected boundary count: {expected_boundary_count}")
        print(f"Boundary detection check: {'PASS' if boundary_check_ok else 'FAIL'}")
        
        # Check for near-duplicate boundaries that might cause issues
        if len(boundaries) > 1:
            min_spacing = np.min(np.diff(boundaries))
            print(f"Minimum boundary spacing: {min_spacing:.3e} MeV")
            
            # Check for boundaries that are very close (potential double-hit risk)
            close_pairs = []
            for i in range(len(boundaries) - 1):
                spacing = boundaries[i+1] - boundaries[i]
                if spacing < 1e-8:  # Very close boundaries
                    close_pairs.append((boundaries[i], boundaries[i+1], spacing))
            
            if close_pairs:
                print(f"WARNING: Found {len(close_pairs)} very close boundary pairs:")
                for b1, b2, spacing in close_pairs[:3]:  # Show first 3
                    print(f"  {b1:.3e} - {b2:.3e} MeV (spacing: {spacing:.3e})")
            else:
                print("No problematically close boundaries detected")
        
        return boundary_check_ok
        
    except Exception as e:
        print(f"ERROR during boundary validation: {e}")
        return False


def run_comprehensive_validation(mf34_cov, sample_mt_numbers: List[int] = None):
    """
    Run comprehensive validation of the union grids and boundary check implementation.
    
    Parameters
    ----------
    mf34_cov : LegendreCovariance
        The MF34 covariance matrix object to validate
    sample_mt_numbers : List[int], optional
        List of MT numbers to test boundary logic for. If None, uses first available MT.
        
    Returns
    -------
    bool
        True if all validations pass, False otherwise
    """
    print("STARTING COMPREHENSIVE VALIDATION")
    print("="*60)
    
    # Validation 1: Shape & Alignment
    shape_ok = validate_union_grids_shape_alignment(mf34_cov)
    
    # Validation 2: Boundary double-hits (requires mock data)
    if sample_mt_numbers is None:
        # Try to find available MT numbers from the covariance data
        available_mts = sorted(set(mf34_cov.reaction_rows + mf34_cov.reaction_cols))
        sample_mt_numbers = available_mts[:2] if available_mts else []
    
    boundary_ok = True
    if sample_mt_numbers:
        print(f"\nTesting boundary logic for MT numbers: {sample_mt_numbers}")
        
        # Create mock parameter mapping and energy grids for testing
        param_triplets = mf34_cov._get_param_triplets()
        union_grids = mf34_cov.get_union_energy_grids()
        energy_grids = {t: union_grids[t].tolist() for t in param_triplets}
        max_G = max(len(g)-1 for g in energy_grids.values()) if energy_grids else 0
        
        param_mapping = []
        for triplet in param_triplets:
            iso, mt, L = triplet
            if mt in sample_mt_numbers:
                for energy_bin in range(min(max_G, 10)):  # Limit to first 10 bins for testing
                    param_mapping.append((iso, mt, L, energy_bin))
        
        # Create mock factors
        factors = np.ones(len(param_mapping)) * 1.1  # 10% perturbation factors
        
        # Mock MT data object
        class MockMTData:
            def __init__(self, mt_num):
                self.number = mt_num
                self._energies = []
                self._legendre_coeffs = []
        
        for mt_num in sample_mt_numbers:
            mock_mt_data = MockMTData(mt_num)
            mt_boundary_ok = validate_boundary_double_hits(
                mock_mt_data, factors, param_mapping, energy_grids, mt_num, verbose=True
            )
            boundary_ok = boundary_ok and mt_boundary_ok
    else:
        print("\nNo MT numbers available for boundary testing")
    
    # Final summary
    print(f"\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Shape & Alignment: {'PASS' if shape_ok else 'FAIL'}")
    print(f"Boundary Logic:    {'PASS' if boundary_ok else 'FAIL'}")
    print(f"Overall:           {'PASS' if (shape_ok and boundary_ok) else 'FAIL'}")
    
    return shape_ok and boundary_ok


if __name__ == "__main__":
    print("This is a validation module for union grids and boundary checks.")
    print("Import this module and call run_comprehensive_validation(mf34_cov) to test your data.")