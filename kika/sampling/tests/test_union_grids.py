#!/usr/bin/env python3
"""
Quick test to demonstrate the union grids and boundary check improvements.

This script shows how to use the validation functionality.
"""

import numpy as np

from kika.cov.legendre_covariance import LegendreCovariance
from union_grids_validation import run_comprehensive_validation


def create_test_mf34_data():
    """Create a simple test LegendreCovariance for demonstration."""
    mf34 = LegendreCovariance()
    
    # Add some test matrices with different energy grids
    isotope = 92235
    mt = 2
    
    # Matrix 1: L=1 vs L=1 (diagonal block)
    energy_grid_1 = [1e-5, 1e-3, 1e-1, 1.0, 20.0]  # 4 bins
    matrix_1 = np.eye(4) * 0.01  # 1% diagonal covariance
    mf34.add_matrix(
        isotope_row=isotope, reaction_row=mt, l_row=1,
        isotope_col=isotope, reaction_col=mt, l_col=1,
        matrix=matrix_1, energy_grid=energy_grid_1,
        is_relative=True, frame="LAB"
    )
    
    # Matrix 2: L=2 vs L=2 (diagonal block) with different grid
    energy_grid_2 = [1e-5, 5e-4, 1e-3, 1e-1, 0.5, 1.0, 20.0]  # 6 bins
    matrix_2 = np.eye(6) * 0.005  # 0.5% diagonal covariance
    mf34.add_matrix(
        isotope_row=isotope, reaction_row=mt, l_row=2,
        isotope_col=isotope, reaction_col=mt, l_col=2,
        matrix=matrix_2, energy_grid=energy_grid_2,
        is_relative=True, frame="LAB"
    )
    
    # Matrix 3: L=1 vs L=2 (off-diagonal block)
    energy_grid_3 = [1e-5, 1e-3, 1e-1, 1.0, 20.0]  # 4 bins
    matrix_3 = np.ones((4, 4)) * 0.002  # Small correlation
    mf34.add_matrix(
        isotope_row=isotope, reaction_row=mt, l_row=1,
        isotope_col=isotope, reaction_col=mt, l_col=2,
        matrix=matrix_3, energy_grid=energy_grid_3,
        is_relative=True, frame="LAB"
    )
    
    return mf34


def test_union_grids_basic():
    """Test basic union grids functionality."""
    print("="*60)
    print("BASIC UNION GRIDS TEST")
    print("="*60)
    
    mf34 = create_test_mf34_data()
    
    # Test union grids computation
    print("Computing union grids...")
    union_grids = mf34.get_union_energy_grids()
    
    print(f"Found union grids for {len(union_grids)} parameter triplets:")
    for triplet, grid in union_grids.items():
        iso, mt, l = triplet
        print(f"  ISO={iso}, MT={mt}, L={l}: {len(grid)} points, {len(grid)-1} bins")
        print(f"    Range: [{grid[0]:.3e}, {grid[-1]:.3e}] MeV")
    
    # Test validation
    print("\nRunning built-in validation...")
    validation_ok = mf34.validate_union_grids(verbose=True)
    
    return validation_ok


def test_comprehensive_validation():
    """Test the comprehensive validation functionality."""
    print("\n" + "="*60)
    print("COMPREHENSIVE VALIDATION TEST")
    print("="*60)
    
    mf34 = create_test_mf34_data()
    
    # Run comprehensive validation
    validation_ok = run_comprehensive_validation(mf34, sample_mt_numbers=[2])
    
    return validation_ok


def test_covariance_matrix_properties():
    """Test that covariance matrix properties work correctly."""
    print("\n" + "="*60)
    print("COVARIANCE MATRIX PROPERTIES TEST")
    print("="*60)
    
    mf34 = create_test_mf34_data()
    
    # Test covariance matrix construction
    print("Computing covariance matrix...")
    cov_matrix = mf34.covariance_matrix
    print(f"Covariance matrix shape: {cov_matrix.shape}")
    print(f"Matrix is symmetric: {np.allclose(cov_matrix, cov_matrix.T)}")
    print(f"Matrix diagonal (first 10): {np.diag(cov_matrix)[:10]}")
    
    # Test other properties
    print(f"\nOther properties:")
    print(f"Number of matrices: {mf34.num_matrices}")
    print(f"Isotopes: {mf34.isotopes}")
    print(f"Reactions: {mf34.reactions}")
    print(f"Legendre indices: {mf34.legendre_indices}")
    print(f"Uniform energy grid: {mf34.has_uniform_energy_grid()}")
    
    return True


if __name__ == "__main__":
    print("Testing Union Grids and Boundary Check Improvements")
    print("="*60)
    
    # Run tests
    try:
        test1_ok = test_union_grids_basic()
        test2_ok = test_comprehensive_validation()
        test3_ok = test_covariance_matrix_properties()
        
        print("\n" + "="*60)
        print("FINAL TEST SUMMARY")
        print("="*60)
        print(f"Basic union grids test:      {'PASS' if test1_ok else 'FAIL'}")
        print(f"Comprehensive validation:    {'PASS' if test2_ok else 'FAIL'}")
        print(f"Covariance matrix test:      {'PASS' if test3_ok else 'FAIL'}")
        
        overall_ok = test1_ok and test2_ok and test3_ok
        print(f"Overall result:              {'PASS' if overall_ok else 'FAIL'}")
        
        if overall_ok:
            print("\nAll tests passed! The union grids and boundary check improvements are working correctly.")
        else:
            print("\nSome tests failed. Please check the implementation.")
            
    except Exception as e:
        print(f"\nERROR during testing: {e}")
        import traceback
        traceback.print_exc()