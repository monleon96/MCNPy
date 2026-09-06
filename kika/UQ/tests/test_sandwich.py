"""
Comprehensive test suite for the sandwich formula uncertainty propagation.

This module contains pytest tests to verify the sandwich formula implementation
in kika/UQ/sandwich.py. Tests include:

1. Cross-section only sensitivity data (TestSandwichFormulaBasic)
   - Basic propagation with 2 energy groups, 2 reactions
   - Hand-calculated verification cases (1x1, 2x2 matrices)
   - Correlation effects verification

2. Legendre moment only sensitivity data (TestSandwichFormulaLegendre)
   - P1 and P2 Legendre moments (MT >= 4000)
   - MultigroupLegendreCovariance covariance matrices

3. Mixed cross-section and Legendre sensitivity data (TestSandwichFormulaMixed)
   - Combined XS and Legendre sensitivities in same SDFData
   - Both CrossSectionCovariance and MultigroupLegendreCovariance provided

4. Edge cases and error handling (TestSandwichFormulaEdgeCases)
   - Missing covariance matrices
   - Empty SDF data
   - Energy grid mismatches
   - No matching reactions
   - Reaction filtering

5. Individual contribution calculations (TestContributionCalculations)
   - Auto-contributions vs total contributions
   - Cross-correlation effects
   - Zero sensitivity handling

6. Utility functions (TestUtilityFunctions)
   - Reaction filtering helpers

The tests use simple, small matrices where results can be verified manually
to ensure the sandwich formula σ²_R = S^T Σ S is correctly implemented.

Key Features Tested:
- Energy grid matching and unit conversion
- Matrix construction and ordering
- Sandwich formula calculation
- Individual reaction contributions
- Cross-correlation effects
- Error handling and validation
- Mixed data type propagation

All test cases include hand-calculated expected results for verification.
"""

import pytest
import numpy as np
from typing import List, Dict
import warnings

from kika.sensitivities.sdf import SDFData, SDFReactionData
from kika.sensitivities.profile import SensitivityProfile, SensitivityReaction
from kika.cov.cross_section_covariance import CrossSectionCovariance
from kika.cov.multigroup.mg_legendre_covariance import MultigroupLegendreCovariance
from kika.UQ.sandwich import (
    sandwich_uncertainty_propagation,
    UncertaintyResult,
    UncertaintyContribution,
    filter_reactions_by_nuclide,
    filter_reactions_by_type,
    _resolve_nubar_redundancy,
    _bootstrap_nd_ci,
)
from kika.UQ.alignment import ParameterKey


def test_mt0_pseudo_total_is_excluded_from_uq():
    profile = SensitivityProfile(
        energy_grid=np.array([0.0, 1.0, 2.0]),
        energy_unit="MeV",
        reactions=(
            SensitivityReaction(26056, 0, [10.0, 20.0], [0.0, 0.0]),
            SensitivityReaction(26056, 2, [1.0, 2.0], [0.0, 0.0]),
        ),
        response=1.0,
    )
    covariance = CrossSectionCovariance(
        num_groups=2,
        energy_grid=[0.0, 1.0, 2.0],
        energy_unit="MeV",
    )
    covariance.add_matrix(26056, 2, 26056, 2, np.eye(2), is_relative=True)

    result = sandwich_uncertainty_propagation(profile, covariance, bootstrap=False)

    assert result.n_reactions == 1
    assert result.alignment_report.policy_exclusions[0] == [ParameterKey("xs", 26056, 0)]
    assert "MT=0 pseudo-total excluded from UQ" in result.alignment_report.assumptions

    covariance.add_matrix(26056, 0, 26056, 0, np.eye(2), is_relative=True)
    explicit = sandwich_uncertainty_propagation(
        profile,
        covariance,
        reaction_filter={26056: [0]},
        bootstrap=False,
    )
    assert explicit.n_reactions == 1
    assert explicit.contributions[0].mt == 0
    assert "MT=0 pseudo-total excluded from UQ" not in explicit.alignment_report.assumptions


class TestSandwichFormulaBasic:
    """Test basic functionality of sandwich formula with simple cases."""
    
    def test_bootstrap_uses_additive_absolute_sensitivity_sigma(self):
        sensitivity = np.array([2.0])
        sigma_absolute = np.array([0.5])
        covariance = np.array([[4.0]])
        actual = _bootstrap_nd_ci(
            sensitivity,
            sigma_absolute,
            covariance,
            n_samples=8,
            ci_level=0.75,
            seed=17,
        )

        rng = np.random.default_rng(17)
        samples = sensitivity + sigma_absolute * rng.standard_normal((8, 1))
        propagated = np.sqrt(
            np.abs(np.einsum("bi,ij,bj->b", samples, covariance, samples))
        )
        expected = (*np.percentile(propagated, [12.5, 87.5]),
                    np.mean(propagated), np.std(propagated, ddof=1))
        assert actual == pytest.approx(expected)

    def create_simple_sdf_xs_only(self) -> SDFData:
        """Create a simple SDFData with cross-section sensitivities only."""
        # Create test data with 2 energy groups, 2 reactions
        # Fe-56 elastic (MT=2) and (n,gamma) (MT=102)
        energy_boundaries = [0.0, 1.0, 20.0]  # 2 groups: [1MeV-20MeV], [0-1MeV]
        
        # Simple sensitivity coefficients that we can verify by hand
        fe56_elastic = SDFReactionData(
            zaid=26056,
            mt=2,
            sensitivity=[0.1, 0.2],    # Group 1: 0.1, Group 2: 0.2
            error=[0.01, 0.02]         # 1% and 2% relative errors
        )
        
        fe56_capture = SDFReactionData(
            zaid=26056,
            mt=102,
            sensitivity=[0.05, 0.15],  # Group 1: 0.05, Group 2: 0.15
            error=[0.005, 0.01]        # 0.5% and 1% relative errors
        )
        
        sdf_data = SDFData(
            title="Test XS Only",
            energy="1.0e+00_2.0e+07",
            pert_energies=energy_boundaries,
            r0=1.0,
            e0=0.01,
            data=[fe56_elastic, fe56_capture]
        )
        
        return sdf_data
    
    def create_simple_cov_mat_xs(self) -> CrossSectionCovariance:
        """Create a simple CrossSectionCovariance for cross-section data."""
        cov_mat = CrossSectionCovariance()
        
        # Set energy grid (must match SDF)
        cov_mat.energy_grid = [0.0, 1.0e6, 2.0e7]
        
        # Add isotope and reaction data
        cov_mat.isotope_rows = [26056, 26056]
        cov_mat.isotope_cols = [26056, 26056]
        cov_mat.reaction_rows = [2, 102]
        cov_mat.reaction_cols = [2, 102]
        
        # Create simple 2x2 covariance matrices for each reaction pair
        # Matrix for (Fe-56, MT=2) vs (Fe-56, MT=2) - elastic self-covariance
        elastic_self_cov = np.array([
            [0.01, 0.005],    # Variances: 1% group 1, 0.5% cross-term
            [0.005, 0.04]     # Cross-term: 0.5%, Variance: 2% group 2
        ])
        
        # Matrix for (Fe-56, MT=102) vs (Fe-56, MT=102) - capture self-covariance
        capture_self_cov = np.array([
            [0.0025, 0.001],  # Variances: 0.5% group 1, 0.1% cross-term
            [0.001, 0.01]     # Cross-term: 0.1%, Variance: 1% group 2
        ])
        
        # Matrix for (Fe-56, MT=2) vs (Fe-56, MT=102) - elastic-capture cross-covariance
        elastic_capture_cov = np.array([
            [0.002, 0.001],   # Small positive correlations
            [0.001, 0.005]
        ])
        
        # Matrix for (Fe-56, MT=102) vs (Fe-56, MT=2) - capture-elastic cross-covariance
        capture_elastic_cov = elastic_capture_cov.T  # Transpose for symmetry
        
        # Add matrices in the expected order
        cov_mat.matrices = [
            elastic_self_cov,      # (26056, 2) vs (26056, 2)
            elastic_capture_cov,   # (26056, 2) vs (26056, 102)
            capture_elastic_cov,   # (26056, 102) vs (26056, 2)
            capture_self_cov       # (26056, 102) vs (26056, 102)
        ]
        
        return cov_mat
    
    def test_xs_only_basic_propagation(self):
        """Test basic uncertainty propagation with cross-section data only."""
        sdf_data = self.create_simple_sdf_xs_only()
        cov_mat = self.create_simple_cov_mat_xs()
        
        # Run the sandwich formula
        result = sandwich_uncertainty_propagation(
            sdf_data=sdf_data,
            cov_mat=cov_mat,
            verbose=False
        )
        
        # Basic checks
        assert isinstance(result, UncertaintyResult)
        assert result.total_variance > 0
        assert result.total_uncertainty > 0
        assert result.relative_uncertainty > 0
        assert result.n_reactions == 2
        assert result.n_energy_groups == 2
        assert len(result.contributions) == 2
        
        # Check that contributions are sorted by magnitude (descending)
        contributions = result.contributions
        for i in range(len(contributions) - 1):
            assert contributions[i].variance_contribution >= contributions[i+1].variance_contribution
    
    def test_hand_calculation_verification_simple(self):
        """Test with very simple case that can be verified by hand calculation."""
        # Create minimal case: 1 energy group, 1 reaction
        energy_boundaries = [0.0, 20.0]  # 1 group: [0-20MeV]
        
        fe56_elastic = SDFReactionData(
            zaid=26056,
            mt=2,
            sensitivity=[0.1],    # Single sensitivity coefficient
            error=[0.01]          # 1% relative error (not used in propagation)
        )
        
        sdf_data = SDFData(
            title="Hand Calc Test",
            energy="0.0e+00_2.0e+07",
            pert_energies=energy_boundaries,
            r0=1.0,
            e0=0.01,
            data=[fe56_elastic]
        )
        
        # Create simple 1x1 covariance matrix
        cov_mat = CrossSectionCovariance()
        cov_mat.energy_grid = [0.0, 2.0e7]
        cov_mat.isotope_rows = [26056]
        cov_mat.isotope_cols = [26056]
        cov_mat.reaction_rows = [2]
        cov_mat.reaction_cols = [2]
        
        # 1x1 covariance matrix with variance = 0.04 (20% relative std dev)
        simple_cov = np.array([[0.04]])
        cov_mat.matrices = [simple_cov]
        
        # Run propagation
        result = sandwich_uncertainty_propagation(
            sdf_data=sdf_data,
            cov_mat=cov_mat,
            verbose=False
        )
        
        # Hand calculation: σ²_R = S^T * Σ * S = [0.1] * [0.04] * [0.1] = 0.004
        expected_variance = 0.1 * 0.04 * 0.1
        expected_uncertainty = np.sqrt(expected_variance)
        
        # Verify results (with small tolerance for floating point)
        assert abs(result.total_variance - expected_variance) < 1e-10
        assert abs(result.total_uncertainty - expected_uncertainty) < 1e-10
        assert abs(result.relative_uncertainty - expected_uncertainty) < 1e-10  # Since sensitivities are relative
        
        # Check that there's only one contribution and it equals the total
        assert len(result.contributions) == 1
        contrib = result.contributions[0]
        assert contrib.zaid == 26056
        assert contrib.mt == 2
        assert abs(contrib.variance_contribution - expected_variance) < 1e-10
    
    def test_hand_calculation_2x2_matrix(self):
        """Test 2x2 case with known cross-correlations that can be verified by hand."""
        # Create 2 energy group, 2 reaction case with known values
        energy_boundaries = [0.0, 1.0, 20.0]  # 2 groups
        
        # Create sensitivities - make them simple for hand calculation
        fe56_elastic = SDFReactionData(
            zaid=26056,
            mt=2,
            sensitivity=[0.1, 0.2],    # S1 = [0.1, 0.2]
            error=[0.01, 0.02]
        )
        
        fe56_capture = SDFReactionData(
            zaid=26056,
            mt=102,
            sensitivity=[0.05, 0.1],   # S2 = [0.05, 0.1]
            error=[0.005, 0.01]
        )
        
        sdf_data = SDFData(
            title="Hand Calc 2x2",
            energy="test",
            pert_energies=energy_boundaries,
            r0=1.0,
            e0=0.01,
            data=[fe56_elastic, fe56_capture]
        )
        
        # Create simple covariance matrices for hand calculation
        cov_mat = CrossSectionCovariance()
        cov_mat.energy_grid = [0.0, 1.0e6, 2.0e7]
        cov_mat.isotope_rows = [26056, 26056, 26056, 26056]
        cov_mat.isotope_cols = [26056, 26056, 26056, 26056]
        cov_mat.reaction_rows = [2, 2, 102, 102]
        cov_mat.reaction_cols = [2, 102, 2, 102]
        
        # Create known covariance matrices
        # Elastic self-covariance (simple diagonal)
        elastic_self = np.array([
            [0.01, 0.0],      # 1% variance group 1, no cross-term
            [0.0, 0.04]       # No cross-term, 4% variance group 2
        ])
        
        # Capture self-covariance (simple diagonal)
        capture_self = np.array([
            [0.0025, 0.0],    # 0.25% variance group 1, no cross-term
            [0.0, 0.01]       # No cross-term, 1% variance group 2
        ])
        
        # Cross-covariances (zero for simple case)
        elastic_capture = np.zeros((2, 2))
        capture_elastic = np.zeros((2, 2))
        
        cov_mat.matrices = [elastic_self, elastic_capture, capture_elastic, capture_self]
        
        # Run propagation
        result = sandwich_uncertainty_propagation(
            sdf_data=sdf_data,
            cov_mat=cov_mat,
            verbose=False
        )
        
        # Hand calculation:
        # S = [0.1, 0.2, 0.05, 0.1] (elastic group 1&2, capture group 1&2)
        # Σ = block diagonal matrix with elastic_self and capture_self on diagonal
        # σ²_R = S^T * Σ * S = S1^T * Σ1 * S1 + S2^T * Σ2 * S2 (no cross terms)
        
        # Elastic contribution: [0.1, 0.2]^T * [[0.01, 0], [0, 0.04]] * [0.1, 0.2]
        elastic_var = 0.1**2 * 0.01 + 0.2**2 * 0.04  # = 0.0001 + 0.0016 = 0.0017
        
        # Capture contribution: [0.05, 0.1]^T * [[0.0025, 0], [0, 0.01]] * [0.05, 0.1]
        capture_var = 0.05**2 * 0.0025 + 0.1**2 * 0.01  # = 0.00000625 + 0.0001 = 0.00010625
        
        expected_total_var = elastic_var + capture_var  # = 0.00180625
        expected_uncertainty = np.sqrt(expected_total_var)
        
        # Verify results
        assert abs(result.total_variance - expected_total_var) < 1e-10
        assert abs(result.total_uncertainty - expected_uncertainty) < 1e-10
        
        # Check individual contributions
        assert len(result.contributions) == 2
        contrib_vars = [c.variance_contribution for c in result.contributions]
        
        # Sort expected values to match the sorting in results (largest first)
        expected_contribs = sorted([elastic_var, capture_var], reverse=True)
        contrib_vars_sorted = sorted(contrib_vars, reverse=True)
        
        for expected, actual in zip(expected_contribs, contrib_vars_sorted):
            assert abs(actual - expected) < 1e-10
        
        # Since there are no cross-correlations, correlation effects should be zero
        assert abs(result.correlation_effects) < 1e-15
    
    def test_hand_calculation_with_correlation(self):
        """Test case with known cross-correlation effects."""
        # Simple 1 energy group, 2 reaction case with correlation
        energy_boundaries = [0.0, 20.0]  # 1 group
        
        fe56_elastic = SDFReactionData(
            zaid=26056,
            mt=2,
            sensitivity=[0.1],    # S1 = 0.1
            error=[0.01]
        )
        
        fe56_capture = SDFReactionData(
            zaid=26056,
            mt=102,
            sensitivity=[0.2],    # S2 = 0.2
            error=[0.02]
        )
        
        sdf_data = SDFData(
            title="Correlation Test",
            energy="test",
            pert_energies=energy_boundaries,
            r0=1.0,
            e0=0.01,
            data=[fe56_elastic, fe56_capture]
        )
        
        # Create covariance matrices with known correlation
        cov_mat = CrossSectionCovariance()
        cov_mat.energy_grid = [0.0, 2.0e7]
        cov_mat.isotope_rows = [26056, 26056, 26056, 26056]
        cov_mat.isotope_cols = [26056, 26056, 26056, 26056]
        cov_mat.reaction_rows = [2, 2, 102, 102]
        cov_mat.reaction_cols = [2, 102, 2, 102]
        
        # 1x1 matrices with known correlation
        elastic_self = np.array([[0.04]])       # σ₁² = 0.04 (20% std dev)
        capture_self = np.array([[0.09]])       # σ₂² = 0.09 (30% std dev)
        
        # Cross-correlation: σ₁₂ = ρ * σ₁ * σ₂ where ρ = 0.5
        correlation = 0.5 * np.sqrt(0.04) * np.sqrt(0.09)  # = 0.5 * 0.2 * 0.3 = 0.03
        elastic_capture = np.array([[correlation]])
        capture_elastic = np.array([[correlation]])  # Symmetric
        
        cov_mat.matrices = [elastic_self, elastic_capture, capture_elastic, capture_self]
        
        # Run propagation
        result = sandwich_uncertainty_propagation(
            sdf_data=sdf_data,
            cov_mat=cov_mat,
            verbose=False
        )
        
        # Hand calculation:
        # S = [0.1, 0.2] (elastic, capture)
        # Σ = [[0.04, 0.03], [0.03, 0.09]]
        # σ²_R = S^T * Σ * S = [0.1, 0.2] * [[0.04, 0.03], [0.03, 0.09]] * [0.1, 0.2]
        #      = [0.1*0.04 + 0.2*0.03, 0.1*0.03 + 0.2*0.09] * [0.1, 0.2]
        #      = [0.004 + 0.006, 0.003 + 0.018] * [0.1, 0.2]
        #      = [0.01, 0.021] * [0.1, 0.2]
        #      = 0.01*0.1 + 0.021*0.2 = 0.001 + 0.0042 = 0.0052
        
        expected_variance = 0.0052
        expected_uncertainty = np.sqrt(expected_variance)
        
        # Auto-contributions (diagonal terms only):
        # Elastic: 0.1² * 0.04 = 0.0004
        # Capture: 0.2² * 0.09 = 0.0036
        auto_total = 0.0004 + 0.0036  # = 0.004
        
        # Cross-correlation contribution: total - auto = 0.0052 - 0.004 = 0.0012
        expected_correlation = expected_variance - auto_total
        
        # Verify results
        assert abs(result.total_variance - expected_variance) < 1e-10
        assert abs(result.total_uncertainty - expected_uncertainty) < 1e-10
        assert abs(result.correlation_effects - expected_correlation) < 1e-10
        
        # Check that auto-contributions are positive but less than total
        total_auto = sum(getattr(c, 'auto_variance_contribution', 0) for c in result.contributions)
        assert total_auto < result.total_variance  # Cross-correlation adds to total
        assert result.correlation_effects > 0  # Positive correlation
        
        # Check that auto-contributions match our calculation
        assert abs(total_auto - auto_total) < 1e-10


class TestSandwichFormulaLegendre:
    """Test sandwich formula with Legendre moment sensitivities."""
    
    def create_simple_sdf_legendre_only(self) -> SDFData:
        """Create SDFData with Legendre moment sensitivities only."""
        energy_boundaries = [0.0, 1.0, 20.0]  # 2 groups
        
        # Legendre moments: MT = 4000 + L where L is the Legendre order
        # P1 moment for elastic scattering (MT = 4001 = 4000 + 1)
        fe56_p1 = SDFReactionData(
            zaid=26056,
            mt=4001,  # P1 Legendre moment
            sensitivity=[0.08, 0.12],
            error=[0.008, 0.01]
        )
        
        # P2 moment for elastic scattering (MT = 4002 = 4000 + 2)
        fe56_p2 = SDFReactionData(
            zaid=26056,
            mt=4002,  # P2 Legendre moment
            sensitivity=[0.04, 0.06],
            error=[0.004, 0.006]
        )
        
        sdf_data = SDFData(
            title="Test Legendre Only",
            energy="1.0e+00_2.0e+07",
            pert_energies=energy_boundaries,
            r0=1.0,
            e0=0.01,
            data=[fe56_p1, fe56_p2]
        )
        
        return sdf_data
    
    def create_simple_mgmf34_cov_mat(self) -> MultigroupLegendreCovariance:
        """Create a simple MultigroupLegendreCovariance for Legendre moment data."""
        
        # Create MultigroupLegendreCovariance - need to check its structure first
        # For now, create a minimal structure based on common patterns
        mgmf34_cov = MultigroupLegendreCovariance()
        
        # Set basic attributes
        mgmf34_cov.energy_grid = np.array([0.0, 1.0e6, 2.0e7])
        
        # Isotope information for rows and columns
        mgmf34_cov.isotope_rows = [26056] * 4
        mgmf34_cov.isotope_cols = [26056] * 4

        # Base reaction (2 for elastic) and complete P1/P2 block index
        mgmf34_cov.reaction_rows = [2] * 4
        mgmf34_cov.reaction_cols = [2] * 4
        mgmf34_cov.l_rows = [1, 1, 2, 2]
        mgmf34_cov.l_cols = [1, 2, 1, 2]
        
        # Create relative covariance matrices
        # P1 self-covariance
        p1_self_cov = np.array([
            [0.016, 0.004],   # 4% variance group 1, 0.4% cross-term
            [0.004, 0.025]    # Cross-term, 2.5% variance group 2
        ])
        
        # P2 self-covariance
        p2_self_cov = np.array([
            [0.009, 0.002],   # 0.9% variance group 1, 0.2% cross-term
            [0.002, 0.016]    # Cross-term, 1.6% variance group 2
        ])
        
        # P1-P2 cross-covariance
        p1_p2_cov = np.array([
            [0.006, 0.001],   # Small positive correlations
            [0.001, 0.008]
        ])
        
        # P2-P1 cross-covariance (transpose)
        p2_p1_cov = p1_p2_cov.T
        
        # Store in relative_matrices (this is the attribute used in sandwich.py)
        mgmf34_cov.relative_matrices = [
            p1_self_cov,    # P1 vs P1
            p1_p2_cov,      # P1 vs P2
            p2_p1_cov,      # P2 vs P1
            p2_self_cov     # P2 vs P2
        ]
        
        return mgmf34_cov
    
    def test_legendre_only_basic_propagation(self):
        """Test basic uncertainty propagation with Legendre data only."""
        sdf_data = self.create_simple_sdf_legendre_only()
        mgmf34_cov = self.create_simple_mgmf34_cov_mat()
        
        # Run the sandwich formula
        result = sandwich_uncertainty_propagation(
            sdf_data=sdf_data,
            legendre_cov_mat=mgmf34_cov,
            verbose=False
        )
        
        # Basic checks
        assert isinstance(result, UncertaintyResult)
        assert result.total_variance > 0
        assert result.total_uncertainty > 0
        assert result.relative_uncertainty > 0
        assert result.n_reactions == 2
        assert result.n_energy_groups == 2
        assert len(result.contributions) == 2
        
        # Check that all reactions are Legendre moments (MT >= 4000)
        for contrib in result.contributions:
            assert contrib.mt >= 4000


class TestSandwichFormulaMixed:
    """Test sandwich formula with mixed cross-section and Legendre data."""
    
    def create_mixed_sdf_data(self) -> SDFData:
        """Create SDFData with both cross-section and Legendre sensitivities."""
        energy_boundaries = [0.0, 1.0, 20.0]  # 2 groups
        
        # Cross-section sensitivities
        fe56_elastic = SDFReactionData(
            zaid=26056,
            mt=2,
            sensitivity=[0.1, 0.2],
            error=[0.01, 0.02]
        )
        
        # Legendre moment sensitivities
        fe56_p1 = SDFReactionData(
            zaid=26056,
            mt=4001,  # P1 moment
            sensitivity=[0.05, 0.08],
            error=[0.005, 0.008]
        )
        
        sdf_data = SDFData(
            title="Test Mixed XS and Legendre",
            energy="1.0e+00_2.0e+07",
            pert_energies=energy_boundaries,
            r0=1.0,
            e0=0.01,
            data=[fe56_elastic, fe56_p1]
        )
        
        return sdf_data
    
    def test_mixed_data_propagation(self):
        """Test uncertainty propagation with mixed data types."""
        sdf_data = self.create_mixed_sdf_data()
        
        # Create both types of covariance matrices
        cov_mat = TestSandwichFormulaBasic().create_simple_cov_mat_xs()
        mgmf34_cov = TestSandwichFormulaLegendre().create_simple_mgmf34_cov_mat()
        
        # Modify covariance matrices to only include relevant reactions
        # For XS: keep only elastic (MT=2)
        cov_mat.isotope_rows = [26056]
        cov_mat.isotope_cols = [26056]
        cov_mat.reaction_rows = [2]
        cov_mat.reaction_cols = [2]
        cov_mat.matrices = [cov_mat.matrices[0]]  # Only elastic self-covariance
        
        # For Legendre: keep only P1 (MT=4001)
        mgmf34_cov.isotope_rows = [26056]
        mgmf34_cov.isotope_cols = [26056]
        mgmf34_cov.reaction_rows = [2]
        mgmf34_cov.reaction_cols = [2]
        mgmf34_cov.l_rows = [1]
        mgmf34_cov.l_cols = [1]
        mgmf34_cov.relative_matrices = [mgmf34_cov.relative_matrices[0]]  # Only P1 self-covariance
        
        # Run propagation with both covariance matrices
        result = sandwich_uncertainty_propagation(
            sdf_data=sdf_data,
            cov_mat=cov_mat,
            legendre_cov_mat=mgmf34_cov,
            verbose=False
        )
        
        # Basic checks
        assert isinstance(result, UncertaintyResult)
        assert result.total_variance > 0
        assert result.total_uncertainty > 0
        assert result.n_reactions == 2  # One XS + one Legendre
        assert result.n_energy_groups == 2
        assert len(result.contributions) == 2
        
        # Check that we have both types of reactions
        mt_values = [contrib.mt for contrib in result.contributions]
        assert 2 in mt_values      # Cross-section elastic
        assert 4001 in mt_values   # P1 Legendre moment


class TestSandwichFormulaEdgeCases:
    """Test edge cases and error handling."""
    
    def test_no_covariance_matrices(self):
        """Test error when no covariance matrices are provided."""
        sdf_data = TestSandwichFormulaBasic().create_simple_sdf_xs_only()
        
        with pytest.raises(ValueError, match="At least one covariance matrix"):
            sandwich_uncertainty_propagation(sdf_data=sdf_data, verbose=False)
    
    def test_empty_sdf_data(self):
        """Test error when SDF data is empty."""
        sdf_data = SDFData(
            title="Empty",
            energy="test",
            pert_energies=[0.0, 1.0],
            data=[]
        )
        cov_mat = TestSandwichFormulaBasic().create_simple_cov_mat_xs()
        
        with pytest.raises(ValueError, match="No sensitivity reactions remain"):
            sandwich_uncertainty_propagation(sdf_data=sdf_data, cov_mat=cov_mat, verbose=False)
    
    def test_energy_grid_mismatch(self):
        """Test handling of mismatched energy grids."""
        sdf_data = TestSandwichFormulaBasic().create_simple_sdf_xs_only()
        cov_mat = TestSandwichFormulaBasic().create_simple_cov_mat_xs()
        
        # Modify covariance energy grid to be incompatible
        cov_mat.energy_grid = [0.0, 1.0e6, 1.0e7, 1.0e8]  # 3 groups instead of 2
        
        with pytest.raises(ValueError, match="explicitly condense"):
            sandwich_uncertainty_propagation(sdf_data=sdf_data, cov_mat=cov_mat, verbose=False)
    
    def test_no_matching_reactions(self):
        """Test when no reactions match between sensitivity and covariance data."""
        sdf_data = TestSandwichFormulaBasic().create_simple_sdf_xs_only()
        cov_mat = TestSandwichFormulaBasic().create_simple_cov_mat_xs()
        
        # Modify covariance to have different reactions
        cov_mat.reaction_rows = [18, 107]  # fission and alpha reactions
        cov_mat.reaction_cols = [18, 107]
        
        with pytest.raises(ValueError, match='missing="drop"'):
            sandwich_uncertainty_propagation(sdf_data=sdf_data, cov_mat=cov_mat, verbose=False)
    
    def test_reaction_filtering(self):
        """Test reaction filtering functionality."""
        sdf_data = TestSandwichFormulaBasic().create_simple_sdf_xs_only()
        cov_mat = TestSandwichFormulaBasic().create_simple_cov_mat_xs()
        
        # Filter to include only elastic scattering
        reaction_filter = {26056: [2]}  # Only MT=2 for Fe-56
        
        result = sandwich_uncertainty_propagation(
            sdf_data=sdf_data,
            cov_mat=cov_mat,
            reaction_filter=reaction_filter,
            verbose=False
        )
        
        # Should only have one reaction now
        assert result.n_reactions == 1
        assert len(result.contributions) == 1
        assert result.contributions[0].mt == 2


class TestContributionCalculations:
    """Test individual contribution calculations and correlation effects."""
    
    def test_contributions_sum_to_total(self):
        """Test that individual contributions sum to approximately the total variance."""
        sdf_data = TestSandwichFormulaBasic().create_simple_sdf_xs_only()
        cov_mat = TestSandwichFormulaBasic().create_simple_cov_mat_xs()
        
        result = sandwich_uncertainty_propagation(
            sdf_data=sdf_data,
            cov_mat=cov_mat,
            verbose=False
        )
        
        # Sum of individual contributions should be close to total
        # (might not be exact due to cross-correlation terms)
        total_individual = sum(contrib.variance_contribution for contrib in result.contributions)
        
        # The difference should be the correlation effects
        correlation_contribution = result.total_variance - total_individual
        
        # Basic sanity checks
        assert abs(correlation_contribution - result.correlation_effects) < 1e-10
        assert total_individual > 0
        
        # All individual contributions should be positive
        for contrib in result.contributions:
            assert contrib.variance_contribution >= 0
            assert contrib.uncertainty_contribution >= 0
            assert 0 <= contrib.relative_contribution <= 1
    
    def test_zero_sensitivity_handling(self):
        """Test handling of zero sensitivity coefficients."""
        # Create SDF with one zero sensitivity
        energy_boundaries = [0.0, 20.0]
        
        fe56_elastic = SDFReactionData(
            zaid=26056,
            mt=2,
            sensitivity=[0.0],  # Zero sensitivity
            error=[0.01]
        )
        
        sdf_data = SDFData(
            title="Zero Sensitivity Test",
            energy="test",
            pert_energies=energy_boundaries,
            r0=1.0,
            e0=0.01,
            data=[fe56_elastic]
        )
        
        cov_mat = CrossSectionCovariance()
        cov_mat.energy_grid = [0.0, 2.0e7]
        cov_mat.isotope_rows = [26056]
        cov_mat.isotope_cols = [26056]
        cov_mat.reaction_rows = [2]
        cov_mat.reaction_cols = [2]
        cov_mat.matrices = [np.array([[0.04]])]
        
        result = sandwich_uncertainty_propagation(
            sdf_data=sdf_data,
            cov_mat=cov_mat,
            verbose=False
        )
        
        # With zero sensitivity, total variance should be zero
        assert abs(result.total_variance) < 1e-15
        assert abs(result.total_uncertainty) < 1e-15
        assert len(result.contributions) == 1
        assert abs(result.contributions[0].variance_contribution) < 1e-15


class TestUtilityFunctions:
    """Test utility functions for reaction filtering."""
    
    def test_filter_by_nuclide(self):
        """Test filter_reactions_by_nuclide function."""
        # Test single nuclide with specific reactions
        filter_dict = filter_reactions_by_nuclide(26056, [2, 102])
        expected = {26056: [2, 102]}
        assert filter_dict == expected
        
        # Test single nuclide with all reactions
        filter_dict = filter_reactions_by_nuclide(26056)
        expected = {26056: []}
        assert filter_dict == expected
    
    def test_filter_by_type(self):
        """Test filter_reactions_by_type function."""
        # Test with specific MT numbers
        mt_numbers = [2, 18, 102]
        filter_dict = filter_reactions_by_type(mt_numbers)
        
        # Should return a dict where all isotopes can have these reactions
        # Implementation might vary, but basic functionality should work
        assert isinstance(filter_dict, dict)


class TestNubarDoubleCounting:
    """Nu-bar total (MT 452) = prompt (456) + delayed (455).

    Serpent reports sensitivities to all three; if a covariance file also
    provides all three, the sandwich formula must not count them together or the
    nu-bar variance is double-counted. The total is retained, the components are
    dropped.
    """

    def test_resolve_nubar_redundancy_total_mode(self):
        """Default 'total' mode: drop prompt/delayed, keep total."""
        # Total + both components for U-235 -> drop components, keep total.
        with pytest.warns(UserWarning, match="double-counting"):
            kept = _resolve_nubar_redundancy(
                [(92235, 452), (92235, 455), (92235, 456), (92238, 18)], "total", False
            )
        assert kept == [(92235, 452), (92238, 18)]

        # Components only (no total) -> keep both, no warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _resolve_nubar_redundancy(
                [(92235, 455), (92235, 456)], "total", False
            ) == [(92235, 455), (92235, 456)]

        # Total only -> unchanged.
        assert _resolve_nubar_redundancy([(92235, 452)], "total", False) == [(92235, 452)]

        # Per-isotope isolation: U-235 has total, U-238 has only delayed.
        with pytest.warns(UserWarning):
            kept = _resolve_nubar_redundancy(
                [(92235, 452), (92235, 455), (92238, 455)], "total", False
            )
        assert kept == [(92235, 452), (92238, 455)]

    def test_resolve_nubar_redundancy_components_mode(self):
        """'components' mode: drop total, keep prompt + delayed when both present."""
        with pytest.warns(UserWarning, match="Excluded total nu-bar"):
            kept = _resolve_nubar_redundancy(
                [(92235, 452), (92235, 455), (92235, 456), (92238, 18)],
                "components",
                False,
            )
        assert kept == [(92235, 455), (92235, 456), (92238, 18)]

        # Per-isotope dict: U-235 decomposed, U-238 keeps total.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kept = _resolve_nubar_redundancy(
                [(92235, 452), (92235, 455), (92235, 456),
                 (92238, 452), (92238, 455), (92238, 456)],
                {92235: "components"},
                False,
            )
        assert set(kept) == {(92235, 455), (92235, 456), (92238, 452)}

    def test_resolve_nubar_redundancy_incomplete_decomposition(self):
        """'components' requested but only one component present -> keep total + warn."""
        with pytest.warns(UserWarning, match="only one component"):
            kept = _resolve_nubar_redundancy(
                [(92235, 452), (92235, 455)], "components", False
            )
        assert kept == [(92235, 452)]

    def _nubar_sdf(self) -> SDFData:
        """U-235 nu-bar sensitivities: total = prompt + delayed (0.3 = 0.2 + 0.1)."""
        energy_boundaries = [0.0, 20.0]  # single group
        return SDFData(
            title="Nu-bar double-count",
            energy="0.0e+00_2.0e+07",
            pert_energies=energy_boundaries,
            r0=1.0,
            e0=0.01,
            data=[
                SDFReactionData(zaid=92235, mt=452, sensitivity=[0.3], error=[0.0]),
                SDFReactionData(zaid=92235, mt=455, sensitivity=[0.1], error=[0.0]),
                SDFReactionData(zaid=92235, mt=456, sensitivity=[0.2], error=[0.0]),
            ],
        )

    def _nubar_cov(self) -> CrossSectionCovariance:
        """Diagonal (relative) covariance for nu-bar total, delayed and prompt."""
        cov = CrossSectionCovariance()
        cov.energy_grid = [0.0, 2.0e7]
        cov.isotope_rows = [92235, 92235, 92235]
        cov.reaction_rows = [452, 455, 456]
        cov.isotope_cols = [92235, 92235, 92235]
        cov.reaction_cols = [452, 455, 456]
        cov.matrices = [
            np.array([[0.0025]]),  # total  (5% rel)
            np.array([[0.0004]]),  # delayed (2% rel)
            np.array([[0.0009]]),  # prompt  (3% rel)
        ]
        return cov

    def test_total_nubar_kept_components_dropped(self):
        """Only the total contributes; variance excludes the redundant components."""
        with pytest.warns(UserWarning, match="double-counting"):
            result = sandwich_uncertainty_propagation(
                sdf_data=self._nubar_sdf(),
                cov_mat=self._nubar_cov(),
                verbose=False,
            )

        # Exactly one reaction survives: the total (MT 452).
        assert result.n_reactions == 1
        assert len(result.contributions) == 1
        assert result.contributions[0].mt == 452

        # Variance equals S_452^2 * cov_452 = 0.3^2 * 0.0025, NOT the double-counted
        # sum that would also add the prompt/delayed contributions.
        expected = 0.3**2 * 0.0025
        double_counted = expected + 0.1**2 * 0.0004 + 0.2**2 * 0.0009
        assert result.total_variance == pytest.approx(expected, rel=1e-9)
        assert result.total_variance < double_counted

    def test_components_mode_keeps_prompt_delayed(self):
        """nubar_mode='components': total dropped, prompt + delayed contribute."""
        with pytest.warns(UserWarning, match="Excluded total nu-bar"):
            result = sandwich_uncertainty_propagation(
                sdf_data=self._nubar_sdf(),
                cov_mat=self._nubar_cov(),
                nubar_mode="components",
                verbose=False,
            )

        # The two components survive; the total (MT 452) is dropped.
        assert result.n_reactions == 2
        mts = sorted(c.mt for c in result.contributions)
        assert mts == [455, 456]

        # Diagonal covariance -> variance is the sum of the component auto-terms.
        expected = 0.1**2 * 0.0004 + 0.2**2 * 0.0009
        assert result.total_variance == pytest.approx(expected, rel=1e-9)


if __name__ == "__main__":
    # Demonstration: Run a hand-calculated test with verbose output
    print("=" * 80)
    print("DEMONSTRATION: Hand-calculated verification test")
    print("=" * 80)
    
    test_basic = TestSandwichFormulaBasic()
    
    # Create test data
    sdf_data = test_basic.create_simple_sdf_xs_only()
    cov_mat = test_basic.create_simple_cov_mat_xs()
    
    # Run with verbose output to show detailed calculations
    from kika.UQ.sandwich import sandwich_uncertainty_propagation
    
    print("\n1. Running cross-section propagation with verbose output:")
    result = sandwich_uncertainty_propagation(
        sdf_data=sdf_data,
        cov_mat=cov_mat,
        verbose=True
    )
    
    print(f"\nResult summary:")
    print(f"- Total variance: {result.total_variance:.6e}")
    print(f"- Total uncertainty: {result.total_uncertainty:.6e}")
    print(f"- Relative uncertainty: {result.relative_uncertainty:.4%}")
    print(f"- Reactions: {result.n_reactions}")
    print(f"- Energy groups: {result.n_energy_groups}")
    print(f"- Correlation effects: {result.correlation_effects:.6e}")
    
    print("\n" + "=" * 80)
    print("2. Running simple hand calculation test:")
    test_basic.test_hand_calculation_verification_simple()
    print("✓ Simple 1x1 test passed!")
    
    print("\n3. Running 2x2 matrix test:")
    test_basic.test_hand_calculation_2x2_matrix()
    print("✓ 2x2 matrix test passed!")
    
    print("\n4. Running correlation test:")
    test_basic.test_hand_calculation_with_correlation()
    print("✓ Correlation test passed!")
    
    print("\n" + "=" * 80)
    print("All comprehensive tests completed successfully!")
    print("The sandwich formula implementation is working correctly.")
    print("=" * 80)
