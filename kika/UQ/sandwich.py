"""Sandwich formula for uncertainty propagation in nuclear data.

This module implements the sandwich formula σ²_R = S^T Σ S for propagating nuclear data 
uncertainties from sensitivity coefficients and covariance matrices.

The sandwich formula allows propagation of uncertainties from nuclear cross-section 
covariances to integral responses through sensitivity coefficients.
"""

from typing import Dict, List, Tuple, Optional, Union, Set
from dataclasses import dataclass, field
import numpy as np
import warnings
import logging

from kika.sensitivities.sdf import SDFData, SDFReactionData
from kika.cov.cross_section_covariance import CrossSectionCovariance
from kika.cov.multigroup.mg_legendre_covariance import MultigroupLegendreCovariance
from kika._constants import MT_TO_REACTION, ATOMIC_NUMBER_TO_SYMBOL

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class UncertaintyContribution:
    """Container for individual uncertainty contributions from specific reactions.
    
    Attributes
    ----------
    zaid : int
        ZAID of the nuclide
    mt : int
        MT reaction number
    variance_contribution : float
        Contribution to total variance from this reaction
    uncertainty_contribution : float
        Square root of variance contribution (1-sigma)
    relative_contribution : float
        Relative contribution to total variance (fraction)
    nuclide : str
        Nuclide symbol (e.g., 'Fe-56')
    reaction_name : str
        Reaction name (e.g., 'elastic')
    """
    zaid: int
    mt: int
    variance_contribution: float
    uncertainty_contribution: float = field(init=False)
    relative_contribution: float = field(init=False)
    nuclide: str = field(init=False)
    reaction_name: str = field(init=False)
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        self.uncertainty_contribution = np.sqrt(abs(self.variance_contribution))
        
        # Calculate nuclide symbol
        z = self.zaid // 1000
        a = self.zaid % 1000
        
        if z in ATOMIC_NUMBER_TO_SYMBOL:
            self.nuclide = f"{ATOMIC_NUMBER_TO_SYMBOL[z]}-{a}"
        else:
            self.nuclide = f"Z{z}-{a}"
            
        # Calculate reaction name
        if self.mt in MT_TO_REACTION:
            self.reaction_name = MT_TO_REACTION[self.mt]
        else:
            self.reaction_name = f"MT{self.mt}"


@dataclass
class UncertaintyResult:
    """Container for uncertainty propagation results.

    Attributes
    ----------
    total_variance : float
        Total propagated variance
    total_uncertainty : float
        Total uncertainty (1-sigma, square root of variance)
    relative_uncertainty : float
        Relative uncertainty (σ/μ)
    response_value : float
        Reference response value used for relative uncertainty
    response_error : float
        MCNP tally relative error on the unperturbed response. Reported as a
        diagnostic only — it is *not* combined with sigma_ND in this function.
    contributions : List[UncertaintyContribution]
        Individual reaction contributions sorted by magnitude
    n_reactions : int
        Number of reactions included in propagation
    n_energy_groups : int
        Number of energy groups used
    correlation_effects : float
        Contribution from cross-correlations between reactions
    bootstrap_ci_low, bootstrap_ci_high : float, optional
        Lower / upper bound of the bootstrap CI on sigma_ND (relative). Both None
        when bootstrap was disabled.
    bootstrap_mean, bootstrap_std : float, optional
        Mean and std of the bootstrap distribution of sigma_ND (relative).
    bootstrap_n_samples : int, optional
        Number of bootstrap samples drawn. None when bootstrap was disabled.
    ci_level : float
        Confidence level used for the bootstrap CI (default 0.95).
    """
    total_variance: float
    total_uncertainty: float
    relative_uncertainty: float
    response_value: float
    response_error: float
    contributions: List[UncertaintyContribution]
    n_reactions: int
    n_energy_groups: int
    correlation_effects: float = 0.0
    bootstrap_ci_low: Optional[float] = None
    bootstrap_ci_high: Optional[float] = None
    bootstrap_mean: Optional[float] = None
    bootstrap_std: Optional[float] = None
    bootstrap_n_samples: Optional[int] = None
    ci_level: float = 0.95

    def __repr__(self) -> str:
        """Format uncertainty results for display."""
        lines = []
        lines.append("=" * 80)
        lines.append("NUCLEAR DATA UNCERTAINTY PROPAGATION (Sandwich Formula)")
        lines.append("=" * 80)

        nuclear_data_abs = self.relative_uncertainty * abs(self.response_value)
        nuclear_data_rel_pct = self.relative_uncertainty * 100

        lines.append("")
        lines.append(f"Response value:                    {self.response_value:.6e}")
        lines.append("")
        lines.append("NUCLEAR DATA UNCERTAINTY  (sigma_ND = sqrt(S^T C S))")
        lines.append(f"  Nominal:                         ± {nuclear_data_rel_pct:.3f}%")
        lines.append(f"  Absolute:                        ± {nuclear_data_abs:.6e}")

        if self.bootstrap_n_samples is not None and self.bootstrap_ci_low is not None:
            ci_pct = int(round(self.ci_level * 100))
            lines.append(f"  Bootstrap mean:                  ± {self.bootstrap_mean * 100:.3f}%")
            lines.append(
                f"  {ci_pct}% CI (bootstrap, N={self.bootstrap_n_samples}):"
                f"{'':<6}[{self.bootstrap_ci_low * 100:.3f}%, {self.bootstrap_ci_high * 100:.3f}%]"
            )

        lines.append("")
        lines.append("PROPAGATION DETAILS")
        lines.append(f"  Reactions:                       {self.n_reactions}")
        lines.append(f"  Energy groups:                   {self.n_energy_groups}")
        lines.append(f"  Total variance:                  {self.total_variance:.6e}")

        # Always show correlation effects (even if zero)
        if abs(self.correlation_effects) > 1e-15:
            corr_pct = abs(self.correlation_effects) / abs(self.total_variance) * 100 if abs(self.total_variance) > 1e-15 else 0.0
            lines.append(f"  Cross-reaction correlations:     {self.correlation_effects:.6e} ({corr_pct:.1f}% of total)")
        else:
            lines.append(f"  Cross-reaction correlations:     None (single reaction only)")

        lines.append("")
        lines.append("DIAGNOSTIC")
        lines.append(
            f"  MCNP tally rel_err (response):   {self.response_error * 100:.3f}%"
            "   -- info only; not propagated here."
        )
        
        lines.append("\n" + "=" * 70)
        lines.append("INDIVIDUAL REACTION CONTRIBUTIONS")
        lines.append("=" * 70)
        
        # Calculate total auto-contributions for percentage calculation
        total_auto_variance = sum(getattr(c, 'auto_variance_contribution', c.variance_contribution) 
                                for c in self.contributions)
        
        # Show both types of contributions
        lines.append("SINGLE-REACTION VARIANCE (includes energy-to-energy correlations):")
        lines.append(f"{'Rank':<4} {'Nuclide':<12} {'Reaction':<15} {'MT':<6} {'Variance':<12} {'% Auto':<8}")
        lines.append("-" * 68)
        
        # Sort by auto-contributions
        auto_sorted = sorted(self.contributions, 
                           key=lambda x: abs(getattr(x, 'auto_variance_contribution', x.variance_contribution)), 
                           reverse=True)
        
        for rank, contrib in enumerate(auto_sorted, 1):
            auto_var = getattr(contrib, 'auto_variance_contribution', contrib.variance_contribution)
            auto_pct = abs(auto_var) / abs(total_auto_variance) * 100 if abs(total_auto_variance) > 1e-15 else 0.0
            lines.append(f"{rank:<4} {contrib.nuclide:<12} {contrib.reaction_name:<15} {contrib.mt:<6} "
                        f"{auto_var:.4e} {auto_pct:>6.2f}%")
        
        lines.append(f"\nSum of single-reaction variances: {total_auto_variance:.6e}")
        lines.append("")
        
        lines.append("TOTAL VARIANCE (includes cross-reaction correlations):")
        lines.append(f"{'Rank':<4} {'Nuclide':<12} {'Reaction':<15} {'MT':<6} {'Variance':<12} {'% Total':<8}")
        lines.append("-" * 68)
        
        # Sort contributions by total magnitude
        sorted_contribs = sorted(self.contributions, 
                               key=lambda x: abs(x.variance_contribution), 
                               reverse=True)
        
        for rank, contrib in enumerate(sorted_contribs, 1):
            pct = contrib.relative_contribution * 100
            lines.append(f"{rank:<4} {contrib.nuclide:<12} {contrib.reaction_name:<15} {contrib.mt:<6} "
                        f"{contrib.variance_contribution:.4e} {pct:>6.2f}%")
        
        lines.append(f"\nTotal variance: {self.total_variance:.6e}")
        off_diagonal_contribution = self.total_variance - total_auto_variance
        off_diagonal_pct = abs(off_diagonal_contribution) / abs(self.total_variance) * 100 if abs(self.total_variance) > 1e-15 else 0.0
        lines.append(f"Cross-REACTION correlation: {off_diagonal_contribution:.6e} ({off_diagonal_pct:.1f}% of total)")
        
        lines.append("=" * 70)
        
        # Add summary interpretation
        if self.relative_uncertainty > 1.0:  # > 100%
            lines.append("⚠️  WARNING: Very high relative uncertainty detected!")
            lines.append("   This may indicate incompatible sensitivity/covariance data")
            lines.append("   or issues with absolute/relative conversion.")
        elif self.relative_uncertainty > 0.5:  # > 50%
            lines.append("⚠️  High relative uncertainty - please verify results")
        elif self.relative_uncertainty > 0.1:  # > 10%
            lines.append("✓ Moderate uncertainty level")
        else:
            lines.append("✓ Low uncertainty level")
            
        return "\n".join(lines)


def _format_nuclide(zaid: int) -> str:
    """Format a ZAID as a human-readable nuclide string."""
    z = zaid // 1000
    a = zaid % 1000

    if z in ATOMIC_NUMBER_TO_SYMBOL:
        return f"{ATOMIC_NUMBER_TO_SYMBOL[z]}-{a}"
    else:
        return f"Z{z}-{a}"


def _bootstrap_nd_ci(
    sensitivity_vector: np.ndarray,
    sigma_s_rel_vector: np.ndarray,
    covariance_matrix: np.ndarray,
    n_samples: int,
    ci_level: float,
    seed: Optional[int],
) -> Tuple[float, float, float, float]:
    """Bootstrap a confidence interval on sigma_ND = sqrt(S^T C S).

    Draws S_b = S0 * (1 + sigma_S_rel * eps_b), eps_b ~ N(0, I), then computes
    sigma_ND,b = sqrt(|S_b^T C S_b|) for each draw. Returns the (low, high) percentile
    bounds, the bootstrap mean of sigma_ND, and its standard deviation.

    The per-bin error sigma_S_rel is the MCNP PERT relative error on each sensitivity
    coefficient (i.e. SDFReactionData.error). Bins with sigma_S_rel == 0 (e.g. missing
    or perfectly converged) reduce to the nominal S0 in every draw.
    """
    rng = np.random.default_rng(seed)
    # eps_b ~ N(0, I): shape (n_samples, len(S))
    eps = rng.standard_normal((n_samples, sensitivity_vector.size))
    # S_b = S0 * (1 + sigma_s_rel * eps) -- broadcasts row-wise
    s_samples = sensitivity_vector[None, :] * (1.0 + sigma_s_rel_vector[None, :] * eps)
    # sigma_ND,b = sqrt(|S_b^T C S_b|), vectorised over b
    variances = np.einsum('bi,ij,bj->b', s_samples, covariance_matrix, s_samples)
    sigmas = np.sqrt(np.abs(variances))

    alpha = (1.0 - ci_level) / 2.0
    low, high = np.percentile(sigmas, [100 * alpha, 100 * (1.0 - alpha)])
    return float(low), float(high), float(np.mean(sigmas)), float(np.std(sigmas, ddof=1))


def _merge_covariance_matrices(
    sens_data: List[SDFReactionData],
    cov_mat_list: List[CrossSectionCovariance],
    verbose: bool
) -> Tuple[Optional[CrossSectionCovariance], Set[int]]:
    """Merge multiple CrossSectionCovariance objects into one, checking for missing isotopes.
    
    Parameters
    ----------
    sens_data : List[SDFReactionData]
        Sensitivity data to check against
    cov_mat_list : List[CrossSectionCovariance]
        List of covariance matrices to merge
    verbose : bool
        Whether to print detailed information
        
    Returns
    -------
    merged_cov_mat : CrossSectionCovariance or None
        Merged covariance matrix, or None if no data found
    missing_isotopes : Set[int]
        Set of ZAIDs that were not found in any covariance matrix
    """
    if not cov_mat_list:
        return None, set()
    
    # Get all isotopes needed from sensitivity data
    needed_isotopes = {r.zaid for r in sens_data}
    
    # Get all available isotopes from covariance matrices
    available_isotopes = set()
    for cov_mat in cov_mat_list:
        reactions_by_iso = cov_mat.reactions_by_isotope()
        available_isotopes.update(reactions_by_iso.keys())
    
    # Find missing isotopes
    missing_isotopes = needed_isotopes - available_isotopes
    
    if verbose and len(cov_mat_list) > 1:
        logger.info(f"  Merging {len(cov_mat_list)} covariance matrices...")
        logger.info(f"  Found data for {len(available_isotopes)} isotopes")
    
    # If only one matrix, return it directly
    if len(cov_mat_list) == 1:
        return cov_mat_list[0], missing_isotopes
    
    # Merge multiple matrices - combine all data into first matrix
    # All matrices should have the same energy grid (will be validated later)
    base_mat = cov_mat_list[0]
    
    # Create new lists to hold merged data
    merged_isotope_rows = list(base_mat.isotope_rows)
    merged_reaction_rows = list(base_mat.reaction_rows)
    merged_isotope_cols = list(base_mat.isotope_cols)
    merged_reaction_cols = list(base_mat.reaction_cols)
    merged_matrices = list(base_mat.matrices)
    
    # Add data from other matrices
    for cov_mat in cov_mat_list[1:]:
        # Check if matrices need to be added (avoid duplicates)
        for i in range(len(cov_mat.isotope_rows)):
            key = (cov_mat.isotope_rows[i], cov_mat.reaction_rows[i],
                   cov_mat.isotope_cols[i], cov_mat.reaction_cols[i])
            
            # Check if this combination already exists
            already_exists = False
            for j in range(len(merged_isotope_rows)):
                existing_key = (merged_isotope_rows[j], merged_reaction_rows[j],
                              merged_isotope_cols[j], merged_reaction_cols[j])
                if key == existing_key:
                    already_exists = True
                    break
            
            if not already_exists:
                merged_isotope_rows.append(cov_mat.isotope_rows[i])
                merged_reaction_rows.append(cov_mat.reaction_rows[i])
                merged_isotope_cols.append(cov_mat.isotope_cols[i])
                merged_reaction_cols.append(cov_mat.reaction_cols[i])
                merged_matrices.append(cov_mat.matrices[i])
    
    # Create merged CrossSectionCovariance object
    merged_cov_mat = CrossSectionCovariance(
        energy_grid=base_mat.energy_grid,
        isotope_rows=merged_isotope_rows,
        reaction_rows=merged_reaction_rows,
        isotope_cols=merged_isotope_cols,
        reaction_cols=merged_reaction_cols,
        matrices=merged_matrices
    )
    
    return merged_cov_mat, missing_isotopes


def _merge_legendre_covariance_matrices(
    sens_data: List[SDFReactionData],
    cov_mat_list: List[MultigroupLegendreCovariance],
    verbose: bool
) -> Tuple[Optional[MultigroupLegendreCovariance], Set[int]]:
    """Merge multiple MultigroupLegendreCovariance objects into one, checking for missing isotopes.
    
    Parameters
    ----------
    sens_data : List[SDFReactionData]
        Sensitivity data to check against
    cov_mat_list : List[MultigroupLegendreCovariance]
        List of Legendre covariance matrices to merge
    verbose : bool
        Whether to print detailed information
        
    Returns
    -------
    merged_cov_mat : MultigroupLegendreCovariance or None
        Merged Legendre covariance matrix, or None if no data found
    missing_isotopes : Set[int]
        Set of ZAIDs that were not found in any covariance matrix
    """
    if not cov_mat_list:
        return None, set()
    
    # Get all isotopes needed from sensitivity data
    needed_isotopes = {r.zaid for r in sens_data}
    
    # Get all available isotopes from Legendre covariance matrices
    available_isotopes = set()
    for cov_mat in cov_mat_list:
        available_isotopes.update(cov_mat.isotope_rows)
        available_isotopes.update(cov_mat.isotope_cols)
    
    # Find missing isotopes
    missing_isotopes = needed_isotopes - available_isotopes
    
    if verbose and len(cov_mat_list) > 1:
        logger.info(f"  Merging {len(cov_mat_list)} Legendre covariance matrices...")
        logger.info(f"  Found data for {len(available_isotopes)} isotopes")
    
    # If only one matrix, return it directly
    if len(cov_mat_list) == 1:
        return cov_mat_list[0], missing_isotopes
    
    # Merge multiple matrices
    base_mat = cov_mat_list[0]
    
    # Create new lists to hold merged data
    merged_isotope_rows = list(base_mat.isotope_rows)
    merged_reaction_rows = list(base_mat.reaction_rows)
    merged_isotope_cols = list(base_mat.isotope_cols)
    merged_reaction_cols = list(base_mat.reaction_cols)
    merged_matrices = list(base_mat.relative_matrices)
    
    # Add data from other matrices
    for cov_mat in cov_mat_list[1:]:
        # Check if matrices need to be added (avoid duplicates)
        for i in range(len(cov_mat.isotope_rows)):
            key = (cov_mat.isotope_rows[i], cov_mat.reaction_rows[i],
                   cov_mat.isotope_cols[i], cov_mat.reaction_cols[i])
            
            # Check if this combination already exists
            already_exists = False
            for j in range(len(merged_isotope_rows)):
                existing_key = (merged_isotope_rows[j], merged_reaction_rows[j],
                              merged_isotope_cols[j], merged_reaction_cols[j])
                if key == existing_key:
                    already_exists = True
                    break
            
            if not already_exists:
                merged_isotope_rows.append(cov_mat.isotope_rows[i])
                merged_reaction_rows.append(cov_mat.reaction_rows[i])
                merged_isotope_cols.append(cov_mat.isotope_cols[i])
                merged_reaction_cols.append(cov_mat.reaction_cols[i])
                merged_matrices.append(cov_mat.relative_matrices[i])
    
    # Create merged MultigroupLegendreCovariance object
    merged_cov_mat = MultigroupLegendreCovariance(
        energy_grid=base_mat.energy_grid,
        isotope_rows=merged_isotope_rows,
        reaction_rows=merged_reaction_rows,
        isotope_cols=merged_isotope_cols,
        reaction_cols=merged_reaction_cols,
        relative_matrices=merged_matrices,
        legendre_indices=base_mat.legendre_indices
    )
    
    return merged_cov_mat, missing_isotopes


def sandwich_uncertainty_propagation(
    sdf_data: SDFData,
    cov_mat: Optional[Union[CrossSectionCovariance, List[CrossSectionCovariance]]] = None,
    legendre_cov_mat: Optional[Union[MultigroupLegendreCovariance, List[MultigroupLegendreCovariance]]] = None,
    reaction_filter: Optional[Dict[int, List[int]]] = None,
    energy_tolerance: float = 1e-6,
    verbose: bool = False,
    bootstrap: bool = True,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    bootstrap_seed: Optional[int] = None,
) -> UncertaintyResult:
    """
    Apply the sandwich formula σ²_R = S^T Σ S to propagate nuclear data uncertainties.
    
    This function automatically handles:
    - Energy grid matching and validation
    - Conversion from absolute to relative sensitivities (to match relative covariances)
    - Matrix construction and sandwich formula application
    - Individual contribution analysis and cross-correlation effects
    - Multiple covariance matrices for different isotopes
    
    The function supports both cross-section and Legendre moment sensitivities:
    - Cross-section sensitivities (MT < 1000) use the regular CrossSectionCovariance covariance matrix
    - Legendre moment sensitivities (MT >= 4000) use the MultigroupLegendreCovariance covariance matrix
    - Both types can be provided simultaneously for mixed propagation
    
    Parameters
    ----------
    sdf_data : SDFData
        Sensitivity data containing sensitivity coefficients for various reactions.
        Can contain both cross-section sensitivities (MT < 1000) and Legendre 
        moment sensitivities (MT >= 4000, where MT = 4000 + L order).
    cov_mat : CrossSectionCovariance or List[CrossSectionCovariance], optional
        Covariance matrix data for nuclear cross sections (in relative form).
        Used for cross-section sensitivities (MT < 1000).
        Can be a single CrossSectionCovariance or a list of CrossSectionCovariance objects for different isotopes.
        When a list is provided, the function will search for each isotope's data
        across all provided matrices.
    legendre_cov_mat : MultigroupLegendreCovariance or List[MultigroupLegendreCovariance], optional
        Covariance matrix data for Legendre moments (in relative form).
        Used for Legendre moment sensitivities (MT >= 4000).
        Can be a single matrix or a list of matrices for different isotopes.
    reaction_filter : Dict[int, List[int]], optional
        Dictionary mapping ZAID to list of MT numbers to include in propagation.
        If None, all matching reactions between sensitivity and covariance data are used.
        Example: {26056: [2, 102]} includes only elastic and (n,γ) for Fe-56
                 {26056: [4001, 4002]} includes only P1 and P2 Legendre moments
    energy_tolerance : float, optional
        Tolerance for matching energy grid boundaries (default: 1e-6)
    verbose : bool, optional
        Print detailed information about the propagation process
    bootstrap : bool, optional
        If True (default), compute a bootstrap CI on sigma_ND that propagates the
        per-bin MC error on the sensitivity coefficients (SDFReactionData.error,
        from MCNP PERT) into the propagated nuclear-data uncertainty.
    n_bootstrap : int, optional
        Number of bootstrap samples to draw (default: 1000).
    ci_level : float, optional
        Confidence level for the bootstrap interval (default: 0.95).
    bootstrap_seed : int, optional
        Seed for the bootstrap RNG. If None (default), draws are non-deterministic
        (percentile drift across runs is sub-percent at N=1000 for typical inputs).

    Returns
    -------
    UncertaintyResult
        Complete uncertainty propagation results including individual contributions
        and cross-correlation effects
        
    Raises
    ------
    ValueError
        If neither covariance matrix is provided, if energy grids don't match, 
        or if no matching reactions are found
    
    Notes
    -----
    The function assumes:
    - Sensitivity coefficients in SDF data are in relative form (dR/R)/(dσ/σ)
    - Covariance matrices are in relative form (as typically provided by SCALE/NJOY)
    - Energy grids are compatible between sensitivity and covariance data
    
    The sandwich formula σ²_R = S^T Σ S gives the relative variance of the response,
    from which the relative uncertainty is calculated as σ_R = sqrt(σ²_R).
    This relative uncertainty represents the fractional uncertainty in the response
    due to nuclear data uncertainties.
    """
    
    if verbose:
        logger.info("Starting sandwich uncertainty propagation")
        logger.info("=" * 60)
    
    # Step 1: Validate inputs
    if not sdf_data.data:
        raise ValueError("SDF data contains no sensitivity information")
    
    if cov_mat is None and legendre_cov_mat is None:
        raise ValueError("At least one covariance matrix (cov_mat or legendre_cov_mat) must be provided")
    
    # Convert single matrices to lists for uniform handling
    cov_mat_list = [cov_mat] if isinstance(cov_mat, CrossSectionCovariance) else (cov_mat if cov_mat is not None else [])
    legendre_cov_mat_list = [legendre_cov_mat] if isinstance(legendre_cov_mat, MultigroupLegendreCovariance) else (legendre_cov_mat if legendre_cov_mat is not None else [])
    
    # Validate that provided covariance matrices contain data
    for i, cm in enumerate(cov_mat_list):
        if not cm.matrices:
            raise ValueError(f"Cross-section covariance matrix {i} contains no data")
    
    for i, lcm in enumerate(legendre_cov_mat_list):
        if not lcm.relative_matrices:
            raise ValueError(f"Legendre covariance matrix {i} contains no data")
    
    if verbose:
        logger.info("✓ Input validation complete")
        if cov_mat_list:
            total_matrices = sum(len(cm.matrices) for cm in cov_mat_list)
            logger.info(f"  Cross-section covariances: {len(cov_mat_list)} files with {total_matrices} total matrices")
        if legendre_cov_mat_list:
            total_matrices = sum(len(lcm.relative_matrices) for lcm in legendre_cov_mat_list)
            logger.info(f"  Legendre covariances: {len(legendre_cov_mat_list)} files with {total_matrices} total matrices")
    
    # Separate sensitivities by type
    xs_sensitivities = [r for r in sdf_data.data if r.mt < 4000]  # Cross-section sensitivities
    leg_sensitivities = [r for r in sdf_data.data if r.mt >= 4000]  # Legendre sensitivities
    
    # Check for and exclude MT=1 (total cross section) to avoid double-counting
    mt1_found = [r for r in xs_sensitivities if r.mt == 1]
    if mt1_found:
        xs_sensitivities = [r for r in xs_sensitivities if r.mt != 1]
    
    if verbose:
        logger.info(f"  Found {len(xs_sensitivities)} cross-section sensitivities (excluding MT=1)")
        logger.info(f"  Found {len(leg_sensitivities)} Legendre moment sensitivities")
    
    # We'll handle each type separately and then combine
    results = []
    
    # Step 2a: Handle cross-section sensitivities
    if xs_sensitivities and cov_mat_list:
        if verbose:
            logger.info("Processing cross-section sensitivities...")
        
        # Merge covariance matrices and check for missing isotopes
        merged_cov_mat, missing_isotopes = _merge_covariance_matrices(
            xs_sensitivities,
            cov_mat_list,
            verbose
        )
        
        # Warn about missing isotopes
        if missing_isotopes:
            missing_str = ", ".join([_format_nuclide(zaid) for zaid in missing_isotopes])
            warnings.warn(
                f"No covariance data found for the following isotopes in cross-section sensitivities: {missing_str}. "
                "These isotopes will be excluded from uncertainty propagation."
            )
            if verbose:
                logger.warning(f"⚠ Missing covariance data for {len(missing_isotopes)} isotopes: {missing_str}")
        
        if merged_cov_mat is not None:
            # Match energy grids for cross-sections
            xs_energy_mapping = _match_energy_grids(
                sdf_data.pert_energies, 
                merged_cov_mat.energy_grid,
                energy_tolerance,
                verbose
            )
            
            if not xs_energy_mapping:
                raise ValueError("No matching energy groups found between sensitivity and cross-section covariance data")
            
            # Find matching cross-section reactions
            xs_matching_reactions = _find_matching_reactions(
                xs_sensitivities, 
                merged_cov_mat, 
                reaction_filter,
                verbose,
                "cross-section"
            )
            
            if xs_matching_reactions:
                # Build matrices for cross-sections
                xs_result = {
                    'type': 'cross_section',
                    'energy_mapping': xs_energy_mapping,
                    'matching_reactions': xs_matching_reactions,
                    'sdf_data': xs_sensitivities,
                    'cov_mat': merged_cov_mat
                }
                results.append(xs_result)
                
                if verbose:
                    logger.info(f"✓ Cross-section processing complete: {len(xs_matching_reactions)} reactions")
    
    # Step 2b: Handle Legendre sensitivities  
    if leg_sensitivities and legendre_cov_mat_list:
        if verbose:
            logger.info("Processing Legendre moment sensitivities...")
        
        # Merge Legendre covariance matrices and check for missing isotopes
        merged_leg_cov_mat, missing_leg_isotopes = _merge_legendre_covariance_matrices(
            leg_sensitivities,
            legendre_cov_mat_list,
            verbose
        )
        
        # Warn about missing isotopes
        if missing_leg_isotopes:
            missing_str = ", ".join([_format_nuclide(zaid) for zaid in missing_leg_isotopes])
            warnings.warn(
                f"No Legendre covariance data found for the following isotopes: {missing_str}. "
                "These isotopes will be excluded from uncertainty propagation."
            )
            if verbose:
                logger.warning(f"⚠ Missing Legendre covariance data for {len(missing_leg_isotopes)} isotopes: {missing_str}")
        
        if merged_leg_cov_mat is not None:
            # Match energy grids for Legendre coefficients
            leg_energy_mapping = _match_energy_grids(
                sdf_data.pert_energies, 
                merged_leg_cov_mat.energy_grid,
                energy_tolerance,
                verbose
            )
            
            if not leg_energy_mapping:
                raise ValueError("No matching energy groups found between sensitivity and Legendre covariance data")
            
            # Find matching Legendre reactions
            leg_matching_reactions = _find_matching_legendre_reactions(
                leg_sensitivities, 
                merged_leg_cov_mat, 
                reaction_filter,
                verbose
            )
            
            if leg_matching_reactions:
                # Build matrices for Legendre moments
                leg_result = {
                    'type': 'legendre',
                    'energy_mapping': leg_energy_mapping,
                    'matching_reactions': leg_matching_reactions,
                    'sdf_data': leg_sensitivities,
                    'cov_mat': merged_leg_cov_mat
                }
                results.append(leg_result)
                
                if verbose:
                    logger.info(f"✓ Legendre processing complete: {len(leg_matching_reactions)} reactions")
    
    # Check if we have any valid results
    if not results:
        raise ValueError("No matching reactions found between sensitivities and any provided covariance data")
    
    # Step 3: Build combined matrices
    if verbose:
        logger.info("Building combined sensitivity vector and covariance matrix...")
    
    sensitivity_vector, covariance_matrix, sigma_s_rel_vector, reaction_indices, reaction_spans = _build_combined_matrices(
        results,
        verbose
    )

    if verbose:
        logger.info("✓ Matrix construction complete")
        logger.info("=" * 60)

    # Apply sandwich formula: σ²_R = S^T Σ S
    total_variance = float(sensitivity_vector.T @ covariance_matrix @ sensitivity_vector)
    total_uncertainty = np.sqrt(abs(total_variance))

    # Bootstrap CI on sigma_ND: propagate per-bin MC error on sensitivities through
    # the sandwich formula by sampling S_b ~ S0 * (1 + sigma_s_rel * eps_b).
    if bootstrap:
        ci_low, ci_high, boot_mean, boot_std = _bootstrap_nd_ci(
            sensitivity_vector,
            sigma_s_rel_vector,
            covariance_matrix,
            n_samples=n_bootstrap,
            ci_level=ci_level,
            seed=bootstrap_seed,
        )
        if verbose:
            logger.info(
                f"Bootstrap CI on sigma_ND (N={n_bootstrap}, {int(ci_level*100)}%): "
                f"[{ci_low:.4%}, {ci_high:.4%}], mean = {boot_mean:.4%}"
            )
    else:
        ci_low = ci_high = boot_mean = boot_std = None
    
    # Calculate relative uncertainty
    # Since both sensitivities and covariances are relative, the sandwich formula
    # gives relative variance directly. Therefore, sqrt(variance) IS the relative uncertainty.
    
    # Validate response value is properly provided
    if sdf_data.r0 is None or sdf_data.r0 == 0:
        raise ValueError(
            "Response value (r0) must be provided and non-zero in SDF data. "
            "This is required to properly calculate absolute uncertainties. "
            "Please provide the actual response value when creating the SDF data."
        )
    
    response_value = sdf_data.r0
    response_error = sdf_data.e0 if sdf_data.e0 is not None else 0.0
    relative_uncertainty = total_uncertainty  # Already relative!
    
    # Calculate total number of reactions across all types
    total_reactions = sum(len(result['matching_reactions']) for result in results)
    
    # Calculate individual contributions
    contributions = _calculate_individual_contributions(
        sensitivity_vector,
        covariance_matrix, 
        reaction_indices,
        reaction_spans,
        total_variance,
        verbose
    )
    
    # Calculate correlation effects
    correlation_effects = _calculate_correlation_effects(
        sensitivity_vector,
        covariance_matrix,
        reaction_indices,
        reaction_spans
    )
    
    if verbose:
        logger.info(f"Propagation complete: {relative_uncertainty:.2%} relative uncertainty")
    
    # Calculate representative energy groups from reaction spans
    # (use the first reaction's group count as representative value)
    representative_n_groups = reaction_spans[0][1] if reaction_spans else 0
    
    return UncertaintyResult(
        total_variance=total_variance,
        total_uncertainty=total_uncertainty,
        relative_uncertainty=relative_uncertainty,
        response_value=response_value,
        response_error=response_error,
        contributions=contributions,
        n_reactions=total_reactions,
        n_energy_groups=representative_n_groups,
        correlation_effects=correlation_effects,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        bootstrap_mean=boot_mean,
        bootstrap_std=boot_std,
        bootstrap_n_samples=n_bootstrap if bootstrap else None,
        ci_level=ci_level,
    )


def _match_energy_grids(
    sens_energies: List[float],
    cov_energies: Optional[List[float]],
    tolerance: float,
    verbose: bool
) -> Dict[int, int]:
    """Match energy grids between sensitivity and covariance data.
    
    This function automatically handles unit conversions between MeV and eV.
    If the energy grids don't match directly, it tries converting the sensitivity
    grid from MeV to eV (multiply by 1e6) or vice versa.
    
    Returns mapping from sensitivity group index to covariance group index.
    """
    if cov_energies is None:
        raise ValueError("Covariance matrix must have energy grid information")
    
    sens_array = np.array(sens_energies)
    cov_array = np.array(cov_energies)
    
    # First try direct matching
    energy_mapping = _try_energy_grid_matching(sens_array, cov_array, tolerance)
    
    # If no matches, try unit conversions
    if not energy_mapping:
        if verbose:
            logger.info("No direct energy grid match found, trying unit conversions...")
        
        # Try converting sensitivity from MeV to eV (multiply by 1e6)
        sens_array_eV = sens_array * 1e6
        energy_mapping = _try_energy_grid_matching(sens_array_eV, cov_array, tolerance)
        
        if energy_mapping and verbose:
            logger.info("✓ Energy grids matched after converting sensitivity MeV → eV")
    
    # If still no matches, try converting covariance from eV to MeV (divide by 1e6)
    if not energy_mapping:
        cov_array_MeV = cov_array / 1e6
        energy_mapping = _try_energy_grid_matching(sens_array, cov_array_MeV, tolerance)
        
        if energy_mapping and verbose:
            logger.info("✓ Energy grids matched after converting covariance eV → MeV")
    
    # Report results
    n_sens_groups = len(sens_energies) - 1
    if verbose:
        logger.info(f"Matched {len(energy_mapping)}/{n_sens_groups} energy groups")
        if len(energy_mapping) < n_sens_groups:
            logger.warning(f"Only {len(energy_mapping)} out of {n_sens_groups} energy groups matched")
    
    return energy_mapping


def _try_energy_grid_matching(
    sens_array: np.ndarray,
    cov_array: np.ndarray, 
    tolerance: float
) -> Dict[int, int]:
    """Try to match energy grids with given arrays and tolerance.

    Supports two cases:
    1. Same length grids — exact 1:1 match (existing behavior).
    2. Sensitivity grid is a contiguous subset of the covariance grid — maps
       each sensitivity group to the corresponding covariance group.
    """
    energy_mapping = {}

    # Case 1: Same length — exact match
    if len(sens_array) == len(cov_array):
        all_match = True
        for i in range(len(sens_array)):
            if abs(sens_array[i] - cov_array[i]) > tolerance:
                all_match = False
                break
        if all_match:
            for i in range(len(sens_array) - 1):
                energy_mapping[i] = i
            return energy_mapping

    # Case 2: Sensitivity grid is a contiguous subset of covariance grid
    if len(sens_array) < len(cov_array):
        # Find where the first sensitivity boundary matches in the covariance grid
        start_idx = None
        for c_idx in range(len(cov_array)):
            if abs(sens_array[0] - cov_array[c_idx]) <= tolerance:
                start_idx = c_idx
                break

        if start_idx is not None:
            n_sens = len(sens_array)
            # Check that all consecutive sensitivity boundaries match
            if start_idx + n_sens <= len(cov_array):
                all_match = True
                for i in range(n_sens):
                    if abs(sens_array[i] - cov_array[start_idx + i]) > tolerance:
                        all_match = False
                        break

                if all_match:
                    # Map: sens group i → cov group (start_idx + i)
                    for i in range(n_sens - 1):
                        energy_mapping[i] = start_idx + i

    return energy_mapping


def _find_matching_reactions(
    sens_data: List[SDFReactionData],
    cov_mat: CrossSectionCovariance,
    reaction_filter: Optional[Dict[int, List[int]]],
    verbose: bool,
    description: str = "reactions"
) -> List[Tuple[int, int]]:
    """Find reactions that exist in both sensitivity and covariance data."""
    
    # Get available reactions from covariance matrix using CrossSectionCovariance's built-in method
    cov_reactions = set()
    cov_reactions_by_isotope = cov_mat.reactions_by_isotope()
    for isotope, reactions in cov_reactions_by_isotope.items():
        for reaction in reactions:
            cov_reactions.add((isotope, reaction))
    
    # Get available reactions from sensitivity data
    sens_reactions = {(r.zaid, r.mt) for r in sens_data}
    
    # Find intersection
    matching_reactions = list(sens_reactions.intersection(cov_reactions))
    
    # Apply reaction filter if provided
    if reaction_filter:
        filtered_reactions = []
        
        # Handle special case for filtering by reaction type across all nuclides
        if "ALL_NUCLIDES" in reaction_filter:
            allowed_mts = reaction_filter["ALL_NUCLIDES"]
            for zaid, mt in matching_reactions:
                if mt in allowed_mts:
                    filtered_reactions.append((zaid, mt))
        else:
            # Normal filtering by specific nuclide and reactions
            for zaid, mt in matching_reactions:
                if zaid in reaction_filter:
                    # If empty list, include all reactions for this nuclide
                    if not reaction_filter[zaid] or mt in reaction_filter[zaid]:
                        filtered_reactions.append((zaid, mt))
        
        matching_reactions = filtered_reactions
        
        if verbose:
            logger.info(f"Applied reaction filter for {description}, {len(matching_reactions)} reactions selected")
    
    if verbose:
        logger.info(f"Found {len(matching_reactions)} matching {description}")
        for zaid, mt in matching_reactions[:5]:  # Show first 5
            z = zaid // 1000
            a = zaid % 1000
            nuclide = f"{ATOMIC_NUMBER_TO_SYMBOL.get(z, f'Z{z}')}-{a}"
            reaction = MT_TO_REACTION.get(mt, f"MT{mt}")
            logger.info(f"  {nuclide} {reaction}")
        if len(matching_reactions) > 5:
            logger.info(f"  ... and {len(matching_reactions) - 5} more")
    
    return matching_reactions


def _find_matching_legendre_reactions(
    sens_data: List[SDFReactionData],
    cov_mat: MultigroupLegendreCovariance,
    reaction_filter: Optional[Dict[int, List[int]]],
    verbose: bool
) -> List[Tuple[int, int, int]]:
    """Find Legendre reactions that exist in both sensitivity and covariance data.
    
    Returns list of (zaid, mt_base, legendre_order) tuples where:
    - zaid: isotope identifier
    - mt_base: base MT number (2 for elastic)
    - legendre_order: order of Legendre coefficient (1, 2, 3, ...)
    """
    
    # Get available reactions from Legendre covariance matrix
    # Build reactions_by_isotope directly from the MultigroupLegendreCovariance attributes
    cov_reactions_by_isotope = {}
    for i, iso in enumerate(cov_mat.isotope_rows):
        if iso not in cov_reactions_by_isotope:
            cov_reactions_by_isotope[iso] = set()
        cov_reactions_by_isotope[iso].add(cov_mat.reaction_rows[i])
    
    # Also check column reactions
    for i, iso in enumerate(cov_mat.isotope_cols):
        if iso not in cov_reactions_by_isotope:
            cov_reactions_by_isotope[iso] = set()
        cov_reactions_by_isotope[iso].add(cov_mat.reaction_cols[i])
    
    # Convert to sorted lists
    cov_reactions_by_isotope = {iso: sorted(reactions) for iso, reactions in cov_reactions_by_isotope.items()}
    
    cov_reactions = set()
    for isotope, reactions in cov_reactions_by_isotope.items():
        for reaction in reactions:
            # For each MT and available Legendre orders
            for l_order in cov_mat.legendre_indices:
                cov_reactions.add((isotope, reaction, l_order))
    
    # Convert sensitivity MTs to (zaid, mt_base, legendre_order)
    sens_reactions = set()
    for r in sens_data:
        if r.mt >= 4000:  # Legendre sensitivity
            l_order = r.mt - 4000  # Extract Legendre order
            mt_base = 2  # Only elastic scattering supported for now
            sens_reactions.add((r.zaid, mt_base, l_order))
    
    # Find intersection
    matching_reactions = list(sens_reactions.intersection(cov_reactions))
    
    # Apply reaction filter if provided (convert to MT format for filtering)
    if reaction_filter:
        filtered_reactions = []
        
        for zaid, mt_base, l_order in matching_reactions:
            mt_sensitivity = 4000 + l_order  # Convert back to sensitivity MT
            
            if "ALL_NUCLIDES" in reaction_filter:
                allowed_mts = reaction_filter["ALL_NUCLIDES"]
                if mt_sensitivity in allowed_mts:
                    filtered_reactions.append((zaid, mt_base, l_order))
            else:
                if zaid in reaction_filter:
                    # If empty list, include all reactions for this nuclide
                    if not reaction_filter[zaid] or mt_sensitivity in reaction_filter[zaid]:
                        filtered_reactions.append((zaid, mt_base, l_order))
        
        matching_reactions = filtered_reactions
        
        if verbose:
            logger.info(f"Applied reaction filter for Legendre moments, {len(matching_reactions)} reactions selected")
    
    if verbose:
        logger.info(f"Found {len(matching_reactions)} matching Legendre moment reactions")
        for zaid, mt_base, l_order in matching_reactions[:5]:  # Show first 5
            z = zaid // 1000
            a = zaid % 1000
            nuclide = f"{ATOMIC_NUMBER_TO_SYMBOL.get(z, f'Z{z}')}-{a}"
            logger.info(f"  {nuclide} P{l_order} (MT={4000+l_order})")
        if len(matching_reactions) > 5:
            logger.info(f"  ... and {len(matching_reactions) - 5} more")
    
    return matching_reactions


def _build_matrices(
    sdf_data: SDFData,
    cov_mat: CrossSectionCovariance,
    matching_reactions: List[Tuple[int, int]],
    energy_mapping: Dict[int, int],
    verbose: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, Tuple[int, int]]]:
    """Build sensitivity vector and covariance matrix in consistent order.

    Important: Both sensitivity coefficients and covariance matrices are in relative form.
    This means the sandwich formula σ²_R = S^T Σ S will give relative variance,
    and the uncertainty will be a relative uncertainty that can be directly
    compared to the relative uncertainties in the covariance matrix.

    Also returns a per-bin relative MC error vector (sigma_S_rel) aligned with the
    sensitivity vector, sourced from ``SDFReactionData.error``. This is used by the
    bootstrap CI on the propagated nuclear-data uncertainty.
    """

    n_groups = len(energy_mapping)
    n_reactions = len(matching_reactions)
    total_size = n_groups * n_reactions

    if verbose:
        logger.info(f"Building matrices: {n_reactions} reactions × {n_groups} groups = {total_size} total elements")

    # Create reaction index mapping
    reaction_indices = {i: reaction for i, reaction in enumerate(matching_reactions)}

    # Build sensitivity vector using ONLY sensitivity coefficients (not errors)
    sensitivity_vector = np.zeros(total_size)
    # Per-bin relative MC error on each sensitivity coefficient (from MCNP PERT)
    sigma_s_rel_vector = np.zeros(total_size)

    # Create lookup for sensitivity data
    sens_lookup = {(r.zaid, r.mt): r for r in sdf_data.data}

    for i, (zaid, mt) in enumerate(matching_reactions):
        reaction_data = sens_lookup[(zaid, mt)]

        for sens_group, cov_group in energy_mapping.items():
            if sens_group < len(reaction_data.sensitivity):
                vector_idx = i * n_groups + sens_group
                # Use ONLY the sensitivity coefficient, NOT the error
                sensitivity_vector[vector_idx] = reaction_data.sensitivity[sens_group]
                if reaction_data.error is not None and sens_group < len(reaction_data.error):
                    sigma_s_rel_vector[vector_idx] = reaction_data.error[sens_group]
    
    # Build covariance matrix (in relative form)
    covariance_matrix = np.zeros((total_size, total_size))
    
    # Create lookup for covariance matrices
    for ir, rr, ic, rc, matrix in zip(cov_mat.isotope_rows, cov_mat.reaction_rows,
                                      cov_mat.isotope_cols, cov_mat.reaction_cols, 
                                      cov_mat.matrices):
        
        # Find reaction indices in our ordered list
        try:
            row_reaction_idx = matching_reactions.index((ir, rr))
            col_reaction_idx = matching_reactions.index((ic, rc))
        except ValueError:
            continue  # Skip if reaction not in our filtered list
        
        # Map the covariance matrix block
        for sens_i, cov_i in energy_mapping.items():
            for sens_j, cov_j in energy_mapping.items():
                if cov_i < matrix.shape[0] and cov_j < matrix.shape[1]:
                    matrix_row = row_reaction_idx * n_groups + sens_i
                    matrix_col = col_reaction_idx * n_groups + sens_j
                    value = matrix[cov_i, cov_j]
                    
                    # Set both (i,j) and (j,i) to ensure symmetry
                    covariance_matrix[matrix_row, matrix_col] = value
                    if matrix_row != matrix_col:  # Don't double-set diagonal elements
                        covariance_matrix[matrix_col, matrix_row] = value
    
    if verbose:
        # Check matrix properties
        nonzero_sens = np.count_nonzero(sensitivity_vector)
        nonzero_cov = np.count_nonzero(covariance_matrix)
        logger.info(f"Sensitivity vector: {nonzero_sens}/{total_size} non-zero elements")
        logger.info(f"Covariance matrix: {nonzero_cov}/{total_size**2} non-zero elements")
        
        # Report sensitivity coefficient ranges
        max_sens = np.max(np.abs(sensitivity_vector))
        logger.info(f"Max sensitivity coefficient: {max_sens:.6e}")
        
        # Report covariance matrix ranges
        max_cov = np.max(np.abs(covariance_matrix))
        typical_uncertainty = np.sqrt(np.mean(np.diag(covariance_matrix)[np.diag(covariance_matrix) > 0]))
        logger.info(f"Max covariance element: {max_cov:.6e}")
        logger.info(f"Typical relative uncertainty: {typical_uncertainty:.4f} ({typical_uncertainty*100:.2f}%)")
        
        # Check if covariance matrix is symmetric
        is_symmetric = np.allclose(covariance_matrix, covariance_matrix.T, rtol=1e-10)
        if not is_symmetric:
            logger.warning("Covariance matrix is not symmetric")
        
        # Important note about units
        logger.info("Using relative sensitivities with relative covariances")
        logger.info("Result will be in relative uncertainty units")

    return sensitivity_vector, covariance_matrix, sigma_s_rel_vector, reaction_indices


def _build_combined_matrices(
    results: List[Dict],
    verbose: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, Tuple], Dict[int, Tuple[int, int]]]:
    """Build combined sensitivity vector and covariance matrix for both cross-section and Legendre data.

    Also returns a flat per-bin relative MC error vector (sigma_s_rel) aligned with the
    sensitivity vector. The MC error on each sensitivity coefficient is read from
    ``SDFReactionData.error`` (populated from MCNP PERT) and is used by the bootstrap
    CI on the propagated nuclear-data uncertainty.

    Returns
    -------
    Tuple containing:
    - sensitivity_vector : np.ndarray
    - covariance_matrix : np.ndarray
    - sigma_s_rel : np.ndarray
        Per-bin relative MC error aligned with sensitivity_vector.
    - reaction_indices : Dict[int, Tuple]
    - reaction_spans : Dict[int, Tuple[int, int]]
    """

    # Build matrices for each type
    sub_matrices = []

    for result in results:
        if result['type'] == 'cross_section':
            # Build cross-section matrices using existing function
            # Convert to SDFData-like object for compatibility
            class MockSDFData:
                def __init__(self, data):
                    self.data = data
            mock_sdf = MockSDFData(result['sdf_data'])

            sens_vec, cov_mat, sigma_vec, reaction_idx = _build_matrices(
                mock_sdf,
                result['cov_mat'],
                result['matching_reactions'],
                result['energy_mapping'],
                verbose
            )

            sub_matrices.append({
                'type': 'cross_section',
                'sensitivity_vector': sens_vec,
                'covariance_matrix': cov_mat,
                'sigma_s_rel_vector': sigma_vec,
                'reaction_indices': reaction_idx,
                'n_groups': len(result['energy_mapping']),
                'n_reactions': len(result['matching_reactions'])
            })

        elif result['type'] == 'legendre':
            # Build Legendre matrices
            sens_vec, cov_mat, sigma_vec, reaction_idx = _build_legendre_matrices(
                result['sdf_data'],
                result['cov_mat'],
                result['matching_reactions'],
                result['energy_mapping'],
                verbose
            )

            sub_matrices.append({
                'type': 'legendre',
                'sensitivity_vector': sens_vec,
                'covariance_matrix': cov_mat,
                'sigma_s_rel_vector': sigma_vec,
                'reaction_indices': reaction_idx,
                'n_groups': len(result['energy_mapping']),
                'n_reactions': len(result['matching_reactions'])
            })

    total_size = sum(m['sensitivity_vector'].size for m in sub_matrices)

    if verbose:
        logger.info(f"Combining {len(sub_matrices)} matrix blocks, total size: {total_size}")

    # Create combined matrices
    combined_sensitivity = np.zeros(total_size)
    combined_covariance = np.zeros((total_size, total_size))
    combined_sigma_s_rel = np.zeros(total_size)
    combined_reaction_indices: Dict[int, Tuple] = {}
    combined_reaction_spans: Dict[int, Tuple[int, int]] = {}

    # Fill combined matrices block by block
    current_offset = 0
    global_reaction_idx = 0

    for sub in sub_matrices:
        sub_vec = sub['sensitivity_vector']
        sub_cov = sub['covariance_matrix']
        sub_sigma = sub['sigma_s_rel_vector']
        n_groups = sub['n_groups']
        n_reac = sub['n_reactions']

        # Copy sensitivity vector
        combined_sensitivity[current_offset:current_offset + sub_vec.size] = sub_vec

        # Copy covariance matrix
        combined_covariance[current_offset:current_offset + sub_vec.size,
                          current_offset:current_offset + sub_vec.size] = sub_cov

        # Copy per-bin relative MC error vector
        combined_sigma_s_rel[current_offset:current_offset + sub_vec.size] = sub_sigma

        # Map reactions and their spans
        for local_idx, reaction_id in sub['reaction_indices'].items():
            start = current_offset + local_idx * n_groups
            combined_reaction_indices[global_reaction_idx + local_idx] = reaction_id
            combined_reaction_spans[global_reaction_idx + local_idx] = (start, n_groups)

        current_offset += sub_vec.size
        global_reaction_idx += n_reac

    # Enforce exact symmetry once
    combined_covariance = 0.5 * (combined_covariance + combined_covariance.T)

    if verbose:
        logger.info(f"Combined matrix construction complete")
        logger.info(f"  Total sensitivity elements: {np.count_nonzero(combined_sensitivity)}/{total_size}")
        logger.info(f"  Total covariance elements: {np.count_nonzero(combined_covariance)}/{total_size**2}")
        logger.info(f"  Total reactions: {len(combined_reaction_indices)}")

    return combined_sensitivity, combined_covariance, combined_sigma_s_rel, combined_reaction_indices, combined_reaction_spans


def _build_legendre_matrices(
    sens_data: List[SDFReactionData],
    cov_mat: MultigroupLegendreCovariance,
    matching_reactions: List[Tuple[int, int, int]],
    energy_mapping: Dict[int, int],
    verbose: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, Tuple[int, int, int]]]:
    """Build sensitivity vector and covariance matrix for Legendre moment data.

    Returns
    -------
    Tuple containing:
    - sensitivity_vector : np.ndarray
    - covariance_matrix : np.ndarray
    - sigma_s_rel_vector : np.ndarray
        Per-bin relative MC error on each Legendre sensitivity coefficient.
    - reaction_indices : Dict[int, Tuple[int, int, int]]
    """

    n_groups = len(energy_mapping)
    n_reactions = len(matching_reactions)
    total_size = n_groups * n_reactions

    if verbose:
        logger.info(f"Building Legendre matrices: {n_reactions} reactions × {n_groups} groups = {total_size} total elements")

    # Create reaction index mapping
    reaction_indices = {i: reaction for i, reaction in enumerate(matching_reactions)}

    # Build sensitivity vector
    sensitivity_vector = np.zeros(total_size)
    sigma_s_rel_vector = np.zeros(total_size)

    # Create lookup for sensitivity data (convert MT back from Legendre order)
    sens_lookup = {}
    for r in sens_data:
        if r.mt >= 4000:
            l_order = r.mt - 4000
            mt_base = 2  # Only elastic supported for now
            sens_lookup[(r.zaid, mt_base, l_order)] = r

    for i, (zaid, mt_base, l_order) in enumerate(matching_reactions):
        if (zaid, mt_base, l_order) in sens_lookup:
            reaction_data = sens_lookup[(zaid, mt_base, l_order)]

            for sens_group, cov_group in energy_mapping.items():
                if sens_group < len(reaction_data.sensitivity):
                    vector_idx = i * n_groups + sens_group
                    sensitivity_vector[vector_idx] = reaction_data.sensitivity[sens_group]
                    if reaction_data.error is not None and sens_group < len(reaction_data.error):
                        sigma_s_rel_vector[vector_idx] = reaction_data.error[sens_group]
    
    # Build covariance matrix from MultigroupLegendreCovariance
    covariance_matrix = np.zeros((total_size, total_size))
    
    # Map covariance matrix blocks
    nan_matrices_count = 0
    nan_values_replaced = 0
    for ir, rr, lr, ic, rc, lc, matrix in zip(
        cov_mat.isotope_rows, cov_mat.reaction_rows, cov_mat.l_rows,
        cov_mat.isotope_cols, cov_mat.reaction_cols, cov_mat.l_cols,
        cov_mat.relative_matrices
    ):
        
        # Check for NaN values in this matrix and replace with zeros
        if np.isnan(matrix).any():
            nan_matrices_count += 1
            nan_count_in_matrix = np.isnan(matrix).sum()
            if verbose:
                logger.info(f"Matrix L{lr}×L{lc} has {nan_count_in_matrix} NaN values - replacing with zeros")
            # Replace NaN values with zeros instead of skipping the matrix
            matrix = np.nan_to_num(matrix, nan=0.0)
            nan_values_replaced += nan_count_in_matrix
        
        # Find reaction indices in our ordered list
        try:
            row_reaction_idx = matching_reactions.index((ir, rr, lr))
            col_reaction_idx = matching_reactions.index((ic, rc, lc))
        except ValueError:
            continue  # Skip if reaction not in our filtered list
        
        # Map the covariance matrix block
        for sens_i, cov_i in energy_mapping.items():
            for sens_j, cov_j in energy_mapping.items():
                if cov_i < matrix.shape[0] and cov_j < matrix.shape[1]:
                    matrix_row = row_reaction_idx * n_groups + sens_i
                    matrix_col = col_reaction_idx * n_groups + sens_j
                    value = matrix[cov_i, cov_j]
                    
                    # Set both (i,j) and (j,i) to ensure symmetry
                    covariance_matrix[matrix_row, matrix_col] = value
                    covariance_matrix[matrix_col, matrix_row] = value
    
    if verbose:
        # Check matrix properties
        nonzero_sens = np.count_nonzero(sensitivity_vector)
        nonzero_cov = np.count_nonzero(covariance_matrix)
        logger.info(f"Legendre sensitivity vector: {nonzero_sens}/{total_size} non-zero elements")
        logger.info(f"Legendre covariance matrix: {nonzero_cov}/{total_size**2} non-zero elements")
        if nan_matrices_count > 0:
            logger.info(f"Processed {nan_matrices_count} matrices with NaN values - replaced {nan_values_replaced} NaN values with zeros")
        
        max_sens = np.max(np.abs(sensitivity_vector))
        logger.info(f"Max Legendre sensitivity coefficient: {max_sens:.6e}")

    return sensitivity_vector, covariance_matrix, sigma_s_rel_vector, reaction_indices


def _calculate_individual_contributions(
    sensitivity_vector: np.ndarray,
    covariance_matrix: np.ndarray,
    reaction_indices: Dict[int, Tuple],
    reaction_spans: Dict[int, Tuple[int, int]],
    total_variance: float,
    verbose: bool
) -> List[UncertaintyContribution]:
    """Calculate individual reaction contributions to total uncertainty.
    
    For each reaction i, calculates two types of contributions:
    1. Auto-contribution (without cross-covariances): S_i^T Σ_ii S_i (diagonal only)
    2. Total contribution (with cross-covariances): Σ_j S_i^T Σ_ij S_j (full row)
    
    Both sets sum to meaningful totals and provide different insights.
    
    The reaction_indices can contain either:
    - (zaid, mt) tuples for cross-section reactions
    - (zaid, mt_base, l_order) tuples for Legendre moment reactions
    """
    
    contributions: List[UncertaintyContribution] = []
    
    # One matvec gives all row-sum pieces (efficient O(n²) instead of O(n³))
    sigma_s = covariance_matrix @ sensitivity_vector
    
    # Accumulate diagnostics
    total_auto = 0.0
    total_rowsum = 0.0
    
    for i, reaction_info in reaction_indices.items():
        start_i, n_g_i = reaction_spans[i]
        s_i = sensitivity_vector[start_i: start_i + n_g_i]
        t_i = sigma_s[start_i: start_i + n_g_i]
        
        # Projection (row-sum) piece: c_i = s_i^T t_i
        total_contribution = float(s_i.T @ t_i)
        
        # Auto piece: s_i^T Σ_ii s_i
        block_ii = covariance_matrix[start_i: start_i + n_g_i, start_i: start_i + n_g_i]
        auto_contribution = float(s_i.T @ block_ii @ s_i)
        
        # Build display MT
        if len(reaction_info) == 2:
            zaid, mt = reaction_info
            mt_display = mt
        elif len(reaction_info) == 3:
            zaid, mt_base, l_order = reaction_info
            mt_display = 4000 + l_order
        else:
            raise ValueError(f"Unexpected reaction format: {reaction_info}")
        
        contrib = UncertaintyContribution(
            zaid=zaid,
            mt=mt_display,
            variance_contribution=total_contribution
        )
        
        denom_total = total_variance if total_variance != 0 else 1.0
        contrib.relative_contribution = total_contribution / denom_total
        
        # Attach auto pieces as attributes for reporting
        contrib.auto_variance_contribution = auto_contribution
        # We'll set auto_relative_contribution after computing total_auto
        contrib.auto_relative_contribution = 0.0  # Temporary
        
        contributions.append(contrib)
        
        total_auto += auto_contribution
        total_rowsum += total_contribution
    
    # Sort by |total| for display
    contributions.sort(key=lambda c: abs(c.variance_contribution), reverse=True)
    
    if verbose:
        logger.info(f"Sum row-sum contributions = {total_rowsum:.6e} (should equal total variance {total_variance:.6e})")
        logger.info(f"Sum auto (diagonal)        = {total_auto:.6e}")
    
    # Now that we know total_auto, set auto_relative_contribution safely
    if abs(total_auto) > 0:
        for c in contributions:
            c.auto_relative_contribution = getattr(c, 'auto_variance_contribution', 0.0) / total_auto
    else:
        for c in contributions:
            c.auto_relative_contribution = 0.0
    
    if verbose:
        # Show top contributors with both auto and total contributions
        logger.info("Top uncertainty contributors:")
        for i, contrib in enumerate(contributions[:5]):
            total_pct = contrib.relative_contribution * 100
            auto_pct = contrib.auto_relative_contribution * 100
            logger.info(f"  {contrib.nuclide} {contrib.reaction_name}: {total_pct:.2f}% (total), {auto_pct:.2f}% (auto)")
    
    return contributions


def _calculate_correlation_effects(
    sensitivity_vector: np.ndarray,
    covariance_matrix: np.ndarray,
    reaction_indices: Dict[int, Tuple],
    reaction_spans: Dict[int, Tuple[int, int]]
) -> float:
    """Calculate the contribution from cross-correlations between reactions.
    
    Computes the sum of all off-diagonal terms in the contribution matrix:
    Cross-correlation effect = Σ_{i≠j} S_i^T Σ_ij S_j
    
    This represents the total contribution from correlations between different 
    reactions (different Legendre orders or cross-sections).
    
    Uses the efficient t = Σ S trick for O(n²) complexity.
    """
    
    sigma_s = covariance_matrix @ sensitivity_vector
    corr = 0.0
    
    for i in reaction_indices.keys():
        start_i, n_g_i = reaction_spans[i]
        s_i = sensitivity_vector[start_i: start_i + n_g_i]
        
        # Full row-sum S_i^T (Σ S)_i
        rowsum_i = float(s_i.T @ sigma_s[start_i: start_i + n_g_i])
        
        # Subtract auto block S_i^T Σ_ii S_i to leave only off-diagonals for this row
        block_ii = covariance_matrix[start_i: start_i + n_g_i, start_i: start_i + n_g_i]
        auto_i = float(s_i.T @ block_ii @ s_i)
        
        corr += (rowsum_i - auto_i)
    
    return corr


def filter_reactions_by_nuclide(zaid: int, mt_list: Optional[List[int]] = None) -> Dict[int, List[int]]:
    """
    Convenience function to create reaction filter for a single nuclide.
    
    Parameters
    ----------
    zaid : int
        ZAID of the nuclide
    mt_list : List[int], optional
        List of MT numbers to include. If None, includes all reactions for this nuclide.
        
    Returns
    -------
    Dict[int, List[int]]
        Reaction filter dictionary suitable for sandwich_uncertainty_propagation
        
    Examples
    --------
    >>> # Include all reactions for Fe-56
    >>> filter_dict = filter_reactions_by_nuclide(26056)
    >>> 
    >>> # Include only elastic and inelastic for Fe-56
    >>> filter_dict = filter_reactions_by_nuclide(26056, [2, 4])
    """
    return {zaid: mt_list} if mt_list else {zaid: []}


def filter_reactions_by_type(mt_numbers: List[int]) -> Dict[int, List[int]]:
    """
    Convenience function to create reaction filter by reaction type across all nuclides.
    
    Parameters
    ----------
    mt_numbers : List[int]
        List of MT numbers to include
        
    Returns
    -------
    Dict[int, List[int]]
        Reaction filter dictionary (note: this returns a special marker that the 
        main function should interpret as "these MTs for all nuclides")
        
    Examples
    --------
    >>> # Include only elastic scattering for all nuclides
    >>> filter_dict = filter_reactions_by_type([2])
    >>>
    >>> # Include elastic and (n,γ) for all nuclides  
    >>> filter_dict = filter_reactions_by_type([2, 102])
    """
    # Return special format - the main function handles this case
    return {"ALL_NUCLIDES": mt_numbers}

