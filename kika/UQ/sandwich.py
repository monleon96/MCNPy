"""Sandwich formula for uncertainty propagation in nuclear data.

This module implements the sandwich formula σ²_R = S^T Σ S for propagating nuclear data 
uncertainties from sensitivity coefficients and covariance matrices.

The sandwich formula allows propagation of uncertainties from nuclear cross-section 
covariances to integral responses through sensitivity coefficients.
"""

from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import numpy as np
import warnings
import logging

from kika.sensitivities.sdf import SDFData
from kika.sensitivities.profile import SensitivityProfile
from kika.UQ.alignment import AlignmentReport, ParameterKey, align_sensitivity_covariance
from kika.cov.cross_section_covariance import CrossSectionCovariance
from kika.cov.multigroup.mg_legendre_covariance import MultigroupLegendreCovariance
from kika._constants import (
    MT_TO_REACTION,
    ATOMIC_NUMBER_TO_SYMBOL,
    NUBAR_TOTAL_MT,
    NUBAR_COMPONENT_MTS,
)

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
        Absolute one-sigma uncertainty on the unperturbed response. Reported as a
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
    alignment_report: Optional[AlignmentReport] = None

    @property
    def response_relative_error(self) -> float:
        if self.response_value == 0:
            return 0.0 if self.response_error == 0 else float("inf")
        return self.response_error / abs(self.response_value)

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
        lines.append(f"  Response statistical sigma:      {self.response_error:.6e} (absolute)")
        lines.append(
            f"  Response statistical rel_err:    {self.response_relative_error * 100:.3f}%"
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
    sigma_s_abs_vector: np.ndarray,
    covariance_matrix: np.ndarray,
    n_samples: int,
    ci_level: float,
    seed: Optional[int],
) -> Tuple[float, float, float, float]:
    """Bootstrap a confidence interval on sigma_ND = sqrt(S^T C S).

    Draws S_b = S0 + sigma_S_abs * eps_b, eps_b ~ N(0, I), then computes
    sigma_ND,b = sqrt(|S_b^T C S_b|) for each draw. Returns the (low, high) percentile
    bounds, the bootstrap mean of sigma_ND, and its standard deviation.

    The per-bin sigma_S_abs is the absolute standard deviation stored in
    SDFReactionData.error. Zero standard deviations reduce to the nominal S0.
    """
    rng = np.random.default_rng(seed)
    # eps_b ~ N(0, I): shape (n_samples, len(S))
    eps = rng.standard_normal((n_samples, sensitivity_vector.size))
    # S_b = S0 + sigma_s_abs * eps -- broadcasts row-wise
    s_samples = sensitivity_vector[None, :] + sigma_s_abs_vector[None, :] * eps
    # sigma_ND,b = sqrt(|S_b^T C S_b|), vectorised over b
    variances = np.einsum('bi,ij,bj->b', s_samples, covariance_matrix, s_samples)
    sigmas = np.sqrt(np.abs(variances))

    alpha = (1.0 - ci_level) / 2.0
    low, high = np.percentile(sigmas, [100 * alpha, 100 * (1.0 - alpha)])
    return float(low), float(high), float(np.mean(sigmas)), float(np.std(sigmas, ddof=1))


def _profile_for_sandwich(
    data: Union[SDFData, SensitivityProfile],
    reaction_filter: Optional[Dict[int, List[int]]],
    nubar_mode: Union[str, Dict[int, str]],
    verbose: bool,
) -> SensitivityProfile:
    profile = data.to_sensitivity_profile() if isinstance(data, SDFData) else data
    if not isinstance(profile, SensitivityProfile):
        raise TypeError("sdf_data must be SDFData or SensitivityProfile")

    selected = []
    mt1_excluded = False
    for reaction in profile.reactions:
        if reaction.mt == 1:
            mt1_excluded = True
            continue
        if reaction_filter:
            if "ALL_NUCLIDES" in reaction_filter:
                if reaction.mt not in reaction_filter["ALL_NUCLIDES"]:
                    continue
            elif reaction.zaid not in reaction_filter:
                continue
            elif reaction_filter[reaction.zaid] and reaction.mt not in reaction_filter[reaction.zaid]:
                continue
        selected.append(reaction)

    xs_pairs = [(reaction.zaid, reaction.mt) for reaction in selected if reaction.mt < 4000]
    kept_xs = set(_resolve_nubar_redundancy(xs_pairs, nubar_mode, verbose))
    selected = [
        reaction for reaction in selected
        if reaction.mt >= 4000 or (reaction.zaid, reaction.mt) in kept_xs
    ]
    if not selected:
        raise ValueError("No sensitivity reactions remain after applying filters")
    metadata = dict(profile.metadata)
    metadata["sandwich_mt1_excluded"] = mt1_excluded
    return SensitivityProfile(
        energy_grid=profile.energy_grid,
        energy_unit=profile.energy_unit,
        reactions=tuple(selected),
        response=profile.response,
        response_uncertainty=profile.response_uncertainty,
        label=profile.label,
        metadata=metadata,
    )


def sandwich_uncertainty_propagation(
    sdf_data: Union[SDFData, SensitivityProfile],
    cov_mat: Optional[Union[CrossSectionCovariance, List[CrossSectionCovariance]]] = None,
    legendre_cov_mat: Optional[Union[MultigroupLegendreCovariance, List[MultigroupLegendreCovariance]]] = None,
    reaction_filter: Optional[Dict[int, List[int]]] = None,
    nubar_mode: Union[str, Dict[int, str]] = "total",
    energy_tolerance: float = 1e-6,
    verbose: bool = False,
    bootstrap: bool = True,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    bootstrap_seed: Optional[int] = None,
    alias_policy: str = "exact",
    missing: str = "error",
) -> UncertaintyResult:
    """Propagate nuclear-data uncertainty with ``S.T @ C @ S``.

    ``sdf_data`` may be an :class:`SDFData` or a format-neutral
    :class:`SensitivityProfile`. Alignment is strict by default. Use
    ``missing="drop"`` only when excluding uncovered parameters is intended;
    the returned ``alignment_report`` records every exclusion and alias.
    """
    if cov_mat is None and legendre_cov_mat is None:
        raise ValueError("At least one covariance matrix (cov_mat or legendre_cov_mat) must be provided")
    if not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level must be between zero and one")
    if n_bootstrap < 2 and bootstrap:
        raise ValueError("n_bootstrap must be at least 2")

    source_profile = (
        sdf_data.to_sensitivity_profile() if isinstance(sdf_data, SDFData) else sdf_data
    )
    profile = _profile_for_sandwich(source_profile, reaction_filter, nubar_mode, verbose)
    aligned = align_sensitivity_covariance(
        [profile],
        covariance=cov_mat,
        legendre_covariance=legendre_cov_mat,
        alias_policy=alias_policy,
        missing=missing,
        energy_rtol=energy_tolerance,
    )
    source_keys = {
        ParameterKey.from_sensitivity(r.zaid, r.mt) for r in source_profile.reactions
    }
    selected_keys = {
        ParameterKey.from_sensitivity(r.zaid, r.mt) for r in profile.reactions
    }
    exclusions = sorted(source_keys - selected_keys)
    if exclusions:
        aligned.report.policy_exclusions[0] = exclusions
    if profile.metadata.get("sandwich_mt1_excluded"):
        aligned.report.assumptions.append("MT=1 excluded to avoid total/reaction double counting")
    aligned.report.assumptions.append(f"nu-bar redundancy resolved with nubar_mode={nubar_mode!r}")

    sensitivity_vector = aligned.sensitivity_vectors[0]
    sigma_s_abs_vector = aligned.sensitivity_uncertainties[0]
    covariance_matrix = aligned.covariance
    reaction_spans = aligned.reaction_spans
    reaction_indices = {}
    for index, key in enumerate(aligned.parameter_keys):
        if key.kind == "legendre":
            reaction_indices[index] = (key.zaid, key.mt, key.order)
        else:
            reaction_indices[index] = (key.zaid, key.mt)

    total_variance = float(sensitivity_vector.T @ covariance_matrix @ sensitivity_vector)
    total_uncertainty = float(np.sqrt(abs(total_variance)))

    if bootstrap:
        if not np.all(np.isfinite(sigma_s_abs_vector)):
            raise ValueError(
                "Bootstrap requires absolute sensitivity uncertainties for every aligned "
                "parameter. This profile contains unknown uncertainties; provide them or "
                "repeat with bootstrap=False."
            )
        ci_low, ci_high, boot_mean, boot_std = _bootstrap_nd_ci(
            sensitivity_vector, sigma_s_abs_vector, covariance_matrix,
            n_samples=n_bootstrap, ci_level=ci_level, seed=bootstrap_seed,
        )
    else:
        ci_low = ci_high = boot_mean = boot_std = None

    if profile.response is None or profile.response == 0:
        raise ValueError(
            "A finite non-zero response value is required to report absolute response uncertainty"
        )
    response_error = profile.response_uncertainty if profile.response_uncertainty is not None else 0.0
    contributions = _calculate_individual_contributions(
        sensitivity_vector, covariance_matrix, reaction_indices, reaction_spans,
        total_variance, verbose,
    )
    correlation_effects = _calculate_correlation_effects(
        sensitivity_vector, covariance_matrix, reaction_indices, reaction_spans
    )
    n_groups = reaction_spans[0][1] if reaction_spans else 0

    return UncertaintyResult(
        total_variance=total_variance,
        total_uncertainty=total_uncertainty,
        relative_uncertainty=total_uncertainty,
        response_value=profile.response,
        response_error=response_error,
        contributions=contributions,
        n_reactions=len(aligned.parameter_keys),
        n_energy_groups=n_groups,
        correlation_effects=correlation_effects,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        bootstrap_mean=boot_mean,
        bootstrap_std=boot_std,
        bootstrap_n_samples=n_bootstrap if bootstrap else None,
        ci_level=ci_level,
        alignment_report=aligned.report,
    )

def _resolve_nubar_redundancy(
    matching_reactions: List[Tuple[int, int]],
    nubar_mode: Union[str, Dict[int, str]],
    verbose: bool,
) -> List[Tuple[int, int]]:
    """Resolve redundant nu-bar reactions according to the requested view.

    Nu-bar is described by three quantities: total (MT 452), prompt (MT 456) and
    delayed (MT 455), with total = prompt + delayed (and, in relative terms,
    S_total = S_prompt + S_delayed). When a covariance file provides the total
    *and* its components, the sandwich formula would count the same nu-bar
    uncertainty more than once, so for each isotope only one of the two views is
    kept:

    - ``"total"`` (default): keep the total (MT 452), drop prompt/delayed. The
      total is the standard integral quantity for k-eff uncertainty.
    - ``"components"``: keep prompt + delayed, drop the total — but only when the
      decomposition is complete (both MT 455 and MT 456 present). With only one
      component available, dropping the total would silently undercount the
      missing piece, so the total is kept and a warning is emitted instead.

    ``nubar_mode`` may be a single string applied to every isotope, or a dict
    mapping ZAID -> mode (isotopes absent from the dict default to ``"total"``).
    Isotopes that have only the total, or only components, are returned unchanged.
    """
    def mode_for(zaid: int) -> str:
        if isinstance(nubar_mode, dict):
            return nubar_mode.get(zaid, "total")
        return nubar_mode or "total"

    isotopes_with_total = {z for z, mt in matching_reactions if mt == NUBAR_TOTAL_MT}
    if not isotopes_with_total:
        return matching_reactions

    components_by_isotope: Dict[int, set] = {}
    for zaid, mt in matching_reactions:
        if mt in NUBAR_COMPONENT_MTS and zaid in isotopes_with_total:
            components_by_isotope.setdefault(zaid, set()).add(mt)

    kept: List[Tuple[int, int]] = []
    dropped_components: List[Tuple[int, int]] = []
    dropped_total: List[Tuple[int, int]] = []
    incomplete: List[int] = []
    for zaid, mt in matching_reactions:
        present_components = components_by_isotope.get(zaid, set())
        # Only the (total + >=1 component) isotopes carry a real choice.
        if zaid not in components_by_isotope:
            kept.append((zaid, mt))
            continue

        want_components = mode_for(zaid) == "components"
        complete = set(NUBAR_COMPONENT_MTS).issubset(present_components)

        if want_components and complete:
            # Keep the decomposition, drop the total.
            if mt == NUBAR_TOTAL_MT:
                dropped_total.append((zaid, mt))
            else:
                kept.append((zaid, mt))
        else:
            # Keep the total, drop the components.
            if want_components and not complete and mt == NUBAR_TOTAL_MT:
                incomplete.append(zaid)
            if mt in NUBAR_COMPONENT_MTS:
                dropped_components.append((zaid, mt))
            else:
                kept.append((zaid, mt))

    def _labels(pairs):
        return ", ".join(
            f"{_format_nuclide(z)} {MT_TO_REACTION.get(mt, f'MT{mt}')}" for z, mt in pairs
        )

    if dropped_components:
        labels = _labels(dropped_components)
        warnings.warn(
            "Excluded prompt/delayed nu-bar to avoid double-counting with total "
            f"nu-bar (MT 452): {labels}."
        )
        if verbose:
            logger.info(f"  Kept total nu-bar (MT 452), dropped components: {labels}")

    if dropped_total:
        labels = _labels(dropped_total)
        warnings.warn(
            "Excluded total nu-bar (MT 452) to avoid double-counting with its "
            f"prompt/delayed components: {labels}."
        )
        if verbose:
            logger.info(f"  Kept prompt/delayed nu-bar, dropped total: {labels}")

    if incomplete:
        labels = ", ".join(_format_nuclide(z) for z in incomplete)
        warnings.warn(
            "Requested prompt/delayed nu-bar decomposition but only one component "
            f"is available for {labels}; kept total nu-bar (MT 452) instead to "
            "avoid undercounting."
        )

    return kept


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

