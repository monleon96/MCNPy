"""Covariance-weighted similarity of integral-response sensitivity profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from kika.UQ.alignment import (
    AlignmentReport,
    LegendreCovarianceInput,
    ParameterIndex,
    ParameterKey,
    PreparedCovariance,
    CovarianceInput,
    align_sensitivity_covariance,
)
from kika.sensitivities.profile import SensitivityProfile
from kika.sensitivities.sdf import SDFData


class ZeroSimilarityVarianceError(ValueError):
    """Raised when c-k is undefined because either covariance norm is non-positive."""


@dataclass(frozen=True)
class ReactionSimilarity:
    """Non-additive c-k diagnostic from one diagonal reaction block."""

    key: ParameterKey
    value: Optional[float]
    variance_a: float
    variance_b: float
    cross_covariance: float


@dataclass
class SimilarityResult:
    """Covariance-weighted similarity result and alignment provenance."""

    value: float
    variance_a: float
    variance_b: float
    cross_covariance: float
    reaction_similarity: Tuple[ReactionSimilarity, ...]
    index: Tuple[ParameterIndex, ...]
    parameter_keys: Tuple[ParameterKey, ...]
    alignment_report: AlignmentReport

    @property
    def ck(self) -> float:
        """Covariance-weighted similarity coefficient."""
        return self.value


def _as_profile(value: Union[SensitivityProfile, SDFData]) -> SensitivityProfile:
    if isinstance(value, SensitivityProfile):
        return value
    if isinstance(value, SDFData):
        return value.to_sensitivity_profile()
    raise TypeError("similarity profiles must be SensitivityProfile or SDFData")


def _nubar_mode_for(mode, zaid: int) -> str:
    return mode.get(zaid, "total") if isinstance(mode, dict) else mode


def _prepare_profile(
    value: Union[SensitivityProfile, SDFData],
    include: Optional[Iterable[Tuple[int, int]]],
    exclude: Optional[Iterable[Tuple[int, int]]],
    include_mt1: bool,
    nubar_mode,
) -> SensitivityProfile:
    profile = _as_profile(value)
    include_set = set(include) if include is not None else None
    exclude_set = set(exclude or ())
    selected = [
        reaction for reaction in profile.reactions
        if (include_mt1 or reaction.mt != 1)
        and (include_set is None or (reaction.zaid, reaction.mt) in include_set)
        and (reaction.zaid, reaction.mt) not in exclude_set
    ]

    by_zaid = {}
    for reaction in selected:
        if reaction.mt in (452, 455, 456):
            by_zaid.setdefault(reaction.zaid, set()).add(reaction.mt)
    kept = []
    for reaction in selected:
        mts = by_zaid.get(reaction.zaid, set())
        if 452 not in mts or not ({455, 456} & mts):
            kept.append(reaction)
            continue
        mode = _nubar_mode_for(nubar_mode, reaction.zaid)
        complete = {455, 456}.issubset(mts)
        if mode == "components" and complete:
            if reaction.mt != 452:
                kept.append(reaction)
        elif reaction.mt not in (455, 456):
            kept.append(reaction)

    if not kept:
        raise ValueError("No sensitivity reactions remain after applying similarity filters")
    return SensitivityProfile(
        energy_grid=profile.energy_grid,
        energy_unit=profile.energy_unit,
        reactions=tuple(kept),
        response=profile.response,
        response_uncertainty=profile.response_uncertainty,
        label=profile.label,
        metadata=profile.metadata,
    )


def _checked_ck(cross: float, variance_a: float, variance_b: float) -> float:
    if not np.isfinite(variance_a) or not np.isfinite(variance_b):
        raise ZeroSimilarityVarianceError("c_k is undefined because a covariance norm is non-finite")
    if variance_a <= 0.0 or variance_b <= 0.0:
        raise ZeroSimilarityVarianceError(
            "c_k is undefined because S.T @ C @ S is zero or negative for at least one profile"
        )
    value = float(cross / np.sqrt(variance_a * variance_b))
    if not np.isfinite(value):
        raise ZeroSimilarityVarianceError("c_k is undefined because its normalized value is non-finite")
    if abs(value) > 1.0 + 1.0e-10:
        raise ValueError(
            f"c_k={value:.12g} lies outside [-1, 1]; verify covariance symmetry and definiteness"
        )
    return float(np.clip(value, -1.0, 1.0))


def similarity_ck(
    profile_a: Union[SensitivityProfile, SDFData],
    profile_b: Union[SensitivityProfile, SDFData],
    covariance: Optional[CovarianceInput] = None,
    legendre_covariance: Optional[LegendreCovarianceInput] = None,
    *,
    alias_policy: str = "exact",
    missing: str = "error",
    include: Optional[Sequence[Tuple[int, int]]] = None,
    exclude: Optional[Sequence[Tuple[int, int]]] = None,
    include_mt1: bool = False,
    nubar_mode: Union[str, dict] = "total",
    energy_tolerance: float = 1.0e-8,
    prepared_covariance: Optional[PreparedCovariance] = None,
) -> SimilarityResult:
    """Calculate covariance-weighted c-k between two sensitivity profiles."""
    raw_a = _as_profile(profile_a)
    raw_b = _as_profile(profile_b)
    a = _prepare_profile(raw_a, include, exclude, include_mt1, nubar_mode)
    b = _prepare_profile(raw_b, include, exclude, include_mt1, nubar_mode)
    aligned = align_sensitivity_covariance(
        [a, b],
        covariance=covariance,
        legendre_covariance=legendre_covariance,
        alias_policy=alias_policy,
        missing=missing,
        energy_rtol=energy_tolerance,
        prepared_covariance=prepared_covariance,
    )
    for profile_index, (raw, selected) in enumerate(((raw_a, a), (raw_b, b))):
        raw_keys = {ParameterKey.from_sensitivity(r.zaid, r.mt) for r in raw.reactions}
        selected_keys = {
            ParameterKey.from_sensitivity(r.zaid, r.mt) for r in selected.reactions
        }
        exclusions = sorted(raw_keys - selected_keys)
        if exclusions:
            aligned.report.policy_exclusions[profile_index] = exclusions
    if not include_mt1 and any(
        reaction.mt == 1 for profile in (raw_a, raw_b) for reaction in profile.reactions
    ):
        aligned.report.assumptions.append("MT=1 excluded from c_k")
    aligned.report.assumptions.append(
        f"nu-bar redundancy resolved with nubar_mode={nubar_mode!r}"
    )

    s_a, s_b = aligned.sensitivity_vectors
    matrix = aligned.covariance
    variance_a = float(s_a @ matrix @ s_a)
    variance_b = float(s_b @ matrix @ s_b)
    cross = float(s_a @ matrix @ s_b)
    value = _checked_ck(cross, variance_a, variance_b)

    diagnostics: List[ReactionSimilarity] = []
    for index, key in enumerate(aligned.parameter_keys):
        start, length = aligned.reaction_spans[index]
        stop = start + length
        block = matrix[start:stop, start:stop]
        sub_a = s_a[start:stop]
        sub_b = s_b[start:stop]
        var_a = float(sub_a @ block @ sub_a)
        var_b = float(sub_b @ block @ sub_b)
        sub_cross = float(sub_a @ block @ sub_b)
        local = None
        if var_a > 0.0 and var_b > 0.0:
            local = _checked_ck(sub_cross, var_a, var_b)
        diagnostics.append(ReactionSimilarity(key, local, var_a, var_b, sub_cross))

    return SimilarityResult(
        value=value,
        variance_a=variance_a,
        variance_b=variance_b,
        cross_covariance=cross,
        reaction_similarity=tuple(diagnostics),
        index=aligned.index,
        parameter_keys=aligned.parameter_keys,
        alignment_report=aligned.report,
    )
