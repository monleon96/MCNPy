"""Explicit, format-neutral condensation of multigroup sensitivities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

from .profile import EnergyUnit, SensitivityProfile, SensitivityReaction


class CondensationError(ValueError):
    """Raised when a sensitivity profile cannot be condensed exactly."""


@dataclass(frozen=True)
class CondensationReport:
    """Diagnostics and assumptions for one exact condensation operation."""

    source_groups: int
    target_groups: int
    source_energy_grid: Tuple[float, ...]
    target_energy_grid: Tuple[float, ...]
    source_energy_unit: EnergyUnit
    target_energy_unit: EnergyUnit
    boundary_indices: Tuple[int, ...]
    energy_rtol: float
    uncertainty_method: str
    assumptions: Tuple[str, ...]
    max_integral_drift: float


@dataclass(frozen=True)
class CondensationResult:
    """A condensed profile together with its operator and diagnostics."""

    profile: SensitivityProfile
    operator: np.ndarray
    report: CondensationReport


def _validated_target_grid(
    target_grid: Iterable[float],
    target_energy_unit: EnergyUnit,
) -> np.ndarray:
    grid = np.asarray(target_grid, dtype=float)
    if grid.ndim != 1 or len(grid) < 2:
        raise CondensationError(
            "target_grid must be a one-dimensional array with at least two boundaries"
        )
    if not np.all(np.isfinite(grid)) or np.any(grid < 0.0):
        raise CondensationError(
            "target_grid must contain finite non-negative boundaries"
        )
    if np.any(np.diff(grid) <= 0.0):
        raise CondensationError("target_grid must be strictly increasing")
    if target_energy_unit not in ("eV", "MeV"):
        raise CondensationError("target_energy_unit must be 'eV' or 'MeV'")
    return grid


def _grid_to_mev(grid: np.ndarray, unit: EnergyUnit) -> np.ndarray:
    return grid if unit == "MeV" else grid * 1.0e-6


def _exact_boundary_indices(
    source_grid_mev: np.ndarray,
    target_grid_mev: np.ndarray,
    *,
    energy_rtol: float,
) -> Tuple[int, ...]:
    if energy_rtol < 0.0 or not np.isfinite(energy_rtol):
        raise CondensationError("energy_rtol must be finite and non-negative")

    if not np.isclose(
        target_grid_mev[0], source_grid_mev[0], rtol=energy_rtol, atol=0.0
    ) or not np.isclose(
        target_grid_mev[-1], source_grid_mev[-1], rtol=energy_rtol, atol=0.0
    ):
        raise CondensationError(
            "Exact condensation requires the target grid to cover the same energy "
            "range as the source grid"
        )

    indices = []
    missing = []
    for boundary in target_grid_mev:
        matches = np.flatnonzero(
            np.isclose(source_grid_mev, boundary, rtol=energy_rtol, atol=0.0)
        )
        if len(matches) == 0:
            missing.append(float(boundary))
            continue
        nearest = matches[np.argmin(np.abs(source_grid_mev[matches] - boundary))]
        indices.append(int(nearest))

    if missing:
        preview = ", ".join(f"{value:.8g}" for value in missing[:5])
        if len(missing) > 5:
            preview += f", ... ({len(missing)} total)"
        raise CondensationError(
            "Exact condensation requires every target boundary to exist on the "
            f"source grid after unit conversion. Missing boundaries in MeV: {preview}. "
            "A non-nested grid requires a separate, explicitly requested projection."
        )

    if len(indices) != len(target_grid_mev) or np.any(np.diff(indices) <= 0):
        raise CondensationError(
            "Target boundaries do not define a strictly coarser subset of the source grid"
        )
    return tuple(indices)


def _exact_condensation_operator(
    source_groups: int,
    boundary_indices: Tuple[int, ...],
) -> np.ndarray:
    operator = np.zeros((len(boundary_indices) - 1, source_groups), dtype=float)
    for target_group, (start, stop) in enumerate(
        zip(boundary_indices[:-1], boundary_indices[1:])
    ):
        operator[target_group, start:stop] = 1.0
    return operator


def condense_sensitivity_profile(
    profile: SensitivityProfile,
    target_grid: Iterable[float],
    *,
    target_energy_unit: Optional[EnergyUnit] = None,
    energy_rtol: float = 1.0e-5,
) -> CondensationResult:
    """Condense a sensitivity profile onto an exactly nested coarser grid.

    Group-wise sensitivity coefficients are integral contributions, so source
    groups inside each target group are summed. Absolute one-sigma statistical
    uncertainties are combined in quadrature, which assumes independent source-
    group estimates. Unknown uncertainties remain unknown.

    The operation is deliberately strict: the target grid must span the same
    range and every target boundary must be present in the source grid after
    explicit unit conversion. Non-nested grid projection is not performed.

    Parameters
    ----------
    profile
        Validated, format-neutral source profile.
    target_grid
        Ascending boundaries for the coarser output grid.
    target_energy_unit
        Unit of ``target_grid`` and of the returned profile. Defaults to the
        source profile unit.
    energy_rtol
        Relative tolerance used only to identify physically equal boundaries.
        The default covers the six-digit boundary rounding used by DICE SDFs.
    """
    if not isinstance(profile, SensitivityProfile):
        raise TypeError("profile must be a SensitivityProfile")

    output_unit = (
        profile.energy_unit if target_energy_unit is None else target_energy_unit
    )
    target = _validated_target_grid(target_grid, output_unit)
    source_mev = profile.grid_in("MeV")
    target_mev = _grid_to_mev(target, output_unit)
    boundary_indices = _exact_boundary_indices(
        source_mev,
        target_mev,
        energy_rtol=energy_rtol,
    )
    operator = _exact_condensation_operator(profile.n_groups, boundary_indices)

    reactions = []
    max_integral_drift = 0.0
    has_uncertainty = False
    for reaction in profile.reactions:
        sensitivity = operator @ reaction.sensitivity
        uncertainty = None
        if reaction.uncertainty is not None:
            has_uncertainty = True
            uncertainty = np.sqrt((operator * operator) @ (reaction.uncertainty**2))

        source_integral = float(np.sum(reaction.sensitivity))
        target_integral = float(np.sum(sensitivity))
        max_integral_drift = max(
            max_integral_drift,
            abs(target_integral - source_integral),
        )
        reactions.append(
            SensitivityReaction(
                zaid=reaction.zaid,
                mt=reaction.mt,
                sensitivity=sensitivity,
                uncertainty=uncertainty,
                label=reaction.label,
            )
        )

    uncertainty_method = (
        "independent source-group quadrature" if has_uncertainty else "not available"
    )
    assumptions = (
        (
            "Source-group sensitivity estimates were treated as statistically "
            "independent when propagating absolute one-sigma uncertainties."
        ),
    ) if has_uncertainty else ()

    metadata = dict(profile.metadata)
    history = list(metadata.get("condensation_history", ()))
    history.append(
        {
            "method": "exact_nested_sum",
            "source_groups": profile.n_groups,
            "target_groups": len(target) - 1,
            "source_energy_unit": profile.energy_unit,
            "target_energy_unit": output_unit,
            "energy_rtol": energy_rtol,
        }
    )
    metadata["condensation_history"] = history

    condensed = SensitivityProfile(
        energy_grid=target,
        energy_unit=output_unit,
        reactions=tuple(reactions),
        response=profile.response,
        response_uncertainty=profile.response_uncertainty,
        label=profile.label,
        metadata=metadata,
    )
    report = CondensationReport(
        source_groups=profile.n_groups,
        target_groups=condensed.n_groups,
        source_energy_grid=tuple(float(value) for value in profile.energy_grid),
        target_energy_grid=tuple(float(value) for value in target),
        source_energy_unit=profile.energy_unit,
        target_energy_unit=output_unit,
        boundary_indices=boundary_indices,
        energy_rtol=energy_rtol,
        uncertainty_method=uncertainty_method,
        assumptions=assumptions,
        max_integral_drift=max_integral_drift,
    )
    return CondensationResult(profile=condensed, operator=operator, report=report)


__all__ = [
    "CondensationError",
    "CondensationReport",
    "CondensationResult",
    "condense_sensitivity_profile",
]
