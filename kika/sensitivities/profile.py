"""Format-neutral sensitivity profiles used by UQ calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Literal, Mapping, Optional, Tuple

import numpy as np


EnergyUnit = Literal["eV", "MeV"]


def _validated_vector(values: Iterable[float], name: str, size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) != size:
        raise ValueError(f"{name} must be a one-dimensional vector of length {size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


@dataclass
class SensitivityReaction:
    """One format-neutral multigroup sensitivity profile."""

    zaid: int
    mt: int
    sensitivity: np.ndarray
    uncertainty: Optional[np.ndarray] = None
    label: Optional[str] = None

    def validated(self, n_groups: int) -> "SensitivityReaction":
        sensitivity = _validated_vector(self.sensitivity, "sensitivity", n_groups)
        uncertainty = None
        if self.uncertainty is not None:
            uncertainty = _validated_vector(self.uncertainty, "uncertainty", n_groups)
            if np.any(uncertainty < 0.0):
                raise ValueError("uncertainty must contain non-negative absolute standard deviations")
        return SensitivityReaction(
            zaid=int(self.zaid),
            mt=int(self.mt),
            sensitivity=sensitivity,
            uncertainty=uncertainty,
            label=self.label,
        )

    @property
    def key(self) -> Tuple[int, int]:
        return self.zaid, self.mt


@dataclass
class SensitivityProfile:
    """Validated sensitivity data independent of SDF or database storage.

    Sensitivities are dimensionless relative coefficients. ``uncertainty`` is
    the absolute one-sigma uncertainty on each sensitivity coefficient.
    """

    energy_grid: np.ndarray
    reactions: Tuple[SensitivityReaction, ...]
    energy_unit: EnergyUnit = "MeV"
    response: Optional[float] = None
    response_uncertainty: Optional[float] = None
    label: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        grid = np.asarray(self.energy_grid, dtype=float)
        if grid.ndim != 1 or len(grid) < 2:
            raise ValueError("energy_grid must be a one-dimensional array with at least two boundaries")
        if not np.all(np.isfinite(grid)) or np.any(grid < 0.0):
            raise ValueError("energy_grid must contain finite non-negative boundaries")
        if np.any(np.diff(grid) <= 0.0):
            raise ValueError("energy_grid must be strictly increasing")
        if self.energy_unit not in ("eV", "MeV"):
            raise ValueError("energy_unit must be 'eV' or 'MeV'")

        n_groups = len(grid) - 1
        validated = tuple(reaction.validated(n_groups) for reaction in self.reactions)
        keys = [reaction.key for reaction in validated]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(f"duplicate sensitivity reaction keys: {duplicates}")

        if self.response is not None and not np.isfinite(self.response):
            raise ValueError("response must be finite when provided")
        if self.response_uncertainty is not None:
            if not np.isfinite(self.response_uncertainty) or self.response_uncertainty < 0.0:
                raise ValueError("response_uncertainty must be a finite non-negative absolute sigma")

        self.energy_grid = grid.copy()
        self.reactions = validated
        self.metadata = dict(self.metadata)

    @property
    def n_groups(self) -> int:
        return len(self.energy_grid) - 1

    @property
    def reaction_map(self) -> Dict[Tuple[int, int], SensitivityReaction]:
        return {reaction.key: reaction for reaction in self.reactions}

    def grid_in(self, unit: EnergyUnit) -> np.ndarray:
        if unit not in ("eV", "MeV"):
            raise ValueError("unit must be 'eV' or 'MeV'")
        if unit == self.energy_unit:
            return self.energy_grid.copy()
        factor = 1.0e6 if self.energy_unit == "MeV" else 1.0e-6
        return self.energy_grid * factor
