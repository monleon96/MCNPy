"""Weight window values container and operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class WeightWindowValues:
    """Container for weight window value arrays and in-place operations.

    Parameters
    ----------
    ww_values : dict[int, np.ndarray]
        Mapping from particle index (0-based) to a 5-D array of shape
        ``(nt, ne, nfz, nfy, nfx)``.
    """

    ww_values: dict[int, np.ndarray] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_particles(
        self, particles: Optional[list[int] | int] = None
    ) -> list[int]:
        """Return a list of particle indices to operate on."""
        if particles is None:
            return list(self.ww_values.keys())
        if isinstance(particles, int):
            return [particles]
        return list(particles)

    def _validate_particles(self, particles: list[int]) -> None:
        for p in particles:
            if p not in self.ww_values:
                raise ValueError(
                    f"Particle index {p} not found. "
                    f"Available: {sorted(self.ww_values.keys())}"
                )

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def multiply(
        self,
        factor: float = 1.2,
        particles: Optional[list[int] | int] = None,
    ) -> None:
        """Scale weight window values by *factor* (in-place)."""
        targets = self._resolve_particles(particles)
        self._validate_particles(targets)
        for p in targets:
            self.ww_values[p] *= factor

    def soften(
        self,
        power: float = 0.6,
        particles: Optional[list[int] | int] = None,
    ) -> None:
        """Raise values to *power* (in-place).  power < 1 softens."""
        targets = self._resolve_particles(particles)
        self._validate_particles(targets)
        for p in targets:
            self.ww_values[p] = np.power(self.ww_values[p], power)

    def apply_ratio_threshold(
        self,
        threshold: float = 10.0,
        particles: Optional[list[int] | int] = None,
    ) -> int:
        """Zero out cells whose max-neighbor ratio exceeds *threshold*.

        Operates on the spatial dimensions (last 3) of each time/energy
        slice.  Returns the total number of zeroed cells.
        """
        from kika.wwinp.utils import calculate_max_ratio_array

        targets = self._resolve_particles(particles)
        self._validate_particles(targets)

        total_changes = 0
        for p in targets:
            arr = self.ww_values[p]  # (nt, ne, nfz, nfy, nfx)
            for t in range(arr.shape[0]):
                for e in range(arr.shape[1]):
                    spatial = arr[t, e]  # (nfz, nfy, nfx) — view, not copy
                    ratios = calculate_max_ratio_array(spatial)
                    mask = ratios > threshold
                    total_changes += int(np.count_nonzero(mask))
                    spatial[mask] = 0.0
        return total_changes
