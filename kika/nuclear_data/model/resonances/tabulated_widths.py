"""GNDS-2.1 §19.4.1 ``tabulatedWidths``: the unresolved region.

Average widths and level spacings as functions of energy, with a degrees-of-
freedom count per channel. ENDF's LSSF flag — whether the URR cross sections are
already in MF3 or must be computed from these parameters — is kept, because
getting it wrong double-counts the unresolved region.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

__all__ = ["UnresolvedChannel", "UnresolvedSpinGroup", "TabulatedWidths"]


@dataclass
class UnresolvedChannel:
    """Average width for one channel, constant or tabulated against energy."""

    label: str
    degreesOfFreedom: float = 1.0
    widths: Optional[np.ndarray] = None
    constantWidth: Optional[float] = None

    def __post_init__(self) -> None:
        if self.widths is not None:
            self.widths = np.asarray(self.widths, dtype=float)


@dataclass
class UnresolvedSpinGroup:
    """Averages for one (L, J)."""

    L: int
    J: float
    levelSpacing: Optional[np.ndarray] = None
    channels: List[UnresolvedChannel] = field(default_factory=list)
    atomicWeightRatio: Optional[float] = None

    def __post_init__(self) -> None:
        if self.levelSpacing is not None:
            self.levelSpacing = np.asarray(self.levelSpacing, dtype=float)


@dataclass
class TabulatedWidths:
    """§19.4.1. The unresolved resonance region."""

    spinGroups: List[UnresolvedSpinGroup] = field(default_factory=list)
    energyGrid: Optional[np.ndarray] = None
    scatteringRadius: Optional[float] = None
    #: ENDF LSSF: 0 = compute the URR cross sections from these parameters,
    #: 1 = they are already in MF3 and these are for self-shielding only.
    selfShieldingOnly: bool = False
    label: Optional[str] = None

    def __post_init__(self) -> None:
        if self.energyGrid is not None:
            self.energyGrid = np.asarray(self.energyGrid, dtype=float)
