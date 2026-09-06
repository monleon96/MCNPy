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
    #: The energies ``widths`` is tabulated against, when they are **not** the
    #: block's :attr:`TabulatedWidths.energyGrid`. ENDF's URR puts every average
    #: on one grid per range, which is why this field did not exist; GNDS gives
    #: each width its own ``XYs1d``, and **66 of the library's 351 unresolved
    #: blocks use more than one grid** — up to seven of them. Without this,
    #: those 66 come out with one grid's energies attached to another grid's
    #: values, which is a *wrong* average width rather than a missing one.
    #:
    #: ``None`` means "the block's grid", which is the common case and what an
    #: ENDF-decoded evaluation always says. Same two-field shape as
    #: :class:`~kika.nuclear_data.model.resonances.ScatteringRadius`.
    energies: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.widths is not None:
            self.widths = np.asarray(self.widths, dtype=float)
        if self.energies is not None:
            self.energies = np.asarray(self.energies, dtype=float)
            size = 0 if self.widths is None else self.widths.size
            if self.energies.size != size:
                raise ValueError(
                    f"channel {self.label!r} has {self.energies.size} energies "
                    f"and {size} widths"
                )


@dataclass
class UnresolvedSpinGroup:
    """Averages for one (L, J)."""

    L: int
    J: float
    levelSpacing: Optional[np.ndarray] = None
    channels: List[UnresolvedChannel] = field(default_factory=list)
    atomicWeightRatio: Optional[float] = None
    #: The energies ``levelSpacing`` is tabulated against, when they are not the
    #: block's grid. See :attr:`UnresolvedChannel.energies`.
    levelSpacingEnergies: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.levelSpacing is not None:
            self.levelSpacing = np.asarray(self.levelSpacing, dtype=float)
        if self.levelSpacingEnergies is not None:
            self.levelSpacingEnergies = np.asarray(
                self.levelSpacingEnergies, dtype=float
            )


@dataclass
class TabulatedWidths:
    """§19.4.1. The unresolved resonance region."""

    spinGroups: List[UnresolvedSpinGroup] = field(default_factory=list)
    energyGrid: Optional[np.ndarray] = None
    scatteringRadius: Optional[float] = None
    #: The unit the radius above was read with. Same field, same reason as
    #: :attr:`~kika.nuclear_data.model.resonances.r_matrix.Channel.radiusUnit`.
    radiusUnit: Optional[str] = None
    #: §19.4.1's ``resonanceReactions`` — the channels the averages are for,
    #: each with the link to the reaction it is. All 351 unresolved blocks in
    #: ENDF/B-VIII.1-GNDS carry them, and the schema makes the ``<link>`` inside
    #: each one mandatory, so a writer that reconstructed the list from the
    #: channel labels alone could not produce a valid file. Typed as the
    #: resolved region's :class:`~.r_matrix.ResonanceReaction` because it is the
    #: same node; an ENDF-decoded evaluation leaves it empty, as ENDF states
    #: the URR channels only by position.
    resonanceReactions: List[object] = field(default_factory=list)
    #: ENDF LSSF: 0 = compute the URR cross sections from these parameters,
    #: 1 = they are already in MF3 and these are for self-shielding only.
    selfShieldingOnly: bool = False
    label: Optional[str] = None
    #: §19.4.1's local ``PoPs``, holding the target and — the part that matters
    #: for a calculation — its **spin**. The URR cross sections carry the same
    #: ``g_J = (2J+1) / (2(2I+1))`` factor the resolved region does, so a
    #: reconstructor needs I here as much as there. The resolved formalisms
    #: (:class:`~.breit_wigner.BreitWigner`, :class:`~.r_matrix.RMatrix`) have
    #: carried this field since 3b; the unresolved one did not, which made the
    #: two halves of the same calculation ask for the same number in different
    #: ways. Added in phase 4.
    PoPs: Optional[object] = None

    def __post_init__(self) -> None:
        if self.energyGrid is not None:
            self.energyGrid = np.asarray(self.energyGrid, dtype=float)
