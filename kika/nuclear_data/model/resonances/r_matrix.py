"""GNDS-2.1 §19.3.1-19.3.5 ``RMatrix``: spin groups and channels.

Reich-Moore and R-Matrix-Limited. Widths here belong to a *channel*, not to a
record position, which is the structural reason ``c3..c6`` cannot express them:
under Reich-Moore the ENDF positions mean ``GN, GG, GFA, GFB`` — two fission
widths — and there is no fourth name for a formalism with more channels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

__all__ = ["Channel", "ResonanceReaction", "RMatrixSpinGroup", "RMatrix"]


@dataclass
class Channel:
    """§19.3.4. One reaction channel of a spin group."""

    label: str
    resonanceReaction: str
    L: Optional[int] = None
    channelSpin: Optional[float] = None
    columnIndex: Optional[int] = None
    #: §19.3.4 lets a channel carry its own radius. This is where ENDF's APL
    #: lands: under Reich-Moore the radius is per l-block, which is per spin
    #: group here, so every channel of that group gets it. Phase 1 found this
    #: value was being dropped entirely.
    scatteringRadius: Optional[float] = None


@dataclass
class ResonanceReaction:
    """§19.3.3. A reaction the resonances can decay through."""

    label: str
    ejectile: Optional[str] = None
    Q: Optional[float] = None
    eliminated: bool = False
    scatteringRadius: Optional[float] = None


@dataclass
class RMatrixSpinGroup:
    """§19.3.5. Resonances of one total spin and parity, over several channels.

    ``widths`` is ``[[width per channel] per resonance]``, parallel to
    ``channels`` — so a width is identified by *which channel it belongs to*,
    which is the whole point.
    """

    label: str
    spin: Optional[float] = None
    parity: Optional[int] = None
    channels: List[Channel] = field(default_factory=list)
    energies: List[float] = field(default_factory=list)
    widths: List[List[float]] = field(default_factory=list)
    #: Per-resonance J. In a true R-matrix spin group every resonance shares the
    #: group's ``spin``, and this is empty. **ENDF's LRF=3 is not that shape**:
    #: it groups by *l*, not by J, and writes AJ on every resonance record. The
    #: first version of the decoder kept only energies and widths, which dropped
    #: AJ silently — caught when the phase 3d façade tried to project back to
    #: ``ResonanceRecord.spin`` and had nothing to read.
    spins: List[float] = field(default_factory=list)
    atomicWeightRatio: Optional[float] = None

    def __post_init__(self) -> None:
        if self.spins and len(self.spins) != len(self.energies):
            raise ValueError(
                f"{len(self.spins)} spins for {len(self.energies)} resonances"
            )
        if self.widths and len(self.widths) != len(self.energies):
            raise ValueError(
                f"{len(self.widths)} width rows for {len(self.energies)} resonances"
            )
        for row in self.widths:
            if len(row) != len(self.channels):
                raise ValueError(
                    f"a width row has {len(row)} entries for {len(self.channels)} channels"
                )

    def __len__(self) -> int:
        return len(self.energies)


@dataclass
class RMatrix:
    """§19.3.1. A resolved region in the R-Matrix formalism."""

    approximation: Optional[str] = None   # 'ReichMoore', 'RMatrixLimited', ...
    resonanceReactions: List[ResonanceReaction] = field(default_factory=list)
    spinGroups: List[RMatrixSpinGroup] = field(default_factory=list)
    PoPs: Optional[object] = None
    label: Optional[str] = None
    boundaryCondition: Optional[str] = None

    @property
    def numberOfResonances(self) -> int:
        return sum(len(g) for g in self.spinGroups)
