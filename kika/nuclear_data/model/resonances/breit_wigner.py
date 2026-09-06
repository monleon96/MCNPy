"""GNDS-2.1 §19.3.6 ``BreitWigner``: SLBW and MLBW, with the widths **named**.

**This is where ``c3..c6`` dies.** The flat ``ResonanceRecord`` carries
``energy, spin, c3, c4, c5, c6`` — ENDF *record positions*, whose meaning
depends on the formalism: under SLBW/MLBW they are ``GN, GG, GF, -`` and under
Reich-Moore ``GN, GG, GFA, GFB``. The docstring on the flat class says the names
exist so the formula functions can duck-type over both
``kika.endf.classes.mf2.mf2mt151.Resonance`` and ``ResonanceRecord``.

A blanket rename to one set of physical names would be *wrong*, because no one
set is right for both formalisms. GNDS's answer — a node per formalism, each
with the names that formalism actually uses — is the correct one, and it is what
this module implements.

The flat class is **not touched**. It keeps ``c3..c6`` and keeps working;
:meth:`Resonance.toFlat` produces the positional tuple for the adapter, one way
only. That is deliberate: the roadmap's own risk assessment is that a direct
rename "can produce silently wrong reconstructed cross sections, with no test in
existence to catch it".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

__all__ = ["BreitWignerApproximation", "Resonance", "ResonanceParameters",
           "SpinGroup", "BreitWigner"]


class BreitWignerApproximation(str, Enum):
    """§19.3.6. Which Breit-Wigner approximation the parameters are for."""

    singleLevel = "SingleLevel"
    multiLevel = "MultiLevel"


@dataclass
class Resonance:
    """One resonance, with its widths named rather than numbered."""

    energy: float
    spin: float
    totalWidth: float = 0.0
    neutronWidth: float = 0.0
    captureWidth: float = 0.0
    fissionWidth: float = 0.0

    def toFlat(self) -> Tuple[float, float, float, float, float, float]:
        """``(energy, spin, c3, c4, c5, c6)`` for the ENDF adapter.

        One-way, and used only by ``kika/endf/model_adapter``. Under SLBW/MLBW
        the ENDF record positions are ``GT, GN, GG, GF``, which is the order
        returned here.
        """
        return (
            self.energy, self.spin,
            self.totalWidth, self.neutronWidth, self.captureWidth, self.fissionWidth,
        )

    @classmethod
    def fromFlat(cls, energy: float, spin: float, c3: float, c4: float,
                 c5: float, c6: float) -> "Resonance":
        """The inverse, for the decoder. ``c3..c6`` are ``GT, GN, GG, GF`` here."""
        return cls(energy=energy, spin=spin, totalWidth=c3, neutronWidth=c4,
                   captureWidth=c5, fissionWidth=c6)


@dataclass
class SpinGroup:
    """Resonances sharing an orbital angular momentum ``L``.

    ENDF calls this an l-block. ``scatteringRadius`` is the l-dependent radius
    that phase 1 found MF2/151 was dropping entirely — on the committed Fe-56
    slice, keeping it moved elastic by a median 2% and up to 42% locally.
    """

    L: int
    resonances: List[Resonance] = field(default_factory=list)
    scatteringRadius: Optional[float] = None
    atomicWeightRatio: Optional[float] = None

    def __len__(self) -> int:
        return len(self.resonances)


@dataclass
class ResonanceParameters:
    """The table of resonances, grouped by L."""

    spinGroups: List[SpinGroup] = field(default_factory=list)

    @property
    def numberOfResonances(self) -> int:
        return sum(len(g) for g in self.spinGroups)


@dataclass
class BreitWigner:
    """§19.3.6. A resolved region in the Breit-Wigner formalism."""

    approximation: BreitWignerApproximation = BreitWignerApproximation.multiLevel
    resonanceParameters: ResonanceParameters = field(default_factory=ResonanceParameters)
    scatteringRadius: Optional[float] = None
    #: The unit the radius above was read with. Same field, same reason as
    #: :attr:`~kika.nuclear_data.model.resonances.r_matrix.Channel.radiusUnit`.
    radiusUnit: Optional[str] = None
    PoPs: Optional[object] = None
    label: Optional[str] = None
    #: §19.3.6. ENDF's NAPS=0 — compute the channel radius from the target mass
    #: instead of using ``scatteringRadius``. 325 of the library's 386
    #: ``BreitWigner`` nodes set it. See
    #: :attr:`~kika.nuclear_data.model.resonances.r_matrix.RMatrix.calculateChannelRadius`.
    calculateChannelRadius: bool = False

    @property
    def numberOfResonances(self) -> int:
        return self.resonanceParameters.numberOfResonances
