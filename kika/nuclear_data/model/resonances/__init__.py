"""GNDS-2.1 §19: resonances, split by *formalism* rather than by record position.

``resolved`` holds a :class:`~.breit_wigner.BreitWigner` or an
:class:`~.r_matrix.RMatrix`; ``unresolved`` holds
:class:`~.tabulated_widths.TabulatedWidths`. Which one a file uses is a property
of the evaluation, not of the reader, and the difference is what makes ENDF's
``c3..c6`` positional widths unnameable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .breit_wigner import (BreitWigner, BreitWignerApproximation, Resonance,
                           ResonanceParameters, SpinGroup)
from .r_matrix import Channel, RMatrix, RMatrixSpinGroup, ResonanceReaction
from .tabulated_widths import (TabulatedWidths, UnresolvedChannel,
                               UnresolvedSpinGroup)

__all__ = [
    "BreitWigner", "BreitWignerApproximation", "Resonance", "ResonanceParameters",
    "SpinGroup", "Channel", "RMatrix", "RMatrixSpinGroup", "ResonanceReaction",
    "TabulatedWidths", "UnresolvedChannel", "UnresolvedSpinGroup",
    "ScatteringRadius", "ResolvedRegion", "UnresolvedRegion", "Resonances",
]


@dataclass
class ScatteringRadius:
    """§19. The scattering radius, constant or energy-dependent.

    ENDF's NRO=1 tables and the l-dependent APL both land here. Phase 1 found
    that MF2/151's l-dependent radius was being dropped outright; the precedence
    it settled on — an energy-dependent table still wins, because ENDF gives no
    per-l version of one — is a property of the ENDF adapter, not of this node.
    """

    constant: Optional[float] = None
    energies: Optional[object] = None
    values: Optional[object] = None
    #: The table's ``(NBT, INT)`` pairs. A table without them is a set of points
    #: that only a convention connects, and the convention is not always
    #: lin-lin — so a reconstruction reading this node would have to assume one,
    #: on the one quantity that sets the hard-sphere phase shift. They lived in
    #: ENDF provenance alone until phase 4, which is enough to write the file
    #: back and not enough to compute from it. Additive: the encoder still
    #: writes the table from provenance, so no round trip moves.
    interpolation: Optional[object] = None
    #: The unit the radius is stated in, or ``None`` for *not stated*.
    #:
    #: **The two readers do not agree on the number, and this is what says so.**
    #: For ENDF/B-VIII.1's Fe-56 the GNDS path gives 5.444 and the ENDF path
    #: 0.5444, because ENDF writes AP in units of 10^-12 cm — ten femtometres —
    #: while GNDS writes it in fm with an axis that says so. Both are the same
    #: radius. Until this field existed both landed in ``constant`` and a
    #: consumer reading it got a silent factor of ten depending on which
    #: encoding the evaluation happened to arrive in.
    #:
    #: kika does **not** convert on read. The ENDF reconstructor works in ENDF's
    #: units throughout and rescaling underneath it would move a number the
    #: thesis pipeline depends on. What changes here is that the unit is stated
    #: where the format states it, so the mismatch is visible rather than latent;
    #: making the two paths agree on one canonical unit is a separate change
    #: with its own gate.
    unit: Optional[str] = None

    @property
    def isEnergyDependent(self) -> bool:
        return self.values is not None


@dataclass
class ResolvedRegion:
    """§19.2. One resolved energy range and the formalism describing it."""

    domainMin: float
    domainMax: float
    domainUnit: str = "eV"
    formalism: Optional[object] = None   # BreitWigner | RMatrix


@dataclass
class UnresolvedRegion:
    """§19.4. The unresolved energy range."""

    domainMin: float
    domainMax: float
    domainUnit: str = "eV"
    tabulatedWidths: Optional[TabulatedWidths] = None


@dataclass
class Resonances:
    """§19. The ``resonances`` child of a ``reactionSuite``."""

    scatteringRadius: Optional[ScatteringRadius] = None
    resolved: List[ResolvedRegion] = field(default_factory=list)
    unresolved: Optional[UnresolvedRegion] = None

    @property
    def domain(self) -> Optional[Tuple[float, float]]:
        """``(low, high)`` over every real region, or ``None`` when there is none.

        The model's counterpart to ``detect_resonance_bounds``, which the app
        calls and which reads ENDF structures directly.
        """
        bounds = [(r.domainMin, r.domainMax) for r in self.resolved]
        if self.unresolved is not None:
            bounds.append((self.unresolved.domainMin, self.unresolved.domainMax))
        if not bounds:
            return None
        return min(b[0] for b in bounds), max(b[1] for b in bounds)

    def __repr__(self) -> str:
        formalisms = ", ".join(
            type(r.formalism).__name__ if r.formalism is not None else "unset"
            for r in self.resolved
        ) or "none"
        domain = self.domain
        span = "no region" if domain is None else f"{domain[0]:g}-{domain[1]:g} eV"
        return (
            f"Resonances(resolved=[{formalisms}], "
            f"unresolved={self.unresolved is not None}, {span})"
        )
