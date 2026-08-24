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
from .r_matrix import (Channel, EXTERNAL_R_MATRIX_REQUIRED_TERMS,
                       EXTERNAL_R_MATRIX_TYPES, ExternalRMatrix, RMatrix,
                       RMatrixSpinGroup, ResonanceReaction)
from .tabulated_widths import (TabulatedWidths, UnresolvedChannel,
                               UnresolvedSpinGroup)

__all__ = [
    "BreitWigner", "BreitWignerApproximation", "Resonance", "ResonanceParameters",
    "SpinGroup", "Channel", "RMatrix", "RMatrixSpinGroup", "ResonanceReaction",
    "ExternalRMatrix", "EXTERNAL_R_MATRIX_TYPES",
    "EXTERNAL_R_MATRIX_REQUIRED_TERMS",
    "TabulatedWidths", "UnresolvedChannel", "UnresolvedSpinGroup",
    "ScatteringRadius", "ResolvedRegion", "UnresolvedRegion", "Resonances",
    "MODEL_RADIUS_UNIT", "FM_PER_ENDF_RADIUS", "radiusFromEndf", "radiusToEndf",
    "radiusFromStatedUnit",
]

#: **The unit every radius on this model is stated in, without exception.**
#:
#: Decided 2026-08-20 (owner: Juan): the canonical unit is GNDS's, because GNDS
#: is the newer format and the one that states its units at all. Before that
#: date the model carried whichever number its reader happened to produce —
#: 5.444 through GNDS and 0.5444 through ENDF for the same Fe-56 radius — and
#: :attr:`ScatteringRadius.unit` existed to *say so* rather than to fix it.
#: ``gnds_endf_conflicts.md`` §4.1 and §7.2 are the account of that.
#:
#: The invariant is now: **anything on this model that is a radius is in fm.**
#: That covers :attr:`ScatteringRadius.constant` and ``values``, and the bare
#: floats — :attr:`Channel.scatteringRadius`, :attr:`Channel.hardSphereRadius`,
#: :attr:`ResonanceReaction.scatteringRadius`, and the per-formalism
#: ``scatteringRadius`` of :class:`BreitWigner`, :class:`RMatrix` and
#: :class:`TabulatedWidths`. The ``unit`` / ``radiusUnit`` fields stay, and what
#: they record is what the *source* said; they are ``"fm"`` for anything that
#: came through either reader, and ``None`` only for a model built by hand.
MODEL_RADIUS_UNIT = "fm"

#: How many femtometres one ENDF radius unit is. ENDF-6 §2.2 writes AP, APL,
#: APT and APE in units of 10^-12 cm, and 10^-12 cm is ten femtometres.
FM_PER_ENDF_RADIUS = 10.0

#: Unit strings a reader may meet, and what one of them is in fm. ``None`` and
#: the empty string mean *the source stated nothing*, which for a radius that
#: reached us through the ENDF adapter means ENDF's unit — but the adapter
#: converts at the boundary and never calls this with ``None``, so a ``None``
#: here is a **GNDS** file whose axis carries no unit, and that is a file
#: defect rather than an ENDF convention. It is reported, not assumed.
_RADIUS_UNITS_IN_FM = {"fm": 1.0, "10*fm": FM_PER_ENDF_RADIUS,
                       "1e-12*cm": FM_PER_ENDF_RADIUS}


def radiusFromEndf(value):
    """An ENDF radius (AP, APL, APT, APE) → fm. ``None`` stays ``None``.

    Works on a scalar or a numpy array, because NRO=1 tabulates the radius and
    the whole table crosses the boundary at once.
    """
    return None if value is None else value * FM_PER_ENDF_RADIUS


def radiusToEndf(value):
    """fm → an ENDF radius. The inverse of :func:`radiusFromEndf`.

    **Called at exactly two kinds of place**, and it is worth knowing which:
    the encoder writes the file-level AP back from *provenance*, in ENDF's own
    units, untouched — so this is only for the radii that live on the model and
    nowhere else. Those are LRF=3's per-l APL and LRF=7's APT/APE. See
    ``kika/endf/model_adapter/resonances.py``.
    """
    return None if value is None else value / FM_PER_ENDF_RADIUS


def radiusFromStatedUnit(value, unit: Optional[str]):
    """A radius in *unit* → fm, or ``(value, None)`` when the unit is unknown.

    Returns ``(converted, problem)``: *problem* is ``None`` on success and a
    sentence naming the unit when it is not one this function knows. The caller
    decides whether that is a report entry or an error, because a reader and a
    consumer mean different things by it.
    """
    if value is None:
        return None, None
    if not unit:
        return value, ("the radius axis carries no unit, so the number cannot "
                       "be placed on the model's fm scale; it is kept as read")
    factor = _RADIUS_UNITS_IN_FM.get(unit)
    if factor is None:
        return value, (f"the radius axis is in {unit!r}, which is not a unit "
                       f"this reader converts; the number is kept as read and "
                       f"is therefore not on the model's fm scale")
    return value * factor, None


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
    #: ~~kika does **not** convert on read.~~ **It does, since 2026-08-20.**
    #: Both readers now put fm on the model — see :data:`MODEL_RADIUS_UNIT` —
    #: so ``constant`` is 5.444 for that Fe-56 whichever format it arrived in,
    #: and this field records what the *source* said rather than what the
    #: number means. It is ``"fm"`` for anything either reader produced.
    #:
    #: **What that cost, and where it was paid.** The ENDF reconstructor works
    #: in ENDF's units throughout, so converting here would have moved the
    #: reconstruction — the reason this was deferred. It does not, because the
    #: conversion back happens at the boundary into
    #: :mod:`kika.processing.resonance_formulas` rather than inside it, and
    #: ``test_numeric_goldens`` is the gate that says so.
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
    #: Not a GNDS node — the same escape hatch ``Reaction.provenance`` is, and
    #: here for a sharper reason. ``encodeMF2MT151`` takes ``(resonances,
    #: provenance)`` because QX, LRX, LAD and the twelve particle-pair columns
    #: have no model node; ``decodeReactionSuite`` used to **discard** that
    #: second return value, so a suite decoded from a tape carried resonances
    #: nobody could write back. Nothing noticed while the only caller was a test
    #: that held both halves itself; the whole-file writer (§2.8) is the first
    #: caller that has only the suite.
    provenance: Optional[object] = None

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
