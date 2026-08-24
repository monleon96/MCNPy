"""GNDS-2.1 §19.3.1-19.3.5 ``RMatrix``: spin groups and channels.

Reich-Moore and R-Matrix-Limited. Widths here belong to a *channel*, not to a
record position, which is the structural reason ``c3..c6`` cannot express them:
under Reich-Moore the ENDF positions mean ``GN, GG, GFA, GFB`` — two fission
widths — and there is no fourth name for a formalism with more channels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..quantities import PhysicalQuantity

__all__ = ["ExternalRMatrix", "Channel", "ResonanceReaction",
           "RMatrixSpinGroup", "RMatrix"]

#: The two parametrisations §19.3.4 admits, and the *only* two: ``type`` is an
#: ``xs:enumeration`` in the schema (``gnds.xsd:935-940``), unlike the term
#: labels below it, which the schema deliberately leaves open — its own comment
#: there reads ``FIXME redefine the 'double' and restrict allowed labels?``.
EXTERNAL_R_MATRIX_TYPES = ("Froehner", "SAMMY")

#: The two terms both parametrisations require. §19.3.4 writes each as a
#: singularity of the external R-matrix, so neither formula can be evaluated
#: without them; FUDGE raises on their absence for the same reason
#: (``fudge/resonances/externalRMatrix.py:30-34``). Every other term defaults
#: to zero, which is why the schema allows as few as two ``<double>``.
EXTERNAL_R_MATRIX_REQUIRED_TERMS = ("singularityEnergyBelow",
                                    "singularityEnergyAbove")


@dataclass
class ExternalRMatrix:
    """§19.3.4's ``externalRMatrix``: the R-matrix *outside* the fitted range.

    A contribution added to the R-matrix diagonal during reconstruction, standing
    in for levels the evaluator did not fit. Two parametrisations exist and they
    are not variants of one formula: SAMMY's is a polynomial in energy plus a
    logarithmic term and is purely real; Froehner's is an arctangent plus a
    genuinely **imaginary** part. A consumer that reads the terms without reading
    :attr:`type` has no way to tell which it holds.

    **Why the terms are a list of labelled quantities and not seven fields.**
    ``gnds.xsd:942-948`` types the children as two to seven ``PQU_double`` and
    says of them, in its own comment, *"FIXME redefine the 'double' and restrict
    allowed labels?"* — the label set is the parametrisation's, not the schema's.
    Naming them as fields would put SAMMY's five and Froehner's three optional
    terms in one class and let a caller build a SAMMY holding
    ``averageRadiationWidth``. A list keyed by the label the file states carries
    both parametrisations with no branch, which matters here more than usual:
    **the corpus contains no Froehner node at all** — 7 ``externalRMatrix`` in
    the 558 distributed neutron evaluations, all in ``n-038_Sr_088``, all SAMMY
    (measured 2026-08-24; every other ``Froehner`` in the library is the
    evaluator's prose citing the man). A branch for a form nothing exercises is
    untested code by construction, and this design has none.

    Each term is a :class:`~kika.nuclear_data.model.quantities.PhysicalQuantity`
    because the units are not uniform and not guessable: ``linearExternalR`` is
    ``1/eV``, ``quadraticExternalR`` is ``1/eV**2``, the singularity energies are
    ``eV`` and the constants are dimensionless. §2.3.3's ``label`` exists for
    exactly this — telling apart several assignments of the same kind of scalar.

    **ENDF states this too**, which is why the node is not a GNDS curiosity:
    LRF=7's KBK/LBK records are the same object, with LBK=2 SAMMY's and LBK=3
    Froehner's. ``kika/endf/model_adapter/resonances.py`` still carries those
    verbatim in provenance rather than through here — see
    ``docs/library/gnds_endf_conflicts.md`` §6.4, which sizes that migration and
    notes the one form this node cannot take: LBK=1 tabulates the background as
    two curves, and GNDS has no ``externalRMatrix`` form for a table.
    """

    #: ``'SAMMY'`` or ``'Froehner'``. Required, and restricted by the schema.
    type: str
    #: The parametrisation's terms, in the order the source stated them.
    terms: List[PhysicalQuantity] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.type not in EXTERNAL_R_MATRIX_TYPES:
            raise ValueError(
                f"externalRMatrix type {self.type!r} is not one of "
                f"{EXTERNAL_R_MATRIX_TYPES}; gnds.xsd:935-940 restricts it"
            )
        labels = [term.label for term in self.terms]
        duplicated = {label for label in labels if labels.count(label) > 1}
        if duplicated:
            raise ValueError(
                f"externalRMatrix has more than one term labelled "
                f"{sorted(duplicated)}; the label is what identifies a term"
            )
        missing = [name for name in EXTERNAL_R_MATRIX_REQUIRED_TERMS
                   if name not in labels]
        if missing:
            raise ValueError(
                f"externalRMatrix is missing {missing}; §19.3.4's two "
                f"parametrisations both place a singularity at each end of the "
                f"range and neither can be evaluated without them"
            )

    def term(self, label: str) -> Optional[PhysicalQuantity]:
        """The term stated under *label*, or ``None`` where the file omits one.

        ``None`` and zero are not the same statement — the file says nothing and
        the reconstruction takes it as zero — so this returns the first and
        leaves the second to the caller, which is what FUDGE's ``getTerm`` folds
        together at the point of use rather than at the point of reading.
        """
        for term in self.terms:
            if term.label == label:
                return term
        return None


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
    #:
    #: Under LRF=7 this is ENDF's **APT**, the *true* channel radius — the one
    #: that enters the penetrability and shift factors.
    scatteringRadius: Optional[float] = None
    #: §19.3.4. ENDF's **APE**, the *effective* channel radius, which sets the
    #: hard-sphere phase shift. LRF=7 writes APE and APT as separate columns and
    #: they are routinely different; collapsing them into one radius made an
    #: evaluation that distinguishes them unrepresentable, so a round trip could
    #: not reproduce the file and a reconstruction would use one radius where the
    #: evaluator asked for two.
    hardSphereRadius: Optional[float] = None
    #: §19.3.4. ENDF's **BND**, the channel's R-matrix boundary condition. Read
    #: by the parser, held by no model node until now, so an LRF=7 round trip
    #: wrote back a boundary condition of zero whatever the file said.
    boundaryConditionValue: Optional[float] = None
    #: §19.3.4's ``externalRMatrix``, the R-matrix outside the fitted range —
    #: ENDF's LBK=2/LBK=3 background R-matrix under LRF=7. ``None`` where the
    #: channel states none, which is all but 7 channels of the 558 distributed
    #: neutron evaluations. Modelled 2026-08-24, closing the last open row of
    #: ``docs/library/gnds_endf_conflicts.md`` §6.4; before that it was read,
    #: reported and dropped, so a file kika round tripped came back with the
    #: evaluator's background R-matrix silently gone. It is the schema's **first**
    #: child of a channel (``gnds.xsd:915-919``, an ``xs:sequence`` ahead of both
    #: radii), which is where the writer puts it.
    externalRMatrix: Optional["ExternalRMatrix"] = None
    #: The unit the two radii above were read with — ``fm`` from GNDS, ``None``
    #: from ENDF, which declares none. Both share one field because they come
    #: off the same node and no file states them differently.
    #:
    #: It exists because the alternative was writing a *false* unit: the GNDS
    #: writer used to label these ``fm`` unconditionally, and an ENDF-sourced
    #: radius is a tenth of a femtometre count (AP in units of 10^-12 cm). See
    #: :attr:`~kika.nuclear_data.model.resonances.ScatteringRadius.unit`, which
    #: makes the same statement for the file-level radius, and note the type is
    #: deliberately *not* ``ScatteringRadius``: ``reconstruct.py`` reads
    #: ``channels[0].scatteringRadius`` as a number into the penetrability.
    radiusUnit: Optional[str] = None


@dataclass
class ResonanceReaction:
    """§19.3.3. A reaction the resonances can decay through."""

    label: str
    ejectile: Optional[str] = None
    Q: Optional[float] = None
    eliminated: bool = False
    scatteringRadius: Optional[float] = None
    #: §19.3.3's ``<link href=.../>``: the xPath of the ``reaction`` this channel
    #: *is*. All 902 ``resonanceReaction`` nodes in ENDF/B-VIII.1-GNDS carry one,
    #: and it is the only formal tie between the resonance block and the
    #: reactions — a ``label`` that matches a reaction label is a convention the
    #: files keep, not a statement the format makes. Kept unfollowed, the same
    #: treatment :class:`~kika.nuclear_data.model.cross_section_forms.Reference`
    #: gets. ``None`` for an ENDF-decoded evaluation, which has no link to give.
    href: Optional[str] = None
    #: The unit ``scatteringRadius`` was read with. Same field, same reason as
    #: :attr:`Channel.radiusUnit`, and it covers ``hardSphereRadius`` too — they
    #: come off sibling nodes and no file states them differently.
    radiusUnit: Optional[str] = None
    #: §19.3.3 allows a hard-sphere radius here as well as on each channel
    #: (§19.3.4). **Four nodes in three files** carry one — V-51, Ca-40 and
    #: Cl-35, measured over the 558 distributed neutron evaluations on
    #: 2026-08-24 — and until then the reader dropped it with a report entry,
    #: on the argument that the channel's is the one the phase shift uses.
    #: That argument is about which radius the *physics* reads, not about which
    #: one the file states, so it justified not consuming the number and never
    #: justified not carrying it: a reader that drops it cannot write the file
    #: back. Modelled now, one field beside its sibling. See
    #: ``docs/library/gnds_endf_conflicts.md`` §6.4.
    hardSphereRadius: Optional[float] = None


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
    #: The range's own scattering radius — ENDF's **AP**, the scalar on the range
    #: record. :class:`~.breit_wigner.BreitWigner` and
    #: :class:`~.tabulated_widths.TabulatedWidths` have carried this since 3b and
    #: ``RMatrix`` did not, which left the Reich-Moore range's AP reachable only
    #: through ``Resonances.scatteringRadius`` — a **file-level** slot filled by
    #: whichever range happened to be first. For the single-range tapes to hand
    #: those are the same number; for a file whose ranges declare different radii
    #: they are not, and a reconstruction reading the file-level one would use
    #: another range's radius without saying so. Added in phase 4, additively:
    #: the encoder writes AP from provenance, so no round trip moves.
    #:
    #: Per-*channel* radii (ENDF's APL under LRF=3, APT/APE under LRF=7) are a
    #: different quantity and stay on :class:`Channel`, which is where §19.3.4
    #: puts them.
    scatteringRadius: Optional[float] = None
    #: The unit the radius above was read with. Same field, same reason as
    #: :attr:`Channel.radiusUnit`.
    radiusUnit: Optional[str] = None
    boundaryCondition: Optional[str] = None
    #: §19.3.1. ENDF's **IFG**. ``False`` — the default and the common case —
    #: means ``widths`` are widths in eV; ``True`` means they are reduced-width
    #: amplitudes, in eV^½, and are *not* interchangeable with them.
    #:
    #: This is the one place this class deviates from the written form of
    #: decision 1(a), which listed IFG as provenance. The principle that decision
    #: states — physics on the model, bookkeeping in provenance — puts it here:
    #: a consumer holding ``widths`` and unable to ask what they mean is exactly
    #: the ``c3..c6`` problem this package exists to fix, one level up. GNDS
    #: gives it an attribute for the same reason.
    reducedWidthAmplitudes: bool = False
    #: §19.3.1. ENDF's **KRL**: ``True`` selects relativistic kinematics. Never
    #: read at all before B1a — not by the model and not by the flat path.
    relativisticKinematics: bool = False
    #: §19.3.1. ``True`` means the channel radius is to be **computed from the
    #: target mass** rather than taken from ``scatteringRadius`` — ENDF's NAPS=0.
    #: Physics, not bookkeeping: a reconstructor that ignores it puts the
    #: evaluator's AP into the penetrability where the evaluator asked for
    #: 0.123 A^(1/3) + 0.08, and the two differ by percent.
    calculateChannelRadius: bool = False

    @property
    def numberOfResonances(self) -> int:
        return sum(len(g) for g in self.spinGroups)
