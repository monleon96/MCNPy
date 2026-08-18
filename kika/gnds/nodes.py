"""Every GNDS node kika reads, every node kika writes, and where the two differ.

**The problem this exists for.** The reader dispatches on XML tag strings; the
writer dispatches on ``isinstance`` of model classes and emits a tag literal.
The two halves are not indexed by the same thing, so nothing compares them —
and "the reader knows a node the writer does not" is not a failure any test can
observe. Four such asymmetries were found by writing this table, one of them a
node kika **emitted into files** that its own reader silently dropped and that
the schema does not admit at all — fixed in the commit after this table landed,
and the count in :data:`KNOWN_DEFECTS` is what recorded it.

**What it is.** One entry per node at a *choice point*, keyed by ``(family,
tag)``. Each entry says which side handles it and — when only one does — why
that is correct, or that it is not. The dispatch tables the reader already had
(``primitives.FUNCTION_1D``, ``covariances.COVARIANCE_FORMS``) are **derived**
from this rather than living beside it, so there is one list and not two.

**Why choice points and not every node.** The criterion is the schema, not
taste. An ``xs:choice`` is where the reader can land on a form whose writer does
not exist; an ``xs:sequence`` is fixed grammar, where the node is mandatory and
the round trip catches a missing writer on its own. §19's resonance grammar is
all sequence — a missing ``spinGroup`` moves every resonance and
``test_resonance_oracle`` sees it — so it is not here. Its real asymmetries are
of *attribute* and of *child*, which a tag table cannot express; those are in
:data:`KNOWN_DEFECTS` and :data:`ONE_SIDED_ATTRIBUTES`, and the honest thing to
say about them is that writing this table is what found them, not that the
table asserts them.

**Why the key is a pair.** The plan for this module said "indexed by the tag
string", and the data says otherwise: ``constant1d`` is both a §6 functional and
a §17.3 multiplicity form, ``unspecified`` is both a multiplicity form and a
distribution form, ``reference`` is a form of three different things, and
``isotropic2d`` means one thing under ``angularTwoBody`` and — until this
landed — something invalid under ``distribution``. The reader dispatches on the
tag *within a context*, so the context is half the key.

**What makes the invariant violable rather than tautological.** If one dict held
both halves and both sides read from it, "the key sets are equal" would be true
by construction and would assert nothing. So the table below is *static* — hand
written, never derived from the code — and the two sides **register against it**
with :func:`reads` and :func:`writes`, which raise at **import time** for a tag
the table does not declare. Adding a §18 law reader in phase 7b without
declaring it fails ``import kika.gnds``, not a test somebody might not run. The
other direction is :func:`check`, which lists the entries whose declared status
disagrees with what actually registered.

The writer's ``isinstance`` chains are deliberately **not** extracted into this
table. What is registered is the family's entry point; that the chain inside it
reaches the right tag is pinned behaviourally in ``test_nodes.py`` — build a
minimal ``spec.cls``, call the writer, assert the emitted tag is the key. That
exercises the chain, where a second copy of it here would only exercise itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional, Tuple

from kika.nuclear_data.model import (AngularEnergy, AngularTwoBody,
                                     AngularDistributionReconstructed,
                                     Constant1d, CoulombPlusNuclearElastic,
                                     CovarianceMatrix,
                                     CrossSectionReconstructed, EnergyAngular,
                                     Evaluated, Gridded1d, GriddedCrossSection,
                                     Heated, HeatedMultiGroup, Isotropic2d,
                                     KalbachMann, Legendre, Mixed,
                                     Polynomial1d, Realization, Reference,
                                     Regions1d, Regions2d, Regions3d,
                                     ResonancesWithBackground,
                                     ShortRangeSelfScalingVariance, Sum,
                                     ThermalNeutronScatteringLaw1d,
                                     URR_probabilityTables,
                                     URR_probabilityTables1d, Uncorrelated,
                                     Unspecified, XYs1d, XYs2d, XYs3d, Ys1d)
from kika.nuclear_data.model.styles import (AverageProductData,
                                            CoulombPlusNuclearElasticMuCutoff,
                                            MonteCarlo_cdf, SnElasticUpScatter)

__all__ = ["Status", "NodeSpec", "NODES", "KNOWN_DEFECTS",
           "NOT_A_DISTRIBUTION_FORM", "ONE_SIDED_ATTRIBUTES", "reads", "writes",
           "readersOf", "check", "UndeclaredNode"]


class Status(str, Enum):
    """What the table *claims* about a node. :func:`check` tests the claim."""

    #: Both sides handle it. The ordinary case, and the only one needing no
    #: reason.
    PAIRED = "paired"
    #: The reader names it — usually to report it — and no writer emits it.
    READ_ONLY = "readOnly"
    #: The writer emits it and no reader takes it back.
    WRITE_ONLY = "writeOnly"
    #: The model declares the class and neither side touches the node. Where a
    #: later phase lands, and the reason says which.
    NEITHER = "neither"


@dataclass(frozen=True)
class NodeSpec:
    """One node at one choice point.

    ``cls`` is the model class the node becomes, or ``None`` where there is no
    class — a ``branching1d`` multiplicity is *named* by the reader and turned
    into nothing. It is the class→tag direction, and it lives here rather than
    on the class: ``gndsNodeName`` exists on exactly three families of model
    class and in all three it means "the node name is **not** the class name
    lower-cased", i.e. it marks an exception. Adding it to the twenty-one
    classes that already round trip would silently turn a marker into a map,
    and would edit five modules of the model for the convenience of this one.
    """

    tag: str
    family: str
    section: str
    cls: Optional[type]
    status: Status
    reason: Optional[str] = None

    @property
    def key(self) -> Tuple[str, str]:
        return (self.family, self.tag)


def _spec(*args, **kwargs) -> Tuple[Tuple[str, str], NodeSpec]:
    spec = NodeSpec(*args, **kwargs)
    return spec.key, spec


#: Phase 7b's landing strip: the §18 laws and the multiplicity forms kika
#: declares are already here as ``READ_ONLY``/``NEITHER`` entries with a reason.
#: When their writers land they become ``PAIRED``, and :func:`check` is what
#: notices that the declaration and the code agree again.
#:
#: **Four §18 entries carry this, not five.** ``gnds.xsd:1647-1662`` gives
#: §18.1.1's choice twelve members; kika names six of them below and the count
#: of *unimplemented* ones is four — ``uncorrelated``, ``energyAngular``,
#: ``angularEnergy``, ``KalbachMann``. ``branching3d`` is deliberately absent,
#: as the guinea pig of the import-time ratchet (see :func:`reads` and its
#: test). The remaining five — ``reference``, ``CoulombPlusNuclearElastic``,
#: ``coherentPhotonScattering``, ``incoherentPhotonScattering``,
#: ``thermalNeutronScatteringLaw`` — occur **zero times** across the 558 neutron
#: evaluations the census walked, which is why they are not entries.
_PHASE_7B = ("declared in the model and not implemented — "
             "kika/nuclear_data/model/distributions.py:163 lists them — "
             "and scheduled for phase 7b")

#: Two of the six names in
#: :data:`~kika.nuclear_data.model.distributions.NOT_IMPLEMENTED_DISTRIBUTIONS`
#: are **not** members of §18.1.1's choice, so they cannot have a
#: ``distributionForm`` entry. Their real choice point is named here rather than
#: left to be rediscovered, because the model's list reads like a list of §18
#: laws and is not one.
NOT_A_DISTRIBUTION_FORM = {
    "recoil": "angularTwoBodyForm — gnds.xsd:1670, where it is already PAIRED "
              "as a bare <recoil href=…/> onto AngularTwoBody.recoilHref",
    "NBodyPhaseSpace": "§18.3's energy choice — gnds.xsd:1703, inside "
                       "uncorrelated/energy, so it lands with uncorrelated",
}

NODES: Dict[Tuple[str, str], NodeSpec] = dict([
    # -- §5-6 functionals. `primitives.FUNCTION_1D` / `_2D` derive from these.
    _spec("XYs1d", "function1d", "§6.2.2", XYs1d, Status.PAIRED),
    _spec("regions1d", "function1d", "§6.3.2", Regions1d, Status.PAIRED),
    _spec("constant1d", "function1d", "§6.2.1", Constant1d, Status.PAIRED),
    _spec("Legendre", "function1d", "§6.2.4", Legendre, Status.PAIRED),
    _spec("polynomial1d", "function1d", "§6.2.3", Polynomial1d, Status.PAIRED),
    _spec("Ys1d", "function1d", "§6.2.5", Ys1d, Status.NEITHER,
          "modelled; no distributed neutron evaluation carries one, and "
          "gnds.xsd:1206 does not admit it at any choice point kika reads"),
    _spec("gridded1d", "function1d", "§6.2.6", Gridded1d, Status.NEITHER,
          "modelled; §6.2.6 is a processed representation and kika reads "
          "evaluated data"),

    _spec("XYs2d", "function2d", "§6.4.2", XYs2d, Status.PAIRED),
    _spec("regions2d", "function2d", "§6.4.3", Regions2d, Status.PAIRED),
    _spec("XYs3d", "function3d", "§6.5", XYs3d, Status.NEITHER,
          "kika/nuclear_data/model/functions/higher.py:307 — declared, "
          "unimplemented, no phase scheduled"),
    _spec("regions3d", "function3d", "§6.5", Regions3d, Status.NEITHER,
          "kika/nuclear_data/model/functions/higher.py:311 — declared, "
          "unimplemented, no phase scheduled"),

    # -- §16.1.1 crossSection forms. gnds.xsd:1206, an xs:choice.
    #    XYs1d and regions1d are also legal here and are delegated to the
    #    functional family above rather than declared twice.
    _spec("resonancesWithBackground", "crossSectionForm", "§16.1.1",
          ResonancesWithBackground, Status.PAIRED),
    _spec("reference", "crossSectionForm", "§16.1.1", Reference, Status.PAIRED),
    _spec("CoulombPlusNuclearElastic", "crossSectionForm", "§16.1.1",
          CoulombPlusNuclearElastic, Status.NEITHER,
          "modelled; §16.1.1 admits it for charged-particle elastic, and it is "
          "absent from every neutron evaluation kika reads"),
    _spec("thermalNeutronScatteringLaw1d", "crossSectionForm", "§16.1.1",
          ThermalNeutronScatteringLaw1d, Status.NEITHER,
          "modelled; §16.1.1 admits it, and TSL evaluations reach kika through "
          "kika/endf/classes/mf7, not here"),
    _spec("URR_probabilityTables1d", "crossSectionForm", "§16.1.1",
          URR_probabilityTables1d, Status.NEITHER,
          "modelled, and **produced live** by "
          "kika/ace/model_adapter/decode.py:284 — so an ACE-decoded suite "
          "already holds one and neither half of kika/gnds can move it"),

    # -- §17.3 multiplicity forms. gnds.xsd:1626, an xs:choice.
    _spec("constant1d", "multiplicityForm", "§17.3", Constant1d, Status.PAIRED),
    _spec("reference", "multiplicityForm", "§17.3", None, Status.READ_ONLY,
          "kika/gnds/decode.py:576-597 names it and reports it; kika's Multiplicity "
          "is a constant or a function of E, and a link needs a node the model "
          "does not have. " + _PHASE_7B),
    _spec("unspecified", "multiplicityForm", "§17.3", None, Status.READ_ONLY,
          "kika/gnds/decode.py:576-597 names it and reports it. " + _PHASE_7B),
    _spec("branching1d", "multiplicityForm", "§17.3", None, Status.READ_ONLY,
          "kika/gnds/decode.py:576-597 names it and reports it; an isomeric "
          "branching is not a multiplicity kika can evaluate. " + _PHASE_7B),

    # -- §18.1.1 distribution forms. gnds.xsd:1647, an xs:choice.
    _spec("angularTwoBody", "distributionForm", "§18.2", AngularTwoBody,
          Status.PAIRED),
    _spec("unspecified", "distributionForm", "§18.1.1", Unspecified,
          Status.PAIRED),
    _spec("uncorrelated", "distributionForm", "§18.3", Uncorrelated,
          Status.NEITHER, _PHASE_7B),
    _spec("energyAngular", "distributionForm", "§18.4", EnergyAngular,
          Status.NEITHER, _PHASE_7B),
    _spec("angularEnergy", "distributionForm", "§18.5", AngularEnergy,
          Status.NEITHER, _PHASE_7B),
    _spec("KalbachMann", "distributionForm", "§18.6", KalbachMann,
          Status.NEITHER, _PHASE_7B),

    # -- §18.2's angularTwoBody forms. gnds.xsd:1665, an xs:choice.
    #    XYs2d and regions2d delegate to the functional family.
    _spec("isotropic2d", "angularTwoBodyForm", "§18.2", Isotropic2d,
          Status.PAIRED),
    _spec("recoil", "angularTwoBodyForm", "§18.2", None, Status.PAIRED,
          "read onto AngularTwoBody.recoilHref and written back from it; the "
          "link has no class of its own"),

    # -- §25 covariance forms. covariances.xsd:83-87, an xs:choice.
    _spec("covarianceMatrix", "covarianceForm", "§25.2", CovarianceMatrix,
          Status.PAIRED),
    _spec("mixed", "covarianceForm", "§25.2", Mixed, Status.PAIRED),
    _spec("sum", "covarianceForm", "§25.2", Sum, Status.PAIRED),
    _spec("shortRangeSelfScalingVariance", "covarianceForm", "§25.2",
          ShortRangeSelfScalingVariance, Status.PAIRED),

    # -- §9 styles. Declaration only: `writeStyles` is already table-driven by
    #    `gndsNodeName`, so there is no dispatch here to convert -- what the
    #    table adds is the count. Two are read, twelve can be written, and
    #    `RS_StylesType` (gnds.xsd:87-93) admits four.
    _spec("evaluated", "style", "§9.2.1", Evaluated, Status.PAIRED),
    _spec("crossSectionReconstructed", "style", "§9.3.1",
          CrossSectionReconstructed, Status.PAIRED),
    _spec("realization", "style", "§9.2.2", Realization, Status.WRITE_ONLY,
          "modelled and writable; kika/gnds/styles.py:58 reports it on read "
          "rather than guessing, because no distributed neutron evaluation "
          "carries one and reading it would be untested. Not in RS_StylesType"),
    _spec("angularDistributionReconstructed", "style", "§9.3.2",
          AngularDistributionReconstructed, Status.WRITE_ONLY,
          "as `realization`. Not in RS_StylesType (gnds.xsd:87-93)"),
    _spec("CoulombPlusNuclearElasticMuCutoff", "style", "§9.3.3",
          CoulombPlusNuclearElasticMuCutoff, Status.WRITE_ONLY,
          "as `realization`. Not in RS_StylesType (gnds.xsd:87-93)"),
    _spec("heated", "style", "§9.4.1", Heated, Status.WRITE_ONLY,
          "as `realization`, and **built live** by "
          "kika/ace/model_adapter/decode.py:141 — an ACE-decoded suite written "
          "with kika.write emits a style its own reader will not take back. "
          "Legal in RS_StylesType"),
    _spec("averageProductData", "style", "§9.4.2", AverageProductData,
          Status.WRITE_ONLY, "as `realization`. Legal in RS_StylesType (gnds.xsd:87-93)"),
    _spec("MonteCarlo_cdf", "style", "§9.4.3", MonteCarlo_cdf,
          Status.WRITE_ONLY, "as `realization`. Not in RS_StylesType (gnds.xsd:87-93)"),
    _spec("griddedCrossSection", "style", "§9.4.4", GriddedCrossSection,
          Status.WRITE_ONLY,
          "as `heated`, built by kika/ace/model_adapter/decode.py:141, and "
          "**not** in RS_StylesType — so the file kika writes from an ACE "
          "decode is rejected by FUDGE's own schema"),
    _spec("URR_probabilityTables", "style", "§9.4.5", URR_probabilityTables,
          Status.WRITE_ONLY, "as `griddedCrossSection`. Not in RS_StylesType (gnds.xsd:87-93)"),
    _spec("heatedMultiGroup", "style", "§9.4.6", HeatedMultiGroup,
          Status.WRITE_ONLY, "as `realization`. Not in RS_StylesType (gnds.xsd:87-93)"),
    _spec("SnElasticUpScatter", "style", "§9.4.7", SnElasticUpScatter,
          Status.WRITE_ONLY, "as `realization`. Not in RS_StylesType (gnds.xsd:87-93)"),
])


@dataclass(frozen=True)
class Defect:
    """An asymmetry that is wrong, as opposed to one that is deliberate.

    Fixed **by count**, not by list: ``len(KNOWN_DEFECTS)`` is asserted, so the
    table lands green declaring what it found, and each repair lowers the
    number and makes the test the thing that notices.
    """

    what: str
    where: str
    caughtBy: str


#: What the two halves disagree about, where the disagreement is a defect.
#: One-sided entries that are *correct* live in :data:`NODES` with a reason and
#: are not here — the point of the distinction is that this list may only get
#: shorter.
KNOWN_DEFECTS: Tuple[Defect, ...] = (
    Defect(
        what="supportsAngularReconstruction is read and reported and never "
             "written, so a file kika round trips loses FUDGE's hint",
        where="kika/gnds/resonances.py:222 vs encode_resonances.py",
        caughtBy="attribute level, as above",
    ),
)

#: The asymmetries that are of *attribute*, not of node — kept beside the table
#: precisely so the commit message cannot suggest the key-set test found them.
#: It did not and cannot; walking both sides to write the table did. There were
#: three; ``relativisticKinematics`` and ``reducedWidthAmplitudes`` are read now.
ONE_SIDED_ATTRIBUTES = (
    ("RMatrix", "supportsAngularReconstruction", "read, never written"),
)


class UndeclaredNode(LookupError):
    """A side registered for a node :data:`NODES` does not declare.

    Raised at **import time**, which is the whole point: a phase 7b reader for
    an §18 law that nobody added to the table fails ``import kika.gnds`` rather
    than passing every test until someone reads the file.
    """


#: Which side registered for which key. Filled by the decorators at import.
_READERS: Dict[Tuple[str, str], Callable] = {}
_WRITERS: Dict[Tuple[str, str], Callable] = {}


def _declare(registry, side: str, family: str, tags: Tuple[str, ...], function):
    for tag in tags:
        key = (family, tag)
        if key not in NODES:
            raise UndeclaredNode(
                f"{function.__module__}.{getattr(function, '__qualname__', '?')} "
                f"{side} <{tag}> as a {family}, and kika/gnds/nodes.py does not "
                f"declare that node. Add a NodeSpec for it — the table is what "
                f"makes 'the reader knows a node the writer does not' something "
                f"a test can see"
            )
        registry[key] = function
    return function


def reads(family: str, *tags: str):
    """Declare that the decorated function reads *tags* of *family*."""

    def decorate(function):
        return _declare(_READERS, "reads", family, tags, function)

    return decorate


def writes(family: str, *tags: str):
    """Declare that the decorated function writes *tags* of *family*.

    A writer's entry point covers a whole family, so this usually takes several
    tags. What it does **not** do is mirror the ``isinstance`` chain inside —
    see the module docstring.
    """

    def decorate(function):
        return _declare(_WRITERS, "writes", family, tags, function)

    return decorate


def readersOf(family: str) -> Dict[str, Callable]:
    """``{tag: reader}`` for one family — what the old dispatch dicts were.

    Derived rather than duplicated: ``primitives.FUNCTION_1D`` and
    ``covariances.COVARIANCE_FORMS`` are filled from this, so a reader that
    exists is in the table by construction and a table entry claiming a reader
    that does not exist is what :func:`check` reports.
    """
    return {tag: function for (fam, tag), function in _READERS.items()
            if fam == family}


def check() -> Tuple[str, ...]:
    """Every entry whose declared status disagrees with what registered.

    Empty is the invariant. This is the half that can fail: the decorators stop
    an *undeclared* node at import, and this stops a declaration that no longer
    describes the code — a ``PAIRED`` entry whose writer was deleted, a
    ``READ_ONLY`` entry that grew one, a one-sided entry with no reason given.

    **It imports both halves itself**, and that is not tidiness. The registries
    fill as a side effect of importing the modules that decorate into them, so
    a caller who imported only this module would find every entry
    ``reader=False writer=False`` — and a test written the other way round,
    "assert no entry is registered on both sides", would pass in an empty room.
    """
    import kika.gnds.covariances  # noqa: F401  - registers the covariance forms
    import kika.gnds.decode       # noqa: F401  - registers the reader's side
    import kika.gnds.encode       # noqa: F401  - registers the writer's side
    import kika.gnds.primitives   # noqa: F401  - registers the functionals
    import kika.gnds.styles       # noqa: F401  - registers §9

    problems = []
    for key, spec in NODES.items():
        hasReader = key in _READERS
        hasWriter = key in _WRITERS
        expected = {
            Status.PAIRED: (True, True),
            Status.READ_ONLY: (True, False),
            Status.WRITE_ONLY: (False, True),
            Status.NEITHER: (False, False),
        }[spec.status]
        if (hasReader, hasWriter) != expected:
            problems.append(
                f"{spec.family}/<{spec.tag}> is declared {spec.status.value} "
                f"and registered reader={hasReader} writer={hasWriter}"
            )
        if spec.status is not Status.PAIRED and not spec.reason:
            problems.append(
                f"{spec.family}/<{spec.tag}> is one-sided and gives no reason; "
                f"cite a § or a file:line, or the entry weakens the test it is in"
            )
    return tuple(problems)
