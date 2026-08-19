"""GNDS-2.1 §18 ``distribution`` — every law the library uses, implemented.

The roadmap's rule was **empty slots exist, they are not absent**: each law was
declared before it was filled, so a reader meeting one could say which GNDS node
it could not handle instead of failing somewhere unrelated, and so that adding
the implementation later restructured nothing. Phase 7b filled them, and the
rule is what made that a series of small commits rather than one large one.

**All seven §18.1.1 members the library contains are here** —
``angularTwoBody``, ``unspecified``, ``uncorrelated``, ``energyAngular``,
``angularEnergy``, ``KalbachMann`` and ``branching3d``. The other five members
of that choice occur **zero times** across the 558 distributed neutron
evaluations, so they are not modelled and ``kika/gnds/nodes.py`` says so with
the count.

What remains declared-and-empty is :class:`Recoil`, which is not a
``<distribution>`` form at all and whose docstring explains where it really
lives.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from .axes import Axes
from .enums import Frame
from .functions import Function1d, Function2d, Regions2d, XYs2d, XYs3d
from .quantities import PhysicalQuantity

__all__ = [
    "Distribution", "AngularTwoBody", "Isotropic2d", "Unspecified",
    "Uncorrelated", "DiscreteGamma", "PrimaryGamma", "NBodyPhaseSpace",
    "EnergyAngular", "AngularEnergy", "KalbachMann", "Branching3d", "Recoil",
    "NOT_IMPLEMENTED_DISTRIBUTIONS",
]


@dataclass
class AngularTwoBody:
    """§18. P(mu|E) for two-body kinematics — what ENDF MF4 carries.

    ``angular`` is a two-dimensional form: an :class:`XYs2d` whose children are
    ``Legendre`` (a coefficient representation) or ``XYs1d`` (a tabulated one),
    or a :class:`Regions2d` when the file uses more than one region — including
    ENDF's LTT=3, where a Legendre region and a tabulated region meet.

    **It was a dict, keyed on energy, and that was wrong.** The committed Fe-56
    slice repeats an incident energy inside one Legendre grid (a discontinuity)
    and repeats the LTT=3 boundary energy across the two representations. A dict
    drops one entry in each case, silently, and the file that comes back out is
    a valid ENDF file missing two angular distributions. The 2-d forms carry an
    ordered list precisely so that cannot happen; see
    ``kika/nuclear_data/model/functions/higher.py``.

    **A tabulated function is only one of the four shapes §18 allows here.**
    ``angularTwoBody`` admits ``XYs2d``, ``regions2d``, ``isotropic2d`` and
    ``recoil``, and across ENDF/B-VIII.1-GNDS's 45 080 of them the split is
    21 649 / 111 / 780 / 22 540 — so the *commonest* shape carries no numbers at
    all. ``isotropic2d`` goes in ``angular`` because it is a statement about
    P(mu|E) like the other two, just one that needs no table; putting it
    elsewhere would make "does this product have an angular distribution?" a
    two-part question.

    ``recoilHref`` is the fourth, and it is a link rather than a function: in
    two-body kinematics the residual's distribution is the ejectile's mirrored,
    so the evaluation states it once and points at it. It gets a field of its
    own rather than a fabricated table — the treatment
    :class:`~kika.nuclear_data.model.cross_section_forms.Reference` gets for the
    same reason. ``angular`` and ``recoilHref`` are alternatives; a node carries
    one of them.
    """

    angular: Optional[Union[XYs2d, Regions2d, "Isotropic2d"]] = None
    productFrame: Frame = Frame.centerOfMass
    label: Optional[str] = None
    recoilHref: Optional[str] = None

    def __post_init__(self) -> None:
        self.productFrame = Frame(self.productFrame)

    @property
    def isRecoil(self) -> bool:
        """This product's P(mu|E) is another product's, mirrored."""
        return self.recoilHref is not None

    @property
    def energies(self) -> List[float]:
        """The incident energies, in file order, **duplicates included**.

        Empty for the two shapes that tabulate nothing — a ``recoil`` link and
        an ``isotropic2d`` — which is why it is not a proxy for "has data".
        """
        if not isinstance(self.angular, (XYs2d, Regions2d)):
            return []
        if isinstance(self.angular, Regions2d):
            return [f.outerDomainValue for f in self.angular.function1ds]
        return list(self.angular.outerDomainValues)

    @property
    def function1ds(self) -> List[Function1d]:
        """Every per-energy angular function, in file order."""
        if not isinstance(self.angular, (XYs2d, Regions2d)):
            return []
        if isinstance(self.angular, Regions2d):
            return self.angular.function1ds
        return list(self.angular.function1ds)


@dataclass
class Isotropic2d:
    """§18. P(mu|E) is flat in mu at every energy — ENDF's LI=1.

    Distinct from :class:`Unspecified`: this *is* the distribution, stated
    positively, and it is not an absence to be filled in later.
    """

    label: Optional[str] = None
    productFrame: Frame = Frame.centerOfMass

    def __post_init__(self) -> None:
        self.productFrame = Frame(self.productFrame)


@dataclass
class Unspecified:
    """§18. The distribution is deliberately not given."""

    label: Optional[str] = None
    productFrame: Frame = Frame.lab

    def __post_init__(self) -> None:
        # Coerced for the reason `ReactionSuite.__post_init__` coerces
        # `projectileFrame`: `Frame` subclasses `str`, so a raw string compares
        # equal and everything appears to work until a writer asks for `.value`
        # and gets an AttributeError three phases later.
        self.productFrame = Frame(self.productFrame)


class _UnimplementedDistribution:
    gndsNodeName = "?"
    #: Why this one is empty. A phase number was the *usual* answer and never
    #: the only one — ``angularEnergy``'s emptiness was a decision, and the
    #: census settled it — so the sentence belongs to the class rather than
    #: being spelled once for everybody. The 2-d/3-d functionals learned the
    #: same lesson; see
    #: :attr:`~kika.nuclear_data.model.functions.higher._UnimplementedNode.plannedFor`.
    #:
    #: The default is no longer a phase, because :class:`Recoil` is the only
    #: subclass left and it is not waiting for one — it is a node that belongs
    #: somewhere else entirely.
    plannedFor = "no phase scheduled"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            f"GNDS distribution {self.gndsNodeName!r} is declared but not "
            f"implemented ({self.plannedFor}). It is present rather than absent "
            f"so that a reader meeting one is told what is missing."
        )


@dataclass
class DiscreteGamma:
    """§18.3. A gamma line at one fixed energy — ``gnds.xsd:1777``.

    Not a functional: there is no table, no interpolation and nothing to
    evaluate between points. The whole content is the line's energy and the
    incident-energy range over which the reaction emits it, which is why this
    is three numbers and an ``axes`` rather than an :class:`XYs2d` of one point.

    ``axes`` is a **required** child of the node (the schema says so, and the
    three distributed fixtures all carry ``energy_in`` / ``energy_out`` /
    ``P(energy_out|energy_in)``), so it is kept and written back rather than
    reconstructed from a convention.
    """

    value: float
    domainMin: float
    domainMax: float
    axes: Optional[Axes] = None


@dataclass
class PrimaryGamma:
    """§18.3. A gamma whose energy tracks the incident one — ``gnds.xsd:1786``.

    The same four members as :class:`DiscreteGamma` plus the residual level the
    transition lands on, and **not a subclass of it**: ``value`` means a
    different thing here — the binding-energy term of E_gamma = value +
    (A/(A+1))·E, not an emitted energy — and the writer dispatches on
    ``isinstance``, where a subclass would let one be written as the other.
    """

    value: float
    domainMin: float
    domainMax: float
    axes: Optional[Axes] = None
    finalState: Optional[str] = None


@dataclass
class NBodyPhaseSpace:
    """§18.3's *energy* form for an N-body break-up — ``gnds.xsd:1703``.

    **Not a `<distribution>` form**, which is the mistake worth not making: it
    is one of the eleven choices inside ``uncorrelated/energy``, so it arrives
    with :class:`Uncorrelated` and never on its own. See
    :data:`kika.gnds.nodes.NOT_A_DISTRIBUTION_FORM`.

    ``mass`` is the total mass of the N products, and it comes with its unit
    stated (``amu`` in the library) — a :class:`PhysicalQuantity` rather than a
    bare float, for the reason §4.1's scattering radius made expensive: a
    number whose unit lives in a convention instead of beside it is a factor of
    ten waiting to happen.
    """

    numberOfProducts: int
    mass: Optional[PhysicalQuantity] = None


@dataclass
class Uncorrelated:
    """§18.3. P(mu|E) and P(E'|E) stated independently — ``gnds.xsd:1676``.

    The commonest distribution in the library by a wide margin: 126 501 of the
    223 916 forms in ENDF/B-VIII.1-GNDS. On the ENDF side it is MF4 and MF5
    read together, which is why the two halves are separate objects here rather
    than one surface — they come from different sections and either can be the
    one kika cannot read.

    **Both halves are required by the schema** (the node is an ``xs:sequence``,
    not a choice), so a half-filled one is invalid rather than merely partial.
    The fields still default to ``None`` because the reader fills what it can
    and reports the rest; refusing to write a half node is the *writer's*
    judgement, made in one place, and it is the same judgement as the empty
    ``<distribution/>``.
    """

    angular: Optional[Union[XYs2d, Isotropic2d]] = None
    energy: Optional[Union[XYs2d, Regions2d, DiscreteGamma, PrimaryGamma,
                           NBodyPhaseSpace]] = None
    label: Optional[str] = None
    productFrame: Frame = Frame.lab

    def __post_init__(self) -> None:
        self.productFrame = Frame(self.productFrame)

    @property
    def isComplete(self) -> bool:
        """Both halves present, which is the only shape the schema admits."""
        return self.angular is not None and self.energy is not None


@dataclass
class EnergyAngular:
    """§18.4. P(E′,mu|E) as one node — ``gnds.xsd:1797``, ``DistributionAEType``.

    The correlated counterpart of :class:`Uncorrelated`: where that one states
    P(mu|E) and P(E′|E) separately because the evaluation did, this one states
    the joint distribution, which is what ENDF MF6 carries.

    **The field is named after its type because the schema admits exactly one.**
    ``DistributionAEType`` is an ``xs:sequence`` of a single ``XYs3d`` — no
    choice, no ``regions3d`` (which cannot occur anywhere; see
    :class:`~kika.nuclear_data.model.functions.higher.Regions3d`), no isotropic
    shorthand. ``AngularTwoBody.angular`` is spelled neutrally because four
    shapes can land there; here a neutral name would suggest an abstraction that
    does not exist.

    **Why this is not :class:`AngularEnergy` with a flag**, though the two share
    a complexType exactly. The nesting order is the physics: ``energyAngular``
    is P(E′|E) outermost with P(mu|E,E′) inside it, and ``angularEnergy`` is the
    same two variables the other way round. Writing one as the other produces a
    file that **validates and states the wrong physics** — no schema can catch
    it, and the axes labels are the only thing that would disagree. The writer
    dispatches on ``isinstance``, so two classes make that failure impossible
    rather than merely unlikely. A shared class with a boolean makes it one
    wrong default away.
    """

    xys3d: Optional[XYs3d] = None
    label: Optional[str] = None
    productFrame: Frame = Frame.lab

    def __post_init__(self) -> None:
        self.productFrame = Frame(self.productFrame)

    @property
    def isComplete(self) -> bool:
        """The one child present, which is the only shape the schema admits."""
        return self.xys3d is not None


@dataclass
class AngularEnergy:
    """§18.5. P(mu,E′|E) — ``gnds.xsd:1797``, the *same* ``DistributionAEType``.

    :class:`EnergyAngular`'s mirror, and everything that class's docstring says
    about why the two are separate classes applies here from the other side. The
    complexType is shared exactly; the element name is the **only** thing in the
    file that says which variable is outermost, so a reader that decoded one
    into the other would produce a model that is wrong in a way no schema can
    see. Hence two dataclasses and an ``isinstance`` dispatch, not one class
    with a flag.

    **It exists because the census counted, and the count was not zero.** It was
    left unwritten while the answer was unknown — the standing recommendation
    was to retire the declaration if it turned out to occur nowhere, as
    ``forward``, ``regions3d``, ``Ys1d``, ``gridded1d``, ``gridded3d``, ``Watt``
    and ``MadlandNix`` all do. It occurs **twice**, both in
    ``n-004_Be_009.endf.gnds.xml``, which is the entire population across the
    558 distributed neutron evaluations. Two is not zero: the sentence "no
    distributed evaluation carries one" would have been **false**, and with a
    witness in hand and the complexType already implemented for its mirror,
    writing it costs less than documenting why it is absent.
    """

    xys3d: Optional[XYs3d] = None
    label: Optional[str] = None
    productFrame: Frame = Frame.lab

    def __post_init__(self) -> None:
        self.productFrame = Frame(self.productFrame)

    @property
    def isComplete(self) -> bool:
        """The one child present, which is the only shape the schema admits."""
        return self.xys3d is not None


@dataclass
class KalbachMann:
    """§18.6. The Kalbach-Mann systematics — ``gnds.xsd:1805-1814``.

    A pre-equilibrium emission spectrum stated as **shape and asymmetry rather
    than as a table of P(mu,E′|E)**: ``f`` is the energy spectrum and ``r`` the
    pre-equilibrium fraction, and the angular distribution is reconstructed from
    them by the systematics. That is why this is a law of its own and not an
    ``energyAngular`` — the file gives the parameters, not the joint function.

    **An ``xs:sequence``, so the order is the schema's and not a preference**:
    ``f``, then ``r``, then ``a``. Each of the three is an ``XYs2dWrapperType``
    (``:2204-2208``) — a bare wrapper element with **no attributes of its own**
    holding exactly one primary ``XYs2d``, which carries its own ``axes``.

    **``f`` and ``r`` are required and ``a`` is not**, which is the schema's
    statement (``minOccurs="0"`` on ``a`` alone) and also the library's: the
    census counted **3 730 KalbachMann nodes across 272 evaluations and every
    one of them is ``f`` + ``r``**. ``a`` appears zero times. It is modelled
    because the schema admits it and the data could exist tomorrow, and the
    writer emits it only when it is there — a *reading* branch for a node nobody
    has ever seen is the ``Ys1d`` case, and this stops short of it.
    """

    f: Optional[XYs2d] = None
    r: Optional[XYs2d] = None
    #: Zero occurrences in 3 730. Present so a file that has one round trips.
    a: Optional[XYs2d] = None
    label: Optional[str] = None
    productFrame: Frame = Frame.centerOfMass

    def __post_init__(self) -> None:
        self.productFrame = Frame(self.productFrame)

    @property
    def isComplete(self) -> bool:
        """``f`` and ``r``, which is what ``:1808-1809`` makes mandatory.

        ``a`` is deliberately not in this test: a node without it is the
        ordinary case and the only one the distribution contains.
        """
        return self.f is not None and self.r is not None


@dataclass
class Branching3d:
    """§18.1.1. ``gnds.xsd:1816-1819`` — the distribution half of a branching.

    ``DistributionBranching3dType`` is two attributes and no content: the node
    states that this photon's angle-energy distribution follows from the
    isomeric transition rather than from a tabulated law. Like its multiplicity
    half it is **format fidelity and not evaluation** — kika reads and writes it
    back and does not resolve the transition against PoPs ``decayData``, which
    is a §12 walk the model does not do.

    See :class:`~kika.nuclear_data.model.output_channel.Branching1d`, which
    accompanies every one of these: 14 032 of each across the distribution, in
    the same 282 files, on the same products.
    """

    label: str
    productFrame: Frame = Frame.lab

    def __post_init__(self) -> None:
        # Coerced for the reason `Unspecified.__post_init__` gives: `Frame`
        # subclasses `str`, so a raw string compares equal and everything
        # appears to work until a writer asks for `.value`.
        self.productFrame = Frame(self.productFrame)


class Recoil(_UnimplementedDistribution):
    """**Not a `<distribution>` form either, and not unimplemented.**
    ``gnds.xsd:1647-1662`` does not admit ``recoil`` there; it occurs only
    inside ``angularTwoBody`` (``gnds.xsd:1670``) as a bare
    ``<recoil href=…/>``, where it is the library's commonest shape and kika
    both reads and writes it — onto :attr:`AngularTwoBody.recoilHref`, which is
    where a link belongs. The class is kept because it is exported, and named
    here so that nobody restores it to the §18.1.1 table on the strength of
    this list."""

    gndsNodeName = "recoil"


#: What a reader meeting one of these can be told is missing. **The one name
#: left is not a member of §18.1.1's choice at all** — see :class:`Recoil` above
#: — which is why :data:`kika.gnds.nodes.NOT_A_DISTRIBUTION_FORM` names its real
#: choice point and a test holds the two lists against each other. It was three
#: until §18.5 and §18.6 landed and :class:`AngularEnergy` and
#: :class:`KalbachMann` became dataclasses.
#:
#: **It is down to a dict that no longer contains an unimplemented §18 law**,
#: which is what phase 7b set out to do. The entry that remains is a misplaced
#: name and not a gap; the test above is what keeps the distinction.
NOT_IMPLEMENTED_DISTRIBUTIONS = {
    cls.gndsNodeName: cls
    for cls in (Recoil,)
}


@dataclass
class Distribution:
    """§18.1.1. A product's distribution, in as many forms as the file carries."""

    forms: Dict[str, object] = field(default_factory=dict)

    def __getitem__(self, label: str):
        return self.forms[label]

    def __setitem__(self, label: str, form: object) -> None:
        self.forms[label] = form

    def __len__(self) -> int:
        return len(self.forms)

    def __bool__(self) -> bool:
        # A declared slot is *present* even when empty; only `len()` speaks to
        # content. Without this, `if reaction.crossSection:` would read a
        # reaction whose forms have not been decoded yet as one that has no
        # cross section at all.
        return True

    def __contains__(self, label: str) -> bool:
        return label in self.forms
