"""GNDS-2.1 §17: ``outputChannel``, ``product`` and the Q value.

§17.1.1: *"A reaction is composed of a crossSection and an output channel which
includes the reaction Q-value and a list of products."* Products may themselves
break up or decay, and the result is another ``outputChannel`` — which is why
``Product.outputChannel`` exists and the structure is recursive.

**Where the Q = 0 defect lands.** ``kika/processing/reconstruct.py:269``
hardcodes ``qm = qi = 0.0`` on every reconstructed section, and the flat
``CrossSection`` carries Q in an untyped ``metadata`` dict that ACE cannot fill.
Here Q is a node of the output channel, where it belongs, and its absence is
representable as ``None`` rather than as a silent zero. Deciding *whose* Q a
reconstructed MT102 inherits is a physics question held for P6.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional

from .component import EVAL_LABEL, Component
from .functions import Function1d
from .quantities import PhysicalQuantity

__all__ = ["Q", "Product", "Products", "Multiplicity", "Branching1d",
           "UnspecifiedMultiplicity", "OutputChannel",
           "DelayedNeutron", "DelayedNeutrons", "FissionFragmentData"]


@dataclass
class Branching1d:
    """§17.3. ``gnds.xsd:1638-1640`` — and **a branching is not a number**.

    ``Branching1dMultiplicityType`` has one attribute and no content at all. The
    node says *"this photon's multiplicity is the isomeric branching ratio
    recorded in PoPs ``decayData`` for the level this channel decays from"*, and
    resolving it means walking §12's decay chains. kika does not do that and
    this class does not pretend to: it exists so the node reads and writes back
    identically, and so :attr:`Multiplicity.isEvaluable` can say ``False`` out
    loud instead of the multiplicity merely looking empty.

    It always arrives with a
    :class:`~kika.nuclear_data.model.distributions.Branching3d` on the same
    product's ``distribution`` — the two counts across the library are equal at
    **14 032**, in the same 282 files, because they are the two halves of one
    statement about one product. They are declared in separate modules all the
    same, because §17.3 and §18.1.1 are separate choice points with separate
    complexTypes and this package is laid out by chapter.
    """

    label: str


@dataclass
class UnspecifiedMultiplicity:
    """§17.3. ``gnds.xsd:1642-1644`` — the evaluator declining to say how many.

    **Not** :class:`~kika.nuclear_data.model.distributions.Unspecified`, though
    both nodes are spelled ``unspecified``. ``UnspecifiedMultiplicityType``
    carries a label and nothing else; ``DistributionUnspecifiedType``
    (``gnds.xsd:1841``) carries a label **and a required ``productFrame``**.
    Sharing the class would hand every multiplicity a frame the node cannot
    hold, and the writer would then have to choose between emitting an attribute
    the schema rejects and dropping one the model claims to have.

    The two say different things anyway: ``unspecified`` under a
    ``distribution`` is the evaluator declining to give P(E′,mu\\|E) *in a stated
    frame*; here it is declining to say how many come out, and there is no frame
    in which that could be said.

    The name breaks the package's class-is-the-node-name-capitalised rule, and
    has to: two nodes in two chapters share a spelling. ``kika/gnds/nodes.py``
    already names this exact collision as one of the reasons its key is
    ``(family, tag)``, so the registry carries both without ambiguity.
    """

    label: str


@dataclass
class Q:
    """§17.1.1. The reaction Q value.

    ``None`` means *not known*, which is a different statement from ``0.0`` and
    the distinction the flat classes could not make.

    **The domain is the file's and not the evaluation's.** §17.1.1 writes a Q as
    a ``constant1d``, and ``xData_constant1d`` (``gnds.xsd:2099``) makes
    ``domainMin`` and ``domainMax`` required — for a threshold reaction they are
    the threshold, not the evaluation's full range. Until these two fields
    existed the writer filled them from the ``evaluated`` style's
    ``projectileEnergyDomain``, which turned H-2's (n,2n) Q of −2 225 002 eV
    over ``3.339e6 … 1.5e8`` into the same Q over ``1e-5 … 1.5e8`` — a statement
    the evaluation does not make, and one the read → write fixed point cannot
    see because both writes agree on it.

    They stay ``Optional``: a Q assembled from ENDF has no GNDS domain to carry,
    and inventing one is what this exists to stop. The writer falls back to the
    style's domain only when they are absent, and says so.
    """

    value: Optional[float] = None
    unit: str = "eV"
    label: Optional[str] = None
    #: The ``constant1d``'s own domain, when the Q was read from GNDS.
    domainMin: Optional[float] = None
    domainMax: Optional[float] = None

    @property
    def isKnown(self) -> bool:
        return self.value is not None


@dataclass(init=False, repr=False)
class Multiplicity(Component):
    """§17.3. How many of a product come out.

    **§17.3 is an ``xs:choice``, so this holds one form and not a field per
    member.** ``gnds.xsd:1626-1636`` gives a multiplicity seven alternatives,
    and a class with one optional slot per alternative can represent states GNDS
    cannot — a constant *and* a branching, say. One field says what the schema
    says: exactly one of them, or none because the node was empty.

    **It is a :class:`~kika.nuclear_data.model.component.Component`, and it
    became one on 2026-09-06 because the premise that said it should not
    stopped holding.** That premise was a census and it still stands as a
    census: across the 558 distributed neutron evaluations all 230 562
    ``<multiplicity>`` nodes carry exactly one form and all are labelled
    ``eval``. What changed is that the question is no longer "what do
    distributed libraries contain" but "what does kika write": a perturbation
    run puts a realisation of the nu-bar beside its evaluation, under a
    ``realization-0007`` label, exactly as it does for ``crossSection`` and
    ``distribution``. The old docstring named the trigger and the consequence --
    *"if a library ever ships a second-labelled multiplicity, ``form`` becomes
    ``forms['eval']`` and nothing else moves"* -- and that is what happened,
    with kika as the library.

    The alternative, kept for one afternoon and then dropped, was for the
    realisation to **replace** the form and be put back afterwards. It works and
    it is worse in a way that shows up in the file rather than in the code: a
    GNDS document written from a suite in that state carries the perturbed
    nu-bar *instead of* the evaluated one, while its cross sections and
    distributions carry both. §9.3's ``realization`` style exists to say "this
    is a draw from that evaluation", and it cannot say it about a form that has
    displaced the evaluation it was drawn from.

    **``form`` still works and still means the evaluated form.** It is a property
    over ``forms`` now, so every reader that asks a multiplicity for "the" form
    keeps getting the one it always got, and only code that means a *specific*
    label has to say so.

    **The form is an object and not a number, and that is what makes the round
    trip faithful.** ``xData_constant1d`` (``gnds.xsd:2099``) makes
    ``domainMin`` and ``domainMax`` required; a bare float could not hold them,
    so the writer took them from the ``evaluated`` style's
    ``projectileEnergyDomain`` instead. That silently widened every threshold
    multiplicity — ``n-001_H_002.endf.gnds.xml:757`` states ``value="2"`` over
    ``3.339e6 … 1.5e8`` and came back over ``1e-5 … 1.5e8``, which the read →
    write fixed point cannot see because both writes agree. A
    :class:`~kika.nuclear_data.model.functions.simple.Constant1d` carries its
    own domain.
    """

    #: Not a GNDS node. ENDF states a multiplicity in a section of its own
    #: (MF1/452, /455, /456) with its own ZA/AWR/MAT header and its own LNU,
    #: and none of the four has a GNDS counterpart -- the same argument
    #: `Product.provenance` makes for MF4's LTT/LI/LCT/NM. Kept here rather
    #: than on the node the multiplicity hangs from, because that node is a
    #: `product` for the prompt nu-bar and a `multiplicitySum` for the total.
    provenance: Optional[object] = None

    gndsNodeName: ClassVar[str] = "multiplicity"

    def __init__(self, form: Optional[object] = None,
                 forms: Optional[Dict[str, Any]] = None,
                 provenance: Optional[object] = None) -> None:
        """``Multiplicity(form=X)`` still builds the one-form node it always did.

        The form is filed under its **own** label when it carries one -- §17.3
        puts the label on the form and not on the node -- and under ``eval``
        when it does not, which is what every decoder produced before this was a
        mapping.
        """
        self.forms = dict(forms or {})
        if form is not None:
            self.forms[getattr(form, "label", None) or EVAL_LABEL] = form
        self.provenance = provenance

    @property
    def form(self) -> Optional[object]:
        """The evaluated form -- "the" multiplicity, for a reader that wants one.

        Falls back to the single form when there is exactly one and it is not
        labelled ``eval``: a GNDS document may label its only multiplicity
        anything, and before this class was a mapping such a node still answered
        ``.form``. Returns ``None`` for an empty node, as it always did.

        With several forms and none labelled ``eval`` there is no defensible
        answer, and it raises rather than picking one: that state can only be
        reached by putting a realisation on a node whose evaluation was never
        read, and guessing there writes an arbitrary draw into a file as if it
        were the evaluation.
        """
        if EVAL_LABEL in self.forms:
            return self.forms[EVAL_LABEL]
        if len(self.forms) == 1:
            return next(iter(self.forms.values()))
        if not self.forms:
            return None
        raise KeyError(
            f"this multiplicity carries {sorted(self.forms)} and none of them "
            f"is {EVAL_LABEL!r}, so 'the' form is not a question with one "
            f"answer; ask for the label you mean"
        )

    @form.setter
    def form(self, value: Optional[object]) -> None:
        """Replace the evaluated form. Kept so that code that built a node by
        assignment keeps working; a *realisation* is ``multiplicity[label] = x``
        and does not come through here."""
        if value is None:
            self.forms.pop(EVAL_LABEL, None)
            return
        self.forms[getattr(value, "label", None) or EVAL_LABEL] = value

    @property
    def label(self) -> Optional[str]:
        """The form's label, because §17.3 puts it there and not on the node.

        ``MultiplicityType`` has no attributes at all: every one of its seven
        members carries its own ``label``. Reading it off the form is what stops
        the two spellings the writer used to have — ``multiplicity.label`` on
        the ``constant1d`` path and ``form.label`` on the functional one — from
        disagreeing, which nothing would have caught because the census says
        both always hold ``eval``.
        """
        return getattr(self.form, "label", None)

    @property
    def isEvaluable(self) -> bool:
        """This multiplicity is a number kika can compute at an energy.

        Until phase 7b a multiplicity held a number or a curve, so "no constant
        and no function" could be read as "kika did not decode this". §17.3 has
        three members that carry no numbers **by design** — a ``reference`` to
        another product's multiplicity (3 539 in the library), an isomeric
        ``branching1d`` (14 032) and an ``unspecified`` (178) — and once those
        are represented rather than dropped, an empty multiplicity and a
        deliberately number-free one look alike to ``form is None``. This is the
        question worth asking before evaluating.
        """
        return isinstance(self.form, Function1d)

    def evaluate(self, energy):
        """The multiplicity at an energy, or a refusal that names the form.

        **Below a threshold this now returns 0, where it used to return the
        constant.** A :class:`Constant1d` evaluates to zero outside its own
        domain (``outOfRange="zero"``), and the domain is the file's: H-2's
        (n,2n) multiplicity of 2 is 2 above 3.339 MeV and 0 below it. It used to
        answer 2 at 1 eV because the model held a bare float with no domain to
        compare against.
        """
        if self.isEvaluable:
            return self.form.evaluate(energy)
        if self.form is None:
            raise ValueError("this multiplicity is empty; the <multiplicity> "
                             "node held no form kika could read")
        raise ValueError(
            f"this multiplicity is a {type(self.form).__name__}, which states "
            f"no number: §17.3's reference, branching1d and unspecified are the "
            f"evaluator declining to give one, not kika failing to read it"
        )


@dataclass
class Product:
    """§17.2.1. One outgoing particle, its multiplicity and its distribution."""

    pid: str
    label: Optional[str] = None
    multiplicity: Optional[Multiplicity] = None
    distribution: Optional[object] = None
    outputChannel: Optional["OutputChannel"] = None  # breakup or decay
    #: Not a GNDS node. A distribution decoded from ENDF MF4 needs LTT, LI, LCT
    #: and NM to be written back, and those belong to the MF4 section rather
    #: than to the reaction — MF3 and MF4 carry separate headers.
    provenance: Optional[object] = None


@dataclass
class Products:
    """The ordered list of a channel's products."""

    products: List[Product] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.products)

    def __iter__(self):
        return iter(self.products)

    def __getitem__(self, index):
        return self.products[index]

    def byPid(self, pid: str) -> List[Product]:
        return [p for p in self.products if p.pid == pid]


@dataclass
class DelayedNeutron:
    """§18.4. One delayed-neutron precursor family: its decay rate and its product.

    ``rate`` is the precursor's decay constant lambda, and it is a
    :class:`~kika.nuclear_data.model.quantities.PhysicalQuantity` rather than a
    bare float because ENDF writes it in 1/s while a library that stores half
    lives would write it in s — the unit is the difference between the two and
    belongs with the number.

    **``product.multiplicity`` is routinely empty, and that is not a defect.**
    ENDF MF1/455 gives the *aggregate* delayed nu-bar and the NNF decay
    constants; the per-family split lives in the MF5/455 subsection weights.
    Until MF5 is decoded (phase 7b) the families exist with their rates and no
    multiplicity, and the aggregate is held by the ``multiplicitySum`` that
    §21 provides for exactly this. An invented equal split would look like data.
    """

    label: str
    rate: Optional[PhysicalQuantity] = None
    product: Optional[Product] = None


@dataclass
class DelayedNeutrons:
    """§18.4. The precursor families of one fission channel, in file order."""

    delayedNeutrons: List[DelayedNeutron] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.delayedNeutrons)

    def __iter__(self):
        return iter(self.delayedNeutrons)

    def __getitem__(self, index):
        return self.delayedNeutrons[index]

    def __bool__(self) -> bool:
        # Present-and-empty is not absent. Same rule as the reactionSuite's
        # children: see `suite.py`'s docstring.
        return True

    def append(self, delayedNeutron: DelayedNeutron) -> None:
        self.delayedNeutrons.append(delayedNeutron)


@dataclass
class FissionFragmentData:
    """§18.4. What a fission channel carries beyond its products.

    Only ``delayedNeutrons`` is filled from ENDF today. ``fissionEnergyReleases``
    (MF1/458) and ``productYields`` (MF8/454, /459) have slots so that the two
    files which would fill them do not have to restructure this node when they
    land; both are absent from every decoder for now.
    """

    delayedNeutrons: DelayedNeutrons = field(default_factory=DelayedNeutrons)
    fissionEnergyReleases: List[object] = field(default_factory=list)
    productYields: List[object] = field(default_factory=list)


@dataclass
class OutputChannel:
    """§17.1.1. The result of a reaction: a Q value and a list of products."""

    genre: Optional[str] = None       # 'twoBody' or 'NBody'
    process: Optional[str] = None
    Q: Q = field(default_factory=Q)
    products: Products = field(default_factory=Products)
    #: §18.4, and ``None`` on purpose for every channel that is not fission.
    #: This is the one child that is *absent* rather than present-and-empty:
    #: a scattering channel with an empty ``fissionFragmentData`` would assert
    #: that kika looked for delayed neutrons in it, which is meaningless.
    fissionFragmentData: Optional[FissionFragmentData] = None

    @property
    def isTwoBody(self) -> bool:
        return self.genre == "twoBody"

    def ensureProduct(self, pid: str, label: Optional[str] = None) -> Product:
        """The channel's product with this ``pid`` and ``label``, created if absent.

        A Python verb on a GNDS noun, and it exists because §17.2.1 puts a
        product's multiplicity and its distribution on **one** node while ENDF
        states them in different files. The ENDF decoder reads MF1's nu-bar and
        MF4's angular distribution in separate passes; each used to append a
        product of its own, so a fissile tape carrying both came out with two
        neutrons on the fission channel -- one holding the multiplicity, one
        holding the distribution, and every consumer of ``byPid('n')[0]``
        getting whichever pass ran first.

        **``label`` is part of the question and not only of the answer**, which
        it was not until MF6 arrived. One channel may legitimately hold two
        products of the same particle -- ENDF/B-VIII.0's alpha+He4 MT2 lists the
        elastically scattered He4 and its recoil, and t+Li7's MT24 emits two of
        them -- and §17.2.1 tells those apart by ``label``, which is why
        :meth:`Products.byPid` answers with a list. Matching on ``pid`` alone
        returned the first and quietly ignored the argument, so the second
        subsection decorated the first product instead of getting its own.

        Called with no ``label`` the two questions coincide, which is what every
        MF1, MF4 and MF5 caller does.
        """
        wanted = label if label is not None else pid
        for product in self.products:
            if (product.label or product.pid) == wanted:
                return product
        product = Product(pid=pid, label=wanted)
        self.products.products.append(product)
        return product
