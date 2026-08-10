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
from typing import List, Optional

from .functions import Function1d
from .quantities import PhysicalQuantity

__all__ = ["Q", "Product", "Products", "Multiplicity", "OutputChannel",
           "DelayedNeutron", "DelayedNeutrons", "FissionFragmentData"]


@dataclass
class Q:
    """§17.1.1. The reaction Q value.

    ``None`` means *not known*, which is a different statement from ``0.0`` and
    the distinction the flat classes could not make.
    """

    value: Optional[float] = None
    unit: str = "eV"
    label: Optional[str] = None

    @property
    def isKnown(self) -> bool:
        return self.value is not None


@dataclass
class Multiplicity:
    """§17.3. How many of a product come out, as a constant or a function of E."""

    constant: Optional[float] = None
    function: Optional[Function1d] = None
    label: Optional[str] = None
    #: Not a GNDS node. ENDF states a multiplicity in a section of its own
    #: (MF1/452, /455, /456) with its own ZA/AWR/MAT header and its own LNU,
    #: and none of the four has a GNDS counterpart -- the same argument
    #: `Product.provenance` makes for MF4's LTT/LI/LCT/NM. Kept here rather
    #: than on the node the multiplicity hangs from, because that node is a
    #: `product` for the prompt nu-bar and a `multiplicitySum` for the total.
    provenance: Optional[object] = None

    def evaluate(self, energy):
        if self.function is not None:
            return self.function.evaluate(energy)
        if self.constant is not None:
            return self.constant
        raise ValueError("this multiplicity has neither a constant nor a function")


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
        """The channel's product with this ``pid``, created if it has none.

        A Python verb on a GNDS noun, and it exists because §17.2.1 puts a
        product's multiplicity and its distribution on **one** node while ENDF
        states them in different files. The ENDF decoder reads MF1's nu-bar and
        MF4's angular distribution in separate passes; each used to append a
        product of its own, so a fissile tape carrying both came out with two
        neutrons on the fission channel — one holding the multiplicity, one
        holding the distribution, and every consumer of ``byPid('n')[0]``
        getting whichever pass ran first.
        """
        existing = self.products.byPid(pid)
        if existing:
            return existing[0]
        product = Product(pid=pid, label=label if label is not None else pid)
        self.products.products.append(product)
        return product
