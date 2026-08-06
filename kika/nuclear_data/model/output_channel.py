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

__all__ = ["Q", "Product", "Products", "Multiplicity", "OutputChannel"]


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
class OutputChannel:
    """§17.1.1. The result of a reaction: a Q value and a list of products."""

    genre: Optional[str] = None       # 'twoBody' or 'NBody'
    process: Optional[str] = None
    Q: Q = field(default_factory=Q)
    products: Products = field(default_factory=Products)
    fissionFragmentData: Optional[object] = None

    @property
    def isTwoBody(self) -> bool:
        return self.genre == "twoBody"
