"""GNDS-2.1 §18 ``distribution``, with the laws declared and unimplemented.

The roadmap's rule: **empty slots exist, they are not absent.** Filling these is
phase 7b. Until then each law is declared, so a reader meeting one can say which
GNDS node it cannot handle instead of failing somewhere unrelated, and so that
adding the implementation later restructures nothing.

``angularTwoBody`` is the exception: MF4 is inside kika's current parser
coverage, so phase 3c fills it and it is a real container from the start.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .enums import Frame
from .functions import Function1d

__all__ = [
    "Distribution", "AngularTwoBody", "Unspecified",
    "Uncorrelated", "EnergyAngular", "AngularEnergy",
    "KalbachMann", "NBodyPhaseSpace", "Recoil",
    "NOT_IMPLEMENTED_DISTRIBUTIONS",
]


@dataclass
class AngularTwoBody:
    """§18. P(mu|E) for two-body kinematics — what ENDF MF4 carries.

    ``byEnergy`` maps incident energy to the angular function at that energy: a
    ``Legendre`` for a coefficient representation, an ``XYs1d`` for a tabulated
    one. Both appear in MF4 and both must survive a round trip.
    """

    byEnergy: Dict[float, Function1d] = field(default_factory=dict)
    productFrame: Frame = Frame.centerOfMass
    label: Optional[str] = None

    @property
    def energies(self) -> List[float]:
        return sorted(self.byEnergy)


@dataclass
class Unspecified:
    """§18. The distribution is deliberately not given (ENDF's LTT=0 and friends)."""

    label: Optional[str] = None
    productFrame: Frame = Frame.lab


class _UnimplementedDistribution:
    gndsNodeName = "?"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            f"GNDS distribution {self.gndsNodeName!r} is declared but not "
            f"implemented (phase 7b). It is present rather than absent so that a "
            f"reader meeting one is told what is missing."
        )


class Uncorrelated(_UnimplementedDistribution):
    gndsNodeName = "uncorrelated"


class EnergyAngular(_UnimplementedDistribution):
    gndsNodeName = "energyAngular"


class AngularEnergy(_UnimplementedDistribution):
    gndsNodeName = "angularEnergy"


class KalbachMann(_UnimplementedDistribution):
    gndsNodeName = "KalbachMann"


class NBodyPhaseSpace(_UnimplementedDistribution):
    gndsNodeName = "NBodyPhaseSpace"


class Recoil(_UnimplementedDistribution):
    gndsNodeName = "recoil"


NOT_IMPLEMENTED_DISTRIBUTIONS = {
    cls.gndsNodeName: cls
    for cls in (Uncorrelated, EnergyAngular, AngularEnergy, KalbachMann,
                NBodyPhaseSpace, Recoil)
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
