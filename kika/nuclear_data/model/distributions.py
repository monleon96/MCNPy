"""GNDS-2.1 §18 ``distribution``, with the laws declared and unimplemented.

The roadmap's rule: **empty slots exist, they are not absent.** Filling these is
phase 7b. Until then each law is declared, so a reader meeting one can say which
GNDS node it cannot handle instead of failing somewhere unrelated, and so that
adding the implementation later restructures nothing.

``angularTwoBody`` and ``isotropic2d`` are the exceptions: MF4 is inside kika's
current parser coverage, so phase 3c fills them and they are real containers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from .enums import Frame
from .functions import Function1d, Function2d, Regions2d, XYs2d

__all__ = [
    "Distribution", "AngularTwoBody", "Isotropic2d", "Unspecified",
    "Uncorrelated", "EnergyAngular", "AngularEnergy",
    "KalbachMann", "NBodyPhaseSpace", "Recoil",
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
    """

    angular: Optional[Union[XYs2d, Regions2d]] = None
    productFrame: Frame = Frame.centerOfMass
    label: Optional[str] = None

    @property
    def energies(self) -> List[float]:
        """The incident energies, in file order, **duplicates included**."""
        if self.angular is None:
            return []
        if isinstance(self.angular, Regions2d):
            return [f.outerDomainValue for f in self.angular.function1ds]
        return list(self.angular.outerDomainValues)

    @property
    def function1ds(self) -> List[Function1d]:
        """Every per-energy angular function, in file order."""
        if self.angular is None:
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


@dataclass
class Unspecified:
    """§18. The distribution is deliberately not given."""

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
