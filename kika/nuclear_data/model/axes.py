"""GNDS-2.1 §5.1: ``axes``, ``axis`` and ``grid``.

**Index 0 is the dependent axis.** §5.1.1: *"For the function x0(xn, ..., x1),
index 0 is for dependent axis x0, 1 is for independent axis x1, ... and n is for
the independent axis xn."* So a cross section σ(E) has ``axis index=1`` for
energy and ``axis index=0`` for the cross section itself, and the list is
conventionally written highest index first. Getting this backwards silently
mislabels every axis in the library, so :meth:`Axes.dependent` and
:meth:`Axes.independent` exist rather than leaving callers to index by hand.

This is the second and last place a unit lives (the first is
``physicalQuantity``). Arrays stay plain numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .enums import GridStyle, Interpolation

__all__ = ["Axis", "Grid", "Axes"]


@dataclass(frozen=True)
class Axis:
    """§5.1.2. An index, a label and a unit for one axis of a function."""

    index: int
    label: str
    unit: str = ""

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"axis index must be >= 0, got {self.index}")
        from .units import parse_unit

        parse_unit(self.unit)


@dataclass(frozen=True)
class Grid(Axis):
    """§5.1.3. An :class:`Axis` that also carries the values it is evaluated at.

    ``style`` says what the values mean — ``points`` for abscissae,
    ``boundaries`` for group edges, ``parameters`` for things like Legendre
    orders (the spec's own example), ``none`` for absent.
    """

    style: GridStyle = GridStyle.none
    interpolation: Interpolation = Interpolation.linlin
    values: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.style is not GridStyle.none and self.values is None:
            raise ValueError(f"grid style {self.style} needs values")


@dataclass
class Axes:
    """§5.1.1. The axes of a function, one per independent axis plus the dependent one.

    Stored in whatever order it was built; look-up is by ``index``, never by
    position, because the specification's own examples list them descending.
    """

    axes: List[Axis] = field(default_factory=list)
    href: Optional[str] = None

    def __post_init__(self) -> None:
        seen = [a.index for a in self.axes]
        if len(seen) != len(set(seen)):
            raise ValueError(f"duplicate axis index in {sorted(seen)}")

    def __len__(self) -> int:
        return len(self.axes)

    def __iter__(self):
        return iter(self.axes)

    def byIndex(self, index: int) -> Axis:
        for axis in self.axes:
            if axis.index == index:
                return axis
        raise KeyError(f"no axis with index {index}; have {sorted(a.index for a in self.axes)}")

    @property
    def dependent(self) -> Axis:
        """Index 0, per §5.1.1."""
        return self.byIndex(0)

    @property
    def independent(self) -> List[Axis]:
        """Indices 1..n, ascending."""
        return sorted((a for a in self.axes if a.index != 0), key=lambda a: a.index)

    @property
    def dimension(self) -> int:
        """``n`` for ``x0(xn, ..., x1)`` — the number of independent axes."""
        return len(self.axes) - 1

    @classmethod
    def forFunction1d(
        cls, dependentLabel: str, dependentUnit: str,
        independentLabel: str, independentUnit: str,
    ) -> "Axes":
        """The two-axis case, which is almost everything kika handles today."""
        return cls([
            Axis(index=1, label=independentLabel, unit=independentUnit),
            Axis(index=0, label=dependentLabel, unit=dependentUnit),
        ])


#: The axes of a pointwise cross section, which is what most of kika's data is.
def crossSectionAxes() -> Axes:
    return Axes.forFunction1d("crossSection", "b", "energy_in", "eV")


#: The axes of a multiplicity — nu-bar being the case kika reads from ENDF MF1.
#: The dependent unit is the empty string, which §2.3.3 defines as
#: *dimensionless* rather than as unknown: a multiplicity is a count of
#: particles per reaction, and it genuinely has no unit. Writing ``"1"`` here
#: would be a second spelling of the same thing.
def multiplicityAxes() -> Axes:
    return Axes.forFunction1d("multiplicity", "", "energy_in", "eV")
