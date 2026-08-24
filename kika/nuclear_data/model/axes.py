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


#: The axes of a two-body angular distribution — the MF4 case, and the one
#: container in the library that is **three** axes rather than two, so it is
#: built literally instead of through :meth:`Axes.forFunction1d`.
#:
#: The triple is not a guess: all three committed GNDS fixtures carry exactly
#: this on their ``angularTwoBody`` container, and they are trims of
#: FUDGE-distributed files. It is the same triple for LTT=1, 2 and 3 alike —
#: ``mu`` stays index 1 even where the inner function is a Legendre series,
#: because the *container* describes P(mu|E) however its regions spell it.
#:
#: **The caller must share one object across a container and its children.**
#: ``kika/gnds/encode.py:_axesUnlessNested`` decides "this child inherits" by
#: object *identity*, so calling this function once per region would make a
#: nested form look like it carried axes of its own and be reported as a loss.
def angularAxes() -> Axes:
    return Axes([
        Axis(index=2, label="energy_in", unit="eV"),
        Axis(index=1, label="mu", unit=""),
        Axis(index=0, label="P(mu|energy_in)", unit=""),
    ])


#: The axes of a secondary-energy distribution — ENDF MF5, and the ``energy``
#: half of GNDS §18.3's ``uncorrelated``. Three axes, like
#: :func:`angularAxes`, and built literally for the same reason.
#:
#: **The dependent unit is ``1/eV`` and not the empty string**, which is the one
#: place this triple is not the angular one with a label swapped: P(E'|E) is a
#: density in the outgoing energy, so it carries the reciprocal of that axis's
#: unit. ``mu`` is dimensionless and P(mu|E) with it; ``energy_out`` is not.
#: Taken verbatim off ``kika/gnds/tests/data/n-001_H_003.endf.gnds.xml``, whose
#: ``<uncorrelated>/<energy>`` is a FUDGE-distributed node, rather than derived.
#:
#: **The caller must share one object across a container and its children**, for
#: the identity reason written on :func:`angularAxes`.
def energyAxes() -> Axes:
    return Axes([
        Axis(index=2, label="energy_in", unit="eV"),
        Axis(index=1, label="energy_out", unit="eV"),
        Axis(index=0, label="P(energy_out|energy_in)", unit="1/eV"),
    ])


#: The axes of one half of a §18.6 ``KalbachMann`` — its ``f``, ``r`` or ``a``.
#:
#: Three axes like :func:`angularAxes` and :func:`energyAxes`, and the same
#: shape for all three halves except the dependent one, which is why this takes
#: the component rather than there being three near-identical factories.
#:
#: The triple is taken verbatim off a FUDGE-distributed node —
#: ``n-006_C_012.endf.gnds.xml``'s ``KalbachMann`` writes ``f`` with unit
#: ``1/eV`` and ``r`` with none — rather than derived. ``f`` is a density in
#: the outgoing energy and carries that axis's reciprocal unit; ``r`` is the
#: pre-equilibrium fraction and is dimensionless.
#:
#: **``a`` has no witness.** The census counted zero ``<a>`` in the library's
#: 3 730 ``KalbachMann`` nodes, so its unit here is reasoned and not copied: the
#: Kalbach slope multiplies a cosine, which is dimensionless, so it is too.
#:
#: **The caller must share one object across a container and its children**, for
#: the identity reason written on :func:`angularAxes`.
_KALBACH_UNITS = {"f": "1/eV", "r": "", "a": ""}


def kalbachMannAxes(component: str) -> Axes:
    try:
        unit = _KALBACH_UNITS[component]
    except KeyError:
        raise ValueError(
            f"a KalbachMann has the three components f, r and a "
            f"(gnds.xsd:1806-1811), not {component!r}"
        ) from None
    return Axes([
        Axis(index=2, label="energy_in", unit="eV"),
        Axis(index=1, label="energy_out", unit="eV"),
        Axis(index=0, label=component, unit=unit),
    ])


#: The axes of a §18.4 ``energyAngular`` or §18.5 ``angularEnergy`` — the only
#: **four**-axis container the library contains, because it is the only one
#: whose function is of three variables.
#:
#: Taken verbatim off ``n-026_Fe_056.endf.gnds.xml``'s ``energyAngular``, whose
#: ``XYs3d`` writes ``energy_in``/``energy_out``/``mu`` at indices 3, 2 and 1
#: and ``P(energy_out,mu|energy_in)`` in ``1/eV`` at index 0.
#:
#: **``outermost`` is the whole difference between the two nodes.** §18.4 nests
#: P(E'|E) outside P(mu|E,E') and §18.5 the other way round, they share a
#: complexType exactly, and the axis labels are the only thing in the file that
#: disagrees — which is why writing one as the other produces a document that
#: validates and states the wrong physics. Passing the order rather than
#: defaulting it is what makes that impossible to do by omission.
#:
#: **The caller must share one object across a container and its children**, for
#: the identity reason written on :func:`angularAxes`.
def energyAngularAxes(outermost: str = "energy_out") -> Axes:
    if outermost == "energy_out":
        middle, inner, dependent = "energy_out", "mu", "P(energy_out,mu|energy_in)"
    elif outermost == "mu":
        middle, inner, dependent = "mu", "energy_out", "P(mu,energy_out|energy_in)"
    else:
        raise ValueError(
            f"the outer axis of a three-dimensional distribution is either "
            f"'energy_out' (§18.4 energyAngular) or 'mu' (§18.5 angularEnergy), "
            f"not {outermost!r}"
        )
    return Axes([
        Axis(index=3, label="energy_in", unit="eV"),
        Axis(index=2, label=middle, unit="eV" if middle == "energy_out" else ""),
        Axis(index=1, label=inner, unit="eV" if inner == "energy_out" else ""),
        Axis(index=0, label=dependent, unit="1/eV"),
    ])
