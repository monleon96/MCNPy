"""GNDS-2.1 §6 higher-dimensional forms: ``XYs2d``, ``regions2d``, ``XYs3d``.

Phase 5 declared these as ``NotImplementedError`` stubs under the model's rule
that *empty slots exist, they are not absent*. Phase 7b fills the two the ENDF
MF4 decoder needs and then ``XYs3d``, which is what ``energyAngular`` and
``angularEnergy`` hold. ``regions3d`` stays declared and empty **permanently**,
and not for want of a phase: the schema gives it no reachable element. See
:class:`Regions3d`.

**Why a list and not a dict — the finding that forced this module.**
``AngularTwoBody`` was first written as ``byEnergy: Dict[float, Function1d]``,
which reads naturally and is wrong. Two things on the committed Fe-56 slice
break it:

* the Legendre grid of MF4/MT2 contains **3.905 MeV twice** — ENDF's way of
  writing a discontinuity in the angular distribution, and a legitimate one;
* the LTT=3 boundary energy **45 MeV is the last Legendre energy and the first
  tabulated one**, so the two representations genuinely share an abscissa.

A dict keyed on energy silently drops one entry in each case. Nothing raises,
nothing is logged, and the file that comes back out is a valid ENDF file with
two fewer angular distributions in it. So the outer axis is an **ordered list**
whose entries carry their own ``outerDomainValue``, exactly as §6 specifies, and
the two-representation case is a ``regions2d`` — which is what that node is for.

**Why there is no 2-d ``evaluate``.** Interpolating *between* two angular
distributions is governed by ``interpolationQualifier`` (§3.4.5: ``unitBase``,
``correspondingPoints``, ...), and between two ``Legendre`` forms of different
order it also needs a padding convention. Guessing any of that would produce
numbers that look right, so :meth:`XYs2d.evaluate` raises and names what is
missing. Callers that need P(mu|E) today go through the flat
``AngularDistribution``, which is honest about interpolating on a single grid.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

from ..axes import Axes
from ..enums import (ENDF_INT_TO_INTERPOLATION, INTERPOLATION_TO_ENDF_INT,
                     Interpolation, InterpolationQualifier,
                     joinEndfTab2Code, splitEndfTab2Code)
from .base import Function1d

__all__ = [
    "Function2d", "XYs2d", "Regions2d", "XYs3d", "Regions3d",
    "fromEndfTab2", "toEndfTab2", "NOT_IMPLEMENTED_NODES",
]


class Function2d(ABC):
    """Abstract base for the §6 two-dimensional functional containers.

    The same four optional attributes every ``functional`` node carries, so a
    ``regions2d`` can hold ``XYs2d`` children that know their own position.
    """

    axes: Optional[Axes]
    label: Optional[str]
    outerDomainValue: Optional[float]
    index: Optional[int]

    @property
    @abstractmethod
    def domainMin(self) -> float:
        """Lowest outer-axis value the function is defined at."""

    @property
    @abstractmethod
    def domainMax(self) -> float:
        """Highest outer-axis value the function is defined at."""


@dataclass
class XYs2d(Function2d):
    """§6. A function of two variables as one 1-d function per outer value.

    ``function1ds`` is ordered and may repeat an ``outerDomainValue``; see the
    module docstring for why that is not an accident to be cleaned up.
    """

    function1ds: List[Function1d] = field(default_factory=list)
    #: Interpolation along the *outer* axis (ENDF's TAB2 ``INT``).
    interpolation: Interpolation = Interpolation.linlin
    interpolationQualifier: Optional[InterpolationQualifier] = None
    axes: Optional[Axes] = None
    label: Optional[str] = None
    outerDomainValue: Optional[float] = None
    index: Optional[int] = None

    def __post_init__(self) -> None:
        self.interpolation = Interpolation(self.interpolation)
        if self.interpolationQualifier is not None:
            self.interpolationQualifier = InterpolationQualifier(self.interpolationQualifier)

    def __len__(self) -> int:
        return len(self.function1ds)

    def __iter__(self):
        return iter(self.function1ds)

    def __getitem__(self, index: int) -> Function1d:
        return self.function1ds[index]

    @property
    def outerDomainValues(self) -> List[float]:
        """The outer abscissae, in file order, **duplicates included**."""
        return [f.outerDomainValue for f in self.function1ds]

    @property
    def domainMin(self) -> float:
        if not self.function1ds:
            return float("nan")
        return float(self.function1ds[0].outerDomainValue)

    @property
    def domainMax(self) -> float:
        if not self.function1ds:
            return float("nan")
        return float(self.function1ds[-1].outerDomainValue)

    @property
    def endfInterpolationCode(self) -> int:
        """The ENDF-6 ``INT`` for the outer axis, **qualifier included**.

        §3.4.4 adopted ENDF's codes, and §0.5.2.1 puts the two-dimensional
        qualifier in the tens digit of the same number — 21-26 is unit base,
        which 44 of ENDF/B-VIII.1's 487 LF=1 MF5 sections use. GNDS states it
        as a second attribute, so writing only ``interpolation`` back would
        drop a statement the file made.
        """
        return joinEndfTab2Code(self.interpolation, self.interpolationQualifier)

    def evaluate(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "XYs2d.evaluate is deliberately absent. Interpolating between two "
            "1-d functions depends on interpolationQualifier (§3.4.5) and, for "
            "Legendre children of different order, on a padding convention. "
            "Neither is decided, and a wrong answer here is indistinguishable "
            "from a right one. Use the flat AngularDistribution for P(mu|E)."
        )

    def __repr__(self) -> str:
        kinds = {type(f).__name__ for f in self.function1ds}
        return (
            f"XYs2d(n={len(self)}, of={'/'.join(sorted(kinds)) or 'nothing'}, "
            f"interpolation={self.interpolation.value!r})"
        )


@dataclass
class Regions2d(Function2d):
    """§6. Adjacent outer-axis regions, each its own :class:`XYs2d`.

    ENDF reaches this shape two ways: a TAB2 with more than one ``(NBT, INT)``
    pair, and LTT=3, where a Legendre region and a tabulated region meet at a
    shared energy. Both are the same node.

    **A child may itself be a ``regions2d``, and only for one reason.** An LTT=3
    section is two TAB2 records, and either of them may carry several
    interpolation regions of its own. Flattening that into one list of regions
    would lose *where the split between the two records falls*, which is what
    tells the encoder how many energies belong to the Legendre TAB2. So the
    LTT=3 node always has exactly two children, each of which is an ``XYs2d``
    when its record has one region and a ``regions2d`` when it has more.
    """

    function2ds: List[Function2d] = field(default_factory=list)
    axes: Optional[Axes] = None
    label: Optional[str] = None
    outerDomainValue: Optional[float] = None
    index: Optional[int] = None

    def __len__(self) -> int:
        return len(self.function2ds)

    def __iter__(self):
        return iter(self.function2ds)

    def __getitem__(self, index: int) -> XYs2d:
        return self.function2ds[index]

    @property
    def domainMin(self) -> float:
        return self.function2ds[0].domainMin if self.function2ds else float("nan")

    @property
    def domainMax(self) -> float:
        return self.function2ds[-1].domainMax if self.function2ds else float("nan")

    @property
    def function1ds(self) -> List[Function1d]:
        """Every 1-d child of every region, flattened recursively, in file order.

        Duplicated boundary values are **kept**, because they are two distinct
        functions that happen to share an abscissa, not one function written
        twice.
        """
        out: List[Function1d] = []
        for region in self.function2ds:
            out.extend(region.function1ds)
        return out

    @property
    def sharesBoundaries(self) -> bool:
        """Whether each region begins where the previous one ended.

        GNDS expects it and ENDF's own regions satisfy it, but a tape is a tape;
        this is a question a caller may ask rather than a rule enforced at
        construction.
        """
        return all(
            self.function2ds[i].domainMax == self.function2ds[i + 1].domainMin
            for i in range(len(self.function2ds) - 1)
        )

    def __repr__(self) -> str:
        return (
            f"Regions2d(n_regions={len(self)}, "
            f"n_functions={len(self.function1ds)})"
        )


# ---------------------------------------------------------------------------
# The ENDF TAB2 shape, in and out
# ---------------------------------------------------------------------------

def fromEndfTab2(
    function1ds: Sequence[Function1d],
    nbtIntPairs: Sequence[Tuple[int, int]],
    axes: Optional[Axes] = None,
    label: Optional[str] = None,
) -> Function2d:
    """One ENDF TAB2 → an :class:`XYs2d`, or a :class:`Regions2d` when NR > 1.

    ``NBT`` is **cumulative**, exactly as in
    :meth:`~kika.nuclear_data.model.functions.regions1d.Regions1d.fromEndfRegions`
    — region *i* ends at the ``NBT[i]``-th function, one-based, counting from
    the start of the whole list.

    Unlike ``regions1d``, the split here does **not** duplicate the boundary
    entry. A TAB2 stores each sub-function exactly once, so region *k* takes
    records ``NBT[k-1]+1 .. NBT[k]`` and the next region starts at the record
    after. (The boundary record is still the lower end of the next region's
    interpolation interval — that is a statement about interpolation, not about
    storage.) Copying it into both regions the way ``regions1d`` does would
    write an extra angular distribution into the file.
    """
    functions = list(function1ds)
    pairs = [(int(nbt), int(code)) for nbt, code in nbtIntPairs]

    if len(pairs) <= 1:
        code = pairs[0][1] if pairs else 2
        interpolation, qualifier = splitEndfTab2Code(code)
        return XYs2d(
            function1ds=functions,
            interpolation=interpolation,
            interpolationQualifier=qualifier,
            axes=axes,
            label=label,
        )

    regions: List[XYs2d] = []
    previous = 0
    for order, (nbt, code) in enumerate(pairs):
        if nbt <= previous:
            continue
        interpolation, qualifier = splitEndfTab2Code(code)
        regions.append(XYs2d(
            function1ds=functions[previous:nbt],
            interpolation=interpolation,
            interpolationQualifier=qualifier,
            axes=axes,
            index=order,
        ))
        previous = nbt
    return Regions2d(function2ds=regions, axes=axes, label=label)


def toEndfTab2(form: Function2d) -> Tuple[List[Function1d], List[Tuple[int, int]]]:
    """The inverse: the flat function list and the cumulative ``(NBT, INT)`` pairs."""
    if isinstance(form, XYs2d):
        return list(form.function1ds), [(len(form), form.endfInterpolationCode)]

    if isinstance(form, Regions2d):
        functions: List[Function1d] = []
        pairs: List[Tuple[int, int]] = []
        for region in form.function2ds:
            if not isinstance(region, XYs2d):
                raise ValueError(
                    "this regions2d nests another regions2d, so it is not one "
                    "TAB2 and cannot be written as one. That nesting means an "
                    "ENDF LTT=3 section: encode each of its two children "
                    "separately, as the MF4 encoder does."
                )
            functions.extend(region.function1ds)
            pairs.append((len(functions), region.endfInterpolationCode))
        return functions, pairs

    raise TypeError(f"not a two-dimensional form: {type(form).__name__}")


# ---------------------------------------------------------------------------
# §6 three dimensions
# ---------------------------------------------------------------------------

@dataclass
class XYs3d:
    """§6.5. A function of three variables as one 2-d function per outer value.

    Structurally ``XYs2d`` one floor up: the schema's ``xData_XYs3d_primary``
    (``gnds.xsd:2260``) is the same ``seq(axes, <container>, uncertainty?)`` with
    ``function2ds`` in place of ``function1ds``, and that container
    (``gnds.xsd:2253``) is a choice of ``XYs2d`` or ``regions2d``, each of which
    **must** carry an ``outerDomainValue``.

    **Ordered, duplicates allowed**, for the reason the module docstring gives
    for ``XYs2d``: a repeated outer abscissa is how ENDF writes a discontinuity,
    and a dict keyed on energy drops one of the two silently. Nothing about the
    third dimension makes that less likely — an energy-angle distribution has an
    outer grid at least as ragged as an angular one.

    **No ``Function3d`` ABC, and no ``regions3d`` sibling.** ``Function2d``
    exists so a ``regions2d`` can hold children that know their own position;
    ``regions3d`` has no such need because it cannot occur. See
    :class:`Regions3d`.
    """

    function2ds: List[Function2d] = field(default_factory=list)
    #: Interpolation along the *outermost* axis.
    #:
    #: Kept although ``xData_XYs3d_primary`` declares no ``interpolation``
    #: attribute where ``xData_XYs2d_primary`` (``gnds.xsd:2193``) does: the
    #: outer axis of a real energy-angular distribution has an interpolation
    #: law whatever the schema forgot, and dropping the field would lose it on
    #: the way in. §6.1.1's default is lin-lin and the writer omits the
    #: attribute at the default, so the ordinary case still validates.
    interpolation: Interpolation = Interpolation.linlin
    interpolationQualifier: Optional[InterpolationQualifier] = None
    axes: Optional[Axes] = None
    label: Optional[str] = None
    #: Only ever set on a child of a ``regions3d``, which no valid file has.
    outerDomainValue: Optional[float] = None
    index: Optional[int] = None

    def __post_init__(self) -> None:
        self.interpolation = Interpolation(self.interpolation)
        if self.interpolationQualifier is not None:
            self.interpolationQualifier = InterpolationQualifier(self.interpolationQualifier)

    def __len__(self) -> int:
        return len(self.function2ds)

    def __iter__(self):
        return iter(self.function2ds)

    def __getitem__(self, index: int) -> Function2d:
        return self.function2ds[index]

    @property
    def outerDomainValues(self) -> List[float]:
        """The outermost abscissae, in file order, **duplicates included**."""
        return [f.outerDomainValue for f in self.function2ds]

    @property
    def domainMin(self) -> float:
        if not self.function2ds:
            return float("nan")
        return float(self.function2ds[0].outerDomainValue)

    @property
    def domainMax(self) -> float:
        if not self.function2ds:
            return float("nan")
        return float(self.function2ds[-1].outerDomainValue)

    @property
    def endfInterpolationCode(self) -> int:
        """The ENDF-6 ``INT`` for the outermost axis, **qualifier included**.

        :attr:`XYs2d.endfInterpolationCode`'s reason, one dimension up.
        """
        return joinEndfTab2Code(self.interpolation, self.interpolationQualifier)

    def evaluate(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "XYs3d.evaluate is deliberately absent, for XYs2d's reason and "
            "more of it: interpolating between two 2-d functions needs the "
            "interpolationQualifier convention (§3.4.5) at *both* levels, and "
            "the 2-d evaluate it would have to call does not exist either. A "
            "wrong answer here is indistinguishable from a right one."
        )

    def __repr__(self) -> str:
        kinds = {type(f).__name__ for f in self.function2ds}
        return (
            f"XYs3d(n={len(self)}, of={'/'.join(sorted(kinds)) or 'nothing'}, "
            f"interpolation={self.interpolation.value!r})"
        )


# ---------------------------------------------------------------------------
# Still declared, still empty
# ---------------------------------------------------------------------------

class _UnimplementedNode:
    """A declared GNDS node with no implementation behind it yet."""

    gndsNodeName: str = "?"
    plannedFor: str = "no phase scheduled"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            f"GNDS node {self.gndsNodeName!r} is declared in kika's model but not "
            f"implemented ({self.plannedFor}). It is present rather than absent so "
            f"that a reader meeting one is told what is missing instead of failing "
            f"somewhere else."
        )


class Regions3d(_UnimplementedNode):
    gndsNodeName = "regions3d"
    plannedFor = (
        "no phase can: gnds.xsd defines xData_regions_3d_primary (:2286) and "
        "no xs:element anywhere in the schema is of that type, so a regions3d "
        "cannot appear in a valid GNDS-2.1 file at all"
    )


#: Every node still declared without an implementation, for the phase 5 reader
#: to consult when it needs to say "I know this node exists and I cannot read
#: it yet".
NOT_IMPLEMENTED_NODES = {cls.gndsNodeName: cls for cls in (Regions3d,)}
