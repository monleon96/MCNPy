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

import numpy as np

from ..axes import Axes
from ..enums import (ENDF_INT_TO_INTERPOLATION, INTERPOLATION_TO_ENDF_INT,
                     Interpolation, InterpolationQualifier,
                     joinEndfTab2Code, splitEndfTab2Code)
from .base import Function1d

__all__ = [
    "Function2d", "XYs2d", "Regions2d", "XYs3d", "Regions3d",
    "fromEndfTab2", "toEndfTab2", "fromEndfTab3", "toEndfTab3",
    "NOT_IMPLEMENTED_NODES",
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

    # ------------------------------------------------------------------
    # The children as tables: read, integrate, replace
    # ------------------------------------------------------------------
    #
    # **The four capabilities that made perturbing MF5 format work.**
    # ``MF5PartialTabulated`` has ``table``, ``normalisation``,
    # ``group_integrals`` and ``replace_table``; the model node holding the same
    # numbers had none of them, so the PFNS applier had to be written against
    # the ENDF class and the perturbation became a property of ENDF. These are
    # those four on the node, spelled the model's way, and the arithmetic under
    # them is the one :mod:`kika.processing.panel_integrals` gives the ENDF
    # class -- so the two integrate identically by construction rather than by
    # agreement.
    #
    # They are on ``Function2d`` and not on ``XYs2d`` alone because an ENDF
    # TAB2 with more than one interpolation region decodes to a ``Regions2d``,
    # and which of the two a tape produces is not something a caller holding a
    # fission spectrum should have to branch on. ``k`` therefore counts the 1-d
    # children in **file order**, across regions, exactly as the flat TAB2 does.

    # ``function1ds`` is deliberately **not** declared here. ``XYs2d`` holds it
    # as a dataclass field and ``Regions2d`` computes it as a property, and a
    # property on this base would be a data descriptor that ``XYs2d.__init__``
    # could not assign through. Both spellings answer ``self.function1ds``,
    # which is all these methods need.

    def _childSlot(self, k: int) -> Tuple[List[Function1d], int]:
        """``(the list that holds child k, its position in that list)``.

        The one thing the two containers do differently: an ``XYs2d`` holds its
        children directly, a ``regions2d`` holds them one region deep. Written
        as a hook so :meth:`replaceTable` is one implementation and not two.
        """
        raise NotImplementedError

    @property
    def outerDomainValues(self) -> List[float]:
        """The outer abscissae of every 1-d child, in file order."""
        return [f.outerDomainValue for f in self.function1ds]

    def table(self, k: int) -> Tuple["np.ndarray", "np.ndarray"]:
        """``(xs, ys)`` of the 1-d child at position *k*.

        The child's own grid, not a resampling of it: a group integral taken on
        anything else is not the integral the evaluator wrote.
        """
        from .integration import tabulateFunction1d

        xs, ys, _codes = tabulateFunction1d(self.function1ds[k],
                                            f"{type(self).__name__} child {k}")
        return xs, ys

    def normalisation(self, k: int) -> float:
        """``int f_k(x) dx`` over child *k*'s whole domain.

        Named as the spectra call it rather than ``integral``: for an MF5 table
        this is the number ENDF-6 §5 requires to be 1, and what a perturbation
        has to give back unchanged.
        """
        return self.function1ds[k].integrate()

    def groupIntegrals(self, k: int, boundaries) -> "np.ndarray":
        """``P_j`` of child *k* over *boundaries* -- what MF35 is a covariance of."""
        return self.function1ds[k].groupIntegrals(boundaries)

    def replaceTable(self, k: int, xs, ys, regions=None) -> Function1d:
        """Put a new table in child *k*'s place, and return it.

        Keeps the child's ``axes``, ``label``, ``outerDomainValue`` and
        ``index``: the replacement is the same function of the same outer
        coordinate, differently tabulated, and a caller that had to restate
        those would eventually restate one of them wrongly.

        *regions* is an ENDF ``(NBT, INT)`` list when the new table is
        piecewise. Leaving it ``None`` means "one region under the rule this
        child already states", which is defined only when the child *has* one
        rule -- so a multi-region child raises rather than being flattened.
        **That refusal is the point.** ``MF5PartialTabulated.replace_table``
        defaults to a single lin-lin region, which silently relabels every panel
        of a table whose later regions were histogram; no MF5 tape read so far
        has one, and if one turns up the caller has to say what it wants rather
        than find out from the file that comes out.
        """
        from .regions1d import Regions1d
        from .xys1d import XYs1d

        holder, position = self._childSlot(k)
        old = holder[position]
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)

        if regions is None:
            interpolation = getattr(old, "interpolation", None)
            if interpolation is None:
                raise ValueError(
                    f"child {k} is a {type(old).__name__} carrying "
                    f"{len(getattr(old, 'function1ds', ()))} interpolation "
                    f"region(s), so 'the rule it already states' is not one "
                    f"rule. Pass regions=[(NBT, INT), ...] saying what the new "
                    f"table's regions are"
                )
            new: Function1d = XYs1d(xs=xs, ys=ys, interpolation=interpolation,
                                    axes=old.axes)
        else:
            new = Regions1d.fromEndfRegions(
                xs, ys, [(int(a), int(b)) for a, b in regions], axes=old.axes)
            if len(new.function1ds) == 1:
                # gnds.xsd puts minOccurs="2" on the children of regions1d, so
                # a one-region regions1d is a node the schema rejects. The same
                # rule the MF5 decoder applies when it builds these.
                new = new.function1ds[0]

        new.label = old.label
        new.outerDomainValue = old.outerDomainValue
        new.index = old.index
        holder[position] = new
        return new


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

    def _childSlot(self, k: int) -> Tuple[List[Function1d], int]:
        return self.function1ds, k

    # ------------------------------------------------------------------
    # The outer axis: refining it exactly
    # ------------------------------------------------------------------

    def _outerCode(self) -> int:
        """The ENDF INT of the outer axis, refusing anything not exact.

        A qualifier is refused with the code it came in as rather than being
        dropped: ``INT=22`` is unit base, which blends two children after
        mapping both onto a common domain, and doing a plain linear blend of a
        unit-base TAB2 is a wrong answer that looks like a right one. 44 of
        ENDF/B-VIII.1's 487 LF=1 MF5 sections state it, so this is a real
        branch and not a defensive one -- and the ENDF-side twin refuses it too,
        by the same test on the same code.
        """
        code = self.endfInterpolationCode
        if code not in (1, 2):
            raise NotImplementedError(
                f"the outer axis of this XYs2d interpolates with INT={code}"
                + (f" ({self.interpolation.value}, "
                   f"{self.interpolationQualifier.value})"
                   if self.interpolationQualifier is not None
                   else f" ({self.interpolation.value})")
                + "; only histogram and lin-lin refine exactly, and an "
                  "inexact refinement of the outer axis moves the central "
                  "value of every node inserted into it"
            )
        return code

    def evaluateAtOuter(self, value: float) -> Tuple["np.ndarray", "np.ndarray"]:
        """The 1-d function at outer coordinate *value*, as ``(xs, ys)``.

        **An exact refinement of the stated interpolant, not an approximation**,
        and a perturbation that inserts nodes depends on that. With lin-lin on
        both axes the function is linear in the outer coordinate at fixed inner
        one, so evaluating both bracketing children on the union of their grids
        loses nothing -- the union refines each, and each is lin-lin -- and
        blending them linearly reproduces the interpolant at the new node
        exactly. Interpolating between the original child and the new one then
        agrees with the original everywhere.

        Copying the neighbouring child instead -- the obvious shortcut -- is
        about 0.1 % wrong on a real MF5 table and shows up as a central-value
        shift in a zero-perturbation run, which is precisely the defect such a
        run exists to rule out.
        """
        from .integration import evaluateExactly

        outer = np.asarray(self.outerDomainValues, dtype=float)
        if outer.size == 0:
            return np.empty(0), np.empty(0)
        if value <= outer[0]:
            return self.table(0)
        if value >= outer[-1]:
            return self.table(outer.size - 1)

        upper = int(np.searchsorted(outer, value, side="right"))
        lower = upper - 1
        if outer[lower] == value:
            return self.table(lower)

        if self._outerCode() == 1:                # histogram: hold the lower
            xs, ys = self.table(lower)
            return xs.copy(), ys.copy()

        xLow, _ = self.table(lower)
        xHigh, _ = self.table(upper)
        union = np.union1d(xLow, xHigh)
        low = evaluateExactly(self.function1ds[lower], union)
        high = evaluateExactly(self.function1ds[upper], union)
        weight = (value - outer[lower]) / (outer[upper] - outer[lower])
        return union, low + weight * (high - low)

    def _refusesInexactChildren(self, position: int) -> None:
        """Refuse to insert between children a lin-lin table cannot represent.

        The blend :meth:`evaluateAtOuter` returns is evaluated at the union of
        the two bracketing grids and written back as a table, and a table needs
        a rule. Lin-lin is the right one exactly when both children are lin-lin
        throughout -- which is every MF5/LF=1 section of both target libraries.
        Where they are not, the blend is a genuinely different function from
        anything a lin-lin table can hold, so this raises instead of writing a
        table that is a few parts in a thousand off and looks like data.
        """
        for k in (position - 1, position):
            if not 0 <= k < len(self.function1ds):
                continue
            child = self.function1ds[k]
            rule = getattr(child, "interpolation", None)
            if rule != Interpolation.linlin:
                raise NotImplementedError(
                    f"child {k} interpolates as "
                    f"{getattr(rule, 'value', type(child).__name__)!r}, so the "
                    f"blend at a new outer node is not a lin-lin table and "
                    f"cannot be written back as one"
                )

    def insertOuterNode(self, value: float) -> int:
        """Add a 1-d child at outer coordinate *value*, refining exactly.

        Returns its position. An outer coordinate the container already carries
        is not duplicated and its position comes back unchanged, so a caller may
        be careless about a band edge that is already a node -- which, measured
        on both PFNS target libraries, every band edge is.
        """
        from .xys1d import XYs1d

        outer = list(self.outerDomainValues)
        for position, existing in enumerate(outer):
            if existing == value:
                return position

        position = int(np.searchsorted(np.asarray(outer, dtype=float), value))
        self._refusesInexactChildren(position)
        xs, ys = self.evaluateAtOuter(value)
        template = self.function1ds[min(position, len(outer) - 1)]
        child = XYs1d(xs=xs, ys=ys, interpolation=Interpolation.linlin,
                      axes=getattr(template, "axes", None))
        child.outerDomainValue = float(value)
        child.index = position
        self.function1ds.insert(position, child)
        for order, sibling in enumerate(self.function1ds):
            # ``index`` is the child's position in its container (§6), so every
            # sibling above the insertion is now one further along. Leaving them
            # stale writes a TAB2 whose records disagree with their own indices.
            sibling.index = order
        return position

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

    def _childSlot(self, k: int) -> Tuple[List[Function1d], int]:
        position = k
        for region in self.function2ds:
            size = len(region.function1ds)
            if position < size:
                return region._childSlot(position)
            position -= size
        raise IndexError(
            f"this regions2d holds {len(self.function1ds)} 1-d child(ren); "
            f"there is no child {k}"
        )

    def insertOuterNode(self, value: float) -> int:
        """Refused: which region a new outer node joins is not derivable.

        The ENDF twin refuses the same case for the same reason -- see
        ``MF5PartialTabulated._renumber_tab2`` -- and no MF5/LF=1 tape read so
        far states a multi-region TAB2. Growing one region means the later
        regions' ``NBT`` all shift, and *which* region the new node belongs to
        at a shared boundary is a statement about interpolation that the caller
        has and this node does not.
        """
        raise NotImplementedError(
            f"inserting an outer node into a regions2d ({len(self)} regions) "
            f"is not implemented: the region the new node joins is not "
            f"derivable from the value alone. No MF5/LF=1 tape read so far "
            f"states a multi-region TAB2"
        )

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



def fromEndfTab3(
    function2ds: "Sequence[Function2d]",
    nbtIntPairs: Sequence[Tuple[int, int]],
    axes: Optional[Axes] = None,
    label: Optional[str] = None,
) -> "XYs3d":
    """One ENDF TAB2 whose nodes are themselves 2-d → an :class:`XYs3d`.

    The three-dimensional twin of :func:`fromEndfTab2`, and simpler than it by
    one branch: **there is no ``regions3d`` to return.** ``gnds.xsd`` defines
    the complexType and declares no element of it, so a file that needed one
    would not be valid GNDS — see :class:`Regions3d`. An ENDF TAB2 with NR>1 at
    this level therefore has no legal home, and this raises rather than
    flattening the regions into one grid, which would silently drop where the
    interpolation law changes.

    The branch is unreachable on the data measured: every LAW=1 and LAW=7 TAB2
    in ENDF/B-VIII.1 writes a single plain ``INT=2``. The raise is what says so
    if that stops being true.
    """
    functions = list(function2ds)
    pairs = [(int(nbt), int(code)) for nbt, code in nbtIntPairs]

    if len(pairs) > 1:
        raise ValueError(
            f"this TAB2 declares NR={len(pairs)} interpolation regions over its "
            f"{len(functions)} two-dimensional nodes, and GNDS has no regions3d "
            f"to put them in: gnds.xsd defines xData_regions_3d_primary and "
            f"declares no element of that type, so the node cannot be written. "
            f"Regions are {pairs}"
        )

    code = pairs[0][1] if pairs else 2
    interpolation, qualifier = splitEndfTab2Code(code)
    return XYs3d(
        function2ds=functions,
        interpolation=interpolation,
        interpolationQualifier=qualifier,
        axes=axes,
        label=label,
    )


def toEndfTab3(form: "XYs3d") -> Tuple[List["Function2d"], List[Tuple[int, int]]]:
    """The inverse: the 2-d node list and the one ``(NBT, INT)`` pair."""
    if not isinstance(form, XYs3d):
        raise TypeError(f"not a three-dimensional form: {type(form).__name__}")
    return list(form.function2ds), [(len(form), form.endfInterpolationCode)]



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
