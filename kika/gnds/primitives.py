"""GNDS §5-6 primitives → the model's own containers.

``values``, ``axes``/``axis``/``grid``, and the functional containers a
``crossSection``, a ``multiplicity`` or a ``distribution`` is built out of.
Nothing here knows what a reaction is; :mod:`kika.gnds.decode` does.

**Which nodes are implemented, and why not the rest.** The registries below
cover every functional node that occurs in the 558 neutron evaluations of
ENDF/B-VIII.1-GNDS, and nothing else:

===================  =========================================================
``XYs1d``            everywhere
``regions1d``        everywhere
``constant1d``       everywhere
``Legendre``         everywhere, inside ``XYs2d``
``polynomial1d``     81 files, in ``fissionEnergyRelease``
``XYs2d``            everywhere, the outer axis of an angular distribution
``regions2d``        ENDF LTT=3, and any multi-region TAB2
===================  =========================================================

``Ys1d`` and ``gridded1d`` are **implemented in the model and deliberately not
read here**: neither occurs in a single distributed neutron evaluation, so a
reader for them would be code no fixture exercises and no test defends. They are
in :data:`DECLARED_ELSEWHERE` so that meeting one produces a message naming the
node and this decision, rather than "unknown node". ``XYs3d``/``regions3d`` are
declared-and-empty in the model itself (phase 7b), and the §18 distribution laws
the same; the model's own ``NOT_IMPLEMENTED_NODES`` and
``NOT_IMPLEMENTED_DISTRIBUTIONS`` are consulted rather than duplicated.

**Uncertainty on a functional is a known loss.** §7 lets any functional carry an
``uncertainty`` child, and real files do — a ``crossSection`` form pointing at
its own covariance with ``<uncertainty><covariance href="$covariances#…"/>``.
No functional in ``kika/nuclear_data/model`` has a slot for it, so it is
reported as a loss rather than dropped. The information is recoverable from the
other end: §25.2.3's ``rowData/href`` points from the covariance back at the
cross section, and that is the direction :mod:`kika.gnds.decode` reads.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from fractions import Fraction
from typing import Callable, Dict, List, Optional

import numpy as np

from kika.gnds.nodes import reads, readersOf
from kika.nuclear_data.model import (Axes, Axis, Constant1d, Grid, GridStyle,
                                     Interpolation, InterpolationQualifier,
                                     Legendre, Polynomial1d, Regions1d,
                                     Regions2d, ValueType, XYs1d, XYs2d)

__all__ = [
    "UnsupportedNode", "readValues", "readAxes", "readForm",
    "readFunction1d", "readFunction2d", "isFunction1d", "isFunction2d",
    "readFraction", "formatFraction",
    "FUNCTION_1D", "FUNCTION_2D", "DECLARED_ELSEWHERE",
]

#: §6.1.1 and §3.4.4: a functional with no ``interpolation`` attribute is
#: lin-lin. Most ``XYs1d`` nodes in the library omit it, so this default is not
#: a fallback — it is the common case.
DEFAULT_INTERPOLATION = Interpolation.linlin


class UnsupportedNode(ValueError):
    """A GNDS node this reader recognises but does not build a model object for.

    Carries the node name so the caller can name it in a ``ConversionReport``
    instead of turning it into a traceback. Everything that raises this is a
    documented decision, never a gap somebody forgot about.
    """

    def __init__(self, node: str, reason: str) -> None:
        self.node = node
        super().__init__(f"GNDS node {node!r}: {reason}")


#: Nodes the model implements but this reader does not read, with the reason.
#: Kept explicit so the message distinguishes "kika cannot represent this" from
#: "kika can represent it and has never seen one".
DECLARED_ELSEWHERE = {
    "Ys1d": (
        "the model implements it, but no neutron evaluation in "
        "ENDF/B-VIII.1-GNDS contains one, so reading it would be untested code"
    ),
    "gridded1d": (
        "the model implements it, but no neutron evaluation in "
        "ENDF/B-VIII.1-GNDS contains one, so reading it would be untested code"
    ),
    # `discreteGamma` and `primaryGamma` were here until phase 7b modelled
    # them. They are not functionals and never were — they reach the model
    # through `kika.gnds.distributions`, at the `uncorrelated/energy` choice
    # point where the schema puts them (gnds.xsd:1697), and a functional reader
    # should not be the thing that has an opinion about them.
    "branching1d": "no model node; §18 isomeric branching is phase 7b",
}


# ---------------------------------------------------------------------------
# §3.4 fractions
# ---------------------------------------------------------------------------

def readFraction(text: str) -> float:
    """``"1/2"`` → ``0.5``. §3.4's spelling for spins and parities.

    The model stores a spin as a float, because that is what arithmetic on one
    needs, and GNDS writes it as an exact fraction, because a nuclear spin *is*
    a half-integer and ``0.5`` invites the question of whether it was rounded.
    Both statements are right, which is why the conversion is a named function
    with an inverse rather than a ``float()`` call at each site.

    An integer spelling (``"0"``, ``"3"``) is admitted and is the majority in
    the library — a whole spin is not written ``"3/1"``.
    """
    return float(Fraction(text.strip()))


def formatFraction(value: float, denominatorLimit: int = 2) -> str:
    """``0.5`` → ``"1/2"``. The inverse of :func:`readFraction`.

    **The inverse has to reproduce the spelling, not merely the number.** A
    writer emitting ``0.5`` where the file said ``1/2`` produces a valid
    document that no longer round-trips and whose diff against the original is
    noise on every spin in the evaluation.

    ``denominatorLimit`` is 2 because a nuclear spin is an integer or a
    half-integer and nothing else; a value that is neither is a bug upstream,
    and rounding it silently to the nearest half is how that bug would survive.
    """
    fraction = Fraction(value).limit_denominator(denominatorLimit)
    if float(fraction) != value:
        raise ValueError(
            f"{value!r} is not an integer or a half-integer, so it is not a "
            f"spin or a parity and §3.4 has no fraction spelling for it"
        )
    return str(fraction.numerator) if fraction.denominator == 1 else str(fraction)


# ---------------------------------------------------------------------------
# §5.2 values
# ---------------------------------------------------------------------------

def readValues(element: ET.Element) -> np.ndarray:
    """§5.2.1 ``values`` → a 1-D array, zero compression expanded.

    ``start`` and ``length`` are the specification's leading/trailing zero
    compression. **No file in the distribution uses them** — every ``values``
    node carries at most ``label`` and ``valueType`` — but they are read because
    a file that did use them and was read without them would be wrong by
    silence, and the model's :class:`~kika.nuclear_data.model.values.Values`
    already carries the fields.
    """
    valueType = ValueType(element.attrib.get("valueType", ValueType.Float64.value))
    dtype = np.int32 if valueType is ValueType.Integer32 else float

    text = element.text or ""
    stored = np.fromstring(text, dtype=float, sep=" ") if text.strip() else np.empty(0)
    stored = stored.astype(dtype) if dtype is not float else stored

    start = int(element.attrib.get("start", 0))
    length = element.attrib.get("length")
    if start == 0 and length is None:
        return stored

    total = int(length) if length is not None else start + stored.size
    if total < start + stored.size:
        raise ValueError(
            f"a values node declares length={total} but stores {stored.size} "
            f"numbers from start={start}"
        )
    out = np.zeros(total, dtype=dtype)
    out[start:start + stored.size] = stored
    return out


# ---------------------------------------------------------------------------
# §5.1 axes
# ---------------------------------------------------------------------------

def readAxes(parent: ET.Element, resolve: Optional[Callable] = None) -> Optional[Axes]:
    """The ``axes`` child of a functional, or ``None`` when it has none.

    Absent is normal and is not a defect: an ``XYs1d`` inside a ``regions1d``
    or an ``XYs2d`` inherits its parent's axes rather than repeating them, so
    the caller passes those down.

    Parameters
    ----------
    resolve
        Optional ``(href, contextElement) -> Optional[Element]``. A covariance's
        column grid does not repeat its row grid's 124 boundaries; it stores
        ``<link href="../../grid[@index='2']/values"/>`` instead, so reading one
        without a resolver yields a grid with **no values at all**. This module
        holds no resolver of its own — that is :mod:`kika.gnds.xpath`'s job and
        it needs the whole document — so the caller supplies one. Passing
        ``None`` is allowed and gives a plain :class:`Axis` for such a grid,
        which is the honest representation of "this axis exists and its values
        were not reachable".
    """
    element = parent.find("axes")
    if element is None:
        return None
    if "href" in element.attrib and len(element) == 0:
        # §5.1.1 lets one function borrow another's axes wholesale. Kept as the
        # href it is; the model's `Axes.href` field exists for this.
        return Axes(href=element.attrib["href"])

    axes: List[Axis] = []
    for child in element:
        index = int(child.attrib["index"])
        label = child.attrib.get("label", "")
        unit = child.attrib.get("unit", "")
        if child.tag == "axis":
            axes.append(Axis(index=index, label=label, unit=unit))
        elif child.tag == "grid":
            values = _gridValues(child, resolve)
            if values is None:
                # `Grid.__post_init__` refuses a styled grid with no values,
                # and rightly: a `boundaries` grid that cannot say where the
                # boundaries are is not a grid. Degrade to the axis it still is.
                axes.append(Axis(index=index, label=label, unit=unit))
            else:
                axes.append(Grid(
                    index=index, label=label, unit=unit,
                    style=GridStyle(child.attrib.get("style", GridStyle.none.value)),
                    interpolation=_interpolation(child),
                    values=values,
                ))
        else:
            raise UnsupportedNode(
                child.tag, "an <axes> holds <axis> and <grid>, nothing else"
            )
    return Axes(axes=axes)


def _gridValues(grid: ET.Element,
                resolve: Optional[Callable]) -> Optional[np.ndarray]:
    """A grid's values, whether written out or reached through a ``link``."""
    stored = grid.find("values")
    if stored is not None:
        return readValues(stored)

    link = grid.find("link")
    if link is None or resolve is None:
        return None
    target = resolve(link.attrib.get("href", ""), link)
    return None if target is None else readValues(target)


def _interpolation(element: ET.Element) -> Interpolation:
    declared = element.attrib.get("interpolation")
    return DEFAULT_INTERPOLATION if declared is None else Interpolation(declared)


#: Tokens the files are written with that the specification does not allow, and
#: what they mean. **This table exists because the reference implementation and
#: the standard disagree, and the whole published library follows the
#: implementation.**
#:
#: §3.4.5 of NEA/WKP(2025)6 (p. 54) gives ``interpolationQualifier`` exactly four
#: allowed values — ``direct``, ``unitBase``, ``correspondingEnergies``,
#: ``correspondingPoints`` — and spells the second one in camelCase in its own
#: examples on pp. 99 and 236. FUDGE 6.10.0 defines it as
#: ``unitBase = 'unitbase'`` (``xData/enums.py:74``), so every one of the 558
#: neutron evaluations in ENDF/B-VIII.1-GNDS writes ``unitbase``. kika's model
#: follows the standard, because the model is the standard's shape; the reader
#: accepts both, because the data is the data.
#:
#: FUDGE also defines ``unitbase-unscaled``, ``correspondingPoints-unscaled``,
#: ``cumulativePoints`` and ``cumulativePointsUnscaled``, none of which §3.4.5
#: allows, and it does *not* define ``correspondingEnergies``, which §3.4.5 does
#: — its source even carries a comment saying so. None of those occur in the
#: neutron distribution, so none is mapped here: meeting one raises, naming it,
#: rather than being folded onto a near neighbour.
#:
#: Accepted **silently**, without a report entry. A warning on every qualified
#: node would mean roughly 500 warnings on a single evaluation, which is how a
#: report teaches its reader to ignore it — the same reasoning
#: ``kika/_read.py``'s ``_dropRedirectsWeActedOn`` already applies.
SPEC_TOKEN_FOR = {
    "unitbase": InterpolationQualifier.unitBase,
}


def _qualifier(element: ET.Element) -> Optional[InterpolationQualifier]:
    declared = element.attrib.get("interpolationQualifier")
    if declared is None or declared == "":
        # FUDGE spells "no qualifier" as the empty string; §3.4.5 spells it by
        # omitting the attribute. Both mean the model's ``None``.
        return None
    if declared in SPEC_TOKEN_FOR:
        return SPEC_TOKEN_FOR[declared]
    try:
        return InterpolationQualifier(declared)
    except ValueError:
        raise ValueError(
            f"interpolationQualifier={declared!r} is neither one of §3.4.5's "
            f"allowed values ({', '.join(q.value for q in InterpolationQualifier)}) "
            f"nor a FUDGE spelling kika maps ({', '.join(sorted(SPEC_TOKEN_FOR))}). "
            f"It is refused rather than folded onto the nearest neighbour: a "
            f"qualifier says how to interpolate *between* two distributions, and "
            f"guessing it wrong produces numbers that look right."
        ) from None


def _optionalFloat(element: ET.Element, name: str) -> Optional[float]:
    value = element.attrib.get(name)
    return None if value is None else float(value)


def _optionalInt(element: ET.Element, name: str) -> Optional[int]:
    value = element.attrib.get(name)
    return None if value is None else int(value)


def _values(element: ET.Element) -> np.ndarray:
    child = element.find("values")
    if child is None:
        raise ValueError(f"<{element.tag}> has no <values> child")
    return readValues(child)


# ---------------------------------------------------------------------------
# §6 one-dimensional functionals
# ---------------------------------------------------------------------------

@reads("function1d", "XYs1d")
def _readXYs1d(element: ET.Element, axes: Optional[Axes]) -> XYs1d:
    """§6.1.1. One ``values`` node holding ``x0 y0 x1 y1 …``.

    The interleaving is a serialisation detail the model does not keep, so it is
    undone here and redone by the encoder — :meth:`XYs1d.interleaved` is the
    other half.
    """
    flat = _values(element)
    if flat.size % 2:
        raise ValueError(
            f"an XYs1d holds interleaved (x, y) pairs, so its values count must "
            f"be even; this one has {flat.size}"
        )
    return XYs1d(
        xs=flat[0::2], ys=flat[1::2],
        interpolation=_interpolation(element),
        axes=readAxes(element) or axes,
        label=element.attrib.get("label"),
        outerDomainValue=_optionalFloat(element, "outerDomainValue"),
        index=_optionalInt(element, "index"),
    )


@reads("function1d", "regions1d")
def _readRegions1d(element: ET.Element, axes: Optional[Axes]) -> Regions1d:
    """§6.4.1. ``axes`` plus a ``function1ds`` container of 1-d children.

    Adjacent regions share their boundary abscissa in the files as they do in
    the model, so the children are taken as they come — this is *not* the ENDF
    layout, and ``Regions1d.fromEndfRegions`` (which reconstructs the sharing
    from cumulative NBT) must not be used on it.
    """
    own = readAxes(element) or axes
    return Regions1d(
        function1ds=[readFunction1d(child, own) for child in _children(element)],
        axes=own,
        label=element.attrib.get("label"),
        outerDomainValue=_optionalFloat(element, "outerDomainValue"),
        index=_optionalInt(element, "index"),
    )


@reads("function1d", "constant1d")
def _readConstant1d(element: ET.Element, axes: Optional[Axes]) -> Constant1d:
    """§6.2.1. A value and the domain it is constant over."""
    return Constant1d(
        constant=float(element.attrib["value"]),
        domainMin_=float(element.attrib["domainMin"]),
        domainMax_=float(element.attrib["domainMax"]),
        axes=readAxes(element) or axes,
        label=element.attrib.get("label"),
        outerDomainValue=_optionalFloat(element, "outerDomainValue"),
        index=_optionalInt(element, "index"),
    )


@reads("function1d", "Legendre")
def _readLegendre(element: ET.Element, axes: Optional[Axes]) -> Legendre:
    """§6.3.1. The coefficients ``a_l``, in order from l = 0."""
    return Legendre(
        coefficients=_values(element),
        axes=readAxes(element) or axes,
        label=element.attrib.get("label"),
        outerDomainValue=_optionalFloat(element, "outerDomainValue"),
        index=_optionalInt(element, "index"),
    )


@reads("function1d", "polynomial1d")
def _readPolynomial1d(element: ET.Element, axes: Optional[Axes]) -> Polynomial1d:
    """§6.2. Coefficients ascending in power, over a stated domain."""
    return Polynomial1d(
        coefficients=_values(element),
        domainMin_=float(element.attrib["domainMin"]),
        domainMax_=float(element.attrib["domainMax"]),
        axes=readAxes(element) or axes,
        label=element.attrib.get("label"),
        outerDomainValue=_optionalFloat(element, "outerDomainValue"),
        index=_optionalInt(element, "index"),
    )


# ---------------------------------------------------------------------------
# §6 two-dimensional functionals
# ---------------------------------------------------------------------------

@reads("function2d", "XYs2d")
def _readXYs2d(element: ET.Element, axes: Optional[Axes]) -> XYs2d:
    """§6. An ordered list of 1-d functions, one per outer-axis value.

    Ordered and possibly repeating: Fe-56's MF4/MT2 states 3.905 MeV twice, a
    legitimate discontinuity, and an LTT=3 boundary energy belongs to both
    representations. ``XYs2d.function1ds`` is a list for exactly that reason —
    see ``kika/nuclear_data/model/functions/higher.py``.
    """
    own = readAxes(element) or axes
    return XYs2d(
        function1ds=[readFunction1d(child, own) for child in _children(element)],
        interpolation=_interpolation(element),
        interpolationQualifier=_qualifier(element),
        axes=own,
        label=element.attrib.get("label"),
        outerDomainValue=_optionalFloat(element, "outerDomainValue"),
        index=_optionalInt(element, "index"),
    )


@reads("function2d", "regions2d")
def _readRegions2d(element: ET.Element, axes: Optional[Axes]) -> Regions2d:
    own = readAxes(element) or axes
    return Regions2d(
        function2ds=[readFunction2d(child, own) for child in _children(element)],
        axes=own,
        label=element.attrib.get("label"),
        outerDomainValue=_optionalFloat(element, "outerDomainValue"),
        index=_optionalInt(element, "index"),
    )


def _children(element: ET.Element) -> List[ET.Element]:
    """The functional children of a container, ``axes`` and metadata skipped.

    ``regions1d``/``XYs2d``/``regions2d`` wrap their children in a
    ``function1ds`` or ``function2ds`` container in every file examined, but the
    schema does not require the wrapper on every node, so both shapes are
    accepted rather than one being assumed.
    """
    wrapped = [c for c in element if c.tag in ("function1ds", "function2ds")]
    source = wrapped[0] if wrapped else element
    return [c for c in source if c.tag not in ("axes", "uncertainty")]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

#: §6's one-dimensional functionals, ``{tag: reader}``.
#:
#: **Derived from ``kika/gnds/nodes.py``, not written here.** The literal dict
#: this replaces was a second list of node names beside the writer's
#: ``isinstance`` chain, with nothing comparing the two; the registry is the
#: one place both sides are declared. Adding a reader means decorating it with
#: ``@reads`` and declaring the node in the table -- and forgetting the second
#: half fails at import rather than at some later test.
FUNCTION_1D: Dict[str, Callable] = readersOf("function1d")

#: §6's two-dimensional functionals. Derived, as above.
FUNCTION_2D: Dict[str, Callable] = readersOf("function2d")


def isFunction1d(tag: str) -> bool:
    return tag in FUNCTION_1D


def isFunction2d(tag: str) -> bool:
    return tag in FUNCTION_2D


def readFunction1d(element: ET.Element, axes: Optional[Axes] = None):
    """One 1-d functional node → its model object. Raises :class:`UnsupportedNode`."""
    builder = FUNCTION_1D.get(element.tag)
    if builder is None:
        raise _unsupported(element.tag, "one-dimensional")
    return builder(element, axes)


def readFunction2d(element: ET.Element, axes: Optional[Axes] = None):
    """One 2-d functional node → its model object. Raises :class:`UnsupportedNode`."""
    builder = FUNCTION_2D.get(element.tag)
    if builder is None:
        raise _unsupported(element.tag, "two-dimensional")
    return builder(element, axes)


def readForm(element: ET.Element, axes: Optional[Axes] = None):
    """Either dimension, whichever this node is."""
    if isFunction1d(element.tag):
        return readFunction1d(element, axes)
    if isFunction2d(element.tag):
        return readFunction2d(element, axes)
    raise _unsupported(element.tag, "functional")


def _unsupported(tag: str, kind: str) -> UnsupportedNode:
    """One message, assembled from whichever list already knows about this node."""
    if tag in DECLARED_ELSEWHERE:
        return UnsupportedNode(tag, DECLARED_ELSEWHERE[tag])

    from kika.nuclear_data.model.functions.higher import NOT_IMPLEMENTED_NODES

    if tag in NOT_IMPLEMENTED_NODES:
        return UnsupportedNode(
            tag,
            "declared in kika's model and not implemented (phase 7b); it is "
            "present rather than absent so a reader meeting one is told what "
            "is missing",
        )
    return UnsupportedNode(tag, f"not a {kind} node kika reads")
