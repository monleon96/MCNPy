"""``XYs2d`` and ``regions2d``: the outer axis is a list, and that is the point.

These were ``NotImplementedError`` stubs until phase 7b. What forced them open
was a modelling mistake caught on the committed Fe-56 slice: ``AngularTwoBody``
held ``Dict[float, Function1d]``, and MF4 repeats an incident energy — once as
a discontinuity inside one grid, once at the boundary between the Legendre and
tabulated halves of an LTT=3 section. A dict silently keeps one of each pair.

So the invariants tested here are about *not losing entries*, and about the ENDF
TAB2 round trip being exact, because those are the two things a plausible wrong
implementation gets wrong.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.nuclear_data.model import (Legendre, Regions2d, Regions3d, XYs1d,
                                     XYs2d, XYs3d, fromEndfTab2, toEndfTab2)
from kika.nuclear_data.model.enums import Interpolation


def _at(value: float, index: int = 0) -> Legendre:
    return Legendre(coefficients=[1.0, 0.1], outerDomainValue=value, index=index)


# ---------------------------------------------------------------------------
# The list invariant
# ---------------------------------------------------------------------------

def test_a_repeated_outer_domain_value_is_kept():
    """The whole reason these are lists. A dict would report one entry here."""
    form = XYs2d(function1ds=[_at(1.0), _at(2.0), _at(2.0), _at(3.0)])
    assert len(form) == 4
    assert form.outerDomainValues == [1.0, 2.0, 2.0, 3.0]


def test_order_is_file_order_and_is_not_sorted():
    """Sorting looks harmless and destroys the pairing with the file's records."""
    form = XYs2d(function1ds=[_at(3.0), _at(1.0), _at(2.0)])
    assert form.outerDomainValues == [3.0, 1.0, 2.0]


def test_regions_flatten_without_dropping_the_shared_boundary():
    low = XYs2d(function1ds=[_at(1.0), _at(2.0)])
    high = XYs2d(function1ds=[_at(2.0), _at(3.0)])
    form = Regions2d(function2ds=[low, high])

    assert [f.outerDomainValue for f in form.function1ds] == [1.0, 2.0, 2.0, 3.0]
    assert form.sharesBoundaries
    assert (form.domainMin, form.domainMax) == (1.0, 3.0)


def test_a_nested_regions2d_flattens_too():
    """LTT=3 with a multi-region Legendre TAB2 is the only case that nests."""
    inner = Regions2d(function2ds=[
        XYs2d(function1ds=[_at(1.0), _at(2.0)]),
        XYs2d(function1ds=[_at(3.0)]),
    ])
    outer = Regions2d(function2ds=[inner, XYs2d(function1ds=[_at(4.0)])])
    assert [f.outerDomainValue for f in outer.function1ds] == [1.0, 2.0, 3.0, 4.0]


# ---------------------------------------------------------------------------
# The ENDF TAB2 round trip
# ---------------------------------------------------------------------------

def test_one_region_is_an_xys2d_and_not_a_regions2d_of_one():
    form = fromEndfTab2([_at(1.0), _at(2.0)], [(2, 2)])
    assert isinstance(form, XYs2d)
    assert form.interpolation is Interpolation.linlin


def test_more_than_one_region_is_a_regions2d():
    form = fromEndfTab2([_at(float(i)) for i in range(5)], [(2, 2), (5, 5)])
    assert isinstance(form, Regions2d)
    assert [len(r) for r in form] == [2, 3]
    assert form[1].interpolation is Interpolation.loglog


def test_the_split_is_exact_and_loses_no_record():
    """``NBT`` is cumulative and a TAB2 stores each record once, so the region
    lengths must sum to the record count — not to the count plus overlaps."""
    functions = [_at(float(i)) for i in range(7)]
    form = fromEndfTab2(functions, [(3, 1), (5, 2), (7, 4)])
    assert sum(len(r) for r in form) == 7

    rebuilt, pairs = toEndfTab2(form)
    assert [f.outerDomainValue for f in rebuilt] == [float(i) for i in range(7)]
    assert pairs == [(3, 1), (5, 2), (7, 4)]


@pytest.mark.parametrize("pairs", [
    [(4, 2)], [(2, 1), (4, 2)], [(1, 5), (2, 3), (4, 4)],
])
def test_every_region_layout_round_trips(pairs):
    functions = [_at(float(i)) for i in range(4)]
    rebuilt, rebuiltPairs = toEndfTab2(fromEndfTab2(functions, pairs))
    assert len(rebuilt) == 4
    assert rebuiltPairs == pairs


def test_an_empty_tab2_does_not_invent_a_region():
    form = fromEndfTab2([], [])
    assert isinstance(form, XYs2d) and len(form) == 0
    assert np.isnan(form.domainMin)


def test_writing_a_nested_regions2d_as_one_tab2_refuses():
    """It is two TAB2s. Flattening it would lose where the split falls."""
    nested = Regions2d(function2ds=[
        Regions2d(function2ds=[XYs2d(function1ds=[_at(1.0)])]),
        XYs2d(function1ds=[_at(2.0)]),
    ])
    with pytest.raises(ValueError, match="not one TAB2"):
        toEndfTab2(nested)


def test_toEndfTab2_refuses_a_one_dimensional_form():
    with pytest.raises(TypeError, match="two-dimensional"):
        toEndfTab2(XYs1d(xs=[1.0, 2.0], ys=[1.0, 2.0]))


# ---------------------------------------------------------------------------
# What is still declared and empty
# ---------------------------------------------------------------------------

def test_two_dimensional_evaluation_refuses_and_says_why():
    """Guessing an interpolationQualifier would give numbers that look right."""
    form = XYs2d(function1ds=[_at(1.0), _at(2.0)])
    with pytest.raises(NotImplementedError, match="interpolationQualifier"):
        form.evaluate(1.5, 0.0)


def test_regions3d_is_the_one_that_stays_declared_and_empty():
    """Present rather than absent, so a reader is told what is missing.

    This used to be parametrised over ``[XYs3d, Regions3d]``. ``XYs3d`` left the
    list by being implemented; ``Regions3d`` cannot ever leave it, because the
    reason it is empty is not a missing phase — ``gnds.xsd:2286`` defines
    ``xData_regions_3d_primary`` and no ``xs:element`` in the schema is of that
    type, so the node cannot occur in a valid GNDS-2.1 file.
    """
    with pytest.raises(NotImplementedError, match="regions3d") as raised:
        Regions3d()
    assert "no xs:element" in str(raised.value), (
        "the message must carry Regions3d.plannedFor, not a phase number")


def test_three_dimensional_evaluation_refuses_for_the_two_dimensional_reason():
    """One floor up the qualifier question is asked twice, not once."""
    form = XYs3d(function2ds=[XYs2d(function1ds=[_at(1.0)], outerDomainValue=1.0)])
    with pytest.raises(NotImplementedError, match="interpolationQualifier"):
        form.evaluate(1.5, 1.5, 0.0)


def test_xys3d_keeps_a_repeated_outermost_value():
    """The list-not-dict finding of the module docstring, one dimension up."""
    form = XYs3d(function2ds=[
        XYs2d(function1ds=[_at(1.0)], outerDomainValue=3.905e6),
        XYs2d(function1ds=[_at(2.0)], outerDomainValue=3.905e6),
        XYs2d(function1ds=[_at(3.0)], outerDomainValue=7.0e6),
    ])
    assert form.outerDomainValues == [3.905e6, 3.905e6, 7.0e6]
    assert len(form) == 3
    assert form.domainMin == 3.905e6
    assert form.domainMax == 7.0e6
