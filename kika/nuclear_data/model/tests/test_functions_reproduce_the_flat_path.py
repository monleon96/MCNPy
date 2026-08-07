"""The phase 4 gate: the model's functions equal the flat path **bit for bit**.

``np.testing.assert_array_equal``, not ``allclose``. Phase 3's whole claim is
that the new model computes the same numbers as the code it will replace, and
"the same to 1e-12" is a different and weaker claim — one that lets a
reimplementation drift somewhere nobody looks.

This is cheap to guarantee and expensive to fake: ``XYs1d`` and ``Regions1d``
*call* ``kika.processing.interpolation.interpolate_1d`` rather than
reimplementing the interpolation laws, so what is really being tested here is
that the ENDF ``(NBT, INT)`` translation is right — the region boundaries, the
shared abscissa between consecutive regions, and the
``INT`` code ↔ GNDS-string mapping. Those are the parts a second pair of eyes
would get wrong, and they are the parts these tests exercise.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.nuclear_data.model import ENDF_INT_TO_INTERPOLATION, Regions1d, XYs1d
from kika.processing.interpolation import interpolate_1d

#: A grid with something interesting on every scale: decades for the log laws,
#: a flat stretch, and values that are not round.
XS = np.array([1.0e-5, 1.0e-3, 1.0e-1, 1.0, 3.7, 10.0, 137.0, 1.0e3, 2.0e5, 2.0e7])
YS = np.array([12.5, 9.75, 4.125, 3.0, 3.0, 2.5, 1.875, 0.5, 0.03125, 0.0078125])

#: Query points: inside, exactly on knots, and outside both ends.
XQ = np.array([
    1.0e-6, 1.0e-5, 5.0e-5, 1.0e-3, 0.05, 1.0, 2.0, 3.7, 7.0, 10.0,
    99.0, 137.0, 500.0, 1.0e3, 1.0e4, 2.0e5, 1.0e6, 2.0e7, 5.0e7,
])

ALL_INT_CODES = (1, 2, 3, 4, 5)


@pytest.mark.parametrize("intCode", ALL_INT_CODES)
@pytest.mark.parametrize("outOfRange", ["zero", "hold"])
def test_xys1d_equals_the_flat_interpolator(intCode, outOfRange):
    """One region, every INT code, both out-of-range policies."""
    expected = interpolate_1d(XS, YS, [(XS.size, intCode)], XQ, outOfRange)
    function = XYs1d(xs=XS, ys=YS, interpolation=ENDF_INT_TO_INTERPOLATION[intCode])

    np.testing.assert_array_equal(function.evaluate(XQ, outOfRange), expected)


@pytest.mark.parametrize("outOfRange", ["zero", "hold"])
def test_regions1d_equals_the_flat_interpolator_on_a_multi_region_table(outOfRange):
    """Three regions with three different laws — the case the flat class cannot hold.

    ``CrossSection`` would collapse this to one "dominant" scheme and hide the
    rest in ``metadata``. ``Regions1d`` keeps all three, and must still evaluate
    to exactly what ENDF's own layout evaluates to.
    """
    pairs = [(4, 2), (7, 5), (10, 1)]  # lin-lin, then log-log, then flat
    expected = interpolate_1d(XS, YS, pairs, XQ, outOfRange)

    regions = Regions1d.fromEndfRegions(XS, YS, pairs)
    np.testing.assert_array_equal(regions.evaluate(XQ, outOfRange), expected)


def test_the_multi_region_table_really_uses_more_than_one_law():
    """Guards against a fixture that would make the test above vacuous."""
    regions = Regions1d.fromEndfRegions(XS, YS, [(4, 2), (7, 5), (10, 1)])
    laws = {f.interpolation for f in regions.function1ds}
    assert len(laws) == 3, f"expected three distinct laws, got {laws}"


def test_endf_regions_round_trip_exactly():
    """``fromEndfRegions`` then ``toEndfRegions`` returns the input untouched.

    The shared boundary point is the trap: ENDF regions overlap in one abscissa,
    so a naive split-and-rejoin either duplicates or loses a point, and the
    resulting grid still *looks* plausible.
    """
    pairs = [(4, 2), (7, 5), (10, 1)]
    regions = Regions1d.fromEndfRegions(XS, YS, pairs)
    xs, ys, rebuilt = regions.toEndfRegions()

    np.testing.assert_array_equal(xs, XS)
    np.testing.assert_array_equal(ys, YS)
    assert rebuilt == pairs


@pytest.mark.parametrize("intCode", ALL_INT_CODES)
def test_a_single_region_regions1d_matches_the_equivalent_xys1d(intCode):
    """The degenerate case has to agree with itself."""
    regions = Regions1d.fromEndfRegions(XS, YS, [(XS.size, intCode)])
    single = XYs1d(xs=XS, ys=YS, interpolation=ENDF_INT_TO_INTERPOLATION[intCode])

    np.testing.assert_array_equal(regions.evaluate(XQ), single.evaluate(XQ))


def test_scalar_queries_stay_scalar():
    """``interpolate_1d`` returns a float for a float; the model must not widen it."""
    function = XYs1d(xs=XS, ys=YS)
    value = function.evaluate(5.0)
    assert isinstance(value, float)
    assert value == interpolate_1d(XS, YS, [(XS.size, 2)], 5.0, "zero")


# ---------------------------------------------------------------------------
# The INT <-> GNDS mapping, which is the part with a real chance of being wrong
# ---------------------------------------------------------------------------

def test_the_interpolation_mapping_is_a_bijection():
    from kika.nuclear_data.model import INTERPOLATION_TO_ENDF_INT

    assert len(ENDF_INT_TO_INTERPOLATION) == len(INTERPOLATION_TO_ENDF_INT) == 6
    for code, name in ENDF_INT_TO_INTERPOLATION.items():
        assert INTERPOLATION_TO_ENDF_INT[name] == code


def test_lin_log_logs_the_independent_axis_not_the_dependent_one():
    """§3.4.4 footnote 11 read the way it is actually written.

    *"The first string in a name such as 'log-lin' refers to the dependent axis
    (the y-axis) and the second to the independent axis (the x-axis)."* So
    ``lin-log`` = dependent linear, independent logarithmic = ENDF ``INT=3``,
    which is the law where *x* is logged. Getting this backwards would swap two
    interpolation laws throughout the library while every shape check still
    passed, so it is asserted numerically rather than by inspection.
    """
    from kika.nuclear_data.model import Interpolation

    xs = np.array([1.0, 10.0, 100.0])
    ys = np.array([0.0, 1.0, 2.0])

    linlog = XYs1d(xs=xs, ys=ys, interpolation=Interpolation.linlog)
    # y is linear in log(x): halfway in log(x) between 1 and 100 is x = 10,
    # and at x = sqrt(10) ~ 3.1623 the value must be exactly 0.5.
    assert linlog.evaluate(np.sqrt(10.0)) == pytest.approx(0.5, abs=1e-12)
    assert ENDF_INT_TO_INTERPOLATION[3] is Interpolation.linlog

    loglin = XYs1d(xs=np.array([0.0, 1.0]), ys=np.array([1.0, 100.0]),
                   interpolation=Interpolation.loglin)
    # log(y) linear in x: halfway in x is the geometric mean of the ordinates.
    assert loglin.evaluate(0.5) == pytest.approx(10.0, rel=1e-12)
    assert ENDF_INT_TO_INTERPOLATION[4] is Interpolation.loglin
