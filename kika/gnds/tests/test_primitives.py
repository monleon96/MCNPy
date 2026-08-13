"""Phase 5 P2: §5-6 primitives, gated on every functional node in the fixtures.

The shape of this module is deliberate. Two of its tests
(:func:`test_every_functional_node_in_every_fixture_builds` and
:func:`test_every_xys1d_survives_the_interleaving_round_trip`) walk the
committed files and assert on **every** node rather than on examples, because
the failure mode of a primitives reader is not an exception — it is a function
that builds and is quietly wrong by one point, one axis or one interpolation
rule.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from kika.gnds.primitives import (DECLARED_ELSEWHERE, FUNCTION_1D, FUNCTION_2D,
                                  UnsupportedNode, readAxes, readForm,
                                  readFunction1d, readValues)
from kika.gnds.xpath import Document, Resolver
from kika.nuclear_data.model import (Axis, Constant1d, Grid, Interpolation,
                                     InterpolationQualifier, Legendre,
                                     Regions1d, Regions2d, XYs1d, XYs2d)

FUNCTIONALS = set(FUNCTION_1D) | set(FUNCTION_2D)


def _parse(text: str) -> ET.Element:
    return ET.fromstring(text)


def _functionals(root: ET.Element):
    for node in root.iter():
        if node.tag in FUNCTIONALS:
            yield node


# ---------------------------------------------------------------------------
# values
# ---------------------------------------------------------------------------

def test_values_reads_a_plain_list():
    assert readValues(_parse("<values>1 2.5 -3e2</values>")).tolist() == [1.0, 2.5, -300.0]


def test_values_reads_across_newlines():
    """The distribution wraps long ``values`` at ~100 columns."""
    assert readValues(_parse("<values>1 2\n   3\n4</values>")).size == 4


def test_an_empty_values_node_is_an_empty_array():
    assert readValues(_parse("<values></values>")).size == 0
    assert readValues(_parse("<values>\n   </values>")).size == 0


def test_integer32_values_come_back_as_integers():
    """``flattened`` arrays store their ``starts`` and ``lengths`` this way."""
    values = readValues(_parse('<values valueType="Integer32">0 242 1210</values>'))
    assert values.dtype == np.int32
    assert values.tolist() == [0, 242, 1210]


def test_zero_compression_is_expanded():
    """§5.2.1 ``start``/``length``. No distributed file uses it; it is read anyway."""
    values = readValues(_parse('<values start="2" length="6">7 8</values>'))
    assert values.tolist() == [0, 0, 7, 8, 0, 0]


def test_a_length_shorter_than_the_stored_values_is_refused():
    with pytest.raises(ValueError, match="declares length"):
        readValues(_parse('<values start="2" length="3">7 8</values>'))


# ---------------------------------------------------------------------------
# axes
# ---------------------------------------------------------------------------

def test_axes_keep_the_specs_index_convention(micro_fe56_gnds):
    """§5.1.1: index 0 is the *dependent* axis. Backwards mislabels everything."""
    root = ET.parse(micro_fe56_gnds).getroot()
    xys = next(n for n in root.iter("XYs1d") if n.find("axes") is not None)
    axes = readAxes(xys)
    assert axes.dependent.label == "crossSection"
    assert axes.dependent.unit == "b"
    assert [a.label for a in axes.independent] == ["energy_in"]
    assert axes.byIndex(1).unit == "eV"


def test_a_functional_without_axes_inherits_its_parents(h2_gnds):
    """An ``XYs1d`` inside a ``regions1d`` does not repeat the axes.

    H-2 rather than Fe-56: the Fe-56 trim has no ``regions1d`` at all, so this
    would have been a permanent skip dressed up as a passing test.
    """
    root = ET.parse(h2_gnds).getroot()
    regions = next(root.iter("regions1d"))
    built = readForm(regions)
    assert built.axes is not None
    assert all(child.axes is built.axes for child in built.function1ds)


def test_an_axes_href_is_kept_rather_than_dereferenced():
    axes = readAxes(_parse('<x><axes href="/reactionSuite/x/axes"/></x>'))
    assert axes.href == "/reactionSuite/x/axes"
    assert len(axes) == 0


def test_a_grid_link_needs_a_resolver_and_is_honest_without_one(h2_gnds_cov):
    """A covariance's column grid is a link to its row grid, not a second copy.

    Without a resolver the axis is still reported — as a plain :class:`Axis`,
    which says "this axis exists and its values were not reachable" — rather
    than as a :class:`Grid` pretending to boundaries it does not have.
    """
    document = Document.parse(h2_gnds_cov)
    gridded = next(document.root.iter("gridded2d"))

    withoutResolver = readAxes(gridded)
    linked = withoutResolver.byIndex(1)
    assert type(linked) is Axis, "a grid with an unresolved link claimed values"

    resolver = Resolver(document)
    withResolver = readAxes(
        gridded, lambda href, node: resolver.resolve(href, context=node).element
    )
    resolved = withResolver.byIndex(1)
    assert isinstance(resolved, Grid)
    assert resolved.values is not None and resolved.values.size > 1
    # The column grid is the row grid: that is what the link says.
    np.testing.assert_array_equal(resolved.values, withResolver.byIndex(2).values)


# ---------------------------------------------------------------------------
# the fixture sweep
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", ["h2_gnds", "micro_fe56_gnds"])
def test_every_functional_node_in_every_fixture_builds(fixture, request):
    """Every ``XYs1d``, ``regions1d``, ``constant1d``, ``Legendre``, ``XYs2d``…

    Two unmodified-in-substance evaluations, every functional node in them, no
    exceptions and no ``None``. This is the test that says the registry covers
    what real files contain.
    """
    root = ET.parse(request.getfixturevalue(fixture)).getroot()
    counts = {}
    for node in _functionals(root):
        built = readForm(node)
        assert built is not None
        counts[node.tag] = counts.get(node.tag, 0) + 1
    assert counts, f"{fixture} contains no functional nodes at all"
    assert "XYs1d" in counts


@pytest.mark.parametrize("fixture", ["h2_gnds", "micro_fe56_gnds"])
def test_every_xys1d_survives_the_interleaving_round_trip(fixture, request):
    """``interleaved()`` reproduces the file's own ``values``, bit for bit.

    §6.1.1 stores ``x0 y0 x1 y1 …`` in one node; the model keeps two arrays.
    That de-interleaving is the encoder's inverse, so it is gated at full
    precision here rather than at P7 — a rounding introduced in the reader
    would look like a writer defect three increments later.
    """
    root = ET.parse(request.getfixturevalue(fixture)).getroot()
    checked = 0
    for node in root.iter("XYs1d"):
        original = readValues(node.find("values"))
        np.testing.assert_array_equal(readFunction1d(node).interleaved(), original)
        checked += 1
    assert checked, f"{fixture} has no XYs1d to round-trip"


# ---------------------------------------------------------------------------
# the individual forms
# ---------------------------------------------------------------------------

def test_xys1d_splits_the_interleaved_pairs():
    built = readForm(_parse("<XYs1d><values>1 10 2 20 3 30</values></XYs1d>"))
    assert isinstance(built, XYs1d)
    assert built.xs.tolist() == [1, 2, 3]
    assert built.ys.tolist() == [10, 20, 30]


def test_an_odd_values_count_is_refused():
    """Half a point is not a point, and silently dropping it loses an abscissa."""
    with pytest.raises(ValueError, match="must be even"):
        readForm(_parse("<XYs1d><values>1 10 2</values></XYs1d>"))


def test_a_missing_interpolation_means_lin_lin():
    """§3.4.4's default, and the common case: most XYs1d omit the attribute."""
    assert readForm(_parse("<XYs1d><values>1 2</values></XYs1d>")).interpolation \
        is Interpolation.linlin


def test_the_files_own_interpolation_tokens_are_read(h2_gnds):
    """``log-log``, ``log-lin``, ``flat`` — GNDS spellings, straight onto the enum."""
    root = ET.parse(h2_gnds).getroot()
    seen = {
        readFunction1d(node).interpolation
        for node in root.iter("XYs1d") if "interpolation" in node.attrib
    }
    assert seen, "H-2 has no XYs1d declaring an interpolation"
    assert seen <= set(Interpolation)


def test_regions1d_keeps_the_shared_boundary_point(h2_gnds):
    """Adjacent GNDS regions share an abscissa, and the model keeps it.

    ``Regions1d.fromEndfRegions`` *reconstructs* that sharing from cumulative
    NBT; a GNDS ``regions1d`` already has it, so the children are taken as they
    come. Using the ENDF constructor here would duplicate the point a second
    time.
    """
    root = ET.parse(h2_gnds).getroot()
    regions = [readForm(n) for n in root.iter("regions1d")]
    assert regions, "H-2 has no regions1d"
    for built in regions:
        assert isinstance(built, Regions1d)
        for left, right in zip(built.function1ds, built.function1ds[1:]):
            assert left.domainMax == right.domainMin


def test_constant1d_carries_its_domain():
    built = readForm(_parse(
        '<constant1d label="eval" value="5.1977" domainMin="1e-5" domainMax="1e8"/>'
    ))
    assert isinstance(built, Constant1d)
    assert (built.constant, built.domainMin, built.domainMax) == (5.1977, 1e-5, 1e8)
    assert built.label == "eval"


def test_legendre_keeps_its_outer_domain_value():
    built = readForm(_parse(
        '<Legendre outerDomainValue="1e2"><values>1 -9.3e-5 4.1e-9 0</values></Legendre>'
    ))
    assert isinstance(built, Legendre)
    assert built.outerDomainValue == 100.0
    assert built.coefficients.tolist() == [1.0, -9.3e-5, 4.1e-9, 0.0]


def test_polynomial1d_reads_its_coefficients():
    built = readForm(_parse(
        '<polynomial1d domainMin="1e-5" domainMax="2e7">'
        "<values>157731300 -0.08186255 0</values></polynomial1d>"
    ))
    assert built.coefficients.tolist() == [157731300.0, -0.08186255, 0.0]
    assert built.domainMax == 2e7


def test_xys2d_is_an_ordered_list_that_may_repeat_an_outer_value():
    """A repeated incident energy is a discontinuity, not a duplicate to drop.

    A ``dict`` keyed on energy loses one of them silently. The model uses a list
    for this reason; the reader must not undo that by de-duplicating.
    """
    built = readForm(_parse(
        "<XYs2d><function1ds>"
        '<Legendre outerDomainValue="3.905e6"><values>1 0</values></Legendre>'
        '<Legendre outerDomainValue="3.905e6"><values>1 0.5</values></Legendre>'
        "</function1ds></XYs2d>"
    ))
    assert isinstance(built, XYs2d)
    assert built.outerDomainValues == [3.905e6, 3.905e6]
    assert built.function1ds[1].coefficients[1] == 0.5


def test_xys2d_reads_its_qualifier(micro_fe56_gnds):
    """The reference implementation and the standard disagree, and files follow FUDGE.

    §3.4.5 (p. 54) allows ``unitBase``; FUDGE writes ``unitbase``
    (``xData/enums.py:74``) and so does every file in the distribution. The
    model keeps the standard's spelling and the reader accepts the file's.

    Fe-56 rather than H-2: H-2's two ``XYs2d`` carry no qualifier at all, so
    this deviation would go untested against a real file on that fixture — and
    a deviation covered only by hand-written XML is a deviation nobody has
    confirmed still exists.
    """
    root = ET.parse(micro_fe56_gnds).getroot()
    written = {
        n.attrib["interpolationQualifier"] for n in root.iter("XYs2d")
        if "interpolationQualifier" in n.attrib
    }
    assert "unitbase" in written, "the deviation this test exists for is gone"

    qualified = [
        readForm(n) for n in root.iter("XYs2d")
        if n.attrib.get("interpolationQualifier")
    ]
    assert all(
        built.interpolationQualifier is InterpolationQualifier.unitBase
        for built in qualified
    )


def test_the_specs_own_spelling_is_accepted_too():
    """A spec-conformant file must read as well as a FUDGE-written one."""
    built = readForm(_parse(
        '<XYs2d interpolationQualifier="unitBase"><function1ds/></XYs2d>'
    ))
    assert built.interpolationQualifier is InterpolationQualifier.unitBase


def test_an_empty_qualifier_is_no_qualifier():
    """FUDGE spells "none" as ``''``; §3.4.5 spells it by omitting the attribute."""
    assert readForm(_parse(
        '<XYs2d interpolationQualifier=""><function1ds/></XYs2d>'
    )).interpolationQualifier is None


def test_a_fudge_extension_qualifier_is_refused_not_approximated():
    """``cumulativePoints`` is FUDGE's, not §3.4.5's, and means something else.

    Folding it onto ``unitBase`` would change how every distribution between two
    incident energies is interpolated, and nothing downstream could tell.
    """
    with pytest.raises(ValueError, match="refused rather than folded"):
        readForm(_parse(
            '<XYs2d interpolationQualifier="cumulativePoints"><function1ds/></XYs2d>'
        ))


def test_regions2d_nests_two_dimensional_children(h2_gnds):
    root = ET.parse(h2_gnds).getroot()
    built = [readForm(n) for n in root.iter("regions2d")]
    assert built, "H-2 has no regions2d"
    for region in built:
        assert isinstance(region, Regions2d)
        assert region.function1ds, "a regions2d flattened to nothing"


# ---------------------------------------------------------------------------
# refusal
# ---------------------------------------------------------------------------

def test_an_unknown_node_names_itself():
    with pytest.raises(UnsupportedNode) as raised:
        readForm(_parse("<somethingElse/>"))
    assert raised.value.node == "somethingElse"


@pytest.mark.parametrize("node", sorted(DECLARED_ELSEWHERE))
def test_a_deliberately_unread_node_says_why(node):
    """``Ys1d`` and ``gridded1d`` are implemented in the model and not read here.

    The distinction matters to whoever reads the report: "kika cannot represent
    this" and "kika can represent it and has never met one" call for different
    next steps.
    """
    with pytest.raises(UnsupportedNode) as raised:
        readForm(_parse(f"<{node}/>"))
    assert raised.value.node == node
    assert str(raised.value).endswith(DECLARED_ELSEWHERE[node])


def test_a_node_the_model_declares_and_does_not_implement_says_so():
    """``XYs3d`` is a slot phase 7b fills; the message must say that, not "unknown"."""
    with pytest.raises(UnsupportedNode) as raised:
        readForm(_parse("<XYs3d/>"))
    assert "phase 7b" in str(raised.value)
