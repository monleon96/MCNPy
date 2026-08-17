"""The node registry, and the six things it is worth having.

The registry earns nothing by existing. What it has to do is **fail** when the
reader and the writer stop naming the same nodes — so the tests here are chosen
for what they would catch, not for coverage of the table:

1. the declared status matches what actually registered (:func:`nodes.check`);
2. a one-sided entry gives a reason that cites something;
3. where a model class *does* declare ``gndsNodeName``, it agrees with the key —
   the one bridge kept to the model's own naming, three lines;
4. the writer's ``isinstance`` chain really does reach the tag the key claims,
   fixed **behaviourally**: build a minimal instance, call the writer, read the
   tag off the element it produced. Re-listing the chain here would only test
   the copy;
5. the defect list is pinned by count, so each repair has to come through here;
6. the ratchet itself is exercised — planting a broken entry must make the
   checker fail. A trinket nobody has seen fail is not a ratchet.

**And the one that stops the rest from passing in an empty room.** The
registries fill as a side effect of importing the two halves, so every
assertion below is worthless if neither half was imported. ``check()`` imports
them itself, and :func:`test_both_halves_registered` states the count.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from kika.gnds import nodes
from kika.gnds.nodes import NODES, Status, UndeclaredNode


@pytest.fixture(scope="module", autouse=True)
def bothHalvesImported():
    """Nothing here means anything until the decorators have run."""
    import kika.gnds.covariances  # noqa: F401
    import kika.gnds.decode       # noqa: F401
    import kika.gnds.encode       # noqa: F401
    import kika.gnds.primitives   # noqa: F401
    import kika.gnds.styles       # noqa: F401


# ---------------------------------------------------------------------------
# 1-2. the table describes the code
# ---------------------------------------------------------------------------

def test_the_two_halves_agree_with_what_the_table_declares():
    """The invariant. Every failure line names the node and both sides."""
    assert nodes.check() == ()


def test_both_halves_registered():
    """The empty-room guard. A table read by nobody would pass every test above.

    The numbers are deliberately loose — this is not a second copy of the
    table, it is "did the decorators run at all on both sides".
    """
    assert len(nodes._READERS) > 10, "the reader's half did not register"
    assert len(nodes._WRITERS) > 10, "the writer's half did not register"


def test_every_one_sided_entry_says_why():
    """A ``reason`` that cites nothing is how a one-sided entry becomes an
    excuse. Each must name a § of the specification or a ``file:line``."""
    for spec in NODES.values():
        if spec.status is Status.PAIRED:
            continue
        reason = spec.reason or ""
        assert "§" in reason or ".py:" in reason or ".xsd:" in reason, (
            f"{spec.family}/<{spec.tag}> is {spec.status.value} and its reason "
            f"cites neither a § nor a file:line: {reason!r}"
        )


# ---------------------------------------------------------------------------
# 3. the one bridge to the model's own naming
# ---------------------------------------------------------------------------

def test_where_a_class_names_its_node_the_table_agrees():
    """``gndsNodeName`` marks the classes whose node name is *not* the class
    name lower-cased. Where one exists it is authoritative, and the table may
    not contradict it."""
    for spec in NODES.values():
        declared = getattr(spec.cls, "gndsNodeName", None)
        if declared and declared != "?":
            assert declared == spec.tag, f"{spec.cls.__name__} vs {spec.key}"


# ---------------------------------------------------------------------------
# 4. the writer's chain reaches the tag the key claims
# ---------------------------------------------------------------------------

def _minimal(cls):
    """The smallest instance of a model form the writer will accept."""
    from kika.nuclear_data.model import (AngularTwoBody, Constant1d,
                                         CovarianceMatrix, Legendre, Mixed,
                                         Polynomial1d, Reference, Regions1d,
                                         Regions2d, ResonancesWithBackground,
                                         ShortRangeSelfScalingVariance, Sum,
                                         Unspecified, XYs1d, XYs2d)
    builders = {
        XYs1d: lambda: XYs1d(xs=[1.0, 2.0], ys=[3.0, 4.0]),
        Regions1d: Regions1d,
        Constant1d: lambda: Constant1d(constant=1.0, domainMin_=1.0,
                                       domainMax_=2.0),
        Legendre: lambda: Legendre(coefficients=[1.0, 0.5]),
        Polynomial1d: lambda: Polynomial1d(coefficients=[1.0], domainMin_=1.0,
                                           domainMax_=2.0),
        XYs2d: XYs2d,
        Regions2d: Regions2d,
        Reference: lambda: Reference(href="#somewhere"),
        ResonancesWithBackground: ResonancesWithBackground,
        AngularTwoBody: AngularTwoBody,
        Unspecified: Unspecified,
        CovarianceMatrix: lambda: CovarianceMatrix(matrix=np.eye(2)),
        Mixed: Mixed,
        Sum: Sum,
        ShortRangeSelfScalingVariance: lambda: ShortRangeSelfScalingVariance(
            matrix=CovarianceMatrix(matrix=np.eye(2)),
            dependenceOnProcessedGroupWidth="inverse",
        ),
    }
    builder = builders.get(cls)
    return None if builder is None else builder()


#: Which writer to call for a family, and how. Not the chain — the entry point.
def _writeWith(family, form, parent, report):
    from kika.gnds.encode import _covarianceForm, _function

    if family in ("function1d", "function2d"):
        _function(parent, form, report, "test")
    elif family == "covarianceForm":
        _covarianceForm(parent, form, report, "test")
    else:
        raise AssertionError(f"no writer harness for {family}")


@pytest.mark.parametrize("family", ["function1d", "function2d", "covarianceForm"])
def test_the_writer_emits_the_tag_the_key_claims(family):
    """The chain is exercised, not mirrored.

    These three families have a writer entry point that takes a bare form, so
    the round trip class → element → tag can be made in three lines. The form
    families that hang off a ``_Writer`` instance (crossSection, multiplicity,
    distribution) are covered the same way by ``test_encode``'s round trip,
    which walks real files through the same chains.
    """
    from kika.nuclear_data.model import ConversionReport

    checked = 0
    for spec in NODES.values():
        if spec.family != family or spec.status is Status.NEITHER:
            continue
        form = _minimal(spec.cls)
        if form is None:
            continue
        parent = ET.Element("parent")
        _writeWith(family, form, parent, ConversionReport())
        assert [child.tag for child in parent] == [spec.tag], (
            f"a minimal {spec.cls.__name__} was written as "
            f"{[c.tag for c in parent]}, and the table keys it on {spec.tag!r}"
        )
        checked += 1
    assert checked, f"no {family} entry was exercised; the harness went stale"


# ---------------------------------------------------------------------------
# 5. the defects, by count
# ---------------------------------------------------------------------------

def test_the_known_defects_are_pinned_by_count():
    """Four on the day the table landed, three since D19 was fixed. Each repair
    lowers the number and
    this test is what notices — the same shape as
    ``test_the_only_schema_errors_are_the_distributions_phase_7b_will_fill``.

    Two of them are of *attribute*, which the tag table cannot express.
    They are here because writing the table found them, and saying otherwise
    would credit the key-set comparison with a catch it did not make.
    """
    assert len(nodes.KNOWN_DEFECTS) == 3
    assert len(nodes.ONE_SIDED_ATTRIBUTES) == 3
    for defect in nodes.KNOWN_DEFECTS:
        assert defect.where and defect.caughtBy


# ---------------------------------------------------------------------------
# 6. the ratchet, exercised
# ---------------------------------------------------------------------------

def test_a_paired_entry_with_no_writer_fails_the_check():
    """Plant one and watch it fail. Without this the checker could be a no-op
    returning ``()`` and every test above would still pass."""
    from kika.gnds.nodes import NodeSpec

    key = ("distributionForm", "KalbachMann")
    original = NODES[key]
    NODES[key] = NodeSpec(tag=original.tag, family=original.family,
                          section=original.section, cls=original.cls,
                          status=Status.PAIRED)
    try:
        problems = nodes.check()
        assert any("KalbachMann" in problem for problem in problems), problems
    finally:
        NODES[key] = original
    assert nodes.check() == (), "the table was not restored"


def test_registering_an_undeclared_node_fails_at_import_time():
    """The other direction, and the one that makes phase 7b safe: a reader for
    a §18 law that nobody declared cannot be added quietly."""
    assert ("distributionForm", "branching3d") not in NODES
    with pytest.raises(UndeclaredNode, match="branching3d"):
        @nodes.reads("distributionForm", "branching3d")
        def _readBranching3d(element, path):  # pragma: no cover - never runs
            ...


def test_the_derived_dispatch_tables_are_the_registry():
    """``FUNCTION_1D`` and ``COVARIANCE_FORMS`` are views of the table now, not
    a second list beside it. If they drift back to literals this fails."""
    from kika.gnds.covariances import COVARIANCE_FORMS
    from kika.gnds.primitives import FUNCTION_1D, FUNCTION_2D

    assert set(FUNCTION_1D) == {
        spec.tag for spec in NODES.values()
        if spec.family == "function1d" and spec.status is not Status.NEITHER
    }
    assert set(FUNCTION_2D) == {"XYs2d", "regions2d"}
    assert set(COVARIANCE_FORMS) == {
        spec.tag for spec in NODES.values() if spec.family == "covarianceForm"
    }
