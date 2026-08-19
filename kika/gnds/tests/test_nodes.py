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
                                         Unspecified, XYs1d, XYs2d, XYs3d)
    from kika.nuclear_data.model import (AverageParameterCovariance, DataLink,
                                         ParameterCovariance,
                                         ParameterCovarianceMatrix,
                                         ParameterLink)
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
        XYs3d: XYs3d,
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
        ParameterCovariance: lambda: ParameterCovariance(
            label="p", rowData=DataLink(href="#row"),
            form=ParameterCovarianceMatrix(
                matrix=np.eye(2), label="eval",
                parameters=[ParameterLink(label="l", href="#p",
                                          nParameters=2)],
            ),
        ),
        AverageParameterCovariance: lambda: AverageParameterCovariance(
            label="a", rowData=DataLink(href="#row"),
            form=CovarianceMatrix(matrix=np.eye(2)),
        ),
    }
    builder = builders.get(cls)
    return None if builder is None else builder()


#: Which writer to call for a family, and how. Not the chain — the entry point.
def _writeWith(family, form, parent, report):
    from kika.gnds.encode import (_covarianceForm, _function,
                                  _parameterCovariances)

    if family in ("function1d", "function2d", "function3d"):
        _function(parent, form, report, "test")
    elif family == "covarianceForm":
        _covarianceForm(parent, form, report, "test")
    elif family == "parameterCovarianceForm":
        # §25.3's entry point takes the whole list and emits the container the
        # sequence lives in, so the tag the key names is a *grandchild*. The
        # container is unwrapped rather than the inner writers being called
        # directly, so that what this exercises is still the registered
        # function.
        outer = ET.Element("outer")
        _parameterCovariances(outer, [form], report)
        container = outer.find("parameterCovariances")
        for child in ([] if container is None else list(container)):
            parent.append(child)
    else:
        raise AssertionError(f"no writer harness for {family}")


@pytest.mark.parametrize("family", ["function1d", "function2d", "function3d",
                                    "covarianceForm",
                                    "parameterCovarianceForm"])
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
    """Four on the day the table landed, one now: D19 went with the isotropic2d
    fix, and the two R-matrix flags and the nested PoPs went with the commit
    after that. What is left is ``supportsAngularReconstruction``, read and
    reported and never written. Each repair lowers the number and this test is
    what notices — the same shape as
    ``test_the_only_schema_errors_are_the_distributions_phase_7b_will_fill``.

    The survivor is of *attribute*, which the tag table cannot express. It is
    here because writing the table found it, and saying otherwise would credit
    the key-set comparison with a catch it did not make.
    """
    assert len(nodes.KNOWN_DEFECTS) == 1
    assert len(nodes.ONE_SIDED_ATTRIBUTES) == 1
    for defect in nodes.KNOWN_DEFECTS:
        assert defect.where and defect.caughtBy


# ---------------------------------------------------------------------------
# 6. the ratchet, exercised
# ---------------------------------------------------------------------------

def test_a_paired_entry_with_no_writer_fails_the_check():
    """Plant one and watch it fail. Without this the checker could be a no-op
    returning ``()`` and every test above would still pass.

    **The victim has to be an entry that will never be implemented**, or this
    test goes quietly vacuous the day it is: planting ``PAIRED`` on something
    already ``PAIRED`` changes nothing and the assert below stops testing the
    checker. It used to be ``("distributionForm", "KalbachMann")`` and moved
    when §18.6 landed.

    ``("function3d", "regions3d")`` is immune, and not merely unimplemented.
    ``gnds.xsd`` gives ``regions3d`` **no element declaration anywhere**, and
    the two complexTypes that would reach one — ``function3ds_inRegions`` and
    ``xData_regions_3d_primary`` — are referenced only by each other, so the
    branch is unreachable from any document. The census agrees from the other
    end: **0 occurrences in 2 950 ``XYs3d`` across 558 evaluations**. A schema
    argument and a measurement, which is what makes it safe to build a test on.
    """
    from kika.gnds.nodes import NodeSpec

    key = ("function3d", "regions3d")
    original = NODES[key]
    assert original.status is Status.NEITHER, (
        f"{key} was chosen as the victim because it can never be implemented; "
        f"it is now {original.status.value}, so this test is measuring nothing"
    )
    NODES[key] = NodeSpec(tag=original.tag, family=original.family,
                          section=original.section, cls=original.cls,
                          status=Status.PAIRED)
    try:
        problems = nodes.check()
        assert any("regions3d" in problem for problem in problems), problems
    finally:
        NODES[key] = original
    assert nodes.check() == (), "the table was not restored"


def test_every_unimplemented_distribution_is_placed_at_a_choice_point():
    """The model's list of unimplemented distributions is not a list of §18 laws.

    One of its two names is not a member of §18.1.1's choice at all —
    ``recoil`` belongs to ``angularTwoBody`` (``gnds.xsd:1670``), where it is
    already read and written — and ``NBodyPhaseSpace``, which
    :data:`nodes.NOT_A_DISTRIBUTION_FORM` also places, is an §18.3 *energy* form
    (``gnds.xsd:1703``). Both read like laws phase 7b owes, and treating them as
    such would mean adding a ``distributionForm`` entry the schema does not
    admit — the same class of mistake as the ``isotropic2d`` the writer used to
    emit under ``distribution``.

    So each name is either an entry here or explicitly placed elsewhere, and
    this is what stops a third state — named in the model, absent from both.
    """
    from kika.nuclear_data.model.distributions import \
        NOT_IMPLEMENTED_DISTRIBUTIONS

    # Down to two: `uncorrelated`, `energyAngular` and `angularEnergy` have all
    # landed, and what is left is `KalbachMann` plus the misplaced `recoil`. The
    # assert is what stops this loop from passing on an empty dict the day the
    # list finally empties, which would read as "all placed" and mean "none
    # checked".
    assert NOT_IMPLEMENTED_DISTRIBUTIONS
    for name in NOT_IMPLEMENTED_DISTRIBUTIONS:
        elsewhere = nodes.NOT_A_DISTRIBUTION_FORM.get(name)
        if elsewhere is not None:
            assert ("distributionForm", name) not in NODES, (
                f"{name} is declared not to be a distribution form and has an "
                f"entry as one"
            )
            assert "gnds.xsd:" in elsewhere, (
                f"{name}'s placement cites no schema line, so it is an opinion"
            )
            continue
        assert ("distributionForm", name) in NODES, (
            f"{name} is declared unimplemented in the model and appears at no "
            f"choice point — add the entry, or say where it really belongs in "
            f"nodes.NOT_A_DISTRIBUTION_FORM"
        )


def test_the_misplaced_names_stay_misplaced_after_they_are_implemented():
    """The loop above visits only names still in the model's unimplemented list,
    so a placement stops being checked the moment its node lands — which is
    exactly when the mistake becomes tempting again. ``NBodyPhaseSpace`` is
    PAIRED now, under ``uncorrelatedEnergyForm``; the one thing that must never
    happen is it acquiring a ``distributionForm`` entry as well."""
    assert nodes.NOT_A_DISTRIBUTION_FORM
    for name, elsewhere in nodes.NOT_A_DISTRIBUTION_FORM.items():
        assert ("distributionForm", name) not in NODES, (
            f"{name} is declared not to be a distribution form and has an "
            f"entry as one"
        )
        assert "gnds.xsd:" in elsewhere, (
            f"{name}'s placement cites no schema line, so it is an opinion"
        )


def test_registering_an_undeclared_node_fails_at_import_time():
    """The other direction, and the one that made phase 7b safe: a reader for
    a §18 law that nobody declared cannot be added quietly.

    The guinea pig was ``branching3d`` until phase 7b implemented it, which is
    the good reason for a victim to expire. ``coherentPhotonScattering`` takes
    over: it is a real §18.1.1 member (``gnds.xsd:1657``) and the census counted
    it **zero times** in the 558 distributed neutron evaluations, along with the
    four other unnamed members. A node with no occurrences is not going to
    acquire a reader by accident, which is exactly what a victim needs to be.
    """
    victim = ("distributionForm", "coherentPhotonScattering")
    assert victim not in NODES
    with pytest.raises(UndeclaredNode, match="coherentPhotonScattering"):
        @nodes.reads(*victim)
        def _readCoherentPhotonScattering(element, path):  # pragma: no cover
            ...


def test_the_derived_dispatch_tables_are_the_registry():
    """``FUNCTION_1D`` and ``COVARIANCE_FORMS`` are views of the table now, not
    a second list beside it. If they drift back to literals this fails."""
    from kika.gnds.covariances import (COVARIANCE_FORMS,
                                       PARAMETER_COVARIANCE_FORMS)
    from kika.gnds.primitives import FUNCTION_1D, FUNCTION_2D, FUNCTION_3D

    for table, family in ((FUNCTION_1D, "function1d"),
                          (FUNCTION_2D, "function2d"),
                          (FUNCTION_3D, "function3d")):
        assert set(table) == {
            spec.tag for spec in NODES.values()
            if spec.family == family and spec.status is not Status.NEITHER
        }, family
    assert set(COVARIANCE_FORMS) == {
        spec.tag for spec in NODES.values() if spec.family == "covarianceForm"
    }
    assert set(PARAMETER_COVARIANCE_FORMS) == {
        spec.tag for spec in NODES.values()
        if spec.family == "parameterCovarianceForm"
    }
