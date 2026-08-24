"""MF4 and MF5 are one GNDS node, and this is where the two are put together.

§18.3's ``uncorrelated`` is the evaluation stating P(mu|E) and P(E'|E)
separately — which is exactly what a tape carrying both files does. So the MF5
pass **rewrites** what the MF4 pass left rather than appending beside it, and
three shapes come out of that:

* MF4 and no MF5 — an ``angularTwoBody``, untouched. The commonest by far.
* MF4 and MF5 — an ``uncorrelated`` with both halves read from the file.
* MF5 and no MF4 — an ``uncorrelated`` whose angular half is **inferred**.

The third is the one that needs watching. ENDF says nothing about angle when
MF4 is absent; §5 leaves the emission isotropic in the lab by convention, and
the distributed library agrees (126 095 ``isotropic2d`` against 406 ``XYs2d``).
Agreeing with a convention is not reading a number, so it is reported as an
approximation — a loss is visible, an approximation looks like data.
"""
from __future__ import annotations

import pytest

from kika.endf.model_adapter.decode import _angularHalf
from kika.endf.model_adapter import decodeReactionSuite
from kika.endf.read_endf import read_endf
from kika.nuclear_data.model import (AngularTwoBody, ConversionReport,
                                     EVAL_LABEL, Frame, Isotropic2d, Legendre,
                                     Regions2d, Uncorrelated, XYs2d,
                                     angularAxes)


def _report():
    return ConversionReport()


# ---------------------------------------------------------------------------
# The four rules, one test each
# ---------------------------------------------------------------------------

def test_no_mf4_infers_isotropic_and_says_it_inferred_it():
    report = _report()
    angular, frame = _angularHalf(None, 18, report)

    assert isinstance(angular, Isotropic2d)
    assert frame is Frame.lab
    # An approximation, not a loss and not a warning: the number is in the
    # result and it did not come from the file.
    assert report.losses == [] and report.unsupported == []
    assert len(report.approximations) == 1
    assert "not read but inferred" in report.approximations[0]


def test_an_isotropic_mf4_is_taken_as_stated():
    """LTT=0 gives an `Isotropic2d` directly rather than an `angularTwoBody`
    around one, and it is the file's statement — so nothing is approximated."""
    report = _report()
    stated = Isotropic2d(productFrame=Frame.centerOfMass)
    angular, frame = _angularHalf(stated, 18, report)

    assert angular is stated
    assert frame is Frame.centerOfMass
    assert report.isClean, report.summary()


def test_an_mf4_angular_form_is_unwrapped_not_nested():
    """`uncorrelated/angular` holds the XYs2d itself. Putting the whole
    `angularTwoBody` in there would nest a distribution inside a distribution."""
    report = _report()
    axes = angularAxes()
    inner = XYs2d(function1ds=[Legendre(coefficients=[1.0, 0.1],
                                        outerDomainValue=1.0e6)], axes=axes)
    angular, frame = _angularHalf(
        AngularTwoBody(angular=inner, productFrame=Frame.lab), 91, report)

    assert angular is inner
    assert frame is Frame.lab
    assert report.isClean, report.summary()


def test_an_ltt3_mf4_beside_an_mf5_is_refused_rather_than_flattened():
    """§18.3's choice is isotropic2d, XYs2d, forward or recoil — no regions2d.

    LTT=3 is an MF4 with two TAB2 records, and its model form is a `regions2d`.
    No tape in the reference library states one on an MT that also carries MF5
    (LTT=3 is elastic-shaped and MT2 has no MF5), so this branch is built
    rather than found — which is the reason to pin it: the day a tape does it,
    the choice must be refusal and not a quiet flattening that writes a valid
    file stating something the evaluator did not.
    """
    report = _report()
    axes = angularAxes()
    twoRegions = Regions2d(function2ds=[
        XYs2d(function1ds=[Legendre(coefficients=[1.0], outerDomainValue=1.0)],
              axes=axes, index=0),
        XYs2d(function1ds=[Legendre(coefficients=[1.0], outerDomainValue=2.0)],
              axes=axes, index=1),
    ], axes=axes)
    angular, frame = _angularHalf(
        AngularTwoBody(angular=twoRegions, productFrame=Frame.lab), 91, report)

    assert angular is None and frame is None
    assert len(report.unsupported) == 1
    assert "no regions2d" in report.unsupported[0]


# ---------------------------------------------------------------------------
# The same rules on a real tape
# ---------------------------------------------------------------------------

def test_a_tape_with_mf5_and_no_mf4_builds_the_inferred_uncorrelated(micro_pfns_tape):
    suite, report = decodeReactionSuite(read_endf(str(micro_pfns_tape)))

    reaction = suite.findReactionByENDF_MT(18)
    form = reaction.outputChannel.products.byPid("n")[0].distribution[EVAL_LABEL]
    assert isinstance(form, Uncorrelated)
    assert form.isComplete, "the schema admits no half node"
    assert isinstance(form.angular, Isotropic2d)
    assert isinstance(form.energy, (XYs2d, Regions2d))

    # Not two-body: `twoBody` says the kinematics fix E' from mu, and a section
    # that tabulates P(E'|E) independently is the statement that they do not.
    assert reaction.outputChannel.genre == "NBody"
    assert any("not read but inferred" in line for line in report.approximations)


@pytest.mark.parametrize("tape", ["u235_tape", "pu241_tape"])
def test_an_mt_stating_both_files_becomes_one_uncorrelated(request, tape):
    """Under ``--deep``. U-235 and Pu-241 both state MF4/MT18 and MF5/MT18, so
    the isotropy is **read** and not inferred — and the report must not claim
    otherwise for that MT."""
    endf = read_endf(str(request.getfixturevalue(tape)))
    shared = sorted(set(endf.mf.get(4).mt) & set(endf.mf.get(5).mt))
    assert shared, f"{tape} states no MT in both MF4 and MF5"

    suite, report = decodeReactionSuite(endf)
    for mt in shared:
        reaction = suite.findReactionByENDF_MT(mt)
        if reaction is None:
            continue
        form = reaction.outputChannel.products.byPid("n")[0].distribution[EVAL_LABEL]
        assert isinstance(form, Uncorrelated), f"{tape} MT{mt}"
        assert form.isComplete, f"{tape} MT{mt}"
        assert not [line for line in report.approximations
                    if f"MT{mt}:" in line and "inferred" in line]
