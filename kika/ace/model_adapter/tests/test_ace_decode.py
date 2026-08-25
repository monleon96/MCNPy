"""ACE → ``reactionSuite``: the numbers, the style chain, and the Q values.

No fixture is committed — a real ACE file is 16 MB at best and truncating one
means recomputing the internal pointer blocks, which would make the fixture
kika's arithmetic rather than the library's. Everything here is ``tape``-marked
and runs against the shared tree.

**The sharpest test in this file is about Q values**, because it contradicts
what the roadmap recorded. Phase 1 concluded "ACE lacks ``qm``/``qi``" and made
``CrossSection.to_endf()`` raise for an ACE-sourced section on that basis. The
raise is right; the reason was not. ACE's LQR block carries one Q per reaction,
in MeV, positionally aligned with MTR — and reading it back gives Fe-56 capture
+7.646 MeV, which is the right answer. What ACE has no counterpart for is **QM**
and **LR**. So the defect was never the format's; it was ``from_ace`` not
reading a block that is present.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.ace import read_ace
from kika.ace.model_adapter import (GRIDDED_LABEL, HEATED_LABEL, URR_LABEL,
                                    decodeAce, qValuesByMT)
from kika.nuclear_data import CrossSection
from kika.nuclear_data.model import (Evaluated, GriddedCrossSection, Heated,
                                     URR_probabilityTables1d, XYs1d)

pytestmark = pytest.mark.tape


@pytest.fixture(scope="module")
def ace(fe56_ace):
    return read_ace(str(fe56_ace))


@pytest.fixture(scope="module")
def decoded(ace):
    return decodeAce(ace)


# ---------------------------------------------------------------------------
# The numbers
# ---------------------------------------------------------------------------

def test_every_reaction_reproduces_the_flat_path_array_for_array(ace, decoded):
    """``assert_array_equal``, not ``allclose``. The adapter re-expresses; the
    moment it starts recomputing, this is the test that says so."""
    suite, _ = decoded
    assert len(suite.reactions) > 0

    for reaction in suite.reactions:
        mt = reaction.id.ENDF_MT
        expected = CrossSection.from_ace(ace, mt)
        form = reaction.crossSection[GRIDDED_LABEL]

        np.testing.assert_array_equal(form.xs, expected.energies)
        np.testing.assert_array_equal(form.ys, expected.values)


def test_the_composite_mts_are_present(ace, decoded):
    """MT4, MT101 and friends are summed on demand, and are real cross sections.

    Losing them would make the suite quietly smaller than the file's own view of
    itself, which is the failure this whole increment is written against.
    """
    from kika._constants import MT_COMPOSITE_ORDER

    suite, _ = decoded
    decodedMTs = {r.id.ENDF_MT for r in suite.reactions}
    available = {mt for mt in MT_COMPOSITE_ORDER
                 if ace.cross_section._get_or_compute_reaction(mt) is not None}
    assert available <= decodedMTs, f"composites lost: {sorted(available - decodedMTs)}"


def test_energies_are_in_ev_not_mev(decoded):
    """ACE is MeV throughout and the model is eV, as ENDF is. Getting this wrong
    is a factor of a million that still plots as a plausible curve."""
    suite, _ = decoded
    form = suite.reactionByENDF_MT(2).crossSection[GRIDDED_LABEL]
    assert form.domainMax > 1.0e6


def test_the_form_is_an_xys1d_and_not_a_single_region_regions1d(decoded):
    """ACE has no ``(NBT, INT)`` regions: it is lin-lin on one union grid. A
    one-region ``regions1d`` would assert a structure the file does not have."""
    suite, _ = decoded
    form = suite.reactionByENDF_MT(2).crossSection[GRIDDED_LABEL]
    assert isinstance(form, XYs1d)
    assert form.interpolation.value == "lin-lin"


# ---------------------------------------------------------------------------
# The processed styles — what makes ACE not an evaluation
# ---------------------------------------------------------------------------

def test_the_derived_from_chain_resolves_to_an_evaluated_style(decoded):
    suite, _ = decoded
    chain = [s.label for s in suite.styles.chain(GRIDDED_LABEL)]
    assert chain == [GRIDDED_LABEL, HEATED_LABEL, "eval"]
    assert isinstance(suite.styles.evaluatedFor(GRIDDED_LABEL), Evaluated)


def test_the_cross_sections_are_gridded_and_never_evaluated(decoded):
    """The whole point. An ACE file contains no evaluated data, so nothing in it
    may be labelled ``eval`` — otherwise a processed representation starts
    claiming to be the evaluation it came from."""
    suite, _ = decoded
    for reaction in suite.reactions:
        assert "eval" not in reaction.crossSection.forms
        assert GRIDDED_LABEL in reaction.crossSection.forms


def test_the_heated_style_carries_the_temperature_in_kelvin(ace, decoded):
    """ACE's header field is kT in **MeV**; 2.53e-08 is 293.6 K, not 2.53e-08 K.

    ``Header.temperature``'s inline comment says kelvin and its class docstring
    says MeV. They cannot both be right, and the docstring is.
    """
    from kika._constants import BOLTZMANN_CONSTANT

    suite, _ = decoded
    heated = next(s for s in suite.styles if isinstance(s, Heated))
    assert heated.derivedFrom == "eval"
    assert heated.temperature.unit == "K"
    assert heated.temperature.value == pytest.approx(
        ace.header.temperature / BOLTZMANN_CONSTANT
    )
    assert 250.0 < heated.temperature.value < 350.0


def test_a_reconstructed_style_is_not_invented(decoded):
    """A continuous-energy ACE has almost certainly been through RECONR. The
    file does not say so, and a style chain is a claim about provenance."""
    from kika.nuclear_data.model import CrossSectionReconstructed

    suite, _ = decoded
    assert not any(isinstance(s, CrossSectionReconstructed) for s in suite.styles)


def test_the_urr_tables_are_declared_where_they_exist(ace, decoded):
    suite, report = decoded
    if not getattr(ace.unresolved_resonance, "has_data", False):
        pytest.skip("this ACE file has no URR probability tables")

    assert any(s.label == URR_LABEL for s in suite.styles)
    assert suite.styles.chain(URR_LABEL)[1].label == GRIDDED_LABEL

    form = suite.reactionByENDF_MT(102).crossSection[URR_LABEL]
    assert isinstance(form, URR_probabilityTables1d)
    assert form.href
    # Declared, not converted — and the report has to say which.
    assert any("URR probability tables" in entry for entry in report.unsupported)


def test_the_gridded_style_exists_and_says_what_it_is(decoded):
    suite, _ = decoded
    gridded = next(s for s in suite.styles if isinstance(s, GriddedCrossSection))
    assert gridded.derivedFrom == HEATED_LABEL


# ---------------------------------------------------------------------------
# Q values — the finding
# ---------------------------------------------------------------------------

def test_ace_does_carry_q_values_and_they_are_right(ace):
    """Three reactions with independently known Q values, from the LQR block."""
    byMT = qValuesByMT(ace)
    assert byMT, "the LQR block was not read at all"

    # Fe-56: capture is exothermic, (n,2n) and (n,p) are not.
    assert byMT[102] == pytest.approx(7.64617e6, rel=1e-6)
    assert byMT[16] == pytest.approx(-11.19706e6, rel=1e-6)
    assert byMT[103] == pytest.approx(-2.91315e6, rel=1e-6)


def test_elastic_q_is_zero_because_it_is_known_and_not_because_it_defaulted(decoded):
    """MT2 is absent from MTR. Elastic Q is 0 by definition — that is knowledge,
    and ``isKnown`` has to distinguish it from a missing value filled with 0."""
    suite, _ = decoded
    q = suite.reactionByENDF_MT(2).outputChannel.Q
    assert q.value == 0.0
    assert q.isKnown is True


def test_a_composite_has_no_q_and_says_so(decoded):
    """MT4 is a sum over inelastic levels with different Q values, so it has
    none — which is a different statement from Q = 0."""
    suite, _ = decoded
    reaction = suite.findReactionByENDF_MT(4)
    if reaction is None:
        pytest.skip("this ACE file exposes no MT4")
    assert reaction.outputChannel.Q.value is None
    assert reaction.outputChannel.Q.isKnown is False


def test_the_flat_path_reads_the_same_q_values_this_decoder_does(ace):
    """The gap this decoder recorded, now closed on the flat side too.

    ``CrossSection.from_ace`` used to build six ACE metadata keys and no Q at
    all, with ``ace.q_values`` right there — ``docs/library/library-gaps.md`` D4. It
    calls ``qValuesByMT`` now, so there is one alignment convention rather than
    two, and this asserts the two paths agree rather than merely that the flat
    one is non-empty. Agreement is the property that matters: a second copy of
    a positional convention is exactly what drifts.
    """
    assert ace.q_values.has_q_values, (
        "the premise of this test is gone: the file has no LQR block"
    )
    expected = qValuesByMT(ace)

    flat = CrossSection.from_ace(ace, 102)
    assert flat.metadata["qi"] == expected[102]
    # QM is the mass-difference Q and has no ACE counterpart. Still absent, and
    # that is the reason `to_endf` still refuses on an ACE-sourced section.
    assert "qm" not in flat.metadata


def test_the_flat_path_gives_elastic_its_defined_zero(ace):
    """MT 2 is absent from MTR because its Q is zero by definition.

    Filling it in is knowledge, not a default — the same call the decoder
    makes, so the flat path cannot disagree with it.
    """
    assert CrossSection.from_ace(ace, 2).metadata["qi"] == 0.0


def test_a_composite_has_no_qi_on_the_flat_path_either(ace):
    """MT 4 sums levels with different Q values, so it has none.

    Absent rather than zero: the same statement the model makes with
    ``Q.isKnown is False``, made in the only way an untyped dict can make it.
    """
    flat = CrossSection.from_ace(ace, 4)
    assert "qi" not in flat.metadata


def test_the_report_names_qm_and_lr_rather_than_claiming_ace_has_no_q(decoded):
    """The wording matters: the previous formulation was false."""
    _, report = decoded
    text = " ".join(report.unsupported)
    assert "QM" in text and "LR" in text
    assert "QI *is* present" in text


def test_the_report_declares_the_absent_evaluation(decoded):
    """The ``eval`` style is a declared placeholder with nothing behind it."""
    _, report = decoded
    assert any("evaluation it does not contain" in entry for entry in report.losses)


def test_the_report_declares_the_absence_of_covariances(decoded):
    _, report = decoded
    assert any("no covariances" in entry for entry in report.unsupported)
