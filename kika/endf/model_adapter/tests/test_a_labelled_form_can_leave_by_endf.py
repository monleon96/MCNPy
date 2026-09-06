"""A form that is not ``eval`` can be written to ENDF, and is named to be.

**The asymmetry this closes.** §9.1's containers hold one form per style label,
which is how §9.3's ``realization`` puts a drawn sample beside the evaluation it
was drawn from rather than on top of it. The GNDS writer walks every form it
finds (``kika/gnds/encode.py``: ``for label, form in crossSection.items()``).
The ENDF encoders named ``EVAL_LABEL`` inline, so the same suite came out
perturbed through the GNDS door and **unperturbed through the ENDF door** —
which is worse than either failing, because a gate on one format then says
nothing about the other.

``docs/library/library-gaps.md`` D29, ``perturbation_model_roadmap.md`` M0.

**What is deliberately not tested here**: a labelled *nu-bar*. ``Multiplicity``
is not a :class:`~kika.nuclear_data.model.component.Component` — §17.3's census
found one form on all 230 562 nodes, so it holds a single ``form`` and has no
label to select on. MF31 therefore has no labelled path yet, and that is a model
question for M5 rather than something to fake here.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.endf.model_adapter import decodeReactionSuite, encodeMF3MT
from kika.endf.read_endf import read_endf
from kika.endf.writers.assemble import encodeTapeSections, writeEndfTape
from kika.nuclear_data.model import EVAL_LABEL

REALIZATION = "realization-0007"


@pytest.fixture
def suite(micro_tape):
    endf = read_endf(str(micro_tape))
    decoded, _ = decodeReactionSuite(endf)
    return decoded


def _scaledCopy(form, factor: float):
    """A second form over the same grid — what an applier produces."""
    from kika.nuclear_data.model import Regions1d

    xs, ys, pairs = form.toEndfRegions()
    return Regions1d.fromEndfRegions(xs.copy(), ys * factor, pairs,
                                     axes=form.axes, label=REALIZATION)


def test_the_default_is_still_eval(suite):
    """The parameter is new; the behaviour without it is not."""
    reaction = suite.reactionByENDF_MT(2)
    reaction.crossSection[REALIZATION] = _scaledCopy(
        reaction.crossSection[EVAL_LABEL], 2.0)

    section, _ = encodeMF3MT(reaction)
    original = suite.reactionByENDF_MT(2).crossSection[EVAL_LABEL].toEndfRegions()[1]
    np.testing.assert_array_equal(section._cross_sections, list(original))


def test_a_named_form_is_the_one_written(suite):
    reaction = suite.reactionByENDF_MT(2)
    evaluated = reaction.crossSection[EVAL_LABEL]
    reaction.crossSection[REALIZATION] = _scaledCopy(evaluated, 2.0)

    section, _ = encodeMF3MT(reaction, label=REALIZATION)

    expected = evaluated.toEndfRegions()[1] * 2.0
    np.testing.assert_allclose(section._cross_sections, expected, rtol=0, atol=0)
    # The grid, the Q values and the header are the reaction's, not the form's.
    assert section.number == 2
    assert section._np == len(expected)


def test_a_missing_label_names_the_ones_that_are_there(suite):
    """The message is the whole value of the check: "no 'eval' form" told a
    caller nothing about what the container actually holds."""
    reaction = suite.reactionByENDF_MT(2)
    with pytest.raises(ValueError, match=r"has no 'realization-0007'"):
        encodeMF3MT(reaction, label=REALIZATION)
    with pytest.raises(ValueError, match=r"it holds \['eval'\]"):
        encodeMF3MT(reaction, label=REALIZATION)


# ---------------------------------------------------------------------------
# Through the tape writer
# ---------------------------------------------------------------------------

def test_a_partly_perturbed_suite_writes_a_whole_tape(suite):
    """The fallback, and why it is not laziness.

    An applier perturbs the reactions it has a covariance for. Refusing to write
    the rest would mean a partially perturbed suite has no tape at all; writing
    only the perturbed ones would mean a tape with holes. So an MT with no form
    under the realization's label is written from ``eval``, which is what it
    still is.
    """
    perturbed = suite.reactionByENDF_MT(2)
    perturbed.crossSection[REALIZATION] = _scaledCopy(
        perturbed.crossSection[EVAL_LABEL], 3.0)

    sections, report, _ = encodeTapeSections(suite, label=REALIZATION)
    byMt = {mt: section for mf, mt, section in sections if mf == 3}

    assert set(byMt) == {mt for mt in (1, 2, 102)}

    # And the tape says it is mixed. A file written from a partly perturbed
    # suite cannot be told from a fully perturbed one by reading it, and for an
    # ensemble that distinction is the traceability -- so the report carries it.
    (line,) = [m for m in report.warnings if REALIZATION in m]
    assert "MT [2] carry it" in line
    assert "MT [1, 102] fell back" in line

    expected = suite.reactionByENDF_MT(2).crossSection[EVAL_LABEL].toEndfRegions()[1]
    np.testing.assert_allclose(byMt[2]._cross_sections, expected * 3.0)

    untouched = suite.reactionByENDF_MT(102).crossSection[EVAL_LABEL].toEndfRegions()[1]
    np.testing.assert_array_equal(byMt[102]._cross_sections, list(untouched))


def test_the_tape_written_from_a_realization_reads_back_perturbed(suite, tmp_path):
    """End to end: the thing a pipeline would actually do."""
    reaction = suite.reactionByENDF_MT(102)
    reaction.crossSection[REALIZATION] = _scaledCopy(
        reaction.crossSection[EVAL_LABEL], 1.5)

    path = tmp_path / "realization.endf"
    writeEndfTape(suite, path, label=REALIZATION)

    again = read_endf(str(path))
    written = np.asarray(again.mf[3].mt[102].cross_sections, dtype=float)
    original = np.asarray(
        reaction.crossSection[EVAL_LABEL].toEndfRegions()[1], dtype=float)

    # ENDF stores six significant figures, so the tape cannot carry more than
    # that back -- this is the resolution of the format, not of the arithmetic.
    np.testing.assert_allclose(written, original * 1.5, rtol=1e-6)

    # And MT2, which nobody perturbed, came back as itself.
    untouched = np.asarray(again.mf[3].mt[2].cross_sections, dtype=float)
    reference = np.asarray(
        suite.reactionByENDF_MT(2).crossSection[EVAL_LABEL].toEndfRegions()[1],
        dtype=float)
    np.testing.assert_allclose(untouched, reference, rtol=1e-6)


def test_a_tape_written_without_a_label_says_nothing_extra(suite):
    """The declaration is about the realization, so `eval` writes no line.

    Otherwise every ordinary tape would grow a warning that carries no
    information, and a report whose warnings are noise stops being read.
    """
    _sections, report, _ = encodeTapeSections(suite)
    assert not [m for m in report.warnings if "fell back" in m]


def test_a_fully_perturbed_suite_still_declares_itself(suite):
    """"0 fell back" is a claim; its absence is not.

    A run that perturbed everything and a run that perturbed nothing must not
    produce the same report, so the line is written either way.
    """
    for mt in (1, 2, 102):
        reaction = suite.reactionByENDF_MT(mt)
        reaction.crossSection[REALIZATION] = _scaledCopy(
            reaction.crossSection[EVAL_LABEL], 1.1)

    _sections, report, _ = encodeTapeSections(suite, label=REALIZATION)
    (line,) = [m for m in report.warnings if REALIZATION in m]
    assert "MT [1, 2, 102] carry it" in line
    assert "MT none fell back" in line
