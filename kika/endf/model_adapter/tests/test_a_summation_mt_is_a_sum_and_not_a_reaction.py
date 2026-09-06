"""§21.1: an MF3 section the file also gives the partials of is a ``sum``.

D7 of ``docs/library/perturbation_model_roadmap.md``. The adapter used to put
every MF3 section in ``reactions`` and leave ``sums`` empty, so the model
*could* express the redundant accounting and did not carry it. That is a gap
the perturbation track walks into: MT1 and MT2 are not two independent
quantities, and drawing them as if they were double-counts the elastic.

**The decision is derived, not read.** ENDF-6 does not mark a summation MT.
MT103 is a sum in an evaluation that also writes MT600-649 and an ordinary
reaction in one that does not -- the same number, two different things,
distinguished only by what the file carries beside it. So these tests pin both
directions on the same MT.

The rule itself lives in :data:`kika._constants.MF3_SUM_RULES`, shared with
:mod:`kika.endf.writers.redundant`, which rebuilds these sections after an
edit. One table, so the two cannot disagree about what MT4 is made of.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kika.endf import read_endf
from kika.endf.model_adapter import decodeReactionSuite
from kika.endf.model_adapter.decode import _summationMTs

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

def test_the_total_is_a_sum_whenever_the_elastic_is_there():
    assert _summationMTs({1, 2, 102}) == {1}


def test_the_same_mt_is_a_sum_or_a_reaction_depending_on_the_file():
    """MT103 with its levels is redundant; MT103 alone is the reaction."""
    withLevels = {2, 103, 600, 601, 602}
    alone = {2, 103}

    assert 103 in _summationMTs(withLevels)
    assert 103 not in _summationMTs(alone)


def test_a_file_with_no_partials_has_no_sums():
    """A tape cut down to a total and nothing else states a reaction, not a sum.

    There is nothing for MT1 to be the sum *of* here, and calling it a sum would
    invite a re-derivation from an empty set.
    """
    assert _summationMTs({1}) == set()


def test_the_inelastic_total_is_a_sum_of_its_levels_and_not_of_mt50():
    """MT50 is (n,n0) -- the elastic under another name -- and ENDF sums 51-91."""
    assert 4 in _summationMTs({2, 4, 50, 51, 52})
    assert 50 not in _summationMTs({2, 4, 50, 51, 52})


# ---------------------------------------------------------------------------
# On a real tape
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def decoded():
    endf = read_endf(str(DATA / "micro_fe56_structural.endf"))
    suite, _report = decodeReactionSuite(endf)
    return endf, suite


def test_the_total_goes_to_sums_and_the_rest_to_reactions(decoded):
    _endf, suite = decoded
    assert [r.ENDF_MT for r in suite.sums] == [1]
    assert suite.reactions.ENDF_MTs == [2, 102]


def test_every_section_still_reaches_the_model(decoded):
    """The split moves sections, it does not drop any."""
    endf, suite = decoded
    placed = {r.ENDF_MT for r in suite.reactions} | {r.ENDF_MT for r in suite.sums}
    assert placed == set(endf.mf[3].mt)


def test_looking_an_mt_up_by_number_does_not_notice_the_move(decoded):
    """``reactionByENDF_MT`` searches both, which is why nothing else changed."""
    _endf, suite = decoded
    total = suite.reactionByENDF_MT(1)
    assert total is not None and total.ENDF_MT == 1


def test_the_written_tape_still_carries_the_summation_section(decoded, tmp_path):
    """The regression this change could plausibly have caused, pinned.

    The MF3 writer iterated ``suite.reactions``. Moving MT1 out of that list
    without teaching the writer would have silently dropped the total from every
    tape kika writes -- a valid file, missing a section, with nothing to say so.
    """
    from kika.endf.writers.assemble import writeEndfTape

    _endf, suite = decoded
    out = tmp_path / "written.endf"
    writeEndfTape(suite, out)

    written = read_endf(str(out))
    assert sorted(written.mf[3].mt) == [1, 2, 102]
