"""One request, applied to evaluations that do not all state the same things.

`collectEntries` raises when a selection matches nothing, and that is right for
a request written against a known file. It is the wrong answer for a request
written against a *directory* of them: "cross sections and angular
distributions" is a reasonable thing to say about a set of evaluations of which
only some state MF34, and it is what a builder that cannot open the files (they
are on a cluster) has to be able to say.

`onMissing="skip"` is that mode. The gate is that it is **never quiet**: what
was dropped reaches the log, the notes and the metadata, so an ensemble can
always be asked what was actually perturbed in it.
"""
from pathlib import Path

import pytest

from kika.sampling.joint_blocks import Selection, pruneRequest
from kika.sampling.model_perturbation import perturbFromModel

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
FE56 = DATA / "micro_fe56_xs_and_angular.endf"   # MF33 + MF34, no MF31/MF35
CF252 = DATA / "micro_cf252_pfns.endf"           # MF35 only
U235 = DATA / "micro_u235_nubar.endf"            # MF31 only

#: Everything the four quantities can be. The point of the generic request.
EVERYTHING = {33: None, 34: None, 31: None, 35: None}

pytestmark = pytest.mark.skipif(not FE56.is_file(), reason=f"{FE56} missing")


def _covariances(tape):
    from kika.endf import read_endf
    from kika.endf.model_adapter import decodeCovarianceSuite

    suite, _report = decodeCovarianceSuite(read_endf(str(tape)))
    return suite


# ---------------------------------------------------------------------------
# pruneRequest
# ---------------------------------------------------------------------------

def test_pruning_keeps_what_the_file_states_and_reports_the_rest():
    kept, dropped = pruneRequest(_covariances(FE56), EVERYTHING)
    assert sorted(s.mf for s in kept) == [33, 34]
    assert sorted(s.mf for s, _reason in dropped) == [31, 35]
    # The reason travels with the selection: it is what the run records.
    assert all(reason for _s, reason in dropped)


def test_pruning_a_request_the_file_fully_serves_drops_nothing():
    kept, dropped = pruneRequest(_covariances(FE56), {33: None, 34: None})
    assert sorted(s.mf for s in kept) == [33, 34]
    assert dropped == []


def test_pruning_is_per_selection_and_not_per_quantity():
    """An MT the file does not state is dropped like a missing quantity."""
    kept, dropped = pruneRequest(
        _covariances(FE56), {33: {"mt": [2]}, 34: {"mt": [16]}})
    assert [s.mf for s in kept] == [33]
    assert [s.mf for s, _r in dropped] == [34]


def test_pruning_a_file_that_states_only_one_thing():
    kept, dropped = pruneRequest(_covariances(CF252), EVERYTHING)
    assert [s.mf for s in kept] == [35]
    assert sorted(s.mf for s, _r in dropped) == [31, 33, 34]


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def test_the_default_still_refuses_what_the_file_does_not_state():
    """Unchanged. A request against a known file that misses is a mistake."""
    with pytest.raises(ValueError, match="states no section"):
        perturbFromModel(FE56, {35: None}, 1, seed=1, dryRun=True)


def test_skip_perturbs_what_is_there_and_says_what_it_did_not():
    run = perturbFromModel(FE56, EVERYTHING, 2, seed=3, dryRun=True,
                           onMissing="skip")

    perturbed = {key.mf for key in run.samples[0]["set"].components()}
    assert perturbed == {33, 34}

    # Every dropped quantity is a note on the result...
    joined = " ".join(run.notes)
    assert "multiplicity" in joined and "energyDistribution" in joined
    # ...and a warning in the log, so a person reading the report sees it.
    warned = [e for e in run.log.problems() if "was not perturbed" in e.message]
    assert len(warned) == 2


def test_skip_leaves_a_fully_served_request_exactly_as_it_was():
    """Turning the mode on must not move a run that needed nothing skipped."""
    import numpy as np

    strict = perturbFromModel(FE56, {33: None, 34: None}, 3, seed=9, dryRun=True)
    lenient = perturbFromModel(FE56, {33: None, 34: None}, 3, seed=9,
                               dryRun=True, onMissing="skip")

    left = strict.samples[0]["set"].factors
    right = lenient.samples[0]["set"].factors
    assert set(left) == set(right)
    for key in left:
        np.testing.assert_array_equal(left[key], right[key], err_msg=str(key))
    assert lenient.notes == strict.notes


def test_a_request_where_nothing_matches_is_still_an_error():
    """Skipping everything is a run with no perturbation in it."""
    with pytest.raises(ValueError, match="nothing to perturb"):
        perturbFromModel(FE56, {31: None, 35: None}, 1, seed=1, dryRun=True,
                         onMissing="skip")


@pytest.mark.skipif(not CF252.is_file(), reason=f"{CF252} missing")
def test_the_same_generic_request_suits_three_different_evaluations():
    """The case the mode exists for: one request, whatever each file holds."""
    got = {}
    for tape in (FE56, CF252, U235):
        if not tape.is_file():
            continue
        run = perturbFromModel(tape, EVERYTHING, 1, seed=1, dryRun=True,
                               onMissing="skip")
        got[tape.name] = {key.mf for key in run.samples[0]["set"].components()}

    assert got[FE56.name] == {33, 34}
    assert got[CF252.name] == {35}
    if U235.is_file():
        assert got[U235.name] == {31}


def test_an_unknown_mode_is_refused_before_the_tape_is_read():
    with pytest.raises(ValueError, match="onMissing"):
        perturbFromModel(FE56, {33: None}, 1, dryRun=True, onMissing="ignore")


# ---------------------------------------------------------------------------
# Naming reactions a file does not state
# ---------------------------------------------------------------------------

def test_a_partly_matching_reaction_list_perturbs_what_is_there_and_says_so():
    """The case that used to pass in silence.

    `collectEntries` only refuses when a selection matches *nothing*, so asking
    for MT2 and MT16 of a file that states only MT2 perturbed one of the two
    and reported it nowhere. The ensemble then disagreed with its own metadata
    about what had been varied in it.
    """
    run = perturbFromModel(FE56, {33: {"mt": [2, 16, 102]}}, 2, seed=5,
                           dryRun=True, onMissing="skip")

    perturbed = {key.mt for key in run.samples[0]["set"].components()}
    assert 2 in perturbed

    absent = sorted({16, 102} - perturbed)
    if absent:
        joined = " ".join(run.notes)
        for mt in absent:
            assert f"MT{mt}" in joined, f"MT{mt} was dropped without a note"
        warned = [e for e in run.log.problems() if "not perturbed" in e.message]
        assert warned


def test_a_reaction_list_the_file_fully_states_leaves_no_note():
    run = perturbFromModel(FE56, {33: {"mt": [2]}}, 1, seed=5, dryRun=True,
                           onMissing="skip")
    assert not [n for n in run.notes if "not perturbed" in n]


def test_missing_reactions_are_reported_under_raise_too():
    """A partial match is not a failure, so `raise` does not refuse it -- but
    it is still something the run has to record."""
    run = perturbFromModel(FE56, {33: {"mt": [2, 16]}}, 1, seed=5, dryRun=True)
    perturbed = {key.mt for key in run.samples[0]["set"].components()}
    if 16 not in perturbed:
        assert any("MT16" in note for note in run.notes)
