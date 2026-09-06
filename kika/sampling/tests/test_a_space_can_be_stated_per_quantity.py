"""``space`` per quantity, and the ladder it must not disturb.

A cross section is a positive magnitude, so a relative covariance of it is
drawn in log space and its factor stays positive. A Legendre coefficient is
signed: an evaluation stating one near zero states an uncertainty that
straddles zero, which a log draw cannot represent. So one run has to be able to
draw the two in different spaces.

The gate that matters is not that the option exists. It is that turning it on
for a quantity that was *already* getting that space moves nothing: block *i*
keeps the seed it had, so a run that asks for ``{33: "log"}`` draws exactly
what ``space="log"`` drew.
"""
from pathlib import Path

import numpy as np
import pytest

from kika.sampling.model_perturbation import perturbFromModel, resolveSpaces

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
#: Fe-56 with MF3, MF4, MF33 and MF34 of MT2 on one tape.
TAPE = DATA / "micro_fe56_xs_and_angular.endf"
BOTH = {33: None, 34: {"mt": [2], "index": [1, 2, 3]}}

pytestmark = pytest.mark.skipif(not TAPE.is_file(), reason=f"{TAPE} missing")


def _factors(run):
    """``{component: array}`` of sample 0, keyed by ``ComponentKey``."""
    return dict(run.samples[0]["set"].factors)


def _byMf(run, mf):
    return {key: value for key, value in _factors(run).items() if key.mf == mf}


# ---------------------------------------------------------------------------
# resolveSpaces
# ---------------------------------------------------------------------------

def test_a_bare_name_applies_to_every_block():
    blocks = [("a", None), ("b", None)]
    assert resolveSpaces(blocks, {}, "log") == {"a": "log", "b": "log"}
    assert resolveSpaces(blocks, {}, "linear") == {"a": "linear", "b": "linear"}


def test_both_spellings_of_a_mapping_mean_the_same_thing():
    class Key:
        def __init__(self, mf):
            self.mf = mf

    index = {"x": {"components": [Key(33)]}, "y": {"components": [Key(34)]}}
    blocks = [("x", None), ("y", None)]
    byNumber = resolveSpaces(blocks, index, {33: "log", 34: "linear"})
    byName = resolveSpaces(
        blocks, index,
        {"crossSection": "log", "angularDistribution": "linear"})
    assert byNumber == byName == {"x": "log", "y": "linear"}


def test_a_quantity_the_mapping_does_not_name_falls_back_to_log():
    class Key:
        def __init__(self, mf):
            self.mf = mf

    index = {"y": {"components": [Key(34)]}}
    assert resolveSpaces([("y", None)], index, {33: "linear"}) == {"y": "log"}


def test_a_space_that_is_not_one_is_refused_by_name():
    with pytest.raises(ValueError, match="log"):
        resolveSpaces([], {}, "lognormal")
    with pytest.raises(ValueError, match="log"):
        resolveSpaces([], {}, {33: "gaussian"})
    with pytest.raises(ValueError, match="neither an MF number"):
        resolveSpaces([], {}, {"crossSections": "log"})


def test_a_bad_space_is_refused_before_the_tape_is_read():
    """A typo in a keyword should not cost a parse of a 27 MB evaluation."""
    with pytest.raises(ValueError, match="log"):
        perturbFromModel(TAPE, BOTH, 1, space="lognormal", dryRun=True)


# ---------------------------------------------------------------------------
# The draw
# ---------------------------------------------------------------------------

def test_naming_the_space_a_quantity_already_had_moves_nothing():
    """The gate. Same numbers, to the bit, through a different code path."""
    plain = perturbFromModel(TAPE, BOTH, 3, seed=20260906, space="log",
                             dryRun=True)
    mapped = perturbFromModel(TAPE, BOTH, 3, seed=20260906,
                              space={33: "log", 34: "log"}, dryRun=True)

    left, right = _factors(plain), _factors(mapped)
    assert set(left) == set(right)
    for key in left:
        np.testing.assert_array_equal(left[key], right[key], err_msg=str(key))


def test_a_mixed_request_draws_each_quantity_in_its_own_space():
    run = perturbFromModel(TAPE, BOTH, 8, seed=11,
                           space={33: "log", 34: "linear"}, dryRun=True)

    # A log draw is strictly positive by construction.
    for key, values in _byMf(run, 33).items():
        assert np.all(values > 0.0), key

    # The linear one is centred on one and free to sit either side of it,
    # which is the whole reason for asking for it.
    angular = _byMf(run, 34)
    assert angular, "the tape states MF34 for MT2"
    spread = np.concatenate([values for values in angular.values()])
    assert spread.min() < 1.0 < spread.max()


def test_the_two_spaces_produce_different_angular_factors():
    log = perturbFromModel(TAPE, BOTH, 4, seed=5, space="log", dryRun=True)
    linear = perturbFromModel(TAPE, BOTH, 4, seed=5,
                              space={33: "log", 34: "linear"}, dryRun=True)

    # Cross sections are untouched by the change: same space, same ladder rung.
    for key, values in _byMf(log, 33).items():
        np.testing.assert_array_equal(values, _byMf(linear, 33)[key])

    # The angular ones are drawn differently, which is the point.
    moved = [not np.array_equal(values, _byMf(linear, 34)[key])
             for key, values in _byMf(log, 34).items()]
    assert any(moved)


def test_the_run_records_the_space_each_block_was_drawn_in():
    run = perturbFromModel(TAPE, BOTH, 2, seed=7,
                           space={33: "log", 34: "linear"}, dryRun=True)
    spaces = {key: info.get("space") for key, info in run.diagnostics.items()}
    assert set(spaces.values()) == {"log", "linear"}, spaces
