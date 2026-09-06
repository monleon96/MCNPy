"""Taking the covariance from a second tape, and what that costs.

An evaluation normally states its own covariance, and that is the path
everything else in this package assumes. This one exists because a user can
hold the uncertainty for a nuclide in a different file from the evaluation they
want to perturb, and refusing outright would leave them editing tapes by hand
to get the two into one file.

It is not the recommended path and the tests say so as much as the code does:
the gate below is that the run **warns**, every time, because nothing in either
file asserts that the two belong together.
"""
from pathlib import Path

import numpy as np
import pytest

from kika.sampling.model_perturbation import perturbFromModel

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
FE56 = DATA / "micro_fe56_xs_and_angular.endf"
CF252 = DATA / "micro_cf252_pfns.endf"

pytestmark = pytest.mark.skipif(not FE56.is_file(), reason=f"{FE56} missing")


def test_a_tape_can_be_perturbed_with_its_own_covariance_named_explicitly():
    """The degenerate case: the same file, given twice.

    It has to draw exactly what naming it once draws, or the second path is
    reading something different from the first and nothing downstream can be
    compared.
    """
    implicit = perturbFromModel(FE56, {33: None}, 3, seed=42, dryRun=True)
    explicit = perturbFromModel(FE56, {33: None}, 3, seed=42, dryRun=True,
                                covarianceSource=FE56)

    left = implicit.samples[0]["set"].factors
    right = explicit.samples[0]["set"].factors
    assert set(left) == set(right)
    for key in left:
        np.testing.assert_array_equal(left[key], right[key], err_msg=str(key))


def test_the_run_warns_every_time_it_is_used():
    """The whole point. A quiet version of this feature would be a trap."""
    run = perturbFromModel(FE56, {33: None}, 1, seed=1, dryRun=True,
                           covarianceSource=FE56)
    warned = [event for event in run.log.problems()
              if "covariance taken from" in event.message]
    assert warned, "using a separate covariance file must be recorded"
    assert "not one the evaluation declares" in warned[0].message


def test_the_covariance_file_is_named_in_the_log():
    run = perturbFromModel(FE56, {33: None}, 1, seed=1, dryRun=True,
                           covarianceSource=FE56)
    started = run.log.of("started")[0]
    assert started.payload["covarianceSource"].endswith(FE56.name)
    reads = [event for event in run.log.of("read")
             if "separate file" in event.message]
    assert reads, "the second read is its own timed stage"


@pytest.mark.skipif(not CF252.is_file(), reason=f"{CF252} missing")
def test_another_nuclides_covariance_is_refused_by_name():
    """The one thing that can be checked, so it is.

    Whether an uncertainty *describes* an evaluation is a judgement; whether it
    is even about the same nuclide is a fact.
    """
    with pytest.raises(ValueError, match="ZAID"):
        perturbFromModel(FE56, {33: None}, 1, seed=1, dryRun=True,
                         covarianceSource=CF252)


def test_a_missing_covariance_file_says_so():
    with pytest.raises(ValueError, match="covariance file not found"):
        perturbFromModel(FE56, {33: None}, 1, seed=1, dryRun=True,
                         covarianceSource=DATA / "no_such_tape.endf")
