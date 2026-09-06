"""Piece E of ``docs/library/autofix_in_the_model.md``: the log, and the dry run.

Two claims. The run records every stage as a typed, timed event with its
numbers, and the two files it writes -- ``run.log.jsonl`` for a program and
``run.log`` for a person -- are the same events. And a dry run does everything
a run does except write tapes, so that what it says about the sampling is what
the full run will do: same factors table, same log up to the first ``emitted``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from kika.sampling.model_perturbation import TAPE_EMITTERS, perturbFromModel
from kika.sampling.perturbation_set import readFactorsTable
from kika.sampling.run_log import EVENT_KINDS, RunLog

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
TAPE = str(DATA / "micro_fe56_xs_and_angular.endf")
REQUEST = {33: None, 34: {"mt": [2], "index": [1, 2, 3]}}
SEED = 20260906


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    full = perturbFromModel(TAPE, REQUEST, 2, seed=SEED,
                            outputDir=tmp_path_factory.mktemp("full"),
                            formats=TAPE_EMITTERS)
    dry = perturbFromModel(TAPE, REQUEST, 2, seed=SEED,
                           outputDir=tmp_path_factory.mktemp("dry"),
                           formats=TAPE_EMITTERS, dryRun=True)
    return full, dry


# ----------------------------------------------------------------------
# The log
# ----------------------------------------------------------------------

def test_every_stage_is_an_event_with_its_numbers(runs):
    full, _dry = runs
    kinds = [event.kind for event in full.log]
    for expected in ("started", "read", "request", "assembled", "inspected",
                     "conditioned", "drawn", "applied", "checked", "emitted",
                     "written", "finished"):
        assert expected in kinds, expected
    assert all(kind in EVENT_KINDS for kind in kinds)

    drawn = [e for e in full.log.of("drawn") if e.subject]
    assert len(drawn) == 2, "one per block"
    for event in drawn:
        assert {"rank", "n_null", "min_over_max_eigenvalue"} <= set(event.payload)

    emitted = full.log.of("emitted")
    assert len(emitted) == 2 * len(TAPE_EMITTERS)
    assert {e.subject for e in emitted} == set(TAPE_EMITTERS)
    assert all(e.sample in (0, 1) and e.seconds is not None for e in emitted)
    assert all(Path(e.payload["path"]).exists() for e in emitted)


def test_the_two_log_files_are_the_same_events(runs):
    full, _dry = runs
    jsonl, text = full.files["log"], full.files["log-text"]
    assert jsonl.name == "run.log.jsonl" and text.name == "run.log"

    restored = RunLog.read(jsonl)
    assert [e.kind for e in restored] == [e.kind for e in full.log]
    assert [e.message for e in restored] == [e.message for e in full.log]

    rendered = text.read_text(encoding="utf-8")
    for event in full.log:
        assert event.render() in rendered
    # The problems are repeated at the bottom so a person sees them last.
    for event in full.log.problems():
        assert rendered.count(event.render()) == 2


def test_a_python_logger_still_gets_every_line(tmp_path):
    logger = logging.getLogger("test_run_log_forwarding")
    logger.setLevel(logging.DEBUG)
    lines = []

    class Grab(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())
    handler = Grab()
    logger.addHandler(handler)
    try:
        run = perturbFromModel(TAPE, REQUEST, 1, seed=SEED, logger=logger)
    finally:
        logger.removeHandler(handler)
    assert len(lines) == len(run.log)
    assert lines == [e.render() for e in run.log]


def test_a_failing_stage_is_recorded_before_it_raises(tmp_path):
    log = RunLog()
    with pytest.raises(ValueError, match="unknown emitter"):
        perturbFromModel(TAPE, REQUEST, 1, formats=("hdf5",), runLog=log)
    # Refused before anything was read: no events but... none. The refusal is
    # a caller error, not a run event.
    assert len(log) == 0

    log = RunLog()
    with pytest.raises(ValueError, match="states no section"):
        perturbFromModel(TAPE, {33: [999]}, 1, runLog=log)
    errors = log.of("error")
    assert len(errors) == 1 and errors[0].payload["stage"] == "assembled"
    assert "states no section" in errors[0].message


def test_the_log_summary_reaches_the_metadata(runs):
    full, _dry = runs
    metadata = json.loads(full.files["metadata"].read_text("utf-8"))
    assert metadata["log"] == full.log.summary()
    assert metadata["files"]["log"] == "run.log.jsonl"
    assert metadata["dryRun"] is False


# ----------------------------------------------------------------------
# The dry run
# ----------------------------------------------------------------------

def test_a_dry_run_writes_no_tape_and_everything_else(runs):
    full, dry = runs
    assert dry.dryRun and dry.nSamples == 2
    assert all(sample["files"] == {} for sample in dry.samples)
    assert not any(p.is_dir() for p in dry.outputDir.iterdir()), "no sample dirs"
    assert not dry.log.of("emitted")
    for name in ("factors", "conditioning-plan", "metadata", "log", "log-text"):
        assert name in dry.files and dry.files[name].exists(), name
    metadata = json.loads(dry.files["metadata"].read_text("utf-8"))
    assert metadata["dryRun"] is True


def test_a_dry_run_draws_what_the_full_run_draws(runs):
    full, dry = runs
    for number in range(2):
        a = readFactorsTable(full.outputDir, number)
        b = readFactorsTable(dry.outputDir, number)
        assert a.components() == b.components()
        for component in a.components():
            np.testing.assert_array_equal(a.factors[component],
                                          b.factors[component])


def test_a_dry_run_tells_the_same_story_up_to_the_first_tape(runs):
    full, dry = runs
    def story(run):
        return [(e.kind, e.subject, e.sample, e.message)
                for e in run.log
                if e.kind not in ("emitted", "written", "finished", "started")]
    assert story(full) == story(dry)


def test_a_dry_run_checks_every_realisation(runs):
    _full, dry = runs
    checks = dry.log.of("checked")
    assert len(checks) == 2 * 4, "two samples, four components"
    for event in checks:
        assert event.payload["min"] > 0 and np.isfinite(event.payload["max"])
        assert "positive" in event.message


def test_a_linear_draw_that_crosses_zero_is_flagged(tmp_path):
    """The check the dry run exists for, provoked on purpose.

    The MF34 block carries a relative variance of order one on a₁ near a zero
    crossing, so a *linear* draw of it produces a factor below zero somewhere
    in a handful of samples. The log says so instead of the tape being written
    with a negative Legendre coefficient scaled the wrong way.
    """
    run = perturbFromModel(TAPE, {34: {"mt": [2], "index": [1]}}, 8, seed=3,
                           space="linear", dryRun=True)
    flagged = [e for e in run.log.problems() if "factor <= 0" in e.message]
    assert flagged, "expected at least one linear sample to cross zero"
    assert all(e.sample is not None and e.subject.startswith("MF34") for e in flagged)
