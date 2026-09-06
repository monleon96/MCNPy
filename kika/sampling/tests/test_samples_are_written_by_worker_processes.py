"""``nWorkers``: the per-sample stage runs in a ``Pool``, and nothing else moves.

The draw is made once, in the parent, so the factors are the same at any
worker count; what a worker does is take a realisation, put it on its own
copy of the suite, write the files and hand its log events back. The gate is
therefore byte identity: every file a parallel run writes is the file the
serial run writes, the factors table is the same table, and the run's log
carries the same per-sample events -- the parallel one just says, in a note,
how many processes wrote them.

What is refused, and why: a parsed object as source (a worker re-reads the
tape and cannot be handed one), and ``nWorkers < 1``. A dry run takes the
argument and does not use it: there are no tapes to write. A worker that
dies reports which sample and why, instead of a ``Pool`` traceback that
names nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kika.endf import read_endf
from kika.sampling.model_perturbation import TAPE_EMITTERS, perturbFromModel
from kika.sampling.perturbation_set import readFactorsTable

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
TAPE = str(DATA / "micro_fe56_xs_and_angular.endf")
REQUEST = {33: None, 34: {"mt": [2], "index": [1, 2, 3]}}
SEED = 20260906
N = 4


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    serial = perturbFromModel(TAPE, REQUEST, N, seed=SEED,
                              outputDir=tmp_path_factory.mktemp("serial"),
                              formats=TAPE_EMITTERS, writeSets=True)
    parallel = perturbFromModel(TAPE, REQUEST, N, seed=SEED,
                                outputDir=tmp_path_factory.mktemp("parallel"),
                                formats=TAPE_EMITTERS, writeSets=True,
                                nWorkers=2)
    return serial, parallel


def test_every_file_is_the_file_the_serial_run_wrote(runs):
    serial, parallel = runs
    assert parallel.nSamples == serial.nSamples == N
    for one, other in zip(serial.samples, parallel.samples):
        assert one["label"] == other["label"]
        assert set(one["files"]) == set(other["files"]) >= set(TAPE_EMITTERS)
        for name in one["files"]:
            a, b = Path(one["files"][name]), Path(other["files"][name])
            assert a.name == b.name
            assert a.read_bytes() == b.read_bytes(), (name, a.name)


def test_the_factors_are_the_same_table(runs):
    serial, parallel = runs
    for number in range(N):
        one = readFactorsTable(serial.outputDir, number)
        other = readFactorsTable(parallel.outputDir, number)
        assert set(one.factors) == set(other.factors)
        for component in one.factors:
            np.testing.assert_array_equal(one.factors[component],
                                          other.factors[component])
    assert (serial.outputDir / "factors.parquet").read_bytes() == \
        (parallel.outputDir / "factors.parquet").read_bytes()


def test_the_log_carries_the_same_per_sample_events_in_sample_order(runs):
    serial, parallel = runs

    def perSample(run):
        return [(e.kind, e.sample, e.subject, e.level, e.message)
                for e in run.log if e.sample is not None]

    assert perSample(parallel) == perSample(serial)
    # The one thing the parallel log says that the serial one does not.
    notes = [e for e in parallel.log.of("note") if "worker process" in e.message]
    assert len(notes) == 1 and notes[0].payload["nWorkers"] == 2
    assert not [e for e in serial.log.of("note") if "worker" in e.message]
    assert parallel.notes == serial.notes
    assert parallel.log.problems() == [] if not serial.log.problems() else True

    metadata = json.loads((parallel.outputDir / "run_metadata.json").read_text())
    assert metadata["nWorkers"] == 2
    assert metadata["nSamples"] == N


def test_a_dry_run_takes_the_argument_and_says_it_did_not_use_it(tmp_path):
    run = perturbFromModel(TAPE, REQUEST, 2, seed=SEED, outputDir=tmp_path,
                           formats=TAPE_EMITTERS, dryRun=True, nWorkers=4)
    notes = [e for e in run.log.of("note") if "nWorkers=4 not used" in e.message]
    assert len(notes) == 1 and "dry run" in notes[0].message
    assert run.nSamples == 2


def test_a_parsed_object_cannot_be_handed_to_workers(tmp_path):
    with pytest.raises(ValueError, match="needs a path as source"):
        perturbFromModel(read_endf(TAPE), REQUEST, 2, seed=SEED,
                         outputDir=tmp_path, formats=("endf-tape",), nWorkers=2)


def test_fewer_than_one_worker_is_refused(tmp_path):
    with pytest.raises(ValueError, match="at least 1"):
        perturbFromModel(TAPE, REQUEST, 2, seed=SEED, outputDir=tmp_path,
                         formats=("endf-tape",), nWorkers=0)


def test_a_worker_that_dies_names_the_sample_and_the_reason(tmp_path):
    # Sample 1's directory is a *file*, so its mkdir fails in the worker.
    (tmp_path / "0001").write_text("in the way")
    with pytest.raises(RuntimeError, match="sample 1 failed in a worker") as failure:
        perturbFromModel(TAPE, REQUEST, 3, seed=SEED, outputDir=tmp_path,
                         formats=("endf-tape",), nWorkers=2)
    assert "0001" in str(failure.value)
