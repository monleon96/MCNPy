"""Piece D of ``docs/library/autofix_in_the_model.md``: one factors file per run.

A thousand samples used to be a thousand ``perturbation.json`` files, every one
repeating the same bin edges and the same provenance. Now a run writes
``factors.parquet`` and ``factors_index.json``; the gate is that sample *i*
read back from the table is the :class:`PerturbationSet` that was applied, to
the bit, and that reading one sample does not mean reading the others.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kika.sampling.joint_blocks import ComponentKey
from kika.sampling.model_perturbation import perturbFromModel
from kika.sampling.perturbation_set import (PerturbationSet, readFactorsIndex,
                                            readFactorsTable,
                                            writeFactorsTable)

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
TAPE = str(DATA / "micro_fe56_xs_and_angular.endf")
REQUEST = {33: None, 34: {"mt": [2], "index": [1, 2, 3]}}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return perturbFromModel(TAPE, REQUEST, 16, seed=7,
                            outputDir=tmp_path_factory.mktemp("table"),
                            dryRun=True)


def _same(a: PerturbationSet, b: PerturbationSet) -> None:
    assert a.label == b.label
    assert a.components() == b.components()
    assert a.groups == b.groups
    assert a.semantics == b.semantics and a.edgeRule == b.edgeRule
    for component in a.components():
        np.testing.assert_array_equal(a.factors[component], b.factors[component])
        np.testing.assert_array_equal(a.binEdges[component], b.binEdges[component])
        assert a.semanticsOf(component) == b.semanticsOf(component)
    assert a.outerDomains == b.outerDomains
    assert a.provenance == b.provenance


def test_the_run_writes_one_table_and_one_index_and_no_json_per_sample(run):
    files = sorted(p.name for p in run.outputDir.iterdir())
    assert "factors.parquet" in files
    assert not [f for f in files if f.endswith(".json") and "factors" in f], \
        "the index lives inside the parquet, not beside it"
    assert not list(run.outputDir.rglob("perturbation.json"))
    index = readFactorsIndex(run.outputDir)
    assert index["nSamples"] == 16
    assert len(index["labels"]) == 16
    assert index["provenance"]["seed"] == 7 and "sample" not in index["provenance"]
    assert {block["describe"] for block in index["blocks"]} == {
        c.describe() for c in run.samples[0]["set"].components()}


@pytest.mark.parametrize("number", [0, 7, 15])
def test_a_sample_read_back_is_the_set_that_was_applied(run, number):
    _same(run.samples[number]["set"], readFactorsTable(run.outputDir, number))
    _same(run.samples[number]["set"], PerturbationSet.fromRun(run.outputDir, number))


def test_reading_one_sample_reads_only_its_rows(run, monkeypatch):
    import pandas as pd
    seen = {}
    real = pd.read_parquet

    def spy(path, *args, **kwargs):
        frame = real(path, *args, **kwargs)
        seen["rows"] = len(frame)
        seen["filters"] = kwargs.get("filters")
        return frame
    monkeypatch.setattr(pd, "read_parquet", spy)
    pset = readFactorsTable(run.outputDir, 3)
    assert seen["filters"] == [("sample", "==", 3)]
    assert seen["rows"] == sum(v.size for v in pset.factors.values())


def test_a_sample_out_of_range_is_refused(run):
    with pytest.raises(IndexError, match="16 sample"):
        readFactorsTable(run.outputDir, 16)


def test_the_table_refuses_realisations_of_different_requests(run, tmp_path):
    other = perturbFromModel(TAPE, {33: None}, 1, seed=7)
    with pytest.raises(ValueError, match="different requests"):
        writeFactorsTable([run.samples[0]["set"], other.samples[0]["set"]], tmp_path)


def test_a_spectrum_run_keeps_its_bands_in_the_table(tmp_path):
    run = perturbFromModel(str(DATA / "micro_cf252_pfns.endf"), {35: None}, 2,
                           seed=11, outputDir=tmp_path, dryRun=True)
    back = readFactorsTable(tmp_path, 1)
    _same(run.samples[1]["set"], back)
    assert len(back.outerDomains) == 4
    assert all(back.semanticsOf(c) == "additive-absolute" for c in back.components())
    index = readFactorsIndex(tmp_path)
    assert all("outerDomain" in block for block in index["blocks"])
    assert readFactorsIndex(run.files["factors"]) == index, "the path of the parquet works too"


def test_the_per_sample_json_is_still_there_when_asked_for(tmp_path):
    run = perturbFromModel(TAPE, REQUEST, 2, seed=7, outputDir=tmp_path,
                           writeSets=True)
    for number, sample in enumerate(run.samples):
        path = sample["files"]["perturbation-set"]
        assert path == tmp_path / f"{number:04d}" / "perturbation.json"
        _same(PerturbationSet.read(path), readFactorsTable(tmp_path, number))


def test_the_table_size_is_what_the_argument_claimed(run):
    """Sixteen samples in one parquet cost less than half of sixteen JSONs.

    Measured: 22.8 KB against 16 x 5.0 KB. Not the 2x first claimed -- a
    parquet carries its columns and its metadata -- but the point stands, and
    it stands harder the more samples there are.
    """
    table = run.files["factors"].stat().st_size
    oneJson = len(json.dumps(run.samples[0]["set"].to_dict()))
    assert table < run.nSamples * oneJson / 2, (table, oneJson)
