"""Piece F of ``docs/library/autofix_in_the_model.md``: the request by quantity, and a GNDS source.

``{33: None}`` speaks ENDF-6. The model underneath does not: a covariance block
lands on a *quantity* of a *reaction*, and a GNDS file names its reactions
``"n + Fe56"`` and calls nothing MF34. So the same request can be written by
quantity, naming reactions by MT or by label, and it has to assemble the same
blocks bit for bit -- and a GNDS ``reactionSuite`` with its covariance sibling
has to be a source the pipeline takes, drawing the same factors from the same
seed as the tape it was converted from.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import kika
from kika.endf import read_endf
from kika.endf.model_adapter import decodeCovarianceSuite
from kika.sampling.joint_blocks import (MF_OF_QUANTITY, QUANTITY_OF_MF,
                                        Selection, assembleRequest,
                                        collectEntries, normaliseRequest)
from kika.sampling.model_perturbation import perturbFromModel

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
TAPE = str(DATA / "micro_fe56_xs_and_angular.endf")
BY_FILE = {33: None, 34: {"mt": [2], "index": [1, 2, 3]}}
BY_QUANTITY = {"crossSection": None,
               "angularDistribution": {"reaction": "MT2", "order": [1, 2, 3]}}


# ----------------------------------------------------------------------
# The spelling
# ----------------------------------------------------------------------

def test_the_two_vocabularies_are_inverses():
    assert {MF_OF_QUANTITY[q] for q in MF_OF_QUANTITY} == set(QUANTITY_OF_MF)
    for mf, quantity in QUANTITY_OF_MF.items():
        assert MF_OF_QUANTITY[quantity] == mf


def test_numbers_resolve_without_a_suite():
    out = normaliseRequest({"crossSection": [2, 102],
                            "angularDistribution": {"reaction": 2, "order": [1]},
                            "multiplicity": None,
                            "energyDistribution": {"band": [0, 1]}})
    assert out == {33: Selection(33, mt=[2, 102]),
                   34: Selection(34, mt=[2], index=[1]),
                   31: None,
                   35: Selection(35, mt=None, index=[0, 1])}


def test_labels_need_the_suite_and_resolve_through_it():
    with pytest.raises(ValueError, match="named by label"):
        normaliseRequest({"crossSection": ["MT2"]})
    suite = kika.read(TAPE)
    out = normaliseRequest(BY_QUANTITY, suite)
    assert out == {33: None, 34: Selection(34, mt=[2], index=[1, 2, 3])}
    with pytest.raises(KeyError, match="no reaction labelled"):
        normaliseRequest({"crossSection": ["n + Pu239"]}, suite)


def test_a_summed_reaction_resolves_by_label_too():
    """MT1 lives in ``suite.sums``, and a request may name it."""
    suite = kika.read(str(DATA / "micro_fe56_mf33.endf"))
    out = normaliseRequest({"crossSection": ["MT4", "MT16"]}, suite)
    assert out == {33: Selection(33, mt=[4, 16])}


def test_the_wrong_third_coordinate_is_refused_by_name():
    with pytest.raises(ValueError, match="no 'band'; its third coordinate is 'order'"):
        normaliseRequest({"angularDistribution": {"reaction": 2, "band": 1}})
    with pytest.raises(ValueError, match="has no third coordinate"):
        normaliseRequest({"crossSection": {"reaction": 2, "order": 1}})
    with pytest.raises(ValueError, match="unknown field"):
        normaliseRequest({"crossSection": {"reaction": 2, "colour": "blue"}})
    with pytest.raises(ValueError, match="twice"):
        normaliseRequest({"crossSection": None, 33: None})
    with pytest.raises(ValueError, match="neither an MF number nor"):
        normaliseRequest({"crosssection": None})


def test_the_endf_spelling_passes_through_unchanged():
    assert normaliseRequest(BY_FILE) == BY_FILE
    sel = Selection(33, mt=[2])
    assert normaliseRequest(sel) is sel


def test_both_spellings_assemble_the_same_blocks():
    suite = kika.read(TAPE)
    covariances, _ = decodeCovarianceSuite(read_endf(TAPE))
    byFile, _ = assembleRequest(collectEntries(covariances, BY_FILE))
    byQuantity, _ = assembleRequest(
        collectEntries(covariances, normaliseRequest(BY_QUANTITY, suite)))
    assert [k for k, _ in byFile] == [k for k, _ in byQuantity]
    for (_, a), (_, b) in zip(byFile, byQuantity):
        np.testing.assert_array_equal(a, b)


def test_the_run_records_both_spellings(tmp_path):
    run = perturbFromModel(TAPE, BY_QUANTITY, 1, seed=1, outputDir=tmp_path,
                           dryRun=True)
    assert run.request == {33: None, 34: Selection(34, mt=[2], index=[1, 2, 3])}
    metadata = json.loads(run.files["metadata"].read_text("utf-8"))
    assert metadata["request"] == {"crossSection": None,
                                   "angularDistribution": {"reaction": "MT2",
                                                           "order": [1, 2, 3]}}
    assert metadata["requestNormalised"]["34"]["mt"] == [2]
    request = run.log.of("request")[0].payload
    assert request["asGiven"]["angularDistribution"]["reaction"] == "MT2"


# ----------------------------------------------------------------------
# A GNDS source
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def gnds(tmp_path_factory):
    """The fixture tape converted to GNDS by kika, covariance sibling included."""
    directory = tmp_path_factory.mktemp("gnds")
    suite = kika.read(TAPE)
    path = directory / "fe56.gnds.xml"
    kika.write(suite, str(path), format="gnds")
    return path


def test_a_gnds_source_draws_what_its_tape_draws(gnds, tmp_path):
    fromTape = perturbFromModel(TAPE, BY_FILE, 2, seed=5, dryRun=True)
    fromGnds = perturbFromModel(str(gnds), BY_FILE, 2, seed=5, dryRun=True,
                                outputDir=tmp_path)
    assert fromGnds.sourceFormat == "gnds"
    for a, b in zip(fromTape.samples, fromGnds.samples):
        assert a["set"].components() == b["set"].components()
        for component in a["set"].components():
            np.testing.assert_array_equal(a["set"].factors[component],
                                          b["set"].factors[component])
    metadata = json.loads(fromGnds.files["metadata"].read_text("utf-8"))
    assert metadata["sourceFormat"] == "gnds"
    assert fromGnds.log.of("read")[0].payload["format"] == "gnds"


def test_a_gnds_source_takes_the_request_by_label(gnds):
    suite = kika.read(str(gnds))
    labels = [reaction.label for reaction in suite.reactions]
    elastic = next(label for label in labels
                   if suite.reactionByLabel(label).ENDF_MT == 2)
    run = perturbFromModel(str(gnds), {"angularDistribution": {
        "reaction": elastic, "order": [1, 2, 3]}}, 1, seed=5, dryRun=True)
    assert run.request == {34: Selection(34, mt=[2], index=[1, 2, 3])}


def test_a_gnds_source_cannot_be_patched_but_can_be_written(gnds, tmp_path):
    with pytest.raises(ValueError, match="GNDS source has none"):
        perturbFromModel(str(gnds), BY_FILE, 1, seed=5, outputDir=tmp_path,
                         formats=("endf-delta",))
    run = perturbFromModel(str(gnds), BY_FILE, 1, seed=5, outputDir=tmp_path,
                           formats=("gnds",))
    written = run.samples[0]["files"]["gnds"]
    assert written.exists() and written.name == "fe56_0000.gnds.xml"
    back = kika.read(str(written))
    elastic = back.reactionByENDF_MT(2)
    assert "realization-0000" in elastic.crossSection


def test_a_source_of_another_format_is_refused(tmp_path):
    ace = next(DATA.parent.parent.parent.glob("ace/tests/data/*.ace"), None) \
        if (DATA.parent.parent.parent / "ace").exists() else None
    if ace is None:
        pytest.skip("no ACE fixture to refuse")
    with pytest.raises(ValueError, match="perturbation source is an ENDF tape or"):
        perturbFromModel(str(ace), BY_FILE, 1)
