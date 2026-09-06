"""A plan handed to ``perturbFromModel`` is applied, and the run says so.

Found from a notebook on 2026-09-06, not from a test: ``perturbFromModel`` bound
the *tuple* ``apply_plan`` returns as its block list, so every run that passed
``conditioningPlan`` died in the draw with ``unhashable type: 'numpy.ndarray'``.
The pipeline's docstring says conditioning is the design that replaces
``autofix``, and the one path that implements it had never been executed.

Three things are pinned here, on the same fixture as the end-to-end gate:

1. a plan whose steps are all ``none`` draws **exactly** what no plan draws --
   the seed ladder, the block order and the matrices are untouched;
2. a ``clip`` step ahead of the draw under ``psdMethod="none"`` is
   **bit-identical** to no plan under ``psdMethod="clip"``. That is P2's
   acceptance (``test_conditioning_matches_the_sampler.py``) restated at the
   pipeline level, and it holds here for a reason worth stating: the draw
   decomposes the assembled block as given, so the plan and the projection act
   on the same matrix;
3. the run directory carries the plan and the record of what it did, because a
   plan nobody can find afterwards is a log line with extra steps.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kika.cov.conditioning import (ConditioningPlan, PlanStep, apply_plan,
                                   inspect_blocks)
from kika.endf import read_endf
from kika.endf.model_adapter import decodeCovarianceSuite
from kika.sampling.joint_blocks import (assembleRequest, collectEntries,
                                        componentDomains)
from kika.sampling.model_perturbation import perturbFromModel

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
TAPE = str(DATA / "micro_fe56_xs_and_angular.endf")
REQUEST = {33: None, 34: {"mt": [2], "index": [1, 2, 3]}}
SEED = 20260906


@pytest.fixture(scope="module")
def blocks():
    """The blocks the pipeline assembles, built the same way it builds them."""
    covariances, _report = decodeCovarianceSuite(read_endf(TAPE))
    entries = collectEntries(covariances, REQUEST)
    assembled, _index = assembleRequest(
        entries, domains=componentDomains(covariances, REQUEST))
    return assembled


def _factors(run):
    return {component: run.samples[0]["set"].factors[component]
            for component in run.samples[0]["set"].components()}


def test_a_plan_of_none_steps_draws_what_no_plan_draws(blocks, tmp_path):
    plan = ConditioningPlan(steps=tuple(
        PlanStep(key=key, remedy="none", reason="looked at, left alone")
        for key, _matrix in blocks))

    bare = perturbFromModel(TAPE, REQUEST, 1, seed=SEED, conditioningPlan=False)
    planned = perturbFromModel(TAPE, REQUEST, 1, seed=SEED, outputDir=tmp_path,
                               conditioningPlan=plan)

    assert _factors(bare).keys() == _factors(planned).keys()
    for component, factors in _factors(bare).items():
        np.testing.assert_array_equal(factors, _factors(planned)[component])

    assert len(planned.conditioning) == len(blocks)
    assert {record["remedy"] for record in planned.conditioning} == {"none"}
    assert not any(record["changed"] for record in planned.conditioning)


def test_a_clip_step_is_the_draws_own_clip_in_linear_space(blocks, tmp_path):
    """Conditioning first + ``none`` inside == ``clip`` inside, bit for bit."""
    plan = ConditioningPlan(steps=tuple(
        PlanStep(key=key, remedy="clip", reason="pipeline gate")
        for key, _matrix in blocks))

    planned = perturbFromModel(TAPE, REQUEST, 1, seed=SEED, outputDir=tmp_path,
                               conditioningPlan=plan, psdMethod="none",
                               space="linear")
    reference = perturbFromModel(TAPE, REQUEST, 1, seed=SEED, psdMethod="clip",
                                 space="linear", conditioningPlan=False)

    for component, factors in _factors(reference).items():
        np.testing.assert_array_equal(factors, _factors(planned)[component])
    assert [record["remedy"] for record in planned.conditioning] == ["clip"] * len(blocks)
    for key, diagnostics in planned.diagnostics.items():
        assert diagnostics["psd_method"] == "none", key


def test_in_log_space_the_plan_path_moment_matches_the_matrix_it_samples(blocks):
    """The one place the two paths differ, and it is the plan path that is right.

    A log-space draw returns ``exp(y - diag(C)/2)`` so the factors have unit
    mean *for the covariance C that was sampled*. With the projection inside
    the draw, ``diag(C)`` is read off the matrix as **stated** while ``y`` is
    drawn from the matrix as **clipped** -- and clip moves this block's stated
    variances by up to 4.4 %, so the mean correction is taken against a
    diagonal the draw did not use. With the projection ahead of the draw, both
    come from the same matrix.

    Measured on this fixture: the two differ by up to 8.2e-6 on 22 of 42
    entries of every MF34 order, and the ratio between them is
    ``exp(-(diag_clipped - diag_stated)/2)`` to 2.2e-16. First read as
    null-direction debris; it is not -- truncating the null directions leaves
    the number unchanged, and the identity below is exact.
    """
    plan = ConditioningPlan(steps=tuple(
        PlanStep(key=key, remedy="clip", reason="pipeline gate")
        for key, _matrix in blocks))
    conditioned, _applied = apply_plan(blocks, plan)

    planned = perturbFromModel(TAPE, REQUEST, 1, seed=SEED, conditioningPlan=plan,
                               psdMethod="none", space="log")
    reference = perturbFromModel(TAPE, REQUEST, 1, seed=SEED, psdMethod="clip",
                                 space="log", conditioningPlan=False)

    somethingMoved = False
    for (key, stated), (_key, clipped) in zip(blocks, conditioned):
        components = planned.index[key]["components"]
        stride = planned.index[key]["stride"]
        for at, component in enumerate(components):
            rows = slice(at * stride, (at + 1) * stride)
            shift = np.diag(clipped)[rows] - np.diag(stated)[rows]
            ahead = _factors(planned)[component]
            inside = _factors(reference)[component]
            np.testing.assert_allclose(ahead / inside, np.exp(-0.5 * shift),
                                       rtol=0, atol=1e-14)
            somethingMoved |= bool(np.any(shift != 0.0))
    assert somethingMoved, "the fixture no longer exercises the difference"


def test_the_recommended_plan_is_accepted_as_it_comes(blocks, tmp_path):
    """The three-stage loop, end to end: inspect, take the default, run."""
    report = inspect_blocks(blocks)
    plan = report.recommended_plan()
    assert len(plan.steps) == len(blocks), "one step per block, `none` included"

    run = perturbFromModel(TAPE, REQUEST, 1, seed=SEED, outputDir=tmp_path,
                           conditioningPlan=plan)
    assert run.nSamples == 1
    for factors in _factors(run).values():
        assert np.all(np.isfinite(factors)) and np.all(factors > 0)

    # And the loop closes: the conditioned blocks pass their own pre-flight on
    # definiteness, which is the one thing the recommended plan repairs.
    conditioned, _applied = apply_plan(blocks, plan)
    after = inspect_blocks(conditioned)
    assert not any(finding.check == "definiteness" and finding.severity != "note"
                   for block in after.blocks for finding in block.findings)


def test_the_run_directory_carries_the_plan_and_what_it_did(blocks, tmp_path):
    plan = ConditioningPlan(steps=tuple(
        PlanStep(key=key, remedy="clip", reason="because the test says so")
        for key, _matrix in blocks), notes=("nothing left unrepaired",))
    run = perturbFromModel(TAPE, REQUEST, 1, seed=SEED, outputDir=tmp_path,
                           conditioningPlan=plan)

    written = tmp_path / "conditioning_plan.json"
    assert written.exists()
    restored = ConditioningPlan.from_dict(json.loads(written.read_text("utf-8")))
    assert [step.remedy for step in restored.steps] == ["clip"] * len(blocks)
    assert [step.reason for step in restored.steps] == [
        "because the test says so"] * len(blocks)
    assert restored.notes == ("nothing left unrepaired",)

    metadata = json.loads((tmp_path / "run_metadata.json").read_text("utf-8"))
    assert metadata["psd_method"] == "none"
    assert metadata["conditioning"]["plan"] == "conditioning_plan.json"
    applied = metadata["conditioning"]["applied"]
    assert [record["remedy"] for record in applied] == ["clip"] * len(blocks)
    assert {record["block"] for record in applied} == {
        record["block"] for record in run.conditioning}


def test_a_run_that_refuses_conditioning_says_so(tmp_path):
    run = perturbFromModel(TAPE, REQUEST, 1, seed=SEED, outputDir=tmp_path,
                           conditioningPlan=False)
    assert run.conditioningPlan is None and run.conditioning == ()
    assert run.conditioningMode == "none" and run.report is None
    assert not (tmp_path / "conditioning_plan.json").exists()
    metadata = json.loads((tmp_path / "run_metadata.json").read_text("utf-8"))
    assert metadata["conditioning"]["mode"] == "none"
    assert metadata["conditioning"]["plan"] is None
    assert metadata["conditioning"]["applied"] == []


def test_without_a_plan_the_run_conditions_itself_and_writes_what_it_did(
        blocks, tmp_path):
    """Piece A of ``docs/library/autofix_in_the_model.md``.

    ``conditioningPlan=None`` is "inspect and apply the recommendation": the
    clip the draw would have made silently under ``psd_method="auto"``, with a
    name and a reason in the run directory. It must draw exactly what the same
    recommendation handed in explicitly draws -- the point is that the default
    is the explicit plan, not a different repair.
    """
    auto = perturbFromModel(TAPE, REQUEST, 1, seed=SEED, outputDir=tmp_path)
    assert auto.conditioningMode == "auto"
    assert auto.report is not None and auto.report.samplable
    remedies = {record["block"]: record["remedy"] for record in auto.conditioning}
    assert set(remedies.values()) == {"none", "clip"}, remedies
    assert [r for r in remedies if "MF34" in r] and all(
        remedies[r] == "clip" for r in remedies if "MF34" in r)
    assert all(remedies[r] == "none" for r in remedies if "MF33" in r)

    explicit = perturbFromModel(TAPE, REQUEST, 1, seed=SEED,
                                conditioningPlan=inspect_blocks(blocks).recommended_plan())
    for component, factors in _factors(explicit).items():
        np.testing.assert_array_equal(factors, _factors(auto)[component])

    written = ConditioningPlan.from_dict(json.loads(
        (tmp_path / "conditioning_plan.json").read_text("utf-8")))
    assert [s.remedy for s in written.steps] == [s.remedy for s in auto.conditioningPlan.steps]
    metadata = json.loads((tmp_path / "run_metadata.json").read_text("utf-8"))
    assert metadata["conditioning"]["mode"] == "auto"
    assert "block(s)" in metadata["conditioning"]["preflight"]

    # And the pre-flight was told which rows belong together. Measured: the
    # "1 variance at 1.39e3x the median" the whole-block inspection reports on
    # this joint is a₁'s bins judged against a₃'s -- within its own order it is
    # no outlier, and the finding does not fire. Piece C of the note.
    assert auto.report.findings("definiteness")
    assert not auto.report.findings("variance_outliers")
    assert inspect_blocks(blocks).findings("variance_outliers"),         "the whole-block inspection no longer shows the artefact this pins"


def test_a_plan_for_other_blocks_is_refused_before_anything_is_drawn(tmp_path):
    """A plan half-fitting this run is a plan from another run."""
    plan = ConditioningPlan(steps=(
        PlanStep(key="not-a-block-of-this-run", remedy="clip"),))
    with pytest.raises(ValueError, match="not in this set"):
        perturbFromModel(TAPE, REQUEST, 1, seed=SEED, outputDir=tmp_path,
                         conditioningPlan=plan)
    assert not list(tmp_path.iterdir()), "nothing was written"
