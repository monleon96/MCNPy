"""P2's acceptance: conditioning ahead of the draw is the draw's own repair.

The design says a run should decide its repairs in front of a human, apply
exactly those, and then draw with ``psd_method="none"`` because there is nothing
left to repair. That is only a refactor if the two routes give the *same*
numbers — and "the same" has to mean bit for bit, because the alternative is a
pipeline whose ensemble moves the day conditioning is switched on and no way to
tell that move from a modelling change.

So for every PSD remedy::

    draw_samples(apply_plan(blocks, plan), psd_method="none")
      ==  draw_samples(blocks, psd_method=<remedy>)

on the committed micro-tape's real MF34 joint — 252x252, six Legendre orders,
87 null directions, and not positive semi-definite as it stands, which is what
makes the comparison worth making at all.

`kika/cov/tests/test_conditioning_plan.py` holds the unit tests on synthetic
matrices. This file exists for the one property that needs a real covariance.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.cov.conditioning import (
    ConditioningPlan,
    PlanStep,
    apply_plan,
    inspect_blocks,
)
from kika.sampling.core import draw_samples
from kika.sampling.endf_perturbation import load_mf34_suite
from kika.sampling.model_blocks import legendre_covariance_blocks

STRUCTURAL = "kika/endf/tests/data/micro_fe56_structural.endf"

#: What the pipeline ships: linear, svd, random.
DRAW = dict(space="linear", returns="factors", decomposition_method="svd",
            sampling_method="random", seed=12345, verbose=False)


@pytest.fixture(scope="module")
def blocks():
    suite = load_mf34_suite(STRUCTURAL)
    found = legendre_covariance_blocks(
        suite, mt=[2], orders=list(range(1, 7)), relative=True,
    )
    assert found and found[0][1].shape == (252, 252)
    return found


@pytest.mark.parametrize("remedy", ["none", "clip", "clip_rescale", "higham"])
def test_conditioning_first_draws_what_repairing_inside_the_draw_draws(blocks, remedy):
    """The acceptance. Not `allclose` — bit for bit, or it is not a refactor."""
    (key, _matrix), = blocks
    plan = ConditioningPlan(steps=(PlanStep(key=key, remedy=remedy),))

    conditioned, applied = apply_plan(blocks, plan)
    ahead, _ = draw_samples(conditioned, 8, psd_method="none", **DRAW)
    inside, _ = draw_samples(blocks, 8, psd_method=remedy, **DRAW)

    np.testing.assert_array_equal(ahead[key], inside[key])
    assert applied[0]["remedy"] == remedy


def test_the_joint_is_the_case_this_test_is_for(blocks):
    """A guard on the fixture: an already-PSD matrix would pass for free."""
    (_key, matrix), = blocks
    spectrum = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    assert spectrum.min() < 0.0, "the micro-tape joint is no longer indefinite"


def test_the_recommendation_is_actionable_on_a_real_joint(blocks):
    """Inspect, apply what came back, and inspect again — the loop must close.

    This is what the round-off tolerance on the `definiteness` check exists for:
    before it, the report on a freshly clipped matrix still said "not PSD" at
    -1e-17 and no run could ever satisfy its own pre-flight.
    """
    report = inspect_blocks(blocks)
    assert not report.faithful

    conditioned, applied = apply_plan(blocks, report.recommended_plan())
    after = inspect_blocks(conditioned)

    assert not after.findings("definiteness")
    assert applied[0]["remedy"] == "clip"


def test_a_conditioned_draw_reports_no_repair_of_its_own(blocks):
    """`psd_method="none"` is the point: the projection stopped being implicit.

    Today the only record of which projection a run used is a log line. After
    conditioning, the projection is a plan step with a reason attached and the
    sampler makes no repair at all — so the applied log is the whole record.
    """
    report = inspect_blocks(blocks)
    plan = report.recommended_plan()
    conditioned, applied = apply_plan(blocks, plan)

    (record,) = applied
    assert record["changed"] is True
    assert record["reason"]
    assert record["applier_info"]["n_negative"] > 0

    _, diagnostics = draw_samples(conditioned, 4, psd_method="none", **DRAW)
    (info,) = diagnostics.values()
    # The conditioned matrix is PSD, so nothing the draw reports comes from a
    # repair it made. The null directions are the evaluation's own rank
    # deficiency, which conditioning does not and must not fill in.
    assert info["n_null"] > 0
    assert info["rank"] + info["n_null"] == info["n"]


def test_a_derived_plan_closes_the_same_loop_as_a_psd_one():
    """P2's acceptance, on the plan P4 actually ships.

    The parametrised test above holds the loop for a plan whose only step is a
    PSD projection, which is what ``recommended_plan`` produces. The MF33 draw
    ships a plan derived from ``generate_samples``' repair set instead —
    rescales and a cap, no PSD step at all — so the property has to be held
    again for that shape, or P2's guarantee simply does not cover the first
    caller to use it.

    Deliberately the *whole* helper rather than ``apply_plan`` alone: the
    inert-row drop sits outside the plan, and an equivalence that stopped at
    the plan boundary would not notice it going wrong.
    """
    from kika.sampling.carrier_blocks import (
        cross_section_carrier_blocks,
        cross_section_carrier_index,
    )
    from kika.sampling.generators import generate_samples
    from kika.sampling.mf33_sampling import load_mf33_covariance
    from kika.sampling.multigroup_draw import draw_relative_factors

    micro = "kika/endf/tests/data/micro_fe56_mf33.endf"
    cov, _mf3, grid, present = load_mf33_covariance(
        micro, None, [4, 16], energy_unit="eV",
    )

    (key, matrix), = cross_section_carrier_blocks(cov)
    (_key, meta), = cross_section_carrier_index(cov).items()

    old, _mts, _fix = generate_samples(
        cov, 4, space="log", decomposition_method="svd", sampling_method="random",
        seed=12345, mt_numbers=list(present), energy_grid=grid, autofix=None,
        psd_method="auto", max_relative_std=3.0, verbose=False,
    )
    new, info = draw_relative_factors(
        matrix, 4, key=key, pairs=meta["pairs"], stride=meta["stride"], bins=grid,
        space="log", decomposition_method="svd", sampling_method="random",
        seed=12345, psd_method="auto", max_relative_std=3.0, verbose=False,
    )
    np.testing.assert_array_equal(new, old)
    assert [s.remedy for s in info["plan"].steps] == ["cap"]
    assert info["n_inert_dropped"] > 0
