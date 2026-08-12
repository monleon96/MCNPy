"""The plan half of ``kika.cov.conditioning`` — decide, then apply.

`test_conditioning.py` covers stage 1, which reads matrices and changes
nothing. This covers stages 2 and 3: the serialisable decision, and carrying it
out. The two properties worth holding are that **a plan means the same thing
after a round trip through JSON** — it is written to a run directory and read
back by a different process — and that **applying a repair ahead of the draw is
bit-identical to letting the draw make it**, which is what lets a conditioned
run be compared against the ensemble that shipped.

Synthetic matrices here on purpose: this is `kika/cov`, and the acceptance
against a real MF34 joint lives in
`kika/sampling/tests/test_conditioning_matches_the_sampler.py` where reading a
tape is already the module's business.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from kika.cov.conditioning import (
    ConditioningPlan,
    PlanStep,
    apply_plan,
    block_key_text,
    inspect_blocks,
    inspect_matrix,
)
from kika.cov.decomposition import _robust_eigh, clip_projection


def _correlated(n: int, seed: int) -> np.ndarray:
    """A PSD covariance with real off-diagonal structure."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n, max(2, n // 3)))
    matrix = factors @ factors.T + np.eye(n) * 0.1
    return (matrix + matrix.T) / 2.0


def _slightly_indefinite(n: int = 12, seed: int = 7) -> np.ndarray:
    """PSD, then one eigenvalue pushed just below zero.

    The shape a real evaluation arrives in: not a broken matrix, a matrix whose
    last digit of round-off went the wrong way.
    """
    matrix = _correlated(n, seed)
    values, vectors = np.linalg.eigh(matrix)
    values[0] = -1e-6 * values[-1]
    repaired = (vectors * values[None, :]) @ vectors.T
    return (repaired + repaired.T) / 2.0


def _sum_rule_block(n: int = 10, seed: int = 3) -> np.ndarray:
    """``C·1 = 0`` and not quite PSD — the MF35 shape.

    Projecting out the all-ones direction gives the identity exactly; nudging
    one other eigenvalue negative is what makes a PSD choice necessary, and the
    whole point is that the choice must not destroy the identity.
    """
    matrix = _correlated(n, seed)
    ones = np.ones((n, 1)) / np.sqrt(n)
    projector = np.eye(n) - ones @ ones.T
    matrix = projector @ matrix @ projector
    values, vectors = np.linalg.eigh(matrix)
    values[1] = -1e-3 * values[-1]          # [0] is the annihilated direction
    repaired = (vectors * values[None, :]) @ vectors.T
    return (repaired + repaired.T) / 2.0


# ---------------------------------------------------------------------------
# The projection, consolidated
# ---------------------------------------------------------------------------

def test_clip_projection_reproduces_each_call_sites_old_arithmetic():
    """The consolidation moved no bits, and this is what says so.

    Four copies of the projection were merged into one function with three
    flags. The flags exist because the copies were *not* the same projection,
    so the gate is not "the results are close" — it is that each call site's
    exact former arithmetic is still reachable, spelled out here rather than
    referenced.
    """
    matrix = _slightly_indefinite(24, seed=11)
    floor = 1e-12

    # nearest_psd_higham (non-preserving) and clip_and_rescale: floored,
    # symmetrised, always rebuilt.
    w, V = _robust_eigh(matrix, verbose=False)
    expected = (V * np.maximum(w, floor)[None, :]) @ V.T
    expected = (expected + expected.T) / 2.0
    got, _ = clip_projection(matrix, floor=floor, eigvals=w, eigvecs=V, verbose=False)
    np.testing.assert_array_equal(got, expected)

    # Higham's Dykstra loop: the same, without the fold-up to symmetry.
    expected_raw = (V * np.maximum(w, floor)[None, :]) @ V.T
    got_raw, _ = clip_projection(
        matrix, floor=floor, symmetrise=False, eigvals=w, eigvecs=V, verbose=False,
    )
    np.testing.assert_array_equal(got_raw, expected_raw)

    # svd_decomposition: floor 0.0, no fold-up, and the old spelling of the
    # product. `A @ diag(w)` and `A * w[None, :]` agree bit for bit because the
    # gemm adds exact zeros — asserted rather than assumed, since the whole
    # consolidation rests on it.
    expected_svd = V @ np.diag(np.clip(w, 0.0, None)) @ V.T
    got_svd, info = clip_projection(
        matrix, floor=0.0, symmetrise=False, passthrough_when_clean=True,
        eigvals=w, eigvecs=V, verbose=False,
    )
    np.testing.assert_array_equal(got_svd, expected_svd)
    assert info["n_negative"] == 1


def test_a_clean_matrix_passes_through_clip_untouched():
    """Not "unchanged to round-off" — the same object.

    ``psd_method="clip"`` is a no-op on an already-PSD matrix, and it is a no-op
    because the matrix is never rebuilt. Rebuilding it from its own
    eigendecomposition would move the last bits and every downstream draw with
    them.
    """
    matrix = _correlated(15, seed=5)
    got, info = clip_projection(
        matrix, passthrough_when_clean=True, symmetrise=False, verbose=False,
    )
    assert got is matrix
    assert info["clipped"] is False


# ---------------------------------------------------------------------------
# The plan object
# ---------------------------------------------------------------------------

def test_block_keys_survive_json_as_text():
    """A tuple key is not a JSON type, so a plan matches blocks by text."""
    assert block_key_text((26056, 2, 1)) == "(26056, 2, 1)"
    assert block_key_text((np.int64(26056), np.int32(2))) == "(26056, 2)"
    assert block_key_text("mf35-band-0") == "mf35-band-0"
    # The reason for recursing rather than calling str on the tuple: str uses
    # repr on the elements, and NumPy 2 reprs an integer as np.int64(26056).
    assert block_key_text((np.int64(26056),)) != str((np.int64(26056),))


def test_a_plan_round_trips_through_json_with_its_order_intact():
    plan = ConditioningPlan(
        steps=(
            PlanStep(key=(26056, 2, 1), remedy="cap",
                     parameters={"max_variance": 0.25}, reason="run 86 ceiling"),
            PlanStep(key=(26056, 2, 1), remedy="clip", reason="then project"),
            PlanStep(key=(26056, 2, 2), remedy="none", reason="clean"),
        ),
        notes=("(26056, 2, 2): variance_outliers left unrepaired",),
    )
    restored = ConditioningPlan.from_dict(json.loads(json.dumps(plan.to_dict())))

    assert [(block_key_text(s.key), s.remedy) for s in restored.steps] == [
        ("(26056, 2, 1)", "cap"),
        ("(26056, 2, 1)", "clip"),
        ("(26056, 2, 2)", "none"),
    ]
    assert restored.steps[0].parameters == {"max_variance": 0.25}
    assert restored.notes == plan.notes
    assert restored.steps_for((26056, 2, 1))[0].remedy == "cap"


def test_a_plan_from_a_future_version_is_refused_not_guessed_at():
    with pytest.raises(ValueError, match="version"):
        ConditioningPlan.from_dict({"version": 2, "steps": []})


def test_numpy_parameters_are_written_as_json_types():
    """The `variance_outliers` remedy hands back NumPy indices and targets."""
    step = PlanStep(
        key=0, remedy="rescale_to_family_median",
        parameters={"indices": np.array([3, 7]), "targets": np.array([1.5, 2.5])},
    )
    written = json.dumps(step.to_dict())          # raises if anything is NumPy
    assert json.loads(written)["parameters"]["indices"] == [3, 7]


# ---------------------------------------------------------------------------
# The recommendation
# ---------------------------------------------------------------------------

def test_the_recommendation_repairs_definiteness_and_names_the_rest():
    """It fixes what a draw cannot decline to fix, and reports what it left."""
    report = inspect_blocks({"indefinite": _slightly_indefinite()})
    plan = report.recommended_plan()

    (step,) = plan.steps
    assert step.remedy == "clip"                  # what psd_method="auto" resolves to
    assert "λ_min/λ_max" in step.reason


def test_a_clean_block_still_gets_a_step():
    """"Looked at, left alone" is a decision, and a plan has to record it."""
    plan = inspect_blocks({"clean": _correlated(12, seed=1)}).recommended_plan()
    (step,) = plan.steps
    assert step.remedy == "none"
    assert step.reason


def test_a_sum_rule_block_is_never_recommended_a_congruence():
    """C·1 ≈ 0 is destroyed by a congruence, so clip is the only admissible pick."""
    matrix = _sum_rule_block()
    report = inspect_blocks({"band0": matrix})
    assert report.findings("sum_rule"), "fixture no longer carries a sum rule"

    (step,) = report.recommended_plan().steps
    assert step.remedy == "clip"


def test_an_unrepaired_finding_is_written_down_rather_than_dropped():
    matrix = _correlated(30, seed=2)
    matrix[4, 4] = 1e6 * np.median(np.diag(matrix))     # one artefact bin
    plan = inspect_blocks({"outliers": matrix}).recommended_plan()

    assert any("variance_outliers" in note for note in plan.notes)
    assert all(step.remedy in ("none", "clip", "higham") for step in plan.steps)


# ---------------------------------------------------------------------------
# Applying
# ---------------------------------------------------------------------------

def test_apply_plan_returns_blocks_in_the_order_they_arrived():
    """The block index sets the seed offset downstream; reordering reseeds."""
    blocks = [
        ("b", _correlated(6, seed=1)),
        ("a", _slightly_indefinite(6, seed=2)),
        ("c", _correlated(6, seed=3)),
    ]
    plan = ConditioningPlan(steps=(PlanStep(key="a", remedy="clip"),))
    conditioned, applied = apply_plan(blocks, plan)

    assert [key for key, _ in conditioned] == ["b", "a", "c"]
    assert [record["block"] for record in applied] == ["a"]
    assert applied[0]["applier_info"]["n_negative"] == 1


def test_a_plan_naming_a_block_that_is_not_here_raises():
    """A plan that half-fits is a plan from another run, not a typo."""
    plan = ConditioningPlan(steps=(PlanStep(key="elsewhere", remedy="clip"),))
    with pytest.raises(ValueError, match="not in this set"):
        apply_plan({"here": _correlated(5, seed=1)}, plan)


def test_steps_apply_in_the_order_the_plan_lists_them():
    """Cap-then-project and project-then-cap are different matrices."""
    matrix = _slightly_indefinite(10, seed=4)
    cap = PlanStep(key=0, remedy="cap", parameters={"max_variance": 0.5})
    clip = PlanStep(key=0, remedy="clip")

    first, _ = apply_plan({0: matrix}, ConditioningPlan(steps=(cap, clip)))
    second, _ = apply_plan({0: matrix}, ConditioningPlan(steps=(clip, cap)))

    assert not np.array_equal(first[0][1], second[0][1])


def test_preserve_is_refused_because_it_is_a_constraint_not_an_operation():
    plan = ConditioningPlan(steps=(PlanStep(key=0, remedy="preserve"),))
    with pytest.raises(ValueError, match="constraint"):
        apply_plan({0: _correlated(5, seed=1)}, plan)


def test_an_unknown_remedy_is_refused_by_name():
    plan = ConditioningPlan(steps=(PlanStep(key=0, remedy="jitter"),))
    with pytest.raises(ValueError, match="unknown remedy"):
        apply_plan({0: _correlated(5, seed=1)}, plan)


def test_a_cap_with_no_ceiling_is_refused():
    plan = ConditioningPlan(steps=(PlanStep(key=0, remedy="cap"),))
    with pytest.raises(ValueError, match="max_variance"):
        apply_plan({0: _correlated(5, seed=1)}, plan)


def test_applying_the_recommendation_makes_the_block_faithful():
    """The round trip the design is for: inspect, decide, apply, inspect again."""
    blocks = {"band0": _sum_rule_block()}
    report = inspect_blocks(blocks)
    assert not report.faithful

    conditioned, applied = apply_plan(blocks, report.recommended_plan())
    assert inspect_blocks(conditioned).faithful

    # And the identity the fixture exists for survived the repair.
    matrix = conditioned[0][1]
    assert np.max(np.abs(matrix.sum(axis=1))) / np.max(np.abs(matrix)) < 1e-9
    assert applied[0]["frobenius_relative_change"] > 0.0


def test_mask_inert_zeroes_the_rows_the_evaluation_says_nothing_about():
    matrix = _correlated(8, seed=6)
    matrix[3, :] = 0.0
    matrix[:, 3] = 0.0
    matrix[5, 5] = 0.0          # a row with covariance but no variance

    plan = ConditioningPlan(steps=(PlanStep(key=0, remedy="mask_inert"),))
    conditioned, applied = apply_plan({0: matrix}, plan)
    (_key, masked), = conditioned

    assert applied[0]["applier_info"]["n_masked"] == 2
    assert np.all(masked[3, :] == 0.0) and np.all(masked[5, :] == 0.0)
    # Untouched rows keep their entries exactly.
    assert np.array_equal(masked[0, [0, 1, 2, 4, 6, 7]], matrix[0, [0, 1, 2, 4, 6, 7]])


def test_the_applied_log_says_what_moved():
    matrix = _slightly_indefinite(10, seed=8)
    plan = ConditioningPlan(steps=(
        PlanStep(key=0, remedy="clip_rescale", reason="diagonal must be exact"),
    ))
    _, applied = apply_plan({0: matrix}, plan)

    (record,) = applied
    assert record["remedy"] == "clip_rescale"
    assert record["reason"] == "diagonal must be exact"
    assert record["n"] == 10
    assert record["changed"] is True
    # clip_rescale restores the stated diagonal exactly; that is its whole claim.
    assert record["stated_diagonal_max_relative_change"] < 1e-12
    assert record["frobenius_relative_change"] > 0.0
    assert record["seconds"] >= 0.0


def test_inspect_then_apply_is_pure_on_the_input():
    """Stage 1 changes nothing and stage 3 changes a copy."""
    matrix = _slightly_indefinite(9, seed=9)
    original = matrix.copy()

    report = inspect_matrix(matrix, key="x")
    assert report.findings
    apply_plan({"x": matrix}, ConditioningPlan(
        steps=(PlanStep(key="x", remedy="clip"),)
    ))
    np.testing.assert_array_equal(matrix, original)
