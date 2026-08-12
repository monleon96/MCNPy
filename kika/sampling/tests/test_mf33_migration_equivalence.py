"""The MF33 migration gate: `draw_relative_factors` draws what it replaces.

`generate_samples` and `draw_relative_factors` must produce the *same bits* for
the configurations the pipelines ship, on a real evaluation's covariance. Not
``allclose`` — a migration that moves the samples is a migration whose effect
cannot be attributed, and the whole reason for landing an equivalent version
first is that the improvement afterwards has something to be measured against.

**These tests are scheduled to die.** They compare the new draw against the old
one, so they go with `generate_samples`, and again with the commit that turns
null-space truncation on (`null_tol=None` -> the default) and deliberately moves
every drawn column. Until then they are the reference, not dead weight.

Two things the fixture cannot reach, tested on synthetic matrices instead:
the statistical-outlier rescale, which JEFF-4.0's MT4/MT16 do not trigger, and
the multi-isotope ordering.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika.cov.cross_section_covariance import CrossSectionCovariance
from kika.sampling.carrier_blocks import (
    cross_section_carrier_blocks,
    cross_section_carrier_index,
)
from kika.sampling.generators import generate_samples
from kika.sampling.mf33_sampling import extract_mt_param_blocks, load_mf33_covariance
from kika.sampling.multigroup_draw import (
    draw_relative_factors,
    shipped_repair_plan,
)

MICRO_MF33 = str(Path(__file__).resolve().parents[2]
                 / "endf" / "tests" / "data" / "micro_fe56_mf33.endf")

#: What `exfor_to_endf_research.py` ships, so the gate holds the shipped draw.
SEED = 12345
N_SAMPLES = 8
MTS = [4, 16]


@pytest.fixture(scope="module")
def carrier():
    cov, _mf3, grid, present = load_mf33_covariance(
        MICRO_MF33, None, MTS, energy_unit="eV",
    )
    return cov, grid, present


@pytest.fixture(scope="module")
def blocks(carrier):
    cov, grid, _present = carrier
    (key, matrix), = cross_section_carrier_blocks(cov)
    (_key, meta), = cross_section_carrier_index(cov).items()
    return key, matrix, meta, grid


def _thresholds(grid):
    """Thresholds that land strictly inside a bin, one per MT."""
    return {4: float(grid[len(grid) // 3]), 16: float(grid[len(grid) // 2])}


def _old(cov, grid, present, *, space, thresholds):
    factors, _mts, _fix = generate_samples(
        cov, N_SAMPLES, space=space, decomposition_method="svd",
        sampling_method="sobol", seed=SEED, mt_numbers=list(present),
        energy_grid=grid, autofix=None, psd_method="auto",
        max_relative_std=3.0, mt_thresholds=thresholds, verbose=False,
    )
    return factors


def _new(key, matrix, meta, grid, *, space, thresholds, **kwargs):
    return draw_relative_factors(
        matrix, N_SAMPLES, key=key, pairs=meta["pairs"], stride=meta["stride"],
        bins=grid, space=space, decomposition_method="svd",
        sampling_method="sobol", seed=SEED, psd_method="auto",
        max_relative_std=3.0, mt_thresholds=thresholds, verbose=False,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("space, ceiling", [("log", np.log(10.0)), ("linear", 9.0)])
def test_the_plan_is_generate_samples_repair_sequence(blocks, space, ceiling):
    """Order is the guarantee: `apply_plan` walks the steps as listed."""
    key, matrix, meta, grid = blocks
    working = np.log1p(matrix) if space == "log" else matrix

    bare = shipped_repair_plan(
        working, key=key, pairs=meta["pairs"], stride=meta["stride"],
        bins=grid, space=space, max_relative_std=3.0, mt_thresholds=None,
    )
    assert [s.remedy for s in bare.steps] == ["cap"]
    assert bare.steps[-1].parameters["max_variance"] == pytest.approx(ceiling)

    targeted = shipped_repair_plan(
        working, key=key, pairs=meta["pairs"], stride=meta["stride"],
        bins=grid, space=space, max_relative_std=3.0,
        mt_thresholds=_thresholds(grid),
    )
    assert [s.remedy for s in targeted.steps] == [
        "rescale_to_family_median", "cap",
    ]
    # The rescale must come first: the cap acts on residual outliers only.
    assert targeted.steps[0].parameters["indices"]
    assert len(targeted.steps[0].parameters["indices"]) == len(
        targeted.steps[0].parameters["targets"]
    )


def test_the_plan_round_trips_through_json(blocks):
    """A plan that cannot be written to the run directory is not a plan."""
    import json

    from kika.cov.conditioning import ConditioningPlan

    key, matrix, meta, grid = blocks
    plan = shipped_repair_plan(
        np.log1p(matrix), key=key, pairs=meta["pairs"], stride=meta["stride"],
        bins=grid, space="log", mt_thresholds=_thresholds(grid),
    )
    restored = ConditioningPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    assert [s.remedy for s in restored.steps] == [s.remedy for s in plan.steps]
    assert restored.steps[0].parameters["indices"] == \
        plan.steps[0].parameters["indices"]


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("space", ["log", "linear"])
@pytest.mark.parametrize("targeted", [False, True])
def test_the_draw_is_bit_identical(carrier, blocks, space, targeted):
    """The migration gate. `allclose` would not do."""
    cov, grid, present = carrier
    key, matrix, meta, _grid = blocks
    thresholds = _thresholds(grid) if targeted else None

    old = _old(cov, grid, present, space=space, thresholds=thresholds)
    new, info = _new(key, matrix, meta, grid, space=space, thresholds=thresholds)

    assert new.shape == old.shape
    np.testing.assert_array_equal(new, old)
    # and the drop-and-pin was exercised rather than skipped
    assert info["n_inert_dropped"] > 0


def test_the_inert_rows_are_pinned_and_the_cast_is_float32(carrier, blocks):
    cov, grid, present = carrier
    key, matrix, meta, _grid = blocks

    new, info = _new(key, matrix, meta, grid, space="log", thresholds=None)

    assert new.dtype == np.float32
    inert = info["inert"]
    assert inert.sum() == info["n_inert_dropped"]
    # exactly 1.0, not approximately: these bins state no variance at all
    assert np.all(new[:, inert] == np.float32(1.0))
    assert not np.all(new[:, ~inert] == np.float32(1.0))


def test_the_nan_fill_and_the_zero_fill_draw_the_same_factors(blocks):
    """D11's fill is inert for every repair, and this holds it that way.

    The flaggers read only the diagonal and both congruences are
    ``diag(s) @ M @ diag(s)``, so an unstated cross block scales without
    contaminating anything. If an applier ever grows a whole-matrix reduction
    -- a ``nanmax``, a norm -- the two fills diverge and this fails.
    """
    key, matrix, meta, grid = blocks
    assert not np.isfinite(matrix).all(), "fixture must exercise the NaN fill"

    nan_filled, _info = _new(key, matrix, meta, grid, space="log", thresholds=None)
    zeroed = np.where(np.isfinite(matrix), matrix, 0.0)
    zero_filled, _info = _new(key, zeroed, meta, grid, space="log", thresholds=None)

    np.testing.assert_array_equal(nan_filled, zero_filled)


def test_one_block_leaves_the_seed_alone(blocks):
    """`draw_samples` offsets by block index; one block must mean no offset."""
    key, matrix, meta, grid = blocks
    _new_factors, info = _new(key, matrix, meta, grid, space="log", thresholds=None)
    assert info["diagnostics"]["seed"] == SEED
    assert info["diagnostics"]["block_index"] == 0


def test_the_layout_is_the_one_the_appliers_read(carrier, blocks):
    """`extract_mt_param_blocks`' slices, derived from the index instead."""
    cov, _grid, _present = carrier
    _key, _matrix, meta, _g = blocks

    expected = extract_mt_param_blocks(cov)
    stride = meta["stride"]
    derived = {
        mt: slice(i * stride, (i + 1) * stride)
        for i, (_za, mt) in enumerate(meta["pairs"])
    }
    assert derived == expected
    assert meta["dimension"] == len(meta["pairs"]) * stride
    assert set(meta["widths"].values()) == {stride}


def test_a_stride_mismatch_is_refused(blocks):
    """Mis-slicing would attribute one reaction's bins to another, silently."""
    key, matrix, meta, grid = blocks
    with pytest.raises(ValueError, match="does not match"):
        draw_relative_factors(
            matrix, N_SAMPLES, key=key, pairs=meta["pairs"],
            stride=meta["stride"] + 1, bins=grid, seed=SEED, verbose=False,
        )


# ---------------------------------------------------------------------------
# What the fixture cannot reach
# ---------------------------------------------------------------------------

def _synthetic(n_groups=8, mts=(2, 102), isotopes=(26056,), outlier=None):
    """A small relative covariance with a known diagonal, on a shared grid."""
    grid = [10.0 ** k for k in range(n_groups + 1)]
    cov = CrossSectionCovariance(
        num_groups=n_groups, energy_grid=list(grid), energy_unit="eV",
    )
    for za in isotopes:
        for mt in mts:
            diag = np.full(n_groups, 0.01)
            if outlier is not None and (za, mt) == outlier[0]:
                diag[outlier[1]] = outlier[2]
            block = np.diag(diag) + 1.0e-4
            cov.add_matrix(za, mt, za, mt, block,
                           energy_grid=list(grid), is_relative=True)
    return cov, grid


def test_the_outlier_rescale_is_reproduced(carrier):
    """JEFF-4.0's MT4/MT16 never trigger it; a placeholder variance does."""
    # 1e6 rather than 1e4: the flagger compares in the drawn space, and
    # log1p(1e4)/log1p(0.0101) is 917 -- just under the 1000x factor.
    cov, grid = _synthetic(outlier=((26056, 102), 5, 1.0e6))
    (key, matrix), = cross_section_carrier_blocks(cov)
    (_k, meta), = cross_section_carrier_index(cov).items()

    plan = shipped_repair_plan(
        np.log1p(matrix), key=key, pairs=meta["pairs"], stride=meta["stride"],
        bins=grid, space="log", max_relative_std=3.0, mt_thresholds=None,
    )
    assert [s.remedy for s in plan.steps] == ["rescale_to_family_median", "cap"]

    old, _mts, _fix = generate_samples(
        cov, N_SAMPLES, space="log", decomposition_method="svd",
        sampling_method="sobol", seed=SEED, mt_numbers=[2, 102],
        energy_grid=grid, autofix=None, psd_method="auto",
        max_relative_std=3.0, mt_thresholds=None, verbose=False,
    )
    new, _info = draw_relative_factors(
        matrix, N_SAMPLES, key=key, pairs=meta["pairs"], stride=meta["stride"],
        bins=grid, space="log", decomposition_method="svd",
        sampling_method="sobol", seed=SEED, psd_method="auto",
        max_relative_std=3.0, mt_thresholds=None, verbose=False,
    )
    np.testing.assert_array_equal(new, old)


def test_multi_isotope_ordering_is_the_carrier_s(carrier):
    """`sorted(pairs)` is isotope-then-MT; the layout must follow it."""
    cov, grid = _synthetic(mts=(2, 102), isotopes=(26056, 26057))
    (key, matrix), = cross_section_carrier_blocks(cov)
    (_k, meta), = cross_section_carrier_index(cov).items()

    assert meta["pairs"] == sorted(meta["pairs"], key=lambda p: (p[0], p[1]))
    assert meta["dimension"] == matrix.shape[0]

    old, _mts, _fix = generate_samples(
        cov, N_SAMPLES, space="log", decomposition_method="svd",
        sampling_method="sobol", seed=SEED, mt_numbers=[2, 102],
        energy_grid=grid, autofix=None, psd_method="auto",
        max_relative_std=3.0, verbose=False,
    )
    new, _info = draw_relative_factors(
        matrix, N_SAMPLES, key=key, pairs=meta["pairs"], stride=meta["stride"],
        bins=grid, space="log", decomposition_method="svd",
        sampling_method="sobol", seed=SEED, psd_method="auto",
        max_relative_std=3.0, verbose=False,
    )
    np.testing.assert_array_equal(new, old)


def test_a_wholly_inert_block_raises_rather_than_drawing_nothing(carrier):
    cov, grid = _synthetic(n_groups=4)
    (key, matrix), = cross_section_carrier_blocks(cov)
    (_k, meta), = cross_section_carrier_index(cov).items()
    with pytest.raises(ValueError, match="nothing to sample"):
        draw_relative_factors(
            np.zeros_like(matrix), N_SAMPLES, key=key, pairs=meta["pairs"],
            stride=meta["stride"], bins=grid, seed=SEED, verbose=False,
        )


# ---------------------------------------------------------------------------
# What the ACE call sites need that the draw does not: the autofix seam
# ---------------------------------------------------------------------------
# `generate_samples` carried `soft_autofix_failed` as a local flag spanning the
# fix and the decomposition, and both ACE pipelines read the consequences. The
# migration splits those two steps across a call site, so the flag has to
# survive the split — and it decides two separate things, one on each branch.


def test_the_two_exception_classes_are_one_object_under_both_names():
    """A caller catching `generators`' must catch what `multigroup_draw` raises.

    They coexist for as long as the equivalence gate does. If these ever became
    two classes, an ACE isotope whose soft autofix missed would stop being a
    skipped isotope with a diagnosis and become an unhandled exception that
    takes the whole run down.
    """
    from kika.sampling import errors, generators, multigroup_draw

    assert (generators.CovarianceFixError
            is multigroup_draw.CovarianceFixError
            is errors.CovarianceFixError)
    assert (generators.SoftAutofixWarning
            is multigroup_draw.SoftAutofixWarning
            is errors.SoftAutofixWarning)


@pytest.mark.parametrize("level,fix_info,expected", [
    (None, {"converged": False, "soft_threshold_met": False}, False),
    ("soft", None, False),
    ("soft", {"converged": True, "soft_threshold_met": False}, False),
    ("soft", {"converged": False, "soft_threshold_met": True}, False),
    ("soft", {"converged": False, "soft_threshold_met": False}, True),
    ("SOFT", {"converged": False, "soft_threshold_met": False}, True),
    ("medium", {"converged": False, "soft_threshold_met": False}, False),
])
def test_soft_autofix_missed_is_generate_samples_condition(level, fix_info, expected):
    from kika.sampling.multigroup_draw import soft_autofix_missed

    assert soft_autofix_missed(level, fix_info) is expected


def test_the_survivor_stamp_is_the_two_keys_the_ace_summaries_read():
    """`generate_samples` set these on its last lines; the ACE summaries read
    them to write "λ_min below threshold but decomposition succeeded"."""
    from kika.sampling.multigroup_draw import mark_soft_autofix_survived

    fix_info = {"converged": False, "soft_threshold_met": False,
                "min_eigenvalue": -1.0}
    mark_soft_autofix_survived(fix_info)
    assert fix_info["soft_autofix_failed"] is True
    assert fix_info["decomposition_succeeded"] is True

    # `if soft_autofix_failed and fix_info` — an autofix that never ran has no
    # dict to stamp, and stamping one would report a fix that did not happen.
    for empty in (None, {}):
        mark_soft_autofix_survived(empty)
        assert not empty


def test_autofix_none_is_the_identity_the_kika_configs_rely_on(carrier):
    """Every kika-side configuration passes ``autofix=None``.

    On that path `apply_legacy_autofix` must hand back the same object, not a
    copy and not a repaired matrix — the migrated call sites go on to assemble
    carrier blocks from whatever it returns.
    """
    from kika.sampling.multigroup_draw import apply_legacy_autofix

    cov, _grid, present = carrier
    same, mts, fix_info = apply_legacy_autofix(cov, None, mt_numbers=present)
    assert same is cov
    assert mts == list(present)
    assert fix_info is None
