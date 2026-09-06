import numpy as np
import pytest

from kika.UQ.alignment import align_sensitivity_covariance
from kika.cov.cross_section_covariance import CrossSectionCovariance
from kika.energy_grids.grids import SCALE44, SCALE56, SCALE238, SCALE252
from kika.sensitivities import (
    CondensationError,
    condense_sensitivity_profile,
)
from kika.sensitivities.profile import SensitivityProfile, SensitivityReaction


def make_profile(grid, *, unit="MeV", uncertainty=True):
    groups = len(grid) - 1
    values = np.linspace(-0.25, 0.75, groups)
    sigma = np.linspace(0.01, 0.02, groups) if uncertainty else None
    return SensitivityProfile(
        energy_grid=np.asarray(grid, dtype=float),
        energy_unit=unit,
        reactions=(SensitivityReaction(26056, 2, values, sigma, "elastic"),),
        response=1.001,
        response_uncertainty=2.0e-4,
        label="source",
        metadata={"origin": "test"},
    )


def test_exact_condensation_sums_sensitivity_and_propagates_absolute_sigma():
    profile = SensitivityProfile(
        energy_grid=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        reactions=(
            SensitivityReaction(
                26056,
                2,
                np.array([1.0, -2.0, 3.0, 4.0]),
                np.array([0.1, 0.2, 0.3, 0.4]),
                "elastic",
            ),
        ),
        response=1.001,
        response_uncertainty=2.0e-4,
        label="source",
    )

    result = condense_sensitivity_profile(profile, [1.0, 3.0, 5.0])
    reaction = result.profile.reactions[0]

    np.testing.assert_allclose(reaction.sensitivity, [-1.0, 7.0])
    np.testing.assert_allclose(
        reaction.uncertainty,
        [np.hypot(0.1, 0.2), np.hypot(0.3, 0.4)],
    )
    np.testing.assert_array_equal(
        result.operator,
        [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]],
    )
    assert result.report.boundary_indices == (0, 2, 4)
    assert result.report.max_integral_drift == pytest.approx(0.0)
    assert result.profile.response == profile.response
    assert result.profile.response_uncertainty == profile.response_uncertainty
    assert result.profile.label == profile.label
    assert (
        result.profile.metadata["condensation_history"][-1]["method"]
        == "exact_nested_sum"
    )


def test_unknown_uncertainty_remains_unknown():
    profile = make_profile([1.0, 2.0, 3.0], uncertainty=False)
    result = condense_sensitivity_profile(profile, [1.0, 3.0])

    assert result.profile.reactions[0].uncertainty is None
    assert result.report.uncertainty_method == "not available"
    assert result.report.assumptions == ()


def test_target_unit_is_explicit_and_preserved():
    profile = make_profile([1.0, 2.0, 3.0, 4.0], unit="MeV")
    result = condense_sensitivity_profile(
        profile,
        [1.0e6, 3.0e6, 4.0e6],
        target_energy_unit="eV",
    )

    assert result.profile.energy_unit == "eV"
    np.testing.assert_allclose(result.profile.energy_grid, [1.0e6, 3.0e6, 4.0e6])
    assert result.report.boundary_indices == (0, 2, 3)


def test_invalid_target_unit_is_not_treated_as_default():
    profile = make_profile([1.0, 2.0, 3.0])

    with pytest.raises(CondensationError, match="must be 'eV' or 'MeV'"):
        condense_sensitivity_profile(
            profile,
            [1.0, 3.0],
            target_energy_unit="",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "source_grid,target_grid,source_groups,target_groups",
    [
        (SCALE238, SCALE44, 238, 44),
        (SCALE252, SCALE56, 252, 56),
    ],
)
def test_standard_scale_nested_condensations(
    source_grid,
    target_grid,
    source_groups,
    target_groups,
):
    profile = make_profile(source_grid)
    result = condense_sensitivity_profile(profile, target_grid)

    assert result.report.source_groups == source_groups
    assert result.report.target_groups == target_groups
    assert result.operator.shape == (target_groups, source_groups)
    np.testing.assert_allclose(result.operator.sum(axis=0), 1.0)
    assert result.report.max_integral_drift < 1.0e-12


def test_scale_238_to_56_is_rejected_as_non_nested():
    profile = make_profile(SCALE238)

    with pytest.raises(
        CondensationError,
        match="non-nested grid requires a separate, explicitly requested projection",
    ):
        condense_sensitivity_profile(profile, SCALE56)


def test_dice_six_digit_boundary_rounding_is_accepted():
    dice_grid = np.asarray(SCALE238, dtype=float).copy()
    boundary = np.flatnonzero(np.isclose(dice_grid, 8.1e-6, rtol=0.0, atol=0.0))[0]
    dice_grid[boundary] = 8.09999e-6

    result = condense_sensitivity_profile(make_profile(dice_grid), SCALE44)

    assert result.report.target_groups == 44
    assert result.report.energy_rtol == pytest.approx(1.0e-5)


@pytest.mark.parametrize(
    "target,match",
    [
        ([1.0, 2.0, 2.0, 4.0], "strictly increasing"),
        ([2.0, 3.0, 4.0], "same energy range"),
        ([1.0, 2.5, 4.0], "every target boundary"),
    ],
)
def test_invalid_or_incompatible_target_grid_fails(target, match):
    profile = make_profile([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(CondensationError, match=match):
        condense_sensitivity_profile(profile, target)


def test_condensed_profile_aligns_without_implicit_grid_work():
    source = make_profile([1.0, 2.0, 3.0, 4.0])
    condensed = condense_sensitivity_profile(source, [1.0, 3.0, 4.0]).profile
    covariance = CrossSectionCovariance(
        num_groups=2,
        energy_grid=[1.0, 3.0, 4.0],
        energy_unit="MeV",
    )
    covariance.add_matrix(26056, 2, 26056, 2, np.eye(2), is_relative=True)

    aligned = align_sensitivity_covariance([condensed], covariance)

    assert aligned.sensitivity_vectors.shape == (1, 2)
    np.testing.assert_allclose(aligned.profiles[0].energy_grid, [1.0, 3.0, 4.0])
