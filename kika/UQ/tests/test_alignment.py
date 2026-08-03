import numpy as np
import pytest

from kika.UQ.alignment import (
    AlignmentError,
    MissingCovarianceError,
    ParameterKey,
    align_sensitivity_covariance,
)
from kika.cov.cross_section_covariance import CrossSectionCovariance
from kika.sensitivities.profile import SensitivityProfile, SensitivityReaction
from kika.sensitivities.sdf import SDFData, SDFReactionData


GRID_MEV = np.array([1.0e-6, 1.0e-3, 1.0])


def profile(*reactions, unit="MeV", grid=GRID_MEV, label="case"):
    return SensitivityProfile(
        energy_grid=np.asarray(grid),
        energy_unit=unit,
        reactions=tuple(
            SensitivityReaction(z, mt, np.asarray(s), np.asarray(e))
            for z, mt, s, e in reactions
        ),
        response=1.0,
        response_uncertainty=1.0e-4,
        label=label,
    )


def covariance(*blocks, relative=True, unit="MeV", grid=GRID_MEV):
    cov = CrossSectionCovariance(
        num_groups=len(grid) - 1,
        energy_grid=list(grid),
        energy_unit=unit,
    )
    for row, col, matrix in blocks:
        cov.add_matrix(*row, *col, np.asarray(matrix), is_relative=relative)
    return cov


def test_neutral_profile_validation_and_sdf_adapter():
    with pytest.raises(ValueError, match="duplicate"):
        SensitivityProfile(
            energy_grid=GRID_MEV,
            reactions=(
                SensitivityReaction(26056, 2, [1.0, 2.0], [0.1, 0.2]),
                SensitivityReaction(26056, 2, [3.0, 4.0], [0.3, 0.4]),
            ),
        )

    sdf = SDFData(
        title="descending", energy="test", pert_energies=[2.0, 1.0, 0.0],
        r0=1.0, e0=0.01,
        data=[
            SDFReactionData(26056, 2, [10.0, 20.0], [1.0, 2.0]),
            SDFReactionData(92235, 18, [30.0, 40.0], [3.0, 4.0], unit=1),
        ],
    )
    neutral = sdf.to_sensitivity_profile()
    np.testing.assert_allclose(neutral.energy_grid, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(neutral.reactions[0].sensitivity, [20.0, 10.0])
    np.testing.assert_allclose(neutral.reactions[0].uncertainty, [2.0, 1.0])
    assert len(neutral.reactions) == 1


def test_union_alignment_is_deterministic_and_records_inserted_zeros():
    a = profile((92235, 18, [1.0, 2.0], [0.1, 0.2]), label="a")
    b = profile(
        (26056, 2, [3.0, 4.0], [0.3, 0.4]),
        (92235, 18, [5.0, 6.0], [0.5, 0.6]),
        label="b",
    )
    cov = covariance(
        ((26056, 2), (26056, 2), np.eye(2)),
        ((92235, 18), (92235, 18), 2.0 * np.eye(2)),
    )

    result = align_sensitivity_covariance([a, b], cov)

    assert result.parameter_keys == (
        ParameterKey("xs", 26056, 2),
        ParameterKey("xs", 92235, 18),
    )
    np.testing.assert_allclose(result.sensitivity_vectors[0], [0.0, 0.0, 1.0, 2.0])
    assert result.report.zeros_inserted[0] == [ParameterKey("xs", 26056, 2)]
    assert result.report.zero_covariance_blocks == [
        (ParameterKey("xs", 26056, 2), ParameterKey("xs", 92235, 18))
    ]
    assert [(entry.key, entry.group) for entry in result.index] == [
        (ParameterKey("xs", 26056, 2), 0),
        (ParameterKey("xs", 26056, 2), 1),
        (ParameterKey("xs", 92235, 18), 0),
        (ParameterKey("xs", 92235, 18), 1),
    ]


def test_energy_units_are_explicit_and_partial_grid_fails():
    p = profile((92235, 18, [1.0, 2.0], [0.0, 0.0]))
    cov = covariance(
        ((92235, 18), (92235, 18), np.eye(2)),
        unit="eV",
        grid=GRID_MEV * 1.0e6,
    )
    result = align_sensitivity_covariance([p], cov)
    assert result.covariance.shape == (2, 2)
    assert any("energy grid converted from eV to MeV" in note for note in result.report.assumptions)

    cov.energy_grid[-1] *= 0.9
    with pytest.raises(AlignmentError, match="explicitly condense"):
        align_sensitivity_covariance([p], cov)


def test_absolute_covariance_is_converted_with_cross_sections():
    p = profile((92235, 18, [1.0, 2.0], [0.0, 0.0]))
    cov = covariance(
        ((92235, 18), (92235, 18), np.diag([4.0, 36.0])),
        relative=False,
    )
    cov.cross_sections[(92235, 18)] = np.array([2.0, 3.0])

    result = align_sensitivity_covariance([p], cov)
    np.testing.assert_allclose(result.covariance, np.diag([1.0, 4.0]))
    assert any("converted from absolute" in note for note in result.report.assumptions)

    cov.cross_sections.clear()
    with pytest.raises(AlignmentError, match="requires row and column cross sections"):
        align_sensitivity_covariance([p], cov)


def test_unmarked_covariance_is_relative_and_reported():
    p = profile((92235, 18, [1.0, 2.0], [0.0, 0.0]))
    cov = CrossSectionCovariance(num_groups=2, energy_grid=list(GRID_MEV), energy_unit="MeV")
    cov.isotope_rows.append(92235)
    cov.reaction_rows.append(18)
    cov.isotope_cols.append(92235)
    cov.reaction_cols.append(18)
    cov.matrices.append(np.eye(2))

    result = align_sensitivity_covariance([p], cov)
    assert "treated as relative" in result.report.assumptions[0]


@pytest.mark.parametrize(
    "source,target",
    [((4309, 2), (4009, 2)), ((1901, 2), (1001, 2)), ((6312, 2), (6000, 2)),
     ((92235, 101), (92235, 102)), ((92235, -2), (92235, 101))],
)
def test_tsurfer_aliases_are_explicit(source, target):
    p = profile((source[0], source[1], [1.0, 2.0], [0.0, 0.0]))
    cov = covariance((target, target, np.eye(2)))

    with pytest.raises(MissingCovarianceError):
        align_sensitivity_covariance([p], cov)
    result = align_sensitivity_covariance([p], cov, alias_policy="tsurfer")
    assert result.parameter_keys == (ParameterKey("xs", *target),)
    assert result.report.aliases[0].source == ParameterKey("xs", *source)


def test_missing_is_strict_and_drop_reports_coverage():
    p = profile(
        (26056, 2, [3.0, 0.0], [0.0, 0.0]),
        (92235, 18, [1.0, 0.0], [0.0, 0.0]),
    )
    cov = covariance(((92235, 18), (92235, 18), np.eye(2)))

    with pytest.raises(MissingCovarianceError, match='missing="drop"') as error:
        align_sensitivity_covariance([p], cov)
    assert error.value.report.missing_covariance == [ParameterKey("xs", 26056, 2)]

    result = align_sensitivity_covariance([p], cov, missing="drop")
    assert result.report.parameter_coverage == [0.5]
    assert result.report.sensitivity_coverage == [0.25]


def test_alias_collision_fails_loudly():
    p = profile(
        (4309, 2, [1.0, 0.0], [0.0, 0.0]),
        (4509, 2, [2.0, 0.0], [0.0, 0.0]),
    )
    cov = covariance(((4009, 2), (4009, 2), np.eye(2)))
    with pytest.raises(AlignmentError, match="alias collision"):
        align_sensitivity_covariance([p], cov, alias_policy="tsurfer")


def test_other_negative_mt_is_rejected():
    p = profile((92235, -3, [1.0, 0.0], [0.0, 0.0]))
    cov = covariance(((92235, 18), (92235, 18), np.eye(2)))
    with pytest.raises(AlignmentError, match="unsupported negative"):
        align_sensitivity_covariance([p], cov, alias_policy="tsurfer")


def test_exact_key_has_priority_over_tsurfer_fallback():
    p = profile((92235, 101, [1.0, 2.0], [0.0, 0.0]))
    cov = covariance(
        ((92235, 101), (92235, 101), np.eye(2)),
        ((92235, 102), (92235, 102), 2.0 * np.eye(2)),
    )
    result = align_sensitivity_covariance([p], cov, alias_policy="tsurfer")
    assert result.parameter_keys == (ParameterKey("xs", 92235, 101),)
    assert result.report.aliases == []


def test_exact_sensitivity_wins_over_competing_mt_alias():
    p = profile(
        (92235, 101, [10.0, 20.0], [1.0, 2.0]),
        (92235, 102, [1.0, 2.0], [0.1, 0.2]),
    )
    cov = covariance(((92235, 102), (92235, 102), np.eye(2)))

    result = align_sensitivity_covariance([p], cov, alias_policy="tsurfer")

    np.testing.assert_allclose(result.sensitivity_vectors[0], [1.0, 2.0])
    assert result.report.aliases == []
    assert result.report.policy_exclusions[0] == [ParameterKey("xs", 92235, 101)]
    assert any("preferred over" in note for note in result.report.assumptions)


def test_same_mt_bound_alias_wins_over_bound_mt_alias():
    p = profile(
        (1801, 101, [10.0, 20.0], [1.0, 2.0]),
        (1801, 102, [1.0, 2.0], [0.1, 0.2]),
    )
    cov = covariance(((1001, 102), (1001, 102), np.eye(2)))

    result = align_sensitivity_covariance([p], cov, alias_policy="tsurfer")

    np.testing.assert_allclose(result.sensitivity_vectors[0], [1.0, 2.0])
    assert result.report.aliases[0].source == ParameterKey("xs", 1801, 102)
    assert result.report.policy_exclusions[0] == [ParameterKey("xs", 1801, 101)]


def test_incompatible_duplicate_covariance_block_fails():
    p = profile((92235, 18, [1.0, 2.0], [0.0, 0.0]))
    first = covariance(((92235, 18), (92235, 18), np.eye(2)))
    second = covariance(((92235, 18), (92235, 18), 2.0 * np.eye(2)))
    with pytest.raises(AlignmentError, match="incompatible duplicate"):
        align_sensitivity_covariance([p], [first, second])


def test_float32_symmetry_noise_is_removed_and_reported():
    p = profile((92235, 18, [1.0, 2.0], [0.0, 0.0]))
    matrix = np.array([[1.0, 3.1388999e-4], [3.1390000e-4, 2.0]])
    cov = covariance(((92235, 18), (92235, 18), matrix))

    result = align_sensitivity_covariance([p], cov)

    np.testing.assert_array_equal(result.covariance, result.covariance.T)
    np.testing.assert_allclose(result.covariance[0, 1], 3.13894995e-4)
    assert any("float32 rounding tolerance" in note for note in result.report.assumptions)


def test_materially_asymmetric_diagonal_covariance_fails():
    p = profile((92235, 18, [1.0, 2.0], [0.0, 0.0]))
    cov = covariance(
        ((92235, 18), (92235, 18), np.array([[1.0, 0.25], [0.30, 2.0]]))
    )

    with pytest.raises(AlignmentError, match="asymmetric diagonal"):
        align_sensitivity_covariance([p], cov)
