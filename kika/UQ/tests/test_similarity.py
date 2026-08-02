import json
from pathlib import Path

import numpy as np
import pytest

from kika.UQ.alignment import ParameterKey
from kika.UQ.similarity import ZeroSimilarityVarianceError, similarity_ck
from kika.cov.cross_section_covariance import CrossSectionCovariance
from kika.sensitivities.profile import SensitivityProfile, SensitivityReaction


GRID = np.array([0.0, 1.0, 2.0])


def make_profile(vector, label="case"):
    return SensitivityProfile(
        energy_grid=GRID,
        energy_unit="MeV",
        reactions=(
            SensitivityReaction(26056, 2, vector[:2], [0.0, 0.0]),
            SensitivityReaction(92235, 18, vector[2:], [0.0, 0.0]),
        ),
        label=label,
    )


def make_covariance():
    cov = CrossSectionCovariance(num_groups=2, energy_grid=list(GRID), energy_unit="MeV")
    cov.add_matrix(26056, 2, 26056, 2, np.array([[0.04, 0.01], [0.01, 0.09]]), is_relative=True)
    cov.add_matrix(26056, 2, 92235, 18, np.array([[0.002, 0.0], [0.0, 0.003]]), is_relative=True)
    cov.add_matrix(92235, 18, 92235, 18, np.array([[0.16, 0.02], [0.02, 0.25]]), is_relative=True)
    return cov


def test_ck_positive_and_negative_unit_limits():
    a = make_profile(np.array([0.2, -0.1, 0.3, 0.4]))
    cov = make_covariance()
    assert similarity_ck(a, a, cov).value == pytest.approx(1.0)
    assert similarity_ck(a, make_profile(-np.array([0.2, -0.1, 0.3, 0.4])), cov).value == pytest.approx(-1.0)


def test_ck_matches_frozen_calins_reference():
    reference_path = Path(__file__).parent / "data" / "calins_ck_reference.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    result = similarity_ck(
        make_profile(np.array([0.2, -0.1, 0.3, 0.4]), "a"),
        make_profile(np.array([0.1, 0.2, -0.2, 0.5]), "b"),
        make_covariance(),
    )
    assert result.variance_a == pytest.approx(reference["variance_a"])
    assert result.variance_b == pytest.approx(reference["variance_b"])
    assert result.cross_covariance == pytest.approx(reference["cross_covariance"])
    assert result.value == pytest.approx(reference["ck"])
    assert result.ck == pytest.approx(reference["ck"])
    assert len(result.index) == 4
    assert len(result.reaction_similarity) == 2


def test_ck_zero_covariance_norm_is_an_error():
    zero = make_profile(np.zeros(4))
    with pytest.raises(ZeroSimilarityVarianceError, match="zero or negative"):
        similarity_ck(zero, make_profile(np.ones(4)), make_covariance())


def test_similarity_missing_policy_is_forwarded():
    one_reaction_cov = CrossSectionCovariance(
        num_groups=2, energy_grid=list(GRID), energy_unit="MeV"
    )
    one_reaction_cov.add_matrix(
        26056, 2, 26056, 2, np.eye(2), is_relative=True
    )
    a = make_profile(np.array([1.0, 1.0, 3.0, 0.0]))
    b = make_profile(np.array([1.0, 0.0, 1.0, 0.0]))
    with pytest.raises(ValueError, match='missing="drop"'):
        similarity_ck(a, b, one_reaction_cov)
    result = similarity_ck(a, b, one_reaction_cov, missing="drop")
    assert result.alignment_report.parameter_coverage == [0.5, 0.5]


def test_similarity_records_policy_exclusions():
    raw = SensitivityProfile(
        energy_grid=GRID,
        energy_unit="MeV",
        reactions=(
            SensitivityReaction(26056, 1, [9.0, 9.0], [0.0, 0.0]),
            SensitivityReaction(26056, 2, [1.0, 2.0], [0.0, 0.0]),
        ),
    )
    cov = CrossSectionCovariance(num_groups=2, energy_grid=list(GRID), energy_unit="MeV")
    cov.add_matrix(26056, 2, 26056, 2, np.eye(2), is_relative=True)
    result = similarity_ck(raw, raw, cov)
    assert result.alignment_report.policy_exclusions == {
        0: [ParameterKey("xs", 26056, 1)],
        1: [ParameterKey("xs", 26056, 1)],
    }
