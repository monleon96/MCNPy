"""Optional end-to-end regression against the real DICE/SCALE CALINS case."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from kika.UQ.sandwich import sandwich_uncertainty_propagation
from kika.UQ.similarity import similarity_ck
from kika.benchmarks.database import BenchmarksDatabase
from kika.cov.parse_covmat import read_coverx
from kika.energy_grids import SCALE44
from kika.sensitivities.condensation import condense_sensitivity_profile


DB_PATH = Path(
    os.environ.get(
        "KIKA_TEST_BENCHMARK_DB",
        Path.home() / ".kika/benchmarks/kika_benchmarks.db",
    )
)
COVERX_PATH = Path(
    os.environ.get(
        "KIKA_TEST_SCALE44_COVERX",
        "/share_snc/projets/INSIDER/SENSIBILITE/"
        "Covariances_scale_6.2.3_binary/scale.rev05.44groupcov",
    )
)
REFERENCE_PATH = Path(__file__).parent / "data/dice_scale44_calins_reference.json"

pytestmark = pytest.mark.skipif(
    not DB_PATH.is_file() or not COVERX_PATH.is_file(),
    reason="real schema-v3 DICE database or SCALE-44 COVERX unavailable",
)


def test_real_dice_scale44_sandwich_and_ck_match_frozen_calins_reference():
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    with BenchmarksDatabase(str(DB_PATH)) as database:
        application = database.get_sensitivity_profile(
            database.get_preferred_profile(
                reference["provenance"]["application_benchmark"]
            )["profile_id"]
        )
        benchmark = database.get_sensitivity_profile(
            database.get_preferred_profile(
                reference["provenance"]["comparison_benchmark"]
            )["profile_id"]
        )

    application = condense_sensitivity_profile(
        application, SCALE44, target_energy_unit="MeV"
    ).profile
    benchmark = condense_sensitivity_profile(
        benchmark, SCALE44, target_energy_unit="MeV"
    ).profile
    covariance = read_coverx(str(COVERX_PATH), energy_unit="MeV")

    uncertainty = sandwich_uncertainty_propagation(
        application,
        cov_mat=covariance,
        bootstrap=False,
        alias_policy="tsurfer",
        missing="drop",
    )
    similarity = similarity_ck(
        application,
        benchmark,
        covariance,
        alias_policy="tsurfer",
        missing="drop",
    )

    expected = reference["kika"]
    assert uncertainty.total_variance == pytest.approx(
        expected["application_variance"], rel=1.0e-11
    )
    assert uncertainty.total_uncertainty == pytest.approx(
        expected["application_sigma_relative"], rel=1.0e-11
    )
    assert uncertainty.total_uncertainty * 1.0e5 * uncertainty.response_value == pytest.approx(
        expected["application_sigma_pcm"], rel=1.0e-11
    )
    assert uncertainty.n_reactions == expected["application_parameters"]
    assert uncertainty.alignment_report.parameter_coverage == pytest.approx(
        [expected["application_parameter_coverage"]]
    )
    assert uncertainty.alignment_report.sensitivity_coverage == pytest.approx(
        [expected["application_sensitivity_coverage"]]
    )

    assert similarity.value == pytest.approx(expected["ck"], rel=1.0e-11)
    assert similarity.variance_a == pytest.approx(
        expected["ck_variance_application"], rel=1.0e-11
    )
    assert similarity.variance_b == pytest.approx(
        expected["ck_variance_benchmark"], rel=1.0e-11
    )
    assert similarity.cross_covariance == pytest.approx(
        expected["ck_cross_covariance"], rel=1.0e-11
    )
    assert len(similarity.parameter_keys) == expected["ck_parameters"]

    calins = reference["calins"]
    assert uncertainty.total_uncertainty == pytest.approx(
        calins["application_sigma_relative"], abs=5.0e-10
    )
    assert similarity.value == pytest.approx(calins["ck"], abs=5.0e-10)
