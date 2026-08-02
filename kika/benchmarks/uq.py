"""Thin benchmark-database adapters for Kika's format-neutral UQ APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

from kika.UQ.alignment import AlignmentError, prepare_covariance
from kika.UQ.sandwich import UncertaintyResult, sandwich_uncertainty_propagation
from kika.UQ.similarity import SimilarityResult, similarity_ck as uq_similarity_ck
from kika.benchmarks.database import BenchmarksDatabase
from kika.sensitivities.profile import SensitivityProfile
from kika.sensitivities.sdf import SDFData


@dataclass(frozen=True)
class BenchmarkSimilarity:
    """One deterministic row in a benchmark c-k ranking."""

    benchmark_id: str
    profile_id: int
    ck: float
    n_parameters: int
    application_parameter_coverage: float
    benchmark_parameter_coverage: float
    application_sensitivity_coverage: float
    benchmark_sensitivity_coverage: float
    code: Optional[str] = None
    library: Optional[str] = None
    group_structure: Optional[str] = None
    result: Optional[SimilarityResult] = None


def benchmark_uncertainty(
    profile_id: int,
    covariance=None,
    legendre_covariance=None,
    *,
    db_path: Optional[str] = None,
    **kwargs,
) -> UncertaintyResult:
    """Run sandwich propagation for one SQLite benchmark profile."""
    with BenchmarksDatabase(db_path) as database:
        profile = database.get_sensitivity_profile(profile_id)
    return sandwich_uncertainty_propagation(
        profile,
        cov_mat=covariance,
        legendre_cov_mat=legendre_covariance,
        **kwargs,
    )


def similarity_ck(
    application: Union[SensitivityProfile, SDFData],
    benchmark_profile_id: int,
    covariance=None,
    legendre_covariance=None,
    *,
    db_path: Optional[str] = None,
    **kwargs,
) -> SimilarityResult:
    """Calculate c-k between an application and one SQLite benchmark profile."""
    with BenchmarksDatabase(db_path) as database:
        benchmark = database.get_sensitivity_profile(benchmark_profile_id)
    return uq_similarity_ck(
        application,
        benchmark,
        covariance,
        legendre_covariance,
        **kwargs,
    )


def rank_benchmarks_by_ck(
    application: Union[SensitivityProfile, SDFData],
    covariance=None,
    legendre_covariance=None,
    *,
    benchmark_ids: Optional[Sequence[str]] = None,
    preferred_only: bool = True,
    limit: Optional[int] = 100,
    rank_by_absolute: bool = False,
    db_path: Optional[str] = None,
    alias_policy: str = "exact",
    missing: str = "error",
    include=None,
    exclude=None,
    include_mt1: bool = False,
    nubar_mode="total",
    energy_tolerance: float = 1.0e-8,
) -> List[BenchmarkSimilarity]:
    """Rank SQLite benchmark profiles by covariance-weighted c-k.

    Covariance normalization is performed once. In strict mode any candidate
    that cannot be aligned aborts the ranking with its benchmark/profile label;
    no failed candidate is silently omitted.
    """
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    neutral_application = (
        application.to_sensitivity_profile() if isinstance(application, SDFData) else application
    )
    if not isinstance(neutral_application, SensitivityProfile):
        raise TypeError("application must be SDFData or SensitivityProfile")
    prepared = prepare_covariance(
        neutral_application,
        covariance,
        legendre_covariance,
        energy_rtol=energy_tolerance,
    )

    rows: List[BenchmarkSimilarity] = []
    with BenchmarksDatabase(db_path) as database:
        candidates = database.list_similarity_profiles(
            list(benchmark_ids) if benchmark_ids is not None else None,
            preferred_only=preferred_only,
        )
        for candidate in candidates:
            profile = database.get_sensitivity_profile(candidate["profile_id"])
            try:
                result = uq_similarity_ck(
                    neutral_application,
                    profile,
                    alias_policy=alias_policy,
                    missing=missing,
                    include=include,
                    exclude=exclude,
                    include_mt1=include_mt1,
                    nubar_mode=nubar_mode,
                    energy_tolerance=energy_tolerance,
                    prepared_covariance=prepared,
                )
            except AlignmentError as error:
                raise AlignmentError(
                    f"Cannot rank benchmark {candidate['benchmark_id']} "
                    f"(profile_id={candidate['profile_id']}): {error}",
                    getattr(error, "report", None),
                ) from error
            report = result.alignment_report
            rows.append(
                BenchmarkSimilarity(
                    benchmark_id=candidate["benchmark_id"],
                    profile_id=candidate["profile_id"],
                    ck=result.value,
                    n_parameters=len(result.parameter_keys),
                    application_parameter_coverage=report.parameter_coverage[0],
                    benchmark_parameter_coverage=report.parameter_coverage[1],
                    application_sensitivity_coverage=report.sensitivity_coverage[0],
                    benchmark_sensitivity_coverage=report.sensitivity_coverage[1],
                    code=candidate["code"],
                    library=candidate["library"],
                    group_structure=candidate["group_structure"],
                    result=result,
                )
            )

    rows.sort(
        key=lambda row: (
            -(abs(row.ck) if rank_by_absolute else row.ck),
            row.benchmark_id,
            row.profile_id,
        )
    )
    return rows if limit is None else rows[:limit]
