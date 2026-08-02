"""
ICSBEP/DICE benchmark sensitivities for KIKA.

Build a local, self-contained database from your own NEA-licensed DICE sensitivity
data, then screen and rank benchmarks by their sensitivity to a given isotope,
reaction and energy region. kika ships no benchmark data itself.

Quick start:
    >>> import kika.benchmarks as benchmarks
    >>> # One-time build from the DICE sensitivity/ folder:
    >>> benchmarks.build_benchmarks_db(source_dir="/path/to/DiceData/sensitivity",
    ...                                db_path="/path/to/kika_benchmarks.db")
    >>> # Then point kika at it (or set KIKA_BENCHMARKS_DB_PATH):
    >>> benchmarks.configure(db_path="/path/to/kika_benchmarks.db")
    >>> hits = benchmarks.find_sensitive_benchmarks("Fe-56", reaction="elastic",
    ...                                             energy_region="fast")
"""

from kika.benchmarks.config import (
    configure,
    get_config,
    get_db_path,
    get_source_dir,
    reset_config,
)
from kika.benchmarks.database import BenchmarksDatabase
from kika.benchmarks.exceptions import (
    BenchmarkNotFoundError,
    BenchmarksError,
    DatabaseNotConfiguredError,
)
from kika.benchmarks.ingest import build_benchmarks_db
from kika.benchmarks.plotting import plot_profile
from kika.benchmarks.uq import (
    BenchmarkSimilarity,
    benchmark_uncertainty,
    rank_benchmarks_by_ck,
    similarity_ck,
)
from kika.benchmarks.screening import (
    find_sensitive_benchmarks,
    get_balance,
    get_benchmark,
    get_benchmark_dashboard,
    get_experimental_keff,
    get_input_deck,
    get_profile_vector,
    get_sensitivity_profile,
    get_spectrum,
    rank_benchmarks,
)

__all__ = [
    "configure",
    "get_config",
    "get_db_path",
    "get_source_dir",
    "reset_config",
    "BenchmarksDatabase",
    "build_benchmarks_db",
    "find_sensitive_benchmarks",
    "rank_benchmarks",
    "get_benchmark",
    "get_benchmark_dashboard",
    "get_profile_vector",
    "get_sensitivity_profile",
    "get_experimental_keff",
    "get_spectrum",
    "get_balance",
    "get_input_deck",
    "plot_profile",
    "BenchmarkSimilarity",
    "benchmark_uncertainty",
    "similarity_ck",
    "rank_benchmarks_by_ck",
    "BenchmarksError",
    "DatabaseNotConfiguredError",
    "BenchmarkNotFoundError",
]
