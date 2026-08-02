"""
Public screening API for the benchmarks database.

Find and rank ICSBEP/DICE benchmarks by their sensitivity to a given isotope,
reaction, and energy region — the core "which benchmarks matter for nuclide X"
question.
"""

from typing import List, Optional, Union

from kika._constants import MT_TO_REACTION
from kika._utils import symbol_to_zaid
from kika.benchmarks._constants import REACTION_ALIASES
from kika.benchmarks.database import BenchmarksDatabase

# Reverse of MT_TO_REACTION for name -> MT resolution (lowercased canonical labels).
_REACTION_NAME_TO_MT = {v.lower(): k for k, v in MT_TO_REACTION.items()}


def _normalize_isotope(isotope: Union[str, int]) -> int:
    """Normalize an isotope to a ZAID. Accepts 26056, 'Fe56', or 'Fe-56'."""
    if isinstance(isotope, int):
        return isotope
    if isinstance(isotope, str):
        return symbol_to_zaid(isotope.replace("-", "").strip())
    raise TypeError(f"isotope must be int or str, got {type(isotope).__name__}")


def _normalize_reaction(reaction: Optional[Union[str, int]]) -> Optional[int]:
    """Normalize a reaction to an MT integer. Accepts 2, 'elastic', '(n,el)', None."""
    if reaction is None:
        return None
    if isinstance(reaction, int):
        return reaction
    if isinstance(reaction, str):
        key = reaction.strip().lower()
        if key.isdigit():
            return int(key)
        if key in REACTION_ALIASES:
            return REACTION_ALIASES[key]
        if key in _REACTION_NAME_TO_MT:
            return _REACTION_NAME_TO_MT[key]
        raise ValueError(f"Unrecognized reaction: {reaction!r}")
    raise TypeError(f"reaction must be int, str or None, got {type(reaction).__name__}")


def find_sensitive_benchmarks(
    isotope: Union[str, int],
    reaction: Optional[Union[str, int]] = None,
    energy_region: str = "total",
    sensitivity_threshold: float = 0.0,
    fraction_threshold: Optional[float] = None,
    preferred_only: bool = True,
    limit: int = 100,
    db_path: Optional[str] = None,
) -> List[dict]:
    """
    Find benchmarks sensitive to an isotope/reaction in an energy region.

    Parameters
    ----------
    isotope : str or int
        Isotope as ZAID (26056), symbol ('Fe56') or hyphenated symbol ('Fe-56').
    reaction : str or int, optional
        Reaction as MT (2), alias ('elastic', 'capture') or canonical label
        ('(n,el)'). None matches any reaction of the isotope.
    energy_region : str
        'thermal', 'epithermal', 'fast' or 'total' (default).
    sensitivity_threshold : float
        Minimum absolute integral sensitivity in the region.
    fraction_threshold : float, optional
        Minimum |region sensitivity| / (sum of |sensitivity|) for the reaction.
    preferred_only : bool
        Only consider the preferred profile per benchmark (default True).
    limit : int
        Maximum number of results.
    db_path : str, optional
        Explicit database path (otherwise resolved from config/env/default).

    Returns
    -------
    list of dict
        One row per matching benchmark reaction, ranked by absolute sensitivity
        (descending). Each row includes benchmark id/category/spectrum, profile
        code/library, keff, the region sensitivity and its fraction of the total.
    """
    zaid = _normalize_isotope(isotope)
    mt = _normalize_reaction(reaction)
    with BenchmarksDatabase(db_path) as db:
        return db.screen(
            zaid=zaid,
            mt=mt,
            region=energy_region,
            sensitivity_threshold=sensitivity_threshold,
            fraction_threshold=fraction_threshold,
            preferred_only=preferred_only,
            limit=limit,
        )


def rank_benchmarks(
    isotope: Union[str, int],
    reaction: Optional[Union[str, int]] = None,
    energy_region: str = "total",
    limit: int = 100,
    db_path: Optional[str] = None,
) -> List[dict]:
    """Rank benchmarks by absolute sensitivity to an isotope/reaction/region.

    Convenience wrapper over :func:`find_sensitive_benchmarks` with no thresholds.
    """
    return find_sensitive_benchmarks(
        isotope,
        reaction=reaction,
        energy_region=energy_region,
        sensitivity_threshold=0.0,
        fraction_threshold=None,
        limit=limit,
        db_path=db_path,
    )


def get_benchmark(benchmark_id: str, db_path: Optional[str] = None) -> dict:
    """Return a benchmark's metadata and its list of profile variants."""
    with BenchmarksDatabase(db_path) as db:
        return db.get_benchmark(benchmark_id)


def get_benchmark_dashboard(benchmark_id: str, db_path: Optional[str] = None) -> dict:
    """Return the whole per-benchmark record (metadata, keff, vectors, spectrum,
    balance, input codes) in a single bundle for the app dashboard."""
    with BenchmarksDatabase(db_path) as db:
        return db.get_benchmark_dashboard(benchmark_id)


def get_profile_vector(
    profile_id: int,
    zaid: Optional[int] = None,
    mt: Optional[int] = None,
    db_path: Optional[str] = None,
) -> dict:
    """Return a profile's full per-group sensitivity vectors (SDFSensitivityData shape)."""
    with BenchmarksDatabase(db_path) as db:
        return db.get_profile_vector(profile_id, zaid=zaid, mt=mt)


def get_sensitivity_profile(profile_id: int, db_path: Optional[str] = None):
    """Return a profile as a validated format-neutral sensitivity object."""
    with BenchmarksDatabase(db_path) as db:
        return db.get_sensitivity_profile(profile_id)


def get_experimental_keff(benchmark_id: str, db_path: Optional[str] = None) -> dict:
    """Return a benchmark's experimental keff ``{'keff', 'keff_unc'}`` (empty if unknown)."""
    with BenchmarksDatabase(db_path) as db:
        return db.get_experimental_keff(benchmark_id)


def get_spectrum(benchmark_id: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Return a benchmark's 299-group flux spectrum ``{'energies', 'flux'}`` or None."""
    with BenchmarksDatabase(db_path) as db:
        return db.get_spectrum(benchmark_id)


def get_balance(benchmark_id: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Return a benchmark's reaction-rate balance ``{'summary', 'text'}`` or None."""
    with BenchmarksDatabase(db_path) as db:
        return db.get_balance(benchmark_id)


def get_input_deck(
    benchmark_id: str, code: Optional[str] = None, db_path: Optional[str] = None
) -> list:
    """Return a benchmark's raw input deck(s) as ``[{'code', 'text'}]`` (verbatim)."""
    with BenchmarksDatabase(db_path) as db:
        return db.get_input_deck(benchmark_id, code=code)
