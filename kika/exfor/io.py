"""
EXFOR data loading functions.

This module provides the main I/O functions for loading EXFOR data from
JSON files or the X4Pro database, and organizing datasets by energy.

Supports multiple data sources:
- JSON files: Local JSON files in standardized format
- Database: X4Pro SQLite database (full 2025 version with JSON schema)
- Auto: Database with JSON fallback for missing entries
"""

import os
import glob
import math
from typing import Dict, List, Optional, Tuple, Union

from kika.exfor._constants import ENERGY_MATCH_ABS_TOL
from kika.exfor.config import get_db_path
from kika.exfor.angular_distribution import ExforAngularDistribution

def read_exfor(filepath: str) -> "ExforAngularDistribution":
    """
    Read an EXFOR angular distribution from a JSON file.

    This is the primary function for loading EXFOR data in kika. It supports
    both the standardized v1.0 schema and legacy formats.

    Parameters:
        filepath: Path to the JSON file

    Returns:
        ExforAngularDistribution object

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid

    Example:
        >>> from kika.exfor import read_exfor
        >>> exfor = read_exfor('/path/to/27673002.json')
        >>> print(exfor.label)
        Gkatis et al. (2025)
        >>> print(exfor.energies())
        [1.0098, 1.0202, ...]
    """
    from kika.exfor.angular_distribution import ExforAngularDistribution

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"EXFOR file not found: {filepath}")

    ad = ExforAngularDistribution.from_json(filepath)
    # Manifest application is pipeline-specific and lives in
    # scripts/uncertainty_manifest.py. The kika library returns the raw
    # ExforAngularDistribution; the pipeline (e.g. build_exfor_cache_from_objects)
    # calls apply_manifest_to_exfor() before building the cache.
    # JSON path: no raw EXFOR columns. The pipeline resolver treats an empty
    # list the same as None and synthesizes a DATA-ERR component from the
    # per-point uncertainty_stat -- see apply_manifest_to_exfor in
    # scripts/uncertainty_manifest.py.
    ad._raw_uncertainty_components = []
    ad.sigma_sys_scalar_relative = 0.0
    ad.sigma_sys_indep_relative = 0.0
    ad.uncertainty_manifest_flag = "default"
    return ad


def read_all_exfor(
    target: Union[str, int, List[str], List[int]] = None,
    mt: int = None,
    source: str = "database",
    directory: str = None,
    pattern: str = "*.json",
    group_by_energy: bool = True,
    db_path: str = None,
    projectile: str = "n",
    energy_range: Tuple[float, float] = None,
    supplementary_json_files: List[str] = None,
    return_load_status: bool = False,
    exclude_experiments: List[str] = None,
    # Deprecated parameter for backwards compatibility
    target_zaid: Union[int, List[int]] = None,
) -> Union[Dict[float, List], Dict[str, "ExforAngularDistribution"], Tuple[Dict, Dict]]:
    """
    Read EXFOR angular distribution data.

    This is the main function for loading EXFOR experimental data. By default,
    it loads from the X4Pro database (189k+ datasets). You can also load from
    your own curated JSON files.

    Parameters
    ----------
    target : str, int, List[str], or List[int], optional
        Target isotope(s). Accepts multiple formats:
        - "Fe56" (symbol + mass)
        - 26056 (ZAID)
        - "Fe-56" (database format)
        - "26-FE-56" (EXFOR notation)
        - [26056, 26000] (list of ZAIDs for multiple targets)
        - ["Fe-56", "Fe-0"] (list of targets)
    mt : int, optional
        ENDF MT number (e.g., 2 for elastic scattering)
    source : str, optional
        Data source (default: "database"):
        - "database": Load from X4Pro database
        - "json": Load from JSON files only
        - "auto": Database with JSON fallback
    directory : str, optional
        Path to JSON files (required if source="json")
    group_by_energy : bool, optional
        If True (default), group by energy. If False, return dict by ID.
    db_path : str, optional
        Path to X4Pro database (uses default if None)
    projectile : str, optional
        Projectile (default: "n" for neutrons)
    energy_range : Tuple[float, float], optional
        Energy range (min, max) in MeV
    supplementary_json_files : List[str], optional
        List of additional JSON file paths to load. Useful for loading
        experiments not yet in the database (e.g., recent publications).
        These are loaded after the main source and deduplicated by ID.
    return_load_status : bool, optional
        If True, return a tuple of (data, supplementary_status) where
        supplementary_status is a dict with 'loaded', 'failed', and 'skipped'
        lists. Useful for logging which supplementary files were processed.
    exclude_experiments : List[str], optional
        List of experiments to exclude from loading. Accepts multiple formats:
        - "20743" - excludes all subentries starting with 20743
        - "20743002" - excludes specific dataset
        - "20743/002" - same as above

    Returns
    -------
    Dict[float, List[ExforAngularDistribution]] or Tuple
        Dictionary mapping energy (MeV) to list of datasets.
        If group_by_energy=False: Dict mapping ID to dataset.
        If return_load_status=True: Tuple of (data, supplementary_status).

    Examples
    --------
    >>> # Load Fe-56 elastic scattering data
    >>> data = read_all_exfor(target="Fe56", mt=2)

    >>> # Load using ZAID
    >>> data = read_all_exfor(target=26056, mt=2)

    >>> # Load both Fe-56 and natural iron (recommended for better data coverage)
    >>> data = read_all_exfor(target=[26056, 26000], mt=2)

    >>> # Load only from your curated JSON files
    >>> data = read_all_exfor(source="json", directory="/path/to/data/")

    >>> # Load with energy filter
    >>> data = read_all_exfor(target="Fe56", energy_range=(1.0, 3.0))

    >>> # Load from database plus a supplementary JSON file
    >>> data = read_all_exfor(
    ...     target="Fe56", mt=2,
    ...     supplementary_json_files=["C:/EXFOR/27673002.json"]
    ... )
    """
    from kika.exfor.angular_distribution import ExforAngularDistribution

    source = source.lower()
    valid_sources = {"json", "database", "auto", "both"}
    if source not in valid_sources:
        raise ValueError(f"Invalid source '{source}'. Must be one of: {valid_sources}")

    # Handle backwards compatibility: target_zaid -> target
    if target_zaid is not None and target is None:
        target = target_zaid

    all_datasets: List[ExforAngularDistribution] = []
    loaded_ids: set = set()

    # Load from database if requested
    if source in ("database", "auto", "both"):
        try:
            db_datasets = _load_from_database(
                db_path=db_path,
                target=target,
                projectile=projectile,
                mt=mt,
                energy_range=energy_range,
            )
            for ds in db_datasets:
                ds_id = f"{ds.entry}{ds.subentry}"
                if ds_id not in loaded_ids:
                    all_datasets.append(ds)
                    loaded_ids.add(ds_id)
        except FileNotFoundError as e:
            if source == "database":
                raise
            # For auto/both, continue to JSON fallback
            print(f"Warning: Database not available, falling back to JSON: {e}")
        except Exception as e:
            if source == "database":
                raise
            print(f"Warning: Database query failed: {e}")

    # Load from JSON files if requested
    if source in ("json", "auto", "both"):
        if directory is not None:
            try:
                json_datasets = _load_from_json_files(directory, pattern)
                for ds in json_datasets:
                    ds_id = f"{ds.entry}{ds.subentry}"
                    if ds_id not in loaded_ids:
                        all_datasets.append(ds)
                        loaded_ids.add(ds_id)
            except FileNotFoundError as e:
                if source == "json":
                    raise
                # For auto/both, database data is sufficient
                if not all_datasets:
                    raise FileNotFoundError(
                        f"No data available. Database not found and {e}"
                    )

    # Load supplementary JSON files (for experiments not in database)
    supplementary_status = {'loaded': [], 'failed': [], 'skipped': []}

    if supplementary_json_files:
        for json_file in supplementary_json_files:
            if not os.path.exists(json_file):
                supplementary_status['failed'].append({
                    'file': json_file,
                    'error': 'File not found'
                })
                continue
            try:
                exfor = read_exfor(json_file)
                ds_id = f"{exfor.entry}{exfor.subentry}"
                if ds_id not in loaded_ids:
                    all_datasets.append(exfor)
                    loaded_ids.add(ds_id)
                    supplementary_status['loaded'].append({
                        'file': json_file,
                        'id': ds_id,
                        'label': getattr(exfor, 'label', 'unknown'),
                        'n_energies': len(exfor.energies()),
                    })
                else:
                    supplementary_status['skipped'].append({
                        'file': json_file,
                        'id': ds_id,
                        'reason': 'Already in database'
                    })
            except Exception as e:
                supplementary_status['failed'].append({
                    'file': json_file,
                    'error': str(e)
                })

    if not all_datasets:
        raise FileNotFoundError("No EXFOR data found from any source")

    # Filter out excluded experiments
    if exclude_experiments:
        all_datasets = _filter_excluded_experiments(all_datasets, exclude_experiments)

    if not all_datasets:
        raise FileNotFoundError("All EXFOR data was excluded by exclude_experiments filter")

    # Organize results
    if not group_by_energy:
        result = {f"{ds.entry}{ds.subentry}": ds for ds in all_datasets}
        if return_load_status:
            return result, supplementary_status
        return result

    # Group by energy
    energy_data: Dict[float, List[ExforAngularDistribution]] = {}
    for exfor in all_datasets:
        energies_mev = exfor.energies(unit='MeV')
        for energy in energies_mev:
            key = _resolve_energy_key(energy, energy_data)
            if key not in energy_data:
                energy_data[key] = []
            if exfor not in energy_data[key]:
                energy_data[key].append(exfor)

    if return_load_status:
        return energy_data, supplementary_status
    return energy_data


def _load_from_database(
    db_path: str = None,
    target: Union[str, int, List[str], List[int]] = None,
    projectile: str = "n",
    mt: int = None,
    energy_range: Tuple[float, float] = None,
) -> List["ExforAngularDistribution"]:
    """
    Load EXFOR data from X4Pro database.

    Parameters
    ----------
    db_path : str, optional
        Path to database file
    target : str, int, List[str], or List[int], optional
        Target (accepts "Fe56", 26056, "Fe-56", [26056, 26000], etc.)
    projectile : str, optional
        Projectile (default: "n")
    mt : int, optional
        ENDF MT number
    energy_range : Tuple[float, float], optional
        Energy range (min, max) in MeV

    Returns
    -------
    List[ExforAngularDistribution]
        List of datasets from database
    """
    from kika.exfor.database import X4ProDatabase

    with X4ProDatabase(db_path) as db:
        # Normalize target(s) to database format
        target_pattern = db._normalize_target(target) if target else None

        return db.query_angular_distributions(
            target=target_pattern,
            projectile=projectile,
            mt=mt,
            energy_range=energy_range,
        )


def _load_from_json_files(
    directory: str,
    pattern: str = "*.json",
) -> List["ExforAngularDistribution"]:
    """
    Load EXFOR data from JSON files in a directory.

    Parameters
    ----------
    directory : str
        Path to directory containing JSON files
    pattern : str, optional
        Glob pattern for file matching

    Returns
    -------
    List[ExforAngularDistribution]
        List of datasets from JSON files
    """
    from kika.exfor.angular_distribution import ExforAngularDistribution

    search_dir = _find_data_directory(directory)
    if search_dir is None:
        raise FileNotFoundError(f"No JSON files found in {directory}")

    json_files = glob.glob(os.path.join(search_dir, pattern))
    if not json_files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {search_dir}")

    datasets = []
    for json_file in json_files:
        try:
            exfor = read_exfor(json_file)
            datasets.append(exfor)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    return datasets


def _find_data_directory(directory: str) -> Optional[str]:
    """
    Find directory containing JSON files.

    Checks the given directory, then looks for 'data' subdirectory.

    Parameters:
        directory: Starting directory path

    Returns:
        Path to directory with JSON files, or None if not found
    """
    # Check if directory exists and has JSON files
    if os.path.isdir(directory):
        if glob.glob(os.path.join(directory, "*.json")):
            return directory
        # Try 'data' subdirectory
        data_subdir = os.path.join(directory, "data")
        if os.path.isdir(data_subdir) and glob.glob(os.path.join(data_subdir, "*.json")):
            return data_subdir

    # Try parent directory with 'data' subdirectory
    parent_dir = os.path.dirname(directory.rstrip("/\\"))
    data_subdir = os.path.join(parent_dir, "data")
    if os.path.isdir(data_subdir) and glob.glob(os.path.join(data_subdir, "*.json")):
        return data_subdir

    return None


def _resolve_energy_key(energy: float, energy_dict: Dict[float, List]) -> float:
    """
    Find existing energy key within tolerance, or return the new energy.

    Parameters:
        energy: Energy value to match
        energy_dict: Existing dictionary with energy keys

    Returns:
        Existing key if within tolerance, otherwise the input energy
    """
    for existing in energy_dict.keys():
        if math.isclose(existing, energy, rel_tol=0.0, abs_tol=ENERGY_MATCH_ABS_TOL):
            return existing
    return energy


def _filter_excluded_experiments(
    datasets: List["ExforAngularDistribution"],
    exclude_experiments: List[str],
) -> List["ExforAngularDistribution"]:
    """
    Filter out experiments matching exclusion patterns.

    Parameters
    ----------
    datasets : List[ExforAngularDistribution]
        List of datasets to filter
    exclude_experiments : List[str]
        List of experiment IDs to exclude. Accepts:
        - "20743" - excludes all subentries starting with 20743
        - "20743002" - excludes specific dataset
        - "20743/002" - same as above

    Returns
    -------
    List[ExforAngularDistribution]
        Filtered list with excluded experiments removed
    """
    if not exclude_experiments:
        return datasets

    # Parse exclusion patterns
    exclusion_set = set()
    prefix_exclusions = set()

    for item in exclude_experiments:
        item = item.strip()
        if not item:
            continue

        # Handle "entry/subentry" format
        if "/" in item:
            parts = item.split("/")
            entry = parts[0].strip()
            subentry = parts[1].strip() if len(parts) > 1 else ""
            exclusion_set.add(entry + subentry)
        elif len(item) <= 5:
            # Short ID - treat as entry prefix (matches all subentries)
            prefix_exclusions.add(item)
        else:
            # Full dataset ID
            exclusion_set.add(item)

    # Filter datasets
    filtered = []
    for ds in datasets:
        ds_id = f"{ds.entry}{ds.subentry}"

        # Check exact match
        if ds_id in exclusion_set:
            continue

        # Check prefix match
        excluded_by_prefix = False
        for prefix in prefix_exclusions:
            if ds_id.startswith(prefix):
                excluded_by_prefix = True
                break

        if not excluded_by_prefix:
            filtered.append(ds)

    return filtered
