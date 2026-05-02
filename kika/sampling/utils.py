"""
Common utilities for sampling module.

This module contains shared functions used across different perturbation modules
to avoid code duplication and maintain consistency.
"""

import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

from kika.cov.parse_covmat import read_coverx, read_covfil, read_boxer, read_scale_covmat, read_njoy_covmat
from kika._utils import zaid_to_symbol


def _format_energy_group_name(energy_grid: List[float], group_index: int) -> str:
    """
    Format energy group name using actual energy boundary values in scientific notation.
    
    Parameters
    ----------
    energy_grid : List[float]
        Energy grid boundaries
    group_index : int
        Energy group index (0-based)
        
    Returns
    -------
    str
        Formatted energy group name (e.g., "1.000e-05_1.234e-02")
    """
    # Check bounds to prevent index out of range errors
    if group_index >= len(energy_grid) - 1:
        # Return a placeholder for padding bins
        return f"PADDING_BIN_{group_index}"
    
    # Get the lower and upper energy boundaries for this group
    e_low = energy_grid[group_index]
    e_high = energy_grid[group_index + 1]
    
    # Format in scientific notation with 3 decimal places
    e_low_str = f"{e_low:.3e}"
    e_high_str = f"{e_high:.3e}"
    
    return f"{e_low_str}_{e_high_str}"


class DualLogger:
    """Logger that writes detailed info to file and minimal info to console.
    
    The logger writes ALL messages to the log file, but only prints messages
    to console when explicitly requested via the console parameter in the
    info/warning/error methods. This ensures users can read the complete
    story in the log file without console spam.
    """
    
    def __init__(self, log_file: str, console_level: str = 'CRITICAL'):
        self.log_file = log_file
        
        # Create logger
        self.logger = logging.getLogger('sampling_perturbation')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler - gets everything at DEBUG level
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler - set to CRITICAL by default to suppress automatic console output
        # Messages will only appear in console when explicitly printed via console=True parameter
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.CRITICAL + 1)  # Effectively disable automatic console output
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str, console: bool = False):
        """Log info message. If console=True, also show in console."""
        self.logger.info(message)
        if console:
            print(message)

    def warning(self, message: str, console: bool = True):
        """Log warning message. If console=True, also show in console."""
        self.logger.warning(message)
        if console:
            print(message)

    def error(self, message: str, console: bool = True):
        """Log error message. If console=True, also show in console."""
        self.logger.error(message)
        if console:
            print(message)


# Global logger instance
_logger = None

def _get_logger():
    """Get the global logger instance."""
    return _logger


def _set_logger(logger):
    """Set the global logger instance."""
    global _logger
    _logger = logger


def load_covariance(path):
    """
    Load covariance matrix from file, trying different formats.
    
    Parameters
    ----------
    path : str
        Path to covariance file
        
    Returns
    -------
    Covariance matrix object or None if loading failed
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Covariance file not found: {path}")
    
    # Try to get logger from global instance
    try:
        logger = _get_logger()
    except:
        logger = None
    
    # Map readers to format names for logging
    reader_formats = {
        read_covfil: "NJOY COVFIL",
        read_coverx: "COVERX",
        read_boxer: "BOXER",
    }

    for reader in (read_covfil, read_coverx, read_boxer):
        try:
            cov = reader(path)
            if logger:
                logger.info(f"Successfully loaded {reader_formats[reader]} covariance from: {path}")
            return cov
        except Exception:
            continue
    
    # If we get here, no reader succeeded
    if logger:
        logger.error(f"Failed to load covariance matrix from: {path}")
    
    return None


def _initialize_master_perturbation_matrix(output_dir: str, timestamp: str, num_samples: int) -> str:
    """
    Initialize directory for incremental master perturbation matrix building.
    
    Parameters
    ----------
    output_dir : str
        Base output directory
    timestamp : str
        Timestamp string for file naming
    num_samples : int
        Number of samples expected
        
    Returns
    -------
    str
        Path to the matrix directory for incremental updates
    """
    matrix_dir = os.path.join(output_dir, f"perturbation_matrix_{timestamp}_parts")
    os.makedirs(matrix_dir, exist_ok=True)
    
    # Create metadata file
    metadata_file = os.path.join(matrix_dir, "metadata.json")
    metadata = {
        "timestamp": timestamp,
        "num_samples": num_samples,
        "isotopes_processed": [],
        "created": True
    }
    
    import json
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return matrix_dir


def _write_isotope_parquet(
    matrix_dir: str,
    zaid: int,
    factors: np.ndarray,
    mt_numbers: List[int],
    energy_grid: List[float],
    verbose: bool = True,
    logger=None,
) -> Optional[Dict]:
    """
    Write per-isotope perturbation factors to a parquet file.

    Parquet writes are atomic and use a unique filename per zaid, so this
    is safe to call concurrently from multiple worker processes. Metadata
    updates are deferred — this function returns a fragment dict that the
    caller passes to ``_merge_isotope_metadata`` after all workers finish.

    Parameters
    ----------
    matrix_dir : str
        Directory for matrix parts.
    zaid : int
        ZAID number.
    factors : np.ndarray
        Perturbation factors array, shape (n_samples, n_params).
    mt_numbers : List[int]
        MT reaction numbers (column ordering).
    energy_grid : List[float]
        Energy grid boundaries.
    verbose : bool
        Enable verbose output.
    logger : optional
        Logger to use; if None, falls back to ``_get_logger()``. Workers
        should pass their own ``BufferedLogger`` to avoid touching the
        shared module-level logger.

    Returns
    -------
    fragment : dict or None
        ``{"zaid": int}`` on success, or ``None`` if no columns were
        produced (caller should not append to metadata in that case).
    """
    symbol = zaid_to_symbol(zaid)

    n_groups = len(energy_grid) - 1
    columns_data = {'Sample_ID': np.arange(1, factors.shape[0] + 1, dtype='int32')}

    for mt_idx, mt in enumerate(mt_numbers):
        for grp in range(n_groups):
            energy_group_name = _format_energy_group_name(energy_grid, grp)
            col_name = f"{symbol}_MT{mt}_{energy_group_name}"
            start_idx = mt_idx * n_groups + grp
            columns_data[col_name] = factors[:, start_idx]

    log = logger if logger is not None else _get_logger()

    if len(columns_data) == 1:  # Only Sample_ID
        if verbose and log is not None:
            log.warning(f"[MATRIX] No valid columns generated for ZAID {zaid}")
        return None

    df = pd.DataFrame(columns_data)
    isotope_file = os.path.join(matrix_dir, f"isotope_{zaid}.parquet")
    df.to_parquet(isotope_file, index=False, engine='pyarrow')

    if verbose and log is not None:
        n_cols = len(columns_data) - 1
        log.info(f"[MATRIX] Saved {n_cols} columns for ZAID {zaid}")

    return {"zaid": zaid}


def _merge_isotope_metadata(matrix_dir: str, fragments: List[Dict]) -> None:
    """
    Apply per-isotope metadata fragments to ``metadata.json``.

    Must be called from the parent process after all workers complete —
    serializes the JSON read-modify-write so concurrent worker writes
    cannot race.

    Parameters
    ----------
    matrix_dir : str
        Directory holding ``metadata.json``.
    fragments : list of dict
        Each fragment must have a ``"zaid"`` key; appended in input order.
    """
    if not fragments:
        return

    metadata_file = os.path.join(matrix_dir, "metadata.json")
    import json
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        for frag in fragments:
            zaid = frag.get("zaid")
            if zaid is not None and zaid not in metadata["isotopes_processed"]:
                metadata["isotopes_processed"].append(zaid)
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception:
        pass


def _update_master_perturbation_matrix(
    matrix_dir: str,
    zaid: int,
    factors: np.ndarray,
    mt_numbers: List[int],
    energy_grid: List[float],
    verbose: bool = True
) -> None:
    """
    Sequential-path wrapper: write one isotope's parquet AND update metadata.

    Equivalent to ``_write_isotope_parquet`` followed immediately by
    ``_merge_isotope_metadata`` for the single fragment. The parallel
    dry-run path uses the split helpers directly so workers never touch
    ``metadata.json``.

    Parameters
    ----------
    matrix_dir : str
        Directory for matrix parts
    zaid : int
        ZAID number
    factors : np.ndarray
        Perturbation factors array
    mt_numbers : List[int]
        MT numbers
    energy_grid : List[float]
        Energy grid boundaries
    verbose : bool
        Enable verbose output
    """
    fragment = _write_isotope_parquet(
        matrix_dir, zaid, factors, mt_numbers, energy_grid, verbose=verbose,
    )
    if fragment is not None:
        _merge_isotope_metadata(matrix_dir, [fragment])


def _finalize_master_perturbation_matrix(matrix_dir: str, verbose: bool = True) -> str:
    """
    Combine all isotope-specific parquet files into a single master matrix.
    
    Parameters
    ----------
    matrix_dir : str
        Directory containing isotope-specific parquet files
    verbose : bool
        Enable verbose output
        
    Returns
    -------
    str
        Path to the final master parquet file
    """
    logger = _get_logger()
    
    # Read metadata
    metadata_file = os.path.join(matrix_dir, "metadata.json")
    import json
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        timestamp = metadata.get("timestamp", "unknown")
    except Exception:
        timestamp = "unknown"
    
    # Find all isotope parquet files
    isotope_files = [f for f in os.listdir(matrix_dir) if f.startswith("isotope_") and f.endswith(".parquet")]
    
    if not isotope_files:
        if verbose and logger:
            logger.warning("[MATRIX] [FINALIZE] No isotope files found to combine")
        return ""
    
    # Load and combine all DataFrames
    dataframes = []
    sample_id_df = None
    
    for isotope_file in sorted(isotope_files):
        file_path = os.path.join(matrix_dir, isotope_file)
        df = pd.read_parquet(file_path)
        
        if sample_id_df is None:
            sample_id_df = df[['Sample_ID']].copy()
        
        # Add all columns except Sample_ID
        factor_cols = [col for col in df.columns if col != 'Sample_ID']
        if factor_cols:
            dataframes.append(df[factor_cols])
    
    if not dataframes:
        if verbose and logger:
            logger.warning("[MATRIX] [FINALIZE] No factor columns found")
        return ""
    
    # Combine all factor columns with Sample_ID
    combined_df = pd.concat([sample_id_df] + dataframes, axis=1)
    
    # Save master file
    parent_dir = os.path.dirname(matrix_dir)
    master_file = os.path.join(parent_dir, f"perturbation_matrix_{timestamp}_master.parquet")
    combined_df.to_parquet(master_file, index=False, engine='pyarrow')
    
    if verbose and logger:
        n_samples, n_cols = combined_df.shape
        n_factor_cols = n_cols - 1  # Exclude Sample_ID
        logger.info(f"[MATRIX] [FINALIZE] Master matrix created: {n_samples} samples × {n_factor_cols} parameters")
        logger.info(f"[MATRIX] [FINALIZE] File: {os.path.basename(master_file)}")
    
    # Clean up temporary files
    try:
        import shutil
        shutil.rmtree(matrix_dir)
        if verbose and logger:
            logger.info(f"[MATRIX] [FINALIZE] Cleaned up temporary directory")
    except Exception as e:
        if verbose and logger:
            logger.warning(f"[MATRIX] [FINALIZE] Could not clean up temporary directory: {e}")
    
    return master_file

def normalize_mt_list(
    mt_list: Union[List[int], List[List[int]], None],
    n_units: int,
    unit_label: str = "files",
) -> List[List[int]]:
    """Normalize mt_list into a per-unit list-of-lists of length ``n_units``.

    Accepts either a flat ``List[int]`` (broadcast to every unit) or a
    ``List[List[int]]`` (one list per unit). An empty inner list means
    "perturb every reaction available in the covariance" for that unit.

    Parameters
    ----------
    mt_list : List[int] or List[List[int]] or None
        The user-provided MT selection.
    n_units : int
        The number of ace/endf files (or ZAIDs) the run operates on.
    unit_label : str, default "files"
        Word used in error messages (e.g. "files", "ZAIDs").

    Returns
    -------
    List[List[int]]
        A list of length ``n_units``. Each inner list is a copy of the input.

    Raises
    ------
    TypeError
        If the input is neither a flat list of ints nor a list of lists of ints.
    ValueError
        If a nested list is provided whose length does not match ``n_units``.
    """
    if mt_list is None or len(mt_list) == 0:
        return [[] for _ in range(n_units)]

    if all(isinstance(x, int) for x in mt_list):
        return [list(mt_list) for _ in range(n_units)]

    if not all(isinstance(x, (list, tuple)) for x in mt_list):
        raise TypeError(
            "mt_list must be a flat List[int] or a List[List[int]]; "
            f"got a mixed/unsupported structure: {mt_list!r}"
        )

    if len(mt_list) != n_units:
        raise ValueError(
            f"mt_list has {len(mt_list)} entries but {n_units} {unit_label} were provided; "
            "they must match."
        )

    for i, sub in enumerate(mt_list):
        if not all(isinstance(x, int) for x in sub):
            raise TypeError(
                f"mt_list[{i}] must contain only integers; got {sub!r}"
            )

    return [list(sub) for sub in mt_list]
