"""
MF34 (Angular Distribution Covariance) creation and writing utilities.

This module provides functions to:
1. Create MF34MT objects from Legendre coefficient covariance matrices
2. Write MF34 sections to ENDF files
3. Support for LB=5 format (full symmetric matrix storage)

The covariance matrix is expected to be organized with:
- Rows/columns representing (energy, Legendre order) pairs
- Layout: [a_1(E_1), a_2(E_1), ..., a_L(E_1), a_1(E_2), ..., a_L(E_N)]

Example:
    >>> from kika.endf.writers import create_mf34_from_covariance, write_mf34_to_file
    >>>
    >>> # Create MF34 from covariance matrix
    >>> mf34 = create_mf34_from_covariance(
    ...     cov_matrix=cov,
    ...     energy_grid_ev=energy_boundaries,
    ...     max_order=8,
    ...     za=26056.0,
    ...     awr=55.47,
    ...     mat=2631,
    ...     mt=2,
    ... )
    >>>
    >>> # Write to ENDF file
    >>> write_mf34_to_file('base.endf', mf34, 'output.endf')
"""
from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING
import numpy as np

from ..classes.mf34.mf34 import (
    MF34MT,
    Subsection,
    SubSubsection,
    SubSubsectionRecord,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_lb5_record(
    matrix: np.ndarray,
    energy_grid: List[float],
) -> SubSubsectionRecord:
    """Create an LB=5 symmetric upper-triangle SubSubsectionRecord.

    Only appropriate for symmetric matrices (diagonal L=L' blocks).
    For asymmetric cross-correlation matrices (L≠L'), use
    :func:`_make_lb6_record` instead.

    Parameters
    ----------
    matrix : np.ndarray
        Square **symmetric** covariance matrix of shape (m, m)
        where m = len(energy_grid) - 1.
    energy_grid : List[float]
        Energy boundary points (m + 1 values).

    Returns
    -------
    SubSubsectionRecord
    """
    m = len(energy_grid) - 1
    if matrix.shape != (m, m):
        raise ValueError(
            f"Matrix shape {matrix.shape} doesn't match energy grid "
            f"with {m} intervals ({len(energy_grid)} boundaries)"
        )
    record = SubSubsectionRecord()
    record.ls = 1  # symmetric upper triangle
    record.lb = 5
    record.ne = len(energy_grid)
    record.energies = list(energy_grid)
    triu_rows, triu_cols = np.triu_indices(m)
    record.matrix = matrix[triu_rows, triu_cols].tolist()
    record.nt = len(energy_grid) + len(record.matrix)
    return record


def _make_lb6_record(
    matrix: np.ndarray,
    row_energy_grid: List[float],
    col_energy_grid: List[float],
) -> SubSubsectionRecord:
    """Create an LB=6 rectangular-matrix SubSubsectionRecord.

    Use this for asymmetric cross-correlation matrices such as
    off-diagonal (L≠L') Legendre covariance blocks where
    ``Cov(a_L(E_i), a_L'(E_j)) ≠ Cov(a_L(E_j), a_L'(E_i))``.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix of shape (r, c) where r = len(row_energy_grid) - 1
        and c = len(col_energy_grid) - 1.
    row_energy_grid : List[float]
        Row energy boundary points (r + 1 values).
    col_energy_grid : List[float]
        Column energy boundary points (c + 1 values).

    Returns
    -------
    SubSubsectionRecord
    """
    r = len(row_energy_grid) - 1
    c = len(col_energy_grid) - 1
    if matrix.shape != (r, c):
        raise ValueError(
            f"Matrix shape {matrix.shape} doesn't match energy grids "
            f"with {r} row intervals and {c} column intervals"
        )
    record = SubSubsectionRecord()
    record.ls = 0
    record.lb = 6
    record.row_energies = list(row_energy_grid)
    record.col_energies = list(col_energy_grid)
    record.rect_matrix = matrix.ravel().tolist()
    # NT = NER + NEC + NER_intervals * NEC_intervals
    record.nt = len(row_energy_grid) + len(col_energy_grid) + r * c
    record.ne = len(row_energy_grid)  # NE field stores NER for LB=6
    return record


def _split_matrix_excluding_range(
    matrix: np.ndarray,
    energy_grid: List[float],
    exclude_min_ev: float,
    exclude_max_ev: float,
) -> List[Tuple[np.ndarray, List[float]]]:
    """Split a covariance matrix to exclude an energy range.

    Computes interval midpoints and removes intervals whose midpoint falls
    within [exclude_min_ev, exclude_max_ev].  Returns 0, 1, or 2 contiguous
    sub-matrices (below and/or above the excluded range).

    Parameters
    ----------
    matrix : np.ndarray
        Square covariance matrix (m, m).
    energy_grid : List[float]
        Energy boundaries (m + 1 values).
    exclude_min_ev, exclude_max_ev : float
        Energy range to exclude (inclusive on midpoints).

    Returns
    -------
    List[Tuple[np.ndarray, List[float]]]
        List of (sub_matrix, sub_energy_grid) for each contiguous kept region.
    """
    grid = np.asarray(energy_grid, dtype=float)
    midpoints = 0.5 * (grid[:-1] + grid[1:])
    keep = (midpoints < exclude_min_ev) | (midpoints > exclude_max_ev)

    if keep.all():
        return [(matrix, list(energy_grid))]
    if not keep.any():
        return []

    # Find contiguous runs of True in keep
    results: List[Tuple[np.ndarray, List[float]]] = []
    m = len(midpoints)
    i = 0
    while i < m:
        if not keep[i]:
            i += 1
            continue
        # Start of a kept run
        j = i
        while j < m and keep[j]:
            j += 1
        # Indices i..j-1 are kept
        idx = np.arange(i, j)
        sub_matrix = matrix[np.ix_(idx, idx)]
        # Energy grid: boundaries i through j (inclusive)
        sub_grid = list(grid[i:j + 1])
        results.append((sub_matrix, sub_grid))
        i = j

    return results


def create_mf34_from_covariance(
    cov_matrix: np.ndarray,
    energy_grid_ev: np.ndarray,
    max_order: int,
    za: float,
    awr: float,
    mat: int,
    mt: int,
    ltt: int = 1,
    mt1: Optional[int] = None,
    frame: str = "same-as-MF4",
) -> MF34MT:
    """
    Create MF34MT object from Legendre coefficient covariance matrix.

    This function constructs an MF34 (Angular Distribution Covariance) section
    from a pre-computed covariance matrix of Legendre polynomial coefficients.
    The output uses LB=5 format with symmetric upper-triangle storage.

    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix with shape (N_energies * L_max, N_energies * L_max).
        The matrix should be organized with Legendre orders as the fast index:
        index = i_energy * max_order + (l - 1)

        Layout: [a_1(E_1), a_2(E_1), ..., a_L(E_1), a_1(E_2), ..., a_L(E_N)]

    energy_grid_ev : np.ndarray
        Energy boundary points in eV. For N_energies energy intervals,
        provide N_energies + 1 boundary points.

    max_order : int
        Maximum Legendre order (L_max). The covariance includes orders 1 to L_max
        (following ENDF convention where a_0 is implicit from normalization).

    za : float
        ZA identifier (1000*Z + A). For example, 26056.0 for Fe-56.

    awr : float
        Atomic weight ratio (mass of target in neutron mass units).

    mat : int
        MAT number for the material in the ENDF file.

    mt : int
        MT reaction number. Common values:
        - MT=2: Elastic scattering
        - MT=18: Total fission

    ltt : int, default 1
        LTT flag indicating Legendre representation:
        - LTT=1: Coefficients start with a_1 (standard ENDF convention)
        - LTT=2: Coefficients start with a_0

    mt1 : int, optional
        MT1 for cross-correlation between different reactions.
        If None, defaults to mt (self-correlation of the same reaction).

    frame : str, default "same-as-MF4"
        Reference frame for angular distributions:
        - "same-as-MF4": Use same frame as MF4 section (LCT=0)
        - "LAB": Laboratory frame (LCT=1)
        - "CM": Center-of-mass frame (LCT=2)

    Returns
    -------
    MF34MT
        MF34MT object ready for serialization to ENDF format using str(mf34).

    Raises
    ------
    ValueError
        If covariance matrix dimensions don't match expected size.

    Notes
    -----
    The output uses LB=5 format (full matrix storage) with LS=1 (symmetric,
    upper-triangle only). For each (L, L1) Legendre pair, a sub-subsection
    is created containing the energy-energy covariance block.

    Only the upper triangle of (L, L1) pairs is stored since the covariance
    is symmetric: Cov(a_L, a_{L1}) = Cov(a_{L1}, a_L).

    Covariance Interpretation
    -------------------------
    The input covariance matrix must contain RELATIVE (fractional) covariance:
        Cov_rel(i, j) = Cov_abs(i, j) / (mean_i * mean_j)

    This is written to MF34 with LB=5 format, which ENDF-6 defines as
    relative covariance. The MF34 resampling model is multiplicative:
        a_new = a_nominal * (1 + Y),  Y ~ N(0, Cov_rel)

    Use ``compute_covariance_from_samples()`` in ``exfor_utils.py`` to
    obtain the relative covariance matrix from MC samples.

    Examples
    --------
    Create MF34 for Fe-56 elastic scattering with 8 Legendre orders:

    >>> import numpy as np
    >>> # Suppose we have 10 energy intervals and 8 Legendre orders
    >>> n_energies = 10
    >>> max_order = 8
    >>> cov = np.eye(n_energies * max_order) * 0.01  # Example diagonal covariance
    >>> energy_grid = np.linspace(1e6, 20e6, n_energies + 1)  # eV
    >>>
    >>> mf34 = create_mf34_from_covariance(
    ...     cov_matrix=cov,
    ...     energy_grid_ev=energy_grid,
    ...     max_order=8,
    ...     za=26056.0,
    ...     awr=55.47,
    ...     mat=2631,
    ...     mt=2,
    ... )
    >>>
    >>> # Get ENDF-formatted string
    >>> endf_text = str(mf34)
    """
    if mt1 is None:
        mt1 = mt

    n_energies = len(energy_grid_ev) - 1  # Number of energy intervals

    # Validate covariance matrix dimensions
    expected_size = n_energies * max_order
    if cov_matrix.shape != (expected_size, expected_size):
        raise ValueError(
            f"Covariance matrix shape {cov_matrix.shape} doesn't match "
            f"expected ({expected_size}, {expected_size}) for "
            f"{n_energies} energy intervals and {max_order} Legendre orders"
        )

    # Validate finite values in inputs
    if not np.all(np.isfinite(cov_matrix)):
        n_inf = int(np.sum(np.isinf(cov_matrix)))
        n_nan = int(np.sum(np.isnan(cov_matrix)))
        inf_indices = np.argwhere(~np.isfinite(cov_matrix))
        raise ValueError(
            f"Covariance matrix contains {n_inf} inf and {n_nan} NaN values. "
            f"First 5 non-finite positions (row, col): {inf_indices[:5].tolist()}"
        )
    if not np.all(np.isfinite(energy_grid_ev)):
        raise ValueError(
            f"Energy grid contains non-finite values: "
            f"{energy_grid_ev[~np.isfinite(energy_grid_ev)]}"
        )

    # Create MF34MT structure
    mf34 = MF34MT(number=mt)
    mf34._za = za
    mf34._awr = awr
    mf34._mat = mat
    mf34._ltt = ltt
    mf34._mf = 34

    # Create subsection for MT1 correlation
    subsection = Subsection()
    subsection.mt1 = mt1
    subsection.nl = max_order   # Number of Legendre coefficients for MT
    subsection.nl1 = max_order  # Number of Legendre coefficients for MT1
    subsection.mat1 = 0.0

    # LCT value based on frame
    lct_map = {"same-as-MF4": 0, "LAB": 1, "CM": 2}
    lct = lct_map.get(frame, 0)

    # Create sub-subsections for each (L, L1) pair
    # Only upper triangle: L <= L1 (symmetric covariance)
    for l in range(1, max_order + 1):
        for l1 in range(l, max_order + 1):
            sub_subsec = SubSubsection()
            sub_subsec.l = l
            sub_subsec.l1 = l1
            sub_subsec.lct = lct
            sub_subsec.ni = 1  # One LIST record

            # Extract sub-matrix for this (L, L1) block
            # Indices: for each energy E_i, coeff a_l is at index: i * max_order + (l-1)
            row_indices = [i * max_order + (l - 1) for i in range(n_energies)]
            col_indices = [i * max_order + (l1 - 1) for i in range(n_energies)]

            sub_matrix = cov_matrix[np.ix_(row_indices, col_indices)]

            if l == l1:
                # Diagonal block: symmetric → LB=5 LS=1
                sub_subsec.records = [_make_lb5_record(sub_matrix, list(energy_grid_ev))]
            else:
                # Off-diagonal block: asymmetric → LB=6
                sub_subsec.records = [_make_lb6_record(
                    sub_matrix, list(energy_grid_ev), list(energy_grid_ev)
                )]
            subsection.sub_subsections.append(sub_subsec)

    mf34._nmt1 = 1  # One subsection
    mf34._subsections = [subsection]

    return mf34


def write_mf34_to_file(
    source_endf: str,
    mf34: MF34MT,
    output_path: str,
    replace_existing: bool = True,
    update_directory: bool = True,
) -> str:
    """
    Write MF34 section to an ENDF file.

    This function takes a source ENDF file as a template and either replaces
    an existing MF34 section or inserts a new one before the MEND marker.

    Parameters
    ----------
    source_endf : str or Path
        Path to source ENDF file that serves as the template.
        All content except MF34 will be preserved.

    mf34 : MF34MT
        MF34MT object to write. Use create_mf34_from_covariance() to create this.

    output_path : str or Path
        Path for the output ENDF file.

    replace_existing : bool, default True
        If True and MF34 already exists in source, replace it.
        If False and MF34 exists, raise FileExistsError.

    Returns
    -------
    str
        Path to the output file.

    Raises
    ------
    FileNotFoundError
        If source_endf file doesn't exist.
    FileExistsError
        If MF34 exists in source and replace_existing=False.

    Examples
    --------
    Add MF34 to an ENDF file:

    >>> mf34 = create_mf34_from_covariance(...)
    >>> write_mf34_to_file('evaluation.endf', mf34, 'evaluation_with_cov.endf')
    'evaluation_with_cov.endf'
    """
    # Read source file
    with open(source_endf, 'r') as f:
        lines = f.readlines()

    # Find MF34 boundaries if it exists
    mf34_start, mf34_end = _find_mf34_boundaries(lines)
    has_mf34 = mf34_start is not None

    if has_mf34 and not replace_existing:
        raise FileExistsError(
            f"MF34 already exists in {source_endf}. "
            f"Set replace_existing=True to replace it."
        )

    # Convert MF34MT to string
    mf34_content = str(mf34)
    mf34_lines = [line + '\n' for line in mf34_content.split('\n') if line.strip()]

    # Add FEND line after MF34 content (MAT=mat, MF=0, MT=0)
    from ..utils import format_endf_data_line, ENDF_FORMAT_INT
    mat_num = mf34._mat or 0
    fend_line = format_endf_data_line(
        [0, 0, 0, 0, 0, 0], mat_num, 0, 0, 0,
        formats=[ENDF_FORMAT_INT] * 6
    ) + '\n'
    mf34_lines.append(fend_line)

    if has_mf34:
        # Replace existing MF34
        # Skip past the old FEND line (MF=0, MT=0) that follows the MF34 SEND
        skip_end = mf34_end
        if skip_end < len(lines) and len(lines[skip_end]) >= 75:
            try:
                old_mf = int(lines[skip_end][70:72].strip() or '0')
                old_mt = int(lines[skip_end][72:75].strip() or '0')
                if old_mf == 0 and old_mt == 0:
                    skip_end += 1  # Skip the old FEND line
            except ValueError:
                pass
        new_lines = lines[:mf34_start] + mf34_lines + lines[skip_end:]
    else:
        # Insert MF34 before MEND marker
        insert_idx = _find_mend_marker(lines)
        new_lines = lines[:insert_idx] + mf34_lines + lines[insert_idx:]

    # Write output
    with open(output_path, 'w') as f:
        f.writelines(new_lines)

    if update_directory:
        from .update_directory import update_mf1_directory
        update_mf1_directory(output_path)

    return output_path


def _find_mf34_boundaries(lines: List[str]) -> tuple:
    """
    Find start and end line indices of MF34 section.

    Parameters
    ----------
    lines : List[str]
        Lines from ENDF file.

    Returns
    -------
    tuple
        (start_index, end_index) or (None, None) if MF34 not found.
    """
    mf34_start = None
    mf34_end = None

    for i, line in enumerate(lines):
        if len(line) >= 75:
            try:
                mf = int(line[70:72].strip() or '0')
                if mf == 34:
                    if mf34_start is None:
                        mf34_start = i
                    mf34_end = i + 1
            except ValueError:
                continue

    return mf34_start, mf34_end


def merge_mf34(
    original_mf34: MF34MT,
    pipeline_mf34: MF34MT,
    pipeline_energy_min_ev: float,
    pipeline_energy_max_ev: float,
) -> MF34MT:
    """
    Merge original and pipeline MF34 covariance data.

    For each (L, L1) pair present in either source, builds a union energy grid
    and selects data from the pipeline where interval midpoints fall within the
    pipeline energy range, and from the original otherwise.  Cross-source
    covariance cells are set to zero (independent analyses).

    Parameters
    ----------
    original_mf34 : MF34MT
        MF34 section parsed from the original ENDF evaluation.
    pipeline_mf34 : MF34MT
        MF34 section produced by the EXFOR sampling pipeline.
    pipeline_energy_min_ev : float
        Lower bound of the pipeline energy range in eV.
    pipeline_energy_max_ev : float
        Upper bound of the pipeline energy range in eV.

    Returns
    -------
    MF34MT
        Merged MF34MT object.  Sub-subsections that exist only in the
        original evaluation have their pipeline energy range excised,
        potentially yielding multiple NI records (one per contiguous
        sub-range outside the pipeline window).
    """
    from kika.cov.legendre_covariance import LegendreCovariance  # noqa: F811 – local import

    # --- Convert both to LegendreCovariance to get per-(L,L1) matrices ----------
    orig_cov = original_mf34.to_ang_covmat()
    pipe_cov = pipeline_mf34.to_ang_covmat()

    # Validate finite values from both sources
    for label, covmat in [("original", orig_cov), ("pipeline", pipe_cov)]:
        for i in range(covmat.num_matrices):
            mat = covmat.matrices[i]
            if not np.all(np.isfinite(mat)):
                raise ValueError(
                    f"{label.capitalize()} MF34 (L={covmat.l_rows[i]}, "
                    f"L1={covmat.l_cols[i]}) contains non-finite values: "
                    f"{int(np.sum(np.isinf(mat)))} inf, "
                    f"{int(np.sum(np.isnan(mat)))} NaN"
                )

    # Build lookup: (l_row, l_col) -> (matrix, energy_grid) for each source
    def _build_ll_map(covmat):
        ll_map = {}
        for i in range(covmat.num_matrices):
            key = (covmat.l_rows[i], covmat.l_cols[i])
            ll_map[key] = (covmat.matrices[i], list(covmat.energy_grids[i]))
        return ll_map

    orig_map = _build_ll_map(orig_cov)
    pipe_map = _build_ll_map(pipe_cov)

    all_ll_pairs = sorted(set(orig_map.keys()) | set(pipe_map.keys()))

    # --- Reconstruct MF34MT with merged data -----------------------------
    merged = MF34MT(number=pipeline_mf34.number)
    merged._za = pipeline_mf34._za
    merged._awr = pipeline_mf34._awr
    merged._mat = pipeline_mf34._mat
    merged._ltt = pipeline_mf34._ltt
    merged._mf = 34

    # Determine max Legendre order from both sources
    all_l_values = set()
    for l, l1 in all_ll_pairs:
        all_l_values.add(l)
        all_l_values.add(l1)
    max_order = max(all_l_values) if all_l_values else 1

    subsection = Subsection()
    subsection.mt1 = pipeline_mf34.number
    subsection.nl = max_order
    subsection.nl1 = max_order
    subsection.mat1 = 0.0

    for l, l1 in all_ll_pairs:
        orig_data = orig_map.get((l, l1))
        pipe_data = pipe_map.get((l, l1))
        is_offdiag = (l != l1)

        if orig_data is None and pipe_data is not None:
            # Pipeline-only pair: single record
            mat, egrid = pipe_data
            egrid_f = [float(e) for e in egrid]
            if is_offdiag:
                records = [_make_lb6_record(mat, egrid_f, egrid_f)]
            else:
                records = [_make_lb5_record(mat, egrid_f)]

        elif pipe_data is None and orig_data is not None:
            # Original-only pair: exclude the pipeline energy range.
            # The original and pipeline are independent analyses — the
            # original's data in the pipeline range is stale.
            mat, egrid = orig_data
            splits = _split_matrix_excluding_range(
                mat, egrid, pipeline_energy_min_ev, pipeline_energy_max_ev,
            )
            if not splits:
                continue  # pipeline covers entire original range → skip pair
            if is_offdiag:
                records = [_make_lb6_record(s_mat, [float(e) for e in s_grid],
                                            [float(e) for e in s_grid])
                           for s_mat, s_grid in splits]
            else:
                records = [_make_lb5_record(s_mat, [float(e) for e in s_grid])
                           for s_mat, s_grid in splits]

        else:
            # Both sources have data – merge on union grid
            orig_mat, orig_grid = orig_data
            pipe_mat, pipe_grid = pipe_data

            # Build union energy grid
            union_grid = sorted(set(orig_grid) | set(pipe_grid))
            n_intervals = len(union_grid) - 1

            merged_matrix = np.zeros((n_intervals, n_intervals))

            # Classify each interval by midpoint (vectorized)
            union_arr = np.asarray(union_grid)
            midpoints = 0.5 * (union_arr[:-1] + union_arr[1:])
            is_pipe = (midpoints >= pipeline_energy_min_ev) & (midpoints <= pipeline_energy_max_ev)

            # Map union intervals to native bins via searchsorted
            pipe_arr = np.asarray(pipe_grid, dtype=float)
            orig_arr = np.asarray(orig_grid, dtype=float)
            pipe_bins = np.searchsorted(pipe_arr, midpoints, side='right') - 1
            orig_bins = np.searchsorted(orig_arr, midpoints, side='right') - 1
            np.clip(pipe_bins, 0, len(pipe_grid) - 2, out=pipe_bins)
            np.clip(orig_bins, 0, len(orig_grid) - 2, out=orig_bins)

            # Fill same-source blocks with fancy indexing
            pipe_idx = np.where(is_pipe)[0]
            orig_idx = np.where(~is_pipe)[0]

            if pipe_idx.size > 0:
                pb = pipe_bins[pipe_idx]
                merged_matrix[np.ix_(pipe_idx, pipe_idx)] = pipe_mat[np.ix_(pb, pb)]

            if orig_idx.size > 0:
                ob = orig_bins[orig_idx]
                merged_matrix[np.ix_(orig_idx, orig_idx)] = orig_mat[np.ix_(ob, ob)]

            # Cross-source cells remain zero (already initialized)
            union_grid_f = [float(e) for e in union_grid]
            if is_offdiag:
                records = [_make_lb6_record(merged_matrix, union_grid_f, union_grid_f)]
            else:
                records = [_make_lb5_record(merged_matrix, union_grid_f)]

        sub_subsec = SubSubsection()
        sub_subsec.l = l
        sub_subsec.l1 = l1
        sub_subsec.lct = 0  # same-as-MF4
        sub_subsec.ni = len(records)
        sub_subsec.records = records
        subsection.sub_subsections.append(sub_subsec)

    merged._nmt1 = 1
    merged._subsections = [subsection]
    return merged


def remove_mf34_from_file(filepath: str, update_directory: bool = True) -> bool:
    """
    Remove MF34 section from an ENDF file if present.

    Parameters
    ----------
    filepath : str or Path
        Path to the ENDF file to modify in place.
    update_directory : bool, default True
        If True, update MF1/MT451 directory after removal.

    Returns
    -------
    bool
        True if MF34 was found and removed, False if no MF34 was present.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    start, end = _find_mf34_boundaries(lines)
    if start is None:
        return False

    new_lines = lines[:start] + lines[end:]
    with open(filepath, 'w') as f:
        f.writelines(new_lines)

    if update_directory:
        from .update_directory import update_mf1_directory
        update_mf1_directory(filepath)

    return True


def _find_mend_marker(lines: List[str]) -> int:
    """
    Find insertion point (line index before MEND marker).

    The MEND marker is identified by a line with MAT = 0, MF = 0, MT = 0.

    Parameters
    ----------
    lines : List[str]
        Lines from ENDF file.

    Returns
    -------
    int
        Line index where MF34 should be inserted.
    """
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if len(line) >= 75:
            try:
                mat = int(line[66:70].strip() or '0')
                mf = int(line[70:72].strip() or '0')
                mt = int(line[72:75].strip() or '0')
                if mat == 0 and mf == 0 and mt == 0:
                    return i
            except ValueError:
                continue
    return len(lines)
