"""
MF34 (Angular Distribution Covariance) creation and writing utilities.

This module provides functions to:
1. Build MF34MT objects from Legendre coefficient covariance matrices
2. Write MF34 sections into ENDF files (insert or replace)
3. Merge two MF34 sections by energy range
4. Remove an MF34 section from an ENDF file

Diagonal blocks (L == L1) are stored with LB=5 (symmetric upper triangle);
off-diagonal blocks (L != L1) are stored with LB=6 (rectangular).  Both the
``ltt=1`` representation (orders start at a_1) and the ``ltt=2``
representation (orders start at a_0) are supported; the latter naturally
includes the L=0 sub-subsections of the upper triangle.
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


# ---- helpers ---------------------------------------------------------------


def _l_min_for_ltt(ltt) -> int:
    """Lowest Legendre index L allowed by the given LTT representation."""
    return 0 if int(ltt or 1) == 2 else 1


def _make_lb5_record(
    matrix: np.ndarray,
    energy_grid: List[float],
) -> SubSubsectionRecord:
    """Build an LB=5 (LS=1, symmetric upper triangle) LIST record.

    Use for diagonal blocks (L == L1) where the covariance is symmetric.

    Parameters
    ----------
    matrix : np.ndarray
        Square symmetric covariance matrix of shape (m, m), where
        m = len(energy_grid) - 1.
    energy_grid : list of float
        Energy boundaries (m + 1 values).
    """
    m = len(energy_grid) - 1
    if matrix.shape != (m, m):
        raise ValueError(
            f"Matrix shape {matrix.shape} doesn't match energy grid "
            f"with {m} intervals ({len(energy_grid)} boundaries)"
        )
    record = SubSubsectionRecord()
    record.ls = 1
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
    """Build an LB=6 (rectangular matrix) LIST record.

    Use for off-diagonal blocks (L != L1) where the matrix is asymmetric.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix of shape (r, c) where r = len(row_energy_grid) - 1
        and c = len(col_energy_grid) - 1.
    row_energy_grid, col_energy_grid : list of float
        Row and column energy boundaries.
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
    record.nt = len(row_energy_grid) + len(col_energy_grid) + r * c
    record.ne = len(row_energy_grid)
    return record


def _split_matrix_excluding_range(
    matrix: np.ndarray,
    energy_grid: List[float],
    exclude_min_ev: float,
    exclude_max_ev: float,
) -> List[Tuple[np.ndarray, List[float]]]:
    """Split a covariance matrix to exclude an energy range.

    Computes interval midpoints and removes intervals whose midpoint falls
    within ``[exclude_min_ev, exclude_max_ev]``.  Returns 0, 1, or 2
    contiguous sub-matrices for the surviving regions.
    """
    grid = np.asarray(energy_grid, dtype=float)
    midpoints = 0.5 * (grid[:-1] + grid[1:])
    keep = (midpoints < exclude_min_ev) | (midpoints > exclude_max_ev)

    if keep.all():
        return [(matrix, list(energy_grid))]
    if not keep.any():
        return []

    results: List[Tuple[np.ndarray, List[float]]] = []
    m = len(midpoints)
    i = 0
    while i < m:
        if not keep[i]:
            i += 1
            continue
        j = i
        while j < m and keep[j]:
            j += 1
        idx = np.arange(i, j)
        sub_matrix = matrix[np.ix_(idx, idx)]
        sub_grid = list(grid[i:j + 1])
        results.append((sub_matrix, sub_grid))
        i = j

    return results


# ---- MF34 builder ----------------------------------------------------------


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
    """Build an MF34MT object from a Legendre-coefficient covariance matrix.

    The covariance matrix is laid out with energies as the slow index and
    Legendre orders as the fast index::

        idx = i_energy * n_orders + (l - l_min)

    where ``l_min`` is 0 for ``ltt=2`` (a_0 included) and 1 otherwise, and
    ``n_orders = max_order - l_min + 1``.

    Parameters
    ----------
    cov_matrix : np.ndarray
        Square covariance matrix of shape
        (N_energies * n_orders, N_energies * n_orders).  Must be **relative**
        covariance: ``Cov_rel(i, j) = Cov_abs(i, j) / (mean_i * mean_j)``.
        ENDF-6 LB=5/LB=6 entries are interpreted as relative.
    energy_grid_ev : np.ndarray
        Energy boundaries in eV (``N_energies + 1`` values).
    max_order : int
        Highest Legendre order included.  With ``ltt=1`` orders span
        1..max_order; with ``ltt=2`` they span 0..max_order.
    za : float
        ZA identifier (1000*Z + A).
    awr : float
        Atomic weight ratio of the target.
    mat : int
        MAT number of the material.
    mt : int
        MT reaction number for this section.
    ltt : int, default 1
        LTT representation flag.  1: orders start at a_1.  2: orders start
        at a_0 (L=0 sub-subsections are included in the upper triangle).
    mt1 : int, optional
        Cross-correlation MT.  Defaults to ``mt`` (self-correlation).
    frame : str, default "same-as-MF4"
        Reference frame: "same-as-MF4" (LCT=0), "LAB" (LCT=1), or "CM" (LCT=2).

    Returns
    -------
    MF34MT
    """
    if mt1 is None:
        mt1 = mt

    l_min = _l_min_for_ltt(ltt)
    if max_order < l_min:
        raise ValueError(
            f"max_order={max_order} must be >= l_min={l_min} (LTT={ltt})"
        )
    n_orders = max_order - l_min + 1
    n_energies = len(energy_grid_ev) - 1
    expected_size = n_energies * n_orders

    if cov_matrix.shape != (expected_size, expected_size):
        raise ValueError(
            f"Covariance matrix shape {cov_matrix.shape} doesn't match "
            f"expected ({expected_size}, {expected_size}) for "
            f"{n_energies} energy intervals and {n_orders} Legendre orders "
            f"(LTT={ltt}, l_min={l_min}, max_order={max_order})"
        )

    if not np.all(np.isfinite(cov_matrix)):
        n_inf = int(np.sum(np.isinf(cov_matrix)))
        n_nan = int(np.sum(np.isnan(cov_matrix)))
        bad = np.argwhere(~np.isfinite(cov_matrix))
        raise ValueError(
            f"Covariance matrix contains {n_inf} inf and {n_nan} NaN values. "
            f"First 5 non-finite positions (row, col): {bad[:5].tolist()}"
        )
    if not np.all(np.isfinite(energy_grid_ev)):
        raise ValueError(
            f"Energy grid contains non-finite values: "
            f"{energy_grid_ev[~np.isfinite(energy_grid_ev)]}"
        )

    mf34 = MF34MT(number=mt)
    mf34._za = za
    mf34._awr = awr
    mf34._mat = mat
    mf34._ltt = ltt
    mf34._mf = 34

    subsection = Subsection()
    subsection.mt1 = mt1
    subsection.nl = max_order
    subsection.nl1 = max_order
    subsection.mat1 = 0.0

    lct_map = {"same-as-MF4": 0, "LAB": 1, "CM": 2}
    lct = lct_map.get(frame, 0)

    grid = list(energy_grid_ev)
    for l in range(l_min, max_order + 1):
        for l1 in range(l, max_order + 1):
            row_indices = [i * n_orders + (l - l_min) for i in range(n_energies)]
            col_indices = [i * n_orders + (l1 - l_min) for i in range(n_energies)]
            sub_matrix = cov_matrix[np.ix_(row_indices, col_indices)]

            if l == l1:
                records = [_make_lb5_record(sub_matrix, grid)]
            else:
                records = [_make_lb6_record(sub_matrix, grid, grid)]

            sub_subsec = SubSubsection(l=l, l1=l1, lct=lct, ni=1, records=records)
            subsection.sub_subsections.append(sub_subsec)

    mf34._nmt1 = 1
    mf34._subsections = [subsection]
    return mf34


# ---- merge ----------------------------------------------------------------


def merge_mf34(
    base_mf34: MF34MT,
    overlay_mf34: MF34MT,
    overlay_energy_min_ev: float,
    overlay_energy_max_ev: float,
) -> MF34MT:
    """Merge two MF34 sections, with the overlay taking precedence in a window.

    For each (L, L1) pair present in either source, the result is built on the
    union energy grid.  Inside ``[overlay_energy_min_ev, overlay_energy_max_ev]``
    the overlay's data is used; outside, the base's data is used.  Cells where
    rows and columns straddle the overlay/base boundary are set to zero (the
    two sources are treated as independent analyses).

    Pairs that exist only in the base have the overlay window excised; pairs
    that exist only in the overlay are kept on their native grid.

    Returns
    -------
    MF34MT
        New MF34MT whose LTT is set to 2 if any L=0 pair appears in the
        merged data, otherwise to the overlay's LTT (or 1 by default).
    """
    from kika.cov.legendre_covariance import LegendreCovariance  # noqa: F401

    base_cov = base_mf34.to_ang_covmat()
    overlay_cov = overlay_mf34.to_ang_covmat()

    for label, covmat in [("base", base_cov), ("overlay", overlay_cov)]:
        for i in range(covmat.num_matrices):
            mat = covmat.matrices[i]
            if not np.all(np.isfinite(mat)):
                raise ValueError(
                    f"{label.capitalize()} MF34 (L={covmat.l_rows[i]}, "
                    f"L1={covmat.l_cols[i]}) contains non-finite values: "
                    f"{int(np.sum(np.isinf(mat)))} inf, "
                    f"{int(np.sum(np.isnan(mat)))} NaN"
                )

    def _build_ll_map(covmat):
        out = {}
        for i in range(covmat.num_matrices):
            key = (covmat.l_rows[i], covmat.l_cols[i])
            out[key] = (covmat.matrices[i], list(covmat.energy_grids[i]))
        return out

    base_map = _build_ll_map(base_cov)
    overlay_map = _build_ll_map(overlay_cov)

    all_ll_pairs = sorted(set(base_map.keys()) | set(overlay_map.keys()))

    merged = MF34MT(number=overlay_mf34.number)
    merged._za = overlay_mf34._za
    merged._awr = overlay_mf34._awr
    merged._mat = overlay_mf34._mat
    merged._mf = 34

    all_l_values = {l for pair in all_ll_pairs for l in pair}
    max_order = max(all_l_values) if all_l_values else 1
    has_l0 = 0 in all_l_values
    merged._ltt = 2 if has_l0 else (overlay_mf34._ltt or 1)

    subsection = Subsection()
    subsection.mt1 = overlay_mf34.number
    subsection.nl = max_order
    subsection.nl1 = max_order
    subsection.mat1 = 0.0

    for l, l1 in all_ll_pairs:
        base_data = base_map.get((l, l1))
        overlay_data = overlay_map.get((l, l1))
        is_offdiag = (l != l1)

        if base_data is None and overlay_data is not None:
            mat, egrid = overlay_data
            egrid_f = [float(e) for e in egrid]
            records = (
                [_make_lb6_record(mat, egrid_f, egrid_f)]
                if is_offdiag else
                [_make_lb5_record(mat, egrid_f)]
            )

        elif overlay_data is None and base_data is not None:
            mat, egrid = base_data
            splits = _split_matrix_excluding_range(
                mat, egrid, overlay_energy_min_ev, overlay_energy_max_ev,
            )
            if not splits:
                continue
            if is_offdiag:
                records = [
                    _make_lb6_record(s_mat, [float(e) for e in s_grid],
                                     [float(e) for e in s_grid])
                    for s_mat, s_grid in splits
                ]
            else:
                records = [
                    _make_lb5_record(s_mat, [float(e) for e in s_grid])
                    for s_mat, s_grid in splits
                ]

        else:
            base_mat, base_grid = base_data
            overlay_mat, overlay_grid = overlay_data

            union_grid = sorted(set(base_grid) | set(overlay_grid))
            n_intervals = len(union_grid) - 1
            merged_matrix = np.zeros((n_intervals, n_intervals))

            union_arr = np.asarray(union_grid)
            midpoints = 0.5 * (union_arr[:-1] + union_arr[1:])
            in_overlay = (
                (midpoints >= overlay_energy_min_ev)
                & (midpoints <= overlay_energy_max_ev)
            )

            overlay_arr = np.asarray(overlay_grid, dtype=float)
            base_arr = np.asarray(base_grid, dtype=float)
            overlay_bins = np.searchsorted(overlay_arr, midpoints, side='right') - 1
            base_bins = np.searchsorted(base_arr, midpoints, side='right') - 1
            np.clip(overlay_bins, 0, len(overlay_grid) - 2, out=overlay_bins)
            np.clip(base_bins, 0, len(base_grid) - 2, out=base_bins)

            overlay_idx = np.where(in_overlay)[0]
            base_idx = np.where(~in_overlay)[0]

            if overlay_idx.size > 0:
                ob = overlay_bins[overlay_idx]
                merged_matrix[np.ix_(overlay_idx, overlay_idx)] = (
                    overlay_mat[np.ix_(ob, ob)]
                )
            if base_idx.size > 0:
                bb = base_bins[base_idx]
                merged_matrix[np.ix_(base_idx, base_idx)] = (
                    base_mat[np.ix_(bb, bb)]
                )

            union_grid_f = [float(e) for e in union_grid]
            records = (
                [_make_lb6_record(merged_matrix, union_grid_f, union_grid_f)]
                if is_offdiag else
                [_make_lb5_record(merged_matrix, union_grid_f)]
            )

        sub_subsec = SubSubsection()
        sub_subsec.l = l
        sub_subsec.l1 = l1
        sub_subsec.lct = 0
        sub_subsec.ni = len(records)
        sub_subsec.records = records
        subsection.sub_subsections.append(sub_subsec)

    merged._nmt1 = 1
    merged._subsections = [subsection]
    return merged


# ---- file I/O --------------------------------------------------------------


def write_mf34_to_file(
    source_endf: str,
    mf34: MF34MT,
    output_path: str,
    replace_existing: bool = True,
    update_directory: bool = True,
) -> str:
    """Write an MF34 section into an ENDF file.

    Uses ``source_endf`` as a template; either replaces the existing MF34
    section or inserts a new one immediately before the MEND marker.

    Parameters
    ----------
    source_endf : str
        Path to source ENDF file.
    mf34 : MF34MT
        Section to write.
    output_path : str
        Destination ENDF file path.
    replace_existing : bool, default True
        Replace any existing MF34 in the source.  Raises ``FileExistsError``
        if False and an MF34 is present.
    update_directory : bool, default True
        Refresh the MF1/MT451 directory after writing.
    """
    with open(source_endf, 'r') as f:
        lines = f.readlines()

    mf34_start, mf34_end = _find_mf34_boundaries(lines)
    has_mf34 = mf34_start is not None

    if has_mf34 and not replace_existing:
        raise FileExistsError(
            f"MF34 already exists in {source_endf}. "
            f"Set replace_existing=True to replace it."
        )

    mf34_content = str(mf34)
    mf34_lines = [line + '\n' for line in mf34_content.split('\n') if line.strip()]

    from ..utils import format_endf_data_line, ENDF_FORMAT_INT
    mat_num = mf34._mat or 0
    fend_line = format_endf_data_line(
        [0, 0, 0, 0, 0, 0], mat_num, 0, 0, 0,
        formats=[ENDF_FORMAT_INT] * 6
    ) + '\n'
    mf34_lines.append(fend_line)

    if has_mf34:
        skip_end = mf34_end
        if skip_end < len(lines) and len(lines[skip_end]) >= 75:
            try:
                old_mf = int(lines[skip_end][70:72].strip() or '0')
                old_mt = int(lines[skip_end][72:75].strip() or '0')
                if old_mf == 0 and old_mt == 0:
                    skip_end += 1
            except ValueError:
                pass
        new_lines = lines[:mf34_start] + mf34_lines + lines[skip_end:]
    else:
        insert_idx = _find_mend_marker(lines)
        new_lines = lines[:insert_idx] + mf34_lines + lines[insert_idx:]

    with open(output_path, 'w') as f:
        f.writelines(new_lines)

    if update_directory:
        from .update_directory import update_mf1_directory
        update_mf1_directory(output_path, added_sections={(34, mf34.number)})

    return output_path


def remove_mf34_from_file(filepath: str, update_directory: bool = True) -> bool:
    """Remove the MF34 section from an ENDF file in place.

    Returns True if MF34 was found and removed, False otherwise.
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


def _find_mf34_boundaries(lines: List[str]) -> Tuple[Optional[int], Optional[int]]:
    """Locate the MF34 block; return (start_idx, end_idx) or (None, None)."""
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


def _find_mend_marker(lines: List[str]) -> int:
    """Find the line index of the MEND marker (MAT=0, MF=0, MT=0)."""
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
