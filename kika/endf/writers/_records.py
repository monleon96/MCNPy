"""Shared LB=5 / LB=6 covariance LIST-record builders.

Both MF34 (``SubSubsectionRecord``) and MF33 (``NISubSubsectionRecord``) store
their LB=5 symmetric and LB=6 rectangular covariance blocks with identical field
names — ``ls/lb/ne/nt`` plus ``energies/matrix`` (LB=5) and
``row_energies/col_energies/rect_matrix`` (LB=6).  These helpers populate a
caller-supplied record instance so the numeric packing lives in one place and
the two writers cannot drift apart.
"""
from __future__ import annotations

from typing import List
import numpy as np


def populate_lb5_record(record, matrix: np.ndarray, energy_grid: List[float]):
    """Populate ``record`` as an LB=5 (LS=1, symmetric upper triangle) block.

    Use for diagonal blocks (self-covariance) where the matrix is symmetric.

    Parameters
    ----------
    record : object
        A record instance (MF34 ``SubSubsectionRecord`` or MF33
        ``NISubSubsectionRecord``) whose ``ls/lb/ne/nt/energies/matrix``
        attributes are set in place.
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
    record.ls = 1
    record.lb = 5
    record.ne = len(energy_grid)
    record.energies = list(energy_grid)
    triu_rows, triu_cols = np.triu_indices(m)
    record.matrix = matrix[triu_rows, triu_cols].tolist()
    record.nt = len(energy_grid) + len(record.matrix)
    return record


def populate_lb6_record(
    record,
    matrix: np.ndarray,
    row_energy_grid: List[float],
    col_energy_grid: List[float],
):
    """Populate ``record`` as an LB=6 (rectangular matrix) block.

    Use for off-diagonal / cross blocks where the matrix is asymmetric or the
    row and column grids differ.

    Parameters
    ----------
    record : object
        A record instance whose ``ls/lb/ne/nt/row_energies/col_energies/
        rect_matrix`` attributes are set in place.
    matrix : np.ndarray
        Matrix of shape (r, c) where r = len(row_energy_grid) - 1 and
        c = len(col_energy_grid) - 1.
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
    record.ls = 0
    record.lb = 6
    record.row_energies = list(row_energy_grid)
    record.col_energies = list(col_energy_grid)
    record.rect_matrix = matrix.ravel().tolist()
    record.nt = len(row_energy_grid) + len(col_energy_grid) + r * c
    record.ne = len(row_energy_grid)
    return record
