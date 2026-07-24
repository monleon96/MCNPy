"""MF33 (Cross-Section Covariance) creation and writing utilities.

Mirrors :mod:`kika.endf.writers.mf34_writer` for the magnitude channel.  MF33
carries the covariance of a File-3 cross section (here, elastic MT2), so — unlike
MF34 — there is no Legendre (L, L1) triangle: a single reaction pair (mt, mt1)
maps to one square self-block (LB=5) or, for a future magnitude-grid ≠ shape-grid
case, one rectangular cross-block (LB=6).

The MF33 read side already exists (:class:`~kika.endf.classes.mf33.mf33.MF33MT`
with a full ``__str__`` serializer, and ``parse_mf33`` / ``parse_mf33_mt``); only
the from-covariance builder and file insertion were missing.  The LB=5/LB=6 record
packing and the insert-or-replace file writer are shared with the MF34 path via
``_records`` and ``_section_writer``.
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from ..classes.mf33.mf33 import (
    MF33MT,
    Subsection,
    NISubSubsectionRecord,
)
from ._records import populate_lb5_record, populate_lb6_record
from ._section_writer import write_mf_section_to_file


def create_mf33_from_covariance(
    cov_matrix: np.ndarray,
    energy_grid_ev: np.ndarray,
    za: float,
    awr: float,
    mat: int,
    mt: int,
    mt1: Optional[int] = None,
    lb: int = 5,
    col_energy_grid_ev: Optional[np.ndarray] = None,
    mtl: int = 0,
) -> MF33MT:
    """Build an ``MF33MT`` object from a cross-section covariance matrix.

    The covariance is a single square block indexed by energy interval::

        idx = i_energy      (N_energies = len(energy_grid_ev) - 1)

    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix.  For ``lb=5`` it is square symmetric of shape
        (N, N) with N = len(energy_grid_ev) - 1.  For ``lb=6`` it is
        rectangular (N_row, N_col).  Must be **relative** covariance
        (``Cov_abs(i, j) / (mean_i * mean_j)``): ENDF-6 LB=5/LB=6 entries are
        interpreted as relative.
    energy_grid_ev : np.ndarray
        Energy boundaries in eV (N_energies + 1 values).  For ``lb=6`` these
        are the row boundaries.
    za : float
        ZA identifier (1000*Z + A).
    awr : float
        Atomic weight ratio of the target.
    mat : int
        MAT number of the material.
    mt : int
        MT reaction number for this section (e.g. 2 for elastic).
    mt1 : int, optional
        Cross-correlation MT.  Defaults to ``mt`` (self-covariance).
    lb : int, default 5
        Covariance pattern flag.  5: symmetric square block (self, ``mt1==mt``).
        6: rectangular block, for a future magnitude-grid ≠ shape-grid case
        (exposed but not yet used by the pipeline).
    col_energy_grid_ev : np.ndarray, optional
        Column energy boundaries (eV) for ``lb=6``.  Defaults to
        ``energy_grid_ev`` when omitted.
    mtl : int, default 0
        Lumped-reaction flag (non-zero marks this MT as a component of a
        lumped MTL); 0 for an ordinary self-covariance section.

    Returns
    -------
    MF33MT
    """
    if mt1 is None:
        mt1 = mt

    if lb not in (5, 6):
        raise ValueError(f"lb must be 5 or 6, got {lb}")

    n_row = len(energy_grid_ev) - 1
    if lb == 5:
        n_col = n_row
    else:
        col_grid = energy_grid_ev if col_energy_grid_ev is None else col_energy_grid_ev
        n_col = len(col_grid) - 1
    expected_shape = (n_row, n_col)

    if cov_matrix.shape != expected_shape:
        raise ValueError(
            f"Covariance matrix shape {cov_matrix.shape} doesn't match "
            f"expected {expected_shape} for {n_row} row and {n_col} column "
            f"energy intervals (lb={lb})"
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

    record = NISubSubsectionRecord()
    if lb == 5:
        populate_lb5_record(record, cov_matrix, list(energy_grid_ev))
    else:
        col_grid = energy_grid_ev if col_energy_grid_ev is None else col_energy_grid_ev
        if not np.all(np.isfinite(col_grid)):
            raise ValueError(
                f"Column energy grid contains non-finite values: "
                f"{np.asarray(col_grid)[~np.isfinite(col_grid)]}"
            )
        populate_lb6_record(record, cov_matrix, list(energy_grid_ev), list(col_grid))

    subsection = Subsection(
        xmf1=0.0,
        xlfs1=0.0,
        mat1=0,
        mt1=mt1,
        nc=0,
        ni=1,
        ni_records=[record],
    )

    mf33 = MF33MT(number=mt)
    mf33._za = za
    mf33._awr = awr
    mf33._mat = mat
    mf33._mtl = mtl
    mf33._mf = 33
    mf33._nl = 1
    mf33._subsections = [subsection]
    return mf33


def write_mf33_to_file(
    source_endf: str,
    mf33: MF33MT,
    output_path: str,
    replace_existing: bool = True,
    update_directory: bool = True,
) -> str:
    """Write an MF33 section into an ENDF file.

    Thin wrapper over the shared insert-or-replace section writer: replaces an
    existing MF33 block (e.g. the host MF33 MT2) or inserts a new one before the
    MEND marker, then refreshes the MF1/MT451 directory.

    Parameters
    ----------
    source_endf : str
        Path to the source ENDF file used as a template.
    mf33 : MF33MT
        Section to write.
    output_path : str
        Destination ENDF file path.
    replace_existing : bool, default True
        Replace any existing MF33.  Raises ``FileExistsError`` if False and an
        MF33 is present.
    update_directory : bool, default True
        Refresh the MF1/MT451 directory after writing.
    """
    return write_mf_section_to_file(
        source_endf, mf33, output_path,
        replace_existing=replace_existing, update_directory=update_directory,
    )
