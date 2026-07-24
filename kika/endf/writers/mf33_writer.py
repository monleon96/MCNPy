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

import logging
from typing import Optional
import numpy as np

from ..classes.mf33.mf33 import (
    MF33MT,
    Subsection,
    NISubSubsectionRecord,
)
from ._records import populate_lb5_record, populate_lb6_record
from ._section_writer import write_mf_section_to_file

_module_logger = logging.getLogger(__name__)


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


def merge_mf33_covariance_into_host(
    host_endf: str,
    cov_matrix: np.ndarray,
    energy_grid_ev: np.ndarray,
    mt: int = 2,
    za: Optional[float] = None,
    awr: Optional[float] = None,
    mat: Optional[int] = None,
    boundary_rtol: float = 1e-6,
    logger=None,
) -> MF33MT:
    """Merge a new relative covariance into the host's MF33 MT section by range.

    The new matrix replaces the host self-covariance **inside** its energy span
    ``[energy_grid_ev[0], energy_grid_ev[-1]]``; the host values survive
    unchanged **outside** it (LB=5 is piecewise-constant, so a host bin
    straddling a boundary keeps its value exactly on the outside part).  The
    cross blocks between the replaced range and the host remainder are set to
    zero — the host's own in↔out correlations refer to the replaced in-range
    model and the new ones are unknown (documented factorization, logged).

    Guards
    ------
    - Cross-reaction subsections (``mt1 != mt``) in the host section are
      **dropped with a warning** — a cross block cannot be range-split
      meaningfully once its self side is replaced.
    - NC-type (derived) records in the self subsection raise ``ValueError``
      (a derived covariance cannot be merged additively).
    - A non-relative host self block raises ``ValueError``.
    - No host MF33 section for ``mt`` → plain new section (nothing to
      preserve), logged.

    Parameters
    ----------
    host_endf : str
        Path to the ENDF file carrying the host MF33 (usually the write
        template itself).
    cov_matrix : np.ndarray
        New **relative** covariance, square (N, N), N = len(energy_grid_ev)-1.
    energy_grid_ev : np.ndarray
        Energy boundaries of the new matrix (eV), strictly increasing.
    mt : int, default 2
        MT section to merge into.
    za, awr, mat : optional
        Header values; inherited from the host section when omitted (required
        if the host has no MF33 section for ``mt``).
    boundary_rtol : float, default 1e-6
        Relative tolerance for deduplicating host edges that coincide with the
        new range boundaries.
    logger : optional
        ``.info``/``.warning`` sink (e.g. the pipeline ``DualLogger``);
        defaults to the module logger.

    Returns
    -------
    MF33MT
        Single-subsection self-covariance section on the merged grid (one
        LB=5 record), ready for :func:`write_mf33_to_file`.
    """
    log = logger if logger is not None else _module_logger

    from ...endf import read_endf  # local import — avoids a circular import

    cov_matrix = np.asarray(cov_matrix, dtype=float)
    new_grid = np.asarray(energy_grid_ev, dtype=float)
    if np.any(np.diff(new_grid) <= 0):
        raise ValueError("energy_grid_ev must be strictly increasing")

    endf = read_endf(host_endf, mf_numbers=[33])
    mf33_file = endf.get_file(33)
    host_section = (
        mf33_file.sections.get(mt) if mf33_file is not None else None
    )

    if host_section is not None:
        if za is None:
            za = host_section._za
        if awr is None:
            awr = host_section._awr
        if mat is None:
            mat = host_section._mat

    if host_section is None:
        log.info(
            f"[MF33 merge] Host has no MF33 MT{mt} section — writing the new "
            f"covariance as-is (nothing to preserve)."
        )
        if za is None or awr is None or mat is None:
            raise ValueError(
                "za/awr/mat are required when the host has no MF33 section "
                f"for MT{mt} to inherit them from"
            )
        return create_mf33_from_covariance(
            cov_matrix, new_grid, za=za, awr=awr, mat=mat, mt=mt,
        )

    # Guard: cross-reaction subsections are dropped, loudly.
    dropped_mt1 = sorted(
        int(sub.mt1) for sub in host_section._subsections if int(sub.mt1) != mt
    )
    if dropped_mt1:
        log.warning(
            f"[MF33 merge] Host MF33 MT{mt} carries cross-reaction subsections "
            f"MT1={dropped_mt1}; they are DROPPED — a cross block cannot be "
            f"range-split once its self side is replaced."
        )

    # Guard: derived (NC) records cannot be merged additively.
    self_sub = next(
        (s for s in host_section._subsections if int(s.mt1) == mt), None
    )
    if self_sub is not None and self_sub.nc_records:
        raise ValueError(
            f"Host MF33 MT{mt} self subsection contains NC-type (derived) "
            f"records; a range merge of a derived covariance is not supported."
        )

    host_bundle = host_section._self_covariance_matrix()
    if host_bundle is None:
        log.warning(
            f"[MF33 merge] Host MF33 MT{mt} self block failed to decode — "
            f"writing the new covariance as-is."
        )
        return create_mf33_from_covariance(
            cov_matrix, new_grid, za=za, awr=awr, mat=mat, mt=mt,
        )
    host_matrix, host_grid, host_is_rel = host_bundle
    host_matrix = np.asarray(host_matrix, dtype=float)
    host_grid = np.asarray(host_grid, dtype=float)
    if not host_is_rel:
        raise ValueError(
            f"Host MF33 MT{mt} self covariance is absolute; merging with a "
            f"relative matrix is not supported (host is expected relative)."
        )

    # Merged grid: host edges strictly outside the new range + the new grid.
    e_lo, e_hi = float(new_grid[0]), float(new_grid[-1])
    below = host_grid[host_grid < e_lo * (1.0 - boundary_rtol)]
    above = host_grid[host_grid > e_hi * (1.0 + boundary_rtol)]
    merged_grid = np.concatenate([below, new_grid, above])
    if np.any(np.diff(merged_grid) <= 0):
        raise ValueError("merged energy grid is not strictly increasing")

    n_below = len(below)          # bins below e_lo (edges below + e_lo)
    n_new = len(new_grid) - 1
    n_above = len(above)          # bins above e_hi (e_hi + edges above)
    n_tot = n_below + n_new + n_above

    merged = np.zeros((n_tot, n_tot), dtype=float)
    merged[n_below:n_below + n_new, n_below:n_below + n_new] = cov_matrix

    if n_below + n_above > 0:
        # Each outside merged bin lies within exactly one host bin (its edges
        # are host edges, or the range boundary subdividing a host bin) —
        # midpoint lookup maps it; piecewise-constant values carry over exactly.
        mids = 0.5 * (merged_grid[:-1] + merged_grid[1:])
        out_idx = np.concatenate([
            np.arange(n_below), np.arange(n_below + n_new, n_tot),
        ])
        hmap = np.searchsorted(host_grid, mids[out_idx], side='right') - 1
        hmap = np.clip(hmap, 0, host_matrix.shape[0] - 1)
        merged[np.ix_(out_idx, out_idx)] = host_matrix[np.ix_(hmap, hmap)]
        # in↔out stays zero (documented factorization).

    merged = 0.5 * (merged + merged.T)

    min_eig = float(np.min(np.linalg.eigvalsh(merged)))
    if min_eig < -1e-8:
        log.warning(
            f"[MF33 merge] Merged MT{mt} relative covariance not PSD "
            f"(min eig {min_eig:.2e}) — warn only, not repaired."
        )

    log.info(
        f"[MF33 merge] MT{mt}: replaced [{e_lo:.6g}, {e_hi:.6g}] eV "
        f"({n_new} bins) inside the host block; preserved {n_below} host bins "
        f"below and {n_above} above; in↔out cross terms zeroed."
    )

    return create_mf33_from_covariance(
        merged, merged_grid, za=za, awr=awr, mat=mat, mt=mt,
    )


def write_mf33_to_file(
    source_endf: str,
    mf33: MF33MT,
    output_path: str,
    replace_existing: bool = True,
    update_directory: bool = True,
) -> str:
    """Write an MF33 section into an ENDF file.

    Thin wrapper over the shared insert-or-replace section writer: replaces
    only the matching (MF33, MT) section — sibling MF33 MT sections survive —
    or inserts the section (in MT order, or before MEND when no MF33 block
    exists), then refreshes the MF1/MT451 directory.

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
