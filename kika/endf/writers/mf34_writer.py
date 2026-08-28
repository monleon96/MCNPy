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

from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import warnings
import numpy as np

from ..classes.mf34.mf34 import (
    MF34MT,
    Subsection,
    SubSubsection,
    SubSubsectionRecord,
)
from ._records import populate_lb5_record, populate_lb6_record
from ._section_writer import _find_mf_boundaries, write_mf_section_to_file

if TYPE_CHECKING:
    from pathlib import Path


# ---- helpers ---------------------------------------------------------------


def _l_min_for_ltt(ltt) -> int:
    """Lowest Legendre index L allowed by the given LTT representation.

    LTT=2 (coefficients start at a_0) and LTT=3 ("if either L or L1=0 anywhere
    in the Section", ENDF-6 manual Sec. 34.2) both admit a_0; every other value
    starts at a_1.
    """
    return 0 if int(ltt or 1) in (2, 3) else 1


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
    return populate_lb5_record(SubSubsectionRecord(), matrix, energy_grid)


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
    return populate_lb6_record(
        SubSubsectionRecord(), matrix, row_energy_grid, col_energy_grid
    )


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


def _normalize_shape_mesh(
    cov_matrix: Union[np.ndarray, Dict[Tuple[int, int], np.ndarray]],
    energy_grid_ev: Union[np.ndarray, Dict[int, np.ndarray]],
    l_min: int,
    max_order: int,
) -> Tuple[Dict[int, List[float]], Dict[Tuple[int, int], np.ndarray]]:
    """Resolve either mesh form into per-order grids and upper-triangle blocks.

    Two input forms are accepted, and both leave this function by the same door
    so the emitting loop has one code path:

    *One mesh for every order* -- ``energy_grid_ev`` an array of N+1 boundaries
    and ``cov_matrix`` the square matrix laid out ``idx = i_energy * n_orders +
    (l - l_min)``.  The blocks are the same slices the loop used to take inline,
    so this form is byte-identical to the pre-existing behaviour.

    *A mesh per order* -- ``energy_grid_ev`` a dict ``{l: boundaries}`` and
    ``cov_matrix`` a dict ``{(l, l1): block}`` covering the upper triangle,
    where block ``(l, l1)`` has shape ``(n_l, n_l1)``.  ENDF-6 asks for nothing
    more than this: each (L, L1) sub-subsection carries its own grids, LB=5 on
    one for the diagonal and LB=6 on a row/column pair off it, so orders that
    need different resolution do not have to share the finest one.

    Returns
    -------
    grids : dict
        ``{l: list of boundaries}`` for l in ``l_min..max_order``.
    blocks : dict
        ``{(l, l1): array}`` for the upper triangle, shaped ``(n_l, n_l1)``.
    """
    orders = list(range(l_min, max_order + 1))
    per_order = isinstance(energy_grid_ev, dict)
    if per_order != isinstance(cov_matrix, dict):
        raise ValueError(
            "per-order meshes need both a dict of grids and a dict of blocks; "
            f"got energy_grid_ev={type(energy_grid_ev).__name__}, "
            f"cov_matrix={type(cov_matrix).__name__}"
        )

    if not per_order:
        grid = np.asarray(energy_grid_ev, dtype=float)
        if not np.all(np.isfinite(grid)):
            raise ValueError(
                f"Energy grid contains non-finite values: "
                f"{grid[~np.isfinite(grid)]}"
            )
        n_energies = len(grid) - 1
        n_orders = len(orders)
        expected = n_energies * n_orders
        cov = np.asarray(cov_matrix, dtype=float)
        if cov.shape != (expected, expected):
            raise ValueError(
                f"Covariance matrix shape {cov.shape} doesn't match "
                f"expected ({expected}, {expected}) for "
                f"{n_energies} energy intervals and {n_orders} Legendre orders "
                f"(l_min={l_min}, max_order={max_order})"
            )
        if not np.all(np.isfinite(cov)):
            n_inf = int(np.sum(np.isinf(cov)))
            n_nan = int(np.sum(np.isnan(cov)))
            bad = np.argwhere(~np.isfinite(cov))
            raise ValueError(
                f"Covariance matrix contains {n_inf} inf and {n_nan} NaN "
                f"values. First 5 non-finite positions (row, col): "
                f"{bad[:5].tolist()}"
            )
        grids = {l: list(energy_grid_ev) for l in orders}
        blocks = {}
        for l in orders:
            rows = [i * n_orders + (l - l_min) for i in range(n_energies)]
            for l1 in range(l, max_order + 1):
                cols = [i * n_orders + (l1 - l_min) for i in range(n_energies)]
                blocks[(l, l1)] = cov[np.ix_(rows, cols)]
        return grids, blocks

    missing = [l for l in orders if l not in energy_grid_ev]
    if missing:
        raise ValueError(
            f"energy_grid_ev is missing a mesh for order(s) {missing}; "
            f"every order in {l_min}..{max_order} needs one"
        )
    grids, sizes = {}, {}
    for l in orders:
        g = np.asarray(energy_grid_ev[l], dtype=float)
        if not np.all(np.isfinite(g)):
            raise ValueError(
                f"Energy grid for order {l} contains non-finite values: "
                f"{g[~np.isfinite(g)]}"
            )
        if len(g) < 2:
            raise ValueError(
                f"Energy grid for order {l} needs at least 2 boundaries, "
                f"got {len(g)}"
            )
        grids[l], sizes[l] = list(g), len(g) - 1

    blocks = {}
    for l in orders:
        for l1 in range(l, max_order + 1):
            if (l, l1) not in cov_matrix:
                raise ValueError(
                    f"cov_matrix is missing the ({l}, {l1}) block; the whole "
                    f"upper triangle over {l_min}..{max_order} is required"
                )
            blk = np.asarray(cov_matrix[(l, l1)], dtype=float)
            if blk.shape != (sizes[l], sizes[l1]):
                raise ValueError(
                    f"cov_matrix[({l}, {l1})] shape {blk.shape} doesn't match "
                    f"expected ({sizes[l]}, {sizes[l1]}) from the meshes of "
                    f"orders {l} and {l1}"
                )
            if not np.all(np.isfinite(blk)):
                raise ValueError(
                    f"cov_matrix[({l}, {l1})] contains "
                    f"{int(np.sum(~np.isfinite(blk)))} non-finite values"
                )
            blocks[(l, l1)] = blk
    return grids, blocks




def _normalize_cross_cov(
    cross_cov: Union[Dict[int, np.ndarray], np.ndarray],
    max_order: int,
    n0: int,
    n_shape: Union[int, Dict[int, int]],
    zero_warn_threshold: float,
) -> Dict[int, np.ndarray]:
    """Validate and normalize the (L=0, L1) cross-covariance blocks.

    Accepts a dict ``{l1: (n0, n_shape) array}`` for ``l1`` in ``1..max_order``
    or a stacked array of shape ``(max_order, n0, n_shape)``.  ``n_shape`` may
    be a dict ``{l1: n}`` when the shape orders carry a mesh each, in which case
    the stacked-array form is refused -- it cannot hold ragged blocks.  Returns a dict
    keyed by ``l1``.  Rejects non-finite entries.  Because these are *relative*
    cross-covariances ``Cov(sigma, a_l)/(sigma*a_l)``, entries blow up where an
    ``a_l`` crosses zero; a warn-only diagnostic (consistent with the PSD-check
    policy) flags — but never modifies — cells whose magnitude exceeds
    ``zero_warn_threshold``.
    """
    if isinstance(cross_cov, dict):
        keys = sorted(int(k) for k in cross_cov)
        if keys != list(range(1, max_order + 1)):
            raise ValueError(
                f"cross_cov keys must be exactly 1..{max_order}, got {keys}"
            )
        # ⚑ ``None`` = DECLARED NULL, and it is not the same as an array of
        # zeros. MF34 stores no NSS field: the count is derived from NL, so a
        # (0, L1) sub-subsection cannot be omitted while a_L1 keeps its own
        # variance block. The only way to say "no cross term here" is to write
        # the pair null -- and a null block does not need the full rectangle,
        # exactly as the (0, 0) self block does not. One interval spanning the
        # same range asserts the same zero for a few bytes instead of N0 x N1.
        blocks = {l1: (None if cross_cov[l1] is None
                       else np.asarray(cross_cov[l1], dtype=float))
                  for l1 in range(1, max_order + 1)}
    else:
        if isinstance(n_shape, dict):
            raise ValueError(
                "with a mesh per order the cross blocks are ragged, so the "
                "stacked-array form cannot express them; pass a dict "
                "{l1: (n0, n_l1) array}"
            )
        arr = np.asarray(cross_cov, dtype=float)
        if arr.shape != (max_order, n0, n_shape):
            raise ValueError(
                f"cross_cov array shape {arr.shape} doesn't match expected "
                f"({max_order}, {n0}, {n_shape})"
            )
        blocks = {l1: arr[l1 - 1] for l1 in range(1, max_order + 1)}

    def _cols(l1: int) -> int:
        return n_shape[l1] if isinstance(n_shape, dict) else n_shape

    big_per_order: Dict[int, int] = {}
    for l1, blk in blocks.items():
        if blk is None:
            continue
        if blk.shape != (n0, _cols(l1)):
            raise ValueError(
                f"cross_cov[{l1}] shape {blk.shape} doesn't match expected "
                f"({n0}, {_cols(l1)})"
            )
        if not np.all(np.isfinite(blk)):
            raise ValueError(
                f"cross_cov[{l1}] contains non-finite values "
                f"({int(np.sum(~np.isfinite(blk)))} cells)"
            )
        n_big = int(np.sum(np.abs(blk) > zero_warn_threshold))
        if n_big:
            big_per_order[l1] = n_big

    if big_per_order:
        total = sum(big_per_order.values())
        warnings.warn(
            f"MF34 (L=0, L1) cross block: {total} relative cross-covariance "
            f"entries exceed {zero_warn_threshold:g} (likely a_l near a zero "
            f"crossing); values kept as-is. Per order: {big_per_order}.",
            RuntimeWarning,
            stacklevel=2,
        )
    return blocks


def create_mf34_from_covariance(
    cov_matrix: Union[np.ndarray, Dict[Tuple[int, int], np.ndarray]],
    energy_grid_ev: Union[np.ndarray, Dict[int, np.ndarray]],
    max_order: int,
    za: float,
    awr: float,
    mat: int,
    mt: int,
    ltt: int = 1,
    mt1: Optional[int] = None,
    frame: str = "same-as-MF4",
    cross_cov: Optional[Union[Dict[int, np.ndarray], np.ndarray]] = None,
    cross_energy_grid_ev: Optional[np.ndarray] = None,
    cross_zero_warn_threshold: float = 1.0e3,
) -> MF34MT:
    """Build an MF34MT object from a Legendre-coefficient covariance matrix.

    The covariance matrix is laid out with energies as the slow index and
    Legendre orders as the fast index::

        idx = i_energy * n_orders + (l - l_min)

    where ``l_min`` is 0 for ``ltt=2`` (a_0 included) and 1 otherwise, and
    ``n_orders = max_order - l_min + 1``.

    **The orders may instead carry a mesh each.**  ENDF-6 asks for nothing more
    than that: every (L, L1) sub-subsection states its own grids -- LB=5 on one
    grid for a diagonal block, LB=6 on a row/column pair off it -- so an order
    whose coefficient is well resolved does not have to be written on the mesh
    the noisiest one needs.  Pass ``energy_grid_ev`` as ``{l: boundaries}`` and
    ``cov_matrix`` as ``{(l, l1): block}`` over the upper triangle to use it;
    the two dict forms go together.  With one mesh for every order the output is
    byte-identical to the pre-existing behaviour, which
    ``_normalize_shape_mesh`` keeps true by construction: both forms leave it by
    the same door, and the shared form slices exactly as this function used to
    inline.

    Parameters
    ----------
    cov_matrix : np.ndarray or dict
        Square covariance matrix of shape
        (N_energies * n_orders, N_energies * n_orders), or -- with a mesh per
        order -- a dict ``{(l, l1): array}`` covering the upper triangle, block
        ``(l, l1)`` shaped ``(n_l, n_l1)``.  Must be **relative**
        covariance: ``Cov_rel(i, j) = Cov_abs(i, j) / (mean_i * mean_j)``.
        ENDF-6 LB=5/LB=6 entries are interpreted as relative.
    energy_grid_ev : np.ndarray or dict
        Energy boundaries in eV (``N_energies + 1`` values), or a dict
        ``{l: boundaries}`` giving order ``l`` its own mesh.
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
    cross_cov : dict or np.ndarray, optional
        Opt-in sigma↔a_l cross covariance emitted as (L=0, L1) sub-subsections.
        A dict ``{l1: (N0, N) array}`` for ``l1`` in ``1..max_order`` or a
        stacked array ``(max_order, N0, N)``, where ``N0`` is the number of
        cross-grid intervals and ``N`` the shape-grid intervals.  Entries are
        **relative** ``Cov(sigma, a_l1)/(sigma*a_l1)``.  When given, the section
        is written in the LTT=2 representation: the L>=1 shape blocks are
        byte-identical to the default (so ``cov_matrix`` must be the LTT=1
        a_1..a_L layout, i.e. pass ``ltt=1``), the (L=0, L1>=1) blocks carry the
        cross terms as LB=6 rectangular records on the coarse×shape grids, and
        the (0, 0) magnitude self-block is written **null (zeros)** — magnitude
        self-covariance belongs in MF33, not MF34, which is what the format
        prescribes (manual Sec. 34.3).  The section is emitted with **LTT=3**
        and NL = NL1 = ``max_order + 1``.  Default ``None`` leaves the output
        byte-identical to the pre-existing behavior.
    cross_energy_grid_ev : np.ndarray, optional
        Coarse (magnitude) energy boundaries in eV for the L=0 dimension of the
        cross blocks (``N0 + 1`` values).  Defaults to ``energy_grid_ev``, and
        is **required** when the shape orders carry a mesh each -- there is then
        no single shape grid to default to.  With per-order meshes ``cross_cov``
        must be the dict form: block ``l1`` is ``(N0, n_l1)``, and the stacked
        array cannot hold ragged blocks.
    cross_zero_warn_threshold : float, default 1e3
        Magnitude above which a relative cross-covariance entry triggers a
        warn-only diagnostic (a_l near a zero crossing).  Never modifies data.

    Returns
    -------
    MF34MT
    """
    if mt1 is None:
        mt1 = mt

    if cross_cov is not None:
        return _create_mf34_with_cross(
            cov_matrix, energy_grid_ev, max_order, za, awr, mat, mt,
            mt1=mt1, ltt=ltt, frame=frame,
            cross_cov=cross_cov, cross_energy_grid_ev=cross_energy_grid_ev,
            cross_zero_warn_threshold=cross_zero_warn_threshold,
        )

    l_min = _l_min_for_ltt(ltt)
    if max_order < l_min:
        raise ValueError(
            f"max_order={max_order} must be >= l_min={l_min} (LTT={ltt})"
        )
    n_orders = max_order - l_min + 1
    grids, blocks = _normalize_shape_mesh(
        cov_matrix, energy_grid_ev, l_min, max_order
    )

    mf34 = MF34MT(number=mt)
    mf34._za = za
    mf34._awr = awr
    mf34._mat = mat
    mf34._ltt = ltt
    mf34._mf = 34

    subsection = Subsection()
    subsection.mt1 = mt1
    # NL/NL1 are the NUMBER of Legendre coefficients carried, not the highest
    # index (ENDF-6 manual Sec. 34.2).  With l_min = 1 the count equals
    # max_order, so the LTT=1 output is byte-identical to before.
    subsection.nl = n_orders
    subsection.nl1 = n_orders
    subsection.mat1 = 0.0

    lct_map = {"same-as-MF4": 0, "LAB": 1, "CM": 2}
    lct = lct_map.get(frame, 0)

    for l in range(l_min, max_order + 1):
        for l1 in range(l, max_order + 1):
            sub_matrix = blocks[(l, l1)]

            if l == l1:
                records = [_make_lb5_record(sub_matrix, grids[l])]
            else:
                records = [_make_lb6_record(sub_matrix, grids[l], grids[l1])]

            sub_subsec = SubSubsection(l=l, l1=l1, lct=lct, ni=1, records=records)
            subsection.sub_subsections.append(sub_subsec)

    mf34._nmt1 = 1
    mf34._subsections = [subsection]
    return mf34


def _create_mf34_with_cross(
    cov_matrix: Union[np.ndarray, Dict[Tuple[int, int], np.ndarray]],
    energy_grid_ev: Union[np.ndarray, Dict[int, np.ndarray]],
    max_order: int,
    za: float,
    awr: float,
    mat: int,
    mt: int,
    *,
    mt1: int,
    ltt: int,
    frame: str,
    cross_cov: Union[Dict[int, np.ndarray], np.ndarray],
    cross_energy_grid_ev: Optional[np.ndarray],
    cross_zero_warn_threshold: float,
) -> MF34MT:
    """LTT=3 builder that appends (L=0, L1) sigma↔a_l cross blocks.

    The L>=1 shape blocks are built from ``cov_matrix`` exactly as the default
    LTT=1 path (so the shape covariance is unchanged); the L=0 row uses the
    coarse magnitude grid, with a null (zeros) (0, 0) block and LB=6 (0, L1)
    cross blocks.  The full upper triangle (including the zero (0, 0)) is
    emitted, which is NSS = NL*(NL+1)/2 for NL = max_order + 1 — the count the
    format defines for a section whose first coefficient is a_0.  ``cross_cov``
    may map an order to ``None``, which writes that (0, L1) pair null over one
    interval: the pair still exists (it has to — NSS is derived from NL, not
    stored), but it declares no covariance and costs a few bytes instead of a
    dense N0 x N_L1 rectangle of zeros.
    """
    if int(ltt or 1) != 1:
        raise ValueError(
            "cross_cov requires the shape cov_matrix in the LTT=1 (a_1..a_L) "
            "layout; pass ltt=1."
        )
    if mt1 != mt:
        raise ValueError(
            "cross_cov is only supported for the self pair (mt1 == mt); "
            f"got mt={mt}, mt1={mt1}."
        )
    if max_order < 1:
        raise ValueError(f"max_order={max_order} must be >= 1 for cross blocks")

    shape_grids, shape_blocks = _normalize_shape_mesh(
        cov_matrix, energy_grid_ev, 1, max_order
    )
    n_shape_per_order = {l: len(shape_grids[l]) - 1
                         for l in range(1, max_order + 1)}
    # Only ragged meshes force the dict form on ``cross_cov``. When every order
    # ends up on the same number of intervals the stacked (max_order, N0, N)
    # array is still expressible, and the API has always accepted it.
    _sizes = set(n_shape_per_order.values())
    n_shape_arg = n_shape_per_order if len(_sizes) > 1 else _sizes.pop()

    if cross_energy_grid_ev is None:
        if isinstance(energy_grid_ev, dict):
            # "defaults to energy_grid_ev" has no meaning once the shape orders
            # stop sharing one: there is no single grid to fall back to.
            raise ValueError(
                "cross_energy_grid_ev is required when the shape orders carry "
                "a mesh each -- there is no single shape grid to default to"
            )
        cross_grid_arr = energy_grid_ev
    else:
        cross_grid_arr = cross_energy_grid_ev
    if not np.all(np.isfinite(cross_grid_arr)):
        raise ValueError(
            f"Cross energy grid contains non-finite values: "
            f"{np.asarray(cross_grid_arr)[~np.isfinite(cross_grid_arr)]}"
        )
    n0 = len(cross_grid_arr) - 1

    blocks = _normalize_cross_cov(
        cross_cov, max_order, n0, n_shape_arg, cross_zero_warn_threshold
    )

    mf34 = MF34MT(number=mt)
    mf34._za = za
    mf34._awr = awr
    mf34._mat = mat
    # LTT=3 is the flag the format defines for this case: "LTT=3 if either L or
    # L1=0 anywhere in the Section" (ENDF-6 manual Sec. 34.2), and the same
    # section's NL definition reads "(The first coefficient is a0 if LTT=3,
    # a1 if LTT=1)".  LTT=2 was written here previously.
    mf34._ltt = 3
    mf34._mf = 34

    subsection = Subsection()
    subsection.mt1 = mt1
    # a_0 .. a_max_order -> max_order + 1 coefficients.  NSS = NL*(NL+1)/2 then
    # gives the (max_order+1)(max_order+2)/2 sub-subsections emitted below.
    subsection.nl = max_order + 1
    subsection.nl1 = max_order + 1
    subsection.mat1 = 0.0

    lct_map = {"same-as-MF4": 0, "LAB": 1, "CM": 2}
    lct = lct_map.get(frame, 0)

    cross_grid = list(cross_grid_arr)

    for l in range(0, max_order + 1):
        for l1 in range(l, max_order + 1):
            row_grid = cross_grid if l == 0 else shape_grids[l]
            col_grid = cross_grid if l1 == 0 else shape_grids[l1]

            if l >= 1:
                sub_matrix = shape_blocks[(l, l1)]
            elif l1 == 0:
                # Null magnitude self-block (belongs in MF33).
                sub_matrix = np.zeros((n0, n0), dtype=float)
            else:
                sub_matrix = blocks[l1]

            if l == 0 and l1 >= 1 and sub_matrix is None:
                # Declared null on one interval spanning the block's own range.
                records = [_make_lb6_record(
                    np.zeros((1, 1), dtype=float),
                    [float(row_grid[0]), float(row_grid[-1])],
                    [float(col_grid[0]), float(col_grid[-1])],
                )]
            elif l == l1:
                records = [_make_lb5_record(sub_matrix, row_grid)]
            else:
                records = [_make_lb6_record(sub_matrix, row_grid, col_grid)]

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
        New MF34MT whose LTT is set to 3 if any L=0 pair appears in the
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
    # LTT=3 is the flag for a section carrying a_0, and NL/NL1 are the NUMBER of
    # Legendre coefficients, not the highest index (ENDF-6 manual Sec. 34.2 —
    # the same pair of defects fixed in `_create_mf34_with_cross`; this was the
    # dormant second copy).  Both branches below are UNCHANGED when a_0 is
    # absent: l_min is then 1, so the count equals max_order and LTT passes
    # through.  That is what keeps `exfor_to_endf_sampling_v2` — the frozen
    # thesis pipeline and this function's only production caller, which never
    # emits a_0 blocks — bit-identical.
    merged._ltt = 3 if has_l0 else (overlay_mf34._ltt or 1)

    subsection = Subsection()
    subsection.mt1 = overlay_mf34.number
    subsection.nl = max_order + 1 if has_l0 else max_order
    subsection.nl1 = subsection.nl
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
    return write_mf_section_to_file(
        source_endf, mf34, output_path,
        replace_existing=replace_existing, update_directory=update_directory,
    )


def remove_mf34_from_file(filepath: str, update_directory: bool = True) -> bool:
    """Remove the MF34 section from an ENDF file in place.

    Returns True if MF34 was found and removed, False otherwise.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    start, end = _find_mf_boundaries(lines, 34)
    if start is None:
        return False

    new_lines = lines[:start] + lines[end:]
    with open(filepath, 'w') as f:
        f.writelines(new_lines)

    if update_directory:
        from .update_directory import update_mf1_directory
        update_mf1_directory(filepath)

    return True
