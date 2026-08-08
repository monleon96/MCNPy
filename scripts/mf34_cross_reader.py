"""Read the MF33↔MF34 cross term out of MF34's a₀ blocks, from ONE parse.

WHY THIS MODULE EXISTS
----------------------
The cross covariance between the cross-section magnitude and the Legendre
coefficients has a home in the format: ENDF-6 Sec. 34.1 says "one expresses
covariances with the a0 Legendre coefficients even though a0 = 1 in the ENDF
system", and Sec. 34.3 says the (0,0) magnitude self-block belongs to MF33 and
must not be repeated. `kika.endf.writers.mf34_writer` already emits exactly
that: (L=0, L1) sub-subsections as LB=6 **rectangular** records with an
independent row (magnitude) grid and column (shape) grid, LTT=3, NL = L_max+1,
and a null (0,0).

The χ² could not read them, for two separate reasons:

1. `eval_covariance.build_mf34_block` skips any block with `l_r < 1`. That skip
   is deliberate and it STAYS — it is what makes "no double counting"
   structural rather than asserted (roadmap §L8).
2. `MF34MT.to_ang_covmat` projects every LB=6 record onto ``union(row, col)``
   and hands back a SQUARE matrix on one grid, because `LegendreCovariance`
   stores one energy grid per block. The projection is value-preserving, but
   the two distinct grids are gone — and the magnitude grid is exactly what the
   fold needs, because its leg must be built by the same call
   `build_mf33_block` makes.

So this module parses MF34 once and splits it:

    * the L≥1 family goes to `to_ang_covmat()` as before, with the a₀
      sub-subsections REMOVED first — otherwise they are lifted onto a ~3020-bin
      union and retained at ~440 MB per library load, in every χ² scenario, for
      nothing;
    * the a₀ blocks come back on their native grids, in the
      ``{"l", "shape_grid_ev", "matrix", "is_relative"}`` shape that
      `build_mf33_mf34_cross_block` already consumes, guarded and correct.

⚑ ONE PARSE, AND THAT IS THE POINT. Routing the cross term through a second
reader is the shape of every failure in this track's history — §L, §L3 and §L9
are all "a joint was certified that was not the joint being shipped". Returning
both objects from a single `read_endf` makes "same file, same MT, same MT1,
same LTT" structural. Do not add an entry point that reads the a₀ blocks
without also producing the marginals they must be Cauchy–Schwarz-compatible
with.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kika.endf import read_endf

__all__ = ["read_mf34_split", "MF34WithCross"]


class MF34WithCross(tuple):
    """``(mf34, cross, info)`` — the Legendre marginals and the a₀ cross blocks.

    A tuple so callers can unpack it, with names so they do not have to.
    """

    __slots__ = ()

    def __new__(cls, mf34, cross, info):
        return super().__new__(cls, (mf34, cross, info))

    @property
    def mf34(self):
        """`LegendreCovariance` over L ≥ 1 only. Never carries an a₀ block."""
        return self[0]

    @property
    def cross(self) -> List[Dict]:
        """Block dicts for `build_mf33_mf34_cross_block`. Empty if the file
        ships no a₀ blocks, which is the case for JEFF-4.0 and JENDL-5."""
        return self[1]

    @property
    def info(self) -> Dict:
        return self[2]


def _self_subsection(mt_sec, mt: int):
    """The (MT, MT1=MT) subsection — the only one that can carry a₀ blocks.

    `mf34_writer` refuses `cross_cov` for `mt1 != mt`, and a magnitude↔shape
    correlation across two different reactions is not what MF33/MT2 describes.
    """
    subs = list(getattr(mt_sec, "_subsections", []) or [])
    for sub in subs:
        if int(sub.mt1 or 0) == int(mt):
            return sub
    return subs[0] if subs else None


def _self_block_is_null(sub_subsec) -> Tuple[bool, float, int]:
    """Is the (0,0) magnitude self-block null? ``(ok, worst, n_bins)``.

    It is LB=5, not LB=6 — square blocks always are — and its grid may be
    either the full magnitude axis or a single interval spanning it. Both mean
    the same thing and both are accepted: the second is the cheap form, because
    a full 2317-bin upper triangle of zeros is 2.685 M numbers of ASCII nothing.
    """
    worst, n_bins = 0.0, 0
    for rec in sub_subsec.records or []:
        vals = np.asarray(list(rec.matrix) + list(rec.raw_list_values or []),
                          dtype=float)
        if vals.size:
            worst = max(worst, float(np.abs(vals).max()))
        n_bins = max(n_bins, max(len(rec.energies) - 1, 0))
    return worst == 0.0, worst, n_bins


def _a0_block_from(sub_subsec, l1: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(row_edges_ev, col_edges_ev, matrix)`` from one (0, L1) LB=6 record."""
    recs = list(sub_subsec.records or [])
    if len(recs) != 1:
        raise ValueError(
            f"MF34 a_0 block (0, {l1}) carries {len(recs)} NI records; the "
            f"cross term is written as exactly one LB=6 rectangular record, so "
            f"this file was not produced by `create_mf34_from_covariance` and "
            f"its magnitude axis cannot be identified."
        )
    rec = recs[0]
    lb = int(rec.lb or 0)
    if lb != 6:
        raise ValueError(
            f"MF34 a_0 block (0, {l1}) is LB={lb}, expected LB=6. Only LB=6 "
            f"carries an independent row grid, and without one the magnitude "
            f"leg cannot be placed on the MF33 grid (roadmap §10.7-2(a))."
        )
    row = np.asarray(rec.row_energies, dtype=float)
    col = np.asarray(rec.col_energies, dtype=float)
    mat = np.asarray(rec.rect_matrix, dtype=float).reshape(row.size - 1, col.size - 1)
    return row, col, mat


def _grids_equal(a: np.ndarray, b: np.ndarray, rtol: float) -> bool:
    if a.shape != b.shape:
        return False
    if rtol <= 0.0:
        return bool(np.array_equal(a, b))
    scale = np.maximum(np.abs(a), np.abs(b))
    scale[scale == 0.0] = 1.0
    return bool((np.abs(a - b) / scale <= rtol).all())


def read_mf34_split(
    path,
    *,
    isotope: int = 26056,
    mt: int = 2,
    l_max: int = 6,
    mf33_grid_ev: Optional[Sequence[float]] = None,
    energy_unit: str = "MeV",
    grid_rtol: float = 1e-12,
    require_cross: bool = False,
) -> MF34WithCross:
    """Parse MF34 once; return the L≥1 marginals and the a₀ cross blocks.

    Parameters
    ----------
    path : str or Path
        The ENDF file.
    isotope, mt : int
        Filter applied to the returned `LegendreCovariance`, matching what
        `load_library_lib_c0` did with `filter_by_isotope_reaction`.
    l_max : int
        Orders kept. An a₀ block for ``L1 > l_max`` is dropped, like the self
        blocks, so the parameter space the fold sees stays square.
    mf33_grid_ev : (M33+1,) array, optional
        The grid the file's own MF33 comes back on. When given, every a₀
        block's ROW grid must equal it — see the guards below. Pass it whenever
        the blocks are going to be folded; omit only for inspection.
    grid_rtol : float
        Tolerance for the grid comparisons, and the number is measured rather
        than chosen for comfort (Phase 0.7, run 86's shipped ``_mg.endf``).

        Bit-exact is *not* achievable across MF sections: only 1126 of 1739 of
        our fine edges come back bit-identical, because the parser evaluates
        ``mantissa * 10**exp`` and e.g. ``2.000500+6`` lands on
        ``2000500.0000000002`` while the in-memory grid holds
        ``2000499.9999999998``. Worst observed disagreement: **2.3e-16
        relative — one ULP.**

        It *is* achievable when the a_0 row grid is written from the array MF33
        was read back into, because that round trip is idempotent. Do that
        anyway; this tolerance is the belt, not the braces. 1e-12 sits four
        orders above the observed noise and nine below the narrowest bin
        (~1.9 keV in ~2 MeV, i.e. ~1e-3), so it cannot mask a real regridding.
    require_cross : bool
        Raise if the file carries no a₀ blocks. Use when the caller was told to
        score a cross term — silently scoring zero is the failure this whole
        section exists to prevent.

    Guards
    ------
    All four exist to buy back the guarantee that reading the cross term
    separately from the marginals would otherwise cost:

    * every a₀ block's COLUMN grid equals every L≥1 block's energy grid. If it
      does not, the fold applies `PointMap.nearest(col_grid)` to the cross and
      `PointMap.nearest(block_grid)` to the self blocks, and Σ_eval stops being
      a congruence. This is the live risk once anything routes through
      `merge_mf34`, whose per-pair union produces four different grids
      (roadmap §L18, §10.7-4 step 4).
    * every a₀ block's ROW grid equals ``mf33_grid_ev``, so the magnitude leg is
      literally `_mf33_magnitude_map`'s output with no regridding.
    * every a₀ block is relative, matching the family (§L13).
    * the (0,0) block is null, because the magnitude self-covariance belongs to
      MF33 and counting it twice would double the magnitude variance.
    """
    from kika.cov import MF34CovMat  # noqa: F401  (kept local; heavy import)

    path = str(path)
    endf = read_endf(path, mf_numbers=[34])
    mf34_file = endf.get_file(34)
    if mf34_file is None or mt not in mf34_file.sections:
        raise ValueError(f"{path} has no MF34/MT={mt}")
    mt_sec = mf34_file.sections[mt]

    sub = _self_subsection(mt_sec, mt)
    if sub is None:
        raise ValueError(f"{path} MF34/MT={mt} has no subsections")

    # ---- split, before any projection happens ------------------------------
    keep, a0 = [], []
    for ss in list(sub.sub_subsections or []):
        l_r, l_c = int(ss.l or 0), int(ss.l1 or 0)
        (a0 if (l_r == 0 or l_c == 0) else keep).append(ss)

    ltt = int(getattr(mt_sec, "_ltt", 0) or 0)
    info: Dict = {
        "form": "file_a0",
        "path": path,
        "ltt": ltt,
        "n_a0_subsubsections": len(a0),
        "n_shape_subsubsections": len(keep),
    }

    if a0 and ltt != 3:
        raise ValueError(
            f"{path} carries {len(a0)} a_0 sub-subsections but declares LTT="
            f"{ltt}. The manual (Sec. 34.2) reserves LTT=3 for 'either L or "
            f"L1=0 anywhere in the Section'; a file that disagrees with itself "
            f"here will be misread by every code that trusts LTT."
        )

    # ⚑ Remove the a_0 blocks BEFORE to_ang_covmat. Left in, each is projected
    # onto union(row, col) — for a 2317-row magnitude axis against a 703-group
    # shape axis that is a ~3020² dense matrix per order, retained for the
    # lifetime of the load, consumed by nothing.
    sub.sub_subsections = keep
    covmat = mt_sec.to_ang_covmat()
    if hasattr(covmat, "filter_by_isotope_reaction"):
        covmat = covmat.filter_by_isotope_reaction(isotope, mt)

    if not a0:
        if require_cross:
            raise ValueError(
                f"{path} ships no MF34 a_0 blocks, so there is no cross term to "
                f"read. The caller asked for one; scoring a silent zero instead "
                f"would misattribute the result."
            )
        info["n_orders"] = 0
        return MF34WithCross(covmat, [], info)

    # ---- the shape grid every leg must share -------------------------------
    shape_grids = [np.asarray(g, dtype=float) for g in covmat.energy_grids]
    if not shape_grids:
        raise ValueError(f"{path} carries a_0 blocks but no L>=1 blocks to "
                         f"correlate them with")
    ref_shape = shape_grids[0]
    for i, g in enumerate(shape_grids[1:], start=1):
        if not _grids_equal(g, ref_shape, grid_rtol):
            raise ValueError(
                f"{path} MF34 block {i} (L={covmat.l_rows[i]},"
                f"{covmat.l_cols[i]}) sits on a grid of {g.size} edges while "
                f"block 0 has {ref_shape.size}. A cross term written against "
                f"one shape grid cannot be folded next to self blocks on "
                f"several — that is §L18's four-grid problem, and it silently "
                f"breaks the congruence. Force one grid in `merge_mf34` "
                f"(§10.7-4 step 4) before shipping a_0 blocks."
            )

    mf33_grid = (None if mf33_grid_ev is None
                 else np.asarray(mf33_grid_ev, dtype=float))

    cross: List[Dict] = []
    row_ref = None
    for ss in a0:
        l_r, l_c = int(ss.l or 0), int(ss.l1 or 0)
        l1 = l_c if l_r == 0 else l_r

        if l1 == 0:
            ok, worst, n_bins = _self_block_is_null(ss)
            if not ok:
                raise ValueError(
                    f"{path} MF34 (0,0) block is not null (max |value| = "
                    f"{worst:.6g}). The magnitude self-covariance belongs to "
                    f"MF33 (manual Sec. 34.3); a non-null (0,0) here would be "
                    f"folded on top of the MF33 self block and double the "
                    f"magnitude variance."
                )
            info["null_self_block_bins"] = n_bins
            continue

        if l1 > l_max:
            continue

        row, col, mat = _a0_block_from(ss, l1)

        if not _grids_equal(col, ref_shape, grid_rtol):
            raise ValueError(
                f"{path} a_0 block (0, {l1}) has a column grid of {col.size} "
                f"edges, but the L>=1 blocks are on {ref_shape.size}. The "
                f"shape leg of the cross term must reach the points through "
                f"the SAME map as the self blocks or Sigma_eval is not "
                f"M J M^T (roadmap §10.7-2(a))."
            )
        if mf33_grid is not None and not _grids_equal(row, mf33_grid, grid_rtol):
            d = ("shape mismatch" if row.shape != mf33_grid.shape
                 else f"max rel diff {np.abs(row - mf33_grid).max():.3e}")
            raise ValueError(
                f"{path} a_0 block (0, {l1}) has a magnitude grid of "
                f"{row.size} edges but the file's own MF33 comes back on "
                f"{mf33_grid.size} ({d}). The magnitude leg must be "
                f"`_mf33_magnitude_map` on the MF33 grid, unmodified — a cross "
                f"term is Cauchy-Schwarz-compatible only with the marginals it "
                f"was built from (§10.1)."
            )
        if row_ref is None:
            row_ref = row
        elif not _grids_equal(row, row_ref, grid_rtol):
            raise ValueError(
                f"{path} a_0 blocks disagree on their magnitude grid "
                f"({row.size} vs {row_ref.size} edges); they are legs of ONE "
                f"parameter family and cannot sit on two axes."
            )

        cross.append({
            "l": l1,
            "shape_grid_ev": col,
            "matrix": mat,
            # LB=5/6 are relative by definition (only LB=0 is absolute, and it
            # carries no off-diagonal structure), so a file that reached here
            # is relative on both axes. Stated rather than inferred, because
            # `build_mf33_mf34_cross_block` refuses a mismatch against the
            # family's flag and the reason must be readable when it does.
            "is_relative": True,
        })

    cross.sort(key=lambda b: b["l"])
    info.update(
        n_orders=len(cross),
        orders=[b["l"] for b in cross],
        n_mag_bins=(0 if row_ref is None else int(row_ref.size - 1)),
        n_shape_bins=int(ref_shape.size - 1),
        mag_range_ev=(None if row_ref is None
                      else (float(row_ref[0]), float(row_ref[-1]))),
        max_abs=float(max((np.abs(b["matrix"]).max() for b in cross), default=0.0)),
    )
    return MF34WithCross(covmat, cross, info)
