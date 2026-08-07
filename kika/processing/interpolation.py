"""One-dimensional interpolation on ENDF/GNDS interpolation-law codes.

Moved down from ``kika/endf/utils.py`` by phase 2 of the GNDS roadmap. Law
codes 1-5 are **not** ENDF-specific -- GNDS uses the same vocabulary (§3.4.4) --
so this is calculation-layer code that happened to live in the format package,
which is exactly the inverted arrow phase 2 exists to straighten.

``kika.endf.utils.interpolate_1d_endf`` stays as a live re-export of
:func:`interpolate_1d`: eight call sites inside ``kika/endf`` import that name.
It is a working adapter, not a shim awaiting deletion.
"""
from typing import List, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["interpolate_1d"]


def _regionize(nbt_int_pairs: Sequence[Tuple[int, int]], np_len: int) -> List[Tuple[int, int, int]]:
    """
    Convert ENDF (NBT, INT) pairs into 0-based [start_idx, end_idx, INT] regions.
    NBT is 1-based index of the *last* point in each region in ENDF.
    """
    if not nbt_int_pairs:
        # single region with default linear
        return [(0, np_len - 1, 2)]
    regions: List[Tuple[int, int, int]] = []
    start = 0
    for nbt, int_code in nbt_int_pairs:
        end = min(max(nbt - 1, 0), np_len - 1)
        if end >= start:
            regions.append((start, end, int_code))
        start = min(max(nbt, 0), np_len)  # next region starts at nbt (1-based → 0-based)
        if start >= np_len:
            break
    # Guard if list does not cover the tail
    if regions and regions[-1][1] < np_len - 1:
        regions.append((regions[-1][1], np_len - 1, regions[-1][2]))
    if not regions:
        regions = [(0, np_len - 1, 2)]
    return regions


def _base_int_code(int_code: int) -> int:
    """Map 11–15 → 1–5 and 21–25 → 1–5 for 1-D use."""
    if int_code >= 10:
        return int_code % 10 if int_code % 10 != 0 else 5
    return int_code


def _interp_pair(x: float, x1: float, y1: float, x2: float, y2: float, int_code: int) -> float:
    """
    Interpolate y(x) between (x1,y1) and (x2,y2) using ENDF INT code semantics (1–5).
    For INT=6 or unsupported codes → fall back to linear-linear.
    """
    if x1 == x2:
        return y1
    t = (x - x1) / (x2 - x1)
    code = _base_int_code(int_code)
    if code == 1:  # histogram/constant
        return y1
    elif code == 2:  # lin-lin
        return (1.0 - t) * y1 + t * y2
    elif code == 3:  # lin-log (y linear in ln x)
        if x1 <= 0 or x2 <= 0 or x <= 0:
            return (1.0 - t) * y1 + t * y2
        lx1, lx2, lx = math.log(x1), math.log(x2), math.log(x)
        tt = (lx - lx1) / (lx2 - lx1)
        return (1.0 - tt) * y1 + tt * y2
    elif code == 4:  # log-lin (ln y linear in x)
        if y1 <= 0 or y2 <= 0:
            return (1.0 - t) * y1 + t * y2
        ln_y = (1.0 - t) * math.log(y1) + t * math.log(y2)
        return math.exp(ln_y)
    elif code == 5:  # log-log (ln y linear in ln x)
        if y1 <= 0 or y2 <= 0 or x1 <= 0 or x2 <= 0 or x <= 0:
            return (1.0 - t) * y1 + t * y2
        lx1, lx2, lx = math.log(x1), math.log(x2), math.log(x)
        tt = (lx - lx1) / (lx2 - lx1)
        ln_y = (1.0 - tt) * math.log(y1) + tt * math.log(y2)
        return math.exp(ln_y)
    else:  # fallback for INT=6 etc.
        return (1.0 - t) * y1 + t * y2


def _interp_pair_vec(
    xq: np.ndarray, x1: np.ndarray, y1: np.ndarray,
    x2: np.ndarray, y2: np.ndarray, int_code: int,
) -> np.ndarray:
    """Vectorized interpolation between paired arrays using ENDF INT code."""
    out = np.empty_like(xq, dtype=float)
    same = x1 == x2
    if same.all():
        return y1.copy()
    diff = ~same
    dx = np.where(diff, x2 - x1, 1.0)
    t = (xq - x1) / dx
    code = _base_int_code(int_code)
    if code == 1:
        out[:] = y1
    elif code == 2:
        out[:] = (1.0 - t) * y1 + t * y2
    elif code == 3:
        safe = diff & (x1 > 0) & (x2 > 0) & (xq > 0)
        out[:] = (1.0 - t) * y1 + t * y2  # fallback
        if safe.any():
            lx1 = np.log(x1[safe]); lx2 = np.log(x2[safe]); lx = np.log(xq[safe])
            tt = (lx - lx1) / (lx2 - lx1)
            out[safe] = (1.0 - tt) * y1[safe] + tt * y2[safe]
    elif code == 4:
        safe = diff & (y1 > 0) & (y2 > 0)
        out[:] = (1.0 - t) * y1 + t * y2
        if safe.any():
            ln_y = (1.0 - t[safe]) * np.log(y1[safe]) + t[safe] * np.log(y2[safe])
            out[safe] = np.exp(ln_y)
    elif code == 5:
        safe = diff & (x1 > 0) & (x2 > 0) & (xq > 0) & (y1 > 0) & (y2 > 0)
        out[:] = (1.0 - t) * y1 + t * y2
        if safe.any():
            lx1 = np.log(x1[safe]); lx2 = np.log(x2[safe]); lx = np.log(xq[safe])
            tt = (lx - lx1) / (lx2 - lx1)
            ln_y = (1.0 - tt) * np.log(y1[safe]) + tt * np.log(y2[safe])
            out[safe] = np.exp(ln_y)
    else:
        out[:] = (1.0 - t) * y1 + t * y2
    if same.any():
        out[same] = y1[same]
    return out


def interpolate_1d(
    x_grid: ArrayLike,
    y_grid: ArrayLike,
    nbt_int_pairs: Sequence[Tuple[int, int]],
    xq: Union[float, ArrayLike],
    out_of_range: str = "zero",
) -> Union[float, np.ndarray]:
    """
    ENDF one-dimensional interpolation using (NBT, INT) regions (Table of INT codes).
    - out_of_range: 'zero'  → return 0 outside grid
                    'hold'  → hold edge value
    """
    x = np.asarray(x_grid, dtype=float)
    y = np.asarray(y_grid, dtype=float)
    scalar = np.ndim(xq) == 0
    if x.size == 0:
        return 0.0 if scalar else np.zeros_like(np.asarray(xq, dtype=float))
    regions = _regionize(nbt_int_pairs, len(x))
    xq_arr = np.atleast_1d(np.asarray(xq, dtype=float))
    n = xq_arr.size

    # Vectorized interval lookup
    k = np.searchsorted(x, xq_arr, side="right") - 1
    np.clip(k, 0, len(x) - 2, out=k)

    # Out-of-range masks
    lo_mask = xq_arr < x[0]
    hi_mask = xq_arr > x[-1]
    in_mask = ~(lo_mask | hi_mask)

    out = np.empty(n, dtype=float)
    if out_of_range == "zero":
        out[lo_mask] = 0.0
        out[hi_mask] = 0.0
    else:
        out[lo_mask] = y[0]
        out[hi_mask] = y[-1]

    if not in_mask.any():
        return float(out[0]) if scalar else out

    # In-range points
    k_in = k[in_mask]
    xq_in = xq_arr[in_mask]

    # Fast path: single region
    if len(regions) == 1:
        _, _, ic = regions[0]
        base_ic = _base_int_code(ic)
        if base_ic == 2:
            # Pure linear-linear → numpy builtin
            out[in_mask] = np.interp(xq_in, x, y)
        else:
            out[in_mask] = _interp_pair_vec(
                xq_in, x[k_in], y[k_in], x[k_in + 1], y[k_in + 1], ic
            )
    else:
        # Assign INT codes per query point from regions
        int_codes = np.full(k_in.size, 2, dtype=int)
        for start, end, ic in regions:
            rmask = (k_in + 1 >= start) & (k_in + 1 <= end)
            int_codes[rmask] = ic
        unique_codes = np.unique(int_codes)
        if unique_codes.size == 1:
            ic = int(unique_codes[0])
            if _base_int_code(ic) == 2:
                out[in_mask] = np.interp(xq_in, x, y)
            else:
                out[in_mask] = _interp_pair_vec(
                    xq_in, x[k_in], y[k_in], x[k_in + 1], y[k_in + 1], ic
                )
        else:
            result_in = np.empty(k_in.size, dtype=float)
            for ic in unique_codes:
                cm = int_codes == ic
                ki = k_in[cm]
                result_in[cm] = _interp_pair_vec(
                    xq_in[cm], x[ki], y[ki], x[ki + 1], y[ki + 1], int(ic)
                )
            out[in_mask] = result_in

    return float(out[0]) if scalar else out

