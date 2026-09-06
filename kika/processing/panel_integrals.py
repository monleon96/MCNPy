"""Exact panel arithmetic on an ENDF/GNDS interpolation-law table.

Three things every tabulated function needs and no interpolator provides: the
law code of each *interval*, the exact cumulative integral under those laws, and
the exact integral up to an arbitrary limit inside a panel. Plus evaluation,
which the general interpolator does do but not with the "zero outside the
table's own support" convention a probability density needs.

**Moved down from** ``kika/endf/classes/mf5/partials.py``, the same way
:mod:`kika.processing.interpolation` came down from ``kika/endf/utils.py`` in
phase 2 of the GNDS roadmap, and for the same reason: law codes 1-5 are GNDS
vocabulary (§3.4.4) as much as ENDF's, so integrating a table under them is
calculation-layer work that happened to live in a format package. The move was
forced rather than tidy — the model's own
:mod:`~kika.nuclear_data.model.functions` needed the same four functions to give
an energy-distribution node the group integrals MF35 is the covariance *of*, and
``kika/nuclear_data`` may not import ``kika.endf``
(``kika/tests/test_layering.py``). The alternative was a second implementation
of the same integral, which is exactly the drift that makes a normalisation
silently wrong.

``kika.endf.classes.mf5.partials`` re-exports all four under their original
names, so the MF5 classes and :mod:`kika.endf.classes.mf5.analytic` import them
from where they always did.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

__all__ = ["EXACT_INT_CODES", "exact_segment_codes", "cumulative_integral",
           "integral_to", "evaluate_table"]

#: Interpolation codes that can be integrated and refined exactly here.
#: 1 is histogram, 2 is lin-lin. Everything else — log in either axis — has no
#: exact group integral in closed form here and no exact node insertion, and
#: the whole normalisation argument of the PFNS sampler rests on both being
#: exact. So they raise rather than approximate.
EXACT_INT_CODES = (1, 2)


def exact_segment_codes(x: Sequence[float],
                        interp: Sequence[Tuple[int, int]],
                        what: str) -> np.ndarray:
    """The ENDF INT code of every interval of a table, restricted to 1 and 2.

    Rejects anything else, naming the code and *what* the table is. See
    :data:`EXACT_INT_CODES` for why approximating would be the wrong favour to
    do the caller.
    """
    n = len(x)
    pairs = list(interp) or [(n, 2)]
    codes = np.empty(max(n - 1, 0), dtype=int)
    start = 0
    for nbt, code in pairs:
        stop = min(int(nbt) - 1, n - 1)
        if stop > start:
            codes[start:stop] = int(code)
        start = max(start, stop)
    if start < n - 1:                      # NBT short of NP: hold the last
        codes[start:] = int(pairs[-1][1])
    bad = sorted(set(int(c) for c in codes) - set(EXACT_INT_CODES))
    if bad:
        raise NotImplementedError(
            f"{what} uses interpolation code(s) {bad}; only "
            f"{list(EXACT_INT_CODES)} (histogram, lin-lin) have an exact "
            f"group integral and an exact node insertion here"
        )
    return codes


def cumulative_integral(x: np.ndarray, y: np.ndarray,
                        codes: np.ndarray) -> np.ndarray:
    """``[0, int_{x0}^{x1}, int_{x0}^{x2}, ...]`` -- exact on the stated law."""
    if x.size < 2:
        return np.zeros(x.size, dtype=float)
    dx = np.diff(x)
    panel = np.where(codes == 1, y[:-1] * dx, 0.5 * (y[:-1] + y[1:]) * dx)
    return np.concatenate(([0.0], np.cumsum(panel)))


def integral_to(x: np.ndarray, y: np.ndarray, codes: np.ndarray,
                cumulative: np.ndarray, limit: float) -> float:
    """``int_{x0}^{limit}``, with *limit* anywhere inside or outside the table."""
    if x.size < 2:
        return 0.0
    if limit <= x[0]:
        return 0.0
    if limit >= x[-1]:
        return float(cumulative[-1])
    i = int(np.searchsorted(x, limit, side="right")) - 1
    i = min(max(i, 0), x.size - 2)
    span = x[i + 1] - x[i]
    t = 0.0 if span == 0 else (limit - x[i]) / span
    if codes[i] == 1:
        partial = y[i] * (limit - x[i])
    else:
        y_at = y[i] + t * (y[i + 1] - y[i])
        partial = 0.5 * (y[i] + y_at) * (limit - x[i])
    return float(cumulative[i] + partial)


def evaluate_table(x: np.ndarray, y: np.ndarray, codes: np.ndarray,
                   points: np.ndarray) -> np.ndarray:
    """The table evaluated at *points*, zero outside its own support."""
    out = np.zeros(np.shape(points), dtype=float)
    if x.size == 0:
        return out
    points = np.asarray(points, dtype=float)
    inside = (points >= x[0]) & (points <= x[-1])
    if not np.any(inside):
        return out
    if x.size == 1:
        out[inside] = y[0]
        return out
    idx = np.clip(np.searchsorted(x, points[inside], side="right") - 1,
                  0, x.size - 2)
    span = x[idx + 1] - x[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(span > 0, (points[inside] - x[idx]) / span, 0.0)
    linear = y[idx] + t * (y[idx + 1] - y[idx])
    out[inside] = np.where(codes[idx] == 1, y[idx], linear)
    return out
