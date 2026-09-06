"""Exact integrals of a tabulated 1-d function, under the law it states.

**Why the model needs this and did not have it.** MF35 is the covariance of the
*group-integrated probabilities* of an MF5 spectrum — measured, not assumed; see
``docs/pfns/pfns_mf5_mf35_roadmap.md`` fact 4 — so perturbing a fission spectrum
from its own covariance means integrating the node over the covariance's groups,
scaling, and integrating again to check what was written. Nothing in
:mod:`kika.nuclear_data.model.functions` could integrate anything: a
:class:`~kika.nuclear_data.model.functions.xys1d.XYs1d` knew how to *evaluate*
itself and nothing else, so every integral of a model node lived in whichever
format class happened to need one.

``MF5PartialTabulated`` had exactly the four operations the node lacked —
``group_integrals``, ``normalisation``, ``table``, ``replace_table`` — which is
why perturbing MF5 was format work by construction. These functions and the
2-d methods that use them are what moves that capability onto the model; the
arithmetic underneath is :mod:`kika.processing.panel_integrals`, the same
functions the ENDF class calls, so the two cannot drift.

**Exact, or it raises.** Only histogram and lin-lin panels have a closed-form
group integral here, and the whole normalisation argument of a PFNS draw rests
on the integral being the evaluator's own rather than a quadrature of it. A
log-interpolated panel therefore raises rather than being approximated — see
:data:`kika.processing.panel_integrals.EXACT_INT_CODES`.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike

from ....processing.panel_integrals import (cumulative_integral, evaluate_table,
                                            exact_segment_codes, integral_to)

__all__ = ["tabulateFunction1d", "integrateFunction1d", "groupIntegralsOf",
           "evaluateExactly"]


def tabulateFunction1d(function1d, what: str = "") -> Tuple[np.ndarray, np.ndarray,
                                                            np.ndarray]:
    """``(xs, ys, per-interval INT codes)`` of a tabulated 1-d function.

    Works off :meth:`toEndfRegions`, which both
    :class:`~kika.nuclear_data.model.functions.xys1d.XYs1d` and
    :class:`~kika.nuclear_data.model.functions.regions1d.Regions1d` answer, so
    one region and many are the same call. A node that has no such method —
    a Legendre child, a polynomial, an isotropic distribution — is not a table
    and is refused by name rather than duck-typed into a wrong answer.
    """
    what = what or type(function1d).__name__
    toRegions = getattr(function1d, "toEndfRegions", None)
    if toRegions is None:
        raise TypeError(
            f"{what} is a {type(function1d).__name__}, which is not a tabulated "
            f"function: it has no grid to integrate over. Only XYs1d and "
            f"Regions1d carry one"
        )
    xs, ys, pairs = toRegions()
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    return xs, ys, exact_segment_codes(xs, pairs, what)


def integrateFunction1d(function1d, domainMin: Optional[float] = None,
                        domainMax: Optional[float] = None,
                        what: str = "") -> float:
    """``int_{domainMin}^{domainMax} f(x) dx``, exactly on the stated law.

    Either limit may fall inside a panel, and either may be outside the table
    altogether — the part outside contributes nothing, which is the convention
    a probability density stated on a finite grid needs: the function *is* zero
    there, it is not merely unknown.
    """
    xs, ys, codes = tabulateFunction1d(function1d, what)
    if xs.size < 2:
        return 0.0
    cumulative = cumulative_integral(xs, ys, codes)
    lo = float(xs[0]) if domainMin is None else float(domainMin)
    hi = float(xs[-1]) if domainMax is None else float(domainMax)
    if hi <= lo:
        return 0.0
    return (integral_to(xs, ys, codes, cumulative, hi)
            - integral_to(xs, ys, codes, cumulative, lo))


def groupIntegralsOf(function1d, boundaries: ArrayLike,
                     what: str = "") -> np.ndarray:
    """``P_j = int_{g_j}^{g_j+1} f`` for every group of *boundaries*.

    One cumulative integral for the whole table rather than one per group, and
    the limits are clipped into the table's own panels rather than the grid
    being refined first. That is what makes ``P_j`` the same number the
    evaluator's own integral gives — which matters here because it is the
    quantity an MF35 matrix is the covariance *of*, not a discretisation of it.
    """
    xs, ys, codes = tabulateFunction1d(function1d, what)
    edges = np.asarray(boundaries, dtype=float)
    if xs.size < 2:
        return np.zeros(max(edges.size - 1, 0), dtype=float)
    cumulative = cumulative_integral(xs, ys, codes)
    totals = np.array([integral_to(xs, ys, codes, cumulative, edge)
                       for edge in edges], dtype=float)
    return np.diff(totals)


def evaluateExactly(function1d, points: ArrayLike, what: str = "") -> np.ndarray:
    """*function1d* at *points*, zero outside its own support.

    :meth:`XYs1d.evaluate` answers the same question through
    :func:`kika.processing.interpolation.interpolate_1d`, which is the general
    interpolator and knows every law. This one is restricted to the two laws
    that refine exactly, and it exists so that a node inserted into a table by
    the same arithmetic that integrates it cannot disagree with the integral by
    a rounding of a different code path.
    """
    xs, ys, codes = tabulateFunction1d(function1d, what)
    return evaluate_table(xs, ys, codes, np.asarray(points, dtype=float))
