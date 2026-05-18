"""
Positivity check / projection for MF4 angular distributions.

The MF4 angular probability density is

    f(mu, E) = (1/2) * sum_l (2l+1) * a_l(E) * P_l(mu),   a_0 = 1

with a_l stored in ENDF (a_0 = 1 implicit). f(mu) >= 0 is the physicality
requirement; it is equivalent to the angular CDF being non-decreasing in mu.

These helpers take ENDF a_l vectors (a_0 included) and either check
positivity on a mu-grid or project to the L^2-nearest non-negative set via
SLSQP. The (2l+1) weighting is applied internally so callers pass the raw
ENDF a-vector.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from numpy.polynomial.legendre import legval, legvander


def _legendre_weights(L_plus_one: int) -> np.ndarray:
    return 2.0 * np.arange(L_plus_one) + 1.0


def check_mf4_positivity(
    a_endf: np.ndarray,
    n_points: int = 101,
) -> tuple[bool, float]:
    """
    Test whether f(mu) = (1/2) sum_l (2l+1) a_l P_l(mu) >= 0 on a mu-grid.

    Parameters
    ----------
    a_endf : np.ndarray
        ENDF Legendre coefficients including a_0. Shape (L+1,).
    n_points : int
        Number of evenly-spaced check points in [-1, 1].

    Returns
    -------
    is_positive : bool
    sigma_min : float
        Minimum value of the (un-normalized) polynomial
        sum_l (2l+1) a_l P_l(mu) on the grid. Negative iff the sample is
        unphysical.
    """
    a = np.asarray(a_endf, dtype=float)
    weights = _legendre_weights(len(a))
    b = weights * a
    mu = np.linspace(-1.0, 1.0, n_points)
    sigma = legval(mu, b)
    sigma_min = float(np.min(sigma))
    return sigma_min >= 0.0, sigma_min


def project_mf4_to_positive(
    a_endf: np.ndarray,
    n_points: int = 101,
    frozen_indices: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """
    Project ENDF a_l to the L^2-nearest set producing f(mu) >= 0.

    Minimizes ||a* - a||^2 subject to
        sum_l (2l+1) a*_l P_l(mu_j) >= 0  for all grid points
        a*_0 = 1                          (always pinned)
        a*_i = v                          for (i, v) in frozen_indices
    via SLSQP.

    Returns the input unchanged if already non-negative on the grid.

    Parameters
    ----------
    a_endf : np.ndarray
        Input ENDF a-vector (a_0 included).
    n_points : int
        Number of constraint points.
    frozen_indices : dict, optional
        Coefficient index -> pinned value. Used to hold the high-l tail
        (outside the perturbed band) at its baseline. a_0 is always pinned
        to 1.0 and does not need to be included here.

    Returns
    -------
    np.ndarray
        Projected coefficients, same shape as input.
    """
    from scipy.optimize import minimize

    a = np.asarray(a_endf, dtype=float).copy()
    n = len(a)
    weights = _legendre_weights(n)

    mu = np.linspace(-1.0, 1.0, n_points)
    # vander[j, l] = P_l(mu_j); C = vander * weights along axis=1
    vander = legvander(mu, n - 1)
    C = vander * weights[np.newaxis, :]

    if np.all(C @ a >= 0.0):
        return a

    def objective(c):
        diff = c - a
        return 0.5 * float(diff @ diff)

    def grad(c):
        return c - a

    constraints = {
        "type": "ineq",
        "fun": lambda c: C @ c,
        "jac": lambda c: C,
    }

    # a_0 = 1 always, plus any extra pins from the caller.
    pins: Dict[int, float] = {0: 1.0}
    if frozen_indices:
        pins.update(frozen_indices)

    bounds = [(None, None)] * n
    for idx, val in pins.items():
        if not (0 <= idx < n):
            continue
        bounds[idx] = (val, val)
        a[idx] = val  # also seed x0 so SLSQP starts feasible on bounds

    x0 = np.asarray(a_endf, dtype=float).copy()
    for idx, val in pins.items():
        if 0 <= idx < n:
            x0[idx] = val

    result = minimize(
        objective,
        x0=x0,
        jac=grad,
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
        options={"ftol": 1e-12, "maxiter": 500},
    )

    return np.asarray(result.x, dtype=float)
