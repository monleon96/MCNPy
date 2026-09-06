"""Putting a drawn fission spectrum back on the constraint it must satisfy.

A perturbed spectrum has to stay a spectrum: every group probability
non-negative, and their sum unchanged. A linear draw from MF35 almost gets
there on its own, because ``C·1 ≈ 0`` — but only almost, by a measured
``sqrt(1ᵀC1)`` of up to 3.2e-5. This module closes that gap in closed form.

**Why the textbook simplex projection is the wrong tool, measured.** The
Euclidean projection onto ``{P ≥ 0, ΣP = 1}`` is ``P_i = max(y_i − θ, 0)`` for
one scalar θ. On Cf-252 the group probabilities span 1e-17 to 1e-1, so absorbing
a 3e-5 drift across ~122 groups gives θ ≈ 2.5e-7 — which **zeroes every group
below 2.5e-7**, roughly a third of the grid and the entire low-energy tail, to
repair a 0.003 % normalisation error. It is a correct projection of the wrong
thing: it treats a probability of 1e-17 and one of 1e-1 as equally movable in
absolute terms, when what is physically comparable between them is the
*fraction* by which each moves.

So the projection here is P⁰-weighted. It has the same closed form, the same
``O(n log n)`` cost and no optimiser, and the shift is uniform in the **ratio**
rather than in absolute probability. A group that is 1e-17 of the spectrum
moves by 1e-17 × t, not by t.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

__all__ = ["project_ratios_to_simplex", "check_ratios"]


def check_ratios(ratios: Sequence[float], p0: Sequence[float],
                 s_target: float) -> dict:
    """Report how far a ratio vector is from feasible, without changing it.

    The ``check_*``/``project_*`` split that :mod:`kika.sampling.mf4_positivity`
    uses: measure first, so the projection's effect is a difference between two
    measurements rather than an assertion about itself.
    """
    ratios = np.asarray(ratios, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    total = float(np.dot(p0, ratios))
    negative = ratios < 0.0
    return {
        "sum": total,
        "sum_error": total - float(s_target),
        "relative_sum_error": (
            (total - float(s_target)) / float(s_target) if s_target else 0.0
        ),
        "n_negative": int(negative.sum()),
        "negative_mass_fraction": (
            float(np.dot(p0[negative], -ratios[negative]) / s_target)
            if s_target and negative.any() else 0.0
        ),
        "min_ratio": float(ratios.min()) if ratios.size else 1.0,
        "max_ratio": float(ratios.max()) if ratios.size else 1.0,
        "is_feasible": bool(not negative.any()),
    }


def project_ratios_to_simplex(
    ratios: Sequence[float],
    p0: Sequence[float],
    s_target: float,
    *,
    metric: str = "probability",
) -> Tuple[np.ndarray, float, int]:
    """P⁰-weighted Euclidean projection onto ``{x ≥ 0, Σ P⁰x = S}``.

    Minimises ``Σ_j P⁰_j (x_j − r_j)²`` subject to ``Σ_j P⁰_j x_j = S`` and
    ``x ≥ 0``. The KKT conditions give ``x_j = max(r_j + t, 0)`` for a single
    scalar ``t`` — a uniform shift in the ratio. Sorting ``r`` descending and
    prefix-summing ``P⁰`` in that order makes ``t`` closed-form.

    Parameters
    ----------
    ratios
        ``r_j = 1 + δ_j / P⁰_j``, the drawn perturbation of each group.
    p0
        The unperturbed group probabilities. Groups with ``P⁰_j = 0`` carry no
        weight and are pinned to ``x_j = r_j``: they contribute nothing to the
        sum, so the constraint says nothing about them.
    s_target
        The sum to restore, ``Σ_j P⁰_j``, **as measured** — not 1. Part of the
        spectrum can lie outside MF35's outgoing coverage, and forcing the
        covered part to 1 would move mass that was never sampled.
    metric
        ``"probability"`` only. See below.

    Returns
    -------
    (x, t, n_clipped)
    """
    if metric != "probability":
        raise NotImplementedError(
            f"metric={metric!r} is not implemented. A variance-weighted shift "
            f"(x_j = r_j + t·σ_j²/P⁰_j, so groups the evaluator declared "
            f"certain barely move) is defensible, but it concentrates the "
            f"whole correction in a few groups when a band's variance is "
            f"concentrated, and it is undefined for a band whose variance is "
            f"zero everywhere. The parameter exists so adding it later is not "
            f"an API break."
        )

    r = np.asarray(ratios, dtype=float)
    weights = np.asarray(p0, dtype=float)
    if r.shape != weights.shape:
        raise ValueError(f"{r.size} ratios for {weights.size} probabilities")

    s_target = float(s_target)
    active = weights > 0.0
    if not np.any(active):
        return r.copy(), 0.0, 0

    r_active = r[active]
    w_active = weights[active]

    # Already feasible? Then t = 0 and x = r *exactly*, no arithmetic applied.
    # This is the common case — the drift is ~3e-5 — and passing the vector
    # through untouched is what keeps the projection from being a second,
    # invisible perturbation on every sample.
    current = float(np.dot(w_active, r_active))
    if r_active.min() >= 0.0 and current == s_target:
        return r.copy(), 0.0, 0

    # KKT: with x_j = max(r_j + t, 0), the j that stay positive are exactly the
    # largest r_j. Sort descending, and for each prefix ask what t would make
    # that prefix the active set and satisfy the constraint.
    order = np.argsort(r_active)[::-1]
    r_sorted = r_active[order]
    w_sorted = w_active[order]

    w_prefix = np.cumsum(w_sorted)
    rw_prefix = np.cumsum(r_sorted * w_sorted)

    # t_k solves  Σ_{j≤k} w_j (r_j + t) = S
    with np.errstate(divide="ignore", invalid="ignore"):
        t_candidates = (s_target - rw_prefix) / w_prefix

    # The valid prefix is the largest k for which every included r_j + t_k ≥ 0,
    # i.e. r_sorted[k] + t_k ≥ 0 (the smallest included one).
    feasible = (r_sorted + t_candidates) >= 0.0
    if np.any(feasible):
        k = int(np.max(np.nonzero(feasible)[0]))
        t = float(t_candidates[k])
    else:
        # Every prefix would clip its own last element. Only the largest ratio
        # survives, and t is set from it alone.
        k = 0
        t = float(t_candidates[0])

    x = r.copy()
    shifted = np.maximum(r_active + t, 0.0)
    x[active] = shifted
    n_clipped = int(np.sum(r_active + t < 0.0))

    return x, t, n_clipped
