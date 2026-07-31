"""Information-criterion model averaging over Legendre order — pure numerics.

Phase 1 of ``docs/mf4_research_roadmap.md`` §5. Deliberately free of file I/O,
global configuration and pipeline imports, so it can be tested in isolation and
reused by whichever driver ends up consuming it.

The problem
-----------
For each energy bin the fit scans candidate Legendre degrees and keeps the
information-criterion score of each. Production then takes the **winner's**
coefficient vector as the nominal and throws the rest away. Measured on run 82,
the winner carries a median of only 0.505 of the IC weight, with ~3.25
effectively-supported models per bin — so winner-take-all discards real support.

This module implements the alternative: treat the candidate degrees as a
mixture, weighted by their Akaike weights.

Zero padding is a modelling statement, not bookkeeping
------------------------------------------------------
A degree-3 model embedded in a six-order space contributes an **exact zero** for
``a_4..a_6``. That zero is a prediction ("this order is absent"), not a missing
value. Because the Legendre representation is linear, the weighted mean
coefficient is then the exact coefficient of the weighted-average angular
distribution — see ``mixture_moments`` and the angular-equivalence test.

The consequence is shrinkage toward zero, and it is intended. With inclusion
probability ``p``, conditional mean ``m`` and conditional variance ``s²``:

    E[a] = p·m            Var(a) = p·s² + p(1-p)·m²

The mean shrinks; the variance does not. A large *relative* uncertainty at a
near-zero mean is therefore a diagnostic state, not a covariance failure —
``p=0.2, m=0.02, s=0.01`` gives a relative sigma of ~230 % with nothing wrong.
``conditional_mean`` divides the shrinkage out, but it silently conditions on
the order being present; keep it as a diagnostic, never as the shipped central.

Conventions
-----------
Vectors are ordered ``[a_1, a_2, ..., a_{l_max}]``. Order 0 is excluded on
purpose: under the ENDF normalisation ``a_0 ≡ 1`` is not a free parameter, so
averaging it is meaningless. Callers working in unnormalised ``c``-space must
say so themselves — this module never normalises, because which space the
averaging happens in is a modelling choice the roadmap wants quantified, not a
default to be buried here.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

__all__ = [
    "ic_weights",
    "effective_n_models",
    "pad_to",
    "stack_padded",
    "nested_masks",
    "inclusion_probabilities",
    "mixture_moments",
    "conditional_mean",
]


def ic_weights(scores: Sequence[float], *, floor: float = 0.0) -> np.ndarray:
    """Akaike weights ``w_L ∝ exp(-Δ_L/2)`` from information-criterion scores.

    Computed by shifting to the minimum before exponentiating — the standard
    log-sum-exp guard. Without it, scores of a few hundred (routine for a χ²
    based IC) overflow ``exp`` and silently produce NaN weights.

    Non-finite scores are treated as candidates that failed to fit and get zero
    weight. That is deliberate: a NaN score reaching ``exp`` would poison the
    whole vector, and the pipeline can produce one when a degree is
    rank-deficient in a bin.

    Parameters
    ----------
    scores
        IC score per candidate, lower is better. ``chi2 + 2k`` for AIC.
    floor
        Drop candidates whose weight falls below this and renormalise. The
        production scan applies an arbitrary 1 % cutoff (roadmap §4.1); the
        default here is 0.0 — keep everything — and any cutoff must be an
        explicit choice at the call site.

    Returns
    -------
    Weights summing to 1. All-zero only if every score is non-finite.
    """
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {s.shape}")
    if s.size == 0:
        return np.zeros(0, dtype=float)

    ok = np.isfinite(s)
    w = np.zeros_like(s)
    if not ok.any():
        return w

    delta = s[ok] - s[ok].min()
    raw = np.exp(-0.5 * delta)
    w[ok] = raw / raw.sum()

    if floor > 0.0:
        w = np.where(w >= floor, w, 0.0)
        total = w.sum()
        if total > 0:
            w = w / total
    return w


def effective_n_models(weights: Sequence[float]) -> float:
    """``exp(H)`` of the weight vector — how many models the evidence supports.

    1.0 means one model carries everything; 3.0 means the support is spread as
    if over three. Reported per bin in the Gate 2 analysis.
    """
    w = np.asarray(weights, dtype=float)
    nz = w[w > 0]
    if nz.size == 0:
        return 0.0
    return float(np.exp(-np.sum(nz * np.log(nz))))


def pad_to(coeffs: Sequence[float], n_out: int) -> np.ndarray:
    """Zero-pad (or validate) a coefficient vector to length ``n_out``.

    Padding with zeros is the model statement described in the module
    docstring. Truncation is refused — silently dropping a fitted coefficient
    would be a different model than the one that was scored.
    """
    c = np.asarray(coeffs, dtype=float)
    if c.ndim != 1:
        raise ValueError(f"coeffs must be 1-D, got shape {c.shape}")
    if c.size > n_out:
        raise ValueError(
            f"cannot pad a length-{c.size} vector to {n_out}: that would "
            "truncate fitted coefficients, which is a different model"
        )
    out = np.zeros(n_out, dtype=float)
    out[: c.size] = c
    return out


def stack_padded(candidates: Sequence[Sequence[float]], n_out: int) -> np.ndarray:
    """``(n_models, n_out)`` array of zero-padded candidate coefficient vectors."""
    if len(candidates) == 0:
        return np.zeros((0, n_out), dtype=float)
    return np.vstack([pad_to(c, n_out) for c in candidates])


def nested_masks(degrees: Sequence[int], n_out: int) -> np.ndarray:
    """Boolean ``(n_models, n_out)``: which orders each candidate actually has.

    For the nested Legendre family a degree-``L`` model carries orders 1..L, so
    ``mask[i, l-1] = (l <= L_i)``. Kept separate from the coefficients so a
    non-nested candidate set can supply its own mask.
    """
    d = np.asarray(degrees, dtype=int)
    orders = np.arange(1, n_out + 1)[None, :]
    return orders <= d[:, None]


def inclusion_probabilities(
    weights: Sequence[float],
    masks: np.ndarray,
) -> np.ndarray:
    """``q_l`` — total IC weight of the candidates that include order ``l``.

    This is the ``p`` of the ``E[a] = p·m`` reframing. For a nested family it is
    the tail sum ``q_l = Σ_{L≥l} w_L``.
    """
    w = np.asarray(weights, dtype=float)
    m = np.asarray(masks, dtype=bool)
    if m.ndim != 2 or m.shape[0] != w.size:
        raise ValueError(
            f"masks must be (n_models, n_out) matching {w.size} weights, "
            f"got shape {m.shape}"
        )
    return w @ m.astype(float)


def mixture_moments(
    weights: Sequence[float],
    means: np.ndarray,
    covariances: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Law of total covariance over the candidate mixture.

    .. math::
        \\bar{a} = \\sum_L w_L a_L
        \\qquad
        V = \\underbrace{\\sum_L w_L V_L}_{\\text{within}}
          + \\underbrace{\\sum_L w_L (a_L-\\bar a)(a_L-\\bar a)^T}_{\\text{between}}

    This is the exact covariance of the zero-padded mixture that the current
    discrete degree-sampling MC is trying to estimate, computed analytically.
    That is the point: sampling estimates the same quantity but puts only a
    handful of draws on the high orders, so ``a_5`` and ``a_6`` are reconstructed
    from a few dozen of 10,000 samples. Here they are exact.

    Parameters
    ----------
    weights
        Length ``n_models``, need not be normalised — they are renormalised
        here, and a zero-sum raises rather than dividing by zero.
    means
        ``(n_models, n_out)`` zero-padded candidate coefficient vectors.
    covariances
        ``(n_models, n_out, n_out)`` within-model covariances, or None for
        between-model spread only (useful when only the central value is
        wanted, and honest about contributing no within-model term).

    Returns
    -------
    dict with ``mean``, ``within``, ``between``, ``total``. ``total`` is
    symmetrised; it is PSD whenever every input covariance is.
    """
    w = np.asarray(weights, dtype=float)
    mu = np.asarray(means, dtype=float)
    if mu.ndim != 2:
        raise ValueError(f"means must be 2-D (n_models, n_out), got {mu.shape}")
    if w.size != mu.shape[0]:
        raise ValueError(
            f"{w.size} weights against {mu.shape[0]} candidate means"
        )
    total_w = w.sum()
    if not np.isfinite(total_w) or total_w <= 0:
        raise ValueError("weights must contain at least one positive finite entry")
    w = w / total_w

    n_out = mu.shape[1]
    mean = w @ mu

    dev = mu - mean[None, :]
    between = np.einsum("i,ij,ik->jk", w, dev, dev)

    if covariances is None:
        within = np.zeros((n_out, n_out), dtype=float)
    else:
        cov = np.asarray(covariances, dtype=float)
        if cov.shape != (w.size, n_out, n_out):
            raise ValueError(
                f"covariances must be {(w.size, n_out, n_out)}, got {cov.shape}"
            )
        within = np.einsum("i,ijk->jk", w, cov)

    within = 0.5 * (within + within.T)
    between = 0.5 * (between + between.T)
    total = within + between
    return {
        "mean": mean,
        "within": within,
        "between": between,
        "total": 0.5 * (total + total.T),
    }


def conditional_mean(
    weights: Sequence[float],
    means: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    """Mean of order ``l`` over only the candidates that include it — DIAGNOSTIC.

    ``m_l = (Σ_{L∋l} w_L a_{lL}) / q_l``, i.e. the zero-padded mean divided by
    the inclusion probability.

    Not a candidate for the shipped central value. It answers "what is ``a_5``
    given that order 5 is certainly present", which conditions on a fact the
    evidence does not establish — that is exactly the question the averaging is
    supposed to stop asking. Its use is comparing against ``mixture_moments``'s
    mean to see how much of a small coefficient is shrinkage and how much is a
    genuinely small conditional value.

    Orders with zero inclusion probability come back as 0.0, not NaN, since no
    candidate offers an opinion about them.
    """
    w = np.asarray(weights, dtype=float)
    mu = np.asarray(means, dtype=float)
    m = np.asarray(masks, dtype=bool)
    total_w = w.sum()
    if not np.isfinite(total_w) or total_w <= 0:
        raise ValueError("weights must contain at least one positive finite entry")
    w = w / total_w

    q = inclusion_probabilities(w, m)
    numer = w @ (mu * m)
    return np.divide(numer, q, out=np.zeros_like(numer), where=q > 0)
