"""The multigroup cross-section draw: repairs, then :func:`draw_samples`.

**Why this exists between the two.** :mod:`kika.sampling.core` carries no
repairs, deliberately — its docstring argues that a covariance's defects are
the evaluation's business and not the sampler's, and names PFNS as a caller for
which three of the repairs below are actively wrong. :func:`generate_samples`
carries six of them, hard-wired, in an order nothing states. This module is the
seam: it reproduces that order **exactly**, for the callers that need it, and
expresses it as a :class:`~kika.cov.conditioning.ConditioningPlan` a human can
read, edit, serialise and later refuse.

So the migration off ``generate_samples`` is not a port of the sampler. It is
this: the repairs become a plan, the plan is applied ahead of the draw, and
``draw_samples`` gets a matrix that needs nothing done to it.

**The sequence, and it is `generators.py`'s.** For the record, because the
whole value of this module is that the order is written down rather than
implied by line numbers:

1. ``log1p`` when drawing in log space — all
   :attr:`CrossSectionCovariance.log_covariance_matrix` ever did;
2. targeted rescale of threshold-spanning bins, when the caller knows the
   reaction thresholds (PENDF and ACE do, nu-bar does not);
3. statistical-outlier rescale, on what step 2 left;
4. the global variance cap;
5. drop the inert rows and decompose the reduced matrix;
6. draw, then pin the dropped rows to factor 1.0 and cast to float32.

Steps 2-4 are the plan. Step 5 is **not**, and that is a decision rather than
an oversight — see :func:`draw_relative_factors`. Step 6 is arithmetic the draw
and this module split between them.

**One matrix, not a block mapping.** :func:`draw_samples` offsets each block's
seed by its index, so a single block gets the caller's seed unchanged. Passing
one matrix makes that property structural instead of something a reader has to
verify by counting blocks.
"""
from __future__ import annotations

from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from kika.cov.conditioning import (
    ConditioningPlan,
    PlanStep,
    apply_plan,
    block_key_text,
    inert_row_mask,
)
from kika.cov.decomposition import (
    flag_outlier_variance_bins,
    flag_threshold_bins,
)
from kika.sampling.core import draw_samples

__all__ = [
    "CovarianceFixError",
    "SoftAutofixWarning",
    "OUTLIER_FACTOR",
    "apply_legacy_autofix",
    "draw_relative_factors",
    "pin_dropped_rows",
    "shipped_repair_plan",
]


#: Variance above this multiple of its per-MT median is an evaluator artefact.
#: The literal ``generate_samples`` passes to ``flag_outlier_variance_bins``.
OUTLIER_FACTOR = 1000.0


class CovarianceFixError(Exception):
    """Autofix could not bring the covariance above the eigenvalue threshold."""


class SoftAutofixWarning(Exception):
    """Soft autofix missed the threshold; the decomposition is tried anyway."""


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def shipped_repair_plan(
    matrix: np.ndarray,
    *,
    key: Hashable,
    pairs: Sequence[Tuple[int, int]],
    stride: int,
    bins: Sequence[float],
    space: str = "log",
    max_relative_std: Optional[float] = 3.0,
    mt_thresholds: Optional[Dict[int, float]] = None,
    outlier_factor: float = OUTLIER_FACTOR,
    verbose: bool = False,
    logger=None,
    label: str = "",
) -> ConditioningPlan:
    """The repairs ``generate_samples`` makes, as a plan, in its order.

    *matrix* must already be in the space it will be drawn in — this reads its
    diagonal to decide what to flag, and the flags differ between a relative
    covariance and its ``log1p``.

    **It derives; it does not apply.** The second flagger has to see what the
    first rescale produced (``generators.py`` reassigns ``cov_mat`` between
    them), so this runs the appliers on interim copies and throws them away.
    The matrix that comes back from :func:`apply_plan` is then the only one
    anything downstream sees, and the plan written to the run directory is the
    plan that ran — which is the whole point of having one.

    Not in the plan: the PSD projection, which stays inside the decomposition
    where both draws already call the same function. Hoisting it would run
    ``psd_method="auto"``'s threshold decision on the full matrix rather than
    on the inert-reduced one, and that is a different matrix and possibly a
    different remedy.
    """
    from kika.cov.decomposition import rescale_threshold_bins_congruence

    steps: List[PlanStep] = []
    notes: List[str] = []
    working = matrix
    n_pairs = len(pairs)
    threshold_flagged: List[int] = []

    ctx = f" [{label}]" if label else ""
    log_kwargs = dict(
        param_pairs=pairs, num_groups=stride, bins=bins,
        verbose=verbose, logger=logger, label=label,
    )

    def _metric(name: str, value: int) -> None:
        """The greppable summary lines `generate_samples` writes, kept."""
        if logger is not None:
            logger.info(f">> {name}{ctx} = {value}")

    # 2a — threshold-spanning bins, when the caller knows the thresholds.
    if mt_thresholds and stride > 0 and n_pairs > 0:
        flagged, targets, _detection = flag_threshold_bins(
            working, mt_thresholds, pairs, stride, bins,
            space=space, verbose=verbose, logger=logger, label=label,
        )
        threshold_flagged = list(flagged)
        rescaled = 0
        if flagged:
            steps.append(PlanStep(
                key=key,
                remedy="rescale_to_family_median",
                parameters={"indices": [int(i) for i in flagged],
                            "targets": [float(t) for t in targets]},
                reason=(
                    f"{len(flagged)} bin(s) straddle a reaction threshold; "
                    "the below-threshold tail pulls <sigma> down and inflates "
                    "the relative variance"
                ),
            ))
            working, info = rescale_threshold_bins_congruence(
                working, flagged, targets, **log_kwargs,
            )
            rescaled = int(info["n_actually_rescaled"])
        _metric("threshold_bins_flagged", len(threshold_flagged))
        _metric("threshold_bins_rescaled", rescaled)

    # 2a-bis — statistical outliers, on what the threshold rescale left.
    if stride > 0 and n_pairs > 0:
        flagged, targets, _detection = flag_outlier_variance_bins(
            working, pairs, stride, bins,
            outlier_factor=outlier_factor,
            skip_indices=threshold_flagged,
            space=space, verbose=verbose, logger=logger, label=label,
        )
        rescaled = 0
        if flagged:
            steps.append(PlanStep(
                key=key,
                remedy="rescale_to_family_median",
                parameters={"indices": [int(i) for i in flagged],
                            "targets": [float(t) for t in targets]},
                reason=(
                    f"{len(flagged)} bin(s) exceed {outlier_factor:g}x their "
                    "per-MT median variance — an evaluator placeholder rather "
                    "than heterogeneity"
                ),
            ))
            working, info = rescale_threshold_bins_congruence(
                working, flagged, targets, **log_kwargs,
            )
            rescaled = int(info["n_actually_rescaled"])
        _metric("outlier_bins_flagged", len(flagged))
        _metric("outlier_bins_rescaled", rescaled)

    # 2b — the global cap.
    if max_relative_std is not None and max_relative_std > 0:
        ceiling = (float(np.log(1.0 + max_relative_std ** 2)) if space == "log"
                   else float(max_relative_std ** 2))
        steps.append(PlanStep(
            key=key,
            remedy="cap",
            parameters={"max_variance": ceiling},
            reason=(
                f"defence in depth behind the rescales: nothing is sampled "
                f"above {max_relative_std:g} relative sigma"
            ),
        ))

    notes.append(
        "the PSD projection is not a step here: it stays inside the "
        "decomposition, where it acts on the inert-reduced matrix rather than "
        "on this one"
    )
    notes.append(
        "the inert rows are dropped before decomposing and pinned to factor "
        "1.0 after; that changes the dimension, so it cannot be a plan step "
        "and is done by draw_relative_factors"
    )
    return ConditioningPlan(steps=tuple(steps), notes=tuple(notes))


# ---------------------------------------------------------------------------
# The draw
# ---------------------------------------------------------------------------

def pin_dropped_rows(
    reduced: np.ndarray, inert: np.ndarray, *, fill: float = 1.0
) -> np.ndarray:
    """Re-expand *reduced* over the full width, pinning the inert columns.

    A bin the evaluation states no variance for samples to the same value every
    draw, which for a multiplicative factor is 1.0. Dropping it before the
    decomposition and writing it back here is exactly equivalent to leaving it
    in — and it takes the row out of the QMC dimension, which is the reason to
    bother.
    """
    n = int(inert.size)
    out = np.ones((reduced.shape[0], n), dtype=reduced.dtype)
    if fill != 1.0:
        out.fill(fill)
    out[:, ~inert] = reduced
    return out


def draw_relative_factors(
    matrix: np.ndarray,
    n_samples: int,
    *,
    key: Hashable = "MF33",
    pairs: Sequence[Tuple[int, int]],
    stride: int,
    bins: Sequence[float],
    space: str = "log",
    decomposition_method: str = "svd",
    sampling_method: str = "sobol",
    seed: Optional[int] = None,
    psd_method: str = "auto",
    max_relative_std: Optional[float] = 3.0,
    mt_thresholds: Optional[Dict[int, float]] = None,
    plan: Optional[ConditioningPlan] = None,
    null_tol: Optional[float] = None,
    dtype=np.float32,
    verbose: bool = True,
    logger=None,
    label: str = "",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Multiplicative perturbation factors for one multigroup covariance.

    *matrix* is the **relative** covariance, linear, laid out as
    ``len(pairs)`` consecutive blocks of *stride* rows in the order *pairs*
    gives. *bins* is the energy grid, ``stride + 1`` long; both it and *pairs*
    exist so the flaggers can say which reaction and which energy a repair
    touched, and neither changes any arithmetic.

    Returns ``(factors, info)`` with ``factors`` of shape
    ``(n_samples, len(pairs) * stride)``. ``info`` carries the ``plan`` that
    ran, the ``applied`` log, ``n_inert_dropped`` and the draw's diagnostics —
    everything a run directory needs to say what was done to the covariance
    before it was sampled.

    **``null_tol=None`` is the default here and it is not the better draw.**
    Truncating to the retained rank is what :func:`draw_samples` does by
    default and what stops null-direction debris reaching a written file. It is
    held off because turning it on changes the QMC dimension and so moves every
    drawn column, and a migration that moves the numbers cannot be checked
    against the path it replaces. It goes on in its own commit, with MF34, once
    the equivalence has been confirmed.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square covariance, got {matrix.shape}")

    expected = len(pairs) * int(stride)
    if expected and matrix.shape[0] != expected:
        raise ValueError(
            f"{matrix.shape[0]}x{matrix.shape[0]} covariance does not match "
            f"{len(pairs)} pair(s) of stride {stride} (= {expected} rows). The "
            "flaggers slice by pair_index * stride, so a mismatch here would "
            "silently attribute one reaction's bins to another rather than fail"
        )
    if space not in ("linear", "log"):
        raise ValueError(f"space must be 'linear' or 'log', got {space!r}")

    # 1 — the space the draw happens in.
    if space == "log":
        matrix = np.log1p(matrix)

    # 2-4 — the repairs, as a plan.
    if plan is None:
        plan = shipped_repair_plan(
            matrix, key=key, pairs=pairs, stride=stride, bins=bins, space=space,
            max_relative_std=max_relative_std, mt_thresholds=mt_thresholds,
            verbose=verbose, logger=logger, label=label,
        )
    conditioned, applied = apply_plan(
        [(key, matrix)], plan, verbose=verbose, logger=logger,
    )
    repaired = conditioned[0][1]

    ctx = f" [{label}]" if label else ""

    # 5 — drop the inert rows.
    inert = inert_row_mask(repaired)
    n_inert = int(inert.sum())
    if n_inert:
        if n_inert == inert.size:
            raise ValueError(
                f"every row of {block_key_text(key)} states zero variance; "
                "there is nothing to sample"
            )
        if verbose:
            message = (
                f"  [INFO] [INERT-BIN MASK]{ctx} Dropped {n_inert}/{inert.size} "
                f"bin(s) with no stated variance; decomposition runs on "
                f"{inert.size - n_inert}-dim reduced matrix"
            )
            if logger is not None:
                logger.info(message)
                logger.info(f">> inert_bins_dropped{ctx} = {n_inert}")
            else:
                print(message)
        reduced = repaired[np.ix_(~inert, ~inert)]
    else:
        reduced = repaired

    # 6 — the draw, and back to full width.
    samples, diagnostics = draw_samples(
        {key: reduced},
        n_samples,
        space=space,
        returns="factors",
        decomposition_method=decomposition_method,
        sampling_method=sampling_method,
        seed=seed,
        psd_method=psd_method,
        null_tol=null_tol,
        dtype=np.float64,
        verbose=verbose,
        logger=logger,
    )
    factors = pin_dropped_rows(samples[key], inert).astype(dtype)

    return factors, {
        "plan": plan,
        "applied": applied,
        "n_inert_dropped": n_inert,
        "inert": inert,
        "diagnostics": diagnostics[key],
    }


# ---------------------------------------------------------------------------
# autofix, kept alive for kika-app
# ---------------------------------------------------------------------------

def apply_legacy_autofix(
    cov,
    level: Optional[str],
    *,
    mt_numbers: Optional[Sequence[int]] = None,
    high_val_thresh: float = 5.0,
    accept_tol: float = -1.0e-4,
    verbose: bool = True,
    logger=None,
):
    """``generate_samples``' autofix, lifted so the sampler can be retired.

    Returns ``(cov, mt_numbers, fix_info)`` and is a no-op returning the inputs
    when *level* is ``None``, which is what every configuration inside kika
    passes.

    **It is kept because kika-app does not.** ``kika-api``'s sigma pipelines
    default ``autofix='soft'`` and render it into every generated script, so
    dropping it here would be a behaviour change for a live consumer riding on
    a migration. Retiring it is its own change, in that repository first.

    ⚠ **Under D11 this measures nothing on an incomplete covariance.** The
    carrier states an unstated cross block as NaN; every eigenvalue solver
    fails on such a matrix and the fallback returns ``0.0``, which clears any
    negative ``accept_tol``. On the full Fe-56 tape that is 85.7 % of the
    matrix and 51 s of solvers failing. Measured 2026-08-12: the honest
    ``lambda_min`` is -3.8e-05 there, which clears the shipped tolerance too —
    so the verdict happens to be right and the measurement is still not being
    taken.
    """
    if level is None:
        return cov, (list(mt_numbers) if mt_numbers is not None else None), None

    cov_fixed, fix_log = cov.fix_covariance(
        level=level,
        high_val_thresh=high_val_thresh,
        accept_tol=accept_tol,
        verbose=verbose,
        logger=logger,
    )

    if not fix_log.get("converged", False):
        min_eigenvalue = fix_log.get("min_eigenvalue", float("nan"))
        soft_missed = (
            level.lower() == "soft" and not fix_log.get("soft_threshold_met", True)
        )
        if soft_missed:
            if logger:
                logger.info(
                    f"[COVARIANCE] [SOFT AUTOFIX] Threshold not met "
                    f"(lambda_min={min_eigenvalue:.4e} < {accept_tol:.4e}), "
                    "attempting decomposition anyway"
                )
        else:
            error_msg = (
                "Covariance matrix could not be fixed to meet eigenvalue "
                f"threshold.\n  Final minimum eigenvalue: {min_eigenvalue:.4e}\n"
                f"  Required threshold: {accept_tol:.4e}\n"
                f"  Autofix level used: {level}"
            )
            if logger:
                logger.error(f"[COVARIANCE] [ERROR] {error_msg}")
            else:
                print(f"[ERROR] {error_msg}")
            raise CovarianceFixError(
                f"min_eigenvalue={min_eigenvalue:.4e} below threshold={accept_tol:.4e}"
            )
    elif verbose and logger:
        logger.info(
            "[COVARIANCE] [SUCCESS] Matrix successfully fixed "
            f"(final lambda_min={fix_log.get('min_eigenvalue', float('nan')):.4e})"
        )

    surviving = list(mt_numbers) if mt_numbers is not None else None
    if surviving is not None and fix_log.get("removed_pairs"):
        removed: set = set()
        if level.lower() == "medium":
            for ra, rb in fix_log.get("removed_pairs", []):
                if ra == rb:  # a diagonal block means the whole reaction went
                    removed.add(ra)
        elif level.lower() == "hard":
            removed.update(fix_log.get("removal_log", {}).get("removed_mts", []))
        if removed:
            message = f"  [INFO] MTs removed by fix_covariance: {sorted(removed)}"
            if verbose:
                logger.info(message) if logger else print(message)
            surviving = [mt for mt in surviving if mt not in removed]

    return cov_fixed, surviving, fix_log
