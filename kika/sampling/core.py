"""The sampler, with the formatting taken out of it.

``draw_samples`` decomposes a covariance, draws uncorrelated normals and imposes
the correlation. That is all it does. It knows nothing about MT numbers, energy
groups, Legendre orders, incident-energy bands, ENDF, ACE or NJOY, and it holds
no covariance class — a block is a ``(key, matrix)`` pair and the key can be
anything hashable.

**Why this exists beside** :func:`kika.sampling.generators.generate_samples`.
That function is ~400 lines, of which about fifteen are the sampling. The rest
is MT/group bookkeeping, four heuristics that only make sense for a *relative*
covariance, sample diagnostics, and a cast to float32. It is a format-aware
pipeline with a sampler inside it. Every perturbation pipeline in this library
needs the sampler; only some of them want the pipeline.

So the sampler is extracted rather than copied. PFNS is the first caller, and it
is a good first caller precisely because the parts it does *not* want are not
merely off by default for it — they are wrong for it:

* the **inert-bin mask** drops rows before decomposition, which breaks
  ``C·1 = 0`` on the retained subspace; that identity is the whole reason a
  linear draw preserves the normalisation of a spectrum.
* the **statistical-outlier rescale** flags any bin whose variance exceeds
  1000× the median. A fission spectrum's variance spans ~30 decades across the
  outgoing grid *as genuine structure*, so it would fire on correct data.
* the **float32 cast** annihilates absolute deltas: they run many decades below
  float32 epsilon, and stored as ``1 + δ`` all but the largest vanish. Silently
  — the run completes, writes tapes, passes NJOY, and contains no perturbation.

A flag that must always be set one way is a fork wearing a parameter, so those
steps are not given flags here. They live in :mod:`kika.cov.decomposition` as
standalone functions and a caller that wants them applies them to the matrix
before handing it over.

**Nothing existing was migrated onto this.** ``generate_samples`` is untouched,
so no result any current pipeline produces changes. ``returns="factors"`` is
here so that the migration, when it happens, is a call rather than a rewrite.

**Blocks are independent, and seeded independently.** MF35 gives no cross-band
covariance, MF33 gives no cross-material one; a block-diagonal covariance is
sampled block by block. That is also much cheaper — five band-wise SVDs on
ENDF/B-VIII.1 U-235 cost ~1/125 of one 3205×3205 decomposition. The per-block
seed offset is what stops every block drawing the same ``Z``, which is the
quiet failure this arrangement invites: the samples look fine block by block
and are perfectly correlated across blocks.
"""
from __future__ import annotations

from typing import Any, Dict, Hashable, Optional, Tuple

import numpy as np

from ..cov.conditioning import as_blocks
from ..cov.decomposition import eigen_decomposition, svd_decomposition
# The QMC draw, shared rather than copied. It belongs here and will move when
# ``generate_samples`` is migrated onto this module; today the dependency runs
# core → generators and generators does not import core, so there is no cycle.
from .generators import _uncorrelated

__all__ = ["draw_samples", "BLOCK_SEED_STRIDE"]

#: Added to the base seed once per block index. A prime, so that two callers
#: whose seeds differ by a small integer do not end up sharing draws on some
#: pair of blocks. Recorded in the diagnostics of every block, because a
#: reproduction that cannot say which seed produced which block is not one.
BLOCK_SEED_STRIDE = 1009

#: Singular values at or below ``null_tol × S.max()`` count as null directions.
DEFAULT_NULL_TOL = 1e-10


#: What a caller may hand in as *blocks*, resolved to ``(key, matrix)`` pairs.
#:
#: Defined in :mod:`kika.cov.conditioning` rather than here so that inspecting
#: a set of blocks and sampling it agree on what "a set of blocks" is by
#: construction. Two copies of this would drift the day one of them learned a
#: fourth input shape, and the failure would be that a pre-flight passed on a
#: different arrangement of matrices than the one that got sampled.
_as_blocks = as_blocks


def draw_samples(
    blocks,
    n_samples: int,
    *,
    space: str = "linear",
    returns: str = "deltas",
    decomposition_method: str = "svd",
    sampling_method: str = "sobol",
    seed: Optional[int] = None,
    psd_method: str = "auto",
    null_tol: Optional[float] = DEFAULT_NULL_TOL,
    dtype=np.float64,
    verbose: bool = True,
    logger=None,
) -> Tuple[Dict[Hashable, np.ndarray], Dict[Hashable, Dict[str, Any]]]:
    """Draw *n_samples* correlated realisations of each covariance block.

    Parameters
    ----------
    blocks
        ``{key: matrix}``, a sequence of ``(key, matrix)`` pairs, or a bare
        sequence of matrices (then the key is the index). The key is opaque —
        ``(isotope, mt)`` for a cross section, ``(isotope, mt, order)`` for a
        Legendre block, ``(isotope, mt, band)`` for a PFNS band.
    n_samples
        Number of realisations per block.
    space
        ``"linear"`` draws ``Y = Z L^T`` from the matrix as given.
        ``"log"`` treats the matrix as the covariance of ``log`` of the
        quantity and moment-matches so the factors have unit mean. Only
        meaningful together with ``returns="factors"``.
    returns
        ``"deltas"`` returns ``Y`` — absolute departures, which is what an
        absolute covariance describes. ``"factors"`` returns ``Y + 1`` (linear)
        or ``exp(Y + m)`` (log), which is what a relative covariance describes
        and what the ENDF/ACE perturbation drivers multiply by.
    decomposition_method
        ``"svd"`` or ``"eigen"``. **Cholesky is refused**: these matrices are
        built from a handful of model parameters and are severely rank
        deficient by construction — two thirds of a typical MF35 band is null
        space — so a Cholesky factor either fails or is meaningless.
    seed
        Base seed. Block *i* uses ``seed + i × BLOCK_SEED_STRIDE``.
    psd_method
        Passed through to the decomposition. ``"auto"`` routes the tiny
        negative eigenvalues these matrices carry to ``clip``, which rebuilds
        ``V·clip(Λ,0)·Vᵀ`` and so **preserves the eigenvectors** — including
        the near-null direction that carries a sum rule.
    null_tol
        Threshold for counting null directions. They are counted and reported,
        never repaired: the rank deficiency is a property of the evaluation,
        and filling it in would invent uncertainty the evaluator did not claim.

        ``None`` retains **every** direction, including the null ones, and is
        there for one purpose: proving that a migrated call site draws what the
        code it replaces drew. Truncation changes the QMC dimension from *n* to
        the rank, so it moves every drawn column — on the Fe-56 multigroup MF34
        joint, 4218 → 3603 — and a migration that turns it on in the same commit
        that changes which objects the draw comes from cannot say which of the
        two moved the numbers. Set it to ``None`` to establish byte-identity,
        then let it default in a separate, measured commit.

        It is an escape hatch for gates, not a modelling option. The diagnostics
        still report ``n_null`` and ``rank`` at :data:`DEFAULT_NULL_TOL`, so a
        run drawing in null directions says so rather than reporting full rank.

        **The rank reported here and the rank a pre-flight reports are not the
        same number, and neither is wrong.** This one is taken on the singular
        values the draw actually decomposes — *after* ``psd_method`` has acted,
        and SVD folds a negative eigenvalue to its magnitude. A direction the
        file states as negative-variance therefore survives as a retained
        direction here while ``conditioning.inspect_blocks``, which reads the
        eigenvalues of the matrix as stated, counts it null. Measured on that
        same Fe-56 joint: 615 null directions in the draw against 669 in the
        pre-flight, the difference being directions the evaluation states as
        unphysical.
    dtype
        Output dtype. float64 by default and on purpose; see the module
        docstring.

    Returns
    -------
    samples, diagnostics
        ``{key: (n_samples, m_key) array}`` and ``{key: {...}}``. The
        diagnostics carry ``n_null``, ``rank``, ``seed``, ``psd_method``,
        the extreme eigenvalue ratio and the realised-versus-input covariance
        error on the retained subspace.
    """
    if returns not in ("deltas", "factors"):
        raise ValueError(f"returns must be 'deltas' or 'factors', got {returns!r}")
    if space not in ("linear", "log"):
        raise ValueError(f"space must be 'linear' or 'log', got {space!r}")
    if space == "log" and returns == "deltas":
        raise ValueError(
            "space='log' with returns='deltas' is not a thing: a log-space draw "
            "describes multiplicative factors, so ask for returns='factors'"
        )

    method = decomposition_method.lower()
    if method == "cholesky":
        raise ValueError(
            "Cholesky is refused here. These covariances are rank deficient by "
            "construction — an MF35 band typically has two thirds of its "
            "directions null — so a Cholesky factor is either a failure or a "
            "fiction. Use 'svd' (default) or 'eigen'."
        )
    if method not in ("svd", "eigen"):
        raise ValueError(
            f"decomposition_method must be 'svd' or 'eigen', got "
            f"{decomposition_method!r}"
        )

    items = _as_blocks(blocks)
    samples: Dict[Hashable, np.ndarray] = {}
    diagnostics: Dict[Hashable, Dict[str, Any]] = {}

    for index, (key, matrix) in enumerate(items):
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"block {key!r}: a covariance is a square 2-D matrix, got shape "
                f"{matrix.shape}"
            )

        block_seed = None if seed is None else int(seed) + index * BLOCK_SEED_STRIDE
        draw, info = _draw_one_block(
            matrix, n_samples,
            space=space, returns=returns, method=method,
            sampling_method=sampling_method, seed=block_seed,
            psd_method=psd_method, null_tol=null_tol,
            verbose=verbose, logger=logger, label=str(key),
        )
        samples[key] = draw.astype(dtype, copy=False)
        info["seed"] = block_seed
        info["block_index"] = index
        diagnostics[key] = info

    return samples, diagnostics


def _draw_one_block(
    matrix: np.ndarray,
    n_samples: int,
    *,
    space: str,
    returns: str,
    method: str,
    sampling_method: str,
    seed: Optional[int],
    psd_method: str,
    null_tol: Optional[float],
    verbose: bool,
    logger,
    label: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    n = matrix.shape[0]

    finite = np.isfinite(matrix)
    n_non_finite = int((~finite).sum())
    if n_non_finite:
        matrix = np.where(finite, matrix, 0.0)

    if method == "svd":
        basis, spectrum, _ = svd_decomposition(
            space=space, psd_method=psd_method, verbose=verbose,
            logger=logger, matrix=matrix,
        )
        spectrum = np.asarray(spectrum, dtype=float)
    else:
        values, vectors = eigen_decomposition(
            space=space, psd_method=psd_method, verbose=verbose,
            logger=logger, matrix=matrix,
        )
        order = np.argsort(np.asarray(values, dtype=float))[::-1]
        spectrum = np.asarray(values, dtype=float)[order]
        basis = vectors[:, order]

    largest = float(spectrum.max()) if spectrum.size else 0.0
    counting_tol = DEFAULT_NULL_TOL if null_tol is None else null_tol
    null = (spectrum <= counting_tol * largest if largest > 0
            else np.ones(n, bool))
    n_null = int(null.sum())

    # ``null_tol=None`` draws in every direction, null ones included. The count
    # above is still taken at the default tolerance, so the diagnostics report
    # what is true of the matrix rather than what this draw chose to use.
    keep = np.ones(n, bool) if null_tol is None else ~null

    # **Truncate to the retained rank rather than drawing in all n directions.**
    #
    # Not an optimisation — a correctness fix, and the subtlest thing in this
    # module. A null direction has λ ≈ 1e-10·λmax, so its factor column carries
    # an amplitude ≈ 1e-5 of the leading one: numerical debris from the
    # decomposition, not uncertainty the evaluation claims. Keeping those
    # columns adds an absolute delta of ~1e-10 to *every* component, including
    # ones whose stated σ is exactly zero.
    #
    # In absolute terms that is nothing. But PFNS divides each delta by its
    # group probability, and a fission spectrum's lowest groups hold ~1e-17 of
    # the total — so 1e-10 of debris becomes a perturbation ratio of 1e+7, and
    # the written spectrum grows a spike in its low-energy tail that MF35 never
    # asked for. Measured on Cf-252 before this truncation: group 1 has
    # σ/P⁰ = 0.12 and was being perturbed by a factor of 2.6e+7.
    #
    # Truncating makes the sampled subspace exactly the one the covariance
    # spans, so a component with zero stated variance gets zero delta
    # identically — every retained eigenvector has a zero there. It also drops
    # the QMC dimension from n to the rank, which is where Sobol is better
    # behaved anyway.
    factor = basis[:, keep] @ np.diag(np.sqrt(np.clip(spectrum[keep], 0.0, None)))

    z = _uncorrelated(dim=int(keep.sum()), n=n_samples,
                      method=sampling_method, seed=seed)
    y = z @ factor.T

    if returns == "deltas":
        drawn = y
    elif space == "linear":
        drawn = y + 1.0
    else:
        drawn = np.exp(y - 0.5 * np.diag(matrix))

    raw_eigenvalues = np.linalg.eigvalsh(matrix)
    lam_max = float(raw_eigenvalues.max()) if raw_eigenvalues.size else 0.0

    return drawn, {
        "n": n,
        "n_null": n_null,
        "rank": n - n_null,
        "null_fraction": (n_null / n) if n else 0.0,
        "n_non_finite_zeroed": n_non_finite,
        "psd_method": psd_method,
        "decomposition_method": method,
        "space": space,
        "returns": returns,
        "min_over_max_eigenvalue": (
            float(raw_eigenvalues.min() / lam_max) if lam_max > 0 else 0.0
        ),
        "realised_covariance_error": _realised_error(y, matrix, counting_tol),
        "label": label,
    }


def _realised_error(y: np.ndarray, matrix: np.ndarray, null_tol: float) -> float:
    """Relative error of the realised covariance, on the non-null subspace only.

    Comparing ``cov(Y)`` to ``C`` elementwise is meaningless when two thirds of
    ``C`` is null: the null directions contribute nothing to either, but they
    do contribute to any norm taken over the whole matrix, so the ratio comes
    out looking good for the wrong reason. Restricting to the retained
    eigenvectors measures the thing that was actually sampled.
    """
    if y.shape[0] < 2:
        return float("nan")

    values, vectors = np.linalg.eigh(matrix)
    largest = float(values.max()) if values.size else 0.0
    if largest <= 0:
        return float("nan")
    keep = values > null_tol * largest
    if not np.any(keep):
        return float("nan")

    basis = vectors[:, keep]
    projected = y @ basis
    realised = np.cov(projected, rowvar=False)
    target = np.diag(values[keep])
    return float(np.linalg.norm(realised - target) / np.linalg.norm(target))
