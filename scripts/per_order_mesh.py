"""The per-order MF34 mesh as the solution of a problem, not the output of a sweep.

ENDF-6 states the energy grids INSIDE each (L, L1) sub-subsection, so an order
whose coefficient is well resolved does not have to be written at the resolution
the noisiest one needs.  This module chooses each order's partition of the
shipped multigroup mesh, and builds the aggregators that carry the covariance
onto it.

WHY NOT A GREEDY MERGE.  The same criterion under three traversal orders gave
a_6 140/145/159 groups and 3/0/4 residual degenerate slots: the rule was fixed
but the ORDER OF APPLICATION was a free parameter, and picking the traversal
that scores best is the move that voids a result (roadmap §10.8-6).  The DP
removes the traversal entirely.

THE PROBLEM, stated once
------------------------
For each order, choose the partition of the mesh that

    minimises  (number of segments with SNR < 1,  then  information destroyed)

lexicographically -- fix what can be fixed first, and among the ways of fixing
it, destroy the least.  Both parts are parameter-free.

INFORMATION DESTROYED, and why it subsumes a rho condition
----------------------------------------------------------
Merging k groups keeps ONE number (the mean) out of a k-dimensional block B.
The variance carried by the directions dropped is

    destroyed(S) = tr(B_S) - u^T B_S u,      u = w/||w||   (w = group widths)

⛔ EXTENSIVE, not a fraction.  The first version of this DP minimised the
per-segment FRACTION ``1 - uBu/tr(B_S)``, and summing fractions rewards having
fewer segments, so the optimum collapsed a_6 to ONE group spanning 20 MeV.  The
extensive form is 0 for a singleton, (k-1)v for k independent groups of variance
v, and ~0 for perfectly redundant ones, so merging more always costs more.

For a pair with equal variances ``destroyed`` is monotone in rho: 0 at rho = +1
(the two are the same direction, nothing is lost), maximal at rho = -1.  So
minimising it prefers merging what is genuinely redundant, and a singleton costs
exactly 0 -- which is why the DP merges only when forced.

NO-CANCELLATION, as a hard constraint
-------------------------------------
A segment is admissible only if

    Var(mean) >= sum_g w_g^2 V_g / (sum_g w_g)^2

i.e. the merged sigma never falls below what INDEPENDENT averaging would give.
That is the k-group generalisation of rho >= 0, and it is a conservatism
condition: any SNR gain must come from averaging, never from opposite
information cancelling.  A group that can only reach SNR >= 1 by cancellation
therefore stays degenerate -- and is declared, not repaired.

COST.  The segment statistics are O(1) each after prefix sums, so the full
O(n^2) DP runs with NO cap on segment width -- the one free parameter this
design was at risk of needing.

WHAT THE MESH IS ALLOWED TO TOUCH
---------------------------------
Two exclusions, and neither is a tuning knob:

* **Outside the evaluation window.**  The shipped mesh spans the host's full
  range; our fits span 0.847-4.075 MeV.  Groups lying entirely outside are left
  UNTOUCHED and are never offered to the DP.  Without this they get merged into
  their in-window neighbours -- the integral functional inherits MF4's constant
  extrapolation and hands them weight 1.0 on the nearest bin -- which would
  declare our covariance out to 20 MeV, where the shipped object correctly
  declares nothing.

* **Groups with no central value.**  ``a_l`` is exactly zero wherever the group
  has no valid fine bin at that order (~37 % of parameters, rising with order),
  and a relative covariance over a zero mean has no absolute counterpart.  Such
  groups are not degenerate, they are ABSENT: they act as fixed cut points, the
  DP runs on each maximal run of live groups between them, and their own edges
  survive.  Counting them as SNR failures would drag them into live segments and
  fabricate a central value for them.

COLLAPSING IN RELATIVE SPACE, AND WHY NOT VIA THE ABSOLUTE MATRIX
-----------------------------------------------------------------
What reaches the writer is the POST-PROCESSED relative covariance -- near-zero
regularisation, the between-experiment floor, order smoothing and forward-fill
have all run.  Reconstructing an absolute matrix from it (``rel * outer(m, m)``)
and collapsing that would destroy every entry whose mean is zero, which is most
of what those steps put in.  It is also unnecessary, because the collapse is
exact in relative space:

    rel_S = sum_ij (w_i m_i)(w_j m_j) rel_ij / (sum_i w_i m_i)^2

i.e. a MEAN-WEIGHTED average of the relative entries.  Absolute never appears.
The DP's own criterion does need absolute quantities, and there it is formed
only on live slots, where ``rel * m_i * m_j`` is exact.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "segment_tables",
    "seg_stats",
    "solve_order",
    "order_cut_indices",
    "per_order_meshes",
    "order_aggregator",
    "collapse_relative_per_order",
]


# --------------------------------------------------------------------------- #
# the dynamic program
# --------------------------------------------------------------------------- #
def segment_tables(B: np.ndarray, a_g: np.ndarray, w: np.ndarray) -> dict:
    """Prefix sums giving every segment's mean, variance and tr(B) in O(1)."""
    n = len(w)
    wa = np.concatenate(([0.0], np.cumsum(w * a_g)))
    ws = np.concatenate(([0.0], np.cumsum(w)))
    tr = np.concatenate(([0.0], np.cumsum(np.diag(B))))
    w2v = np.concatenate(([0.0], np.cumsum(w ** 2 * np.diag(B))))
    M = (w[:, None] * w[None, :]) * B
    S = np.zeros((n + 1, n + 1))
    S[1:, 1:] = M.cumsum(0).cumsum(1)
    w2 = np.concatenate(([0.0], np.cumsum(w ** 2)))
    return dict(wa=wa, ws=ws, tr=tr, w2v=w2v, S=S, w2=w2)


def seg_stats(T: dict, i: int, j: int):
    """Stats of the segment covering groups ``[i, j)``: mean, var, tr, indep, destroyed."""
    ws = T["ws"][j] - T["ws"][i]
    mean = (T["wa"][j] - T["wa"][i]) / ws
    quad = T["S"][j, j] - T["S"][i, j] - T["S"][j, i] + T["S"][i, i]
    var = quad / ws ** 2
    trB = T["tr"][j] - T["tr"][i]
    indep = (T["w2v"][j] - T["w2v"][i]) / ws ** 2
    uBu = quad / (T["w2"][j] - T["w2"][i])
    destroyed = max(trB - uBu, 0.0)
    return mean, var, trB, indep, destroyed


def solve_order(B: np.ndarray, a_g: np.ndarray, w: np.ndarray):
    """Lexicographic DP over one run of groups: fewest degenerate segments, then least loss.

    Returns ``(cuts, (n_bad, destroyed, -n_segments))`` with ``cuts`` the local
    boundary indices, always starting at 0 and ending at ``len(w)``.
    """
    n = len(w)
    if n == 0:
        return np.array([0]), (0, 0.0, 0)
    T = segment_tables(B, a_g, w)
    INF = (10 ** 9, float("inf"), 10 ** 9)
    f: List[tuple] = [INF] * (n + 1)
    back = [-1] * (n + 1)
    f[0] = (0, 0.0, 0)          # third entry counts -(segments so far)
    for j in range(1, n + 1):
        for i in range(j):
            if f[i] == INF:
                continue
            mean, var, _trB, indep, loss = seg_stats(T, i, j)
            if j - i > 1 and var < indep * (1 - 1e-12):
                continue                       # cancellation: inadmissible
            sd = np.sqrt(max(var, 0.0))
            bad = 1 if (sd <= 0 or abs(mean) / sd < 1.0) else 0
            # ⚠ The third key is NEGATED: the tuple is minimised, so "more
            # segments" has to be the SMALLER key.  It resolves exact ties
            # (perfectly redundant neighbours, destroyed == 0) toward keeping
            # resolution.  Written the obvious way round it does the opposite --
            # a_2, which needs no merge at all, silently lost 16 groups to free
            # merges before this was caught.
            cand = (f[i][0] + bad, f[i][1] + max(loss, 0.0), f[i][2] - 1)
            if cand < f[j]:
                f[j], back[j] = cand, i
    cuts, j = [n], n
    while j > 0:
        j = back[j]
        cuts.append(j)
    return np.array(cuts[::-1]), f[n]


# --------------------------------------------------------------------------- #
# window, live runs, and the resulting edge set
# --------------------------------------------------------------------------- #
def _live_runs(live: np.ndarray, lo: int, hi: int) -> List[Tuple[int, int]]:
    """Maximal runs of consecutive live groups inside ``[lo, hi)``."""
    runs, start = [], None
    for g in range(lo, hi):
        if live[g]:
            if start is None:
                start = g
        elif start is not None:
            runs.append((start, g))
            start = None
    if start is not None:
        runs.append((start, hi))
    return runs


def order_cut_indices(B: np.ndarray, a_g: np.ndarray, w: np.ndarray,
                      live: np.ndarray, lo: int, hi: int):
    """Edge indices of order ``l``'s mesh, as indices into the base edge array.

    Every base edge outside ``[lo, hi]`` survives, as does every edge bounding a
    dead group.  Inside each run of live groups the DP chooses.

    ⚑ THE LOWER WINDOW EDGE IS AN EDGE.  The research fork rebuilt the mesh as
    ``base[:lo] + base[lo + cuts[1:-1]] + base[hi:]``, which drops ``base[lo]``
    in every case -- including when the DP merges nothing at all.  That fused
    the first in-window group to the last out-of-window one, i.e. it declared
    our covariance across the window boundary, which is exactly what excluding
    the outside groups is there to prevent.  It is also why a_2 came out at 702
    of 703 groups while the fork's own reasoning said it needed no merge: the
    single missing group was this edge, not a merge.
    """
    n_g = len(w)
    keep = np.zeros(n_g + 1, dtype=bool)
    keep[:lo + 1] = True                 # everything below the window, inclusive
    keep[hi:] = True                     # everything above the window, inclusive
    for g in range(lo, hi):              # dead groups keep both their edges
        if not live[g]:
            keep[g] = keep[g + 1] = True
    for i0, i1 in _live_runs(live, lo, hi):
        sl = slice(i0, i1)
        cuts, _ = solve_order(B[sl, sl], a_g[sl], w[sl])
        keep[i0 + cuts] = True
    return np.flatnonzero(keep)


def per_order_meshes(edges_ev: np.ndarray, cov_rel: np.ndarray,
                     means: np.ndarray, l_max: int,
                     window_ev: Optional[Tuple[float, float]] = None,
                     logger=None) -> Dict[int, np.ndarray]:
    """One mesh per Legendre order, as a subset of ``edges_ev``.

    Parameters
    ----------
    edges_ev
        The shipped multigroup boundaries, ``G + 1`` values in eV.
    cov_rel
        Relative covariance, group slow / order fast: ``idx = g * l_max + (l-1)``.
    means
        Nominal ``a_l`` on the same layout; exactly zero marks an absent order.
    window_ev
        ``(lo, hi)`` in eV.  Groups lying entirely outside are left untouched.
        ``None`` offers every group to the DP.

    Returns ``{l: boundaries}`` for ``l`` in ``1..l_max``.
    """
    edges_ev = np.asarray(edges_ev, dtype=float)
    n_g = len(edges_ev) - 1
    w = np.diff(edges_ev)
    if cov_rel.shape != (n_g * l_max, n_g * l_max):
        raise ValueError(
            f"cov_rel is {cov_rel.shape}, expected "
            f"{(n_g * l_max, n_g * l_max)} for {n_g} groups x {l_max} orders")
    if means.shape != (n_g * l_max,):
        raise ValueError(f"means is {means.shape}, expected {(n_g * l_max,)}")

    if window_ev is None:
        lo, hi = 0, n_g
    else:
        lo = int(np.searchsorted(edges_ev, window_ev[0] - 1e-3))
        hi = int(np.searchsorted(edges_ev, window_ev[1] + 1e-3)) - 1
        lo, hi = max(lo, 0), min(max(hi, 0), n_g)
    if logger is not None and window_ev is not None:
        logger.info(
            f"  [mesh] window [{window_ev[0]/1e6:.3f}, {window_ev[1]/1e6:.3f}] MeV "
            f"= groups {lo}..{hi} ({hi - lo} of {n_g}); "
            f"{n_g - (hi - lo)} outside are left untouched")

    meshes = {}
    for l in range(1, l_max + 1):
        sel = np.arange(n_g) * l_max + (l - 1)
        m_l = means[sel]
        live = m_l != 0.0
        B = cov_rel[np.ix_(sel, sel)] * np.outer(m_l, m_l)   # absolute on live slots
        idx = order_cut_indices(B, m_l, w, live, lo, hi)
        meshes[l] = edges_ev[idx]
        if logger is not None:
            logger.info(
                f"    a_{l}: {n_g} -> {len(idx) - 1} groups "
                f"({int(live.sum())} live, {int((~live).sum())} absent)")
    return meshes


# --------------------------------------------------------------------------- #
# carrying the covariance onto the meshes
# --------------------------------------------------------------------------- #
def order_aggregator(edges_base: np.ndarray, edges_new: np.ndarray,
                     means: np.ndarray) -> np.ndarray:
    """``W[new, base]``, the MEAN-weighted map that makes the relative collapse exact.

    With widths ``w`` and central values ``m``, the merged group's relative
    covariance is the ``u = w * m`` weighted average of the entries it covers,
    because ``rel_S = sum_ij u_i u_j rel_ij / (sum_i u_i)^2``.  Base groups with
    ``m = 0`` get weight zero -- they carry no central value, so there is nothing
    of theirs for the merged entry to be relative TO -- and a new group covering
    only such bins stays all-zero, a genuine null direction rather than a
    fabricated value.

    This is the same convention as ``multigroup_collapse.build_aggregation_matrix``
    and ``build_group_cross.row_aggregator``, which weight by width over the
    valid bins; here the validity weight is the central value itself, which is
    what relative entries require.
    """
    edges_base = np.asarray(edges_base, dtype=float)
    edges_new = np.asarray(edges_new, dtype=float)
    if not np.isin(edges_new, edges_base).all():
        raise ValueError("edges_new must be a subset of edges_base")
    w = np.diff(edges_base)
    g = np.clip(np.searchsorted(edges_new, 0.5 * (edges_base[:-1] + edges_base[1:]),
                                side="right") - 1, 0, len(edges_new) - 2)
    W = np.zeros((len(edges_new) - 1, len(edges_base) - 1))
    W[g, np.arange(len(w))] = w * np.asarray(means, dtype=float)
    tot = W.sum(axis=1, keepdims=True)
    np.divide(W, tot, out=W, where=tot != 0)
    return W


def collapse_relative_per_order(
    edges_ev: np.ndarray,
    cov_rel: np.ndarray,
    means: np.ndarray,
    l_max: int,
    meshes: Dict[int, np.ndarray],
) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Relative blocks ``{(l, l1): array}`` on the per-order meshes.

    Returns ``(blocks, grids, weights)``: the upper-triangle blocks the writer
    wants, the grids that go with them, and the aggregators, which the caller
    MUST reuse for the cross term.  Handing the cross a mesh the shape block was
    not collapsed with is silently wrong and nothing downstream can detect it.
    """
    W, grids = {}, {}
    for l in range(1, l_max + 1):
        sel = np.arange(len(edges_ev) - 1) * l_max + (l - 1)
        W[l] = order_aggregator(edges_ev, meshes[l], means[sel])
        grids[l] = np.asarray(meshes[l], dtype=float)
    rows = {l: np.arange(len(edges_ev) - 1) * l_max + (l - 1)
            for l in range(1, l_max + 1)}
    blocks = {}
    for l in range(1, l_max + 1):
        for l1 in range(l, l_max + 1):
            fine = cov_rel[np.ix_(rows[l], rows[l1])]
            blocks[(l, l1)] = W[l] @ fine @ W[l1].T
    return blocks, grids, W


def mesh_report(edges_ev: Sequence[float], meshes: Dict[int, np.ndarray],
                l_max: int) -> str:
    """One line summarising what the meshes cost and buy, for the run log."""
    n_g = len(edges_ev) - 1
    sizes = [len(meshes[l]) - 1 for l in range(1, l_max + 1)]
    before = l_max * n_g
    after = sum(sizes)
    # MF34 is quadratic per (L, L1) pair, which is what actually sets file size
    q_before = sum(n_g * n_g for _ in range(l_max) for _ in range(l_max))
    q_after = sum(sizes[a] * sizes[b] for a in range(l_max) for b in range(l_max))
    return (f"per-order mesh: groups {'/'.join(str(s) for s in sizes)} of {n_g}; "
            f"parameters {before:,d} -> {after:,d} ({100*(after/before - 1):+.1f} %); "
            f"MF34 entries {q_before:,d} -> {q_after:,d} "
            f"({100*(q_after/q_before - 1):+.1f} %)")
