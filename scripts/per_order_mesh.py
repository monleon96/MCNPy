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
    "physical_solve_order",
    "segment_tables",
    "seg_stats",
    "solve_order",
    "order_cut_indices",
    "per_order_meshes",
    "order_aggregator",
    "collapse_relative_per_order",
    "folded_variance_ratio",
]


#: Tolerancia RELATIVA con la que el AVISO de no-cancelacion decide que un
#: grupo cancela de verdad.
#:
#: ⚑ EL DEFECTO ERA LA ASIMETRIA, NO LA MALLA.  Las dos rutas comprueban
#: ``Var(mean) >= sum_j u_j^2 V_j / (sum_j u_j)^2`` con ``u = w * a`` -- la misma
#: desigualdad -- pero por caminos aritmeticos distintos: el DP con ``cumsum``
#: sobre la banda invertida, el colapso con ``W @ rel @ W.T``.  El DP MINIMIZA el
#: numero de grupos, asi que su optimo se apoya JUSTO sobre la restriccion; con
#: el DP tolerante (``- 1e-12|vv|``) y el aviso exacto (``- 1e-300``), los
#: segmentos de la frontera salian marcados por diferencias de ultimo bit.
#:
#: Medido sobre la run 103R4 (malla en una etapa, 7 820 parametros, la malla que
#: escribio la run): los **121** grupos que el log daba por violados
#: (12/38/48/23 en a_3..a_6) tienen cociente ``indep/own = 1`` a DIEZ cifras y
#: **ninguno** pasa de 1,001.  No hay ni una cancelacion real: son redondeo.
#:
#: ⚠ ARREGLARLO POR EL LADO DEL DP CUESTA Y NO COMPRA NADA.  Exigir el margen
#: por encima (``g >= vv (1 + 1e-9)``) sube la malla de 7 820 a **8 432**
#: parametros (+7,8 %, ~52 MiB) para negarse a fusiones cuya ganancia sobre el
#: promediado independiente es < 1e-9 -- fusiones que no destruyen nada y si
#: ahorran fichero.  Por eso el DP no se toca (su ``1e-12`` sigue literal, y la
#: malla es bit a bit la misma) y lo que se alinea es el AVISO.
#:
#: ⚑ EL SUELO DEL COLAPSO NO USA ESTA CONSTANTE: sigue siendo exacto
#: (``max(own, indep)``), que es inerte en el ruido y acota donde haga falta.
NOCANCEL_REL_MARGIN = 1e-9


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
def _live_runs(live: np.ndarray, lo: int, hi: int,
               sign: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
    """Maximal runs of consecutive live groups inside ``[lo, hi)``.

    ⚑ A SIGN CHANGE OF ``a_l`` ENDS A RUN, and that is not a refinement -- it is
    the half of the no-cancellation condition that was missing.  The relative
    variance of a merged segment is

        rel_S = Var_S / mean_S^2 = 1 / SNR_S^2 ,

    so what the relative form does under merging is decided by BOTH the variance
    and the mean.  The DP constrains the variance (``Var(mean) >=`` independent
    averaging).  Nothing constrained the mean, and the mean of a segment
    straddling a zero crossing of ``a_l`` is a DIFFERENCE, not an average: it
    can be arbitrarily small while the terms are not, and ``rel`` diverges.

    Measured on run 98, where the mesh merged across crossings:
    ``(sum|w a| / |sum w a|)^2`` reached 4.4 at a_1, 62.8 at a_4 and 2.6e5 at
    a_5, and the emission blew ``max|rel|`` from 0.9945 to 260.4.  a_2 has no
    crossing at all, its ratio is exactly 1, and it is the one order the DP left
    alone.

    With every run of one sign, ``sum|u| == |sum u|`` for every segment, so
    ``|rel_S| <= max|rel|`` termwise: the collapse becomes a CONTRACTION of the
    relative form rather than something a guard has to catch afterwards.

    It costs almost nothing.  On run 98's multigroup mesh the crossings are
    26/0/40/128/198/245 of 660 groups, so the rule still permits meshes down to
    644 parameters of 3960.
    """
    runs, start = [], None
    for g in range(lo, hi):
        if not live[g]:
            if start is not None:
                runs.append((start, g))
                start = None
        elif start is None:
            start = g
        elif sign is not None and sign[g] != sign[start]:
            runs.append((start, g))
            start = g
    if start is not None:
        runs.append((start, hi))
    return runs


# --------------------------------------------------------------------------- #
# the PHYSICAL criterion -- an alternative to the SNR/destroyed DP above
# --------------------------------------------------------------------------- #
def physical_solve_order(B: np.ndarray, a_g: np.ndarray, w: np.ndarray,
                         sigma_E: np.ndarray, *, res_factor: float = 1.0,
                         a_consistency_k: Optional[float] = None,
                         sigma_ratio_max: Optional[float] = None,
                         band: int = 400):
    """Fewest groups that the PHYSICS allows, as cuts into this run.

    ⚑ A DIFFERENT QUESTION FROM ``solve_order``.  That one repairs SNR < 1 and
    then destroys as little as possible; the criterion is a property of OUR
    Monte Carlo.  This one asks what the DATA permits, and ``destroyed`` stops
    being the objective and becomes a reported quantity.  Three conditions, and
    none of them is a knob to tune:

    * **resolution.**  A group may not be wider than ``res_factor`` times the
      worst ``sigma_E`` it covers.  Wider, and there is structure the experiment
      DID resolve and merging destroys it; narrower, and the bins are not
      independent measurements -- the same neutrons enter both -- so what the
      fit put between them is not information.  ``sigma_E`` comes from the TOF
      geometry, not from us.
    * **a_l consistency.**  The file is RELATIVE, so what has to be homogeneous
      inside a group is not sigma, it is the DENOMINATOR ``a_l``.  A group is
      admissible while ``a_l`` stays constant within ``k`` sigma of its most
      precise member.  The sign-change cut is the limiting case of this same
      rule.
    * **sigma ratio.**  ``max(sigma)/min(sigma)`` in the group, PER ORDER, on the
      RELATIVE sigma.  With ``COLLAPSE = max`` under-declaration is impossible by
      arithmetic, so what this bounds is the OVER-declaration -- the waste -- and
      the quantity wasted is ``max_j rel_jj / rel_ii``.  Reading it on the
      absolute sigma, which is what ``B`` carries, bounds the wrong ratio: run 99
      with ``c = 3`` then over-declares 536x at a_4.  It is a file-size budget,
      not physics (§J-7).

    Plus the same no-cancellation constraint as ``solve_order``: a segment is
    admissible only if the merged variance is at least what INDEPENDENT
    averaging would give, so no SNR gain can come from opposite information
    cancelling.

    ⚠ THE ACCUMULATES RUN ON THE REVERSED VECTOR and have to be put back, or the
    constraint lands on the wrong segments and gives the impossible: MORE
    restriction with FEWER groups (a_2 went 948 -> 20 the first time).
    ⚠ ``max == min == 0`` is ratio 1, NOT infinity -- a dead bin has to be able
    to be its own group, or its stretch is unreachable and the whole order costs
    ``inf``.
    """
    n = B.shape[0]
    if n == 0:
        return np.array([0], dtype=int)
    d = np.diag(B).copy()
    a_g = np.asarray(a_g, float)
    Bw = B * np.outer(w, w)
    w2, w2d = w * w, w * w * d
    # ⚑ LA SIGMA QUE EL TOPE TIENE QUE LEER ES LA RELATIVA.  ``B`` llega
    # ABSOLUTA (``per_order_meshes`` hace ``cov_rel * outer(m, m)``), y leer el
    # cociente ahi no acota lo que hay que acotar: con ``COLLAPSE = max`` el
    # consumidor ve ``R_gg / rel_ii = max_j rel_jj / rel_ii``, un cociente de
    # varianzas RELATIVAS.  Los dos difieren por el cociente de ``|a_l|`` dentro
    # del grupo, que es justo lo que la consistencia de a_l deja libre.
    # Medido en la run 99 con c = 3: leido en absoluta el maximo sobre-declara
    # 536x (a_4) y 283x (a_1); leido en relativa, 2,88x.  Cuesta 44 parametros
    # de 9341.  Ver docs/chi2-mf4/pipeline_covariance_to_tape.md §J-7.
    _aa = np.abs(a_g)
    s_rel = np.zeros(n)
    _ok_a = _aa > 0
    s_rel[_ok_a] = np.sqrt(np.maximum(d[_ok_a], 0.0)) / _aa[_ok_a]
    OK = np.zeros((n + 1, band), dtype=bool)
    for j in range(1, n + 1):
        lo = max(0, j - band)
        m = j - lo
        M = Bw[lo:j, lo:j]
        G = M[::-1, ::-1].cumsum(0).cumsum(1)[::-1, ::-1]
        g = np.ascontiguousarray(np.diagonal(G))
        vv = w2d[lo:j][::-1].cumsum()[::-1]
        # ⚠ EL 1e-12 SE QUEDA COMO ESTA, LITERAL.  Subirlo a NOCANCEL_REL_MARGIN
        # aflojaria el DP mil veces y MOVERIA la malla; lo que hay que alinear es
        # el AVISO, que es donde estaba la asimetria. Con esto la malla de la run
        # 103R4 (7 820 parametros) sigue siendo bit a bit la misma.
        ok = g >= vv - 1e-12 * np.abs(vv)                      # no-cancellation
        if sigma_ratio_max is not None:
            sg = s_rel[lo:j]                       # RELATIVA -- ver arriba
            mx = np.maximum.accumulate(sg[::-1])[::-1]
            mn = np.minimum.accumulate(sg[::-1])[::-1]
            with np.errstate(divide="ignore", invalid="ignore"):
                rat = np.where(mn > 0, mx / np.where(mn > 0, mn, 1.0),
                               np.where(mx <= 0, 1.0, np.inf))
            ok &= rat <= sigma_ratio_max
        if sigma_E is not None:
            wid = w[lo:j][::-1].cumsum()[::-1]
            res = np.minimum.accumulate(np.asarray(sigma_E)[lo:j][::-1])[::-1]
            lim = res_factor * res
            # ⛔ UN BIN MAS ANCHO QUE SU PROPIA sigma_E HACIA INADMISIBLE SU
            # PROPIO SINGLETON, Y ESO RINDE EL TRAMO ENTERO.  Sin esta linea
            # `f[n]` sale infinito y `physical_solve_order` cae en su rama de
            # emergencia: TODO el tramo vivo a singletons.  Medido en la run
            # 103R4: 82 bins por encima de 3,47 MeV son 1,48x mas anchos que su
            # resolucion (el ultimo, 4,52x), y como a_2 no cambia de signo
            # NUNCA su unico tramo son los 1738 bins -> el orden entero se
            # rendia (1738 grupos donde el criterio permite 749).  Tocaba a
            # 3 636 de los 10 428 parametros.
            #
            # ⚑ Y NO ES UNA EXCEPCION AD HOC: es la misma regla que ya lleva el
            # tope sigma-ratio unas lineas mas arriba ("max == min == 0 es
            # cociente 1, NO infinito -- un bin muerto tiene que poder ser su
            # propio grupo").  Un bin que existe en la rejilla hay que
            # escribirlo; la resolucion decide con quien se JUNTA, no si se
            # escribe.  Sobre grupos de dos o mas la condicion no se toca.
            lim[-1] = max(lim[-1], wid[-1])
            ok &= wid <= lim
        if a_consistency_k is not None:
            sabs = np.sqrt(np.maximum(d[lo:j], 0.0))
            amx = np.maximum.accumulate(a_g[lo:j][::-1])[::-1]
            amn = np.minimum.accumulate(a_g[lo:j][::-1])[::-1]
            smn = np.minimum.accumulate(sabs[::-1])[::-1]
            ok &= (amx - amn) <= a_consistency_k * np.maximum(smn, 1e-300)
        OK[j, :m] = ok[::-1]
    # fewest groups subject to admissibility
    f = np.full(n + 1, np.inf); f[0] = 0.0
    bk = np.zeros(n + 1, dtype=int)
    for j in range(1, n + 1):
        k = min(band, j)
        I = np.arange(j - 1, j - 1 - k, -1)
        c = np.where(OK[j, :k], f[I] + 1.0, np.inf)
        t = int(np.argmin(c)); f[j] = c[t]; bk[j] = I[t]
    if not np.isfinite(f[n]):
        # no admissible partition: singletons, which are always admissible
        return np.arange(n + 1, dtype=int)
    cuts, j = [], n
    while j > 0:
        cuts.append(int(bk[j])); j = int(bk[j])
    return np.array(sorted(set(cuts + [0, n])), dtype=int)


def order_cut_indices(B: np.ndarray, a_g: np.ndarray, w: np.ndarray,
                      live: np.ndarray, lo: int, hi: int,
                      physical: Optional[dict] = None,
                      sigma_E: Optional[np.ndarray] = None):
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
    # w > 0, so the sign of the emission weight u = w * a is the sign of a.
    for i0, i1 in _live_runs(live, lo, hi, np.sign(np.asarray(a_g, float))):
        sl = slice(i0, i1)
        if physical is None:
            cuts, _ = solve_order(B[sl, sl], a_g[sl], w[sl])
        else:
            cuts = physical_solve_order(
                B[sl, sl], a_g[sl], w[sl],
                None if sigma_E is None else np.asarray(sigma_E)[sl],
                **physical)
        keep[i0 + cuts] = True
    return np.flatnonzero(keep)


def per_order_meshes(edges_ev: np.ndarray, cov_rel: np.ndarray,
                     means: np.ndarray, l_max: int,
                     window_ev: Optional[Tuple[float, float]] = None,
                     logger=None, physical: Optional[dict] = None,
                     sigma_E: Optional[np.ndarray] = None) -> Dict[int, np.ndarray]:
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
        idx = order_cut_indices(B, m_l, w, live, lo, hi,
                                physical=physical, sigma_E=sigma_E)
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


def _group_of(edges_base: np.ndarray, edges_new: np.ndarray) -> np.ndarray:
    """``gi[i]`` = index in ``edges_new`` of the group base bin ``i`` falls in.

    Same rule as :func:`order_aggregator`, kept in one place so the two can
    never drift: a base bin belongs to the new group its MIDPOINT lands in.
    """
    edges_base = np.asarray(edges_base, dtype=float)
    edges_new = np.asarray(edges_new, dtype=float)
    mid = 0.5 * (edges_base[:-1] + edges_base[1:])
    return np.clip(np.searchsorted(edges_new, mid, side="right") - 1,
                   0, len(edges_new) - 2)


def folded_variance_ratio(fine_rel: np.ndarray, grouped_rel: np.ndarray,
                          gi: np.ndarray, a: np.ndarray, edges_base: np.ndarray,
                          sigma_E: np.ndarray, n_centers: int = 400) -> np.ndarray:
    """``v_mesh / v_fine`` at ``n_centers`` energies, folded with the TOF kernel.

    ⚑ THE CONSUMER DOES NOT READ A BIN.  It reads the cross section at its own
    resolution, so the quantity that has to survive grouping is the variance of
    a ``sigma_E``-wide fold, not a diagonal entry.  **Below 1 the mesh
    UNDER-DECLARES, which is inadmissible; above 1 it only wastes.**  The median
    is ~1 for any mesh -- the defect of a bad mesh is the tail -- so what is
    reported and gated is the MINIMUM over the centres.
    """
    edges_base = np.asarray(edges_base, dtype=float)
    w = np.diff(edges_base)
    E = 0.5 * (edges_base[:-1] + edges_base[1:])
    c = np.linspace(E.min(), E.max(), n_centers)
    sE = np.interp(c, E, np.asarray(sigma_E, dtype=float))
    dd = (E[:, None] - c[None, :]) / np.maximum(sE[None, :], 1e-12)
    G = w[:, None] * np.exp(-0.5 * dd * dd)
    tot = G.sum(axis=0)
    G = np.where(tot > 0, G / np.where(tot > 0, tot, 1.0), 0.0)
    aa = np.outer(a, a)
    Cf = fine_rel * aa
    Cm = grouped_rel[np.ix_(gi, gi)] * aa
    vf = np.einsum("ik,ik->k", G, Cf @ G)
    vm = np.einsum("ik,ik->k", G, Cm @ G)
    out = np.full(len(vf), np.nan)
    ok = vf > 0
    out[ok] = vm[ok] / vf[ok]
    return out


def collapse_relative_per_order(
    edges_ev: np.ndarray,
    cov_rel: np.ndarray,
    means: np.ndarray,
    l_max: int,
    meshes: Dict[int, np.ndarray],
    variant: str = "mean",
    sigma_E: Optional[np.ndarray] = None,
    calibrate_margin: bool = False,
    logger=None,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Relative blocks ``{(l, l1): array}`` on the per-order meshes.

    Returns ``(blocks, grids, weights)``: the upper-triangle blocks the writer
    wants, the grids that go with them, and the aggregators, which the caller
    MUST reuse for the cross term.  Handing the cross a mesh the shape block was
    not collapsed with is silently wrong and nothing downstream can detect it.

    ⛔ ``variant="mean"`` (the default, and what has always shipped) IS NOT
    CONSERVATIVE.  The group's entry is a mean-weighted average, so a bin with a
    large sigma averaged against small neighbours declares LESS than the fine
    object does at the resolution the file is read at -- up to 100x less in a_6
    at specific energies.

    ``variant="max"`` sets the group's relative diagonal to the LARGEST of its
    members instead.  The file is relative, so the consumer forms
    ``sigma_abs(i) = sqrt(R_gg) * |a_i|``, and with ``R_gg = max_j B_jj`` that is
    ``>= sqrt(B_ii) |a_i|`` for EVERY member.  The diagonal cannot come up short
    anywhere -- that is arithmetic, not an argument.  It is applied as a
    congruence, so no correlation moves and PSD is preserved.

    ⚠ WHAT ``max`` DOES NOT GIVE YOU is conservatism across group BOUNDARIES: a
    fold wide enough to span two groups can still see less than the fine object.
    ``calibrate_margin`` closes that, and closes it EXACTLY: folding is linear
    in the covariance, so scaling an order's block by ``1/min(ratio)`` puts the
    minimum at exactly 1 in one step -- no iteration, one scalar per order, and
    always ``>= 1``, so it is still a floor.  Needs ``sigma_E``.
    """
    if variant not in ("mean", "max"):
        raise ValueError(f"variant must be 'mean' or 'max', got {variant!r}")
    if calibrate_margin and sigma_E is None:
        raise ValueError("calibrate_margin needs sigma_E")
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

    if variant == "mean" and not calibrate_margin:
        return blocks, grids, W

    # ── the conservative diagonal, as a per-order congruence ────────────────
    S: Dict[int, np.ndarray] = {}
    for l in range(1, l_max + 1):
        own = blocks[(l, l)]
        gi = _group_of(edges_ev, meshes[l])
        d_fine = np.maximum(np.diag(cov_rel[np.ix_(rows[l], rows[l])]), 0.0)
        s = np.ones(own.shape[0])
        if variant == "max":
            target = np.zeros(own.shape[0])
            np.maximum.at(target, gi, d_fine)
            # ⛔ EL 1e-300 QUE HABIA AQUI NO ERA UNA GUARDA, ERA EL MECANISMO.
            # Si un grupo cancela exactamente -- dos bins con rho = -1 --
            # ``diag(own)`` baja a 0 y ``s`` sube a sqrt(target/1e-300) ~ 4e149,
            # que multiplica la fila y la columna enteras del grupo.  Medido
            # sobre el cov_emission de la run 102cp: pares adyacentes con
            # rho <= -0.999 son 0/0/0 en a_1-a_3 y 46/236/485 en a_4/a_5/a_6, o
            # sea el 28 % de los pares en a_6.  Hoy no llega a la cinta porque la
            # etapa 1 promedia el par antes de que el DP lo vea; se vuelve
            # alcanzable en cuanto la malla se elija sobre el FINO
            # (``MF34_MESH_SINGLE_STAGE``).
            #
            # EL SUELO CORRECTO YA ESTA DEFINIDO EN ESTE FICHERO: es la propia
            # restriccion de no-cancelacion del DP, ``Var(mean) >= sum_j w_j^2
            # V_j / (sum_j w_j)^2``, escrita con el MISMO agregador para que no
            # puedan derivar.  Donde la restriccion se cumple -- que es todo lo
            # que el DP admite -- ``diag(own) >= indep`` y este suelo es INERTE:
            # el cambio es bit a bit identico.  Donde se viola, acota ``s`` en
            # vez de dejarlo ir a 1e149.
            indep = (W[l] ** 2) @ d_fine
            _own_d = np.diag(own)
            # ⚑ EL SUELO SIGUE SIENDO EXACTO: donde `own` se queda por debajo de
            # `indep`, aunque sea por 1e-16, se usa `indep`. Es inerte ahi y
            # acota donde hace falta, y no depende de ninguna tolerancia.
            dg = np.maximum(np.maximum(_own_d, indep), 1e-300)
            s = np.sqrt(np.maximum(target / dg, 1.0))       # no-deflation guard
            # ⚑ EL AVISO, EN CAMBIO, LEE LA MISMA TOLERANCIA QUE EL DP.  Con el
            # umbral exacto marcaba 121 grupos en la run 103R4 y los 121 tenian
            # cociente 1 a diez cifras: eran el ultimo bit de dos sumas hechas
            # por caminos distintos, no una malla mal elegida. Y el texto decia
            # "la malla no los vetó", que es justo la lectura equivocada.
            # Se reporta ademas la SEVERIDAD, para que una cancelacion de verdad
            # no pueda esconderse dentro del recuento.
            viol = _own_d < indep * (1.0 - NOCANCEL_REL_MARGIN) - 1e-300
            n_cancel = int(viol.sum())
            if n_cancel and logger is not None:
                r = indep[viol] / np.maximum(_own_d[viol], 1e-300)
                logger.warning(
                    f"    a_{l}: {n_cancel} grupos cancelan por encima de la "
                    f"tolerancia ({NOCANCEL_REL_MARGIN:g}): indep/own p50 "
                    f"{np.median(r):.6g} max {r.max():.6g}, "
                    f"{int((r > 1.001).sum())} por encima de x1.001; "
                    f"el suelo de no-cancelacion los acota")
        if calibrate_margin:
            r = folded_variance_ratio(
                cov_rel[np.ix_(rows[l], rows[l])], own * np.outer(s, s), gi,
                means[rows[l]], edges_ev, sigma_E)
            lo = float(np.nanmin(r)) if np.isfinite(r).any() else 1.0
            if lo > 0:
                s = s * np.sqrt(max(1.0 / lo, 1.0))         # never below 1
        S[l] = s
        if logger is not None:
            logger.info(f"    a_{l}: diagonal '{variant}'"
                        + (", margen plegado" if calibrate_margin else "")
                        + f" -> sobre-declaracion x{s.min():.3f}-{s.max():.3f} "
                          f"(mediana {np.median(s):.3f})")
    for l in range(1, l_max + 1):
        for l1 in range(l, l_max + 1):
            blocks[(l, l1)] = S[l][:, None] * blocks[(l, l1)] * S[l1][None, :]
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
