"""One object for the joint (sigma, a_l) covariance, and ONE map from it to points.

WHY THIS EXISTS (roadmap §10.7-7). Our matrices were never the problem: run 86's
MF33 fine block is PSD at +4.00e-09 and its MF34 fine block at -3.84e-15. What
was broken is the *fold*. Sigma_eval was assembled as three separately-computed
terms

    Sigma_eval = Sigma^MF34 + Sigma^MF33 + Sigma^cross

each carrying its own implicit map from an energy grid to a data point -- an
MF4-bin overlap average for the MF33 leg, a nearest-bin lookup for the MF34 leg,
and a *third* nearest-bin lookup for the magnitude side of the cross leg. A sum
of three such terms is not of the form ``S C S^T``, so nothing guarantees it is
PSD, and measuring PSD on the blocks says nothing about what the chi^2 folds.

Here the joint covariance is one matrix ``C`` over one parameter vector, and the
fold is one sparse operator ``S``:

    Sigma_eval = S @ C @ S.T

which is a congruence, so **Sigma_eval is PSD by theorem whenever C is**, for
every experiment, with no check needed. The runs 87-90 forensics chased a
violation that cannot occur in this formulation.

Two consequences worth stating because they are not obvious:

* Different grids per family are FINE. Sigma may live on 188 groups and a_l on
  660; ``S`` absorbs both. What is forbidden is the *same* variable reaching a
  point by two different maps, and building ``S`` once makes that impossible
  rather than merely unlikely.
* Off-grid masking becomes symmetric for free. Zeroing a row of ``S`` removes
  the point from every block at once, which is what ``build_mf34_block`` had to
  spell out by hand as ``block * np.outer(covered, covered)`` (§10.7-6).

SCOPE. This object holds **our evaluation, over our energy range**. The host
library's covariance outside it is the file's business and is handled where it
belongs, in ``merge_mf33_covariance_into_host`` / ``merge_mf34``. Every negative
eigenvalue in the shipped file was measured to come from there (§10.7-7b), and
none of it is modelled here.

PARAMETER ORDER. ``n = M0 + M1 * L``::

    sigma bin i          ->  i                          (i = 0 .. M0-1)
    a_l(bin j, order l)  ->  M0 + j * L + (l - 1)       (l = 1 .. L)

Energy slow, order fast in the a_l half -- the same layout
``create_mf34_from_covariance`` documents, so the ENDF write is a reshape and
not a transposition waiting to be got wrong.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from numpy.polynomial.legendre import legval

try:                                    # module, or script on sys.path
    from scripts.point_map import PointMap
except ImportError:                     # pragma: no cover - direct execution
    from point_map import PointMap

# ── helpers shared with the legacy fold, so equivalence is checkable ──────────

def _legendre_base_sens(mu: np.ndarray, c0: np.ndarray, l_max: int) -> np.ndarray:
    """base_sens[l-1, j] = c0(E_j) * (2l+1) * P_l(mu_j), l = 1..l_max.

    Identical to ``eval_covariance._legendre_base_sens``; duplicated only so
    this module can be imported without pulling the legacy fold in. The
    equivalence test asserts the two agree exactly.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    c0 = np.asarray(c0, dtype=float).ravel()
    P = np.zeros((l_max, mu.size), dtype=float)
    for l in range(1, l_max + 1):
        coef = np.zeros(l + 1, dtype=float)
        coef[l] = 1.0
        P[l - 1] = legval(mu, coef)
    two_l_p1 = np.arange(3, 2 * l_max + 2, 2, dtype=float)
    return c0[None, :] * two_l_p1[:, None] * P


def overlap_weights(grid_ev: np.ndarray, lo_ev: np.ndarray, hi_ev: np.ndarray) -> np.ndarray:
    """Row-normalised (N, M) overlap of each window [lo, hi] with ``grid_ev``.

    A window with no overlap gets an all-zero row -- which is the off-grid mask,
    for free and on both axes once it goes through ``S``.

    Delegates to ``PointMap``: this module needs the dense array to build ``S``,
    but the RULE lives in one place (§10.7-7), so the legacy fold and this one
    cannot drift apart.
    """
    return PointMap.overlap(grid_ev, lo_ev, hi_ev).dense()


def nearest_weights(grid_ev: np.ndarray, e_ev: np.ndarray) -> np.ndarray:
    """Row-normalised (N, M) one-hot containing-bin lookup, masked off-grid.

    ENDF LB=5/6 matrices are piecewise-constant ON their grid and assert nothing
    outside it, so a point beyond either edge gets an all-zero row rather than
    the edge interval's value. Pinning it is what manufactured a lam_min of
    -6.21e-02 in ``build_group_cross`` (§L18) and a fabricated ``Cov(a_2, a_6)``
    for JEFF in every chi^2 from run 82 to 90 (§10.7-6).

    Delegates to ``PointMap`` -- see :func:`overlap_weights`.
    """
    return PointMap.nearest(grid_ev, e_ev).dense()


def assemble_a_block(mf34, l_max: int) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Put every stored (L, L1) block onto ONE grid: the union of all of them.

    Takes anything exposing the ``MF34CovMat`` fields (``matrices``, ``l_rows``,
    ``l_cols``, ``energy_grids``, ``is_relative``), so it is testable without a
    file.

    The union refines every block's grid exactly and LB=5/6 blocks are
    piecewise-constant, so this is a relabelling, not an interpolation. Union
    bins a block does not span get ZERO from it -- ``merge_mf34`` splices per
    (L, L1) pair on that pair's own union, which is how JEFF's (2, 6) came to
    span only [1 MeV, 20 MeV] while its 20 siblings start at 1e-05 eV, and how
    every chi^2 from run 82 to 90 folded a fabricated ``Cov(a_2, a_6)``
    (§10.7-6). Doing it once, here, is the point of the whole object.

    Returns ``(grid_ev, c34, is_relative)``.
    """
    keep = [i for i in range(len(mf34.matrices))
            if 1 <= int(mf34.l_rows[i]) <= l_max
            and 1 <= int(mf34.l_cols[i]) <= l_max
            and np.asarray(mf34.matrices[i]).size]
    if not keep:
        raise ValueError(f"no MF34 blocks with 1 <= L, L1 <= {l_max}")
    rel = {bool(mf34.is_relative[i]) for i in keep}
    if len(rel) != 1:
        raise ValueError(
            f"MF34 mixes relative and absolute blocks {rel}; the joint carries "
            f"one flag per family and guessing is not safe"
        )
    grid = np.unique(np.concatenate(
        [np.asarray(mf34.energy_grids[i], dtype=float) for i in keep]))
    m1 = grid.size - 1
    mid = 0.5 * (grid[:-1] + grid[1:])
    c34 = np.zeros((m1 * l_max, m1 * l_max), dtype=float)
    for i in keep:
        g = np.asarray(mf34.energy_grids[i], dtype=float)
        blk = np.asarray(mf34.matrices[i], dtype=float)
        b = np.clip(np.searchsorted(g, mid, side="right") - 1, 0, blk.shape[0] - 1)
        cov = (mid >= g[0]) & (mid <= g[-1])
        placed = blk[np.ix_(b, b)] * np.outer(cov, cov)
        lr, lc = int(mf34.l_rows[i]) - 1, int(mf34.l_cols[i]) - 1
        rows = np.arange(m1) * l_max + lr
        cols = np.arange(m1) * l_max + lc
        c34[np.ix_(rows, cols)] += placed
        if lr != lc:
            c34[np.ix_(cols, rows)] += placed.T
    return grid, 0.5 * (c34 + c34.T), rel.pop()


def width_weights(fine_edges_ev: np.ndarray, group_edges_ev: np.ndarray) -> np.ndarray:
    """(G, M) width-weighted collapse operator: group g averages its fine bins.

    The default weighting for :meth:`JointCov.collapse`. It is a plain average
    over bin width; a run whose adaptive collapse used flux or MC weights should
    pass its own ``A`` instead of relying on this. Requires the group edges to
    be a subset of the fine edges (a genuine coarsening), because a group
    boundary that cuts a fine bin is a different operation and silently
    approximating it is how a collapse stops being a congruence.
    """
    fine = np.asarray(fine_edges_ev, dtype=float)
    grp = np.asarray(group_edges_ev, dtype=float)
    M = fine.size - 1
    G = grp.size - 1
    pos = np.searchsorted(fine, grp)
    if not np.allclose(fine[np.clip(pos, 0, M)], grp, rtol=1e-9, atol=0.0):
        raise ValueError(
            "group edges are not a subset of the fine edges; a boundary that "
            "cuts a fine bin is not a coarsening and would not be a congruence"
        )
    widths = np.diff(fine)
    A = np.zeros((G, M), dtype=float)
    for g in range(G):
        s, e = int(pos[g]), int(pos[g + 1])
        w = widths[s:e]
        A[g, s:e] = w / w.sum()
    return A


# ── report ────────────────────────────────────────────────────────────────────

# A correlation reconstructed from a file cannot be tested against 1 + 1e-9.
# ENDF-6 writes six significant digits, so a stored rho of exactly 1 comes back
# from independently-rounded covariance and variance entries off by ~1e-6.
# MEASURED on run 86: the pre-write sidecars give max|rho| = 1.000000000 exactly
# and the same matrix read back out of the _mg file gives 1.000002 -- so this is
# the ASCII round trip and nothing else. Named and defaulted per source rather
# than widened globally, because a joint assembled in memory really should hit
# 1 + 1e-9 and quietly losing that check is how a real violation would hide.
CORR_TOL_IN_MEMORY = 1.0e-9
CORR_TOL_FROM_ENDF = 5.0e-6

@dataclass
class CovReport:
    """What :meth:`JointCov.check` found. ``ok`` is the gate; the rest is why."""
    n: int
    n_sigma: int
    n_a: int
    symmetry_residual: float
    n_non_finite: int
    max_abs_correlation: float
    min_eig: Optional[float]
    n_negative_eig: Optional[int]
    min_eig_sigma: Optional[float]
    min_eig_a: Optional[float]
    negative_tolerance: float
    correlation_tolerance: float = CORR_TOL_IN_MEMORY

    @property
    def ok(self) -> bool:
        return (
            self.n_non_finite == 0
            and self.symmetry_residual <= 1e-12
            and self.max_abs_correlation <= 1.0 + self.correlation_tolerance
            and (self.min_eig is None or self.min_eig >= -self.negative_tolerance)
        )

    def __str__(self) -> str:
        def _f(x):
            return "n/a" if x is None else f"{x:.4e}"
        return (
            f"JointCov[{self.n} params = {self.n_sigma} sigma + {self.n_a} a_l]\n"
            f"  finite        : {'yes' if self.n_non_finite == 0 else str(self.n_non_finite) + ' BAD'}\n"
            f"  symmetry      : {self.symmetry_residual:.3e}\n"
            f"  max |corr|    : {self.max_abs_correlation:.9f}"
            f"   (tol 1 + {self.correlation_tolerance:.1e})\n"
            f"  min eig joint : {_f(self.min_eig)}"
            f"   (negatives: {self.n_negative_eig})\n"
            f"  min eig sigma : {_f(self.min_eig_sigma)}\n"
            f"  min eig a_l   : {_f(self.min_eig_a)}\n"
            f"  VERDICT       : {'OK' if self.ok else 'NOT OK'}"
        )


# ── the object ────────────────────────────────────────────────────────────────

@dataclass
class JointCov:
    """Joint relative covariance of (sigma, a_1..a_L) over one energy range.

    Parameters
    ----------
    grid_sigma_ev, grid_a_ev : (M0+1,), (M1+1,)
        Strictly increasing energy boundaries in eV. They need not agree.
    l_max : int
        Highest Legendre order carried (orders 1..l_max).
    matrix : (n, n)
        Relative covariance in the order documented at module level.
    sigma_is_relative, a_is_relative : bool
        Whether each half stores ``Cov/(mean_i mean_j)``. The fold scales by the
        nominal only where the stored block is relative -- the same dispatch
        ``build_mf34_block`` does per block.
    """

    grid_sigma_ev: np.ndarray
    grid_a_ev: np.ndarray
    l_max: int
    matrix: np.ndarray
    sigma_is_relative: bool = True
    a_is_relative: bool = True

    # ── construction ──────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        self.grid_sigma_ev = np.asarray(self.grid_sigma_ev, dtype=float)
        self.grid_a_ev = np.asarray(self.grid_a_ev, dtype=float)
        self.matrix = np.asarray(self.matrix, dtype=float)
        for name, g in (("grid_sigma_ev", self.grid_sigma_ev),
                        ("grid_a_ev", self.grid_a_ev)):
            if g.ndim != 1 or g.size < 2:
                raise ValueError(f"{name} must have at least 2 edges")
            if np.any(np.diff(g) <= 0):
                raise ValueError(f"{name} must be strictly increasing")
        if self.l_max < 1:
            raise ValueError("l_max must be >= 1")
        if self.matrix.shape != (self.n, self.n):
            raise ValueError(
                f"matrix is {self.matrix.shape}, expected {(self.n, self.n)} "
                f"for {self.n_sigma} sigma bins + {self.n_a_bins} a_l bins "
                f"x {self.l_max} orders"
            )

    @classmethod
    def from_blocks(
        cls,
        c33: np.ndarray,
        c34: np.ndarray,
        grid_sigma_ev: np.ndarray,
        grid_a_ev: np.ndarray,
        l_max: int,
        cross: Optional[np.ndarray] = None,
        sigma_is_relative: bool = True,
        a_is_relative: bool = True,
    ) -> "JointCov":
        """Assemble from the three blocks. ``cross`` is (M0, M1*L), None -> 0.

        The zero-cross default is the factorization the pipeline has shipped
        since run 82, so this reproduces today's numbers exactly.
        """
        c33 = np.asarray(c33, dtype=float)
        c34 = np.asarray(c34, dtype=float)
        m0 = c33.shape[0]
        na = c34.shape[0]
        n = m0 + na
        mat = np.zeros((n, n), dtype=float)
        mat[:m0, :m0] = c33
        mat[m0:, m0:] = c34
        if cross is not None:
            cx = np.asarray(cross, dtype=float)
            if cx.shape != (m0, na):
                raise ValueError(f"cross is {cx.shape}, expected {(m0, na)}")
            mat[:m0, m0:] = cx
            mat[m0:, :m0] = cx.T
        mat = 0.5 * (mat + mat.T)
        return cls(grid_sigma_ev, grid_a_ev, l_max, mat,
                   sigma_is_relative, a_is_relative)

    # ── layout ────────────────────────────────────────────────────────────────

    @property
    def n_sigma(self) -> int:
        return self.grid_sigma_ev.size - 1

    @property
    def n_a_bins(self) -> int:
        return self.grid_a_ev.size - 1

    @property
    def n_a(self) -> int:
        return self.n_a_bins * self.l_max

    @property
    def n(self) -> int:
        return self.n_sigma + self.n_a

    def a_index(self, bin_idx: int, l: int) -> int:
        """Flat index of a_l(bin). ``l`` is 1-based."""
        if not 1 <= l <= self.l_max:
            raise ValueError(f"l={l} outside 1..{self.l_max}")
        return self.n_sigma + int(bin_idx) * self.l_max + (l - 1)

    @property
    def c33(self) -> np.ndarray:
        """View of the magnitude self block (M0, M0)."""
        return self.matrix[:self.n_sigma, :self.n_sigma]

    @property
    def c34(self) -> np.ndarray:
        """View of the shape self block (M1*L, M1*L)."""
        return self.matrix[self.n_sigma:, self.n_sigma:]

    @property
    def cross(self) -> np.ndarray:
        """View of the magnitude<->shape block (M0, M1*L)."""
        return self.matrix[:self.n_sigma, self.n_sigma:]

    @property
    def has_cross(self) -> bool:
        return bool(np.any(self.cross))

    # ── verification ──────────────────────────────────────────────────────────

    def check(self, eigen: bool = True, negative_tolerance: float = 1e-8,
              correlation_tolerance: float = CORR_TOL_IN_MEMORY) -> CovReport:
        """Structural + PSD report. ``eigen=False`` skips the O(n^3) part.

        The per-block eigenvalues are reported too, but note the direction of
        the implication: a principal submatrix of a PSD matrix is PSD, so
        healthy blocks say nothing about the joint. Only ``min_eig`` does.
        """
        m = self.matrix
        n_bad = int(np.count_nonzero(~np.isfinite(m)))
        sym = 0.0 if n_bad else float(np.abs(m - m.T).max())
        d = np.diag(m).copy()
        pos = d > 0
        max_corr = 0.0
        if not n_bad and pos.any():
            sub = m[np.ix_(pos, pos)]
            den = np.sqrt(np.outer(d[pos], d[pos]))
            max_corr = float(np.abs(sub / den).max())
        min_eig = n_neg = e_s = e_a = None
        if eigen and not n_bad:
            sm = 0.5 * (m + m.T)
            ev = np.linalg.eigvalsh(sm)
            min_eig = float(ev.min())
            n_neg = int((ev < -1e-12).sum())
            if self.n_sigma:
                e_s = float(np.linalg.eigvalsh(0.5 * (self.c33 + self.c33.T)).min())
            if self.n_a:
                e_a = float(np.linalg.eigvalsh(0.5 * (self.c34 + self.c34.T)).min())
        return CovReport(
            n=self.n, n_sigma=self.n_sigma, n_a=self.n_a,
            symmetry_residual=sym, n_non_finite=n_bad,
            max_abs_correlation=max_corr,
            min_eig=min_eig, n_negative_eig=n_neg,
            min_eig_sigma=e_s, min_eig_a=e_a,
            negative_tolerance=negative_tolerance,
            correlation_tolerance=correlation_tolerance,
        )

    # ── range restriction ─────────────────────────────────────────────────────

    def restrict(self, e_lo_ev: float, e_hi_ev: float) -> "JointCov":
        """Keep only bins fully inside [e_lo, e_hi], both families.

        A principal submatrix of a PSD matrix is PSD, so restricting can never
        introduce a negative eigenvalue -- which is why "only our range" is a
        safe thing to want, and why the host remainder is not modelled here.
        """
        def _keep(grid):
            return np.flatnonzero((grid[:-1] >= e_lo_ev * (1 - 1e-12))
                                  & (grid[1:] <= e_hi_ev * (1 + 1e-12)))

        ks = _keep(self.grid_sigma_ev)
        ka = _keep(self.grid_a_ev)
        if ks.size == 0 or ka.size == 0:
            raise ValueError(
                f"[{e_lo_ev:.6g}, {e_hi_ev:.6g}] eV keeps "
                f"{ks.size} sigma and {ka.size} a_l bins; at least one of each "
                f"is required"
            )
        if np.any(np.diff(ks) != 1) or np.any(np.diff(ka) != 1):
            raise ValueError("restriction is not contiguous; grids have a gap")
        idx = np.concatenate([
            ks,
            self.n_sigma + (ka[:, None] * self.l_max
                            + np.arange(self.l_max)[None, :]).ravel(),
        ])
        gs = self.grid_sigma_ev[ks[0]:ks[-1] + 2]
        ga = self.grid_a_ev[ka[0]:ka[-1] + 2]
        return JointCov(gs, ga, self.l_max, self.matrix[np.ix_(idx, idx)],
                        self.sigma_is_relative, self.a_is_relative)

    # ── collapse ──────────────────────────────────────────────────────────────

    def collapse(
        self,
        group_edges_sigma_ev: Optional[np.ndarray] = None,
        group_edges_a_ev: Optional[np.ndarray] = None,
        a_sigma: Optional[np.ndarray] = None,
        a_a: Optional[np.ndarray] = None,
    ) -> "JointCov":
        """Coarsen to group grids. ``C' = A C A^T`` with A block-diagonal.

        Because ``A`` is applied to the WHOLE joint at once, the cross block is
        collapsed with exactly the operators its two neighbours are -- so the
        result is a congruence and **grouping cannot destroy PSD**. Collapsing
        the blocks separately, which is what the pipeline does today, has no
        such guarantee: the difference ``C - P C P^T`` it discards is indefinite
        in general (§10.7-2).

        Pass explicit ``a_sigma`` / ``a_a`` to use the run's own collapse
        weights; otherwise width weighting is built from the group edges.
        """
        if a_sigma is None:
            a_sigma = (np.eye(self.n_sigma) if group_edges_sigma_ev is None
                       else width_weights(self.grid_sigma_ev, group_edges_sigma_ev))
        if a_a is None:
            a_a = (np.eye(self.n_a_bins) if group_edges_a_ev is None
                   else width_weights(self.grid_a_ev, group_edges_a_ev))
        a_sigma = np.asarray(a_sigma, dtype=float)
        a_a = np.asarray(a_a, dtype=float)
        if a_sigma.shape[1] != self.n_sigma:
            raise ValueError(f"a_sigma has {a_sigma.shape[1]} columns, "
                             f"expected {self.n_sigma}")
        if a_a.shape[1] != self.n_a_bins:
            raise ValueError(f"a_a has {a_a.shape[1]} columns, "
                             f"expected {self.n_a_bins}")
        # Energy slow, order fast -> the a_l operator is kron(A_a, I_L).
        big = sp.block_diag(
            (sp.csr_matrix(a_sigma), sp.kron(sp.csr_matrix(a_a),
                                             sp.identity(self.l_max, format="csr"))),
            format="csr",
        )
        new = big @ self.matrix @ big.T
        new = np.asarray(new)
        new = 0.5 * (new + new.T)
        gs = (self.grid_sigma_ev if group_edges_sigma_ev is None
              else np.asarray(group_edges_sigma_ev, dtype=float))
        ga = (self.grid_a_ev if group_edges_a_ev is None
              else np.asarray(group_edges_a_ev, dtype=float))
        if gs.size - 1 != a_sigma.shape[0]:
            gs = np.arange(a_sigma.shape[0] + 1, dtype=float)
        if ga.size - 1 != a_a.shape[0]:
            ga = np.arange(a_a.shape[0] + 1, dtype=float)
        return JointCov(gs, ga, self.l_max, new,
                        self.sigma_is_relative, self.a_is_relative)

    # ── the fold ──────────────────────────────────────────────────────────────

    def operator(
        self,
        e_mev: np.ndarray,
        mu: np.ndarray,
        c0: np.ndarray,
        a_l_per_pt: np.ndarray,
        y_eval: np.ndarray,
        sigma_window_ev: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        sigma_map: str = "overlap",
        a_map: str = "nearest",
        drop: Optional[np.ndarray] = None,
    ) -> sp.csr_matrix:
        """The (N, n) sensitivity operator ``S``. This is the only map to points.

        Parameters
        ----------
        e_mev, mu, c0, a_l_per_pt, y_eval
            Per-point arrays, exactly as the legacy fold takes them.
        sigma_window_ev : (lo, hi), optional
            Per-point energy window the magnitude is averaged over, in eV.
            Required for ``sigma_map="overlap"``; this is where the MF4 bin
            edges go today, and where the TOF kernel support would go if the
            magnitude leg is ever made to match ``y_eval``'s own smoothing
            (§10.7-7c sizes that at ~10 % in sigma, so it is a separate,
            single-variable change and NOT bundled here).
        sigma_map, a_map : "overlap" | "nearest"
            Defaults reproduce the legacy fold exactly: W-average for sigma,
            containing-bin for a_l. They may differ between families without
            hurting PSD -- what matters is that each family has ONE.
        drop : (N, l_max) bool, optional
            Zero the sensitivity of these (point, order) pairs -- the §10.6-1
            null-slot removal. Zeroing it in ``S`` removes the parameter's row
            AND column at once, which is the thing the legacy code had to do in
            two places.
        """
        e_mev = np.asarray(e_mev, dtype=float).ravel()
        N = e_mev.size
        if N == 0:
            return sp.csr_matrix((0, self.n))
        e_ev = e_mev * 1e6
        a_l_per_pt = np.asarray(a_l_per_pt, dtype=float).reshape(N, -1)
        if a_l_per_pt.shape[1] < self.l_max:
            raise ValueError(
                f"a_l_per_pt carries {a_l_per_pt.shape[1]} orders, "
                f"need {self.l_max}"
            )

        # ── magnitude leg ────────────────────────────────────────────────────
        if sigma_map == "overlap":
            if sigma_window_ev is None:
                raise ValueError('sigma_map="overlap" needs sigma_window_ev')
            lo, hi = sigma_window_ev
            w_sig = overlap_weights(self.grid_sigma_ev, lo, hi)
        elif sigma_map == "nearest":
            w_sig = nearest_weights(self.grid_sigma_ev, e_ev)
        else:
            raise ValueError(f"unknown sigma_map {sigma_map!r}")
        if not self.sigma_is_relative:
            raise NotImplementedError(
                "an absolute magnitude block needs a reference sigma to define "
                "its sensitivity; MF33 LB=5 is relative and the host merge "
                "refuses anything else, so this path is unreachable today"
            )
        y = np.asarray(y_eval, dtype=float).ravel()
        s_sigma = w_sig * y[:, None]                      # dy/d(dsigma/sigma)

        # ── shape leg ────────────────────────────────────────────────────────
        if a_map == "overlap":
            if sigma_window_ev is None:
                raise ValueError('a_map="overlap" needs sigma_window_ev')
            lo, hi = sigma_window_ev
            w_a = overlap_weights(self.grid_a_ev, lo, hi)
        elif a_map == "nearest":
            w_a = nearest_weights(self.grid_a_ev, e_ev)
        else:
            raise ValueError(f"unknown a_map {a_map!r}")
        base = _legendre_base_sens(mu, c0, self.l_max)     # (L, N)
        sens = base.T.copy()                               # (N, L)
        if self.a_is_relative:
            sens *= a_l_per_pt[:, :self.l_max]
        if drop is not None:
            sens = np.where(np.asarray(drop, dtype=bool)[:, :self.l_max], 0.0, sens)
        # S_a[j, k*L + (l-1)] = sens[j, l-1] * w_a[j, k]
        s_a = (w_a[:, :, None] * sens[:, None, :]).reshape(N, self.n_a)

        return sp.csr_matrix(np.concatenate([s_sigma, s_a], axis=1))

    def fold(
        self,
        e_mev: np.ndarray,
        mu: np.ndarray,
        c0: np.ndarray,
        a_l_per_pt: np.ndarray,
        y_eval: np.ndarray,
        chunk: int = 4096,
        dtype=np.float32,
        **operator_kwargs,
    ) -> np.ndarray:
        """``Sigma_eval = S C S^T``, dense (N, N).

        PSD whenever ``self`` is, by congruence -- there is nothing to check
        per experiment and nothing to repair.

        Computed in row chunks so the (chunk, n) intermediate stays bounded;
        the (N, N) result is the same object the chi^2 already stores, so the
        peak is the existing peak plus ``chunk * n * 8`` bytes.
        """
        S = self.operator(e_mev, mu, c0, a_l_per_pt, y_eval, **operator_kwargs)
        N = S.shape[0]
        out = np.empty((N, N), dtype=dtype)
        St = S.T.tocsc()
        for i in range(0, N, chunk):
            j = min(i + chunk, N)
            # (chunk, n) dense, then (chunk, N).
            block = S[i:j] @ self.matrix
            out[i:j] = (block @ St).astype(dtype, copy=False)
        # One symmetrisation against the asymmetry float addition leaves behind;
        # the algebra is exactly symmetric.
        out = 0.5 * (out + out.T)
        return out

    # ── ENDF ──────────────────────────────────────────────────────────────────

    @classmethod
    def from_endf(
        cls,
        path: str,
        mt: int = 2,
        isotope: int = 26056,
        l_max: int = 6,
        energy_range_ev: Optional[Tuple[float, float]] = None,
    ) -> "JointCov":
        """Read MF33/MT + MF34/MT from one file into a single joint.

        Both families are put on ONE grid each: MF33's own, and for MF34 the
        **union of every stored block's grid**. The union refines each block's
        grid exactly, and LB=5/6 blocks are piecewise-constant, so placing a
        block onto it is exact, not an interpolation. Union bins a block does
        not span get ZERO from that block -- the §10.7-6 rule, applied once
        here instead of three times downstream.

        This is deliberately uniform across libraries: JEFF, JENDL and
        This_work all come through the same path and are folded by the same
        ``S``, which they are not today.
        """
        from kika.cov import MF34CovMat
        from kika.endf import read_endf

        endf = read_endf(path, mf_numbers=[33])
        mf33_file = endf.get_file(33)
        if mf33_file is None or mt not in mf33_file.sections:
            raise ValueError(f"{path}: no MF33/MT{mt}")
        cov33 = mf33_file.sections[mt].to_xs_covmat(energy_unit="MeV")
        idx = next(
            (i for i in range(cov33.num_matrices)
             if cov33.reaction_rows[i] == mt and cov33.reaction_cols[i] == mt),
            None,
        )
        if idx is None:
            raise ValueError(f"{path}: no MT{mt} self block in MF33")
        grid_sigma = np.asarray(cov33.energy_grids[idx], dtype=float)
        c33 = np.asarray(cov33.matrices[idx], dtype=float)
        sigma_rel = bool(cov33.is_relative[idx])

        mf34 = MF34CovMat.from_endf(path, energy_unit="MeV")
        mf34 = mf34.filter_by_isotope_reaction(isotope, mt)
        grid_a, c34, a_rel = assemble_a_block(mf34, l_max)

        obj = cls.from_blocks(c33, c34, grid_sigma, grid_a, l_max,
                              sigma_is_relative=sigma_rel, a_is_relative=a_rel)
        if energy_range_ev is not None:
            obj = obj.restrict(*energy_range_ev)
        return obj

    def collapse_orders(
        self,
        order_weights: Dict[int, np.ndarray],
        order_grids_ev: Dict[int, np.ndarray],
    ) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[int, np.ndarray]]:
        """Coarsen each order onto a mesh of its own, shape blocks and cross alike.

        ``order_weights[l]`` is the (G_l, M1) matrix that averages this object's
        M1 fine a_l bins onto order ``l``'s G_l groups -- the same weights the
        central values are collapsed with, so the ratio the ENDF entry means
        stays the ratio it was built from. ``order_grids_ev[l]`` gives the G_l+1
        boundaries.

        One map does both halves::

            block(l, l1) = W_l C34(l, l1) W_l1^T
            cross(l)     = Cross(l) W_l^T

        which is why they are collapsed here and not by the caller: supplied
        separately they can disagree, and a cross block whose columns sit on a
        different mesh than the shape block it correlates with is silently
        wrong -- nothing downstream can detect it.

        Being a congruence, ``W C W^T`` keeps PSD; no eigenvalue can appear that
        the fine object did not have.
        """
        orders = list(range(1, self.l_max + 1))
        missing = [l for l in orders if l not in order_weights
                   or l not in order_grids_ev]
        if missing:
            raise ValueError(
                f"order_weights/order_grids_ev are missing order(s) {missing}; "
                f"every order in 1..{self.l_max} needs both"
            )
        W, sizes = {}, {}
        for l in orders:
            w = np.asarray(order_weights[l], dtype=float)
            g = np.asarray(order_grids_ev[l], dtype=float)
            if w.ndim != 2 or w.shape[1] != self.n_a_bins:
                raise ValueError(
                    f"order_weights[{l}] is {w.shape}, expected "
                    f"(G_{l}, {self.n_a_bins})"
                )
            if g.size != w.shape[0] + 1:
                raise ValueError(
                    f"order_grids_ev[{l}] has {g.size} edges, expected "
                    f"{w.shape[0] + 1} for {w.shape[0]} groups"
                )
            if np.any(np.diff(g) <= 0):
                raise ValueError(
                    f"order_grids_ev[{l}] must be strictly increasing")
            W[l], sizes[l] = w, w.shape[0]

        c34 = self.c34
        rows = {l: np.arange(self.n_a_bins) * self.l_max + (l - 1)
                for l in orders}
        blocks = {}
        for l in orders:
            for l1 in range(l, self.l_max + 1):
                fine = c34[np.ix_(rows[l], rows[l1])]
                blocks[(l, l1)] = W[l] @ fine @ W[l1].T

        cross_blocks = {}
        if self.has_cross:
            cx = self.cross.reshape(self.n_sigma, self.n_a_bins, self.l_max)
            for l in orders:
                cross_blocks[l] = cx[:, :, l - 1] @ W[l].T
        return blocks, cross_blocks

    def to_endf_sections(self, za: float, awr: float, mat: int, mt: int = 2,
                         ltt: int = 1,
                         order_weights: Optional[Dict[int, np.ndarray]] = None,
                         order_grids_ev: Optional[Dict[int, np.ndarray]] = None):
        """Build the (MF33MT, MF34MT) sections this object represents.

        The a_l half is handed to ``create_mf34_from_covariance`` unreshaped:
        its documented layout ``idx = i_energy * n_orders + (l - l_min)`` is
        this object's layout, on purpose. The cross block, when present, goes
        out as the (L=0, L1) LB=6 sub-subsections that writer already supports.

        Pass ``order_weights`` and ``order_grids_ev`` together to give each
        order a mesh of its own -- see :meth:`collapse_orders`. ENDF-6 states
        the grids inside each (L, L1) sub-subsection, so this is ordinary
        format: an order whose coefficient is well determined need not be
        written at the resolution the noisiest one needs. Without them the
        emission is unchanged, and passing meshes that all coincide with
        ``grid_a_ev`` reproduces it byte for byte.

        The magnitude half never moves: MF33 and the L=0 dimension of the cross
        stay on ``grid_sigma_ev`` either way.

        Writing is left to the caller so the host range-merge stays in one
        place (``merge_mf33_covariance_into_host`` / ``merge_mf34``): outside
        our range is the file's business, not this object's.
        """
        from kika.endf.writers import (
            create_mf33_from_covariance,
            create_mf34_from_covariance,
        )
        if not (self.sigma_is_relative and self.a_is_relative):
            raise ValueError("ENDF LB=5/6 entries are relative; this joint is not")
        if (order_weights is None) != (order_grids_ev is None):
            raise ValueError(
                "order_weights and order_grids_ev go together; got "
                f"order_weights={'set' if order_weights is not None else 'None'}, "
                f"order_grids_ev={'set' if order_grids_ev is not None else 'None'}"
            )
        sec33 = create_mf33_from_covariance(
            self.c33, self.grid_sigma_ev, za=za, awr=awr, mat=mat, mt=mt,
        )

        if order_weights is None:
            a_cov, a_grid = self.c34, self.grid_a_ev
            cross_cov = None
            if self.has_cross:
                cx = self.cross.reshape(self.n_sigma, self.n_a_bins, self.l_max)
                cross_cov = {l + 1: cx[:, :, l] for l in range(self.l_max)}
        else:
            a_cov, cross_blocks = self.collapse_orders(
                order_weights, order_grids_ev)
            a_grid = {l: np.asarray(order_grids_ev[l], dtype=float)
                      for l in range(1, self.l_max + 1)}
            cross_cov = cross_blocks or None

        sec34 = create_mf34_from_covariance(
            a_cov, a_grid, max_order=self.l_max,
            za=za, awr=awr, mat=mat, mt=mt, ltt=ltt,
            cross_cov=cross_cov,
            cross_energy_grid_ev=self.grid_sigma_ev if cross_cov else None,
        )
        return sec33, sec34
