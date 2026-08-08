"""ONE map from a covariance's own energy grid to a set of data points.

WHY THIS MODULE EXISTS (roadmap §10.7-7). The magnitude channel reached a data
point by three different rules -- the TOF kernel in ``y_eval``, the MF4-bin
overlap average ``W`` in ``build_mf33_block``, and a nearest-bin lookup in
``build_mf33_mf34_cross_block`` -- and the Legendre channel by a fourth. That is
not a style problem. The fold is

    Sigma_eval = M J M^T

which is a congruence, hence PSD whenever ``J`` is, **only if every leg of a
given parameter family reaches the points through the same M**. When the cross
term's magnitude leg used nearest-bin while the self block used ``W``, no single
``M`` existed and PSD had nothing to transfer along.

By the time this was diagnosed there were five copies of the idea in tree
(``build_mf34_block``, ``build_mf33_block``, ``build_mf33_mf34_cross_block``,
``null_slot_exposure.mf34_diag``, ``build_group_cross.shipped_c34_rel_on_base``).
The JEFF (2,6) off-grid clip of §10.7-6 had to be fixed in three of them by
hand, and the one that was missed failed four tests the same minute. So the fix
is not "patch the fourth copy", it is **one object every leg goes through**.

WHAT A MAP IS. ``P`` is (N, M): row j says how parameter bins combine into the
value at point j. Two rules are in use and both are legitimate -- what is not
legitimate is a family using both:

``nearest``
    One-hot containing-bin lookup. ENDF LB=5/6 matrices are piecewise-constant
    ON their grid, so this is what the file means. Used by the Legendre family.

``overlap``
    Row-normalised length-weighted average of the bins a window covers. Used by
    the magnitude family, where the window is the MF4 bin containing the point.

OFF-GRID IS AN ALL-ZERO ROW, IN BOTH RULES. A block whose grid does not reach a
point declares nothing there, so it must contribute nothing there -- not the
value of its nearest edge interval. Pinning it is what manufactured a lam_min of
-6.21e-02 in ``build_group_cross`` (§L18) and a fabricated ``Cov(a_2, a_6)`` for
JEFF-4.0 in every chi^2 from run 82 to run 90 (§10.7-6). An all-zero row zeroes
the point's row AND its column once it goes through the sandwich, which is the
symmetric thing the old code had to remember to do by hand on both axes.

PERFORMANCE IS WHY THIS IS A CLASS AND NOT A FUNCTION RETURNING AN ARRAY.
Cierjacks alone is 28631 points against 703 MF34 bins; materialising a dense
(N, M) and multiplying it out would cost far more than the fancy-index the fold
has always used. ``sandwich`` therefore dispatches on the rule -- one-hot pairs
take the O(N^2) indexed path, everything else falls back to the dense matmul --
so the shared definition costs nothing. ``dense()`` exists so tests can assert
the two paths agree.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class PointMap:
    """The linear map ``P`` (N x M) from parameter bins to data points.

    Build with :meth:`nearest` or :meth:`overlap`; never with ``__init__``
    directly. ``P`` is never materialised unless :meth:`dense` is called.
    """

    __slots__ = ("n_points", "n_bins", "covered", "_idx", "_w")

    def __init__(self, n_points: int, n_bins: int, covered: np.ndarray,
                 idx: Optional[np.ndarray], w: Optional[np.ndarray]) -> None:
        self.n_points = int(n_points)
        self.n_bins = int(n_bins)
        self.covered = covered
        self._idx = idx      # (N,) containing-bin index, or None for dense rules
        self._w = w          # (N, M) dense weights, or None for the one-hot rule

    # ── constructors ─────────────────────────────────────────────────────────

    @classmethod
    def nearest(cls, grid_ev: np.ndarray, e_ev: np.ndarray) -> "PointMap":
        """One-hot containing-bin lookup, all-zero row outside ``grid_ev``."""
        grid = np.asarray(grid_ev, dtype=float).ravel()
        e = np.asarray(e_ev, dtype=float).ravel()
        M = max(grid.size - 1, 0)
        if M == 0:
            return cls(e.size, 0, np.zeros(e.size, dtype=bool), None, None)
        idx = np.clip(np.searchsorted(grid, e, side="right") - 1, 0, M - 1)
        covered = (e >= grid[0]) & (e <= grid[-1])
        return cls(e.size, M, covered, idx, None)

    @classmethod
    def overlap(cls, grid_ev: np.ndarray, lo_ev: np.ndarray,
                hi_ev: np.ndarray) -> "PointMap":
        """Row-normalised overlap of each window ``[lo, hi]`` with ``grid_ev``.

        A window with no overlap gets an all-zero row, which is the off-grid
        mask for free.
        """
        grid = np.asarray(grid_ev, dtype=float).ravel()
        lo = np.asarray(lo_ev, dtype=float).ravel()
        hi = np.asarray(hi_ev, dtype=float).ravel()
        N = lo.size
        M = max(grid.size - 1, 0)
        if M == 0:
            return cls(N, 0, np.zeros(N, dtype=bool), None, np.zeros((N, 0)))
        W = np.zeros((N, M), dtype=float)
        for j in range(N):
            a, b = float(lo[j]), float(hi[j])
            if b <= a or b <= grid[0] or a >= grid[-1]:
                continue
            k0 = max(0, int(np.searchsorted(grid, a, side="right") - 1))
            k1 = min(M - 1, int(np.searchsorted(grid, b, side="left")))
            for k in range(k0, k1 + 1):
                w = min(grid[k + 1], b) - max(grid[k], a)
                if w > 0:
                    W[j, k] = w
            s = W[j].sum()
            if s > 0:
                W[j] /= s
        return cls(N, M, W.any(axis=1), None, W)

    # ── introspection ────────────────────────────────────────────────────────

    @property
    def is_one_hot(self) -> bool:
        return self._idx is not None

    @property
    def n_off_grid(self) -> int:
        """Points this map contributes nothing at."""
        return int((~self.covered).sum())

    def dense(self) -> np.ndarray:
        """Materialise ``P`` as (N, M). For tests and small cases only."""
        if self._w is not None:
            return self._w
        P = np.zeros((self.n_points, self.n_bins), dtype=float)
        if self.n_bins:
            P[np.arange(self.n_points), self._idx] = self.covered.astype(float)
        return P

    # ── the fold ─────────────────────────────────────────────────────────────

    def sandwich(self, mat: np.ndarray,
                 other: Optional["PointMap"] = None) -> np.ndarray:
        """``self.P @ mat @ other.P.T``, as (N, N').

        ``other`` defaults to ``self`` (the self-block case). ``mat`` must be
        (self.n_bins, other.n_bins).
        """
        if other is None:
            other = self
        m = np.asarray(mat, dtype=float)
        if m.shape != (self.n_bins, other.n_bins):
            raise ValueError(
                f"matrix is {m.shape}, expected "
                f"({self.n_bins}, {other.n_bins}) for this pair of maps"
            )
        if self.n_bins == 0 or other.n_bins == 0:
            return np.zeros((self.n_points, other.n_points), dtype=float)
        if self.is_one_hot and other.is_one_hot:
            # The path the fold has always taken: O(N*N') gather, no (N, M)
            # intermediate. Identical arithmetic to the dense product, which
            # `test_point_map.py` asserts rather than assumes.
            out = m[np.ix_(self._idx, other._idx)]
            if not (self.covered.all() and other.covered.all()):
                out = out * np.outer(self.covered, other.covered)
            return out
        return self.dense() @ m @ other.dense().T

    def sandwich_diag(self, mat: np.ndarray) -> np.ndarray:
        """``diag(self.P @ mat @ self.P.T)``, as (N,), without the (N, N)."""
        m = np.asarray(mat, dtype=float)
        if m.shape != (self.n_bins, self.n_bins):
            raise ValueError(
                f"matrix is {m.shape}, expected ({self.n_bins}, {self.n_bins})"
            )
        if self.n_bins == 0:
            return np.zeros(self.n_points, dtype=float)
        if self.is_one_hot:
            # Both axes are the same point, so `covered` enters ONCE, not as an
            # outer product.
            return m[self._idx, self._idx] * self.covered
        W = self.dense()
        return np.einsum("jm,mn,jn->j", W, m, W, optimize=True)
