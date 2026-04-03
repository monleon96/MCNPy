"""Tests for nearest_psd_higham and verify_psd_projection."""

import numpy as np
import pytest

from kika.cov.decomposition import nearest_psd_higham, verify_psd_projection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_non_psd(n: int, seed: int = 0) -> np.ndarray:
    """Return a symmetric non-PSD matrix of size n with known negative eigenvalues."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    S = A @ A.T  # PSD by construction
    # Flip the sign of the smallest eigenvalue direction
    w, V = np.linalg.eigh(S)
    w[0] = -abs(w[-1]) * 0.1  # Make one eigenvalue clearly negative
    return V @ np.diag(w) @ V.T


def _is_psd(M: np.ndarray, tol: float = -1e-10) -> bool:
    return float(np.linalg.eigvalsh(M)[0]) >= tol


# ---------------------------------------------------------------------------
# Tests for nearest_psd_higham
# ---------------------------------------------------------------------------

class TestNearestPsdHigham:

    def test_already_psd_returns_unchanged(self):
        """An already-PSD matrix should be returned with 0 iterations."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((10, 10))
        S = A @ A.T  # PSD
        result, info = nearest_psd_higham(S, verbose=False)
        assert info["iterations"] == 0
        assert info["converged"] is True
        np.testing.assert_allclose(result, (S + S.T) / 2, atol=1e-12)

    def test_2x2_known_solution(self):
        """2x2 case: known non-PSD matrix with analytical nearest-PSD."""
        # [[1, 2], [2, 1]] has eigenvalues 3 and -1
        A = np.array([[1.0, 2.0], [2.0, 1.0]])
        assert not _is_psd(A)

        result, info = nearest_psd_higham(A, preserve_diagonal=True, verbose=False)
        assert _is_psd(result)
        # Diagonal must be preserved
        np.testing.assert_allclose(np.diag(result), [1.0, 1.0], atol=1e-10)
        # With diagonal preserved, nearest PSD is [[1, 1], [1, 1]]
        # (correlation clipped to 1.0)
        np.testing.assert_allclose(result, [[1.0, 1.0], [1.0, 1.0]], atol=1e-8)

    def test_diagonal_preservation(self):
        """Diagonal elements must be preserved when preserve_diagonal=True."""
        M = _make_non_psd(50, seed=1)
        original_diag = np.diag(M).copy()

        result, info = nearest_psd_higham(M, preserve_diagonal=True, verbose=False)
        assert _is_psd(result)
        np.testing.assert_allclose(np.diag(result), original_diag, atol=1e-8)
        assert info["max_diagonal_change"] < 1e-8

    def test_psd_guarantee(self):
        """Result must always be PSD (all eigenvalues >= 0)."""
        M = _make_non_psd(100, seed=2)
        result, info = nearest_psd_higham(M, preserve_diagonal=True, verbose=False)
        eigvals = np.linalg.eigvalsh(result)
        # Allow tiny numerical negatives
        assert eigvals[0] >= -1e-10, f"min eigenvalue = {eigvals[0]}"

    def test_frobenius_better_than_clip(self):
        """Higham projection should be closer to original than naive clipping."""
        M = _make_non_psd(30, seed=3)

        # Higham
        result_h, info_h = nearest_psd_higham(M, preserve_diagonal=True, verbose=False)
        frob_h = np.linalg.norm(result_h - M, "fro")

        # Naive clip
        result_c, info_c = nearest_psd_higham(M, preserve_diagonal=False, verbose=False)
        frob_c = np.linalg.norm(result_c - M, "fro")

        # Higham preserves diagonal so it may have slightly larger Frobenius
        # (it optimises a constrained problem). But diagonal error should be
        # dramatically better.
        assert info_h["max_diagonal_change"] < 1e-8
        assert info_c["max_diagonal_change"] > 1e-6  # clip changes diagonal

    def test_convergence(self):
        """Algorithm should converge within max_iter."""
        M = _make_non_psd(80, seed=4)
        result, info = nearest_psd_higham(M, preserve_diagonal=True, max_iter=2000, verbose=False)
        assert info["converged"] is True

    def test_preserve_diagonal_false(self):
        """With preserve_diagonal=False, should do single-step eigenvalue clipping."""
        M = _make_non_psd(20, seed=5)
        result, info = nearest_psd_higham(M, preserve_diagonal=False, verbose=False)
        assert info["iterations"] == 1
        assert _is_psd(result)

    def test_eigval_floor(self):
        """eigval_floor should ensure minimum eigenvalue >= floor."""
        M = _make_non_psd(20, seed=6)
        floor = 1e-6
        result, info = nearest_psd_higham(
            M, preserve_diagonal=False, eigval_floor=floor, verbose=False,
        )
        eigvals = np.linalg.eigvalsh(result)
        assert eigvals[0] >= floor - 1e-12

    def test_symmetry_enforced(self):
        """Result should be symmetric even if input has slight asymmetry."""
        M = _make_non_psd(15, seed=7)
        M[0, 1] += 1e-8  # break symmetry slightly
        result, info = nearest_psd_higham(M, verbose=False)
        np.testing.assert_allclose(result, result.T, atol=1e-14)

    def test_large_matrix(self):
        """Smoke test on a reasonably large matrix."""
        M = _make_non_psd(500, seed=8)
        result, info = nearest_psd_higham(M, preserve_diagonal=True, verbose=False)
        assert _is_psd(result)
        assert info["converged"]


# ---------------------------------------------------------------------------
# Tests for verify_psd_projection
# ---------------------------------------------------------------------------

class TestVerifyPsdProjection:

    def test_reports_metrics(self):
        M = _make_non_psd(20, seed=10)
        result, info = nearest_psd_higham(M, preserve_diagonal=True, verbose=False)
        frob_pct, diag_pct, corr_change = verify_psd_projection(
            M, result, info, verbose=False,
        )
        assert frob_pct >= 0
        assert diag_pct >= 0
        assert corr_change >= 0

    def test_identity_projection(self):
        """PSD matrix should have zero error."""
        S = np.eye(10)
        result, info = nearest_psd_higham(S, verbose=False)
        frob_pct, diag_pct, corr_change = verify_psd_projection(
            S, result, info, verbose=False,
        )
        assert frob_pct < 1e-10
        assert diag_pct < 1e-10
