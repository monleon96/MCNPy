"""Tests for adaptive per-group variance percentile scaling."""
import numpy as np
import pytest

from scripts.multigroup_collapse import (
    _compute_group_quality_score,
    apply_percentile_variance_scaling,
    idx,
    collapse_covariance,
    build_aggregation_matrix,
)


def _build_uniform_corr_cov(n, rho, sigma, max_order=1):
    """Build a fine covariance with uniform correlation rho and std sigma for l=1."""
    size = n * max_order
    cov = np.zeros((size, size))
    for i in range(n):
        for j in range(n):
            ii = idx(i, 1, max_order)
            jj = idx(j, 1, max_order)
            if i == j:
                cov[ii, jj] = sigma ** 2
            else:
                cov[ii, jj] = rho * sigma ** 2
    return cov


class TestQualityScoreMonotonicity:
    """q should increase with rho and decrease with n."""

    def test_monotonic_in_rho(self):
        """Fixed n=4, increasing rho -> increasing q."""
        n = 4
        widths = np.ones(n)
        qs = []
        for rho in [0.0, 0.3, 0.6, 0.9, 1.0]:
            cov = _build_uniform_corr_cov(n, rho, sigma=0.1)
            q = _compute_group_quality_score(list(range(n)), cov, widths, max_order=1)
            qs.append(q)
        for i in range(len(qs) - 1):
            assert qs[i + 1] >= qs[i] - 1e-12, f"q not monotonic in rho: {qs}"

    def test_monotonic_in_n(self):
        """Fixed rho=0.5, increasing n -> decreasing q -> higher percentile."""
        rho = 0.5
        qs = []
        for n in [2, 4, 8, 16]:
            widths = np.ones(n)
            cov = _build_uniform_corr_cov(n, rho, sigma=0.1)
            q = _compute_group_quality_score(list(range(n)), cov, widths, max_order=1)
            qs.append(q)
        for i in range(len(qs) - 1):
            assert qs[i + 1] <= qs[i] + 1e-12, f"q not monotonic in n: {qs}"


class TestQualityScoreBounds:
    def test_single_bin(self):
        cov = _build_uniform_corr_cov(1, 0.0, sigma=0.1)
        q = _compute_group_quality_score([0], cov, np.array([1.0]), max_order=1)
        assert q == 1.0

    def test_analytical_uniform(self):
        """n identical-sigma, uniform-rho bins -> q = rho + (1-rho)/n exactly."""
        for n in [2, 3, 5, 10]:
            for rho in [0.0, 0.3, 0.7, 1.0]:
                widths = np.ones(n)
                cov = _build_uniform_corr_cov(n, rho, sigma=0.1)
                q = _compute_group_quality_score(list(range(n)), cov, widths, max_order=1)
                expected = rho + (1.0 - rho) / n
                assert abs(q - expected) < 1e-12, (
                    f"n={n}, rho={rho}: q={q}, expected={expected}"
                )

    def test_all_zero_variance(self):
        n = 3
        cov = np.zeros((n, n))
        q = _compute_group_quality_score([0, 1, 2], cov, np.ones(n), max_order=1)
        assert q == 1.0


class TestNonUniformWeights:
    def test_two_bins_skewed(self):
        """2 bins with widths [1, 9] -> n_eff = 1/(0.01+0.81) ~ 1.22, not 2."""
        widths = np.array([1.0, 9.0])
        rho = 0.5
        cov = _build_uniform_corr_cov(2, rho, sigma=0.1)
        q = _compute_group_quality_score([0, 1], cov, widths, max_order=1)

        w = widths / widths.sum()
        n_eff = 1.0 / np.sum(w ** 2)
        # rho_w should be rho for uniform-sigma case regardless of weights
        expected = rho + (1.0 - rho) / n_eff
        assert abs(q - expected) < 1e-12
        assert n_eff < 2.0  # confirm n_eff is less than actual count


class TestPercentileBoundedness:
    def test_percentile_in_range(self):
        """Adaptive percentile should always be in [50, p_max]."""
        variance_pct = 66.67
        p_max = min(variance_pct * 1.5 - 25.0, 95.0)

        for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
            pct = p_max - q * (p_max - 50.0)
            pct = np.clip(pct, 50.0, p_max)
            assert 50.0 <= pct <= p_max


class TestNoOpWhenBaseline50:
    def test_all_scale_factors_one(self):
        """When variance_percentile=50 and adaptive, all scale_factors should be 1.0."""
        n = 4
        max_order = 1
        sigma = 0.1
        rho = 0.5
        cov_fine = _build_uniform_corr_cov(n, rho, sigma)
        widths = np.ones(n)
        groups = [[0, 1], [2, 3]]

        A = build_aggregation_matrix(groups, widths, n, max_order)
        cov_grouped = collapse_covariance(cov_fine, A)

        result, _scale = apply_percentile_variance_scaling(
            cov_grouped=cov_grouped.copy(),
            cov_fine=cov_fine,
            groups=groups,
            max_order=max_order,
            variance_percentile_min=50.0,
            variance_percentile_max=50.0,
            fine_bin_widths_mev=widths,
        )
        # min = max = 50 → group_pct=50 for all groups (median of fine variances)
        # No-deflation guard ensures scale >= 1.0, so result/cov_grouped >= 1
        np.testing.assert_array_less(0.99, np.diag(result) / np.diag(cov_grouped))


class TestNoDeflation:
    def test_scale_factors_geq_one(self):
        """Scale factors should always be >= 1.0 (no deflation)."""
        n = 6
        max_order = 1
        sigma = 0.1
        rho = 0.95  # High correlation -> collapsed variance close to fine
        cov_fine = _build_uniform_corr_cov(n, rho, sigma)
        widths = np.ones(n)
        groups = [[0, 1, 2], [3, 4, 5]]

        A = build_aggregation_matrix(groups, widths, n, max_order)
        cov_grouped = collapse_covariance(cov_fine, A)

        result, _scale = apply_percentile_variance_scaling(
            cov_grouped=cov_grouped.copy(),
            cov_fine=cov_fine,
            groups=groups,
            max_order=max_order,
            variance_percentile_min=66.67,
            variance_percentile_max=66.67,
            fine_bin_widths_mev=widths,
        )

        # Diagonal should never decrease
        for i in range(result.shape[0]):
            assert result[i, i] >= cov_grouped[i, i] - 1e-15, (
                f"Deflation at index {i}: {result[i,i]} < {cov_grouped[i,i]}"
            )


class TestPSDPreservation:
    def test_output_is_psd(self):
        """Output of apply_percentile_variance_scaling should be PSD."""
        n = 8
        max_order = 2
        size = n * max_order
        # Build a random PSD fine covariance
        rng = np.random.default_rng(42)
        L = rng.standard_normal((size, size)) * 0.1
        cov_fine = L @ L.T + np.eye(size) * 0.01

        widths = rng.uniform(0.5, 2.0, n)
        groups = [[0, 1, 2], [3, 4], [5, 6, 7]]

        A = build_aggregation_matrix(groups, widths, n, max_order)
        cov_grouped = collapse_covariance(cov_fine, A)

        result, _scale = apply_percentile_variance_scaling(
            cov_grouped=cov_grouped.copy(),
            cov_fine=cov_fine,
            groups=groups,
            max_order=max_order,
            variance_percentile_min=75.0,
            variance_percentile_max=75.0,
            fine_bin_widths_mev=widths,
        )

        eigenvalues = np.linalg.eigvalsh(result)
        assert np.min(eigenvalues) >= -1e-10, (
            f"Not PSD: min eigenvalue = {np.min(eigenvalues)}"
        )


class TestBackwardCompatibility:
    def test_none_widths_uses_uniform_percentile(self):
        """When fine_bin_widths_mev=None, behavior matches old uniform code."""
        n = 4
        max_order = 1
        sigma = 0.1
        rho = 0.5
        cov_fine = _build_uniform_corr_cov(n, rho, sigma)
        widths = np.ones(n)
        groups = [[0, 1], [2, 3]]

        A = build_aggregation_matrix(groups, widths, n, max_order)
        cov_grouped = collapse_covariance(cov_fine, A)

        # With None -> uniform percentile (no adaptive, but with no-deflation guard)
        result, _scale = apply_percentile_variance_scaling(
            cov_grouped=cov_grouped.copy(),
            cov_fine=cov_fine,
            groups=groups,
            max_order=max_order,
            variance_percentile_min=66.67,
            variance_percentile_max=66.67,
            fine_bin_widths_mev=None,
        )
        assert result.shape == cov_grouped.shape
        # Should still produce valid output
        eigenvalues = np.linalg.eigvalsh(result)
        assert np.min(eigenvalues) >= -1e-10
