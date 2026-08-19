"""Tests for :mod:`kika.endf.dcs`.

Three jobs:

1. Pin the physics against closed-form results (normalization, isotropy,
   known Jacobian limits) rather than against whatever the code happens to do.
2. Pin the new recurrence-based reconstruction against the pre-existing
   :func:`kika.utils.energy_folding.endf_angular_distribution`, so the two
   definitions of the same formula can never drift.
3. Pin the FWHM convention, which is the divergence this module exists to end.
"""

import numpy as np
import pytest

from kika._constants import FWHM_TO_SIGMA
from kika.endf import dcs
from kika.utils.energy_folding import endf_angular_distribution


MU = np.linspace(-1.0, 1.0, 401)

# Gauss-Legendre is exact for polynomials up to degree 2n-1, and every quantity
# here is a Legendre expansion. Integrating on a linear grid would leave a ~1e-6
# trapezoid error and force tolerances loose enough to hide a real mistake.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(64)


def integrate_over_mu(y_of_mu) -> float:
    r"""Exact :math:`\int_{-1}^{1} y(\mu)\,d\mu` for an expansion of order < 64."""
    return float(np.asarray(y_of_mu(_GL_NODES)) @ _GL_WEIGHTS)


# ─── Angular reconstruction ─────────────────────────────────────────────────

class TestAngularPdf:
    def test_isotropic_when_no_coefficients(self):
        assert np.allclose(dcs.angular_pdf(MU, []), 0.5)

    @pytest.mark.parametrize("coeffs", [
        [0.1],
        [0.3, 0.1],
        [0.5, 0.25, 0.1, 0.05],
        [0.2, -0.1, 0.05, -0.02, 0.01, 0.004],
    ])
    def test_integrates_to_one(self, coeffs):
        # a_0 = 1 is the MF4 normalization, so ∫f dμ = 1 whatever the a_l are.
        got = integrate_over_mu(lambda mu: dcs.angular_pdf(mu, coeffs))
        assert got == pytest.approx(1.0, rel=1e-12)

    @pytest.mark.parametrize("coeffs", [[], [0.1], [0.5, 0.25, 0.1], [0.2, -0.1, 0.05, -0.02]])
    def test_matches_the_scipy_definition(self, coeffs):
        # The one that already existed. If these ever disagree, one of the two
        # is wrong and this suite says so before the app does.
        assert np.allclose(
            dcs.angular_pdf(MU, coeffs),
            endf_angular_distribution(MU, np.asarray(coeffs)),
            rtol=1e-12, atol=1e-12,
        )

    def test_first_moment_recovers_a1(self):
        # ⟨μ⟩ = a_1 by orthogonality — an independent check on the (2l+1)/2.
        a1 = 0.37
        mean_mu = integrate_over_mu(lambda mu: mu * dcs.angular_pdf(mu, [a1]))
        assert mean_mu == pytest.approx(a1, rel=1e-12)

    def test_legendre_basis_matches_numpy(self):
        basis = dcs.legendre_basis(MU, 8)
        for l in range(9):
            expected = np.polynomial.legendre.Legendre.basis(l)(MU)
            assert np.allclose(basis[l], expected, atol=1e-12)


class TestAngularPdfUncertainty:
    def test_zero_sigmas_give_zero_band(self):
        assert np.allclose(dcs.angular_pdf_uncertainty(MU, [0.3, 0.1], [0.0, 0.0]), 0.0)

    def test_single_order_is_the_analytic_expression(self):
        sigma_a1 = 0.05
        got = dcs.angular_pdf_uncertainty(MU, [0.3], [sigma_a1])
        assert np.allclose(got, np.abs(1.5 * MU) * sigma_a1)

    def test_orders_add_in_quadrature(self):
        one = dcs.angular_pdf_uncertainty(MU, [0.3, 0.1], [0.05, 0.0])
        two = dcs.angular_pdf_uncertainty(MU, [0.3, 0.1], [0.0, 0.02])
        both = dcs.angular_pdf_uncertainty(MU, [0.3, 0.1], [0.05, 0.02])
        assert np.allclose(both, np.hypot(one, two))

    def test_length_mismatch_is_an_error(self):
        with pytest.raises(ValueError, match="equal length"):
            dcs.angular_pdf_uncertainty(MU, [0.3, 0.1], [0.05])


# ─── Frame transform ────────────────────────────────────────────────────────

class TestFrame:
    def test_alpha_prefers_awr(self):
        assert dcs.frame_alpha(awr=55.36, mass_number=56) == pytest.approx(1 / 55.36)

    def test_alpha_falls_back_to_mass_number(self):
        assert dcs.frame_alpha(mass_number=56) == pytest.approx(1.00866491595 / 56)

    def test_alpha_is_zero_when_nothing_is_known(self):
        assert dcs.frame_alpha() == 0.0

    @pytest.mark.parametrize("alpha", [1 / 55.36, 1 / 238.05, 0.5])
    def test_cosine_maps_are_inverses(self, alpha):
        mu_cm = np.linspace(-0.999, 0.999, 101)
        round_trip = dcs.cos_cm_from_cos_lab(dcs.cos_lab_from_cos_cm(mu_cm, alpha), alpha)
        assert np.allclose(round_trip, mu_cm, atol=1e-10)

    def test_heavy_target_is_almost_a_no_op(self):
        # α → 0: LAB and CM coincide, J → 1.
        alpha = 1 / 238.05
        assert np.allclose(dcs.cos_lab_from_cos_cm(MU, alpha), MU, atol=2 * alpha)
        assert np.allclose(dcs.jacobian_cm_to_lab(MU, alpha), 1.0, atol=4 * alpha)

    def test_zero_alpha_is_a_passthrough(self):
        y = dcs.angular_pdf(MU, [0.3])
        mu_out, y_out = dcs.transform_angular_curve(MU, y, 0.0, "cm2lab")
        assert np.allclose(mu_out, MU) and np.allclose(y_out, y)

    @pytest.mark.parametrize("alpha", [1 / 55.36, 1 / 12.0])
    def test_transform_conserves_the_integrated_cross_section(self, alpha):
        # ∮ dσ/dΩ dΩ is frame-invariant: the Jacobian is exactly what makes
        # ∫y dμ survive the change of variable.
        y_cm = dcs.angular_pdf(MU, [0.4, 0.2, 0.05])
        mu_lab, y_lab = dcs.transform_angular_curve(MU, y_cm, alpha, "cm2lab")
        assert np.trapezoid(y_lab, mu_lab) == pytest.approx(np.trapezoid(y_cm, MU), rel=1e-4)

    @pytest.mark.parametrize("alpha", [1 / 55.36, 1 / 12.0])
    def test_transform_round_trips(self, alpha):
        y_cm = dcs.angular_pdf(MU, [0.4, 0.2])
        mu_lab, y_lab = dcs.transform_angular_curve(MU, y_cm, alpha, "cm2lab")
        mu_back, y_back = dcs.transform_angular_curve(mu_lab, y_lab, alpha, "lab2cm")
        assert np.allclose(mu_back, MU, atol=1e-10)
        assert np.allclose(y_back, y_cm, rtol=1e-10)

    def test_bad_direction_is_an_error(self):
        with pytest.raises(ValueError, match="cm2lab"):
            dcs.transform_angular_curve(MU, MU, 0.5, "sideways")


# ─── sigma(E) ───────────────────────────────────────────────────────────────

class TestInterpolateLogLog:
    grid = np.array([1.0, 10.0, 100.0, 1000.0])
    xs = np.array([2.0, 20.0, 200.0, 2000.0])  # exactly σ = 2E, a power law

    def test_exact_on_a_power_law(self):
        got = dcs.interpolate_log_log(self.grid, self.xs, [3.0, 55.0, 700.0])
        assert np.allclose(got, [6.0, 110.0, 1400.0], rtol=1e-12)

    def test_hits_the_nodes(self):
        assert np.allclose(dcs.interpolate_log_log(self.grid, self.xs, self.grid), self.xs)

    def test_clamps_outside_the_range(self):
        assert dcs.interpolate_log_log(self.grid, self.xs, 0.01) == pytest.approx(2.0)
        assert dcs.interpolate_log_log(self.grid, self.xs, 1e6) == pytest.approx(2000.0)

    def test_falls_back_to_linear_across_a_zero(self):
        grid = np.array([1.0, 2.0])
        got = dcs.interpolate_log_log(grid, np.array([0.0, 4.0]), 1.5)
        assert got == pytest.approx(2.0)


class TestBinEdges:
    def test_interior_point_uses_midpoints(self):
        assert dcs.bin_edges_for_energy([1.0, 2.0, 4.0], 1) == (1.5, 3.0)

    def test_first_and_last_mirror_their_half_bin(self):
        assert dcs.bin_edges_for_energy([1.0, 2.0, 4.0], 0) == (0.5, 1.5)
        assert dcs.bin_edges_for_energy([1.0, 2.0, 4.0], 2) == (3.0, 5.0)

    def test_edges_never_go_negative(self):
        lo, _ = dcs.bin_edges_for_energy([1.0, 100.0], 0)
        assert lo >= 0.0

    def test_single_point_grid_is_degenerate(self):
        assert dcs.bin_edges_for_energy([7.0], 0) == (7.0, 7.0)


class TestSigmaBinAveraged:
    def test_constant_cross_section_averages_to_itself(self):
        grid = np.geomspace(1.0, 1e6, 200)
        xs = np.full_like(grid, 3.5)
        assert dcs.sigma_bin_averaged(grid, xs, 100.0, 1000.0) == pytest.approx(3.5, rel=1e-9)

    def test_degenerate_bin_falls_back_to_the_point_value(self):
        grid = np.array([1.0, 10.0, 100.0])
        xs = np.array([1.0, 2.0, 3.0])
        assert dcs.sigma_bin_averaged(grid, xs, 10.0, 10.0) == pytest.approx(2.0)

    def test_a_bin_reaching_zero_is_finite_under_lethargy_weighting(self):
        # The lowest MF4 bin mirrors its half-width below the first point and
        # clamps at 0, where 1/E diverges. The result must still be a number.
        grid = np.geomspace(1e-5, 1e7, 400)
        xs = np.full_like(grid, 2.5)
        got = dcs.sigma_bin_averaged(grid, xs, 0.0, 1.5e-5, "lethargy")
        assert np.isfinite(got)
        assert got == pytest.approx(2.5, rel=1e-9)

    def test_a_bin_entirely_below_the_grid_still_returns_a_value(self):
        grid = np.geomspace(1.0, 1e7, 100)
        xs = np.full_like(grid, 2.5)
        assert dcs.sigma_bin_averaged(grid, xs, 0.0, 1e-3, "lethargy") == pytest.approx(2.5)

    def test_constant_weighting_tolerates_a_zero_edge(self):
        grid = np.geomspace(1e-5, 1e7, 400)
        xs = np.full_like(grid, 2.5)
        assert np.isfinite(dcs.sigma_bin_averaged(grid, xs, 0.0, 1.5e-5, "constant"))

    def test_lethargy_and_constant_weighting_differ_on_a_sloped_xs(self):
        grid = np.geomspace(1.0, 1e4, 500)
        xs = grid / 1e4  # rises with E, so 1/E weighting pulls the mean down
        lethargy = dcs.sigma_bin_averaged(grid, xs, 100.0, 1000.0, "lethargy")
        constant = dcs.sigma_bin_averaged(grid, xs, 100.0, 1000.0, "constant")
        assert lethargy < constant


class TestTofResolution:
    def test_fwhm_convention_is_the_documented_one(self):
        # This is the number the app's TypeScript got wrong: it returned the
        # FWHM width itself, 2.3548x too wide.
        tof = dcs.TofResolution(flight_path_m=27.037, delta_t_ns=10.0, delta_t_is_fwhm=True)
        assert tof.sigma_e_mev(1.0) == pytest.approx(0.0043450, rel=1e-4)

    def test_sigma_convention_is_fwhm_times_the_factor(self):
        as_fwhm = dcs.TofResolution(delta_t_is_fwhm=True).sigma_e_mev(1.0)
        as_sigma = dcs.TofResolution(delta_t_is_fwhm=False).sigma_e_mev(1.0)
        assert as_sigma / as_fwhm == pytest.approx(FWHM_TO_SIGMA, rel=1e-9)

    def test_scales_as_e_to_the_three_halves(self):
        tof = dcs.TofResolution()
        assert tof.sigma_e_mev(4.0) / tof.sigma_e_mev(1.0) == pytest.approx(4 ** 1.5, rel=1e-3)

    def test_floor_applies_at_low_energy(self):
        tof = dcs.TofResolution(min_sigma_e_kev=1.0)
        assert tof.sigma_e_mev(1e-9) == pytest.approx(0.001)

    def test_vectorizes(self):
        tof = dcs.TofResolution()
        got = tof.sigma_e_mev(np.array([1.0, 4.0]))
        assert got.shape == (2,)


class TestSigmaFolded:
    grid = np.linspace(0.0, 1e7, 1001)  # 0..10 MeV in eV
    xs = 2.0 + 3.0 * (grid / 1e6)       # linear in E

    def test_folding_a_linear_xs_returns_the_centroid_value(self):
        tof = dcs.TofResolution(flight_path_m=27.037, delta_t_ns=10.0)
        got = dcs.sigma_folded(self.grid, self.xs, 5e6, tof)
        assert float(got) == pytest.approx(2.0 + 3.0 * 5.0, rel=1e-6)

    def test_folding_broadens_a_peak(self):
        grid = np.linspace(0.0, 2e6, 2001)
        peak = np.exp(-0.5 * ((grid - 1e6) / 5e3) ** 2)
        tof = dcs.TofResolution(flight_path_m=27.037, delta_t_ns=10.0)
        assert float(dcs.sigma_folded(grid, peak, 1e6, tof)) < 1.0


class TestResolveSigma:
    grid = np.geomspace(1.0, 1e7, 500)
    xs = np.full_like(grid, 4.0)

    def test_dispatches_to_each_mode(self):
        assert dcs.resolve_sigma(self.grid, self.xs, 1e5) == pytest.approx(4.0)
        assert dcs.resolve_sigma(
            self.grid, self.xs, 1e5, mode="binavg", bin_edges=(9e4, 1.1e5)
        ) == pytest.approx(4.0)
        assert float(dcs.resolve_sigma(
            self.grid, self.xs, 1e5, mode="folded", tof=dcs.TofResolution()
        )) == pytest.approx(4.0, rel=1e-6)

    def test_missing_inputs_fall_back_to_nominal_instead_of_raising(self):
        # A UI toggle must never be able to punch a hole in the curve.
        assert dcs.resolve_sigma(self.grid, self.xs, 1e5, mode="binavg") == pytest.approx(4.0)
        assert dcs.resolve_sigma(self.grid, self.xs, 1e5, mode="folded") == pytest.approx(4.0)


# ─── Coefficients in energy ─────────────────────────────────────────────────

class TestCoefficientsAtEnergies:
    energies = [1e3, 1e4, 1e5, 1e6]
    wire = {"1": [0.1, 0.2, 0.3, 0.4], "2": [0.01, 0.02, 0.03, 0.04]}

    def test_reproduces_the_grid_points(self):
        got = dcs.coefficients_at_energies(self.energies, self.wire, self.energies)
        assert np.allclose(got[0], self.wire["1"])
        assert np.allclose(got[1], self.wire["2"])

    def test_order_zero_is_dropped(self):
        # MF4 normalizes a_0 = 1; carrying it as data would double-count it.
        wire = {"0": [1.0] * 4, **self.wire}
        got = dcs.coefficients_at_energies(self.energies, wire, self.energies)
        assert got.shape[0] == 2
        assert np.allclose(got[0], self.wire["1"])

    def test_a_missing_order_becomes_a_zero_row(self):
        got = dcs.coefficients_at_energies(self.energies, {"1": [0.1] * 4, "3": [0.3] * 4}, [1e4])
        assert got.shape[0] == 3
        assert got[1, 0] == 0.0 and got[2, 0] == pytest.approx(0.3)

    def test_lin_lin_is_the_default_law(self):
        got = dcs.coefficients_at_energies(self.energies, self.wire, [5.5e3])
        assert got[0, 0] == pytest.approx(0.15)  # halfway between 0.1 and 0.2

    def test_honours_a_log_log_region(self):
        # INT=5 over the whole grid: the a_1 values are a power law in E.
        energies = [1.0, 10.0]
        got = dcs.coefficients_at_energies(
            energies, {"1": [1.0, 100.0]}, [np.sqrt(10.0)], nbt_int_pairs=[(2, 5)]
        )
        assert got[0, 0] == pytest.approx(10.0, rel=1e-9)

    def test_holds_the_edge_outside_the_range(self):
        got = dcs.coefficients_at_energies(self.energies, self.wire, [1e-3, 1e9])
        assert got[0, 0] == pytest.approx(0.1)
        assert got[0, 1] == pytest.approx(0.4)

    def test_rejects_a_ragged_wire_shape(self):
        with pytest.raises(ValueError, match="energy grid"):
            dcs.coefficients_at_energies(self.energies, {"1": [0.1, 0.2]}, [1e4])

    def test_accepts_a_matrix(self):
        got = dcs.coefficients_at_energies(self.energies, np.array([[0.1, 0.2, 0.3, 0.4]]), [1e4])
        assert got[0, 0] == pytest.approx(0.2)


# ─── Plot products ──────────────────────────────────────────────────────────

class TestDifferentialXsVsAngle:
    def test_pdf_when_no_cross_section_is_given(self):
        got = integrate_over_mu(lambda mu: dcs.differential_xs_vs_angle(mu, [0.3])[1])
        assert got == pytest.approx(1.0, rel=1e-12)

    def test_per_steradian_integrates_to_sigma_over_the_sphere(self):
        sigma = 4.2
        got = integrate_over_mu(
            lambda mu: dcs.differential_xs_vs_angle(mu, [0.3, 0.1], sigma, per_steradian=True)[1]
        )
        assert 2 * np.pi * got == pytest.approx(sigma, rel=1e-12)

    def test_per_mu_integrates_to_sigma(self):
        sigma = 4.2
        got = integrate_over_mu(
            lambda mu: dcs.differential_xs_vs_angle(mu, [0.3, 0.1], sigma, per_steradian=False)[1]
        )
        assert got == pytest.approx(sigma, rel=1e-12)

    def test_frame_change_is_applied_when_asked(self):
        alpha = 1 / 55.36
        mu_out, _ = dcs.differential_xs_vs_angle(
            MU, [0.3], 1.0, alpha=alpha, native_frame="cm", output_frame="lab"
        )
        assert not np.allclose(mu_out, MU)

    def test_same_frame_is_left_alone(self):
        mu_out, _ = dcs.differential_xs_vs_angle(
            MU, [0.3], 1.0, alpha=0.5, native_frame="cm", output_frame="cm"
        )
        assert np.allclose(mu_out, MU)


class TestDifferentialXsVsEnergy:
    energies = list(np.geomspace(1e3, 1e7, 60))
    wire = {"1": list(np.linspace(0.0, 0.6, 60)), "2": list(np.linspace(0.0, 0.2, 60))}
    xs_grid = list(np.geomspace(1.0, 1e8, 400))
    xs_vals = [3.0] * 400

    def test_defaults_to_the_mf4_grid(self):
        out = dcs.differential_xs_vs_energy(
            energies_ev=self.energies, coefficients=self.wire, mu=0.5,
            mu_frame="cm", native_frame="cm",
        )
        assert np.allclose(out["energies"], self.energies)
        assert out["values"].shape == (60,)

    def test_agrees_with_the_vs_angle_product_at_every_energy(self):
        # The two products must be the same surface sliced two ways. This is
        # the property that keeps the new curve honest against the old one.
        mu = 0.37
        out = dcs.differential_xs_vs_energy(
            energies_ev=self.energies, coefficients=self.wire, mu=mu,
            xs_energies_ev=self.xs_grid, xs_values=self.xs_vals,
            mu_frame="cm", native_frame="cm",
        )
        for i in (0, 17, 42, 59):
            a_l = [self.wire["1"][i], self.wire["2"][i]]
            _, y = dcs.differential_xs_vs_angle(np.array([mu]), a_l, 3.0, per_steradian=True)
            assert out["values"][i] == pytest.approx(float(y[0]), rel=1e-10)

    def test_pdf_only_when_no_cross_section_is_supplied(self):
        out = dcs.differential_xs_vs_energy(
            energies_ev=self.energies, coefficients=self.wire, mu=0.5,
            mu_frame="cm", native_frame="cm",
        )
        assert out["sigma"] is None
        assert out["y_unit"] == "Probability Density"

    def test_isotropic_slice_is_flat_at_one_half(self):
        flat = {"1": [0.0] * 60}
        out = dcs.differential_xs_vs_energy(
            energies_ev=self.energies, coefficients=flat, mu=-0.8,
            mu_frame="cm", native_frame="cm",
        )
        assert np.allclose(out["values"], 0.5)

    def test_lab_request_on_cm_data_maps_the_angle_and_applies_the_jacobian(self):
        alpha = 1 / 55.36
        mu_lab = 0.5
        out = dcs.differential_xs_vs_energy(
            energies_ev=self.energies, coefficients=self.wire, mu=mu_lab,
            mu_frame="lab", native_frame="cm", alpha=alpha,
        )
        assert out["mu_native"] == pytest.approx(dcs.cos_cm_from_cos_lab(mu_lab, alpha))
        assert out["jacobian"] == pytest.approx(dcs.jacobian_cm_to_lab(out["mu_native"], alpha))
        # And it agrees with transforming the full vs-angle curve then reading
        # off the requested lab angle.
        i = 30
        a_l = [self.wire["1"][i], self.wire["2"][i]]
        mu_lab_grid, y_lab = dcs.differential_xs_vs_angle(
            MU, a_l, alpha=alpha, native_frame="cm", output_frame="lab"
        )
        assert out["values"][i] == pytest.approx(
            float(np.interp(mu_lab, mu_lab_grid, y_lab)), rel=2e-4
        )

    def test_cm_request_on_lab_data_is_the_inverse_transform(self):
        alpha = 1 / 12.0
        mu_cm = 0.4
        out = dcs.differential_xs_vs_energy(
            energies_ev=self.energies, coefficients=self.wire, mu=mu_cm,
            mu_frame="cm", native_frame="lab", alpha=alpha,
        )
        assert out["mu_native"] == pytest.approx(dcs.cos_lab_from_cos_cm(mu_cm, alpha))
        assert out["jacobian"] == pytest.approx(1.0 / dcs.jacobian_cm_to_lab(mu_cm, alpha))

    def test_no_frame_change_leaves_the_jacobian_at_one(self):
        out = dcs.differential_xs_vs_energy(
            energies_ev=self.energies, coefficients=self.wire, mu=0.5,
            mu_frame="cm", native_frame="cm", alpha=1 / 55.36,
        )
        assert out["jacobian"] == 1.0
        assert out["mu_native"] == pytest.approx(0.5)

    def test_query_grid_is_clipped_to_the_mf4_range(self):
        out = dcs.differential_xs_vs_energy(
            energies_ev=self.energies, coefficients=self.wire, mu=0.5,
            query_energies_ev=[1.0, 1e5, 1e12], mu_frame="cm", native_frame="cm",
        )
        assert np.allclose(out["energies"], [1e5])

    def test_denser_query_grid_interpolates_the_coefficients(self):
        out = dcs.differential_xs_vs_energy(
            energies_ev=[1e3, 1e5], coefficients={"1": [0.0, 0.6]}, mu=1.0,
            query_energies_ev=[1e3, 5.05e4, 1e5], mu_frame="cm", native_frame="cm",
        )
        # f(μ=1) = 1/2 + 3/2·a_1, and a_1 is halfway → 0.3 lin-lin.
        assert out["values"][1] == pytest.approx(0.5 + 1.5 * 0.3, rel=1e-9)

    def test_xs_modes_all_produce_a_full_curve(self):
        for mode, kwargs in [
            ("nominal", {}),
            ("binavg", {}),
            ("folded", {"tof": dcs.TofResolution()}),
        ]:
            out = dcs.differential_xs_vs_energy(
                energies_ev=self.energies, coefficients=self.wire, mu=0.5,
                xs_energies_ev=self.xs_grid, xs_values=self.xs_vals,
                mu_frame="cm", native_frame="cm", xs_mode=mode, **kwargs,
            )
            assert out["values"].shape == (60,)
            assert np.all(np.isfinite(out["values"])), mode
            # A constant σ has nothing for any mode to change.
            assert np.allclose(out["sigma"], 3.0, rtol=1e-6), mode

    def test_per_mu_is_two_pi_times_per_steradian(self):
        common = dict(
            energies_ev=self.energies, coefficients=self.wire, mu=0.5,
            xs_energies_ev=self.xs_grid, xs_values=self.xs_vals,
            mu_frame="cm", native_frame="cm",
        )
        sr = dcs.differential_xs_vs_energy(per_steradian=True, **common)
        per_mu = dcs.differential_xs_vs_energy(per_steradian=False, **common)
        assert np.allclose(per_mu["values"], 2 * np.pi * sr["values"])

    def test_empty_energy_grid_is_an_error(self):
        with pytest.raises(ValueError, match="empty"):
            dcs.differential_xs_vs_energy(energies_ev=[], coefficients={}, mu=0.0)


class TestMaxLegendreOrder:
    """MF4 stores a_1..a_NL — a_0 = 1 is the convention and is never written —
    so the row length is the order itself, not the order plus one."""

    class _Section:
        def __init__(self, rows=None, count=None):
            if rows is not None:
                self.legendre_coefficients = rows
            if count is not None:
                self.num_legendre_coefficients = count

    def test_row_length_is_the_order(self):
        assert dcs.max_legendre_order(self._Section(rows=[[0.1, 0.2, 0.3]])) == 3

    def test_takes_the_maximum_across_the_energy_grid(self):
        # NL grows with incident energy; the low-energy rows must not cap it.
        rows = [[0.1] * 3, [0.1] * 32, [0.1] * 4]
        assert dcs.max_legendre_order(self._Section(rows=rows)) == 32

    def test_ignores_empty_rows(self):
        assert dcs.max_legendre_order(self._Section(rows=[[], [0.1, 0.2], []])) == 2

    def test_isotropic_section_is_order_zero(self):
        assert dcs.max_legendre_order(self._Section(rows=[])) == 0
        assert dcs.max_legendre_order(object()) == 0

    def test_falls_back_to_a_declared_count(self):
        assert dcs.max_legendre_order(self._Section(count=8)) == 8

    def test_a_bad_count_is_not_an_error(self):
        assert dcs.max_legendre_order(self._Section(count="oops")) == 0

    def test_agrees_with_what_the_reconstruction_consumes(self):
        # The order reported must be enough to reconstruct every coefficient:
        # feeding it back as max_order must not drop a row.
        rows = [[0.1] * 3, [0.2] * 7]
        order = dcs.max_legendre_order(self._Section(rows=rows))
        wire = {str(l): [0.1, 0.2] for l in range(1, 8)}
        assert dcs.coefficients_at_energies([1.0, 2.0], wire, [1.0], max_order=order).shape[0] == 7


# --- explicit-width bins ----------------------------------------------------


def test_bin_edges_for_width_relative_is_centred():
    lo, hi = dcs.bin_edges_for_width(1e6, mode="relative", relative_width=0.02)
    assert lo == pytest.approx(0.99e6)
    assert hi == pytest.approx(1.01e6)
    assert 0.5 * (lo + hi) == pytest.approx(1e6)


def test_bin_edges_for_width_tof_matches_the_resolution():
    """The TOF window is the folding width, so the two modes are comparable."""
    tof = dcs.TofResolution()
    sigma_ev = tof.sigma_e_mev(1.0) * 1e6
    lo, hi = dcs.bin_edges_for_width(1e6, mode="tof", tof=tof, n_sigma=1.0)
    assert hi - 1e6 == pytest.approx(sigma_ev)
    assert 1e6 - lo == pytest.approx(sigma_ev)


def test_bin_edges_for_width_scales_with_n_sigma():
    one = dcs.bin_edges_for_width(1e6, mode="tof", n_sigma=1.0)
    three = dcs.bin_edges_for_width(1e6, mode="tof", n_sigma=3.0)
    assert (three[1] - three[0]) == pytest.approx(3.0 * (one[1] - one[0]))


def test_bin_edges_for_width_refuses_the_grid_mode():
    """Silently substituting a different window would be worse than failing."""
    with pytest.raises(ValueError, match="bin_edges_for_energy"):
        dcs.bin_edges_for_width(1e6, mode="mf4grid")


def test_bin_edges_for_width_clamps_below_zero():
    lo, hi = dcs.bin_edges_for_width(1.0, mode="relative", relative_width=10.0)
    assert lo == 0.0
    assert hi > 1.0


# --- readings of a_l(E) -----------------------------------------------------


def _smooth_coefficients(n=600):
    """A two-order set of coefficients that varies smoothly with energy."""
    energies = np.logspace(0.0, 7.0, n)
    a1 = 0.30 + 0.05 * np.log10(energies)
    a2 = 0.10 - 0.01 * np.log10(energies)
    return energies, np.vstack([a1, a2])


def test_coefficients_folded_is_exact_on_a_linear_coefficient():
    """Folding a linear function returns its value at the kernel centroid."""
    energies = np.linspace(1e5, 1e7, 400)
    coefficients = np.vstack([2.0 + 3e-7 * energies])
    got = dcs.coefficients_folded(
        energies, coefficients, np.array([5e6]), dcs.TofResolution()
    )
    assert got[0, 0] == pytest.approx(2.0 + 3e-7 * 5e6, rel=1e-9)


def test_coefficients_folded_returns_a_constant_unchanged():
    energies = np.logspace(0.0, 7.0, 300)
    coefficients = np.vstack([np.full_like(energies, 0.42)])
    got = dcs.coefficients_folded(
        energies, coefficients, np.array([1e4, 1e6]), dcs.TofResolution()
    )
    assert got == pytest.approx(0.42)


def test_coefficients_bin_averaged_matches_sigma_bin_averaged_row_by_row():
    """The angular average is the same quadrature, applied per order."""
    energies, coefficients = _smooth_coefficients()
    edges = [(9.0e5, 1.1e6), (4.0e6, 6.0e6)]
    got = dcs.coefficients_bin_averaged(energies, coefficients, edges)
    for i, row in enumerate(coefficients):
        for j, (lo, hi) in enumerate(edges):
            assert got[i, j] == pytest.approx(
                dcs.sigma_bin_averaged(energies, row, lo, hi)
            )


def test_resolve_coefficients_dispatches_like_resolve_sigma():
    energies, coefficients = _smooth_coefficients()
    query = np.array([1e6])
    tof = dcs.TofResolution()
    edges = [(9.0e5, 1.1e6)]

    nominal = dcs.resolve_coefficients(energies, coefficients, query)
    assert nominal == pytest.approx(
        dcs.coefficients_at_energies(energies, coefficients, query)
    )
    assert dcs.resolve_coefficients(
        energies, coefficients, query, mode="folded", tof=tof
    ) == pytest.approx(dcs.coefficients_folded(energies, coefficients, query, tof))
    assert dcs.resolve_coefficients(
        energies, coefficients, query, mode="binavg", bin_edges=edges
    ) == pytest.approx(dcs.coefficients_bin_averaged(energies, coefficients, edges))


def test_resolve_coefficients_falls_back_rather_than_raising():
    """A mode with its inputs missing must not put a hole in the curve."""
    energies, coefficients = _smooth_coefficients()
    query = np.array([1e6])
    nominal = dcs.resolve_coefficients(energies, coefficients, query)
    assert dcs.resolve_coefficients(
        energies, coefficients, query, mode="folded", tof=None
    ) == pytest.approx(nominal)
    assert dcs.resolve_coefficients(
        energies, coefficients, query, mode="binavg", bin_edges=None
    ) == pytest.approx(nominal)


def test_the_four_readings_are_independent():
    """sigma and a_l are chosen separately, giving the four documented cases."""
    energies, coefficients = _smooth_coefficients()
    xs = 3.0 + 2e-7 * energies
    query = np.array([2e6])
    tof = dcs.TofResolution()

    sig_nom = float(np.atleast_1d(dcs.resolve_sigma(energies, xs, query))[0])
    sig_fold = float(np.atleast_1d(
        dcs.resolve_sigma(energies, xs, query, mode="folded", tof=tof))[0])
    a_nom = dcs.resolve_coefficients(energies, coefficients, query)[:, 0]
    a_fold = dcs.resolve_coefficients(
        energies, coefficients, query, mode="folded", tof=tof)[:, 0]

    mu = np.array([-0.5, 0.0, 0.5])
    cases = {
        "nominal": dcs.angular_pdf(mu, a_nom) * sig_nom,
        "sigma_only": dcs.angular_pdf(mu, a_nom) * sig_fold,
        "shape_only": dcs.angular_pdf(mu, a_fold) * sig_nom,
        "both": dcs.angular_pdf(mu, a_fold) * sig_fold,
    }
    # The factor average is the product of the two folds, by construction —
    # this is the property that makes the modes composable.
    assert cases["both"] == pytest.approx(
        cases["sigma_only"] * cases["shape_only"] / cases["nominal"]
    )
