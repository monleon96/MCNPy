"""Perturbing a fission spectrum: the identities, and the measured budgets.

The strongest test here is
:func:`test_r_identically_one_reproduces_chi_pointwise`. Every node the
pipeline inserts — outgoing group boundaries, their shoulders, and the incident
shoulder below each band edge — is supposed to be an *exact refinement* of the
ENDF interpolant. If any of them is not, the baseline moves, the group
integrals stop being what was computed, and the final renormalisation absorbs
the difference into a scalar that hides it. Feeding zero deltas through the
whole path and demanding χ back to 1e-15 is what makes that impossible.

Everything else is a budget, and each is asserted against a number that was
measured rather than chosen.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from kika.endf import read_endf
from kika.endf.classes.mf5.partials import MF5PartialTabulated
import kika.sampling.mf35_sampling as sampling
from kika.sampling.mf35_sampling import (
    band_grids,
    build_pfns_covariance,
    generate_pfns_samples,
    normalisation_residual,
    perturb_pfns_partial,
    realised_covariance_report,
    realised_group_probabilities,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic(micro_pfns_cov_tape):
    """Eight groups, two bands: the fast lane."""
    endf = read_endf(str(micro_pfns_cov_tape), mf_numbers=[5, 35])
    suite, section, bands = build_pfns_covariance(endf, mt=18)
    return suite, section, bands, band_grids(suite)


@pytest.fixture(scope="module")
def cf252(micro_pfns_tape):
    """Four real LB=7 bands over a real Cf-252 MF5 table."""
    endf = read_endf(str(micro_pfns_tape), mf_numbers=[5, 35])
    suite, section, bands = build_pfns_covariance(endf, mt=18)
    return suite, section, bands, band_grids(suite)


def chi_at(partial, energy, points):
    grid, values = partial.evaluate_at_incident(energy)
    return np.interp(points, grid, values, left=0.0, right=0.0)


def zero_deltas(grids):
    return {index: np.zeros(grid.size - 1) for index, grid in enumerate(grids)}


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def test_the_bands_come_off_the_covariance_suite(cf252):
    suite, section, bands, grids = cf252
    assert len(bands) == 4
    assert [g.size for g in grids] == [123] * 4
    assert bands[0][0] == pytest.approx(1e-5)
    assert bands[-1][1] == pytest.approx(2e7)
    assert isinstance(section.partials[0], MF5PartialTabulated)


def test_a_tape_with_no_mf35_is_refused_with_the_reason(micro_tape):
    """Fe-56 has neither, and the message has to say which is missing and why."""
    endf = read_endf(str(micro_tape), mf_numbers=[3])
    with pytest.raises(ValueError, match=r"no MF5/MT18"):
        build_pfns_covariance(endf, mt=18)


def test_the_reserved_parameters_refuse_rather_than_ignore(cf252, micro_pfns_tape):
    """A parameter that is silently ignored is worse than one that is absent."""
    endf = read_endf(str(micro_pfns_tape), mf_numbers=[5, 35])
    with pytest.raises(NotImplementedError, match="covariance_override"):
        build_pfns_covariance(endf, mt=18, covariance_override=object())
    with pytest.raises(NotImplementedError, match="energy_unit"):
        build_pfns_covariance(endf, mt=18, energy_unit="MeV")


def test_cf252_normalisation_residual_matches_the_measured_value(cf252):
    """The input's own departure from ∫χ = 1, before anything is perturbed."""
    _, section, _, _ = cf252
    residual = normalisation_residual(section)
    assert residual["n_nodes"] == 31
    assert residual["max_abs"] == pytest.approx(4.306e-7, rel=1e-3)
    assert residual["max_abs"] < 1e-6


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def test_generate_pfns_samples_returns_float64_and_absolute_deltas(synthetic):
    """Guards the correction that ruled out reusing ``generate_samples``.

    Its linear branch returns ``Y + 1`` and casts to float32. MF35 deltas run
    many decades below float32 epsilon, so stored that way they are annihilated
    for all but the largest groups — and silently: the run completes, writes
    tapes, passes NJOY, and contains no perturbation.
    """
    suite, _, bands, grids = synthetic
    samples, diagnostics = generate_pfns_samples(
        suite, 64, isotope="Cf252", seed=1, verbose=False,
    )

    assert set(samples) == {("Cf252", 18, b) for b in range(len(bands))}
    for key, drawn in samples.items():
        assert drawn.dtype == np.float64
        assert drawn.shape == (64, grids[key[-1]].size - 1)
        assert abs(float(drawn.mean())) < 1e-3, "these are deltas, not factors"
        assert diagnostics[key]["space"] == "linear"
        assert diagnostics[key]["returns"] == "deltas"


def test_generate_pfns_samples_rejects_cholesky(synthetic):
    suite, _, _, _ = synthetic
    with pytest.raises(ValueError, match="Cholesky is refused"):
        generate_pfns_samples(suite, 8, decomposition_method="cholesky",
                              verbose=False)


def test_the_sampling_space_is_linear_and_says_why():
    """A module constant so it cannot be 'fixed' back to log by analogy."""
    assert sampling.SAMPLING_SPACE == "linear"


def test_every_drawn_sample_has_a_vanishing_delta_sum(cf252):
    """``C·1 ≈ 0`` in, ``1ᵀδ ≈ 0`` out. The normalisation argument in one line."""
    suite, _, bands, _ = cf252
    samples, _ = generate_pfns_samples(suite, 128, isotope="Cf252", seed=3,
                                       verbose=False)
    for band in range(len(bands)):
        drawn = samples[("Cf252", 18, band)]
        assert np.max(np.abs(drawn.sum(axis=1))) < 1e-3


def test_the_sigma_clamp_touches_only_groups_that_carry_no_mass(cf252):
    """It removes PSD-repair debris, and must not be shaping the distribution.

    Measured on Cf-252: 4-5 groups of 122 per band are clamped, holding ~1e-14
    of the spectrum between them, while every group holding more than 1e-4 of
    the spectrum keeps its stated standard deviation to within 1 %.
    """
    suite, section, bands, grids = cf252
    samples, diagnostics = generate_pfns_samples(
        suite, 256, isotope="Cf252", seed=7, verbose=False,
    )
    partial = section.partials[0]

    for band in range(len(bands)):
        matrix = np.asarray(list(suite)[band].form.matrix)
        drawn = samples[("Cf252", 18, band)]
        sigma = np.sqrt(np.clip(np.diag(matrix), 0.0, None))
        stated = sigma > 0

        node = next(k for k, e in enumerate(partial.incident_energies)
                    if bands[band][0] <= e < bands[band][1])
        share = partial.group_integrals(node, grids[band])
        share = share / share.sum()

        realised = drawn.std(axis=0, ddof=1)
        significant = share > 1e-4
        np.testing.assert_allclose(
            realised[significant] / sigma[significant], 1.0, atol=0.05,
        )

        clamped_often = (np.abs(drawn) > 5.0 * sigma).mean(axis=0) > 0.01
        assert share[clamped_often].sum() < 1e-10
        assert diagnostics[("Cf252", 18, band)]["sigma_clamp"] == 5.0


def test_realised_covariance_matches_mf35_on_the_non_null_subspace(cf252):
    """The acceptance gate. Elementwise comparison would be meaningless here.

    Half of every Cf-252 band is null space: it contributes to a norm over the
    whole matrix but carries no information, so a bad ensemble scores well.
    The gate is the median over *retained* modes of ``Var(a_k)/λ_k − 1``,
    against the 3σ band of a χ² variance estimate — the median rather than the
    max, because over sixty modes the max is dominated by sampling noise on the
    smallest retained eigenvalue.
    """
    suite, _, bands, _ = cf252
    samples, _ = generate_pfns_samples(suite, 512, isotope="Cf252", seed=7,
                                       sampling_method="random", verbose=False)

    for band in range(len(bands)):
        matrix = np.asarray(list(suite)[band].form.matrix)
        report = realised_covariance_report(samples[("Cf252", 18, band)], matrix)

        assert report["n_modes_null"] > 0, "this data is rank deficient"
        assert report["passes_spectral_gate"], report
        assert report["null_leakage"] < 1e-3, (
            "the ensemble reached a direction MF35 never authorised"
        )


# ---------------------------------------------------------------------------
# Applying a sample — the identities
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", ["synthetic", "cf252"])
def test_r_identically_one_reproduces_chi_pointwise(request, fixture):
    """**The strongest test in this suite.**

    Zero deltas through the whole path — incident shoulder insertion, group
    integration, projection, node insertion, scaling, renormalisation — and χ
    must come back unchanged at 10 000 random ``(E, E')``, *including* at the
    inserted incident shoulder nodes, which is where a non-exact refinement
    would show up.

    The bug this rules out is the tempting shortcut in
    ``_insert_incident_shoulders``: copying the band edge's own table instead
    of blending the two bracketing tables. That is ~0.1 % wrong and would look
    like a physical central-value shift rather than like a bug.
    """
    _, section, bands, grids = request.getfixturevalue(fixture)
    original = copy.deepcopy(section.partials[0])
    perturbed = copy.deepcopy(section.partials[0])

    perturbed, diagnostics = perturb_pfns_partial(
        perturbed, zero_deltas(grids), bands, grids,
    )

    assert diagnostics["n_incident_inserted"] == len(bands) - 1
    assert diagnostics["total_outgoing_inserted"] == 0, (
        "a constant factor steps nowhere, so nothing should be inserted"
    )
    assert diagnostics["max_renormalisation_error"] < 1e-12

    rng = np.random.default_rng(20260810)
    probe_incident = rng.uniform(bands[0][0], bands[-1][1], 200)
    probe_outgoing = rng.uniform(0.0, 2.0e7, 50)
    # The inserted shoulders themselves, which the random draw would miss.
    probe_incident = np.concatenate([
        probe_incident,
        [edge * (1.0 - sampling.INCIDENT_STEP_SHOULDER) for edge, _ in bands[1:]],
    ])

    before = np.array([chi_at(original, e, probe_outgoing) for e in probe_incident])
    after = np.array([chi_at(perturbed, e, probe_outgoing) for e in probe_incident])
    np.testing.assert_allclose(after, before, rtol=5e-14, atol=1e-300)


def test_a_uniform_factor_gives_exactly_r_times_p0(synthetic):
    """No steps, so no shoulder ramp, so the group identity is exact.

    Isolates step 6's scaling from the ramp that step 6's shoulders introduce:
    with one factor everywhere, ``P_j`` must be ``r·P⁰_j`` to machine precision
    before the final renormalisation puts the integral back.
    """
    _, section, bands, grids = synthetic
    original = copy.deepcopy(section.partials[0])
    partial = copy.deepcopy(section.partials[0])

    band, node = 0, 0
    boundaries = grids[band]
    p0 = original.group_integrals(node, boundaries)
    deltas = zero_deltas(grids)
    deltas[band] = 0.05 * p0                      # r = 1.05 in every group

    partial, _ = perturb_pfns_partial(partial, deltas, bands, grids)
    realised = realised_group_probabilities(partial, node, boundaries)

    # The projection removes the 5 % excess as a uniform ratio shift, and the
    # renormalisation restores ∫χ, so what survives is exactly the input.
    np.testing.assert_allclose(realised, p0, rtol=1e-12)


def test_shoulder_ramp_error_stays_inside_its_budget(cf252):
    """What a stepping factor costs, measured two ways.

    The relative form is unbounded wherever the projection drove a group's
    ratio towards zero, so the budget that means anything is the discrepancy as
    a fraction of the whole spectrum. Measured on Cf-252: relative ≤1.6e-2,
    mass-weighted ≤8.4e-7.
    """
    suite, section, bands, grids = cf252
    samples, _ = generate_pfns_samples(suite, 8, isotope="Cf252", seed=7,
                                       verbose=False)

    for index in range(4):
        partial = copy.deepcopy(section.partials[0])
        deltas = {b: samples[("Cf252", 18, b)][index] for b in range(len(bands))}
        partial, diagnostics = perturb_pfns_partial(partial, deltas, bands, grids)

        assert diagnostics["max_group_mass_error"] < 1e-5
        assert diagnostics["max_group_ratio_error"] < 0.1
        assert diagnostics["max_renormalisation_error"] < 1e-4


def test_the_energy_relative_shoulder_would_have_been_much_worse(cf252):
    """Why the shoulder is scaled by group width and not by energy.

    Cf-252's top groups are 2e5 wide at 2e7 eV, so a shoulder at ``E(1-1e-3)``
    sits a tenth of a group below the boundary and the ramp spans a tenth of
    the group. Copying ``NUBAR_STEP_SHOULDER``'s energy-relative form — correct
    there, because MF31 bins span decades — gave a mass error two orders of
    magnitude larger. Pinned so the constant cannot drift back.
    """
    suite, section, bands, grids = cf252
    samples, _ = generate_pfns_samples(suite, 4, isotope="Cf252", seed=7,
                                       verbose=False)
    deltas = {b: samples[("Cf252", 18, b)][0] for b in range(len(bands))}

    good = copy.deepcopy(section.partials[0])
    _, good_diagnostics = perturb_pfns_partial(good, deltas, bands, grids)

    widths = np.diff(grids[-1])
    assert widths[-1] / grids[-1][-1] < 1e-2, (
        "the top groups are no longer narrow relative to their energy, so this "
        "test has stopped being about anything"
    )
    assert good_diagnostics["max_group_mass_error"] < 1e-5


def test_empty_group_freezes_its_delta_and_the_sum_still_closes(synthetic):
    """A group with ``P⁰ = 0`` has no ratio, so its delta is dropped.

    Dropping it re-opens the sum the draw had closed, which is why the freeze
    happens *before* the projection rather than after it.
    """
    _, section, bands, grids = synthetic
    partial = copy.deepcopy(section.partials[0])

    band, node = 0, 0
    boundaries = np.asarray(grids[band], dtype=float).copy()
    # Push the lowest boundary pair below the table so its group integrates to 0.
    boundaries[0] = -2.0
    boundaries[1] = -1.0
    grids_local = list(grids)
    grids_local[band] = boundaries

    p0 = partial.group_integrals(node, boundaries)
    assert p0[0] == 0.0, "the fixture no longer produces an empty group"

    deltas = {b: np.zeros(g.size - 1) for b, g in enumerate(grids_local)}
    deltas[band] = np.full(boundaries.size - 1, 1e-4)

    partial, diagnostics = perturb_pfns_partial(
        partial, deltas, bands, grids_local,
    )

    node_report = diagnostics["per_node"][0]
    assert node_report["n_groups_frozen"] >= 1
    assert abs(node_report["sum_error_after_projection"]) < 1e-12
    assert diagnostics["max_renormalisation_error"] < 1e-3


def test_every_sample_is_normalised_and_positive(cf252):
    """The two properties a perturbed spectrum must have, over a real ensemble."""
    suite, section, bands, grids = cf252
    samples, _ = generate_pfns_samples(suite, 16, isotope="Cf252", seed=11,
                                       verbose=False)

    for index in range(16):
        partial = copy.deepcopy(section.partials[0])
        deltas = {b: samples[("Cf252", 18, b)][index] for b in range(len(bands))}
        partial, _ = perturb_pfns_partial(partial, deltas, bands, grids)

        for node in range(len(partial.incident_energies)):
            assert partial.normalisation(node) == pytest.approx(1.0, abs=1e-6)
            assert min(partial.chi[node]) >= 0.0


def test_the_table_grows_but_stays_inside_the_measured_bound(cf252):
    """NJOY ACER reads MF5 into fixed-size buffers, so growth is a real budget.

    Measured on Cf-252: 11 668 outgoing points become ~18 300, a factor of 1.6.
    The roadmap's estimate from the union of every boundary and shoulder was
    1.41; the difference is that a random draw steps at nearly every boundary.
    """
    suite, section, bands, grids = cf252
    samples, _ = generate_pfns_samples(suite, 4, isotope="Cf252", seed=7,
                                       verbose=False)
    partial = copy.deepcopy(section.partials[0])
    before = sum(len(g) for g in partial.outgoing_grids)

    deltas = {b: samples[("Cf252", 18, b)][0] for b in range(len(bands))}
    partial, _ = perturb_pfns_partial(partial, deltas, bands, grids)

    after = sum(len(g) for g in partial.outgoing_grids)
    assert 1.2 < after / before < 2.5


def test_max_outgoing_points_drops_the_smallest_steps_and_says_so(cf252):
    """The cap NJOY may force. It must never be silent about what it dropped."""
    suite, section, bands, grids = cf252
    samples, _ = generate_pfns_samples(suite, 4, isotope="Cf252", seed=7,
                                       verbose=False)
    deltas = {b: samples[("Cf252", 18, b)][0] for b in range(len(bands))}

    partial = copy.deepcopy(section.partials[0])
    partial, diagnostics = perturb_pfns_partial(
        partial, deltas, bands, grids, max_outgoing_points=400,
    )

    assert diagnostics["total_steps_dropped"] > 0
    assert max(len(g) for g in partial.outgoing_grids) <= 700
    for node in range(len(partial.incident_energies)):
        assert partial.normalisation(node) == pytest.approx(1.0, abs=1e-6)


def test_a_delta_vector_of_the_wrong_length_is_refused(synthetic):
    _, section, bands, grids = synthetic
    partial = copy.deepcopy(section.partials[0])
    deltas = zero_deltas(grids)
    deltas[0] = np.zeros(3)
    with pytest.raises(ValueError, match="3 deltas for 8 groups"):
        perturb_pfns_partial(partial, deltas, bands, grids)


# ---------------------------------------------------------------------------
# The real tapes
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("tape_fixture", ["u235_tape", "u235_b81_tape"])
def test_the_whole_path_holds_on_the_real_u235_tapes(request, tape_fixture):
    """Where the outgoing-grid mismatch is real and the bands differ in order.

    ENDF/B-VIII.1 U-235 is the harder case in every way: five bands of 84 and
    641 groups, an MF5 outgoing grid that varies between incident points, and
    an MF35 grid that matches neither.
    """
    path = request.getfixturevalue(tape_fixture)
    endf = read_endf(str(path), mf_numbers=[5, 35])
    suite, section, bands = build_pfns_covariance(endf, mt=18)
    grids = band_grids(suite)

    original = copy.deepcopy(section.partials[0])
    unperturbed = copy.deepcopy(section.partials[0])
    unperturbed, diagnostics = perturb_pfns_partial(
        unperturbed, zero_deltas(grids), bands, grids,
    )
    rng = np.random.default_rng(1)
    probe_incident = rng.uniform(bands[0][0], bands[-1][1], 40)
    probe_outgoing = rng.uniform(0.0, 2.0e7, 40)
    before = np.array([chi_at(original, e, probe_outgoing) for e in probe_incident])
    after = np.array([chi_at(unperturbed, e, probe_outgoing) for e in probe_incident])
    np.testing.assert_allclose(after, before, rtol=5e-14, atol=1e-300)

    samples, _ = generate_pfns_samples(suite, 4, isotope="U235", seed=5,
                                       verbose=False)
    partial = copy.deepcopy(section.partials[0])
    deltas = {b: samples[("U235", 18, b)][0] for b in range(len(bands))}
    partial, diagnostics = perturb_pfns_partial(partial, deltas, bands, grids)

    assert diagnostics["max_group_mass_error"] < 1e-4
    assert diagnostics["max_renormalisation_error"] < 1e-3
    for node in range(len(partial.incident_energies)):
        assert min(partial.chi[node]) >= 0.0
