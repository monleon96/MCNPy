"""Cov(c0, a_l) over shared MC replicas must measure real correlation only.

WHY THIS FILE EXISTS. Sigma_eval ships with its MF33<->MF34 cross block set to
exactly zero, because the evaluation splits one Monte Carlo into two
one-at-a-time channels (c0 at frozen shape -> MF33; a_l at frozen c0 -> MF34).
Roadmap §9.4.1 bounded what that omission can cost -- |rho| <= 1, so coverage on
`no_Cierjacks` lies in [41.3, 60.9] % -- but a bound cannot give the SIGN, and
the sign decides whether a cross block would raise or lower our uncertainty.

`compute_mf33_mf34_cross` measures it, exploiting the fact that both channels
read the same `Y_perturbed` row per replica (resample_AD.py: one matrix,
`_batch_mc_ridge_solve` for a_l and `fixed_shape_c0_scale` for c0).

The failure mode these tests exist for is a FALSE READING, in either direction:
manufacturing correlation by mispairing replicas, or reporting a confident zero
where the quantity is undefined. Nothing here asserts a physical rho.
"""
import numpy as np
import pytest

from scripts.exfor_to_endf_research import compute_mf33_mf34_cross

MAX_ORDER = 3
N_SAMPLES = 400
BINS = [0, 1, 2]


def _build(rho_target, n=N_SAMPLES, seed=0, bins=BINS):
    """Replicas where a_1 carries a known correlation with c0."""
    rng = np.random.default_rng(seed)
    all_s, c0_s = {}, {}
    for s in range(n):
        z = rng.normal()
        w = rng.normal()
        c0 = 1.0 + 0.1 * z
        # a_1 correlated with c0 by construction; a_2, a_3 independent.
        a1 = 0.5 + 0.1 * (rho_target * z + np.sqrt(max(0.0, 1 - rho_target**2)) * w)
        all_s[s] = {b: np.array([a1, 0.2 + 0.05 * rng.normal(),
                                 0.1 + 0.05 * rng.normal()]) for b in bins}
        c0_s[s] = {b: c0 for b in bins}
    return all_s, c0_s


def test_recovers_a_known_positive_correlation():
    out = compute_mf33_mf34_cross(*_build(0.8), BINS, MAX_ORDER)
    assert out["rho"][:, 0] == pytest.approx(0.8, abs=0.12)


def test_recovers_a_known_negative_correlation():
    """The sign is the entire point -- §9.4.1's bound could not supply it."""
    out = compute_mf33_mf34_cross(*_build(-0.8), BINS, MAX_ORDER)
    assert out["rho"][:, 0] == pytest.approx(-0.8, abs=0.12)


def test_independent_orders_read_as_uncorrelated():
    out = compute_mf33_mf34_cross(*_build(0.8), BINS, MAX_ORDER)
    assert np.abs(out["rho"][:, 1]) == pytest.approx(0.0, abs=0.12)
    assert np.abs(out["rho"][:, 2]) == pytest.approx(0.0, abs=0.12)


def test_rho_is_bounded():
    """A |rho| > 1 would mean the estimator, not the physics, is broken."""
    out = compute_mf33_mf34_cross(*_build(0.95, seed=7), BINS, MAX_ORDER)
    r = out["rho"][np.isfinite(out["rho"])]
    assert np.all(np.abs(r) <= 1.0 + 1e-9)


def test_mispaired_replicas_do_not_manufacture_correlation():
    """Guards the pairing itself -- the one assumption the measurement rests on.

    Shuffling the c0 replica labels destroys the pairing while preserving both
    marginals. A rho that survives that would be an artefact, and would have
    been reported as physics.
    """
    all_s, c0_s = _build(0.9)
    keys = list(c0_s)
    shuffled = {k: c0_s[v] for k, v in zip(keys, np.random.default_rng(1).permutation(keys))}
    intact = compute_mf33_mf34_cross(all_s, c0_s, BINS, MAX_ORDER)
    broken = compute_mf33_mf34_cross(all_s, shuffled, BINS, MAX_ORDER)
    assert np.abs(intact["rho"][:, 0]) == pytest.approx(0.9, abs=0.12)
    assert np.abs(broken["rho"][:, 0]) == pytest.approx(0.0, abs=0.15)


def test_frozen_order_gives_nan_not_zero():
    """Orders restored from nominal have no spread. rho is undefined there.

    Reporting 0.0 would read as 'measured, and uncorrelated' -- the opposite of
    the truth, and it would bias any median taken over orders.
    """
    all_s, c0_s = _build(0.8)
    for s in all_s:
        for b in BINS:
            all_s[s][b][2] = 0.1  # a_3 frozen at nominal in every replica
    out = compute_mf33_mf34_cross(all_s, c0_s, BINS, MAX_ORDER)
    assert np.all(np.isnan(out["rho"][:, 2]))
    assert np.all(np.isfinite(out["rho"][:, 0]))


def test_too_few_paired_replicas_yields_nan():
    out = compute_mf33_mf34_cross(*_build(0.8, n=10), BINS, MAX_ORDER, min_samples=32)
    assert np.all(np.isnan(out["rho"]))
    assert np.all(out["n_pairs"] == 10)


def test_only_replicas_present_in_both_channels_are_paired():
    """A bin can fail in one channel and not the other."""
    all_s, c0_s = _build(0.8)
    for s in list(c0_s)[:100]:
        del c0_s[s][BINS[0]]
    out = compute_mf33_mf34_cross(all_s, c0_s, BINS, MAX_ORDER)
    assert out["n_pairs"][0] == N_SAMPLES - 100
    assert out["n_pairs"][1] == N_SAMPLES
    assert out["rho"][0, 0] == pytest.approx(0.8, abs=0.15)


def test_shapes_and_bin_order_follow_energy_indices():
    out = compute_mf33_mf34_cross(*_build(0.5), [2, 0, 1], MAX_ORDER)
    assert out["cov"].shape == (3, MAX_ORDER)
    assert out["rho"].shape == (3, MAX_ORDER)
    assert out["n_pairs"].shape == (3,)
    assert out["std_a"].shape == (3, MAX_ORDER)
