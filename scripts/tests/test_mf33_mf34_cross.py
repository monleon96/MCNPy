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


# ── The COMPLETE cross block: Cov(c0(E_i), a_l(E_j)) (roadmap §10.1.6) ───────
#
# The within-bin block above is a PARTIAL Level A, and shipping it against
# complete MF33/MF34 diagonals is not a valid covariance -- that is what made
# run 87 non-PSD, and §10.1.5 showed no representation work repairs it. These
# tests pin the complete form and, above all, the ONE COMMON REPLICA SET that
# makes it PSD. Pairwise-complete covariance (intersecting per entry, as the
# per-bin path does per bin) has no PSD guarantee and would silently reinstate
# the original defect.

def _paired_samples(n_s, n_bins, max_order, seed=0, drop=(), corr_len=2.0):
    """Correlated c0/a_l draws in the dict-of-dicts shape the pipeline uses.

    `drop` lists (s_idx, e_idx) pairs to remove from the SHAPE channel only,
    which is how an incomplete replica actually arises.

    `corr_len` makes the latent field SMOOTH IN ENERGY, and that is not
    decoration. With an iid latent (corr_len -> 0) both channels are
    energy-uncorrelated, the transitivity argument of roadmap §10.1.3 has
    nothing to bite on, and zeroing the cross-energy entries leaves the joint
    matrix PSD -- verified while writing these tests. The real MF33 and MF34 are
    strongly correlated across energy, which is exactly why the omission is
    fatal there. A test built on an iid latent would pass while testing nothing.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(n_bins)
    K = np.exp(-0.5 * ((idx[:, None] - idx[None, :]) / corr_len) ** 2)
    Lk = np.linalg.cholesky(K + 1e-10 * np.eye(n_bins))
    latent = rng.normal(size=(n_s, n_bins)) @ Lk.T
    c0, alls = {}, {}
    for s in range(n_s):
        c0[s] = {e: 0.2 + 0.01 * latent[s, e] for e in range(n_bins)}
        alls[s] = {}
        for e in range(n_bins):
            if (s, e) in drop:
                continue
            base = 0.5 * latent[s, e] + 0.5 * rng.normal(size=max_order)
            alls[s][e] = 0.01 * base
    return alls, c0


def test_full_block_has_the_within_bin_block_on_its_diagonal():
    """Same estimator, so the diagonal must agree when nothing is dropped."""
    n_s, n_bins, L = 400, 6, 3
    alls, c0 = _paired_samples(n_s, n_bins, L, seed=1)
    out = compute_mf33_mf34_cross(alls, c0, range(n_bins), L)
    full = out["cov_full"]
    assert out["n_common"] == n_s
    np.testing.assert_allclose(
        np.einsum("iil->il", full), out["cov"], rtol=1e-10, atol=0,
    )


def test_full_block_is_psd_as_a_joint_covariance():
    """THE property the whole change exists for.

    Assemble [[C33, Cx], [Cx^T, C34]] from the same replicas and require PSD.
    This is what the within-bin-only block fails.
    """
    n_s, n_bins, L = 500, 5, 3
    alls, c0 = _paired_samples(n_s, n_bins, L, seed=2)
    out = compute_mf33_mf34_cross(alls, c0, range(n_bins), L)
    full = out["cov_full"]

    C = np.array([[c0[s][e] for e in range(n_bins)] for s in range(n_s)])
    A = np.array([[alls[s][e] for e in range(n_bins)] for s in range(n_s)])
    X = np.hstack([C, A.reshape(n_s, n_bins * L)])
    Xc = X - X.mean(0)
    S = (Xc.T @ Xc) / (n_s - 1)

    # Rebuild the magnitude<->shape corner from cov_full and require it to match
    # the direct sample covariance -- i.e. cov_full IS that corner.
    corner = np.empty((n_bins, n_bins * L))
    for e in range(n_bins):
        for f in range(n_bins):
            corner[e, f * L:(f + 1) * L] = full[e, f, :]
    np.testing.assert_allclose(corner, S[:n_bins, n_bins:], rtol=1e-10, atol=1e-18)
    assert np.linalg.eigvalsh(S)[0] > -1e-10 * np.max(np.abs(np.diag(S)))


def test_zeroing_the_cross_energy_entries_destroys_psd():
    """The measured claim of §10.1.6, pinned as a unit test: the discarded
    entries are load-bearing, so a within-bin-only block is not a covariance."""
    n_s, n_bins, L = 500, 5, 3
    alls, c0 = _paired_samples(n_s, n_bins, L, seed=3)
    C = np.array([[c0[s][e] for e in range(n_bins)] for s in range(n_s)])
    A = np.array([[alls[s][e] for e in range(n_bins)] for s in range(n_s)])
    X = np.hstack([C, A.reshape(n_s, n_bins * L)])
    Xc = X - X.mean(0)
    S = (Xc.T @ Xc) / (n_s - 1)
    scale = float(np.max(np.abs(np.diag(S))))
    assert np.linalg.eigvalsh(S)[0] > -1e-10 * scale          # full: PSD

    S2 = S.copy()
    blk = S[:n_bins, n_bins:].copy()
    keep = np.zeros_like(blk, dtype=bool)
    for e in range(n_bins):
        keep[e, e * L:(e + 1) * L] = True
    S2[:n_bins, n_bins:] = np.where(keep, blk, 0.0)
    S2[n_bins:, :n_bins] = S2[:n_bins, n_bins:].T
    assert np.linalg.eigvalsh(S2)[0] < -1e-8 * scale          # within-bin: NOT PSD


def test_incomplete_replicas_are_excluded_not_pairwise_intersected():
    """A replica missing ONE bin must leave the common set entirely, or the
    block loses its PSD guarantee."""
    n_s, n_bins, L = 300, 4, 2
    alls, c0 = _paired_samples(n_s, n_bins, L, seed=4, drop={(0, 2), (1, 3)})
    out = compute_mf33_mf34_cross(alls, c0, range(n_bins), L)
    assert out["n_common"] == n_s - 2
    # the per-bin path keeps its own per-bin intersection, so it sees more
    assert out["n_pairs"][0] == n_s


def test_full_block_is_withheld_when_too_few_replicas_are_complete():
    """A partial block is exactly what we are trying to stop shipping."""
    n_s, n_bins, L = 40, 3, 2
    alls, c0 = _paired_samples(n_s, n_bins, L, seed=5)
    out = compute_mf33_mf34_cross(alls, c0, range(n_bins), L, min_samples=100)
    assert out["cov_full"] is None
    assert out["n_common"] == n_s


def test_compute_full_false_keeps_the_legacy_return():
    """Runs 86/87 reproducibility: the per-bin outputs must not move."""
    n_s, n_bins, L = 200, 4, 3
    alls, c0 = _paired_samples(n_s, n_bins, L, seed=6)
    a = compute_mf33_mf34_cross(alls, c0, range(n_bins), L, compute_full=False)
    b = compute_mf33_mf34_cross(alls, c0, range(n_bins), L, compute_full=True)
    assert "cov_full" not in a
    for k in ("cov", "rho", "std_c0", "std_a", "n_pairs"):
        np.testing.assert_array_equal(a[k], b[k])
