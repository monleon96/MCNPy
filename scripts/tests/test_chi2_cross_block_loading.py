"""The MF33<->MF34 cross sidecars must reach Sigma_eval in the right currency.

WHY THIS FILE EXISTS. `compute_mf33_mf34_cross` measures Cov(c0, a_l) in
ABSOLUTE units. `build_mf33_mf34_cross_block` consumes a block whose magnitude
axis is RELATIVE (its magnitude sensitivity is `y_eval`, i.e. dy/d(dsigma/sigma))
and whose shape axis is absolute unless the block says otherwise. Nothing in
either signature enforces the conversion between the two, so a wrong denominator
would produce a plausible, silently mis-scaled cross term rather than an error.

The specific trap: the shipped relative MF33 is `absolute / (c0_host x c0_host)`,
NOT `c0_nominal` -- our own c0 sits ~9.5 % above the host's and the pipeline
recentres onto the host. Dividing the cross block by c0_nominal instead would
scale it by 1.095 against its own diagonals, breaking Cauchy-Schwarz consistency
with them while looking entirely reasonable in the log.

These tests pin the conversion, the zero-by-default contract that keeps runs
82-86 reproducible, and the "undefined is not zero" accounting.
"""
import numpy as np
import pytest

from scripts.eval_covariance import build_mf33_mf34_cross_block
from scripts.precompute_chi2_predictive import (
    _min_eig_lanczos,
    load_mf33_mf34_cross,
)

N_BINS = 5
L_MAX = 3


@pytest.fixture()
def run_dir(tmp_path):
    """A minimal pipeline-output directory carrying the three sidecars."""
    rng = np.random.default_rng(0)
    cov = rng.normal(scale=1e-3, size=(N_BINS, 6))
    grid = np.linspace(0.85e6, 4.0e6, N_BINS + 1)
    c0_host = np.full(N_BINS, 0.2)
    np.save(tmp_path / "mf33_mf34_cross_covariance.npy", cov)
    np.save(tmp_path / "mf33_energy_grid_ev.npy", grid)
    np.save(tmp_path / "mf33_c0_host.npy", c0_host)
    return tmp_path, cov, grid, c0_host


def test_magnitude_axis_is_divided_by_c0_host(run_dir):
    """The one conversion that cannot be checked by reading the log."""
    d, cov, _grid, c0_host = run_dir
    blocks, _ = load_mf33_mf34_cross(str(d), l_max=L_MAX)
    for blk in blocks:
        L = blk["l"]
        np.testing.assert_allclose(
            np.diag(blk["matrix"]), cov[:, L - 1] / c0_host, rtol=0, atol=0,
        )


def test_shape_axis_stays_absolute(run_dir):
    """is_relative=False, or the builder would multiply by a_l and divide twice.

    The relative form also divides by a_l_nom, which passes through zero.
    """
    d, *_ = run_dir
    blocks, _ = load_mf33_mf34_cross(str(d), l_max=L_MAX)
    assert all(blk["is_relative"] is False for blk in blocks)


def test_cross_energy_entries_are_zero_not_invented(run_dir):
    """Only within-bin Cov(c0(E), a_l(E)) was ever sampled."""
    d, *_ = run_dir
    blocks, _ = load_mf33_mf34_cross(str(d), l_max=L_MAX)
    for blk in blocks:
        m = blk["matrix"]
        assert m.shape == (N_BINS, N_BINS)
        off = m - np.diag(np.diag(m))
        assert np.all(off == 0.0)


def test_l_max_truncates_the_block_list(run_dir):
    d, *_ = run_dir
    blocks, info = load_mf33_mf34_cross(str(d), l_max=L_MAX)
    assert [b["l"] for b in blocks] == [1, 2, 3]
    assert info["n_orders"] == 3


def test_scale_is_linear_and_zero_kills_the_term(run_dir):
    """Damping must be a plain multiplier, so a PSD failure can be bracketed."""
    d, *_ = run_dir
    base, _ = load_mf33_mf34_cross(str(d), l_max=L_MAX, scale=1.0)
    half, _ = load_mf33_mf34_cross(str(d), l_max=L_MAX, scale=0.5)
    zero, _ = load_mf33_mf34_cross(str(d), l_max=L_MAX, scale=0.0)
    for b, h, z in zip(base, half, zero):
        np.testing.assert_allclose(h["matrix"], 0.5 * b["matrix"])
        assert np.all(z["matrix"] == 0.0)


def test_undefined_slots_are_zeroed_and_counted(run_dir):
    """NaN = 'the MC never determined this', not 'the correlation is zero'.

    Zeroing is the only representable choice, but it must be reported, or a
    restored-from-nominal order reads as a measured zero.
    """
    d, cov, _grid, _c0 = run_dir
    cov = cov.copy()
    cov[0, 0] = np.nan
    cov[2, 1] = np.nan
    np.save(d / "mf33_mf34_cross_covariance.npy", cov)

    blocks, info = load_mf33_mf34_cross(str(d), l_max=L_MAX)
    assert info["n_undefined_slots"] == 2
    assert blocks[0]["matrix"][0, 0] == 0.0
    assert blocks[1]["matrix"][2, 2] == 0.0
    assert np.all(np.isfinite(np.concatenate([b["matrix"].ravel() for b in blocks])))


def test_undefined_is_counted_from_rho_not_from_cov(run_dir):
    """The real sidecars disagree, and rho is the one that knows.

    On run 86, 36 slots have rho = NaN (orders restored from nominal, zero
    spread) while cov at those slots is ~1e-31 — finite. Counting non-finite
    entries in cov would report 0 undefined and claim the MC measured every
    slot.
    """
    d, cov, *_ = run_dir
    rho = np.ones_like(cov)
    rho[1, 0] = np.nan
    rho[3, 2] = np.nan
    rho[4, 4] = np.nan  # order 5: outside l_max, must not be counted
    np.save(d / "mf33_mf34_cross_correlation.npy", rho)

    _blocks, info = load_mf33_mf34_cross(str(d), l_max=L_MAX)
    assert info["n_undefined_slots"] == 2


def test_grid_mismatch_is_a_hard_error(run_dir):
    """Sidecars from different runs must not be silently mixed."""
    d, cov, grid, _c0 = run_dir
    np.save(d / "mf33_energy_grid_ev.npy", grid[:-1])
    with pytest.raises(SystemExit):
        load_mf33_mf34_cross(str(d), l_max=L_MAX)


def test_missing_sidecar_names_the_flag_that_produces_it(run_dir):
    d, *_ = run_dir
    (d / "mf33_mf34_cross_covariance.npy").unlink()
    with pytest.raises(SystemExit) as e:
        load_mf33_mf34_cross(str(d), l_max=L_MAX)
    assert "COMPUTE_MF33_MF34_CROSS" in str(e.value)


def test_loaded_blocks_are_accepted_by_the_builder_and_move_sigma(run_dir):
    """End-to-end: the loader's output is what the builder consumes.

    Guards the layout contract between the two — a renamed key would otherwise
    surface as a silently zero cross term, which is exactly today's behaviour
    and therefore invisible.
    """
    d, _cov, grid, _c0 = run_dir
    blocks, _ = load_mf33_mf34_cross(str(d), l_max=L_MAX)

    n = 12
    e_mev = np.linspace(0.9, 3.9, n)
    mu = np.linspace(-0.9, 0.9, n)
    c0 = np.full(n, 0.2)
    a_l = np.tile(np.array([0.4, 0.2, 0.05]), (n, 1))
    y = c0 * (1.0 + a_l[:, 0])

    zero = build_mf33_mf34_cross_block(None, e_mev, mu, c0, a_l, y)
    got = build_mf33_mf34_cross_block(blocks, e_mev, mu, c0, a_l, y,
                                      mf33_grid_ev=grid,
                                      energies_mf4_mev=grid / 1e6,
                                      a_is_relative=False)

    assert zero.shape == got.shape == (n, n)
    assert np.all(zero == 0.0)
    assert not np.allclose(got, 0.0), "cross block reached the builder as zero"
    np.testing.assert_allclose(got, got.T, rtol=1e-12, atol=1e-18)


def test_scaling_the_block_scales_the_assembled_sigma(run_dir):
    """Sigma_cross is linear in the block, so damping is interpretable."""
    d, _cov, grid, _c0 = run_dir
    n = 8
    e_mev = np.linspace(0.9, 3.9, n)
    mu = np.linspace(-0.8, 0.8, n)
    c0 = np.full(n, 0.2)
    a_l = np.tile(np.array([0.4, 0.2, 0.05]), (n, 1))
    y = c0 * (1.0 + a_l[:, 0])

    b1, _ = load_mf33_mf34_cross(str(d), l_max=L_MAX, scale=1.0)
    b2, _ = load_mf33_mf34_cross(str(d), l_max=L_MAX, scale=0.25)
    kw = dict(mf33_grid_ev=grid, energies_mf4_mev=grid / 1e6,
              a_is_relative=False)
    s1 = build_mf33_mf34_cross_block(b1, e_mev, mu, c0, a_l, y, **kw)
    s2 = build_mf33_mf34_cross_block(b2, e_mev, mu, c0, a_l, y, **kw)
    np.testing.assert_allclose(s2, 0.25 * s1, rtol=1e-12, atol=1e-20)


# ── The PSD accounting the damping decision rests on ──────────────────────────

def _spd_with_known_min(n, lam_min, seed=0):
    """Symmetric matrix with an exactly known smallest eigenvalue."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    ev = np.linspace(lam_min, 40.0, n)
    A = (Q * ev) @ Q.T
    return ((A + A.T) / 2).astype(np.float32)


@pytest.mark.parametrize("lam_min", [-0.7, 1e-3])
def test_lanczos_finds_the_smallest_eigenvalue(lam_min):
    """It has to be right on the NEGATIVE case — that is the one it exists for.

    Cierjacks is 28631 x 28631, so `eigvalsh` is not an option and the diagonal
    alone cannot answer PSD-ness. If this estimate were wrong we would adopt a
    damping scale on a matrix that is still indefinite.
    """
    A = _spd_with_known_min(600, lam_min)
    exact = float(np.linalg.eigvalsh(A.astype(np.float64)).min())
    got = _min_eig_lanczos(A)
    assert got is not None
    assert abs(got - exact) <= 1e-4 * max(abs(exact), 1.0)


def test_lanczos_returns_none_instead_of_raising():
    """A diagnostic must never be the reason a 12-hour job dies."""
    assert _min_eig_lanczos(np.zeros((0, 0), dtype=np.float32)) is None


def test_sigma_eval_is_linear_in_the_scale():
    """The whole damping argument rests on this, so it is pinned rather than assumed.

    Sigma_eval(s) = Sigma^MF33 + Sigma^MF34 + s*Sigma^cross. If the cross block
    were not exactly linear in the scale, one run would not determine s_max and
    the scan would have to be blind.
    """
    rng = np.random.default_rng(3)
    n_bins, l_max, n = 6, 3, 10
    cov = rng.normal(scale=1e-3, size=(n_bins, l_max))
    grid = np.linspace(0.85e6, 4.0e6, n_bins + 1)
    e_mev = np.linspace(0.9, 3.9, n)
    mu = np.linspace(-0.9, 0.9, n)
    c0 = np.full(n, 0.2)
    a_l = np.tile(np.array([0.4, 0.2, 0.05]), (n, 1))
    y = c0 * (1.0 + a_l[:, 0])

    def blocks(scale):
        return [{"l": L, "shape_grid_ev": grid,
                 "matrix": np.diag(scale * cov[:, L - 1]), "is_relative": False}
                for L in range(1, l_max + 1)]

    kw = dict(mf33_grid_ev=grid, energies_mf4_mev=grid / 1e6,
              a_is_relative=False)
    base = build_mf33_mf34_cross_block(blocks(1.0), e_mev, mu, c0, a_l, y, **kw)
    for s in (0.0, 0.25, 0.5, 2.0):
        got = build_mf33_mf34_cross_block(blocks(s), e_mev, mu, c0, a_l, y, **kw)
        np.testing.assert_allclose(got, s * base, rtol=1e-12, atol=1e-20)


# ── The COMPLETE cross block (roadmap §10.1.6) ───────────────────────────────
#
# Run 87 shipped a within-bin-only cross block against complete MF33/MF34
# diagonals and came out non-PSD; §10.1.5 proved a consistent collapse cannot
# fix that, because collapse never creates cross-energy structure. The repair is
# to measure the block completely. These tests pin the two halves of it: the
# loader must PREFER the full sidecar, and the producer must build it from ONE
# common replica set -- pairwise-complete covariance carries no PSD guarantee,
# which is the exact bug being fixed.

@pytest.fixture()
def run_dir_full(run_dir):
    """`run_dir` plus a full (n_bins, n_bins, L) cross sidecar."""
    d, cov, grid, c0_host = run_dir
    rng = np.random.default_rng(7)
    full = rng.normal(scale=1e-3, size=(N_BINS, N_BINS, 6))
    for L in range(6):                      # keep the diagonal consistent
        full[np.arange(N_BINS), np.arange(N_BINS), L] = cov[:, L]
    np.save(d / "mf33_mf34_cross_covariance_full.npy", full)
    return d, cov, grid, c0_host, full


def test_full_sidecar_is_preferred_over_the_within_bin_one(run_dir_full):
    d, _cov, _grid, c0_host, full = run_dir_full
    blocks, info = load_mf33_mf34_cross(str(d), l_max=L_MAX)
    assert info["form"] == "full"
    for blk in blocks:
        L = blk["l"]
        np.testing.assert_allclose(
            blk["matrix"], full[:, :, L - 1] / c0_host[:, None], rtol=0, atol=0,
        )


def test_full_block_keeps_its_off_diagonal_structure(run_dir_full):
    """The whole point: cross-energy entries must survive to Sigma_eval."""
    d, *_ = run_dir_full
    blocks, _ = load_mf33_mf34_cross(str(d), l_max=L_MAX)
    for blk in blocks:
        off = blk["matrix"][~np.eye(N_BINS, dtype=bool)]
        assert np.count_nonzero(off) > 0, "cross-energy structure was flattened"


def test_full_block_is_not_symmetrised(run_dir_full):
    """Row is magnitude, column is shape -- they are different quantities, so
    symmetrising would silently average two unrelated covariances."""
    d, *_ = run_dir_full
    blocks, _ = load_mf33_mf34_cross(str(d), l_max=L_MAX)
    m = blocks[0]["matrix"]
    assert not np.allclose(m, m.T), "block was symmetrised somewhere"


def test_a_mismatched_full_sidecar_is_rejected_not_broadcast(run_dir):
    d, *_ = run_dir
    np.save(d / "mf33_mf34_cross_covariance_full.npy",
            np.zeros((N_BINS + 1, N_BINS + 1, 6)))
    with pytest.raises(SystemExit, match="Full cross block has shape"):
        load_mf33_mf34_cross(str(d), l_max=L_MAX)


def test_missing_full_sidecar_warns_and_still_loads(run_dir):
    """Runs 86/87 must stay reproducible -- but noisily."""
    d, *_ = run_dir
    with pytest.warns(RuntimeWarning, match="WITHIN-BIN-ONLY"):
        _blocks, info = load_mf33_mf34_cross(str(d), l_max=L_MAX)
    assert info["form"] == "within_bin_only"


# ---------------------------------------------------------------------------
# The group-space form (roadmap §10.1.8-J)
# ---------------------------------------------------------------------------

N_MAG, N_SHAPE = 4, 9


@pytest.fixture()
def group_run_dir(run_dir):
    """A run dir that ALSO carries the group-space sidecars.

    Built on top of `run_dir` on purpose: the fine sidecars stay present, so
    these tests pin that the group form *wins* rather than merely working when
    it is the only option.
    """
    d, _cov, grid, _c0h = run_dir
    rng = np.random.default_rng(7)
    grp = rng.normal(scale=1e-3, size=(N_MAG, N_SHAPE, 6))
    mag_ev = np.linspace(grid[0], grid[-1], N_MAG + 1)
    shape_ev = np.linspace(grid[0], grid[-1], N_SHAPE + 1)
    np.save(d / "mf33_mf34_cross_group_covariance.npy", grp)
    np.save(d / "mf33_mf34_cross_group_mag_grid_ev.npy", mag_ev)
    np.save(d / "mf33_mf34_cross_group_shape_grid_ev.npy", shape_ev)
    return d, grp, mag_ev, shape_ev


def test_group_form_wins_over_the_fine_sidecars(group_run_dir):
    """Runs 87/88's fine block is non-PSD; if both exist the group one must win."""
    d, _grp, _m, _s = group_run_dir
    _blocks, info = load_mf33_mf34_cross(str(d), l_max=L_MAX)
    assert info["form"] == "group"


def test_group_block_is_not_divided_by_c0_host_again(group_run_dir):
    """It is built against the shipped RELATIVE MF33, so it is already relative.

    Dividing by c0_host a second time is the exact failure this pins: it would
    rescale the term by 1/0.2 here and look perfectly plausible downstream.
    """
    d, grp, _m, _s = group_run_dir
    blocks, _ = load_mf33_mf34_cross(str(d), l_max=L_MAX)
    for blk in blocks:
        np.testing.assert_allclose(
            blk["matrix"], grp[:, :, blk["l"] - 1], rtol=0, atol=0,
        )


def test_group_block_carries_its_shape_grid_but_no_magnitude_grid(group_run_dir):
    """The shape axis is still the run's own adaptive grid and still independent
    of the magnitude axis — §10.7-2: the two families do NOT have to share a
    grid, only each family with itself.

    ⚑ What changed: the block no longer carries a magnitude grid of its own.
    Since §10.7-7 the magnitude axis IS the shipped MF33 grid, passed to the
    fold once, because that is the only way `Sigma_eval = M J M^T` is a
    congruence. A block that disagrees is caught by row count, not re-binned.
    """
    d, grp, _mag_ev, shape_ev = group_run_dir
    blocks, info = load_mf33_mf34_cross(str(d), l_max=L_MAX)
    assert info["n_bins"] == N_MAG
    for blk in blocks:
        assert "mag_grid_ev" not in blk
        np.testing.assert_array_equal(blk["shape_grid_ev"], shape_ev)
        assert blk["matrix"].shape == (N_MAG, N_SHAPE)
        assert blk["is_relative"] is False


def test_a_group_block_against_the_fine_mf33_is_refused(group_run_dir):
    """⚑ THE GUARD, on the real artefact. This is run 89/90's configuration:
    a cross term certified on the 188-group magnitude axis, folded against the
    fine MF33 the file actually ships. It produced four runs and no χ².

    It is now a loud error instead of an indefinite Σ_eval.
    """
    d, _grp, _mag_ev, _shape_ev = group_run_dir
    blocks, _ = load_mf33_mf34_cross(str(d), l_max=L_MAX)
    fine_ev = np.linspace(0.85e6, 4.0e6, N_MAG * 7 + 1)   # the shipped grid
    with pytest.raises(ValueError, match="magnitude bins"):
        build_mf33_mf34_cross_block(
            blocks, np.linspace(0.9, 3.9, 12), np.linspace(-0.9, 0.9, 12),
            np.full(12, 0.2), np.tile(np.array([0.3, 0.1, 0.05]), (12, 1)),
            np.full(12, 0.2),
            mf33_grid_ev=fine_ev, energies_mf4_mev=fine_ev / 1e6,
            a_is_relative=False,
        )


def test_group_scale_is_applied(group_run_dir):
    d, grp, _m, _s = group_run_dir
    b1, _ = load_mf33_mf34_cross(str(d), l_max=L_MAX, scale=1.0)
    b2, _ = load_mf33_mf34_cross(str(d), l_max=L_MAX, scale=0.25)
    for x, y in zip(b1, b2):
        np.testing.assert_allclose(y["matrix"], 0.25 * x["matrix"])


def test_group_block_without_its_grids_is_refused(group_run_dir):
    """A block whose axes are unknown is worse than no block at all."""
    d, _grp, _m, _s = group_run_dir
    (d / "mf33_mf34_cross_group_shape_grid_ev.npy").unlink()
    with pytest.raises(SystemExit, match="group_shape_grid"):
        load_mf33_mf34_cross(str(d), l_max=L_MAX)


def test_group_block_reaches_sigma_eval_with_both_axes_binned(group_run_dir):
    """End-to-end: two different grids must bin independently, not collapse.

    With 4 magnitude groups and 9 shape groups over the same energy span, points
    in one magnitude group span several shape groups. If the builder used one
    grid for both, Sigma_eval would be piecewise-constant in blocks it should
    resolve -- so assert the assembled matrix actually varies within a magnitude
    group.
    """
    d, _grp, mag_ev, _s = group_run_dir
    blocks, _ = load_mf33_mf34_cross(str(d), l_max=L_MAX)
    e_mev = np.linspace(0.9, 3.9, 12)
    mu = np.linspace(-0.9, 0.9, 12)
    c0 = np.full(12, 0.2)
    a_l = np.tile(np.array([0.3, 0.1, 0.05]), (12, 1))
    # Folded against the MF33 grid the block was BUILT on, which is the only
    # configuration the fold now accepts — the axes still bin independently.
    sigma = build_mf33_mf34_cross_block(
        blocks, e_mev, mu, c0, a_l, y_eval=np.full(12, 0.2),
        mf33_grid_ev=mag_ev, energies_mf4_mev=mag_ev / 1e6,
        a_is_relative=False,
    )
    assert sigma.shape == (12, 12)
    np.testing.assert_allclose(sigma, sigma.T, rtol=0, atol=1e-18)
    assert np.count_nonzero(sigma) > 0
    # Two points inside the SAME magnitude group but different shape groups must
    # not receive identical rows.
    assert not np.allclose(sigma[0], sigma[1])


# ── §L8 inverted: the file is the source, so the sidecar must be absent ───────

def test_the_file_and_the_sidecar_cannot_both_be_the_source():
    """⚑ The no-double-counting property, restated for item 5.

    It used to be structural: `build_mf34_block` skips `l_r < 1`, so the a_0
    blocks were invisible and the sidecar was necessarily the only source. That
    skip is still there, but the a_0 reader now writes the SAME
    `mf33_mf34_cross` key the sidecar loader writes -- so with both enabled the
    magnitude<->shape correlation is folded at 2x and nothing downstream can
    tell. Hence the check, and hence a test for it.
    """
    from scripts.precompute_chi2_predictive import refuse_double_cross_source

    with pytest.raises(SystemExit, match="TWICE"):
        refuse_double_cross_source(True, "/some/run/dir")

    # Each alone, and neither, are all fine.
    refuse_double_cross_source(True, "")
    refuse_double_cross_source(False, "/some/run/dir")
    refuse_double_cross_source(False, "")


@pytest.mark.parametrize("value, want", [
    ("", False), ("0", False), ("false", False), ("False", False),
    ("1", True), ("yes", True),
])
def test_the_from_file_switch_reads_off_the_environment(monkeypatch, value, want):
    """Off by default and off for the spellings of 'no', because a truthy
    "0" would turn the cross term on in every run that tried to disable it."""
    import importlib
    import scripts.precompute_chi2_predictive as p

    monkeypatch.setenv("KIKA_MF33_MF34_CROSS_FROM_FILE", value)
    mod = importlib.reload(p)
    try:
        assert mod.MF33_MF34_CROSS_FROM_FILE is want
    finally:
        monkeypatch.delenv("KIKA_MF33_MF34_CROSS_FROM_FILE")
        importlib.reload(p)
