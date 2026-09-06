"""Tests for the warn-only MF33 magnitude-channel diagnostics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.mf33_diagnostics import (  # noqa: E402
    FOUR_PI,
    fold_host_mf3_at_points,
    folded_c0_comparison_stats,
    between_experiment_c0_spread,
    load_mf33_sidecar,
)


# --- folding ---------------------------------------------------------------

def test_fold_flat_mf3_returns_constant():
    """A flat cross section folds to itself regardless of TOF resolution."""
    e_grid = np.linspace(0.1e6, 5.0e6, 200)
    xs = np.full_like(e_grid, 3.0)  # 3 barns everywhere
    folded = fold_host_mf3_at_points(
        e_grid, xs,
        subentries=["10037024", "99999999"],  # one absent → defaults
        energies_mev=[1.0, 2.5],
        tof_cache={},
    )
    assert folded.shape == (2,)
    np.testing.assert_allclose(folded, 3.0, rtol=1e-6)


def test_fold_delta_limit_interpolates():
    """With a zero-resolution kernel the fold is a plain interpolation."""
    e_grid = np.array([0.5e6, 1.0e6, 2.0e6, 4.0e6])
    xs = np.array([1.0, 2.0, 4.0, 8.0])
    # A subentry present in the cache with tiny time resolution → ~delta kernel.
    cache = {"AAA": {"energy_resolution_input": {
        "distance": {"value": 200.0}, "time_resolution": {"value": 1e-6}}}}
    folded = fold_host_mf3_at_points(
        e_grid, xs, subentries=["AAA"], energies_mev=[2.0], tof_cache=cache,
    )
    assert np.isclose(folded[0], 4.0, rtol=1e-3)


# --- comparison stats ------------------------------------------------------

def _cmp_df():
    # Two campaigns; DCS sits ~10% above folded host for campaign B.
    return pd.DataFrame({
        "campaign": ["A", "A", "B", "B"],
        "energy_mev": [1.0, 3.0, 1.5, 3.5],
        "sigma_folded_b": [2.0, 2.0, 2.0, 2.0],
        "sigma_el_dcs_b": [2.0, 2.0, 2.2, 2.2],
        "rel_sigma_dcs": [0.05, 0.05, 0.05, 0.05],
        "rel_sigma_host": [0.03, 0.03, 0.03, 0.03],
    })


def test_ratio_and_pull_values():
    stats = folded_c0_comparison_stats(_cmp_df(), leave_one_out=["A", "B"])
    ov = stats["overall"]
    assert ov["n"] == 4
    # ratios are [1,1,1.1,1.1] → median 1.05.
    assert np.isclose(ov["median_ratio"], 1.05)
    # mean |ratio-1| = (0+0+0.1+0.1)/4 = 0.05.
    assert np.isclose(ov["mean_abs_rel_diff"], 0.05)
    # DCS pull for the B points: (2.2-2.0)/(2.2*0.05) ≈ 1.818.
    assert stats["overall"]["pull_dcs_mean"] > 0


def test_regions_split_at_edge():
    stats = folded_c0_comparison_stats(_cmp_df(), region_edges_mev=(2.5,))
    assert set(stats["regions"]) == {"min-2.5MeV", "2.5-maxMeV"}
    # Below 2.5: points at 1.0 (ratio 1) and 1.5 (ratio 1.1).
    below = stats["regions"]["min-2.5MeV"]
    assert below["n"] == 2


def test_leave_one_out_excludes_campaign():
    stats = folded_c0_comparison_stats(_cmp_df(), leave_one_out=["B"])
    # Excluding B leaves only ratio-1 points → median ratio 1.0.
    assert np.isclose(stats["leave_one_out"]["excl_B"]["median_ratio"], 1.0)


def test_missing_host_rel_gives_nan_host_pull():
    df = _cmp_df().drop(columns=["rel_sigma_host"])
    stats = folded_c0_comparison_stats(df)
    assert np.isnan(stats["overall"]["pull_host_mean"])
    # DCS pulls unaffected.
    assert np.isfinite(stats["overall"]["pull_dcs_mean"])


# --- between-experiment spread ---------------------------------------------

def test_spread_flags_disagreement_beyond_manifest():
    # Bin 0: experiments agree (small spread, within priors).
    # Bin 1: large disagreement, priors small → flagged.
    per_exp = {
        0: {"exp1": 2.00, "exp2": 2.02, "exp3": 1.98},
        1: {"exp1": 2.0, "exp2": 2.6},
    }
    manifest = {"exp1": 0.03, "exp2": 0.03, "exp3": 0.03}
    out = between_experiment_c0_spread(per_exp, manifest)
    row0 = out[out["bin_key"] == 0].iloc[0]
    row1 = out[out["bin_key"] == 1].iloc[0]
    assert not bool(row0["exceeds"])
    assert bool(row1["exceeds"])
    assert row1["empirical_rel_std"] > row1["expected_rel_sys"]


def test_spread_skips_single_experiment_bins():
    per_exp = {0: {"exp1": 2.0}}  # only one experiment
    out = between_experiment_c0_spread(per_exp, {"exp1": 0.05})
    assert out.empty


# --- sidecar loader --------------------------------------------------------

def test_load_mf33_sidecar_roundtrip(tmp_path):
    grid = np.array([0.8e6, 1.0e6, 1.5e6])
    rel = np.array([[0.01, 0.005], [0.005, 0.02]])
    c0 = np.array([2.0, 2.5])
    np.save(tmp_path / "mf33_energy_grid_ev.npy", grid)
    np.save(tmp_path / "mf33_relative_covariance.npy", rel)
    np.save(tmp_path / "mf33_c0_nominal.npy", c0)
    lib = load_mf33_sidecar(str(tmp_path))
    np.testing.assert_array_equal(lib["mf33_grid_ev"], grid)
    np.testing.assert_array_equal(lib["mf33_rel_cov"], rel)
    np.testing.assert_array_equal(lib["mf33_c0_nominal"], c0)


# --- bin_average_xs (F2 host-MF3 projection) --------------------------------

from scripts.mf33_diagnostics import bin_average_xs  # noqa: E402


def test_bin_average_xs_flat():
    e = np.array([0.0, 10.0])
    xs = np.array([3.0, 3.0])
    grid = np.array([1.0, 4.0, 9.0])
    np.testing.assert_allclose(bin_average_xs(e, xs, grid), [3.0, 3.0])


def test_bin_average_xs_linear():
    """Linear xs(E) -> bin average equals the midpoint value."""
    e = np.array([0.0, 10.0])
    xs = np.array([0.0, 10.0])  # xs(E) = E
    grid = np.array([2.0, 4.0, 8.0])
    np.testing.assert_allclose(bin_average_xs(e, xs, grid), [3.0, 6.0], rtol=1e-12)


def test_bin_average_xs_catches_narrow_peak():
    """A native-grid spike inside a wide bin contributes its integral."""
    # Triangle peak of width 0.2 and height 100 centred at 5.0 on a flat 1.0.
    e = np.array([0.0, 4.9, 5.0, 5.1, 10.0])
    xs = np.array([1.0, 1.0, 101.0, 1.0, 1.0])
    grid = np.array([0.0, 10.0])
    # integral = 10*1.0 + 0.5*0.2*100 = 20 -> mean 2.0
    np.testing.assert_allclose(bin_average_xs(e, xs, grid), [2.0], rtol=1e-6)
