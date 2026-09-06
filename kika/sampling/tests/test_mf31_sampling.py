"""Unit + integration tests for nu-bar (MF31) sampling.

Pure-logic tests (family sum rule, LNU=1 reconstruction) use synthetic MF1
sections and always run. End-to-end tests against the JEFF-4.0 U-235/Th-232/
Pu-241 tapes skip when those files are absent; the NJOY→ACE leg is exercised
only when an NJOY executable is available.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from kika.endf.classes.mf1.mf1mt452 import MF1MT452
from kika.endf.classes.mf1.mf1mt455 import MF1MT455
from kika.endf.classes.mf1.mf1mt456 import MF1MT456
from kika.sampling.mf31_sampling import (
    perturb_nubar_family,
    apply_factors_to_mf1_nubar,
    sum_rule_residual,
    _nubar_as_tabulated,
)

# The U-235 / Th-232 / Pu-241 tapes are resolved by the root conftest through
# $KIKA_TAPES; see the ``u235_tape`` / ``th232_tape`` / ``pu241_tape`` fixtures.

# Shared synthetic grid: 3 covariance bins.
BINS = [1.0e-5, 1.0e3, 1.0e6, 2.0e7]


# ---------------------------------------------------------------------------
# Synthetic MF1 builders (total = prompt + delayed exactly)
# ---------------------------------------------------------------------------

def _tab(cls, energies, nubar, **extra):
    e = list(map(float, energies))
    return cls(
        _za=92235.0, _awr=233.0, _mat=9228, _lnu=2,
        _energies=e, _nubar=list(map(float, nubar)),
        _interpolation=[(len(e), 2)], _nr=1, _np=len(e), **extra,
    )


def _family():
    # Realistic-ish grid: several nodes strictly interior to each of the 3 bins
    # ([1e-5,1e3), [1e3,1e6), [1e6,2e7)) so per-bin factors apply cleanly at the
    # query nodes (1.0e5 in bin 1, 5.0e6 in bin 2).
    e = [1.0e-5, 1.0e-1, 1.0e1, 1.0e3, 1.0e4, 1.0e5, 5.0e5,
         1.0e6, 2.0e6, 5.0e6, 1.0e7, 1.5e7, 2.0e7]
    prompt = [2.40 + 1.0e-7 * x for x in e]
    delayed = [0.0160 - 5.0e-10 * x for x in e]
    total = [p + d for p, d in zip(prompt, delayed)]
    return {
        452: _tab(MF1MT452, e, total),
        455: _tab(MF1MT455, e, delayed, _ldg=0, _nnf=0, _decay_constants=[]),
        456: _tab(MF1MT456, e, prompt),
    }


def _nu(sec, E):
    e, n = _nubar_as_tabulated(sec)
    return float(np.interp(E, e, n))


def _max_sumrule_err(out, n=50):
    errs = []
    for E in np.geomspace(1.0e-3, 1.9e7, n):
        errs.append(abs(_nu(out[452], E) - (_nu(out[455], E) + _nu(out[456], E))))
    return max(errs)


# ---------------------------------------------------------------------------
# Family sum rule — the three MF31 coverage branches
# ---------------------------------------------------------------------------

def test_family_prompt_only_derives_total_and_keeps_delayed():
    secs = _family()
    block = np.array([1.0, 1.05, 0.97])  # per-bin prompt factor
    out, diags = perturb_nubar_family(secs, {456: block}, BINS)

    assert _max_sumrule_err(out) < 1e-12
    # delayed untouched
    np.testing.assert_allclose(
        _nubar_as_tabulated(out[455])[1],
        np.interp(_nubar_as_tabulated(out[455])[0],
                  *(_nubar_as_tabulated(secs[455]))),
        atol=1e-12,
    )
    # prompt moved by the +5% middle-bin factor
    assert _nu(out[456], 1.0e5) / _nu(secs[456], 1.0e5) == pytest.approx(1.05, rel=1e-9)
    assert set(diags) == {456}


def test_family_total_only_scales_all_three_together():
    secs = _family()
    block = np.array([1.0, 1.04, 0.98])
    out, diags = perturb_nubar_family(secs, {452: block}, BINS)

    assert _max_sumrule_err(out) < 1e-12
    # all three share the same per-bin ratio at a mid-bin energy
    r452 = _nu(out[452], 1.0e5) / _nu(secs[452], 1.0e5)
    r455 = _nu(out[455], 1.0e5) / _nu(secs[455], 1.0e5)
    r456 = _nu(out[456], 1.0e5) / _nu(secs[456], 1.0e5)
    assert r452 == pytest.approx(1.04, rel=1e-9)
    assert r455 == pytest.approx(1.04, rel=1e-9)
    assert r456 == pytest.approx(1.04, rel=1e-9)


def test_family_all_three_independent_components_derive_total():
    secs = _family()
    fblocks = {
        455: np.array([1.0, 0.90, 1.10]),
        456: np.array([1.0, 1.03, 0.99]),
        452: np.array([1.0, 1.01, 1.00]),  # redundant — must be ignored
    }
    out, diags = perturb_nubar_family(secs, fblocks, BINS)

    assert _max_sumrule_err(out) < 1e-12
    # components follow their own factors; total is the (derived) sum
    assert _nu(out[455], 1.0e5) / _nu(secs[455], 1.0e5) == pytest.approx(0.90, rel=1e-9)
    assert _nu(out[456], 1.0e5) / _nu(secs[456], 1.0e5) == pytest.approx(1.03, rel=1e-9)
    assert _nu(out[452], 1.0e5) == pytest.approx(
        _nu(out[455], 1.0e5) + _nu(out[456], 1.0e5), rel=1e-12
    )
    assert set(diags) == {455, 456}  # 452 derived, not directly perturbed


def test_family_derived_component_via_override():
    # Mark delayed (455) as the derived redundant member.
    secs = _family()
    fblocks = {452: np.array([1.0, 1.02, 1.00]), 456: np.array([1.0, 1.01, 1.00])}
    out, _ = perturb_nubar_family(secs, fblocks, BINS, derived_mt=455)
    # 455 = 452 - 456 pointwise
    for E in (1.0e2, 1.0e5, 5.0e6):
        assert _nu(out[455], E) == pytest.approx(_nu(out[452], E) - _nu(out[456], E), abs=1e-12)


# ---------------------------------------------------------------------------
# LNU=1 polynomial reconstruction
# ---------------------------------------------------------------------------

def test_apply_lnu1_polynomial_reconstructed_to_tabulated():
    poly = MF1MT456(_za=92235.0, _awr=233.0, _mat=9228, _lnu=1,
                    _nc=2, _coefficients=[2.4, 1.0e-7])  # nu = 2.4 + 1e-7 E
    block = np.array([1.0, 1.10, 1.0])
    new, diag = apply_factors_to_mf1_nubar(poly, block, BINS)
    assert new.lnu == 2
    assert len(new._energies) >= len(BINS)
    # middle-bin (+10%): nu(2e5) baseline = 2.4 + 1e-7*2e5 = 2.42 → 2.662.
    # Query a bin-interior energy (away from the 1e6 edge ramp); the subdivided
    # reconstruction grid makes the per-bin factor clean to <1%.
    e, n = _nubar_as_tabulated(new)
    assert float(np.interp(2.0e5, e, n)) == pytest.approx(2.42 * 1.10, rel=1e-2)


# ---------------------------------------------------------------------------
# Integration: real JEFF-4.0 tapes
# ---------------------------------------------------------------------------

def test_build_from_object_matches_load_from_path(u235_tape):
    # The app endpoints call build_mf31_covariance() on an already-parsed object;
    # it must produce the same covariance as load_mf31_covariance() from a path.
    from kika.sampling.mf31_sampling import (
        load_mf31_covariance,
        build_mf31_covariance,
    )
    from kika.endf.parsers.parse_endf import parse_endf_file

    cov_p, _, grid_p, mts_p = load_mf31_covariance(str(u235_tape))
    cov_o, _, grid_o, mts_o = build_mf31_covariance(parse_endf_file(str(u235_tape)))
    assert mts_p == mts_o
    np.testing.assert_allclose(grid_p, grid_o)
    assert len(cov_p.matrices) == len(cov_o.matrices)
    for a, b in zip(cov_p.matrices, cov_o.matrices):
        np.testing.assert_allclose(a, b)


def test_load_mf31_u235_prompt_block_is_psd(u235_tape):
    from kika.sampling.mf31_sampling import load_mf31_covariance
    cov, secs, grid, mts = load_mf31_covariance(str(u235_tape))
    assert mts == [456]
    assert {452, 455, 456}.issubset(set(secs))
    C = cov.covariance_matrix
    assert C.shape == (22, 22)
    assert np.allclose(C, C.T)
    assert np.linalg.eigvalsh(C).min() > -1e-9
    d = np.sqrt(np.clip(np.diag(C), 0, None))
    assert 0.002 < d.max() < 0.02  # ~0.5–1% prompt nu-bar uncertainty


# The tape is picked per parameter with getfixturevalue, which the conftest's
# fixture-derived auto-marking cannot see at collection time — so mark by hand.
@pytest.mark.tape
@pytest.mark.parametrize("tape_fixture,expected_mts", [
    ("u235_tape", [456]),
    ("th232_tape", [452]),
    ("pu241_tape", [452, 455, 456]),
])
def test_end_to_end_write_satisfies_sum_rule(
    tape_fixture, expected_mts, tmp_path, request,
):
    tape = str(request.getfixturevalue(tape_fixture))
    from kika.sampling.nubar_perturbation import perturb_nubar_files
    from kika.endf.parsers.parse_endf import parse_endf_file

    summ = perturb_nubar_files(
        tape, num_samples=2, generate_ace=False,
        output_dir=str(tmp_path), seed=3, verbose_diagnostics=0,
    )
    iso = next(iter(summ["isotopes"].values()))
    assert iso["n_endf_ok"] == 2
    assert sorted(iso["mts_perturbed"]) == sorted(expected_mts)

    import glob
    pe = sorted(glob.glob(os.path.join(str(tmp_path), "endf", "*", "0001", "*")))[0]
    mf1 = parse_endf_file(pe).get_file(1)
    out = {mt: mf1.sections[mt] for mt in (452, 455, 456) if mt in mf1.sections}
    # sum rule holds to ENDF write precision (6 sig figs on nu-bar ~2–3)
    assert _max_sumrule_err(out) < 1e-4
    # perturbation actually took effect somewhere
    orig = parse_endf_file(tape).get_file(1)
    moved = abs(_nu(out[452], 2.0e6) / _nu(orig.sections[452], 2.0e6) - 1.0)
    assert moved > 1e-6


# ---------------------------------------------------------------------------
# Realised per-bin factor: does the written table express the factor we asked
# for? The block is piecewise constant per MF31 bin, but a duplicate-free
# lin-lin table can only approximate that step, so the answer is "not exactly"
# and these tests pin how close.
# ---------------------------------------------------------------------------

def _bin_average(e, n, lo, hi, m=400):
    """1/E-weighted average of the lin-lin table over [lo, hi]."""
    eq = np.geomspace(max(lo, 1e-11), hi, m)
    return float(np.trapezoid(np.interp(eq, e, n) / eq, eq)
                 / np.trapezoid(1.0 / eq, eq))


def _realised_factors(before, after, grid):
    """Per-bin realised factor (NaN where the baseline table does not reach)."""
    e0, n0 = _nubar_as_tabulated(before)
    e1, n1 = _nubar_as_tabulated(after)
    out = np.full(len(grid) - 1, np.nan)
    for g in range(len(grid) - 1):
        lo, hi = max(grid[g], e0[0]), min(grid[g + 1], e0[-1])
        if hi <= lo:
            continue
        out[g] = _bin_average(e1, n1, lo, hi) / _bin_average(e0, n0, lo, hi)
    return out


def test_top_bin_edge_point_is_perturbed():
    # A point sitting exactly on the top of the MF31 grid falls in the bin
    # *starting* there under the side='right' rule, i.e. outside coverage. For
    # a nu-bar table whose last point IS that edge, leaving it at factor 1.0
    # turns the last bin into a ramp back to the unperturbed value.
    from kika.sampling.mf33_sampling import perturb_pointwise_xs

    bins = np.array([1.0e-5, 1.0e3, 1.0e6, 2.0e7])
    block = np.array([1.1, 1.2, 1.3])
    e = np.array([1.0e-5, 1.0, 1.0e3, 1.0e5, 1.0e6, 1.0e7, 2.0e7])
    xs = np.ones_like(e)

    _, f_off, frac_off = perturb_pointwise_xs(e, xs, block, bins)
    _, f_on, frac_on = perturb_pointwise_xs(e, xs, block, bins, clamp_top_edge=True)

    # Default is unchanged — the MF33 → PENDF path must not move.
    assert f_off[-1] == 1.0 and frac_off > 0
    # Opt-in pulls the top point back into the last bin.
    assert f_on[-1] == pytest.approx(1.3)
    assert frac_on == 0.0
    np.testing.assert_allclose(f_off[:-1], f_on[:-1])


def test_sparse_table_realises_the_block_it_was_given():
    # 6 points across 15 bins is JEFF-4.0 Pu-241 MT456. With only bin edges
    # inserted, an alternating block cancels outright (realised factor ~1.000).
    grid = np.geomspace(1.0e-5, 2.0e7, 16)
    n_g = len(grid) - 1
    sec = _tab(MF1MT456, np.geomspace(1.0e-5, 2.0e7, 6), [2.9] * 6)
    block = 1.0 + 0.05 * ((-1.0) ** np.arange(n_g))

    new, _ = apply_factors_to_mf1_nubar(sec, block, grid)
    realised = _realised_factors(sec, new, grid)
    assert np.nanmax(np.abs(realised - block)) < 0.01


@pytest.mark.tape
@pytest.mark.parametrize("tape_fixture", ["u235_tape", "th232_tape", "pu241_tape"])
def test_realised_factors_track_the_block_on_real_tapes(tape_fixture, request):
    # Worst case for the edge ramp: neighbouring bins pulled opposite ways.
    from kika.sampling.mf31_sampling import load_mf31_covariance

    tape = str(request.getfixturevalue(tape_fixture))
    _cov, secs, grid, _mts = load_mf31_covariance(tape)
    grid = np.asarray(grid)
    block = 1.0 + 0.05 * ((-1.0) ** np.arange(len(grid) - 1))

    for mt, sec in secs.items():
        new, _ = apply_factors_to_mf1_nubar(sec, block, grid)
        realised = _realised_factors(sec, new, grid)
        worst = np.nanmax(np.abs(realised - block))
        assert worst < 0.01, f"MT{mt}: worst per-bin factor error {worst:.4f}"


@pytest.mark.tape
@pytest.mark.parametrize("tape_fixture", ["u235_tape", "th232_tape", "pu241_tape"])
def test_augmented_tables_stay_writable(tape_fixture, request):
    # The shoulder node must stay clear of the ~1e-6 relative resolution of an
    # ENDF float, or it is written as a duplicate energy — the one thing NJOY
    # ACER is documented not to tolerate here.
    from kika.sampling.mf31_sampling import load_mf31_covariance

    tape = str(request.getfixturevalue(tape_fixture))
    _cov, secs, grid, _mts = load_mf31_covariance(tape)
    grid = np.asarray(grid)
    block = np.full(len(grid) - 1, 1.02)

    for mt, sec in secs.items():
        new, _ = apply_factors_to_mf1_nubar(sec, block, grid)
        e = np.asarray(new.energies, dtype=float)
        d = np.diff(e)
        assert np.all(d > 0), f"MT{mt}: non-monotonic energies"
        assert (d / e[:-1]).min() > 1.0e-5, f"MT{mt}: nodes too close to write"


# ---------------------------------------------------------------------------
# Guards: cases that used to produce a plausible, wrong file in silence
# ---------------------------------------------------------------------------

def test_incomplete_family_does_not_overwrite_the_total():
    # A {452, 455} tape is malformed (the manual §1.3.2 makes 456 mandatory
    # whenever 455 is present). Deriving anyway wrote nu_total := nu_delayed.
    secs = _family()
    del secs[456]
    out, _ = perturb_nubar_family(secs, {452: np.array([1.0, 1.0, 1.0])}, BINS)
    assert _nu(out[452], 1.0e5) == pytest.approx(_nu(secs[452], 1.0e5), rel=1e-9)


def test_unperturbable_contributor_raises_instead_of_vanishing():
    # An LNU=1 component with no factor block has no table to sum; skipping it
    # used to drop that component out of the derived total entirely.
    secs = _family()
    secs[455] = MF1MT455(_za=92235.0, _awr=233.0, _mat=9228, _lnu=1, _ldg=0,
                         _nc=1, _coefficients=[0.016], _nnf=0, _decay_constants=[])
    with pytest.raises(ValueError, match="no usable nu-bar table"):
        perturb_nubar_family(secs, {456: np.array([1.0, 1.05, 1.0])}, BINS)


def test_no_nubar_sections_raises():
    with pytest.raises(ValueError, match="no MF1 nu-bar sections"):
        perturb_nubar_family({}, {456: np.ones(3)}, BINS)


def test_factor_block_without_a_section_raises():
    secs = _family()
    del secs[455]
    with pytest.raises(ValueError, match="no matching MF1"):
        perturb_nubar_family(secs, {455: np.ones(3)}, BINS)


def test_lnu1_delayed_section_can_be_perturbed():
    # MF1MT455 used to lack the ``coefficients`` property MT452/456 expose, so
    # a polynomial delayed nu-bar raised AttributeError — swallowed by the
    # per-sample error handler, which silently dropped the sample.
    poly = MF1MT455(_za=94241.0, _awr=239.0, _mat=9546, _lnu=1, _ldg=0,
                    _nc=2, _coefficients=[0.016, 1.0e-10],
                    _nnf=6, _decay_constants=[0.013] * 6)
    new, _ = apply_factors_to_mf1_nubar(poly, np.array([1.0, 1.1, 1.0]), BINS)
    assert new.lnu == 2 and len(new.energies) > len(BINS)
    # The decay-constant block survives the rewrite.
    assert new.decay_constants == poly.decay_constants


def test_lnu1_section_yields_central_values():
    # Without central values an absolute MF31 block cannot be made relative and
    # an NC LTY=0 sub-subsection cannot be resolved, so the covariance would be
    # dropped for exactly the sections that carry no table.
    from kika.sampling.mf31_sampling import _nubar_central_values

    poly = MF1MT456(_za=94241.0, _awr=239.0, _mat=9546, _lnu=1,
                    _nc=1, _coefficients=[2.9])

    class _MF1:
        sections = {456: poly}

    shim = _nubar_central_values(_MF1())
    assert 456 in shim
    np.testing.assert_allclose(shim[456].get_cross_section([1.0, 1.0e6]), 2.9)


# ---------------------------------------------------------------------------
# Sum-rule residual of the input file
# ---------------------------------------------------------------------------

def test_sum_rule_residual_is_zero_for_a_consistent_family():
    res = sum_rule_residual(_family(), BINS)
    assert res is not None
    assert res["max_bin_rel"] < 1e-12
    assert res["n_bins_uncovered"] == 0


def test_sum_rule_residual_ignores_bins_the_family_does_not_span():
    # JEFF-4.0 U-235 tabulates nu_d to 20 MeV and nu_p to 30 MeV. Scoring the
    # missing nu_d as a violation reported a 1e-3 discrepancy that was really
    # just absent data.
    secs = _family()
    e_short = [1.0e-5, 1.0e3, 1.0e6]
    secs[455] = _tab(MF1MT455, e_short, [0.016] * 3,
                     _ldg=0, _nnf=0, _decay_constants=[])
    res = sum_rule_residual(secs, BINS)
    assert res["n_bins_uncovered"] >= 1
    assert np.isnan(res["per_bin_rel"][-1])


@pytest.mark.tape
def test_sum_rule_residual_reported_in_summary(u235_tape, tmp_path):
    from kika.sampling.nubar_perturbation import perturb_nubar_files

    summ = perturb_nubar_files(
        str(u235_tape), num_samples=1, generate_ace=False,
        output_dir=str(tmp_path), seed=3, verbose_diagnostics=0, dry_run=True,
    )
    iso = next(iter(summ["isotopes"].values()))
    res = iso["sum_rule_residual"]
    # Measured on JEFF-4.0 U-235: 6.3e-4 at bin 10, 0.14 of the MF31 1σ there.
    assert res["max_bin_rel"] == pytest.approx(6.3e-4, rel=0.1)
    assert res["max_bin_over_sigma"] == pytest.approx(0.14, abs=0.02)
    assert res["n_bins_uncovered"] == 2
    # 452 is derived, so it is sampled but never applied.
    assert iso["derived_mt"] == 452
    assert iso["mts_perturbed"] == [456] and iso["mts_applied"] == [456]


@pytest.mark.tape
def test_parquet_metadata_flags_the_derived_mt(pu241_tape, tmp_path):
    # The parquet carries a column per (sampled MT, bin), including the derived
    # member's — MF31 gives it a covariance and the flatten order has to match.
    # Nothing downstream can tell from the columns alone that those factors were
    # discarded, so the fact has to survive the run — past the finalize step,
    # which deletes the parts directory the manifest is built in. Pu-241 is the
    # tape where all three MTs are sampled and only two are applied.
    import glob
    import json

    from kika.sampling.nubar_perturbation import perturb_nubar_files

    perturb_nubar_files(
        str(pu241_tape), num_samples=1, generate_ace=False,
        output_dir=str(tmp_path), seed=3, verbose_diagnostics=0, dry_run=True,
    )

    meta_file = glob.glob(os.path.join(
        str(tmp_path), "perturbation_matrix_*_metadata.json"))[0]
    with open(meta_file) as f:
        meta = json.load(f)

    assert 94241 in meta["isotopes_processed"]
    details = meta["isotope_details"]["94241"]
    assert details["derived_mt"] == 452
    assert sorted(details["mts_sampled"]) == [452, 455, 456]
    assert sorted(details["mts_applied"]) == [455, 456]
