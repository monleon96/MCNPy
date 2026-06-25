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
    _nubar_as_tabulated,
)

JEFF40_DIR = "/share_snc/snc/JuanMonleon/jeff40-endf"
U235 = os.path.join(JEFF40_DIR, "92-U-235g.txt")
TH232 = os.path.join(JEFF40_DIR, "90-Th-232g.txt")
PU241 = os.path.join(JEFF40_DIR, "94-Pu-241g.txt")

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

@pytest.mark.skipif(not os.path.exists(U235), reason="U-235 JEFF-4.0 tape not available")
def test_build_from_object_matches_load_from_path():
    # The app endpoints call build_mf31_covariance() on an already-parsed object;
    # it must produce the same covariance as load_mf31_covariance() from a path.
    from kika.sampling.mf31_sampling import (
        load_mf31_covariance,
        build_mf31_covariance,
    )
    from kika.endf.parsers.parse_endf import parse_endf_file

    cov_p, _, grid_p, mts_p = load_mf31_covariance(U235)
    cov_o, _, grid_o, mts_o = build_mf31_covariance(parse_endf_file(U235))
    assert mts_p == mts_o
    np.testing.assert_allclose(grid_p, grid_o)
    assert len(cov_p.matrices) == len(cov_o.matrices)
    for a, b in zip(cov_p.matrices, cov_o.matrices):
        np.testing.assert_allclose(a, b)


@pytest.mark.skipif(not os.path.exists(U235), reason="U-235 JEFF-4.0 tape not available")
def test_load_mf31_u235_prompt_block_is_psd():
    from kika.sampling.mf31_sampling import load_mf31_covariance
    cov, secs, grid, mts = load_mf31_covariance(U235)
    assert mts == [456]
    assert {452, 455, 456}.issubset(set(secs))
    C = cov.covariance_matrix
    assert C.shape == (22, 22)
    assert np.allclose(C, C.T)
    assert np.linalg.eigvalsh(C).min() > -1e-9
    d = np.sqrt(np.clip(np.diag(C), 0, None))
    assert 0.002 < d.max() < 0.02  # ~0.5–1% prompt nu-bar uncertainty


@pytest.mark.parametrize("tape,expected_mts", [
    (U235, [456]),
    (TH232, [452]),
    (PU241, [452, 455, 456]),
])
def test_end_to_end_write_satisfies_sum_rule(tape, expected_mts, tmp_path):
    if not os.path.exists(tape):
        pytest.skip(f"{tape} not available")
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
