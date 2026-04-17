"""Tests for MF33 NC-type LTY=0 (derived/sum-rule) covariance resolution.

Covers two levels:

* **Synthetic fixture** — builds two MF33MT objects representing MT101
  (a "total-like" reaction) and MT102 (a contributor to MT101), then
  resolves a third section MT103 defined as ``MT101 − MT102`` via the
  sum rule. Since the inputs are controlled, the bilinear propagation
  can be checked element-wise.

* **Fe-56 JENDL-5 regression** — the original bug was that
  ``resolve_nc_lty0`` returned a diagonal-only matrix. This test only
  asserts that the resolved covariance is symmetric with non-zero
  off-diagonals and that the output grid is the union of the
  contributors' MF33 grids (no fine subdivision).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika.endf.classes.mf33.mf33 import (
    MF33MT,
    NCSubSubsection,
    NISubSubsectionRecord,
    Subsection,
)
from kika.endf.classes.mf3.mf3mt import MF3MT


# ---------------------------------------------------------------------------
# Helpers for building synthetic MF33 / MF3 fixtures
# ---------------------------------------------------------------------------

def _make_mf3(mt: int, energies, xs_values) -> MF3MT:
    """Build a minimal MF3MT carrying a pointwise σ(E) tabulation."""
    sec = MF3MT(number=mt)
    sec._za = 26056.0
    sec._awr = 55.454
    sec._energies = list(map(float, energies))
    sec._cross_sections = list(map(float, xs_values))
    sec._np = len(sec._energies)
    sec._nr = 1
    sec._interpolation = [(sec._np, 2)]  # lin-lin
    return sec


def _make_mf33_self_cov(mt: int, energies, diag_values) -> MF33MT:
    """Build a minimal MF33MT with a single LB=5 LS=1 symmetric matrix.

    The matrix is diagonal (independent bins) — nothing to do with the
    NC-rewrite; this is just the simplest way to have well-defined
    covariances for contributing MTs inside the synthetic fixture.
    """
    n = len(diag_values)
    assert len(energies) == n + 1, "diag_values must be NE-1 long"
    ni = NISubSubsectionRecord(
        ls=1, lb=5, nt=n + n * (n + 1) // 2, ne=n + 1,
        energies=list(map(float, energies)),
        matrix=[],
    )
    # LS=1 means triu stored; diagonal entries at triu positions
    triu_vals = []
    for i in range(n):
        for j in range(i, n):
            triu_vals.append(float(diag_values[i]) if i == j else 0.0)
    ni.matrix = triu_vals
    subsec = Subsection(
        xmf1=0.0, xlfs1=0.0, mat1=0, mt1=mt, nc=0, ni=1,
        ni_records=[ni],
    )
    sec = MF33MT(number=mt)
    sec._za = 26056.0
    sec._awr = 55.454
    sec._mtl = 0
    sec._nl = 1
    sec._subsections = [subsec]
    return sec


def _make_mf33_derived(mt: int, contrib_coeffs, e_lo: float, e_hi: float) -> MF33MT:
    """Build an MF33MT containing only an NC LTY=0 derivation record.

    ``contrib_coeffs`` is a sequence of ``(coeff, mt_index)`` pairs.
    """
    nci = len(contrib_coeffs)
    nc = NCSubSubsection(
        lty=0, e1=e_lo, e2=e_hi, nci=nci,
        ci=[float(c) for c, _ in contrib_coeffs],
        xmti=[float(m) for _, m in contrib_coeffs],
    )
    subsec = Subsection(
        xmf1=0.0, xlfs1=0.0, mat1=0, mt1=mt, nc=1, ni=0,
        nc_records=[nc],
    )
    sec = MF33MT(number=mt)
    sec._za = 26056.0
    sec._awr = 55.454
    sec._mtl = 0
    sec._nl = 1
    sec._subsections = [subsec]
    return sec


# ---------------------------------------------------------------------------
# Synthetic fixture: MT103 = MT101 − MT102
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_sumrule():
    """Two contributors and one derived MT on a shared 4-bin grid.

    Relative variances are specified directly on the 4 bins; cross
    sections are smooth (linear) so bin-averaged σ is well behaved.
    """
    edges = [1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7]  # 4 bins
    # Relative variances (diagonal of LB=5 relative cov for each MT)
    var_mt101 = [0.01, 0.02, 0.03, 0.04]
    var_mt102 = [0.005, 0.010, 0.015, 0.020]

    # Pointwise σ on a grid that straddles every bin
    e_xs = np.geomspace(edges[0], edges[-1], 50)
    xs_mt101 = 10.0 + 0.0 * e_xs          # σ = 10 b (flat)
    xs_mt102 = 2.0 + 0.0 * e_xs           # σ = 2 b (flat)
    xs_mt103 = xs_mt101 - xs_mt102        # = 8 b (by sum rule)

    mf3 = {
        101: _make_mf3(101, e_xs, xs_mt101),
        102: _make_mf3(102, e_xs, xs_mt102),
        103: _make_mf3(103, e_xs, xs_mt103),
    }
    mf33 = {
        101: _make_mf33_self_cov(101, edges, var_mt101),
        102: _make_mf33_self_cov(102, edges, var_mt102),
        103: _make_mf33_derived(103, [(1.0, 101), (-1.0, 102)],
                                edges[0], edges[-1]),
    }
    return {
        "edges": edges,
        "var_mt101": np.array(var_mt101),
        "var_mt102": np.array(var_mt102),
        "xs_mt101": 10.0,
        "xs_mt102": 2.0,
        "xs_mt103": 8.0,
        "mf3": mf3,
        "mf33": mf33,
    }


def test_sumrule_diagonal(synthetic_sumrule):
    """Diagonal must follow: c_a²·Var(a)·σ_a² + c_b²·Var(b)·σ_b² / σ_d²."""
    fx = synthetic_sumrule
    cov = fx["mf33"][103].to_xs_covmat(
        sibling_sections=fx["mf33"], mf3_sections=fx["mf3"],
    )
    M = np.asarray(cov.matrices[0])
    diag = np.diag(M)

    # Expected per-bin relative variance with σ_a=10, σ_b=2, σ_d=8:
    #   Var_abs = (+1)² · v_a · 10² + (−1)² · v_b · 2²
    #   Var_rel = Var_abs / 8²
    expected = (fx["var_mt101"] * 10.0 ** 2
                + fx["var_mt102"] * 2.0 ** 2) / 8.0 ** 2
    assert np.allclose(diag, expected, rtol=1e-6), (diag, expected)


def test_sumrule_offdiagonal_zero_when_inputs_diagonal(synthetic_sumrule):
    """Contributors have no off-diagonal structure → derived should too."""
    fx = synthetic_sumrule
    cov = fx["mf33"][103].to_xs_covmat(
        sibling_sections=fx["mf33"], mf3_sections=fx["mf3"],
    )
    M = np.asarray(cov.matrices[0])
    off = M - np.diag(np.diag(M))
    assert np.allclose(off, 0.0, atol=1e-12)


def test_sumrule_offdiagonal_propagates():
    """When a contributor has off-diagonal structure, the derived MT inherits it.

    Builds MT101 with a non-trivial correlated LB=5 matrix, then checks
    that MT103 = MT101 − MT102 carries that same off-diagonal pattern
    (scaled by σ_a²/σ_d²).
    """
    edges = [1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7]
    n = 4
    # LB=5 LS=1 correlated matrix for MT101
    corr = np.array([
        [0.01, 0.005, 0.002, 0.001],
        [0.005, 0.02, 0.008, 0.003],
        [0.002, 0.008, 0.03, 0.012],
        [0.001, 0.003, 0.012, 0.04],
    ])
    triu = []
    for i in range(n):
        for j in range(i, n):
            triu.append(float(corr[i, j]))
    ni = NISubSubsectionRecord(
        ls=1, lb=5, nt=n + n * (n + 1) // 2, ne=n + 1,
        energies=list(map(float, edges)),
        matrix=triu,
    )
    sec101 = MF33MT(number=101)
    sec101._za, sec101._awr, sec101._mtl, sec101._nl = 26056.0, 55.454, 0, 1
    sec101._subsections = [Subsection(
        xmf1=0.0, xlfs1=0.0, mat1=0, mt1=101, nc=0, ni=1, ni_records=[ni],
    )]

    sec102 = _make_mf33_self_cov(102, edges, [0.0] * n)
    sec103 = _make_mf33_derived(103, [(1.0, 101), (-1.0, 102)],
                                edges[0], edges[-1])

    e_xs = np.geomspace(edges[0], edges[-1], 50)
    mf3 = {
        101: _make_mf3(101, e_xs, [10.0] * 50),
        102: _make_mf3(102, e_xs, [2.0] * 50),
        103: _make_mf3(103, e_xs, [8.0] * 50),
    }
    mf33 = {101: sec101, 102: sec102, 103: sec103}

    cov = sec103.to_xs_covmat(sibling_sections=mf33, mf3_sections=mf3)
    M = np.asarray(cov.matrices[0])

    # Expected relative cov: c_a² · corr · σ_a² / σ_d² = 1 · corr · 100/64
    expected = corr * (10.0 ** 2) / (8.0 ** 2)
    assert np.allclose(M, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Fe-56 JENDL-5 regression (opt-in — requires the 64 MB ENDF file)
# ---------------------------------------------------------------------------

FE56 = Path(__file__).resolve().parents[1] / "files" / "endf" / "Fe56_jendl5_n.endf"


@pytest.mark.skipif(not FE56.is_file(), reason="Fe56 JENDL-5 file not present")
def test_fe56_mt2_is_full_matrix_not_diagonal():
    """The pre-fix behaviour returned ``np.diag(...)`` — guard against regression."""
    from kika.processing import resolve_derived_covariance

    cov = resolve_derived_covariance(str(FE56), mt=2)
    M = np.asarray(cov.matrices[0])

    # Shape + symmetry
    assert M.ndim == 2 and M.shape[0] == M.shape[1]
    assert np.allclose(M, M.T, atol=1e-12)

    # Regression guard: off-diagonals must exist
    off = M - np.diag(np.diag(M))
    assert np.any(np.abs(off) > 1e-20), "MT2 covariance is diagonal-only (regression)"

    # Diagonal must be non-negative
    assert np.all(np.diag(M) >= -1e-12)

    # Off-diagonal magnitude must be comparable to (not overwhelmed by)
    # the diagonal — a weak sanity check that sign/scale are sensible.
    assert np.abs(off).max() < 10 * np.abs(np.diag(M)).max()


@pytest.mark.skipif(not FE56.is_file(), reason="Fe56 JENDL-5 file not present")
def test_fe56_mt2_output_grid_is_union_not_subdivided():
    """The rewrite drops the ~10/decade fine subdivision — output must match
    the union of the contributing MTs' coarse MF33 grids."""
    from kika.endf.read_endf import read_endf
    from kika.processing import resolve_derived_covariance

    endf = read_endf(str(FE56))
    mf33 = endf.files[33].sections

    # Expected union: every bin edge that appears in any contributor's own
    # MF33 grid, plus edges from MT2's own NI range-extension records.
    mt2 = mf33[2]
    nc = next(nc for ss in mt2.subsections for nc in ss.nc_records
              if nc.lty == 0)
    contrib_mts = [int(round(x)) for x in nc.xmti]

    expected_union: set[float] = set()
    for mt in contrib_mts:
        if mt not in mf33:
            continue
        for ss in mf33[mt].subsections:
            if ss.mt1 != mt:
                continue
            for ni in ss.ni_records:
                if ni.lb == 5:
                    expected_union.update(ni.energies)
                elif ni.lb in (0, 1, 2, 8, 9):
                    expected_union.update(ni.e_table_k)
                elif ni.lb in (3, 4):
                    expected_union.update(ni.e_table_k)
                    expected_union.update(ni.e_table_l)
                elif ni.lb == 6:
                    expected_union.update(ni.row_energies)
                    expected_union.update(ni.col_energies)

    # Also grid points from MT2's own NI (range-extension)
    for ss in mt2.subsections:
        for ni in ss.ni_records:
            if ni.lb in (0, 1, 2, 8, 9):
                expected_union.update(ni.e_table_k)

    cov = resolve_derived_covariance(str(FE56), mt=2)
    got = set(cov.energy_grids[0])

    # The native grid should be a subset of what we'd accumulate from all
    # contributors — it's exact if our extractor visits the same sources.
    # Use "subset or equal" rather than strict equality since obscure LB
    # variants in less-common contributors could add extras we didn't list.
    missing = expected_union - got
    assert not missing, f"Output grid is missing union points: {sorted(missing)[:5]}..."
    # Grid must be much coarser than the pre-fix ~10/decade subdivision.
    # Union grid should be ≤ 500 bins for a file of this shape.
    assert len(cov.energy_grids[0]) < 500
