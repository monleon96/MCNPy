"""Tests for CrossSectionCovariance.transfer_reactions (covariance merge).

Covers the merge semantics used by the app's "Merge" covariance tool:
diagonal + cross-correlation block transfer, the "both-present" cross rule,
replace-on-conflict, energy-grid validation (refuse mismatches), eV/MeV
normalisation, and the is_relative positional-alignment guard. A final
data-file test exercises the real nubar -> boxer merge when the sample
covariance files are available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kika.cov.cross_section_covariance import CrossSectionCovariance, TransferResult

U5 = 92235
_GRID = [1.0e-5, 1.0, 100.0, 1.0e4, 2.0e7]  # 5 boundaries -> 4 groups


def _diag(g: int, val: float) -> np.ndarray:
    return np.eye(g) * val


def _make_source(unit: str = "eV") -> CrossSectionCovariance:
    """Two nubar self-blocks (452, 455), their cross block, and cross-sections."""
    grid = _GRID if unit == "eV" else [e * 1e-6 for e in _GRID]
    cov = CrossSectionCovariance(num_groups=4, energy_grid=list(grid), energy_unit=unit)
    cov.add_matrix(U5, 452, U5, 452, _diag(4, 0.04), is_relative=True)  # unc 0.2
    cov.add_matrix(U5, 455, U5, 455, _diag(4, 0.09), is_relative=True)  # unc 0.3
    cov.add_matrix(U5, 452, U5, 455, np.full((4, 4), 0.01), is_relative=True)
    cov.cross_sections[(U5, 452)] = np.array([1.0, 2.0, 3.0, 4.0])
    cov.cross_sections[(U5, 455)] = np.array([5.0, 6.0, 7.0, 8.0])
    return cov


def _make_dest() -> CrossSectionCovariance:
    """Reaction self-blocks (MT2, MT18), no nubar."""
    cov = CrossSectionCovariance(num_groups=4, energy_grid=list(_GRID), energy_unit="eV")
    cov.add_matrix(U5, 2, U5, 2, _diag(4, 0.01), is_relative=True)   # unc 0.1
    cov.add_matrix(U5, 18, U5, 18, _diag(4, 0.16), is_relative=True)  # unc 0.4
    return cov


def test_full_transfer_brings_diagonals_cross_and_xs():
    src, dst = _make_source(), _make_dest()
    res = dst.transfer_reactions(src, [(U5, 452), (U5, 455)])

    assert isinstance(res, TransferResult)
    assert set(res.diagonal_transferred) == {(U5, 452), (U5, 455)}
    assert (U5, 452, U5, 455) in res.cross_transferred
    assert res.cross_dropped == []
    assert set(res.cross_sections_transferred) == {(U5, 452), (U5, 455)}
    assert res.reactions_replaced == []
    assert res.missing_in_source == []

    m = res.covariance
    assert set(m.reactions_by_isotope(U5)) == {2, 18, 452, 455}
    assert np.allclose(m.get_uncertainty(U5, 452), 0.2)
    assert np.allclose(m.get_uncertainty(U5, 455), 0.3)
    # Inputs are untouched.
    assert set(dst.reactions_by_isotope(U5)) == {2, 18}
    assert set(src.reactions_by_isotope(U5)) == {452, 455}
    # Destination's own reactions are unchanged.
    assert np.allclose(m.get_uncertainty(U5, 18), 0.4)


def test_single_reaction_drops_cross_with_absent_partner():
    src, dst = _make_source(), _make_dest()
    res = dst.transfer_reactions(src, [(U5, 452)])

    assert res.diagonal_transferred == [(U5, 452)]
    assert res.cross_transferred == []
    dropped = {block for block, _ in res.cross_dropped}
    assert (U5, 452, U5, 455) in dropped
    reason = next(r for b, r in res.cross_dropped if b == (U5, 452, U5, 455))
    assert "455" in reason


def test_diagonal_only_mode_skips_all_cross_blocks():
    src, dst = _make_source(), _make_dest()
    res = dst.transfer_reactions(src, [(U5, 452), (U5, 455)], cross_correlation="diagonal-only")
    assert set(res.diagonal_transferred) == {(U5, 452), (U5, 455)}
    assert res.cross_transferred == []
    assert any("diagonal-only" in r for _, r in res.cross_dropped)


def test_always_mode_keeps_cross_even_without_partner():
    src, dst = _make_source(), _make_dest()
    res = dst.transfer_reactions(src, [(U5, 452)], cross_correlation="always")
    assert (U5, 452, U5, 455) in res.cross_transferred
    assert res.cross_dropped == []


def test_replace_removes_existing_block_then_re_adds_source_data():
    src, dst = _make_source(), _make_dest()
    dst.add_matrix(U5, 452, U5, 452, np.zeros((4, 4)), is_relative=True)  # stale placeholder

    res = dst.transfer_reactions(src, [(U5, 452)])
    assert res.reactions_replaced == [(U5, 452)]

    m = res.covariance
    count_452 = sum(
        1 for a, b, c, d in zip(m.isotope_rows, m.reaction_rows, m.isotope_cols, m.reaction_cols)
        if (a, b, c, d) == (U5, 452, U5, 452)
    )
    assert count_452 == 1
    assert np.allclose(m.get_uncertainty(U5, 452), 0.2)  # source data, not the zeros


def test_grid_mismatch_raises():
    src = _make_source()
    other_grid = list(np.linspace(1e-5, 2e7, 9))  # 8 groups
    dst8 = CrossSectionCovariance(num_groups=8, energy_grid=other_grid, energy_unit="eV")
    dst8.add_matrix(U5, 2, U5, 2, _diag(8, 0.01), is_relative=True)
    with pytest.raises(ValueError):
        dst8.transfer_reactions(src, [(U5, 452)])


def test_ev_mev_normalisation_allows_same_physical_grid():
    src_mev = _make_source(unit="MeV")  # physically identical grid, MeV units
    dst = _make_dest()                  # eV
    res = dst.transfer_reactions(src_mev, [(U5, 452)])
    assert (U5, 452) in res.diagonal_transferred
    assert np.allclose(res.covariance.get_uncertainty(U5, 452), 0.2)


def test_is_relative_alignment_with_empty_dest_flags():
    """A dest with no is_relative flags (like BOXER) must keep its blocks relative
    even when an *absolute* source block is appended after them."""
    dst = CrossSectionCovariance(num_groups=4, energy_grid=list(_GRID), energy_unit="eV")
    for mt, val in [(2, 0.01), (18, 0.16)]:  # append manually, leaving is_relative empty
        dst.isotope_rows.append(U5); dst.reaction_rows.append(mt)
        dst.isotope_cols.append(U5); dst.reaction_cols.append(mt)
        dst.matrices.append(_diag(4, val))
    assert dst.is_relative == []

    src_abs = CrossSectionCovariance(num_groups=4, energy_grid=list(_GRID), energy_unit="eV")
    src_abs.add_matrix(U5, 452, U5, 452, _diag(4, 0.04), is_relative=False)  # absolute
    src_abs.cross_sections[(U5, 452)] = np.full(4, 2.0)

    m = dst.transfer_reactions(src_abs, [(U5, 452)]).covariance
    # Dest's MT18 stays relative: unc = sqrt(0.16) = 0.4 (NOT divided by any xs).
    assert np.allclose(m.get_uncertainty(U5, 18), 0.4)
    # The absolute 452 block: unc = sqrt(0.04)/|xs| = 0.2 / 2 = 0.1.
    assert np.allclose(m.get_uncertainty(U5, 452), 0.1)


def test_missing_in_source_is_reported():
    src, dst = _make_source(), _make_dest()
    res = dst.transfer_reactions(src, [(U5, 452), (U5, 999)])
    assert (U5, 999) in res.missing_in_source
    assert (U5, 452) in res.diagonal_transferred


def test_invalid_cross_correlation_mode_raises():
    src, dst = _make_source(), _make_dest()
    with pytest.raises(ValueError):
        dst.transfer_reactions(src, [(U5, 452)], cross_correlation="bogus")


# ── Real sample-file test ─────────────────────────────────────────────────────

# The two sample files are resolved by the root conftest; see the
# ``u5_nubar_covfil_tape`` / ``u5_boxer_tape`` fixtures.
def test_real_nubar_into_boxer(u5_nubar_covfil_tape, u5_boxer_tape):
    from kika.sampling.utils import load_covariance

    src = load_covariance(str(u5_nubar_covfil_tape))  # COVFIL, nubar 452/455/456
    dst = load_covariance(str(u5_boxer_tape))         # BOXER, reactions + lumped 851/852
    assert np.allclose(np.asarray(src.energy_grid, float), np.asarray(dst.energy_grid, float))

    res = dst.transfer_reactions(src, [(U5, 452), (U5, 455), (U5, 456)])
    m = res.covariance

    assert set(res.diagonal_transferred) == {(U5, 452), (U5, 455), (U5, 456)}
    assert res.cross_dropped == []
    merged_mts = set(m.reactions_by_isotope(U5))
    assert {452, 455, 456}.issubset(merged_mts)              # transferred
    assert {1, 2, 18, 102, 851, 852}.issubset(merged_mts)    # original boxer reactions kept
    assert np.allclose(
        np.asarray(src.get_uncertainty(U5, 452)), np.asarray(m.get_uncertainty(U5, 452))
    )
