"""Tests for the opt-in MF34 (L=0, L1) sigma↔a_l cross-block writing.

Covers: structure (LTT=3 upper triangle with a null (0,0) block), round-trip
through ``parse_mf34_mt``, byte-identical shape blocks vs the default path, the
warn-only zero-crossing guard, and input validation.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from kika.endf.writers.mf34_writer import create_mf34_from_covariance
from kika.endf.parsers.parse_mf34 import parse_mf34_mt


ZA, AWR, MAT, MT = 26056.0, 55.454, 2631, 2
MAX_ORDER = 2
SHAPE_GRID = np.array([0.85e6, 1.5e6, 2.5e6, 4.0e6])  # N = 3
CROSS_GRID = np.array([0.85e6, 2.5e6, 4.0e6])          # N0 = 2 (coarse)
N, N0 = 3, 2


def _shape_cov() -> np.ndarray:
    base = np.arange(1, N * MAX_ORDER + 1, dtype=float)
    cov = 0.0005 * np.outer(base, base) + np.diag(0.01 * base)
    return 0.5 * (cov + cov.T)


def _cross() -> dict:
    return {1: 0.01 * np.ones((N0, N)), 2: -0.005 * np.ones((N0, N))}


def _find(section, l, l1):
    for ss in section.subsections[0].sub_subsections:
        if ss.l == l and ss.l1 == l1:
            return ss.records[0]
    return None


def test_cross_structure_and_ltt():
    mf34 = create_mf34_from_covariance(
        _shape_cov(), SHAPE_GRID, MAX_ORDER, ZA, AWR, MAT, MT,
        ltt=1, cross_cov=_cross(), cross_energy_grid_ev=CROSS_GRID,
    )
    # "LTT=3 if either L or L1=0 anywhere in the Section" (ENDF-6 Sec. 34.2).
    assert mf34._ltt == 3
    pairs = [(ss.l, ss.l1, ss.records[0].lb)
             for ss in mf34._subsections[0].sub_subsections]
    assert pairs == [(0, 0, 5), (0, 1, 6), (0, 2, 6), (1, 1, 5), (1, 2, 6), (2, 2, 5)]


def test_declared_nl_is_the_coefficient_count_not_the_max_index():
    """NL must equal the NUMBER of coefficients, so NSS = NL*(NL+1)/2 holds.

    The section carries a_0..a_MAX_ORDER, so NL = MAX_ORDER + 1.  Declaring the
    highest index instead under-counts the sub-subsections, and
    ``parse_mf34_mt`` loops exactly NSS times — so the tail of the section is
    silently dropped rather than raising.
    """
    mf34 = create_mf34_from_covariance(
        _shape_cov(), SHAPE_GRID, MAX_ORDER, ZA, AWR, MAT, MT,
        ltt=1, cross_cov=_cross(), cross_energy_grid_ev=CROSS_GRID,
    )
    sub = mf34._subsections[0]
    assert (sub.nl, sub.nl1) == (MAX_ORDER + 1, MAX_ORDER + 1)
    assert len(sub.sub_subsections) == sub.nl * (sub.nl + 1) // 2


def test_default_ltt1_declares_nl_as_the_count_too():
    """LTT=1 carries a_1..a_max_order, so the count equals max_order."""
    mf34 = create_mf34_from_covariance(
        _shape_cov(), SHAPE_GRID, MAX_ORDER, ZA, AWR, MAT, MT, ltt=1,
    )
    sub = mf34._subsections[0]
    assert (sub.nl, sub.nl1) == (MAX_ORDER, MAX_ORDER)
    assert len(sub.sub_subsections) == sub.nl * (sub.nl + 1) // 2


def test_cross_roundtrip():
    cross = _cross()
    mf34 = create_mf34_from_covariance(
        _shape_cov(), SHAPE_GRID, MAX_ORDER, ZA, AWR, MAT, MT,
        ltt=1, cross_cov=cross, cross_energy_grid_ev=CROSS_GRID,
    )
    parsed = parse_mf34_mt(str(mf34).split("\n"), mt=MT)

    pairs = [(ss.l, ss.l1) for ss in parsed.subsections[0].sub_subsections]
    assert pairs == [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]

    # (0, 0) null; (0, L1) cross blocks recovered exactly.
    assert np.allclose(_find(parsed, 0, 0).matrix, 0.0)
    for l1 in (1, 2):
        rec = _find(parsed, 0, l1)
        rect = np.array(rec.rect_matrix).reshape(N0, N)
        assert np.allclose(rect, cross[l1])


def test_cross_shape_blocks_match_default():
    """The L>=1 shape blocks must be byte-identical to the default LTT=1 output."""
    shape_cov = _shape_cov()
    default = create_mf34_from_covariance(
        shape_cov, SHAPE_GRID, MAX_ORDER, ZA, AWR, MAT, MT, ltt=1,
    )
    withcross = create_mf34_from_covariance(
        shape_cov, SHAPE_GRID, MAX_ORDER, ZA, AWR, MAT, MT,
        ltt=1, cross_cov=_cross(), cross_energy_grid_ev=CROSS_GRID,
    )
    for l in (1, 2):
        for l1 in range(l, MAX_ORDER + 1):
            d = _find(default, l, l1)
            c = _find(withcross, l, l1)
            if l == l1:
                assert d.matrix == c.matrix and d.energies == c.energies
            else:
                assert d.rect_matrix == c.rect_matrix


def test_cross_array_form_equivalent_to_dict():
    dict_cross = _cross()
    arr = np.stack([dict_cross[1], dict_cross[2]], axis=0)  # (max_order, N0, N)
    a = create_mf34_from_covariance(
        _shape_cov(), SHAPE_GRID, MAX_ORDER, ZA, AWR, MAT, MT,
        ltt=1, cross_cov=dict_cross, cross_energy_grid_ev=CROSS_GRID,
    )
    b = create_mf34_from_covariance(
        _shape_cov(), SHAPE_GRID, MAX_ORDER, ZA, AWR, MAT, MT,
        ltt=1, cross_cov=arr, cross_energy_grid_ev=CROSS_GRID,
    )
    assert str(a) == str(b)


def test_cross_defaults_to_shape_grid():
    """Omitting cross_energy_grid_ev uses the shape grid (N0 == N)."""
    cross = {1: 0.01 * np.ones((N, N)), 2: np.zeros((N, N))}
    mf34 = create_mf34_from_covariance(
        _shape_cov(), SHAPE_GRID, MAX_ORDER, ZA, AWR, MAT, MT,
        ltt=1, cross_cov=cross,
    )
    rec = _find(mf34, 0, 1)
    assert rec.row_energies == list(SHAPE_GRID)


def test_cross_zero_crossing_warns_only():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mf34 = create_mf34_from_covariance(
            _shape_cov(), SHAPE_GRID, MAX_ORDER, ZA, AWR, MAT, MT,
            ltt=1,
            cross_cov={1: 1e4 * np.ones((N0, N)), 2: np.zeros((N0, N))},
            cross_energy_grid_ev=CROSS_GRID,
        )
    assert any("zero crossing" in str(x.message) for x in w)
    # Warn-only: the large entries are kept, not clamped.
    rec = _find(mf34, 0, 1)
    assert np.allclose(np.array(rec.rect_matrix), 1e4)


def test_cross_rejects_ltt2_shape_layout():
    with pytest.raises(ValueError, match="LTT=1"):
        create_mf34_from_covariance(
            _shape_cov(), SHAPE_GRID, MAX_ORDER, ZA, AWR, MAT, MT,
            ltt=2, cross_cov=_cross(), cross_energy_grid_ev=CROSS_GRID,
        )


def test_cross_rejects_wrong_keys():
    with pytest.raises(ValueError, match="keys must be"):
        create_mf34_from_covariance(
            _shape_cov(), SHAPE_GRID, MAX_ORDER, ZA, AWR, MAT, MT,
            ltt=1, cross_cov={1: np.ones((N0, N))},  # missing l1=2
            cross_energy_grid_ev=CROSS_GRID,
        )


def test_cross_rejects_wrong_block_shape():
    with pytest.raises(ValueError, match="doesn't match"):
        create_mf34_from_covariance(
            _shape_cov(), SHAPE_GRID, MAX_ORDER, ZA, AWR, MAT, MT,
            ltt=1,
            cross_cov={1: np.ones((N0, N)), 2: np.ones((N0, N + 1))},
            cross_energy_grid_ev=CROSS_GRID,
        )


def test_cross_rejects_cross_pair():
    with pytest.raises(ValueError, match="self pair"):
        create_mf34_from_covariance(
            _shape_cov(), SHAPE_GRID, MAX_ORDER, ZA, AWR, MAT, MT,
            ltt=1, mt1=4, cross_cov=_cross(), cross_energy_grid_ev=CROSS_GRID,
        )
