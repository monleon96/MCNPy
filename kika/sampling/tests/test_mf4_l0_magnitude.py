"""Tests for the dormant L=0 (magnitude) plumbing in the MF4 perturbation path.

In MF4 the isotropic term a_0 is identically 1 — the cross-section magnitude
lives in MF3 sigma(E) — so an L=0 perturbation must never scale the Legendre
shape coefficients.  ``_apply_factors_to_mf4_legendre`` routes any L=0 entry to
an optional ``mf3_magnitude_sink`` (for a future joint MF3+MF4 TMC path) and
otherwise skips it.  Existing MF34-only callers never pass L=0, so behavior is
unchanged for them; these tests pin the L=0 handling directly.
"""
from __future__ import annotations

import numpy as np

from kika.endf.classes.mf4.polynomial import MF4MTLegendre
from kika.sampling.endf_perturbation import _apply_factors_to_mf4_legendre


ISO, MT = 26056, 2


def _make_mt() -> MF4MTLegendre:
    """Two energy points, coeffs [a_1, a_2] each; grid boundaries outside span."""
    return MF4MTLegendre(
        number=MT,
        _energies=[1.0, 2.0],
        _legendre_coeffs=[[0.5, 0.2], [0.4, 0.1]],
    )


def _grids():
    # One bin [0, 3] covering both energies (no interior boundary → both points
    # are interior and get scaled).
    return {(ISO, MT, 0): [0.0, 3.0], (ISO, MT, 1): [0.0, 3.0]}


def test_l0_skipped_without_sink():
    """L=0 factor is ignored (magnitude not applied); L>=1 applied normally."""
    mt = _make_mt()
    pm = [(ISO, MT, 0, 0), (ISO, MT, 1, 0)]
    factors = np.array([1.5, 2.0])  # magnitude 1.5, shape (a_1) 2.0
    _apply_factors_to_mf4_legendre(mt, factors, pm, _grids(), verbose=False)
    # a_1 doubled at both energies; a_2 untouched; magnitude 1.5 dropped.
    assert mt._legendre_coeffs == [[1.0, 0.2], [0.8, 0.1]]


def test_l0_routed_to_sink_coeffs_identical():
    """With a sink, L=0 is recorded there and the MF4 coeffs are unchanged
    relative to the no-sink case."""
    mt_ref = _make_mt()
    mt_sink = _make_mt()
    pm = [(ISO, MT, 0, 0), (ISO, MT, 1, 0)]
    factors = np.array([1.5, 2.0])

    _apply_factors_to_mf4_legendre(mt_ref, factors, pm, _grids(), verbose=False)
    sink: dict = {}
    _apply_factors_to_mf4_legendre(
        mt_sink, factors, pm, _grids(), verbose=False, mf3_magnitude_sink=sink,
    )

    assert sink == {(ISO, MT, 0): 1.5}
    assert mt_sink._legendre_coeffs == mt_ref._legendre_coeffs


def test_l0_only_is_noop_on_coeffs():
    """An L=0-only mapping never mutates the shape coefficients."""
    mt = _make_mt()
    baseline = [row[:] for row in mt._legendre_coeffs]
    _apply_factors_to_mf4_legendre(
        mt, np.array([1.5]), [(ISO, MT, 0, 0)],
        {(ISO, MT, 0): [0.0, 3.0]}, verbose=False,
    )
    assert mt._legendre_coeffs == baseline


def test_l0_sink_records_per_bin():
    """Two magnitude bins record one entry each, keyed by energy bin."""
    mt = _make_mt()
    pm = [(ISO, MT, 0, 0), (ISO, MT, 0, 1)]
    grids = {(ISO, MT, 0): [0.0, 1.5, 3.0]}  # two bins
    sink: dict = {}
    _apply_factors_to_mf4_legendre(
        mt, np.array([1.1, 0.9]), pm, grids, verbose=False,
        mf3_magnitude_sink=sink,
    )
    assert sink == {(ISO, MT, 0): 1.1, (ISO, MT, 1): 0.9}
    # No shape order touched.
    assert mt._legendre_coeffs == [[0.5, 0.2], [0.4, 0.1]]
