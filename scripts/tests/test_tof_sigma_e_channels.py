"""sigma_E channel precedence, and the two channels added on 2026-08-26.

The resolver used to have three levels: a declared EN-RSL* width, a curated
(L, delta_t) pair, and an ORELA-like default. Two things fell into that last
level that should never have: a subentry whose declared width was *quarantined*
for looking like a covered range, and a subentry EXFOR documents nothing about.
Both then got sigma_E = 0.2-0.4% of E, which for the Fe-56 corpus is 6-31x
narrower than what those experiments declare and 3-6x narrower than the median
of the ones that do declare. Under-declaring the resolution is the direction
this evaluation treats as inadmissible, so both now resolve conservatively.

These tests pin the precedence and the two shape conventions. They use hand-
written cache dicts, not the production JSON, so they stay true if the corpus
changes.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.tof_parameters import (
    CORPUS_MEDIAN_REL_SIGMA_E,
    compute_sigma_E,
    get_tof_parameters,
)

E = 2.0  # MeV, an arbitrary point inside the Fe-56 analysis window

DECLARED = {"energy_resolution": {"fwhm_mev": 0.05}}
QUARANTINED = {"energy_resolution": {"fwhm_mev": 0.40, "review_required": True}}
SPREAD_BOX = {"energy_spread": {"full_width_mev": 0.60, "shape": "box"}}
SPREAD_FWHM = {"energy_spread": {"full_width_mev": 0.60, "shape": "fwhm"}}
CURATED_TOF = {"tof": {"flight_path_m": 5.25, "time_resolution_ns": 5.0}}


def _resolve(entry, **kw):
    p = get_tof_parameters("x", {"x": entry}, **kw)
    return p.source, compute_sigma_E(E, p)


def test_declared_width_wins_over_everything():
    source, sigma = _resolve({**DECLARED, **SPREAD_BOX, **CURATED_TOF})
    assert source == "exfor_rsl"
    assert sigma == pytest.approx(0.05 / 2.3548)


def test_curated_spread_outranks_the_tof_pair():
    source, sigma = _resolve({**SPREAD_BOX, **CURATED_TOF})
    assert source == "curated_spread"
    assert sigma == pytest.approx(0.60 / np.sqrt(12))


def test_spread_shape_selects_the_convention():
    """A covered range is a box; a resolution function is a FWHM. 1.48x apart."""
    _, box = _resolve(SPREAD_BOX)
    _, fwhm = _resolve(SPREAD_FWHM)
    assert box == pytest.approx(0.60 / np.sqrt(12))
    assert fwhm == pytest.approx(0.60 / 2.3548)
    assert fwhm > box
    # "box" is the default when the shape is not stated.
    _, implied = _resolve({"energy_spread": {"full_width_mev": 0.60}})
    assert implied == pytest.approx(box)


def test_curated_pair_still_beats_a_quarantined_width():
    """The box channel sits BELOW the curated pair, so it can only ever
    replace the default — never override a value someone curated."""
    source, sigma = _resolve({**QUARANTINED, **CURATED_TOF})
    assert source == "file"


def test_quarantined_width_is_read_as_a_box_not_dropped():
    source, sigma = _resolve(QUARANTINED)
    assert source == "exfor_rsl_box"
    assert sigma == pytest.approx(0.40 / np.sqrt(12))
    # And it must be wider than the old fall-through, not narrower.
    _, old = _resolve(QUARANTINED, quarantined_as_box=False)
    assert sigma > 5 * old


def test_quarantined_as_box_false_restores_the_old_fall_through():
    source, _ = _resolve(QUARANTINED, quarantined_as_box=False)
    assert source == "default"


def test_relative_default_is_off_unless_asked_for():
    """Callers that have not opted in must be bit-identical to before."""
    source, sigma = _resolve({})
    assert source == "default"
    assert sigma == pytest.approx(
        compute_sigma_E(E, get_tof_parameters("absent", {}))
    )


def test_relative_default_scales_with_energy():
    p = get_tof_parameters("absent", {}, default_rel_sigma_E=CORPUS_MEDIAN_REL_SIGMA_E)
    assert p.source == "default_rel"
    for e in (1.0, 2.0, 4.0):
        assert compute_sigma_E(e, p) == pytest.approx(CORPUS_MEDIAN_REL_SIGMA_E * e)


def test_relative_default_is_wider_than_the_orela_default():
    """The point of the change: 27.037 m / 5 ns applied to a Van de Graaff is
    an under-declaration, not a neutral guess."""
    _, old = _resolve({})
    _, new = _resolve({}, default_rel_sigma_E=CORPUS_MEDIAN_REL_SIGMA_E)
    assert new > 3 * old
