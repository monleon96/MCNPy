"""The declared incident-energy resolution EXFOR carries as EN-RSL*.

The conventions under test are LEXFOR's (IAEA-NDS-0208, "Resolution") and
EXFOR Dictionary 24's, not this module's inventions:

  * ``EN-RSL-FW`` is a full width, ``EN-RSL-HW`` a half width, and LEXFOR's
    worked example gives FW = 2 MeV against HW = 1 MeV for one curve, so a half
    width doubles.
  * ``EN-RSL`` is *unspecified*.  LEXFOR says the resolution is "usually
    defined as full-width at half-maximum", so it is read as a FWHM and marked
    as an assumption rather than silently treated as fact.

The X4Pro-shaped fixtures below are trimmed from real subentries; the values
are the ones the database actually holds.
"""

from __future__ import annotations

import pytest

from kika.exfor.database import (
    _declared_fwhm_over_energy,
    _is_resolution_header,
    _parse_x4data_json,
    _read_declared_resolution,
    declared_resolution_fwhm_ev,
)


def _column(header, units, basic_units, *, com1=None, dat1=None, com0=None, dat0=None):
    """One x4data variable, in the shape X4Pro emits."""
    return {
        "fam": "EN",
        "cvar": "dx1",
        "header": header,
        "units": units,
        "basicUnits": basic_units,
        "ifComm": dat1 is None and dat0 is None,
        "com0": com0,
        "com1": com1,
        "dat0": dat0,
        "dat1": dat1,
    }


# --- headings ---------------------------------------------------------------

@pytest.mark.parametrize("header", ["EN-RSL", "EN-RSL-FW", "EN-RSL-HW", "en-rsl-hw1"])
def test_resolution_headers_are_recognised(header):
    assert _is_resolution_header(header)


@pytest.mark.parametrize("header", ["EN", "ANG", "DATA-ERR", "", None])
def test_other_headers_are_not(header):
    assert not _is_resolution_header(header)


# --- conventions ------------------------------------------------------------

def test_full_width_is_taken_as_given():
    rec = _read_declared_resolution(_column("EN-RSL-FW", "MEV", "EV", com1=13000.0))
    assert rec["convention"] == "full_width"
    assert rec["assumed_fwhm"] is False
    assert rec["fwhm_ev"] == 13000.0


def test_half_width_doubles_to_a_full_width():
    """LEXFOR's example: HW = 1 MeV is the same curve as FW = 2 MeV."""
    rec = _read_declared_resolution(_column("EN-RSL-HW", "MEV", "EV", com1=1e6))
    assert rec["convention"] == "half_width"
    assert rec["fwhm_ev"] == pytest.approx(2e6)


def test_bare_heading_is_read_as_fwhm_but_flagged():
    rec = _read_declared_resolution(_column("EN-RSL", "KEV", "EV", com1=20000.0))
    assert rec["convention"] == "unspecified"
    assert rec["assumed_fwhm"] is True
    assert rec["fwhm_ev"] == 20000.0


def test_ratio_headings_are_not_this_datasets_beam():
    """EN-RSL-DN/-NM belong to a REACTION ratio's other side."""
    assert _read_declared_resolution(_column("EN-RSL-DN", "MEV", "EV", com1=1e6)) is None


# --- units ------------------------------------------------------------------

def test_per_cent_is_relative_to_the_incident_energy():
    rec = _read_declared_resolution(_column("EN-RSL", "PER-CENT", "PER-CENT", com1=14.5))
    assert rec["kind"] == "relative"
    assert rec["fwhm_fraction"] == pytest.approx(0.145)
    assert declared_resolution_fwhm_ev(rec, 2e6) == pytest.approx(2.9e5)


def test_reciprocal_velocity_is_already_converted_by_x4pro():
    """`basicUnits` still reads NSEC/M but the converted value is in eV.

    Subentry 21142005 declares 0.06 ns/m at 14.2 MeV and stores 88804.7, which
    is LEXFOR's 2.766e-2 * E^1.5 * dtau expressed in eV. Reading that as a
    timing spread gives a resolution of several GeV.
    """
    rec = _read_declared_resolution(_column("EN-RSL", "NSEC/M", "NSEC/M", com1=88804.7))
    assert rec["kind"] == "absolute"
    assert declared_resolution_fwhm_ev(rec, 14.2e6) == pytest.approx(88804.7)


def test_raw_reciprocal_velocity_still_needs_the_energy():
    """With no converted value, the column is a timing spread per metre."""
    rec = _read_declared_resolution(
        _column("EN-RSL", "NSEC/M", "NSEC/M", com0=0.06, com1=None)
    )
    assert rec["kind"] == "per_flight_path"
    # LEXFOR: dE[MeV] = 2.766e-2 * E[MeV]^1.5 * dtau[ns/m]
    assert declared_resolution_fwhm_ev(rec, 14.2e6) == pytest.approx(88_800, rel=1e-3)


def test_an_unknown_base_unit_is_refused():
    """Read as eV it would fold the evaluation to a width off by decades."""
    assert _read_declared_resolution(_column("EN-RSL", "BARNS", "BARNS", com1=3.0)) is None


# --- per-point columns ------------------------------------------------------

def test_per_point_values_are_kept_and_summarised():
    rec = _read_declared_resolution(
        _column("EN-RSL-HW", "MEV", "EV", dat1=[0.1e6, 0.2e6, 0.3e6])
    )
    assert rec["fwhm_ev_values"] == [0.2e6, 0.4e6, 0.6e6]   # doubled
    assert declared_resolution_fwhm_ev(rec, 5e6) == pytest.approx(0.4e6)  # median


# --- the parser as a whole --------------------------------------------------

def test_a_resolution_column_no_longer_overwrites_the_energies():
    """EN-RSL arrives as fam="EN", so the energy branch must check cvar.

    Without the guard the resolution replaced `energies` — and `energy_unit`
    with it — for every dataset that declares one.
    """
    parsed = _parse_x4data_json({
        "x4data": [
            {"fam": "Data", "cvar": "y", "header": "DATA", "units": "B/SR",
             "dat0": [1.0, 2.0]},
            {"fam": "EN", "cvar": "x1", "header": "EN", "units": "MEV",
             "dat0": [2.0, 3.0]},
            _column("EN-RSL", "KEV", "EV", dat1=[20000.0, 20000.0], dat0=[20.0, 20.0]),
        ]
    })
    assert parsed["energies"] == [2.0, 3.0]
    assert parsed["energy_unit"] == "MEV"
    assert parsed["energy_resolution"]["fwhm_ev_values"] == [20000.0, 20000.0]


def test_no_declaration_is_absent_rather_than_defaulted():
    parsed = _parse_x4data_json({
        "x4data": [
            {"fam": "EN", "cvar": "x1", "header": "EN", "units": "MEV",
             "dat0": [2.0]},
        ]
    })
    assert parsed["energy_resolution"] is None


# --- plausibility -----------------------------------------------------------

def test_ratio_reports_a_width_that_cannot_be_a_resolution():
    """11095002 declares EN-RSL = 5 MeV for EN = 2.45 MeV."""
    rec = _read_declared_resolution(_column("EN-RSL", "MEV", "EV", com1=5e6))
    assert _declared_fwhm_over_energy(rec, [2.45e6]) == pytest.approx(5e6 / 2.45e6)


def test_ratio_is_the_median_not_the_worst_point():
    """A constant width is largest relative to the lowest energy on the grid.

    Taking the extreme would flag an ordinary declaration whose resolution
    merely dominates at the bottom of its own energy range.
    """
    rec = _read_declared_resolution(_column("EN-RSL", "KEV", "EV", com1=20000.0))
    ratio = _declared_fwhm_over_energy(rec, [2e4, 1e6, 2e6, 5e6, 1e7])
    assert ratio == pytest.approx(20000.0 / 2e6)


def test_ratio_is_none_without_energies():
    rec = _read_declared_resolution(_column("EN-RSL", "KEV", "EV", com1=20000.0))
    assert _declared_fwhm_over_energy(rec, []) is None
    assert _declared_fwhm_over_energy(None, [1e6]) is None
