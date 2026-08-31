"""
X4Pro EXFOR Database Interface.

This module provides the X4ProDatabase class for accessing EXFOR data
directly from the X4Pro SQLite database (full 2025 version with JSON schema).

The full database stores measurement data in JSON format in the `x4pro_x5z.jx5z`
column, containing:
- x4data: Raw measurement arrays (energy, angle, cross section, uncertainties)
- c5data: Pre-computed data arrays
- Metadata: author, year, target, projectile, reaction codes

Usage:
    >>> from kika.exfor.database import X4ProDatabase
    >>> db = X4ProDatabase()  # Uses KIKA_X4PRO_DB_PATH env var or default path
    >>> datasets = db.query_angular_distributions(target_zaid=26056, projectile="N")
    >>> print(f"Found {len(datasets)} angular distribution datasets")
"""

import json
import os
import sqlite3
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from kika.exfor._constants import (
    FRAME_LAB,
    FRAME_CM,
    DB_DEFAULT_PATH,
    DB_UNIT_MAPPINGS,
    DB_FAMILY_MAPPINGS,
    EXFOR_QUANTITY_CODES,
    EXFOR_FAMILY_TO_VARIABLE,
)
from kika.exfor.config import get_tof_metadata_path
from kika.exfor.experiment import ExforExperiment
from kika._constants import ATOMIC_NUMBER_TO_SYMBOL, SYMBOL_TO_ATOMIC_NUMBER
from kika._utils import zaid_to_symbol, symbol_to_zaid
from kika.exfor.angular_distribution import ExforAngularDistribution
from kika.exfor.cross_section import ExforCrossSection
from kika.exfor.exfor_entry import ExforEntry

# Module-level cache for TOF metadata
_tof_metadata_cache: Optional[Dict[str, Any]] = None


# Default TOF geometry when an experiment has no curated entry (GELINA-style).
_TOF_DEFAULT_FLIGHT_PATH_M = 27.037
_TOF_DEFAULT_TIME_RESOLUTION_NS = 10.0


def _load_tof_metadata(force_reload: bool = False) -> Dict[str, Any]:
    """
    Load TOF metadata from the configuration file.

    The file (``exfor_tof_parameters.json``) is keyed by EXFOR dataset ID; each
    entry uses the nested ``energy_resolution_input`` schema (``distance`` and
    ``time_resolution`` sub-objects). Supplements the database, which lacks this
    metadata. Returns an empty map if the file is missing or invalid, so every
    experiment falls back to the defaults.

    Parameters
    ----------
    force_reload : bool
        If True, reload from file even if cached
    """
    global _tof_metadata_cache

    if _tof_metadata_cache is not None and not force_reload:
        return _tof_metadata_cache

    metadata_path = get_tof_metadata_path()

    _tof_metadata_cache = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                _tof_metadata_cache = loaded
        except (json.JSONDecodeError, IOError):
            _tof_metadata_cache = {}

    return _tof_metadata_cache


def _get_tof_params_for_experiment(dataset_id: str) -> Dict[str, Any]:
    """
    Get TOF parameters for a specific experiment, with fallback to defaults.

    Reads the nested ``energy_resolution_input`` (or ``energy_resolution_inputs``)
    schema, taking ``distance.value`` as the flight path and
    ``time_resolution.value`` as the timing resolution. When the experiment is
    absent, or either value is missing/null, the defaults are used.

    Parameters
    ----------
    dataset_id : str
        EXFOR dataset ID (e.g., "10037024")

    Returns
    -------
    Dict[str, Any]
        ``{'flight_path_m': float, 'time_resolution_ns': float,
           'source': 'file' | 'default'}``
    """
    metadata = _load_tof_metadata()
    entry = metadata.get(dataset_id) if isinstance(metadata, dict) else None

    if isinstance(entry, dict):
        eri = entry.get("energy_resolution_input") or entry.get("energy_resolution_inputs")
        if isinstance(eri, dict):
            distance = (eri.get("distance") or {}).get("value")
            time_res = (eri.get("time_resolution") or {}).get("value")
            if distance is not None and time_res is not None:
                return {
                    "flight_path_m": float(distance),
                    "time_resolution_ns": float(time_res),
                    "source": "file",
                }

    return {
        "flight_path_m": _TOF_DEFAULT_FLIGHT_PATH_M,
        "time_resolution_ns": _TOF_DEFAULT_TIME_RESOLUTION_NS,
        "source": "default",
    }

@dataclass
class X4ProDataset:
    """
    Raw dataset from X4Pro database before conversion to ExforAngularDistribution.

    This intermediate representation holds the parsed JSON data from the database
    before it is converted to the full ExforAngularDistribution object.
    """

    dataset_id: str
    year: int
    author: str
    target: str
    projectile: str
    mf: int
    mt: int
    quant: str
    ndat: int
    reacode: str

    # Parsed data arrays
    energies_ev: np.ndarray = field(default_factory=lambda: np.array([]))
    angles_deg: np.ndarray = field(default_factory=lambda: np.array([]))
    cross_sections: np.ndarray = field(default_factory=lambda: np.array([]))
    uncertainties: np.ndarray = field(default_factory=lambda: np.array([]))

    # Units as read from database
    energy_unit: str = "EV"
    angle_unit: str = "ADEG"
    xs_unit: str = "B/SR"

    # Frame information
    angle_frame: str = FRAME_LAB

    # Correction information
    is_corrected: bool = False
    correction_notes: List[str] = field(default_factory=list)

    # Raw JSON for debugging/verification
    raw_json: Optional[Dict[str, Any]] = None

    # All x4data dy columns surfaced (DATA-ERR, ERR-1, ERR-S, ERR-T, ...) — used by
    # the uncertainty manifest resolver to derive sigma_stat and sigma_sys.
    uncertainty_components: List[Dict[str, Any]] = field(default_factory=list)

    # The author's own declared incident-energy resolution (EXFOR EN-RSL*),
    # normalised to a FWHM. None when the dataset declares none — which is the
    # majority: see _read_declared_resolution.
    energy_resolution: Optional[Dict[str, Any]] = None


def _zaid_to_target_pattern(zaid: int) -> str:
    """
    Convert ZAID to database target pattern (e.g., 26056 -> "Fe-56").

    Uses kika._utils.zaid_to_symbol internally.
    """
    symbol = zaid_to_symbol(zaid)  # Returns "Fe56" or "Fe" for natural
    # Convert "Fe56" to "Fe-56" format used by database
    match = re.match(r"([A-Za-z]+)(\d*)", symbol)
    if match:
        elem = match.group(1)
        mass = match.group(2) or "0"
        return f"{elem}-{mass}"
    return ""


def _parse_target_from_db(target_str: str) -> Tuple[str, int]:
    """
    Parse target from database format (e.g., "Fe-56" -> ("Fe56", 26056)).

    Uses kika._utils.symbol_to_zaid internally.
    """
    # Try database format first: "Fe-56" -> "Fe56"
    match = re.match(r"([A-Za-z]+)-(\d+)", target_str)
    if match:
        symbol = match.group(1).capitalize()
        a = match.group(2)
        target = f"{symbol}{a}" if a != "0" else symbol
        try:
            zaid = symbol_to_zaid(target)
            return (target, zaid)
        except ValueError:
            pass

    # Try EXFOR format: "26-FE-56"
    match = re.match(r"(\d+)-([A-Z]+)-(\d+)", target_str.upper())
    if match:
        z = int(match.group(1))
        symbol = match.group(2).capitalize()
        a = int(match.group(3))
        target = f"{symbol}{a}" if a > 0 else f"{symbol}0"
        zaid = z * 1000 + a
        return (target, zaid)

    return ("Unknown", 0)


# --- Declared incident-energy resolution (EXFOR EN-RSL*) --------------------
#
# EXFOR lets an author declare the incident-energy resolution of their own
# measurement.  It is the experiment's own number, so it beats anything curated
# externally, and it is far more widely available: 25% of elastic angular
# datasets carry one.
#
# What the value *means* is specified, and the specification is uncomfortable.
# LEXFOR (IAEA-NDS-0208, "Resolution / Incident-Projectile Energy Resolution")
# says the resolution "is usually defined as full-width at half-maximum (FWHM),
# but may be given in other representations such as standard deviation", and
# asks for the shape to be given in free text under INC-SPECT.  EXFOR
# Dictionary 24 then codes three headings:
#
#     EN-RSL-FW   full width
#     EN-RSL-HW   half width
#     EN-RSL      unspecified
#
# `-FW` and `-HW` are exact against each other — LEXFOR's worked example gives
# FW = 2 MeV and HW = 1 MeV for one curve — and Dictionary 24 spells the ratio
# variants `EN-RSL-DN`/`-NM` as "(FWHM)", which is what fixes "full width" as
# meaning FWHM.  The bare heading is the problem: it is three quarters of the
# corpus, and the INC-SPECT escape hatch is not used in practice (measured over
# the full X4Pro: of 2283 bare-EN-RSL angular subentries, 13 have free text
# that says anything about the incident-energy convention).
#
# So a bare EN-RSL is normalised to a FWHM — LEXFOR's "usually" — and flagged
# `convention="unspecified"` so a caller can say out loud that it assumed.
# Nothing here silently decides what it cannot know.

#: EXFOR headings holding a declared incident-energy resolution.
_RESOLUTION_HEADINGS = ("EN-RSL",)

#: Headings whose value is a half width, i.e. half of the FWHM (Dictionary 24).
_HALF_WIDTH_HEADINGS = ("EN-RSL-HW",)


def _is_resolution_header(header: Any) -> bool:
    """Whether an x4data column holds a declared incident-energy resolution."""
    return str(header or "").upper().startswith(_RESOLUTION_HEADINGS)


def _read_declared_resolution(var: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Read one EN-RSL* column into a normalised record, or None.

    X4Pro carries both the value as compiled (``com0``/``dat0``, in ``units``)
    and the same value converted to a base unit (``com1``/``dat1``, in
    ``basicUnits``).  The converted form is used: it saves reimplementing
    Dictionary 25's factor table, and it is what X4Pro itself considers this
    number to be.  ``basicUnits`` collapses every absolute energy unit to EV
    and leaves the two relative forms alone, so it doubles as the discriminator
    between the three kinds of width EXFOR allows.

    Returns a dict with:

    ``heading``
        The raw EXFOR heading, for display.
    ``convention``
        ``"full_width"`` (EN-RSL-FW), ``"half_width"`` (EN-RSL-HW) or
        ``"unspecified"`` (bare EN-RSL).
    ``kind``
        ``"absolute"`` (``fwhm_ev``/``fwhm_ev_values``), ``"relative"``
        (``fwhm_fraction``, of the incident energy) or ``"per_flight_path"``
        (``fwhm_ns_per_m``, a reciprocal velocity).
    ``assumed_fwhm``
        True when the heading declined to say, so the caller can label it.

    Half widths are doubled here, so every ``fwhm_*`` field means a full width
    at half maximum regardless of which heading it came from.  The FWHM→sigma
    step is deliberately *not* done here: :data:`kika._constants.FWHM_TO_SIGMA`
    is the single definition of it and the folding code owns that conversion.
    """
    header = str(var.get("header", "") or "").upper()
    basic = str(var.get("basicUnits", "") or "").upper()

    # Prefer the converted value; fall back to the raw one only when the units
    # were already the base ones, so an unconverted number is never mistaken
    # for a converted one. (`dat1eq` — 4 angular columns — is an expression
    # rather than a value, and is skipped.)
    converted = True
    if var.get("ifComm"):
        value, values = var.get("com1"), None
        if value is None and str(var.get("units", "")).upper() == basic:
            value, converted = var.get("com0"), False
    else:
        value, values = None, var.get("dat1")
        if values is None and str(var.get("units", "")).upper() == basic:
            values, converted = var.get("dat0"), False

    if value is None and not values:
        return None

    if header.startswith(_HALF_WIDTH_HEADINGS):
        convention = "half_width"
    elif header.startswith("EN-RSL-FW"):
        convention = "full_width"
    elif header == "EN-RSL":
        convention = "unspecified"
    else:
        # EN-RSL-DN / -NM belong to a REACTION ratio's numerator or
        # denominator, not to this dataset's own incident beam.
        return None

    # LEXFOR's example fixes HW = FW/2, so a half width doubles to a FWHM.
    scale = 2.0 if convention == "half_width" else 1.0

    record: Dict[str, Any] = {
        "heading": header,
        "convention": convention,
        "assumed_fwhm": convention == "unspecified",
        "units": var.get("units"),
    }

    if basic == "PER-CENT":
        record["kind"] = "relative"
        record["fwhm_fraction"] = (
            float(value) * scale / 100.0 if value is not None else None
        )
        if values:
            record["fwhm_fraction_values"] = [
                float(v) * scale / 100.0 for v in values if v is not None
            ]
    elif basic == "EV" or (converted and basic in ("NSEC/M", "MICROSEC/M")):
        # Electron-volts — and, for a reciprocal velocity, *also* electron-volts
        # whenever the converted value is what we are holding. X4Pro leaves
        # `basicUnits` reading NSEC/M there but has already applied the LEXFOR
        # relation itself: subentry 21142005 declares 0.06 ns/m at 14.2 MeV and
        # stores com1 = 88804.7, which is 2.766e-2 * 14.2^1.5 * 0.06 MeV in eV.
        # Reading that as a timing spread gives a resolution of several GeV.
        record["kind"] = "absolute"
        record["fwhm_ev"] = float(value) * scale if value is not None else None
        if values:
            record["fwhm_ev_values"] = [
                float(v) * scale for v in values if v is not None
            ]
    elif basic in ("NSEC/M", "MICROSEC/M"):
        # The raw column, in the unit as compiled: a timing spread per metre of
        # flight path, which only becomes a width once an energy is supplied.
        per_m = float(value) * scale if value is not None else None
        if basic == "MICROSEC/M" and per_m is not None:
            per_m *= 1000.0
        record["kind"] = "per_flight_path"
        record["fwhm_ns_per_m"] = per_m
    else:
        # An unrecognised base unit is worse than no resolution at all: it
        # would be read as electron-volts and fold the evaluation to a width
        # off by orders of magnitude.
        return None

    return record


def declared_resolution_fwhm_ev(
    resolution: Optional[Dict[str, Any]],
    energy_ev: float,
) -> Optional[float]:
    r"""The declared resolution as a FWHM in eV at one incident energy.

    Resolves the three forms EXFOR allows onto a single number, which is what
    a folding kernel needs.  For the reciprocal-velocity form LEXFOR gives, at
    the non-relativistic limit,

    .. math:: \Delta E = 2.766\times10^{-2}\, E^{3/2}\, |\Delta\tau|

    with :math:`\Delta E`, :math:`E` in MeV and :math:`\Delta\tau` in ns/m
    (cf. Schillebeeckx et al., *Nucl. Data Sheets* **113** (2012) 3054, §II.B).
    The relation converts a width to a width, so it carries the convention
    through unchanged: a FWHM in time gives a FWHM in energy.

    Returns None when there is nothing to compute from, so a caller can fall
    back rather than fold to a fabricated width.
    """
    if not resolution or not energy_ev or energy_ev <= 0:
        return None

    kind = resolution.get("kind")

    if kind == "absolute":
        width = resolution.get("fwhm_ev")
        if width is None:
            values = resolution.get("fwhm_ev_values") or []
            # A per-point column has no single width; the median is the
            # honest summary for a caller that wants one number.
            if not values:
                return None
            ordered = sorted(values)
            width = ordered[len(ordered) // 2]
        return float(width) if width and width > 0 else None

    if kind == "relative":
        frac = resolution.get("fwhm_fraction")
        if frac is None:
            values = resolution.get("fwhm_fraction_values") or []
            if not values:
                return None
            ordered = sorted(values)
            frac = ordered[len(ordered) // 2]
        return float(frac) * energy_ev if frac and frac > 0 else None

    if kind == "per_flight_path":
        ns_per_m = resolution.get("fwhm_ns_per_m")
        if not ns_per_m or ns_per_m <= 0:
            return None
        e_mev = energy_ev / 1e6
        return 2.766e-2 * (e_mev ** 1.5) * float(ns_per_m) * 1e6

    return None


def _parse_x4data_json(jx5z: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the x4data array from the JSON structure.

    The x4data array contains measurement variables:
    - ivar: Variable index
    - cvar: Variable category ('y' for dependent, 'x1', 'x2' for independent)
    - fam: Family code ('Data', 'EN', 'ANG', 'COS', 'dData', etc.)
    - dat0: Data array
    - units: Unit string

    Parameters
    ----------
    jx5z : Dict[str, Any]
        Parsed JSON from jx5z column

    Returns
    -------
    Dict[str, Any]
        Extracted data with keys: 'energies', 'angles', 'values', 'uncertainties',
        'energy_unit', 'angle_unit', 'xs_unit', 'angle_type'
    """
    x4data = jx5z.get("x4data", [])

    result = {
        "energies": [],
        "angles": [],
        "values": [],
        "uncertainties": [],
        "energy_unit": "EV",
        "angle_unit": "ADEG",
        "xs_unit": "B/SR",
        "uncertainty_unit": "",  # Track uncertainty unit for PER-CENT detection
        "angle_type": "ANG",  # 'ANG' or 'COS'
        "uncertainty_components": [],  # All dy columns surfaced (DATA-ERR, ERR-1, ERR-S, ERR-T, ...)
        # The author's own declared incident-energy resolution (EN-RSL*), or
        # None. See _read_declared_resolution.
        "energy_resolution": None,
    }

    for var in x4data:
        fam = var.get("fam", "")
        cvar = var.get("cvar", "")
        dat0 = var.get("dat0", [])
        units = var.get("units", "")

        if fam == "Data" and cvar == "y":
            # Cross section values
            result["values"] = dat0
            result["xs_unit"] = units
        elif fam == "EN" and not cvar.startswith("d"):
            # Incident energy itself. The cvar guard matters: EXFOR's declared
            # resolution (EN-RSL*) is also fam="EN", as cvar="dx1", and without
            # it that column fell through to here and overwrote the energies
            # with resolution widths — plus energy_unit with the resolution's.
            result["energies"] = dat0
            result["energy_unit"] = units
        elif fam == "EN" and _is_resolution_header(var.get("header", "")):
            # Declared incident-energy resolution. See _read_declared_resolution.
            declared = _read_declared_resolution(var)
            if declared is not None:
                result["energy_resolution"] = declared
        elif fam == "ANG":
            # Angle in degrees
            result["angles"] = dat0
            result["angle_unit"] = units
            result["angle_type"] = "ANG"
        elif fam == "COS":
            # Angle as cosine
            result["angles"] = dat0
            result["angle_unit"] = units
            result["angle_type"] = "COS"
        elif cvar == "dy" or fam in ("dData", "DATA-ERR"):
            header = var.get("header", "")
            if_comm = bool(var.get("ifComm", False))

            if if_comm:
                # Common (scalar) uncertainty column: value lives in com0,
                # not dat0. Surface it as a scalar component without
                # touching the legacy per-point fields (which would
                # otherwise be silently overwritten with []).
                scalar_value = var.get("com0")
                if scalar_value is not None:
                    result["uncertainty_components"].append({
                        "header": header,
                        "kind": "scalar",
                        "value": scalar_value,
                        "unit": units,
                    })
            else:
                # Per-point uncertainty column.
                if dat0:
                    result["uncertainty_components"].append({
                        "header": header,
                        "kind": "per_point",
                        "values": dat0,
                        "unit": units,
                    })
                # Preserve legacy "last per-point dy wins" semantics so
                # downstream code (PER-CENT override branch) is unchanged.
                result["uncertainties"] = dat0
                result["uncertainty_unit"] = units

    return result


def _parse_c5data_json(jx5z: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the c5data (corrected) dict from the JSON structure.

    The c5data dict contains pre-computed values with corrections applied
    (when applicable) and standardized units. This is the preferred data
    source as it includes:
    - Decay data corrections (gamma line intensities from ENSDF)
    - Monitor cross section renormalization
    - Standardized units (EV, ADEG, B/SR)

    Parameters
    ----------
    jx5z : Dict[str, Any]
        Parsed JSON from jx5z column

    Returns
    -------
    Dict[str, Any]
        Extracted data with keys: 'energies', 'angles', 'values', 'uncertainties',
        'energy_unit', 'angle_unit', 'xs_unit', 'angle_type', 'is_corrected',
        'correction_notes'
    """
    c5data = jx5z.get("c5data", {})

    result = {
        "energies": [],
        "angles": [],
        "values": [],
        "uncertainties": [],
        "energy_unit": "EV",
        "angle_unit": "ADEG",
        "xs_unit": "B/SR",
        "angle_type": "ANG",
        "is_corrected": False,
        "correction_notes": [],
    }

    if not isinstance(c5data, dict):
        return result

    # Extract y (cross section)
    if "y" in c5data:
        y_data = c5data["y"]
        if isinstance(y_data, dict):
            result["values"] = y_data.get("y", [])
            result["uncertainties"] = y_data.get("dy", [])
            result["xs_unit"] = y_data.get("units", "B/SR")

    # Extract x1 (energy)
    if "x1" in c5data:
        x1_data = c5data["x1"]
        if isinstance(x1_data, dict):
            # Check family is energy (EN)
            if x1_data.get("fam") == "EN":
                result["energies"] = x1_data.get("x1", [])
                result["energy_unit"] = x1_data.get("units", "EV")

    # Extract x2 (angle or cosine)
    if "x2" in c5data:
        x2_data = c5data["x2"]
        if isinstance(x2_data, dict):
            result["angles"] = x2_data.get("x2", [])
            fam = x2_data.get("fam", "")
            if fam == "COS":
                result["angle_type"] = "COS"
                result["angle_unit"] = "NO-DIM"
            else:
                result["angle_type"] = "ANG"
                result["angle_unit"] = x2_data.get("units", "ADEG")

    # Check if corrections were applied
    auto_corr_notes = jx5z.get("autoCorrNotes", [])
    if auto_corr_notes:
        result["is_corrected"] = True
        result["correction_notes"] = auto_corr_notes if isinstance(auto_corr_notes, list) else [auto_corr_notes]

    return result


# Constants for percentage uncertainty detection
_PERCENT_UNIT_INDICATORS = ("PER-CENT", "PC", "%", "PERCENT")


def _is_percent_unit(unit_str: str) -> bool:
    """Check if a unit string indicates percentage values."""
    return unit_str.upper() in _PERCENT_UNIT_INDICATORS


def _convert_percent_to_absolute(
    uncertainties: np.ndarray,
    values: np.ndarray,
    unc_unit: str,
) -> np.ndarray:
    """
    Convert percentage uncertainties to absolute values.

    Parameters
    ----------
    uncertainties : np.ndarray
        Uncertainty values (may be in percent)
    values : np.ndarray
        Cross section values (absolute)
    unc_unit : str
        Unit string for uncertainties (e.g., "PER-CENT", "B/SR")

    Returns
    -------
    np.ndarray
        Absolute uncertainties in the same units as values
    """
    if _is_percent_unit(unc_unit):
        # Values are percentages (e.g., 4.36 means 4.36%)
        return np.abs(values) * (uncertainties / 100.0)
    return uncertainties


def _convert_units(
    values: np.ndarray, from_unit: str, target_unit: str, unit_type: str
) -> np.ndarray:
    """
    Convert values between units.

    Parameters
    ----------
    values : np.ndarray
        Input values
    from_unit : str
        Source unit (e.g., 'EV', 'MEV', 'B/SR', 'MB/SR')
    target_unit : str
        Target unit
    unit_type : str
        Type of unit: 'energy', 'cross_section'

    Returns
    -------
    np.ndarray
        Converted values
    """
    from_unit = from_unit.upper()
    target_unit = target_unit.upper()

    if from_unit == target_unit:
        return values

    mappings = DB_UNIT_MAPPINGS.get(unit_type, {})

    from_factor = mappings.get(from_unit, 1.0)
    to_factor = mappings.get(target_unit, 1.0)

    return values * (from_factor / to_factor)


class X4ProDatabase:
    """
    Interface to X4Pro SQLite database (full 2025 version).

    This class provides methods for querying angular distribution data from the
    X4Pro database and converting it to ExforAngularDistribution objects.

    Parameters
    ----------
    db_path : str, optional
        Path to X4Pro SQLite database. If not provided, uses the
        KIKA_X4PRO_DB_PATH environment variable.

    Attributes
    ----------
    db_path : str
        Path to the database file
    _conn : sqlite3.Connection
        Database connection (lazy-loaded)

    Examples
    --------
    >>> db = X4ProDatabase()
    >>> datasets = db.query_angular_distributions(target_zaid=26056)
    >>> print(f"Found {len(datasets)} datasets for Fe-56")
    """

    def __init__(self, db_path: str = None):
        """Initialize database connection."""
        from kika.exfor.config import get_db_path
        self.db_path = get_db_path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            if self.db_path is None:
                raise FileNotFoundError(
                    "X4Pro database path not configured.\n"
                    "Set KIKA_X4PRO_DB_PATH environment variable or provide db_path parameter."
                )
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(
                    f"X4Pro database not found at: {self.db_path}\n"
                    f"Set KIKA_X4PRO_DB_PATH environment variable or provide db_path."
                )
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def query_dataset_ids(
        self,
        target: Union[str, List[str]] = None,
        target_zaid: Union[int, List[int]] = None,
        projectile: str = "N",
        quantity: str = "DA",
        mf: int = None,
        mt: int = None,
        energy_min_mev: float = None,
        energy_max_mev: float = None,
        year_min: int = None,
        year_max: int = None,
        author: str = None,
    ) -> List[str]:
        """
        Query dataset IDs matching the given criteria.

        Parameters
        ----------
        target : str or List[str], optional
            Target in EXFOR notation (e.g., "26-FE-56" or ["Fe-56", "Fe-0"])
        target_zaid : int or List[int], optional
            Target ZAID (e.g., 26056 for Fe-56, or [26056, 26000] for Fe-56 + natural)
        projectile : str, optional
            Projectile (default: "N" for neutrons)
        quantity : str, optional
            Quantity type (default: "DA" for angular distribution)
        mf : int, optional
            ENDF MF number (4 for angular distributions)
        mt : int, optional
            ENDF MT number (2 for elastic scattering)
        energy_min_mev : float, optional
            Minimum energy in MeV
        energy_max_mev : float, optional
            Maximum energy in MeV
        year_min : int, optional
            Minimum publication year
        year_max : int, optional
            Maximum publication year
        author : str, optional
            Author name (partial match)

        Returns
        -------
        List[str]
            List of DatasetID strings
        """
        conn = self._get_connection()

        # Build query
        conditions = []
        params = []

        # Target filtering - now supports lists
        if target:
            # Convert single value to list for uniform handling
            target_list = [target] if isinstance(target, str) else target
            if len(target_list) == 1:
                conditions.append("Targ1 LIKE ?")
                params.append(f"%{target_list[0]}%")
            else:
                # Multiple targets: use OR condition
                target_conditions = ["Targ1 LIKE ?" for _ in target_list]
                conditions.append(f"({' OR '.join(target_conditions)})")
                params.extend([f"%{t}%" for t in target_list])
        elif target_zaid:
            # Convert ZAID(s) to database target format(s)
            zaid_list = [target_zaid] if isinstance(target_zaid, int) else target_zaid
            target_patterns = []
            for zaid in zaid_list:
                pattern = _zaid_to_target_pattern(zaid)
                if pattern:
                    target_patterns.append(pattern)

            if target_patterns:
                if len(target_patterns) == 1:
                    conditions.append("Targ1 = ?")
                    params.append(target_patterns[0])
                else:
                    # Multiple targets: use OR condition with exact match
                    target_conditions = ["Targ1 = ?" for _ in target_patterns]
                    conditions.append(f"({' OR '.join(target_conditions)})")
                    params.extend(target_patterns)

        # Projectile (database uses lowercase for common particles: n, p, d, a, g)
        if projectile:
            proj_lower = projectile.lower()
            # Common particles are stored lowercase; heavier ions may be mixed case
            conditions.append("(Proj = ? OR Proj = ?)")
            params.extend([proj_lower, projectile.upper()])

        # Quantity (angular distribution)
        if quantity:
            conditions.append("quant1 LIKE ?")
            params.append(f"%{quantity}%")

        # MF/MT numbers
        if mf is not None:
            conditions.append("MF = ?")
            params.append(mf)
        if mt is not None:
            conditions.append("MT = ?")
            params.append(mt)

        # Year range
        if year_min is not None:
            conditions.append("year1 >= ?")
            params.append(year_min)
        if year_max is not None:
            conditions.append("year1 <= ?")
            params.append(year_max)

        # Author
        if author:
            conditions.append("author1 LIKE ?")
            params.append(f"%{author}%")

        # Build final query
        query = "SELECT DatasetID FROM x4pro_ds"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        cursor = conn.execute(query, params)
        return [row[0] for row in cursor.fetchall()]

    def get_dataset_json(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the raw JSON data for a dataset.

        Parameters
        ----------
        dataset_id : str
            Dataset ID (e.g., "10012002")

        Returns
        -------
        Dict[str, Any] or None
            Parsed JSON from jx5z column, or None if not found
        """
        conn = self._get_connection()

        cursor = conn.execute(
            "SELECT jx5z FROM x4pro_x5z WHERE DatasetID = ?", (dataset_id,)
        )
        row = cursor.fetchone()

        if row is None:
            return None

        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return None

    def get_dataset_metadata(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a dataset from x4pro_ds table.

        Parameters
        ----------
        dataset_id : str
            Dataset ID

        Returns
        -------
        Dict[str, Any] or None
            Dataset metadata
        """
        conn = self._get_connection()

        cursor = conn.execute(
            """
            SELECT DatasetID, year1, author1, Targ1, Proj, MF, MT, ndat, quant1, reacode
            FROM x4pro_ds
            WHERE DatasetID = ?
            """,
            (dataset_id,),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        # Extract energy range from JSON data
        e_min, e_max = None, None
        try:
            jx5z = self.get_dataset_json(dataset_id)
            if jx5z:
                e_min, e_max = self._extract_energy_range(jx5z)
        except Exception:
            pass  # Energy extraction failed, continue without energy

        return {
            "dataset_id": row["DatasetID"],
            "year": row["year1"],
            "author": row["author1"],
            "target": row["Targ1"],
            "projectile": row["Proj"],
            "mf": row["MF"],
            "mt": row["MT"],
            "ndat": row["ndat"],
            "quant": row["quant1"],
            "reacode": row["reacode"],
            "e_min": e_min,  # Energy min in MeV
            "e_max": e_max,  # Energy max in MeV
        }

    def parse_dataset(self, dataset_id: str) -> Optional[X4ProDataset]:
        """
        Parse a dataset from the database into X4ProDataset.

        This method prefers c5data (corrected data) over x4data (raw data)
        because c5data contains:
        - Pre-applied corrections (decay data, monitor renormalization)
        - Standardized units (EV, ADEG, B/SR)
        - Cleaner structure

        Parameters
        ----------
        dataset_id : str
            Dataset ID

        Returns
        -------
        X4ProDataset or None
            Parsed dataset, or None if not found
        """
        # Get metadata
        metadata = self.get_dataset_metadata(dataset_id)
        if metadata is None:
            return None

        # Get JSON data
        jx5z = self.get_dataset_json(dataset_id)
        if jx5z is None:
            return None

        # Try c5data first (contains corrected values with standardized units)
        parsed = _parse_c5data_json(jx5z)
        is_corrected = parsed["is_corrected"]
        correction_notes = parsed["correction_notes"]

        # Convert to numpy arrays
        energies = np.array(parsed["energies"], dtype=float) if parsed["energies"] else np.array([])
        angles = np.array(parsed["angles"], dtype=float) if parsed["angles"] else np.array([])
        values = np.array(parsed["values"], dtype=float) if parsed["values"] else np.array([])
        uncertainties = np.array(parsed["uncertainties"], dtype=float) if parsed["uncertainties"] else np.array([])
        energy_unit = parsed["energy_unit"]
        xs_unit = parsed["xs_unit"]
        angle_type = parsed["angle_type"]

        # Check x4data for PER-CENT uncertainties - c5data may have incorrect conversion
        # The X4Pro database sometimes incorrectly processes PER-CENT uncertainties in c5data
        x4_parsed = _parse_x4data_json(jx5z)
        x4_unc_unit = x4_parsed.get("uncertainty_unit", "")
        if _is_percent_unit(x4_unc_unit) and x4_parsed["uncertainties"] and x4_parsed["values"]:
            # x4data has percentage uncertainties - convert properly using x4data values
            x4_values = np.array(x4_parsed["values"], dtype=float)
            x4_unc_raw = np.array(x4_parsed["uncertainties"], dtype=float)
            uncertainties = _convert_percent_to_absolute(x4_unc_raw, x4_values, x4_unc_unit)
            # Note: uncertainties are now in the same units as x4_parsed xs_unit
            # We need to ensure they match the c5data values which may have different scaling
            # Since we're using x4_values for the conversion, apply the same unit conversion
            if x4_parsed["xs_unit"].upper() != xs_unit.upper() and len(values) > 0:
                # Scale uncertainties to match c5data value scale
                # This handles cases where c5data values are in different units than x4data
                scale_factor = np.mean(np.abs(values)) / np.mean(np.abs(x4_values)) if np.mean(np.abs(x4_values)) > 0 else 1.0
                uncertainties = uncertainties * scale_factor

        # Fallback to x4data if c5data is empty/incomplete
        if len(values) == 0:
            # x4_parsed was already computed above for PER-CENT check
            if x4_parsed["values"]:
                values = np.array(x4_parsed["values"], dtype=float)
                xs_unit = x4_parsed["xs_unit"]
            if x4_parsed["uncertainties"]:
                x4_unc_raw = np.array(x4_parsed["uncertainties"], dtype=float)
                x4_unc_unit = x4_parsed.get("uncertainty_unit", "")
                # Convert PER-CENT to absolute if needed
                if _is_percent_unit(x4_unc_unit) and len(values) > 0:
                    uncertainties = _convert_percent_to_absolute(x4_unc_raw, values, x4_unc_unit)
                else:
                    uncertainties = x4_unc_raw

        if len(energies) == 0:
            x4_parsed = _parse_x4data_json(jx5z)
            if x4_parsed["energies"]:
                energies = np.array(x4_parsed["energies"], dtype=float)
                energy_unit = x4_parsed["energy_unit"]

        if len(angles) == 0:
            x4_parsed = _parse_x4data_json(jx5z)
            if x4_parsed["angles"]:
                angles = np.array(x4_parsed["angles"], dtype=float)
                angle_type = x4_parsed["angle_type"]

        # Ensure uncertainties array is initialized
        if len(uncertainties) == 0 and len(values) > 0:
            uncertainties = np.zeros_like(values)

        # Handle angle type (convert cosine to degrees if needed)
        if angle_type == "COS":
            angles = np.degrees(np.arccos(np.clip(angles, -1.0, 1.0)))
            angle_unit = "ADEG"
        else:
            angle_unit = parsed["angle_unit"]

        # Determine frame from reacode
        reacode = metadata.get("reacode", "")
        angle_frame = FRAME_CM if ",DA/DA,," in reacode or angle_type == "COS" else FRAME_LAB

        return X4ProDataset(
            dataset_id=dataset_id,
            year=metadata["year"],
            author=metadata["author"],
            target=metadata["target"],
            projectile=metadata["projectile"],
            mf=metadata["mf"],
            mt=metadata["mt"],
            quant=metadata["quant"],
            ndat=metadata["ndat"],
            reacode=metadata["reacode"],
            energies_ev=energies,
            angles_deg=angles,
            cross_sections=values,
            uncertainties=uncertainties,
            energy_unit=energy_unit,
            angle_unit=angle_unit,
            xs_unit=xs_unit,
            angle_frame=angle_frame,
            is_corrected=is_corrected,
            correction_notes=correction_notes,
            raw_json=jx5z,
            uncertainty_components=x4_parsed.get("uncertainty_components", []),
            # Always from x4data: c5data carries the corrected measurement, not
            # the beam's declared resolution, so there is no c5 equivalent to
            # prefer here the way there is for values and uncertainties.
            energy_resolution=x4_parsed.get("energy_resolution"),
        )

    def query_angular_distributions(
        self,
        target: Union[str, List[str]] = None,
        target_zaid: Union[int, List[int]] = None,
        projectile: str = "N",
        process: str = None,
        energy_range: Tuple[float, float] = None,
        mt: int = None,
        convert_to_objects: bool = True,
    ) -> Union[List["ExforAngularDistribution"], List[X4ProDataset]]:
        """
        Query angular distribution datasets from the database.

        This is the main method for retrieving EXFOR angular distribution data.
        It queries datasets matching the criteria and optionally converts them
        to ExforAngularDistribution objects.

        Parameters
        ----------
        target : str or List[str], optional
            Target in EXFOR notation (e.g., "26-FE-56" or ["Fe-56", "Fe-0"])
        target_zaid : int or List[int], optional
            Target ZAID (e.g., 26056 for Fe-56, or [26056, 26000] for Fe-56 + natural).
            Alternative to target.
        projectile : str, optional
            Projectile (default: "N" for neutrons)
        process : str, optional
            Reaction process (e.g., "EL" for elastic). If provided, filters by MT.
        energy_range : Tuple[float, float], optional
            Energy range (min, max) in MeV. Filters datasets with data in range.
        mt : int, optional
            ENDF MT number (e.g., 2 for elastic scattering)
        convert_to_objects : bool, optional
            If True (default), convert to ExforAngularDistribution objects.
            If False, return X4ProDataset objects.

        Returns
        -------
        List[ExforAngularDistribution] or List[X4ProDataset]
            List of angular distribution datasets

        Examples
        --------
        >>> db = X4ProDatabase()
        >>> # Get all elastic scattering data for Fe-56
        >>> datasets = db.query_angular_distributions(target_zaid=26056, mt=2)
        >>> # Get data for both Fe-56 and natural iron
        >>> datasets = db.query_angular_distributions(target_zaid=[26056, 26000], mt=2)
        >>> # Get data in specific energy range
        >>> datasets = db.query_angular_distributions(
        ...     target_zaid=26056,
        ...     energy_range=(1.0, 3.0)
        ... )
        """
        # Determine MT from process if not specified
        if mt is None and process:
            process_to_mt = {"EL": 2, "INL": 4, "N,N'": 4, "TOT": 1}
            mt = process_to_mt.get(process.upper())

        # Query dataset IDs
        dataset_ids = self.query_dataset_ids(
            target=target,
            target_zaid=target_zaid,
            projectile=projectile,
            quantity="DA",
            mf=4,  # MF=4 for angular distributions
            mt=mt,
        )

        # Parse datasets
        parsed_datasets = []
        for ds_id in dataset_ids:
            dataset = self.parse_dataset(ds_id)
            if dataset is None:
                continue

            # Apply energy filter if specified
            if energy_range is not None:
                e_min, e_max = energy_range
                # Convert energy range to eV for comparison
                e_min_ev = e_min * 1e6
                e_max_ev = e_max * 1e6

                # Check if dataset has data in range
                ds_energies = dataset.energies_ev
                if dataset.energy_unit.upper() == "MEV":
                    ds_energies = ds_energies * 1e6
                elif dataset.energy_unit.upper() == "KEV":
                    ds_energies = ds_energies * 1e3

                if len(ds_energies) == 0:
                    continue
                if np.max(ds_energies) < e_min_ev or np.min(ds_energies) > e_max_ev:
                    continue

            parsed_datasets.append(dataset)

        if not convert_to_objects:
            return parsed_datasets

        # Convert to ExforAngularDistribution objects
        return [self._convert_to_exfor_object(ds) for ds in parsed_datasets]

    def _convert_to_exfor_object(
        self, dataset: X4ProDataset
    ) -> "ExforEntry":
        """
        Convert X4ProDataset to appropriate ExforEntry subclass.

        The object type is determined by the quantity field (quant):
        - "DA" -> ExforAngularDistribution
        - "SIG", "CS" -> ExforCrossSection
        - Other types -> ExforExperiment (generic fallback)

        Parameters
        ----------
        dataset : X4ProDataset
            Parsed dataset from database

        Returns
        -------
        ExforEntry
            Appropriate ExforEntry subclass based on data type
        """
        quantity = dataset.quant.upper() if dataset.quant else ""

        # Check for angular distribution (DA)
        if "DA" in quantity:
            return self._convert_to_angular_distribution(dataset)

        # Check for cross section (SIG, CS)
        if "SIG" in quantity or quantity == "CS" or quantity.startswith("CS"):
            return self._convert_to_cross_section(dataset)

        # Fallback to generic ExforExperiment for other quantities
        return self._convert_to_experiment_generic(dataset)

    def _convert_to_angular_distribution(
        self, dataset: X4ProDataset
    ) -> "ExforAngularDistribution":
        """
        Convert X4ProDataset to ExforAngularDistribution.

        Parameters
        ----------
        dataset : X4ProDataset
            Parsed dataset from database

        Returns
        -------
        ExforAngularDistribution
            Full ExforAngularDistribution object
        """
        from kika.exfor.angular_distribution import ExforAngularDistribution

        # Parse target
        target_name, target_zaid = _parse_target_from_db(dataset.target)

        # Convert energies to MeV
        energies_mev = _convert_units(
            dataset.energies_ev,
            dataset.energy_unit,
            "MEV",
            "energy",
        )

        # Convert cross sections to b/sr
        xs_bsr = _convert_units(
            dataset.cross_sections,
            dataset.xs_unit,
            "B/SR",
            "cross_section",
        )
        unc_bsr = _convert_units(
            dataset.uncertainties,
            dataset.xs_unit,
            "B/SR",
            "cross_section",
        )

        # Group data by energy. Per-point sigma_stat / sigma_sys are populated
        # below by apply_manifest_to_exfor() after the ExforAngularDistribution
        # is constructed; for now we seed uncertainty_stat with the legacy
        # per-point unc_bsr so that fallback behavior is preserved if manifest
        # resolution fails.
        unique_energies = np.unique(energies_mev)
        data_blocks = []

        for energy in unique_energies:
            mask = np.isclose(energies_mev, energy, rtol=1e-6)
            block_angles = dataset.angles_deg[mask]
            block_xs = xs_bsr[mask]
            block_unc = unc_bsr[mask]

            # Sort by angle
            sort_idx = np.argsort(block_angles)
            block_angles = block_angles[sort_idx]
            block_xs = block_xs[sort_idx]
            block_unc = block_unc[sort_idx]

            data_points = []
            for i in range(len(block_angles)):
                data_points.append({
                    "angle": float(block_angles[i]),
                    "cross_section": float(block_xs[i]),
                    "uncertainty_stat": float(block_unc[i]),
                    "uncertainty_sys": 0.0,
                })

            data_blocks.append({
                "value": float(energy),
                "data": data_points,
            })

        # Extract entry/subentry from dataset_id
        entry = dataset.dataset_id[:5]
        subentry = dataset.dataset_id[5:]

        # Build citation
        author_parts = dataset.author.split(".")
        surname = author_parts[-1] if author_parts else dataset.author

        citation = {
            "authors": [dataset.author],
            "year": dataset.year,
            "reference": f"EXFOR {dataset.dataset_id}",
        }

        # Build reaction
        reaction = {
            "target": target_name,
            "target_zaid": target_zaid,
            "projectile": dataset.projectile.lower(),
            "process": "EL" if dataset.mt == 2 else f"MT{dataset.mt}",
            "notation": dataset.reacode,
        }

        # Get TOF parameters from metadata file
        tof_params = _get_tof_params_for_experiment(dataset.dataset_id)
        energy_resolution_input = {
            "distance": {
                "value": tof_params["flight_path_m"],
                "unit": "m",
            },
            "time_resolution": {
                "value": tof_params["time_resolution_ns"],
                "unit": "ns",
            },
            "source": tof_params.get("source", "default"),
        }

        # The experiment's own declared resolution outranks anything curated
        # externally, and covers far more of the corpus. It is *added* rather
        # than substituted: it is a width, not a (L, dt) pair, so a consumer
        # that can only fold from a flight path still has something to use.
        # `source` says which of the two is the authoritative one.
        declared = dataset.energy_resolution
        if declared is not None:
            energy_resolution_input["declared"] = declared
            energy_resolution_input["source"] = {
                "full_width": "exfor_rsl_fw",
                "half_width": "exfor_rsl_hw",
                "unspecified": "exfor_rsl_assumed",
            }.get(declared.get("convention"), "exfor_rsl_assumed")

        method = {
            "type": "TOF",
            "energy_resolution_input": energy_resolution_input,
        }

        ad = ExforAngularDistribution(
            entry=entry,
            subentry=subentry,
            quantity="DA",
            citation=citation,
            reaction=reaction,
            facility={},
            method=method,
            angle_frame=dataset.angle_frame,
            units={"energy": "MeV", "angle": "deg", "cross_section": "b/sr"},
            _data_blocks=data_blocks,
        )
        # Stash raw uncertainty_components on the object so the pipeline-side
        # manifest resolver (scripts/uncertainty_manifest.py) can reference
        # named EXFOR columns when applying ERR-ANALYS-derived overrides.
        # The kika library itself does not apply the manifest — this is
        # pipeline-specific behavior, called from scripts/exfor_utils.py.
        ad._raw_uncertainty_components = list(dataset.uncertainty_components)
        ad.sigma_sys_scalar_relative = 0.0
        ad.sigma_sys_indep_relative = 0.0
        ad.uncertainty_manifest_flag = "default"
        return ad

    def _convert_to_cross_section(
        self, dataset: X4ProDataset
    ) -> "ExforCrossSection":
        """
        Convert X4ProDataset to ExforCrossSection.

        Parameters
        ----------
        dataset : X4ProDataset
            Parsed dataset from database

        Returns
        -------
        ExforCrossSection
            Cross section object with energy-dependent data
        """
        from kika.exfor.cross_section import ExforCrossSection

        # Parse target
        target_name, target_zaid = _parse_target_from_db(dataset.target)

        # Convert energies to MeV
        energies_mev = _convert_units(
            dataset.energies_ev,
            dataset.energy_unit,
            "MEV",
            "energy",
        )

        # Convert cross sections to barns (from b/sr to b for total XS)
        # Note: cross section data from database is typically in barns, not b/sr
        xs_unit = dataset.xs_unit.upper()
        if "/SR" in xs_unit:
            # This is differential - convert to b/sr then use that
            xs_b = _convert_units(
                dataset.cross_sections,
                dataset.xs_unit,
                "B/SR",
                "cross_section",
            )
            unc_b = _convert_units(
                dataset.uncertainties,
                dataset.xs_unit,
                "B/SR",
                "cross_section",
            )
            xs_unit_out = "b/sr"
        else:
            # Total cross section in barns
            # Map common units to barns
            xs_factor_map = {
                "B": 1.0, "MB": 1e-3, "UB": 1e-6, "MUB": 1e-6, "NB": 1e-9,
            }
            factor = xs_factor_map.get(xs_unit.replace("/SR", ""), 1.0)
            xs_b = dataset.cross_sections * factor
            unc_b = dataset.uncertainties * factor
            xs_unit_out = "b"

        # Build DataFrame
        import pandas as pd
        data_df = pd.DataFrame({
            "energy": energies_mev,
            "cross_section": xs_b,
            "error": unc_b,
        })

        # Remove rows with NaN or zero energy
        data_df = data_df[data_df["energy"] > 0].dropna(subset=["energy", "cross_section"])
        data_df = data_df.sort_values("energy").reset_index(drop=True)

        # Extract entry/subentry from dataset_id
        entry = dataset.dataset_id[:5]
        subentry = dataset.dataset_id[5:]

        # Build citation
        citation = {
            "authors": [dataset.author],
            "year": dataset.year,
            "reference": f"EXFOR {dataset.dataset_id}",
        }

        # Build reaction
        reaction = {
            "target": target_name,
            "target_zaid": target_zaid,
            "projectile": dataset.projectile.lower(),
            "process": "TOT" if dataset.mt == 1 else f"MT{dataset.mt}",
            "notation": dataset.reacode,
        }

        return ExforCrossSection(
            entry=entry,
            subentry=subentry,
            quantity=dataset.quant,
            citation=citation,
            reaction=reaction,
            facility={},
            method={},
            units={"energy": "MeV", "cross_section": xs_unit_out},
            _data=data_df,
        )

    def _convert_to_experiment_generic(
        self, dataset: X4ProDataset
    ) -> "ExforExperiment":
        """
        Convert X4ProDataset to generic ExforExperiment.

        This is a fallback for quantity types that don't have specialized classes.

        Parameters
        ----------
        dataset : X4ProDataset
            Parsed dataset from database

        Returns
        -------
        ExforExperiment
            Generic experiment object
        """
        # Parse target
        target_name, target_zaid = _parse_target_from_db(dataset.target)

        # Extract entry/subentry
        entry = dataset.dataset_id[:5]
        subentry = dataset.dataset_id[5:]

        # Build citation
        citation = {
            "authors": [dataset.author],
            "year": dataset.year,
            "reference": f"EXFOR {dataset.dataset_id}",
        }

        # Build reaction
        reaction = {
            "target": target_name,
            "target_zaid": target_zaid,
            "projectile": dataset.projectile.lower(),
            "notation": dataset.reacode,
        }

        # Build generic DataFrame from available data
        import pandas as pd
        data_dict = {}

        if len(dataset.energies_ev) > 0:
            energies_mev = _convert_units(
                dataset.energies_ev,
                dataset.energy_unit,
                "MEV",
                "energy",
            )
            data_dict["energy"] = energies_mev

        if len(dataset.angles_deg) > 0:
            data_dict["angle"] = dataset.angles_deg

        if len(dataset.cross_sections) > 0:
            data_dict["value"] = dataset.cross_sections

        if len(dataset.uncertainties) > 0:
            data_dict["error"] = dataset.uncertainties

        data_df = pd.DataFrame(data_dict)

        # Determine independent variables
        ind_vars = []
        if "energy" in data_df.columns:
            ind_vars.append("energy")
        if "angle" in data_df.columns:
            ind_vars.append("angle")

        return ExforExperiment(
            entry=entry,
            subentry=subentry,
            quantity=dataset.quant,
            citation=citation,
            reaction=reaction,
            facility={},
            method={},
            independent_vars=ind_vars,
            dependent_var="value",
            units={"energy": "MeV"},
            _data=data_df,
        )

    # =========================================================================
    # Cross Section Query Methods
    # =========================================================================

    def query_cross_sections(
        self,
        target: str = None,
        target_zaid: int = None,
        projectile: str = "N",
        mt: int = None,
        energy_range: Tuple[float, float] = None,
        convert_to_objects: bool = True,
    ) -> Union[List["ExforCrossSection"], List[X4ProDataset]]:
        """
        Query cross section datasets from the database.

        This method queries datasets with quantity type SIG/CS and optionally
        converts them to ExforCrossSection objects.

        Parameters
        ----------
        target : str, optional
            Target in EXFOR notation (e.g., "26-FE-56") or "Fe56" format
        target_zaid : int, optional
            Target ZAID (e.g., 26056 for Fe-56). Alternative to target.
        projectile : str, optional
            Projectile (default: "N" for neutrons)
        mt : int, optional
            ENDF MT number (e.g., 1 for total, 2 for elastic, 18 for fission)
        energy_range : Tuple[float, float], optional
            Energy range (min, max) in MeV. Filters datasets with data in range.
        convert_to_objects : bool, optional
            If True (default), convert to ExforCrossSection objects.
            If False, return X4ProDataset objects.

        Returns
        -------
        List[ExforCrossSection] or List[X4ProDataset]
            List of cross section datasets

        Examples
        --------
        >>> db = X4ProDatabase()
        >>> # Get total cross section data for Fe-56
        >>> datasets = db.query_cross_sections(target_zaid=26056, mt=1)
        >>> # Get elastic scattering cross sections
        >>> datasets = db.query_cross_sections(target="Fe56", mt=2)
        """
        # Query dataset IDs - use SIG quantity and MF=3 (cross sections)
        dataset_ids = self.query_dataset_ids(
            target=target,
            target_zaid=target_zaid,
            projectile=projectile,
            quantity="SIG",
            mf=3,  # MF=3 for cross sections
            mt=mt,
        )

        # Also query for CS quantity code
        dataset_ids_cs = self.query_dataset_ids(
            target=target,
            target_zaid=target_zaid,
            projectile=projectile,
            quantity="CS",
            mf=3,
            mt=mt,
        )

        # Combine and deduplicate
        all_dataset_ids = list(set(dataset_ids + dataset_ids_cs))

        # Parse datasets
        parsed_datasets = []
        for ds_id in all_dataset_ids:
            dataset = self.parse_dataset(ds_id)
            if dataset is None:
                continue

            # Apply energy filter if specified
            if energy_range is not None:
                e_min, e_max = energy_range
                e_min_ev = e_min * 1e6
                e_max_ev = e_max * 1e6

                ds_energies = dataset.energies_ev
                if dataset.energy_unit.upper() == "MEV":
                    ds_energies = ds_energies * 1e6
                elif dataset.energy_unit.upper() == "KEV":
                    ds_energies = ds_energies * 1e3

                if len(ds_energies) == 0:
                    continue
                if np.max(ds_energies) < e_min_ev or np.min(ds_energies) > e_max_ev:
                    continue

            parsed_datasets.append(dataset)

        if not convert_to_objects:
            return parsed_datasets

        # Convert to ExforCrossSection objects
        return [self._convert_to_cross_section(ds) for ds in parsed_datasets]

    # =========================================================================
    # General Query Methods (for any quantity type)
    # =========================================================================

    def list_unique_quantities(
        self,
        projectile: str = "n",
        target: str = None,
    ) -> pd.DataFrame:
        """
        List all unique quantity codes in the database.

        Parameters
        ----------
        projectile : str, optional
            Projectile filter (default: "n" for neutrons)
        target : str, optional
            Target filter (e.g., "Fe-56", "Fe56", 26056)

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: quantity, description, count
        """
        conn = self._get_connection()
        conditions = []
        params = []

        proj_lower = projectile.lower()
        conditions.append("(Proj = ? OR Proj = ?)")
        params.extend([proj_lower, projectile.upper()])

        if target:
            target_pattern = self._normalize_target(target)
            conditions.append("Targ1 = ?")
            params.append(target_pattern)

        query = f"""
            SELECT quant1, COUNT(*) as count
            FROM x4pro_ds
            WHERE {' AND '.join(conditions)}
            GROUP BY quant1
            ORDER BY count DESC
        """

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            quant = row[0] if row[0] else "Unknown"
            results.append({
                "quantity": quant,
                "description": EXFOR_QUANTITY_CODES.get(quant, "Unknown/custom"),
                "count": row[1],
            })

        return pd.DataFrame(results)

    def list_unique_reactions(
        self,
        projectile: str = "n",
        target: str = None,
    ) -> pd.DataFrame:
        """
        List all unique reaction codes in the database.

        Parameters
        ----------
        projectile : str, optional
            Projectile filter (default: "n" for neutrons)
        target : str, optional
            Target filter

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: reacode, MT, count
        """
        conn = self._get_connection()
        conditions = []
        params = []

        proj_lower = projectile.lower()
        conditions.append("(Proj = ? OR Proj = ?)")
        params.extend([proj_lower, projectile.upper()])

        if target:
            target_pattern = self._normalize_target(target)
            conditions.append("Targ1 = ?")
            params.append(target_pattern)

        query = f"""
            SELECT reacode, MT, COUNT(*) as count
            FROM x4pro_ds
            WHERE {' AND '.join(conditions)}
            GROUP BY reacode
            ORDER BY count DESC
        """

        cursor = conn.execute(query, params)
        results = [{"reacode": row[0], "MT": row[1], "count": row[2]} for row in cursor.fetchall()]
        return pd.DataFrame(results)

    def query_experiments(
        self,
        targets: Union[str, List[str]] = None,
        projectile: str = "n",
        quantity: str = None,
        mt: int = None,
        mf: int = None,
        energy_min_mev: float = None,
        energy_max_mev: float = None,
        year_min: int = None,
        year_max: int = None,
        author: str = None,
    ) -> List[str]:
        """
        General query for experiments with flexible filtering.

        Supports multiple targets with OR logic.

        Parameters
        ----------
        targets : str or List[str], optional
            Single target or list of targets (OR logic)
        projectile : str, optional
            Projectile (default: "n")
        quantity : str, optional
            Quantity code (e.g., "SIG", "DA", "FY")
        mt : int, optional
            ENDF MT number
        mf : int, optional
            ENDF MF number
        energy_min_mev : float, optional
            Minimum energy in MeV
        energy_max_mev : float, optional
            Maximum energy in MeV
        year_min : int, optional
            Minimum publication year
        year_max : int, optional
            Maximum publication year
        author : str, optional
            Author name (partial match)

        Returns
        -------
        List[str]
            List of dataset IDs
        """
        conn = self._get_connection()
        conditions = []
        params = []

        # Handle multiple targets (OR logic)
        if targets is not None:
            if isinstance(targets, str):
                targets = [targets]
            target_conditions = []
            for t in targets:
                pattern = self._normalize_target(t)
                target_conditions.append("Targ1 = ?")
                params.append(pattern)
            if len(target_conditions) == 1:
                conditions.append(target_conditions[0])
            else:
                conditions.append(f"({' OR '.join(target_conditions)})")

        # Projectile
        if projectile:
            proj_lower = projectile.lower()
            conditions.append("(Proj = ? OR Proj = ?)")
            params.extend([proj_lower, projectile.upper()])

        # Quantity (partial match)
        if quantity:
            conditions.append("quant1 LIKE ?")
            params.append(f"%{quantity}%")

        # MF/MT
        if mf is not None:
            conditions.append("MF = ?")
            params.append(mf)
        if mt is not None:
            conditions.append("MT = ?")
            params.append(mt)

        # Year range
        if year_min is not None:
            conditions.append("year1 >= ?")
            params.append(year_min)
        if year_max is not None:
            conditions.append("year1 <= ?")
            params.append(year_max)

        # Author
        if author:
            conditions.append("author1 LIKE ?")
            params.append(f"%{author}%")

        query = "SELECT DatasetID FROM x4pro_ds"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        cursor = conn.execute(query, params)
        return [row[0] for row in cursor.fetchall()]

    def list_experiments_general(
        self,
        targets: Union[str, List[str]] = None,
        projectile: str = "n",
        quantity: str = None,
        mt: int = None,
        year_min: int = None,
        year_max: int = None,
        author: str = None,
    ) -> pd.DataFrame:
        """
        List experiments matching criteria with detailed info.

        Parameters
        ----------
        targets : str or List[str], optional
            Single target or list of targets (OR logic)
        projectile : str, optional
            Projectile (default: "n")
        quantity : str, optional
            Quantity code
        mt : int, optional
            ENDF MT number
        year_min : int, optional
            Minimum publication year
        year_max : int, optional
            Maximum publication year
        author : str, optional
            Author name (partial match)

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: dataset_id, author, year, target, quantity,
            reacode, ndat, energy_min_mev, energy_max_mev
        """
        dataset_ids = self.query_experiments(
            targets=targets, projectile=projectile, quantity=quantity,
            mt=mt, year_min=year_min, year_max=year_max, author=author
        )

        results = []
        for ds_id in dataset_ids:
            metadata = self.get_dataset_metadata(ds_id)
            if metadata:
                # Get energy range from JSON
                jx5z = self.get_dataset_json(ds_id)
                e_min, e_max = self._extract_energy_range(jx5z)

                results.append({
                    "dataset_id": ds_id,
                    "author": metadata["author"],
                    "year": metadata["year"],
                    "target": metadata["target"],
                    "quantity": metadata["quant"],
                    "reacode": metadata["reacode"],
                    "ndat": metadata["ndat"],
                    "energy_min_mev": e_min,
                    "energy_max_mev": e_max,
                })

        return pd.DataFrame(results)

    def load_experiment_general(self, dataset_id: str) -> "ExforExperiment":
        """
        Load any experiment as a general ExforExperiment object.

        Works with ANY quantity type (SIG, DA, FY, NU, etc.)

        Parameters
        ----------
        dataset_id : str
            EXFOR dataset identifier (e.g., "10571002")

        Returns
        -------
        ExforExperiment
            General experiment object with data and metadata

        Raises
        ------
        ValueError
            If dataset is not found
        """
        from kika.exfor.experiment import ExforExperiment

        metadata = self.get_dataset_metadata(dataset_id)
        if metadata is None:
            raise ValueError(f"Dataset {dataset_id} not found")

        jx5z = self.get_dataset_json(dataset_id)
        if jx5z is None:
            raise ValueError(f"No JSON data for dataset {dataset_id}")

        # Parse data generically
        data_df, units, ind_vars, dep_var = self._parse_general_data(jx5z)

        entry = dataset_id[:5]
        subentry = dataset_id[5:]

        citation = {
            "authors": [metadata["author"]],
            "year": metadata["year"],
            "reference": f"EXFOR {dataset_id}",
        }

        target_name, target_zaid = _parse_target_from_db(metadata["target"])

        reaction = {
            "target": target_name,
            "target_zaid": target_zaid,
            "projectile": metadata["projectile"].lower(),
            "notation": metadata["reacode"],
        }

        return ExforExperiment(
            entry=entry,
            subentry=subentry,
            quantity=metadata["quant"],
            citation=citation,
            reaction=reaction,
            facility={},
            method={},
            units=units,
            independent_vars=ind_vars,
            dependent_var=dep_var,
            _data=data_df,
        )

    def _parse_general_data(
        self, jx5z: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, str], List[str], str]:
        """
        Parse x4data/c5data for any quantity type into DataFrame.

        Parameters
        ----------
        jx5z : Dict[str, Any]
            Parsed JSON from database

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, str], List[str], str]
            (DataFrame, units_dict, independent_vars, dependent_var)
        """
        c5data = jx5z.get("c5data", {})
        x4data = jx5z.get("x4data", [])

        columns = {}
        units = {}
        ind_vars = []
        dep_var = "value"

        # Try c5data first
        if isinstance(c5data, dict):
            if "y" in c5data:
                y_data = c5data["y"]
                columns["value"] = y_data.get("y", [])
                columns["error"] = y_data.get("dy", [])
                units["value"] = y_data.get("units", "")

            for i in [1, 2, 3]:
                key = f"x{i}"
                if key in c5data:
                    x_data = c5data[key]
                    fam = x_data.get("fam", "")
                    var_name = EXFOR_FAMILY_TO_VARIABLE.get(fam, fam.lower() or f"x{i}")
                    columns[var_name] = x_data.get(key, [])
                    units[var_name] = x_data.get("units", "")
                    ind_vars.append(var_name)

        # Fallback to x4data
        if not columns.get("value"):
            for var in x4data:
                fam = var.get("fam", "")
                cvar = var.get("cvar", "")
                dat0 = var.get("dat0", [])
                unit = var.get("units", "")

                if cvar == "y":
                    columns["value"] = dat0
                    units["value"] = unit
                elif cvar.startswith("x"):
                    var_name = EXFOR_FAMILY_TO_VARIABLE.get(fam, fam.lower() or cvar)
                    columns[var_name] = dat0
                    units[var_name] = unit
                    if var_name not in ind_vars:
                        ind_vars.append(var_name)
                elif cvar == "dy":
                    columns["error"] = dat0

        # Ensure all arrays have the same length
        if columns:
            max_length = max(len(arr) if isinstance(arr, list) else 1 for arr in columns.values())
            for key, arr in columns.items():
                if isinstance(arr, list):
                    if len(arr) < max_length:
                        # Pad shorter arrays with appropriate values
                        if key == "error":
                            columns[key] = arr + [0.0] * (max_length - len(arr))
                        else:
                            # For other columns, pad with the last value or NaN
                            pad_value = arr[-1] if arr else np.nan
                            columns[key] = arr + [pad_value] * (max_length - len(arr))
                else:
                    # Convert single values to lists of the appropriate length
                    columns[key] = [arr] * max_length

        df = pd.DataFrame(columns)
        if "error" not in df.columns and "value" in df.columns:
            df["error"] = 0.0

        return df, units, ind_vars, dep_var

    def _extract_energy_range(
        self, jx5z: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Extract energy range from JSON data, return in MeV.

        Parameters
        ----------
        jx5z : Dict[str, Any] or None
            Parsed JSON from database

        Returns
        -------
        Tuple[Optional[float], Optional[float]]
            (min_energy_mev, max_energy_mev) or (None, None)
        """
        if not jx5z:
            return None, None

        # Try c5data first
        c5data = jx5z.get("c5data", {})
        if isinstance(c5data, dict) and "x1" in c5data:
            x1 = c5data["x1"]
            if x1.get("fam") == "EN":
                energies = np.array(x1.get("x1", []), dtype=float)
                unit = x1.get("units", "EV").upper()
                energies = energies[~np.isnan(energies)]
                if len(energies) > 0:
                    if unit == "EV":
                        energies = energies / 1e6
                    elif unit == "KEV":
                        energies = energies / 1e3
                    return float(np.min(energies)), float(np.max(energies))

        # Try x4data
        x4data = jx5z.get("x4data", [])
        for var in x4data:
            if var.get("fam") == "EN":
                energies = np.array(var.get("dat0", []), dtype=float)
                unit = var.get("units", "EV").upper()
                energies = energies[~np.isnan(energies)]
                if len(energies) > 0:
                    if unit == "EV":
                        energies = energies / 1e6
                    elif unit == "KEV":
                        energies = energies / 1e3
                    return float(np.min(energies)), float(np.max(energies))

        return None, None

    def get_statistics(self) -> Dict[str, int]:
        """
        Get database statistics.

        Returns
        -------
        Dict[str, int]
            Statistics including total datasets, angular distributions, etc.
        """
        conn = self._get_connection()

        stats = {}

        # Total datasets
        cursor = conn.execute("SELECT COUNT(*) FROM x4pro_x5z")
        stats["total_datasets"] = cursor.fetchone()[0]

        # Total metadata entries
        cursor = conn.execute("SELECT COUNT(*) FROM x4pro_ds")
        stats["total_metadata"] = cursor.fetchone()[0]

        # Angular distributions (DA)
        cursor = conn.execute(
            "SELECT COUNT(*) FROM x4pro_ds WHERE quant1 LIKE '%DA%'"
        )
        stats["angular_distributions"] = cursor.fetchone()[0]

        # Elastic scattering
        cursor = conn.execute(
            "SELECT COUNT(*) FROM x4pro_ds WHERE quant1 LIKE '%DA%' AND MT = 2"
        )
        stats["elastic_scattering"] = cursor.fetchone()[0]

        return stats

    def list_targets(self, projectile: str = "n") -> List[str]:
        """
        List all unique targets with angular distribution data.

        Parameters
        ----------
        projectile : str, optional
            Filter by projectile (default: "n" for neutrons)

        Returns
        -------
        List[str]
            Sorted list of unique target strings (e.g., ["Fe-54", "Fe-56", "Fe-57"])

        Examples
        --------
        >>> db = X4ProDatabase()
        >>> targets = db.list_targets()
        >>> print(targets[:5])
        ['Ag-0', 'Ag-107', 'Ag-109', 'Al-27', 'Am-241']
        """
        conn = self._get_connection()
        proj_lower = projectile.lower()

        cursor = conn.execute(
            """SELECT DISTINCT Targ1 FROM x4pro_ds
               WHERE (Proj = ? OR Proj = ?) AND quant1 LIKE '%DA%'""",
            (proj_lower, projectile.upper()),
        )
        targets = [row[0] for row in cursor.fetchall() if row[0]]
        return sorted(targets)

    def list_experiments(
        self,
        target: Union[str, int] = None,
        projectile: str = "n",
        mt: int = None,
    ) -> "pd.DataFrame":
        """
        List all experiments for a target with summary info.

        Parameters
        ----------
        target : str or int, optional
            Target isotope. Accepts multiple formats:
            - "Fe56" or "Fe-56" (symbol + mass)
            - 26056 (ZAID)
            - None to list all targets
        projectile : str, optional
            Projectile (default: "n" for neutrons)
        mt : int, optional
            ENDF MT number (e.g., 2 for elastic)

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: dataset_id, author, year, energy_min, energy_max

        Examples
        --------
        >>> db = X4ProDatabase()
        >>> experiments = db.list_experiments("Fe56", mt=2)
        >>> print(experiments)
           dataset_id    author  year  energy_min  energy_max
        0    10037024  Boschung  1971        0.01        0.01
        1    10571002    Kinney  1970        4.07        8.56
        """
        import pandas as pd

        conn = self._get_connection()

        # Build query conditions
        conditions = ["quant1 LIKE '%DA%'"]
        params = []

        # Handle target - accept multiple formats
        if target is not None:
            target_pattern = self._normalize_target(target)
            if target_pattern:
                conditions.append("Targ1 = ?")
                params.append(target_pattern)

        # Projectile
        proj_lower = projectile.lower()
        conditions.append("(Proj = ? OR Proj = ?)")
        params.extend([proj_lower, projectile.upper()])

        # MT number
        if mt is not None:
            conditions.append("MT = ?")
            params.append(mt)

        query = f"""
            SELECT DatasetID, author1, year1, Targ1, MT
            FROM x4pro_ds
            WHERE {' AND '.join(conditions)}
            ORDER BY year1, author1
        """

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()

        # Get energy ranges from JSON data
        results = []
        for row in rows:
            dataset_id = row[0]
            # Get energy range from parsed data
            jx5z = self.get_dataset_json(dataset_id)
            e_min, e_max = None, None
            if jx5z:
                parsed = _parse_x4data_json(jx5z)
                energies = None
                unit = None

                # Try x4data first
                if parsed["energies"]:
                    energies = np.array(parsed["energies"], dtype=float)
                    unit = parsed["energy_unit"].upper()

                # Fallback to c5data if x4data has no energies
                if energies is None or len(energies) == 0:
                    c5data = jx5z.get("c5data", {})
                    if isinstance(c5data, dict) and "x1" in c5data:
                        x1 = c5data["x1"]
                        if x1.get("fam") == "EN" and "x1" in x1:
                            c5_energies = x1.get("x1", [])
                            if c5_energies:
                                energies = np.array(c5_energies, dtype=float)
                                unit = x1.get("units", "EV").upper()

                if energies is not None and len(energies) > 0:
                    # Remove NaN values
                    energies = energies[~np.isnan(energies)]
                    if len(energies) > 0:
                        # Convert to MeV
                        if unit == "EV":
                            energies = energies / 1e6
                        elif unit == "KEV":
                            energies = energies / 1e3
                        e_min = float(np.min(energies))
                        e_max = float(np.max(energies))

            results.append({
                "dataset_id": dataset_id,
                "author": row[1],
                "year": row[2],
                "energy_min": e_min,
                "energy_max": e_max,
            })

        return pd.DataFrame(results)

    def load_experiment(self, dataset_id: str) -> "ExforEntry":
        """
        Load a specific experiment by its dataset ID.

        Returns the appropriate ExforEntry subclass based on data type:
        - Angular distributions (DA) -> ExforAngularDistribution
        - Other types -> NotImplementedError (for now)

        Parameters
        ----------
        dataset_id : str
            EXFOR dataset identifier (e.g., "10037024")

        Returns
        -------
        ExforEntry
            The loaded experiment data as appropriate subclass

        Raises
        ------
        ValueError
            If the dataset is not found
        NotImplementedError
            If the quantity type is not yet supported

        Examples
        --------
        >>> db = X4ProDatabase()
        >>> exp = db.load_experiment("10037024")
        >>> print(exp.label)
        Boschung (1971)
        >>> isinstance(exp, ExforAngularDistribution)
        True
        """
        dataset = self.parse_dataset(dataset_id)
        if dataset is None:
            raise ValueError(f"Dataset {dataset_id} not found in database")
        return self._convert_to_exfor_object(dataset)

    def create_subset_database(
        self,
        output_path: str,
        targets: List[str] = None,
        target_zaids: List[int] = None,
        quantity: str = "DA",
        projectile: str = "n",
        mt: int = None,
    ) -> int:
        """
        Create a new SQLite database with a subset of the data.

        Parameters
        ----------
        output_path : str
            Path for the new database file
        targets : List[str], optional
            List of targets in database format (e.g., ["Fe-0", "Fe-56"])
        target_zaids : List[int], optional
            List of target ZAIDs (e.g., [26000, 26056])
        quantity : str, optional
            Quantity type filter (default: "DA")
        projectile : str, optional
            Projectile filter (default: "n")
        mt : int, optional
            ENDF MT number filter (e.g., 2 for elastic)

        Returns
        -------
        int
            Number of datasets copied

        Examples
        --------
        >>> db = X4ProDatabase()
        >>> count = db.create_subset_database(
        ...     "iron_angular.db",
        ...     target_zaids=[26000, 26056],
        ...     mt=2
        ... )
        >>> print(f"Created database with {count} datasets")
        """
        import os

        # Convert ZAIDs to target patterns if provided
        target_patterns = []
        if targets:
            target_patterns.extend([self._normalize_target(t) for t in targets])
        if target_zaids:
            for zaid in target_zaids:
                pattern = _zaid_to_target_pattern(zaid)
                if pattern and pattern not in target_patterns:
                    target_patterns.append(pattern)

        if not target_patterns:
            raise ValueError("Must provide either targets or target_zaids")

        # Get source connection
        source_conn = self._get_connection()

        # Remove existing output file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)

        # Create new database with same schema
        dest_conn = sqlite3.connect(output_path)
        dest_conn.row_factory = sqlite3.Row

        # Create tables with same schema
        dest_conn.execute("""
            CREATE TABLE IF NOT EXISTS x4pro_ds (
                DatasetID TEXT PRIMARY KEY,
                year1 INTEGER,
                author1 TEXT,
                Targ1 TEXT,
                Proj TEXT,
                MF INTEGER,
                MT INTEGER,
                ndat INTEGER,
                quant1 TEXT,
                reacode TEXT
            )
        """)
        dest_conn.execute("""
            CREATE TABLE IF NOT EXISTS x4pro_x5z (
                DatasetID TEXT PRIMARY KEY,
                jx5z TEXT
            )
        """)

        # Build query conditions for each target
        copied_count = 0
        for target_pattern in target_patterns:
            conditions = ["Targ1 = ?"]
            params = [target_pattern]

            # Add quantity filter
            if quantity:
                conditions.append("quant1 LIKE ?")
                params.append(f"%{quantity}%")

            # Add projectile filter
            proj_lower = projectile.lower()
            conditions.append("(Proj = ? OR Proj = ?)")
            params.extend([proj_lower, projectile.upper()])

            # Add MT filter
            if mt is not None:
                conditions.append("MT = ?")
                params.append(mt)

            # Query matching dataset IDs
            query = f"""
                SELECT DatasetID FROM x4pro_ds
                WHERE {' AND '.join(conditions)}
            """
            cursor = source_conn.execute(query, params)
            dataset_ids = [row[0] for row in cursor.fetchall()]

            # Copy data for each dataset
            for ds_id in dataset_ids:
                # Copy metadata row
                cursor = source_conn.execute(
                    "SELECT * FROM x4pro_ds WHERE DatasetID = ?", (ds_id,)
                )
                row = cursor.fetchone()
                if row:
                    dest_conn.execute(
                        """INSERT OR REPLACE INTO x4pro_ds
                           (DatasetID, year1, author1, Targ1, Proj, MF, MT, ndat, quant1, reacode)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (row["DatasetID"], row["year1"], row["author1"], row["Targ1"],
                         row["Proj"], row["MF"], row["MT"], row["ndat"], row["quant1"],
                         row["reacode"])
                    )

                # Copy JSON data row
                cursor = source_conn.execute(
                    "SELECT * FROM x4pro_x5z WHERE DatasetID = ?", (ds_id,)
                )
                row = cursor.fetchone()
                if row:
                    dest_conn.execute(
                        "INSERT OR REPLACE INTO x4pro_x5z (DatasetID, jx5z) VALUES (?, ?)",
                        (row["DatasetID"], row["jx5z"])
                    )

                copied_count += 1

        dest_conn.commit()
        dest_conn.close()

        return copied_count

    def _normalize_target(self, target: Union[str, int, List[str], List[int]]) -> Union[str, List[str]]:
        """
        Normalize target input to database format (e.g., "Fe-56").

        Accepts single values or lists:
        - "Fe56", "Fe-56", 26056, "26-FE-56", "Fe" (natural element)
        - [26056, 26000] for multiple ZAIDs
        - ["Fe-56", "Fe-0"] for multiple targets

        Returns: "Fe-56" or "Fe-0" for natural elements (database format),
                 or a list of normalized targets if input was a list.
        """
        # Handle list inputs
        if isinstance(target, list):
            return [self._normalize_single_target(t) for t in target]

        return self._normalize_single_target(target)

    def _normalize_single_target(self, target: Union[str, int]) -> str:
        """
        Normalize a single target input to database format (e.g., "Fe-56").

        Accepts: "Fe56", "Fe-56", 26056, "26-FE-56", "Fe" (natural element)
        Returns: "Fe-56" or "Fe-0" for natural elements (database format)
        """
        if isinstance(target, int):
            # ZAID format
            return _zaid_to_target_pattern(target)

        target_str = str(target)

        # Already in database format "Fe-56" or "Fe-0"
        if re.match(r"^[A-Za-z]+-\d+$", target_str):
            return target_str.capitalize().replace(target_str[0], target_str[0].upper(), 1)

        # "Fe56" format -> "Fe-56"
        match = re.match(r"^([A-Za-z]+)(\d+)$", target_str)
        if match:
            elem = match.group(1).capitalize()
            mass = match.group(2)
            return f"{elem}-{mass}"

        # EXFOR format "26-FE-56"
        match = re.match(r"^(\d+)-([A-Za-z]+)-(\d+)$", target_str)
        if match:
            elem = match.group(2).capitalize()
            mass = match.group(3)
            return f"{elem}-{mass}"

        # Bare element symbol "Fe" -> "Fe-0" (natural element)
        # Validate it's a known element symbol
        if re.match(r"^[A-Za-z]{1,2}$", target_str):
            elem = target_str.capitalize()
            if elem in SYMBOL_TO_ATOMIC_NUMBER:
                return f"{elem}-0"

        return target_str


def read_exfor_from_database(
    db_path: str = None,
    target: str = None,
    target_zaid: int = None,
    projectile: str = "N",
    mt: int = None,
    energy_range: Tuple[float, float] = None,
) -> List["ExforAngularDistribution"]:
    """
    Convenience function to read EXFOR data from the X4Pro database.

    Parameters
    ----------
    db_path : str, optional
        Path to database. Uses KIKA_X4PRO_DB_PATH env var or default if None.
    target : str, optional
        Target in EXFOR notation (e.g., "26-FE-56")
    target_zaid : int, optional
        Target ZAID (e.g., 26056)
    projectile : str, optional
        Projectile (default: "N")
    mt : int, optional
        ENDF MT number
    energy_range : Tuple[float, float], optional
        Energy range (min, max) in MeV

    Returns
    -------
    List[ExforAngularDistribution]
        List of angular distribution datasets

    Examples
    --------
    >>> from kika.exfor.database import read_exfor_from_database
    >>> datasets = read_exfor_from_database(target_zaid=26056, mt=2)
    >>> for ds in datasets:
    ...     print(f"{ds.label}: {len(ds.energies())} energies")
    """
    with X4ProDatabase(db_path) as db:
        return db.query_angular_distributions(
            target=target,
            target_zaid=target_zaid,
            projectile=projectile,
            mt=mt,
            energy_range=energy_range,
        )
