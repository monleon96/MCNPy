"""The IAEA download URL, for both of the filename conventions in use.

IAEA serves ``download-endf`` under two naming schemes and the split does not
follow the library family, only when the release was put online: JEFF-3.3 and
JENDL-4.0 use ``n_0125_1-H-1.zip`` while JEFF-4.0 and JENDL-5 use
``n_001-H-1_0125.zip``. :func:`build_iaea_url` used to emit the second one for
every library, so half the catalogue answered 404 and the app reported the
isotope as missing from the library.

The URLs asserted here were checked against the live server once; the test
itself makes no network call.
"""
from __future__ import annotations

import pytest

from kika.endf.remote.iaea_client import (
    build_iaea_url,
    get_endf_mat,
    isotope_key,
    parse_isotope_state,
)

BASE = "https://nds.iaea.org/public/download-endf"


@pytest.mark.parametrize(
    "isotope, library, expected",
    [
        # "za_mat" style
        ("H-1", "endfb8.1", "ENDF-B-VIII.1/n/n_001-H-1_0125.zip"),
        ("U-235", "jeff4.0", "JEFF-4.0/n/n_092-U-235_9228.zip"),
        ("U-238", "jendl5", "JENDL-5/n/n_092-U-238_9237.zip"),
        ("Ne-20", "tendl2023", "TENDL-2023/n/n_010-Ne-20_1025.zip"),
        ("O-16", "cendl3.2", "CENDL-3.2/n/n_008-O-16_0825.zip"),
        # "mat_za" style
        ("H-1", "jeff3.3", "JEFF-3.3/n/n_0125_1-H-1.zip"),
        ("H-1", "jendl4.0", "JENDL-4.0/n/n_0125_1-H-1.zip"),
        ("Pu-239", "jeff3.2", "JEFF-3.2/n/n_9437_94-Pu-239.zip"),
        ("Na-23", "jeff3.1.1", "JEFF-3.1.1/n/n_1125_11-Na-23.zip"),
        ("Fe-56", "endfb8.0", "ENDF-B-VIII.0/n/n_2631_26-Fe-56.zip"),
        ("U-235", "endfb7.1", "ENDF-B-VII.1/n/n_9228_92-U-235.zip"),
        # isomers, in both styles
        ("Am-242m", "endfb8.1", "ENDF-B-VIII.1/n/n_095-Am-242M_9547.zip"),
        ("Am-242m", "jeff3.3", "JEFF-3.3/n/n_9547_95-Am-242M.zip"),
    ],
)
def test_url_matches_the_library_convention(isotope, library, expected):
    z, a, symbol, isomer = parse_isotope_state(isotope)
    assert build_iaea_url(z, a, symbol, library, "n", isomer) == f"{BASE}/{expected}"


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("Fe56", (26, 56, "Fe", 0)),
        ("Fe-56", (26, 56, "Fe", 0)),
        ("fe56", (26, 56, "Fe", 0)),
        (26056, (26, 56, "Fe", 0)),
        ("26056", (26, 56, "Fe", 0)),
        ("Am242m", (95, 242, "Am", 1)),
        ("Am-242m", (95, 242, "Am", 1)),
        ("Ag110M", (47, 110, "Ag", 1)),
        (95642, (95, 242, "Am", 1)),
    ],
)
def test_parse_isotope_state(spec, expected):
    assert parse_isotope_state(spec) == expected


def test_isomers_do_not_share_the_ground_state_cache_key():
    """An isomer and its ground state differ only in MAT, never in ZAID."""
    assert get_endf_mat(95, 242, 0) == 9546
    assert get_endf_mat(95, 242, 1) == 9547
    assert isotope_key("Am-242") == 95242
    assert isotope_key("Am-242m") == 95642


def test_a_nuclide_without_an_isomer_says_so():
    with pytest.raises(ValueError, match="isomeric state"):
        get_endf_mat(26, 56, 1)
