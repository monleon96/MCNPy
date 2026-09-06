"""The IAEA catalogue: filename parsing and the nuclide-first index.

The snapshot is a scrape of the IAEA directory listings, and the
filenames in it come in every convention the server has used over the years.
The parser must read all of them the same way, because a nuclide the parser
misses is a nuclide the app says the library does not have.

Everything here runs on a small in-memory document; the built file is only
opened by the last test, which checks the snapshot itself is sound.
"""
from __future__ import annotations

import pytest

from kika.endf.remote.catalog import (
    PACKAGE_CATALOG,
    Catalog,
    load_catalog,
    nuclide_label,
    parse_catalog_filename,
)
from kika.endf.remote.catalog_build import parse_apache_size, parse_listing
from kika.endf.remote.constants import (
    library_display_name,
    library_family,
    normalize_library_name,
)


@pytest.mark.parametrize(
    "filename, sublib, expected",
    [
        # the two modern conventions
        ("n_026-Fe-56_2631.zip", "n", (26, 56, 0, 2631, "Fe-56")),
        ("n_2631_26-Fe-56.zip", "n", (26, 56, 0, 2631, "Fe-56")),
        # isomers, either side, either case
        ("n_095-Am-242M_9547.zip", "n", (95, 242, 1, 9547, "Am-242m")),
        ("n_9547_95-Am-242M.zip", "n", (95, 242, 1, 9547, "Am-242m")),
        # unpadded Z (PADF), upper-case symbol (ENDF/B-VIII.1 he4), .dat (ADS-2.0)
        ("p_12-Mg-22_0160.zip", "p", (12, 22, 0, 160, "Mg-22")),
        ("he4_002-HE-4_0228.zip", "he4", (2, 4, 0, 228, "He-4")),
        ("n_001-H-1_0125.dat", "n", (1, 1, 0, 125, "H-1")),
        # the neutron itself, as decay sub-libraries spell it
        ("decay_0001_0-nn-1.zip", "decay", (0, 1, 0, 1, "n-1")),
        ("decay_000-Nn-1_0001.zip", "decay", (0, 1, 0, 1, "n-1")),
        ("decay_000-n-1_0001.zip", "decay", (0, 1, 0, 1, "n-1")),
        # elemental files (photo-atomic) carry A=0 and label as the element
        ("photo_0100_1-H-0.zip", "photo", (1, 0, 0, 100, "H")),
    ],
)
def test_nuclide_filenames_parse(filename, sublib, expected):
    parsed = parse_catalog_filename(filename, sublib)
    assert parsed is not None, filename
    assert (parsed["z"], parsed["a"], parsed["isomer"], parsed["mat"], parsed["label"]) == expected


@pytest.mark.parametrize(
    "filename, label, mat",
    [
        ("tsl_H(H2O)_0001.dat", "H(H2O)", 1),
        ("tsl_10Graphite_0031.zip", "10Graphite", 31),
        ("tsl_7Li(7LiD)_3034.zip", "7Li(7LiD)", 3034),
    ],
)
def test_tsl_filenames_keep_the_material_name(filename, label, mat):
    parsed = parse_catalog_filename(filename, "tsl")
    assert parsed == {"z": None, "a": None, "isomer": 0, "mat": mat, "label": label}


@pytest.mark.parametrize(
    "filename, sublib",
    [
        ("endf-b-vi-8_n.zip", "NEUTRON"),  # whole-library tape
        ("Activ87_1120.txt", "mat"),  # MAT-numbered archive, no nuclide
        ("n_NDS148,1_0025.zip", "n"),  # stray listing in ENDF/B-VIII.1
        ("readme.txt", "n"),
    ],
)
def test_non_nuclide_files_are_dropped(filename, sublib):
    assert parse_catalog_filename(filename, sublib) is None


def test_nuclide_label():
    assert nuclide_label(26, 56) == "Fe-56"
    assert nuclide_label(95, 242, 1) == "Am-242m"
    assert nuclide_label(95, 242, 2) == "Am-242m2"
    assert nuclide_label(1, 0) == "H"
    assert nuclide_label(0, 1) == "n-1"


# -- a tiny catalogue ------------------------------------------------------

DOC = {
    "schema": 1,
    "generated_at": "2026-09-05T00:00:00+00:00",
    "base_url": "https://nds.iaea.org/public/download-endf",
    "libraries": [
        {
            "dir": "ENDF-B-VIII.1",
            "sublibs": {
                "n": [
                    ["n_026-Fe-56_2631.zip", 1000, "2024-10-03"],
                    ["n_092-U-235_9228.zip", 5000, "2024-10-03"],
                    ["n_NDS148,1_0025.zip", 5, "2024-10-03"],
                ],
                "tsl": [["tsl_H(H2O)_0001.zip", 300, "2024-10-03"]],
            },
        },
        {
            "dir": "JEFF-3.3",
            "sublibs": {"n": [["n_2631_26-Fe-56.zip", 1100, "2017-11-20"]]},
        },
        {
            "dir": "FENDL-3.2",
            "sublibs": {
                "n": [["n_026-Fe-56_2631.zip", 900, "2020-01-01"]],
                "p": [["p_026-Fe-56_2631.zip", 400, "2020-01-01"]],
            },
        },
    ],
}


@pytest.fixture
def catalog():
    return Catalog(DOC)


def test_library_ids_keep_the_historical_short_names(catalog):
    ids = {lib.id: lib for lib in catalog.libraries()}
    assert set(ids) == {"endfb8.1", "jeff3.3", "fendl-3.2"}
    assert ids["endfb8.1"].name == "ENDF/B-VIII.1"
    assert ids["endfb8.1"].family == "ENDF/B"
    assert ids["fendl-3.2"].directory == "FENDL-3.2"
    assert ids["fendl-3.2"].sublibs == {"n": 1, "p": 1}
    assert ids["endfb8.1"].sublibs == {"n": 2, "tsl": 1}  # the stray file is gone


def test_resolve_library_accepts_ids_aliases_and_directory_names(catalog):
    assert catalog.resolve_library("endfb8.1") == "endfb8.1"
    assert catalog.resolve_library("ENDF/B-VIII.1") == "endfb8.1"
    assert catalog.resolve_library("ENDF-B-VIII.1") == "endfb8.1"
    assert catalog.resolve_library("fendl-3.2") == "fendl-3.2"
    assert catalog.resolve_library("FENDL-3.2") == "fendl-3.2"
    assert catalog.resolve_library("nope") is None


def test_nuclide_first_lookup(catalog):
    entries = catalog.entries("Fe56")
    assert [e.library for e in entries] == ["endfb8.1", "fendl-3.2", "jeff3.3"]
    assert entries[0].url.endswith("/ENDF-B-VIII.1/n/n_026-Fe-56_2631.zip")
    assert entries[2].url.endswith("/JEFF-3.3/n/n_2631_26-Fe-56.zip")
    assert entries[0].cache_key == "26056"
    assert catalog.entries("Fe-56", sublib="p")[0].library == "fendl-3.2"
    assert catalog.entries(26056, sublib=None) and len(catalog.entries(26056, sublib=None)) == 4
    assert catalog.find("jeff3.3", "U235") is None
    assert catalog.find("endfb8.1", "U-235").mat == 9228


def test_nuclide_census_counts_libraries(catalog):
    nuclides = {n.label: n for n in catalog.nuclides("n")}
    assert nuclides["Fe-56"].libraries == 3
    assert nuclides["U-235"].libraries == 1
    assert [n.label for n in catalog.nuclides("p")] == ["Fe-56"]
    assert [n.label for n in catalog.nuclides("n", library="jeff3.3")] == ["Fe-56"]


def test_tsl_entries_are_reachable_by_library(catalog):
    (entry,) = catalog.entries_for_library("endfb8.1", "tsl")
    assert entry.label == "H(H2O)"
    assert entry.zaid is None
    assert entry.cache_key == "tsl_H(H2O)_0001"
    assert entry.nuclide_key is None


def test_search_matches_label_zaid_and_symbol(catalog):
    assert [n.label for n in catalog.search("fe-5")] == ["Fe-56"]
    assert [n.label for n in catalog.search("fe56")] == ["Fe-56"]
    assert [n.label for n in catalog.search("922")] == ["U-235"]
    assert [n.label for n in catalog.search("U")] == ["U-235"]
    assert catalog.search("") == []


def test_sublib_census_is_in_display_order(catalog):
    assert [s["id"] for s in catalog.sublibs()] == ["n", "p", "tsl"]
    assert catalog.sublibs()[0]["name"] == "Neutron"


# -- helpers in constants / builder ------------------------------------------

def test_display_names_and_families():
    assert library_display_name("ENDF-B-VII.1") == "ENDF/B-VII.1"
    assert library_display_name("JEFF-4.0") == "JEFF-4.0"
    assert library_family("ENDF-B-VIII.1") == "ENDF/B"
    assert library_family("JEF-2.2") == "JEFF"
    assert library_family("JENDL-PD-2016") == "JENDL"
    assert library_family("IRDFF-II") == "IRDFF"
    assert library_family("TENDL-2023") == "TENDL"


def test_apache_listing_parsing():
    html = (
        '<tr><td><a href="n_001-H-1_0125.zip">n_001-H-1_0125.zip</a></td>'
        '<td align="right">2024-10-03 11:24  </td><td align="right">226K</td></tr>'
        '<tr><td><a href="tsl/">tsl/</a></td><td align="right">2024-10-03 11:24  </td>'
        '<td align="right">  - </td></tr>'
    )
    dirs, files = parse_listing(html)
    assert dirs == ["tsl"]
    assert files == [("n_001-H-1_0125.zip", 226 * 1024, "2024-10-03")]
    assert parse_apache_size("1.5M") == int(1.5 * 1024 * 1024)
    assert parse_apache_size("807") == 807
    assert parse_apache_size("-") == 0


# -- the shipped snapshot ------------------------------------------------------

@pytest.mark.skipif(not PACKAGE_CATALOG.exists(), reason="snapshot not built")
def test_a_built_snapshot_is_sound():
    catalog = load_catalog(PACKAGE_CATALOG)
    ids = {lib.id for lib in catalog.libraries()}
    # every historically supported id resolves to a directory in the snapshot
    for lib_id in ("endfb8.1", "endfb8.0", "endfb7.1", "jeff4.0", "jeff3.3", "jendl5", "tendl2023", "cendl3.2"):
        assert lib_id in ids, lib_id
        assert normalize_library_name(lib_id) == lib_id
    assert len(catalog) > 100_000
    fe56 = catalog.entries("Fe56")
    assert {e.library for e in fe56} >= {"endfb8.1", "jeff3.3", "jeff4.0", "jendl5", "tendl2023"}
    assert normalize_library_name("FENDL-3.2") == "fendl-3.2"
