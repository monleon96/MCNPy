"""The general loader's canonical units, the reaction-code fields, and the
in-memory catalogue search of ``X4ProDatabase``.

The X4Pro shapes below are trimmed from real subentries: 23662002 (Gkatis,
Fe-0 transmission filed as ``quant1='CS'``), 23365006 (Pirovano, Fe-0 elastic
σ with c5 energies in eV and x4 energies in MeV) and 10037024-style angular
data. The values are the ones the database holds.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from kika.exfor.database import (
    X4ProDatabase,
    canonical_unit_factor,
    exfor_quantity,
    parse_reacode_fields,
)


# ---------------------------------------------------------------------------
# Reaction-code fields
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "reacode, expected",
    [
        ("26-FE-56(N,EL)26-FE-56,,DA", ("26-FE-56", "N", "EL", "26-FE-56", "", "DA")),
        ("26-FE-0(N,TOT),,TRN", ("26-FE-0", "N", "TOT", "", "", "TRN")),
        ("26-FE-54(N,INL)26-FE-54,PAR,DA", ("26-FE-54", "N", "INL", "26-FE-54", "PAR", "DA")),
        ("92-U-235(N,F),,SIG", ("92-U-235", "N", "F", "", "", "SIG")),
        # A ratio: the fields of the first reaction.
        ("(26-FE-56(N,EL)26-FE-56,,SIG)/(6-C-0(N,EL)6-C-0,,SIG)", ("26-FE-56", "N", "EL", "26-FE-56", "", "SIG")),
    ],
)
def test_parse_reacode_fields(reacode, expected):
    f = parse_reacode_fields(reacode)
    assert (f["target"], f["projectile"], f["process"], f["product"], f["sf5"], f["sf6"]) == expected


def test_parse_reacode_fields_tolerates_garbage():
    assert parse_reacode_fields(None)["sf6"] == ""
    assert parse_reacode_fields("")["sf6"] == ""
    assert parse_reacode_fields("not a code")["sf6"] == ""


def test_exfor_quantity_prefers_sf6_over_x4pro_quant():
    # X4Pro files this transmission under CS; EXFOR itself says TRN.
    assert exfor_quantity("26-FE-0(N,TOT),,TRN", fallback="CS") == "TRN"
    assert exfor_quantity("garbage", fallback="CS") == "CS"
    assert exfor_quantity(None, fallback=None) == ""


# ---------------------------------------------------------------------------
# Canonical units
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "unit, factor, label",
    [
        ("EV", 1e-6, "MeV"),
        ("KEV", 1e-3, "MeV"),
        ("MEV", 1.0, "MeV"),
        ("MILLI-EV", 1e-9, "MeV"),
        ("B", 1.0, "b"),
        ("MB", 1e-3, "b"),
        ("MICRO-B", 1e-6, "b"),
        ("MB/SR", 1e-3, "b/sr"),
        ("B/SR", 1.0, "b/sr"),
        ("B/KEV", 1e3, "b/MeV"),
        ("MB/SR/MEV", 1e-3, "b/sr/MeV"),
    ],
)
def test_canonical_unit_factor(unit, factor, label):
    got_factor, got_label = canonical_unit_factor(unit)
    assert got_label == label
    assert got_factor == pytest.approx(factor)


@pytest.mark.parametrize("unit", ["NO-DIM", "PART/FIS", "PC/FIS", "ADEG", "ATOMS/B", "", None, "B/FOO"])
def test_canonical_unit_factor_leaves_other_families_alone(unit):
    assert canonical_unit_factor(unit) == (None, None)


# ---------------------------------------------------------------------------
# General loader
# ---------------------------------------------------------------------------

def _c5(y, dy, y_units, x1, x1_units, dx1=None, fam="EN"):
    block = {
        "y": {"cvar": "y", "fam": "Data", "units": y_units, "y": y, "dy": dy},
        "x1": {"cvar": "x1", "fam": fam, "units": x1_units, "x1": x1},
    }
    if dx1 is not None:
        block["x1"]["dx1"] = dx1
    return block


def test_c5_energies_in_ev_become_mev_and_dx1_is_the_x_error():
    db = X4ProDatabase.__new__(X4ProDatabase)
    jx5z = {
        "c5data": _c5([2.8, 2.76], [0.35, 0.34], "B", [1994800.0, 2009100.0], "EV", dx1=[7100, 7200]),
        "x4data": [],
    }
    df, units, ind_vars, dep_var = db._parse_general_data(jx5z)

    assert ind_vars == ["energy"] and dep_var == "value"
    assert units["energy"] == "MeV" and units["value"] == "b" and units["error"] == "b"
    assert df["energy"].tolist() == pytest.approx([1.9948, 2.0091])
    assert df["value"].tolist() == [2.8, 2.76]
    assert df["error"].tolist() == [0.35, 0.34]
    # The x resolution is its own column, in MeV, never confused with dy.
    assert units["energy_error"] == "MeV"
    assert df["energy_error"].tolist() == pytest.approx([0.0071, 0.0072])


def test_millibarn_values_and_errors_are_scaled_together():
    db = X4ProDatabase.__new__(X4ProDatabase)
    jx5z = {"c5data": _c5([120.0, 80.0], [6.0, 4.0], "MB", [1.0e6, 2.0e6], "EV"), "x4data": []}
    df, units, _, _ = db._parse_general_data(jx5z)
    assert units["value"] == "b"
    assert df["value"].tolist() == pytest.approx([0.12, 0.08])
    assert df["error"].tolist() == pytest.approx([0.006, 0.004])


def test_transmission_stays_dimensionless_and_unconverted():
    db = X4ProDatabase.__new__(X4ProDatabase)
    jx5z = {
        "c5data": {
            **_c5([0.617498, 0.664761], [0.026587, 0.030067], "NO-DIM", [100.001, 99.9822], "EV"),
            "x2": {"cvar": "x2", "fam": "THS", "units": "ATOMS/B", "x2": [0.1014], "rpt": 2},
        },
        "x4data": [],
    }
    df, units, ind_vars, _ = db._parse_general_data(jx5z)
    assert units["value"] == "NO-DIM"
    assert df["value"].tolist() == [0.617498, 0.664761]
    assert units["energy"] == "MeV"
    assert df["energy"].tolist() == pytest.approx([100.001e-6, 99.9822e-6])
    assert units["ths"] == "ATOMS/B" and df["ths"].tolist() == [0.1014, 0.1014]
    assert ind_vars == ["energy", "ths"]


def test_x4_fallback_converts_mev_and_percent_errors():
    """Without c5data the x4 block is used: MeV energies and a PER-CENT error."""
    db = X4ProDatabase.__new__(X4ProDatabase)
    jx5z = {
        "c5data": {},
        "x4data": [
            {"cvar": "y", "fam": "Data", "units": "MB", "dat0": [200.0, 100.0]},
            # A constant systematic error has no dat0 and must not win.
            {"cvar": "dy", "fam": "Data", "units": "PER-CENT", "ifComm": True, "com0": 3.0},
            {"cvar": "dy", "fam": "Data", "units": "PER-CENT", "dat0": [10.0, 5.0]},
            {"cvar": "x1", "fam": "EN", "units": "MEV", "dat0": [1.5, 2.5]},
        ],
    }
    df, units, _, _ = db._parse_general_data(jx5z)
    assert units["energy"] == "MeV" and df["energy"].tolist() == [1.5, 2.5]
    assert units["value"] == "b" and df["value"].tolist() == pytest.approx([0.2, 0.1])
    # 10 % of 200 mb = 20 mb = 0.02 b; 5 % of 100 mb = 0.005 b.
    assert units["error"] == "b" and df["error"].tolist() == pytest.approx([0.02, 0.005])


# ---------------------------------------------------------------------------
# Catalogue search against a tiny X4Pro-shaped database
# ---------------------------------------------------------------------------

def _x5z(energies_ev):
    return json.dumps({
        "c5data": {
            "y": {"cvar": "y", "fam": "Data", "units": "B", "y": [1.0] * len(energies_ev), "dy": [0.1] * len(energies_ev)},
            "x1": {"cvar": "x1", "fam": "EN", "units": "EV", "x1": energies_ev},
        },
        "x4data": [],
    })


_ROWS = [
    # DatasetID, year1, author1, Targ1, Proj, MF, MT, ndat, quant1, reacode, energies (eV)
    ("23365006", 2019, "Pirovano", "Fe-0", "n", 3, 2, 2, "CS", "26-FE-0(N,EL)26-FE-0,,SIG", [1.9948e6, 2.0163e6]),
    ("23662002", 2024, "Gkatis", "Fe-0", "n", 0, 0, 3, "CS", "26-FE-0(N,TOT),,TRN", [100.0, 1.0e6, 2.0e7]),
    ("23661003", 2024, "Gkatis", "Fe-54", "n", 3, 2, 2, "CS", "26-FE-54(N,EL)26-FE-54,,SIG", [1.0e6, 5.0e6]),
    ("23661002", 2024, "Gkatis", "Fe-54", "n", 4, 2, 2, "DA", "26-FE-54(N,EL)26-FE-54,,DA", [1.0e6, 5.0e6]),
    ("10001002", 1971, "Boschung", "Fe-56", "n", 4, 2, 2, "DA", "26-FE-56(N,EL)26-FE-56,,DA", [1.0e4, 1.0e4]),
    ("C0001002", 1990, "Smith", "Fe-56", "p", 3, 2, 2, "CS", "26-FE-56(P,EL)26-FE-56,,SIG", [1.0e7, 2.0e7]),
    ("30001002", 1985, "Nobody", "Fe-56", "n", 3, 102, 2, "CSP", "26-FE-56(N,G)26-FE-57,PAR,SIG", None),
]


@pytest.fixture
def tiny_db(tmp_path):
    path = tmp_path / "x4.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE x4pro_ds (DatasetID TEXT, year1 INT, author1 TEXT, Targ1 TEXT, Proj TEXT, "
        "MF INT, MT INT, ndat INT, quant1 TEXT, reacode TEXT)"
    )
    conn.execute("CREATE TABLE x4pro_x5z (DatasetID TEXT, Subent TEXT, updated TEXT, jx5z TEXT)")
    for ds, year, author, targ, proj, mf, mt, ndat, quant, reacode, energies in _ROWS:
        conn.execute("INSERT INTO x4pro_ds VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (ds, year, author, targ, proj, mf, mt, ndat, quant, reacode))
        if energies is not None:
            conn.execute("INSERT INTO x4pro_x5z VALUES (?,?,?,?)", (ds, ds[:5], "", _x5z(energies)))
    conn.commit()
    conn.close()
    db = X4ProDatabase(str(path))
    yield db
    db.close()


def _ids(results):
    return [r["dataset_id"] for r in results]


def test_search_matches_like_query_dataset_ids(tiny_db):
    # Substring target, substring quantity (CS also finds CSP), neutron in either case.
    hits = tiny_db.search_datasets(targets=["Fe-56"], projectile="n", quantities=["CS"])
    assert _ids(hits) == ["30001002"]
    hits = tiny_db.search_datasets(targets=["Fe-56"], projectile="N")
    assert _ids(hits) == ["10001002", "30001002"]
    # Several targets and MTs are ORed within the filter.
    hits = tiny_db.search_datasets(targets=["Fe-54", "Fe-0"], mts=[2])
    assert _ids(hits) == ["23365006", "23661002", "23661003"]


def test_search_reports_sf6_quantity_energy_range_and_mf(tiny_db):
    hits = {r["dataset_id"]: r for r in tiny_db.search_datasets(targets=["Fe-0"])}
    trn = hits["23662002"]
    assert trn["quant"] == "CS" and trn["sf6"] == "TRN" and trn["mf"] == 0
    assert trn["e_min"] == pytest.approx(1e-4) and trn["e_max"] == pytest.approx(20.0)
    sig = hits["23365006"]
    assert sig["sf6"] == "SIG" and sig["sf5"] == "" and sig["mf"] == 3
    assert sig["e_min"] == pytest.approx(1.9948) and sig["e_max"] == pytest.approx(2.0163)


def test_search_energy_window_overlaps_and_drops_unknown_ranges(tiny_db):
    hits = tiny_db.search_datasets(targets=["Fe"], energy_min_mev=3.0)
    # Fe-56 (n,g) has no JSON at all and is dropped by an energy filter …
    assert "30001002" not in _ids(hits)
    # … Pirovano tops out at 2.02 MeV, Gkatis Fe-54 reaches 5 MeV, the transmission 20 MeV.
    assert _ids(hits) == ["23661002", "23661003", "23662002", "C0001002"] or _ids(hits) == ["23661002", "23661003", "23662002"]
    hits = tiny_db.search_datasets(targets=["Fe"], projectile=None, energy_min_mev=3.0)
    assert "C0001002" in _ids(hits)
    # Without an energy filter, an unknown range is kept and reported as None.
    hits = {r["dataset_id"]: r for r in tiny_db.search_datasets(targets=["Fe-56"], projectile="n")}
    assert hits["30001002"]["e_min"] is None


def test_search_author_year_and_entry_prefix(tiny_db):
    assert _ids(tiny_db.search_datasets(author="gkat")) == ["23661002", "23661003", "23662002"]
    assert _ids(tiny_db.search_datasets(year_min=2020)) == ["23661002", "23661003", "23662002"]
    assert _ids(tiny_db.search_datasets(year_max=1975)) == ["10001002"]
    assert _ids(tiny_db.search_datasets(entry_prefix="2366")) == ["23661002", "23661003", "23662002"]
    assert tiny_db.search_datasets(targets=["Xe-999"]) == []


def test_energy_ranges_are_cached_after_the_first_search(tiny_db):
    tiny_db.search_datasets(targets=["Fe-0"])
    assert tiny_db._energy_range_cache["23365006"] == (pytest.approx(1.9948), pytest.approx(2.0163))
    # A dataset without a JSON row is remembered as unknown, not re-queried.
    tiny_db.search_datasets(targets=["Fe-56"], projectile="n")
    assert tiny_db._energy_range_cache["30001002"] == (None, None)
    # get_dataset_metadata reads the same cache.
    meta = tiny_db.get_dataset_metadata("23365006")
    assert meta["e_max"] == pytest.approx(2.0163)


def test_statistics_and_targets_come_from_the_catalogue_once_loaded(tiny_db):
    tiny_db.load_catalogue()
    stats = tiny_db.get_statistics()
    assert stats["total_datasets"] == len(_ROWS)
    assert stats["angular_distributions"] == 2
    assert stats["cross_sections"] == 5
    assert stats["unique_targets"] == 3
    assert tiny_db.list_targets("n") == ["Fe-54", "Fe-56"]
