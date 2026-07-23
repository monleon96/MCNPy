"""Tests for the benchmarks (ICSBEP/DICE) subpackage: ingest, screening, reads."""

import gzip

import pytest

import kika.benchmarks as bm


def _block(nuclide, reaction, zaid, mt, meta1, sens_line):
    scal = "  1.0E-02  1.0E-03  1.0E-02  0.0  0.0"
    err = "  1.000000E-04  1.000000E-04  1.000000E-04  1.000000E-04"
    return "\n".join(
        [f"{nuclide:<13}{reaction:<17}{zaid:>5}{mt:>7}", meta1,
         "  0.000000E+00  0.000000E+00      0      0", scal, sens_line, err]
    )


def _fixture_sdf():
    header = "\n".join(
        [
            "/scale/test.inp",
            "         4 number of neutron groups",
            "         3   number of sensitivity profiles          2 are region integrated",
            "  1.001000 +/-   0.000200  k-eff from the forward case",
            "energy boundaries:",
            "  2.000000E+07  1.000000E+05  6.250000E-01  1.000000E-05  1.000000E-11",
        ]
    )
    blocks = [
        # region-integrated fe-56 elastic (file order = descending energy)
        _block("fe-56", "elastic", 26056, 2, "      0      0   ",
               "  4.000000E-03  3.000000E-03  2.000000E-03  1.000000E-03"),
        # per-region u-235 fission (unit -1) -> excluded from the DB
        _block("u-235", "fission", 92235, 18, "     -1      0",
               "  9.000000E-01  9.000000E-01  9.000000E-01  9.000000E-01"),
        # region-integrated u-235 fission: descending [0.25,0.15,0.05,0.05]
        _block("u-235", "fission", 92235, 18, "      0      0",
               "  2.500000E-01  1.500000E-01  5.000000E-02  5.000000E-02"),
    ]
    return header + "\n" + "\n".join(blocks) + "\nfile verification information\n"


@pytest.fixture
def built_db(tmp_path):
    """Build a benchmarks DB from a single fixture benchmark and return its path."""
    src = tmp_path / "sensitivity" / "HEU"
    src.mkdir(parents=True)
    name = "HEU-MET-FAST-001-001_KENO_ENDF-B-VII.0---4-Group_SENS.gz"
    with gzip.open(src / name, "wt", encoding="utf-8") as f:
        f.write(_fixture_sdf())
    db_path = str(tmp_path / "kika_benchmarks.db")
    stats = bm.build_benchmarks_db(
        source_dir=str(tmp_path / "sensitivity"), db_path=db_path
    )
    bm.reset_config()
    return db_path, stats


def test_ingest_counts_and_region_filter(built_db):
    db_path, stats = built_db
    assert stats["benchmarks"] == 1
    assert stats["profiles"] == 1
    # only the two region-integrated profiles are stored (per-region excluded)
    assert stats["reactions"] == 2
    assert stats["files_skipped"] == 0


def test_screening_and_integrals(built_db):
    db_path, _ = built_db
    hits = bm.find_sensitive_benchmarks("U-235", reaction="fission", db_path=db_path)
    assert len(hits) == 1
    h = hits[0]
    assert h["benchmark_id"] == "HEU-MET-FAST-001-001"
    assert h["category"] == "HEU" and h["spectrum"] == "FAST"
    # ascending groups [0.05, 0.05, 0.15, 0.25]; total = 0.5
    assert h["s_total"] == pytest.approx(0.5, abs=1e-6)

    # Energy-region split: thermal = 0.10, epithermal = 0.15, fast = 0.25.
    fast = bm.find_sensitive_benchmarks("U-235", "fission", "fast", db_path=db_path)[0]
    assert fast["s_region"] == pytest.approx(0.25, abs=1e-6)
    therm = bm.find_sensitive_benchmarks("U-235", "fission", "thermal", db_path=db_path)[0]
    assert therm["s_region"] == pytest.approx(0.10, abs=1e-6)


def test_isotope_and_reaction_aliases(built_db):
    db_path, _ = built_db
    # zaid, symbol, hyphenated symbol all resolve; 'elastic' alias -> MT 2
    for iso in (26056, "Fe56", "Fe-56"):
        hits = bm.find_sensitive_benchmarks(iso, reaction="elastic", db_path=db_path)
        assert len(hits) == 1 and hits[0]["mt"] == 2
    assert bm.find_sensitive_benchmarks("Fe-56", reaction=2, db_path=db_path)


def test_threshold_filters(built_db):
    db_path, _ = built_db
    # Fe-56 elastic total = 0.01; a threshold above it returns nothing.
    assert bm.find_sensitive_benchmarks(
        "Fe-56", "elastic", "total", sensitivity_threshold=0.1, db_path=db_path
    ) == []


def test_get_benchmark_and_vector(built_db):
    db_path, _ = built_db
    detail = bm.get_benchmark("HEU-MET-FAST-001-001", db_path=db_path)
    assert len(detail["profiles"]) == 1
    prof = detail["profiles"][0]
    assert prof["is_preferred"] == 1

    vec = bm.get_profile_vector(prof["profile_id"], zaid=92235, mt=18, db_path=db_path)
    assert len(vec["pert_energies"]) == 5
    assert len(vec["reactions"]) == 1
    r = vec["reactions"][0]
    assert r["nuclide"] == "U-235"
    assert len(r["sensitivity"]) == 4
    # store_errors now defaults True -> per-group error vector is present.
    assert r["error"] is not None and len(r["error"]) == 4
    assert r["error"][0] == pytest.approx(1e-4, abs=1e-9)
    assert sum(r["sensitivity"]) == pytest.approx(0.5, abs=1e-6)


def test_store_errors_false(tmp_path):
    src = tmp_path / "sensitivity" / "HEU"
    src.mkdir(parents=True)
    name = "HEU-MET-FAST-001-001_KENO_ENDF-B-VII.0---4-Group_SENS.gz"
    with gzip.open(src / name, "wt", encoding="utf-8") as f:
        f.write(_fixture_sdf())
    db_path = str(tmp_path / "db.db")
    bm.build_benchmarks_db(
        source_dir=str(tmp_path / "sensitivity"), db_path=db_path, store_errors=False
    )
    bm.reset_config()
    prof = bm.get_benchmark("HEU-MET-FAST-001-001", db_path=db_path)["profiles"][0]
    vec = bm.get_profile_vector(prof["profile_id"], zaid=92235, mt=18, db_path=db_path)
    assert vec["reactions"][0]["error"] is None


def test_region_profiles_flag(tmp_path):
    src = tmp_path / "sensitivity" / "HEU"
    src.mkdir(parents=True)
    name = "HEU-MET-FAST-001-001_KENO_ENDF-B-VII.0---4-Group_SENS.gz"
    with gzip.open(src / name, "wt", encoding="utf-8") as f:
        f.write(_fixture_sdf())

    # Default (off): the per-region u-235 fission profile (unit -1) is excluded.
    db_off = str(tmp_path / "off.db")
    s_off = bm.build_benchmarks_db(source_dir=str(tmp_path / "sensitivity"), db_path=db_off)
    assert s_off["reactions"] == 2

    # On: the per-region profile is also stored, but screening still returns only
    # the system-total, and the plot vector still excludes non-(0,0).
    db_on = str(tmp_path / "on.db")
    s_on = bm.build_benchmarks_db(
        source_dir=str(tmp_path / "sensitivity"),
        db_path=db_on,
        store_region_profiles=True,
    )
    bm.reset_config()
    assert s_on["reactions"] == 3  # +1 per-region profile
    hits = bm.find_sensitive_benchmarks("U-235", "fission", db_path=db_on)
    assert len(hits) == 1  # screening unaffected
    prof = bm.get_benchmark("HEU-MET-FAST-001-001", db_path=db_on)["profiles"][0]
    vec = bm.get_profile_vector(prof["profile_id"], db_path=db_on)
    assert len(vec["reactions"]) == 2  # only system totals in the plot vector


def test_plot_profile_returns_figure(built_db):
    import matplotlib
    matplotlib.use("Agg")

    db_path, _ = built_db
    fig = bm.plot_profile("HEU-MET-FAST-001-001", db_path=db_path)
    assert fig is not None and len(fig.axes) >= 1


def test_keff_abbrev_crosswalk():
    from kika.benchmarks._naming import benchmark_id_to_keff_abbrevs

    assert benchmark_id_to_keff_abbrevs("HEU-MET-MIXED-005-001") == ["hmm5-1"]
    assert benchmark_id_to_keff_abbrevs("LEU-COMP-THERM-038-007") == ["lct38-7"]
    # MIX yields both single- and double-m candidates.
    assert benchmark_id_to_keff_abbrevs("MIX-COMP-THERM-001-002") == ["mct1-2", "mmct1-2"]
    assert benchmark_id_to_keff_abbrevs("SUB-WEIRD") == []


def test_aux_parsers():
    from kika.benchmarks._aux import parse_balance_summary, parse_spectrum

    spec = (
        " GRP     E-up        FLUX\n"
        "   1   2.000E+07   6.743E-04  1.0  1.0  1.0  1.0\n"
        "   2   1.862E+07   2.258E-02  1.0  1.0  1.0  1.0\n"
        " TOTAL             1.036E+02\n"
    )
    energies, flux = parse_spectrum(spec)
    assert energies == [2.000e7, 1.862e7]
    assert flux == [pytest.approx(6.743e-4), pytest.approx(2.258e-2)]

    bal = "NUMBER OF ZONES IN THE CORE:   8\n K-EFF = 0.9904736\n LEAKAGE = 0.184\n"
    summary = parse_balance_summary(bal)
    assert summary["n_zones"] == 8
    assert summary["keff"] == pytest.approx(0.9904736)
    assert summary["leakage"] == pytest.approx(0.184)


def test_missing_database_raises():
    with pytest.raises(bm.DatabaseNotConfiguredError):
        bm.find_sensitive_benchmarks("Fe-56", db_path="/no/such/benchmarks.db")
