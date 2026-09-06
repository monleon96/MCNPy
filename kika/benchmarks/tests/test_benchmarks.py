"""Tests for the benchmarks (ICSBEP/DICE) subpackage: ingest, screening, reads."""

import gzip
import sqlite3

import numpy as np
import pytest

import kika.benchmarks as bm
from kika.benchmarks.ingest import _iter_sdf_files, _resolve_occurrences
from kika.sensitivities.sdf import SDFReactionData
from kika.cov.cross_section_covariance import CrossSectionCovariance
from kika.UQ.sandwich import sandwich_uncertainty_propagation


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
    # Fe-56 elastic total = 0.01; thresholds are strict.
    assert bm.find_sensitive_benchmarks(
        "Fe-56", "elastic", "total", sensitivity_threshold=0.1, db_path=db_path
    ) == []
    assert bm.find_sensitive_benchmarks(
        "Fe-56",
        "elastic",
        "total",
        sensitivity_threshold=0.01,
        db_path=db_path,
    ) == []


def test_mt_zero_is_only_screened_when_explicit(built_db):
    db_path, _ = built_db
    connection = sqlite3.connect(db_path)
    profile_id = connection.execute(
        "SELECT profile_id FROM profiles LIMIT 1"
    ).fetchone()[0]
    connection.execute(
        "INSERT INTO profile_sensitivities "
        "(profile_id, zaid, mt, unit, region, nuclide, reaction, "
        "s_thermal, s_epithermal, s_fast, s_total, s_abs_total) "
        "VALUES (?, 26056, 0, 0, 0, 'Fe-56', 'aggregate', 2, 2, 2, 2, 2)",
        (profile_id,),
    )
    connection.commit()
    connection.close()

    with bm.BenchmarksDatabase(db_path) as database:
        implicit = database.screen(26056)
        explicit = database.screen(26056, mt=0)

    assert implicit
    assert all(hit["mt"] != 0 for hit in implicit)
    assert len(explicit) == 1
    assert explicit[0]["mt"] == 0


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


def test_neutral_profile_and_uq_adapters(built_db, monkeypatch):
    db_path, _ = built_db
    benchmark = bm.get_benchmark("HEU-MET-FAST-001-001", db_path=db_path)
    profile_id = benchmark["profiles"][0]["profile_id"]
    profile = bm.get_sensitivity_profile(profile_id, db_path=db_path)
    assert profile.energy_unit == "MeV"
    assert profile.response == pytest.approx(1.001)
    assert profile.response_uncertainty == pytest.approx(0.0002)
    assert all(reaction.uncertainty is not None for reaction in profile.reactions)

    cov = CrossSectionCovariance(
        num_groups=profile.n_groups,
        energy_grid=profile.energy_grid.tolist(),
        energy_unit="MeV",
    )
    for reaction in profile.reactions:
        cov.add_matrix(
            reaction.zaid, reaction.mt, reaction.zaid, reaction.mt,
            np.eye(profile.n_groups), is_relative=True,
        )

    direct = sandwich_uncertainty_propagation(profile, cov_mat=cov, bootstrap_seed=7)
    adapted = bm.benchmark_uncertainty(
        profile_id, covariance=cov, db_path=db_path, bootstrap_seed=7
    )
    assert adapted.total_variance == pytest.approx(direct.total_variance)
    assert adapted.bootstrap_ci_low == pytest.approx(direct.bootstrap_ci_low)

    pair = bm.similarity_ck(profile, profile_id, cov, db_path=db_path)
    assert pair.value == pytest.approx(1.0)

    import kika.benchmarks.uq as benchmark_uq
    prepare_calls = 0
    original_prepare = benchmark_uq.prepare_covariance

    def counted_prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(benchmark_uq, "prepare_covariance", counted_prepare)
    ranked = bm.rank_benchmarks_by_ck(profile, cov, db_path=db_path)
    assert prepare_calls == 1
    assert [(row.benchmark_id, row.profile_id) for row in ranked] == [
        ("HEU-MET-FAST-001-001", profile_id)
    ]
    assert ranked[0].ck == pytest.approx(1.0)
    assert ranked[0].application_parameter_coverage == pytest.approx(1.0)
    assert ranked[0].benchmark_sensitivity_coverage == pytest.approx(1.0)


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


def test_iter_sdf_files_excludes_numbered_revisions(tmp_path):
    source = tmp_path / "sensitivity" / "HEU"
    source.mkdir(parents=True)
    canonical = source / "HEU-MET-FAST-001-001_KENO_LIB---4-Group_SENS.gz"
    revision = source / "HEU-MET-FAST-001-001_KENO_LIB---4-Group_SENS.1.gz"
    canonical.touch()
    revision.touch()
    found = list(_iter_sdf_files(tmp_path / "sensitivity", None))
    assert found == [canonical]


def test_occurrence_policies_are_explicit_and_absolute():
    first = SDFReactionData(
        zaid=26056,
        mt=2,
        sensitivity=[1.0, -2.0],
        error=[0.3, 0.4],
        unit=0,
        region=0,
    )
    last = SDFReactionData(
        zaid=26056,
        mt=2,
        sensitivity=[-0.5, 3.0],
        error=[0.4, 0.3],
        unit=0,
        region=0,
    )

    summed = _resolve_occurrences([first, last], "sum")
    assert len(summed) == 1
    assert summed[0].sensitivity == pytest.approx([0.5, 1.0])
    assert summed[0].error == pytest.approx([0.5, 0.5])
    assert _resolve_occurrences([first, last], "first")[0] is first
    assert _resolve_occurrences([first, last], "last")[0] is last
    with pytest.raises(bm.BenchmarksError, match="Repeated sensitivity"):
        _resolve_occurrences([first, last], "error")


def test_schema_v3_metadata_and_absolute_errors(built_db):
    db_path, _ = built_db
    with bm.BenchmarksDatabase(db_path) as database:
        stats = database.get_statistics()
        assert stats["meta"]["schema_version"] == "3"
        assert stats["meta"]["sensitivity_uncertainty_convention"] == "absolute"
        assert stats["meta"]["occurrences_rule"] == "sum"
        profile = database.get_benchmark("HEU-MET-FAST-001-001")["profiles"][0]
        assert "keff_unc" in profile
        assert "keff_rel_err" not in profile
        vector = database.get_profile_vector(profile["profile_id"])
        errors = [
            value
            for reaction in vector["reactions"]
            for value in reaction["error"]
        ]
        assert np.all(np.isfinite(errors))
        assert np.all(np.asarray(errors) >= 0.0)


def test_schema_v2_requires_rebuild(tmp_path):
    path = tmp_path / "old.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    connection.execute(
        "INSERT INTO meta (key, value) VALUES ('schema_version', '2')"
    )


def test_ingest_abnn_30_group_profile_without_invented_uncertainties(tmp_path):
    source = tmp_path / "sensitivity" / "HEU"
    source.mkdir(parents=True)
    grid = tmp_path / "newE"
    grid.mkdir()
    upper_boundaries = np.geomspace(2.0e7, 1.0e-2, 30)
    (grid / "ABBN30.txt").write_text(
        "\n".join(f"{value:.12E}" for value in upper_boundaries),
        encoding="utf-8",
    )
    rows = [
        f"{group:4d} "
        + " ".join(f"{group * scale:.8E}" for scale in range(1, 7))
        for group in range(1, 31)
    ]
    sum_rows = [
        f"{group:4d} {0.01 * group:.8E}"
        for group in range(1, 31)
    ]
    table = (
        "            ZONE         ALL      ISOTOP        PU39      CONCEN  1.0\n\n"
        "   G     FISSION     CAPTURE   INELASTIC     ELASTIC      NU-BAR      MU-BAR\n\n"
        + "\n".join(rows)
        + "\n"
        + "            ZONE         ALL      ISOTOP         SUM\n\n"
        + "   G     FSS.SPECTRA\n\n"
        + "\n".join(sum_rows)
        + "\n"
    )
    filename = "HEU-MET-FAST-002-001_KENO_ABBN-93---299-Group_SENS.gz"
    with gzip.open(source / filename, "wt", encoding="utf-8") as stream:
        stream.write(table)

    db_path = tmp_path / "abbn.db"
    stats = bm.build_benchmarks_db(
        source_dir=str(tmp_path / "sensitivity"),
        db_path=str(db_path),
        n_workers=1,
    )
    assert stats["files_skipped"] == 0
    assert stats["benchmarks"] == 1
    assert stats["profiles"] == 1
    assert stats["reactions"] == 6

    benchmark = bm.get_benchmark("HEU-MET-FAST-002-001", db_path=str(db_path))
    profile = benchmark["profiles"][0]
    assert profile["group_structure"] == "30-Group"
    assert profile["ngroups"] == 30
    assert profile["keff"] is None
    assert profile["keff_unc"] is None
    vector = bm.get_profile_vector(profile["profile_id"], db_path=str(db_path))
    assert vector["pert_energies"][0] == pytest.approx(1.0e-11)
    assert vector["pert_energies"][-1] == pytest.approx(20.0)
    assert all(reaction["error"] is None for reaction in vector["reactions"])
    neutral = bm.get_sensitivity_profile(profile["profile_id"], db_path=str(db_path))
    assert all(reaction.uncertainty is None for reaction in neutral.reactions)
    assert {reaction["zaid"] for reaction in vector["reactions"]} == {94239}
    fission = next(reaction for reaction in vector["reactions"] if reaction["mt"] == 18)
    assert fission["sensitivity"] == pytest.approx(
        [float(group) for group in range(30, 0, -1)]
    )


def test_ck_ranking_tie_breaks_by_benchmark_id(tmp_path):
    src = tmp_path / "sensitivity" / "HEU"
    src.mkdir(parents=True)
    for case in ("002", "001"):
        name = f"HEU-MET-FAST-001-{case}_KENO_ENDF-B-VII.0---4-Group_SENS.gz"
        with gzip.open(src / name, "wt", encoding="utf-8") as stream:
            stream.write(_fixture_sdf())
    db_path = str(tmp_path / "ranking.db")
    bm.build_benchmarks_db(source_dir=str(tmp_path / "sensitivity"), db_path=db_path)
    bm.reset_config()

    first = bm.get_benchmark("HEU-MET-FAST-001-001", db_path=db_path)["profiles"][0]
    application = bm.get_sensitivity_profile(first["profile_id"], db_path=db_path)
    cov = CrossSectionCovariance(
        num_groups=application.n_groups,
        energy_grid=application.energy_grid.tolist(),
        energy_unit="MeV",
    )
    for reaction in application.reactions:
        cov.add_matrix(
            reaction.zaid, reaction.mt, reaction.zaid, reaction.mt,
            np.eye(application.n_groups), is_relative=True,
        )

    ranked = bm.rank_benchmarks_by_ck(application, cov, db_path=db_path)
    assert [row.benchmark_id for row in ranked] == [
        "HEU-MET-FAST-001-001",
        "HEU-MET-FAST-001-002",
    ]
    assert [row.ck for row in ranked] == pytest.approx([1.0, 1.0])
