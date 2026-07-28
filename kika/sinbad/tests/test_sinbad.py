"""Tests for the SINBAD subpackage: package forms, library, objects, tables.

The fixture is a minimal but structurally complete package written to a tmp
directory in both forms, so the tests exercise the thing the design rests on:
a directory and a ``.sinbad`` archive must be indistinguishable to a reader.

It carries two measurement systems and a three-component uncertainty model --
counting statistics (uncorrelated), detector calibration (correlated within one
system) and source strength (correlated across the entry) -- because the block
structure that produces is the part most easily got wrong.
"""

import json
import math
import zipfile

import numpy as np
import pytest

import kika.sinbad as sinbad

BENCHMARK_ID = "TEST-SHIELD-01"

STAT = 0.01
CAL = {"SYS-AL": 0.03, "SYS-AU": 0.02}
SRC = 0.04

SYSTEMS = {
    "SYS-AL": ("Al27", [("A1", 5.0, 2.0e-8), ("A2", 15.0, 4.0e-9)]),
    "SYS-AU": ("Au197", [("A1", 5.0, 1.0e-6), ("A2", 15.0, 3.0e-7),
                         ("A3", 25.0, 9.0e-8)]),
}


def _total(system):
    return math.sqrt(STAT**2 + CAL[system] ** 2 + SRC**2)


def _components(system):
    return [
        {"id": "U0", "type": "countingStatistics", "relative": STAT,
         "coverageFactor": 1, "derivation": "reconstructed",
         "correlationScope": {"kind": "none"}},
        {"id": "U1", "type": "detectorCalibration", "relative": CAL[system],
         "coverageFactor": 1, "derivation": "reported",
         "correlationScope": {"kind": "fullWithinMeasurementSystem",
                              "ref": system}},
        {"id": "U2", "type": "sourceStrength", "relative": SRC,
         "coverageFactor": 1, "derivation": "reported",
         "correlationScope": {"kind": "fullWithinEntry", "ref": "NORM-X"}},
    ]


def _model():
    """Two systems, two libraries, one absent input, one derived deck."""
    measurements = []
    for sys_id, (_nuclide, points) in SYSTEMS.items():
        tag = sys_id.split("-")[1]
        for pos, depth, value in points:
            measurements.append({
                "id": f"EXP-{tag}-{pos}",
                "systemRef": sys_id,
                "observableRef": f"OBS-{tag}",
                "unit": "1/s",
                "value": value,
                "reportedValue": value * 1e3,
                "position": {"label": pos, "axialDepth_cm": depth},
                "uncertainty": {
                    "componentsResolved": True,
                    "reportedTotal": _total(sys_id),
                    "components": _components(sys_id),
                    "missing": [],
                },
            })

    calculations, comparisons = [], []
    for tag, lib, scale in (("L1", "LIB-A", 1.10), ("L2", "LIB-B", 0.95)):
        for sys_id in SYSTEMS:
            short = sys_id.split("-")[1]
            calc_id = f"CALC-{short}-{tag}"
            mine = [m for m in measurements if m["systemRef"] == sys_id]
            results = [
                {
                    "id": f"{calc_id}:{m['id']}",
                    "measurementRef": m["id"],
                    "value": m["value"] * scale,
                    "unit": "1/s",
                    "statisticalRelativeUncertainty": 0.01,
                }
                for m in mine
            ]
            calculations.append({
                "id": calc_id,
                "systemRef": sys_id,
                "code": "MCNP",
                "codeVersion": "6.3",
                "nuclearDataLibrary": lib,
                "librarySelector": f"xsdir-{tag}",
                "historiesRequested": "5E7",
                "varianceReduction": {
                    "method": "weightWindowMesh",
                    "generator": "notReported",
                    "scope": "perMeasurementSystem",
                    "note": "",
                },
                "inputs": [
                    {"artifactRef": "ART-DECK", "role": "transportInput",
                     "appliesTo": None},
                    {"artifactRef": f"ART-WW-{short}", "role": "weightWindow",
                     "appliesTo": None},
                ],
                "results": results,
            })
            comparisons += [
                {
                    "id": f"CMP-{calc_id}-{r['measurementRef']}",
                    "measurementRef": r["measurementRef"],
                    "calculationResultRef": r["id"],
                    "quantity": "C/E",
                    "value": scale,
                }
                for r in results
            ]

    return {
        "identification": {
            "id": BENCHMARK_ID,
            "title": "Synthetic test entry",
            "facility": "Nowhere",
            "year": 1999,
            "category": "shielding/test",
        },
        "documentation": {"references": {}},
        "systematics": [
            {
                "id": "NORM-X",
                "factors": [
                    {
                        "symbol": "P",
                        "value": 1.0,
                        "unit": "W",
                        "relativeUncertainty": None,
                        "uncertaintyMissingReason": "notReported",
                    }
                ],
            }
        ],
        "experiment": {
            "materials": [{"id": "MAT-1", "nuclides": [{"id": "Fe56"}]}],
            "measurementSystems": [
                {
                    "id": sys_id,
                    "targetNuclide": nuclide,
                    "reaction": f"{nuclide}(n,x)",
                    "productNuclide": "X",
                    "effectiveThreshold_MeV": 1.0 if sys_id == "SYS-AL" else None,
                    "dosimetryEvaluation": "IRDFF-II",
                    "foil": {"diameter_mm": 10.0, "thickness_mm": 1.0},
                }
                for sys_id, (nuclide, _pts) in SYSTEMS.items()
            ],
            "observables": [
                {
                    "id": f"OBS-{sys_id.split('-')[1]}",
                    "systemRef": sys_id,
                    "name": "reaction rate",
                    "unit": "1/s",
                    "unit_confidence": "derived" if sys_id == "SYS-AL" else "reported",
                    "unit_note": "reconstructed",
                }
                for sys_id in SYSTEMS
            ],
            "measurements": measurements,
        },
        "calculations": calculations,
        "comparisons": comparisons,
        "sensitivities": [
            {
                "id": "SENS-1",
                "systemRef": "SYS-AL",
                "position": "P1",
                "measurementRef": "M-AL-1",
                "nuclearDataLibrary": "TEST-LIB",
                "parameter": {
                    "kind": "nuclearData",
                    "targetNuclide": "Fe56",
                    "reactions": [2],
                    "reactionNames": ["(n,el)"],
                },
                "convention": "relative-relative",
                "method": "PERT",
                "includesImplicitEffects": False,
                "dataOrigin": "syntheticDemo",
                "coefficientsRef": "arrays.npz#/sens/coefficients",
                "groupStructureRef": "arrays.npz#/sens/boundaries",
            }
        ],
        "artifacts": [
            {
                "id": "ART-DECK",
                "role": "transportInput",
                "format": "MCNP",
                "path": "decks/base.mcnp",
                "sha256": "0" * 64,
                "includedInPackage": False,
                "derivedFrom": None,
                "modification": None,
                "reason": "byReference",
            },
            {
                "id": "ART-DECK-MOD",
                "role": "transportInput",
                "format": "MCNP",
                "path": "decks/mod.mcnp",
                "sha256": "1" * 64,
                "includedInPackage": False,
                "derivedFrom": "ART-DECK",
                "modification": "changed one material",
                "reason": "byReference",
            },
        ]
        + [
            {
                "id": f"ART-WW-{tag}",
                "role": "weightWindow",
                "format": "MCNP wwinp",
                "path": None,
                "sha256": None,
                "includedInPackage": False,
                "derivedFrom": None,
                "modification": None,
                "reason": "notInThisExtract",
            }
            for tag in ("AL", "AU")
        ],
        "dataQuality": [
            {
                "id": "DQ-TEST",
                "severity": "warning",
                "affects": "all measurements",
                "summary": "A finding the entry carries about itself.",
                "detail": "Long form.",
                "evidence": {},
            }
        ],
        "provenance": {"statement": "test fixture"},
    }


@pytest.fixture
def package(tmp_path):
    """Write the fixture entry as a directory and as a .sinbad archive."""
    pkg = tmp_path / "test-shield-01"
    pkg.mkdir()
    (pkg / "benchmark.json").write_text(json.dumps(_model()))
    (pkg / "manifest.xml").write_text("<manifest/>")
    np.savez_compressed(
        pkg / "arrays.npz",
        **{
            "sens/coefficients": np.array([[-0.1, -0.2, -0.3]]),
            "sens/boundaries": np.array([0.1, 1.0, 5.0, 20.0]),
        },
    )

    archive = tmp_path / "test-shield-01.sinbad"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(p.name for p in pkg.iterdir()):
            z.writestr(f"{pkg.name}/{f}", (pkg / f).read_bytes())

    yield {"dir": pkg, "archive": archive, "root": tmp_path}
    sinbad.reset_config()


# -- package forms ---------------------------------------------------------


def test_both_package_forms_open(package):
    for form in ("dir", "archive"):
        b = sinbad.SinbadBenchmark(package[form])
        assert b.id == BENCHMARK_ID
        b.close()


def test_forms_are_indistinguishable(package):
    a = sinbad.SinbadBenchmark(package["dir"])
    z = sinbad.SinbadBenchmark(package["archive"])
    assert a.model == z.model
    assert a.to_dataframe().equals(z.to_dataframe())
    assert a.package.kind == "directory" and z.package.kind == "archive"
    a.close()
    z.close()


def test_arrays_load_from_inside_the_archive(package):
    with sinbad.SinbadBenchmark(package["archive"]) as b:
        s = b.sensitivity("SENS-1")
        assert s.coefficients.shape == (1, 3)
        assert len(s.group_boundaries) == 4


def test_unknown_path_raises(tmp_path):
    with pytest.raises(sinbad.PackageNotFoundError):
        sinbad.SinbadBenchmark(tmp_path / "nothing-here")


# -- library ---------------------------------------------------------------


def test_library_discovery_prefers_the_archive(package):
    sinbad.configure(path=str(package["root"]))
    assert sinbad.list_benchmarks() == [BENCHMARK_ID]
    assert sinbad.scan()[BENCHMARK_ID].suffix == ".sinbad"


def test_catalogue_columns(package):
    sinbad.configure(path=str(package["root"]))
    cat = sinbad.catalogue()
    assert list(cat["id"]) == [BENCHMARK_ID]
    assert cat.loc[0, "measurements"] == 5
    assert cat.loc[0, "libraries"] == "LIB-A, LIB-B"


def test_open_by_identifier_and_by_substring(package):
    sinbad.configure(path=str(package["root"]))
    assert sinbad.open(BENCHMARK_ID).id == BENCHMARK_ID
    assert sinbad.open("SHIELD-01").id == BENCHMARK_ID


def test_open_accepts_a_path_without_configuration(package):
    assert sinbad.SinbadBenchmark.open(package["archive"]).id == BENCHMARK_ID


def test_missing_identifier_lists_what_is_available(package):
    sinbad.configure(path=str(package["root"]))
    with pytest.raises(sinbad.PackageNotFoundError, match=BENCHMARK_ID):
        sinbad.open("NOT-A-BENCHMARK")


def test_unconfigured_library_raises(tmp_path):
    sinbad.configure(path=str(tmp_path / "does-not-exist"))
    with pytest.raises(sinbad.LibraryNotConfiguredError):
        sinbad.list_benchmarks()


# -- measurement systems ---------------------------------------------------


def test_systems_are_objects(package):
    b = sinbad.SinbadBenchmark(package["dir"])
    assert [s.id for s in b.systems] == ["SYS-AL", "SYS-AU"]
    assert b.system("Al27").target_nuclide == "Al27"
    # A capture reaction has no threshold, and that is not the same as unknown.
    assert b.system("SYS-AU").threshold_mev is None


def test_system_resolves_by_id_nuclide_or_fragment(package):
    b = sinbad.SinbadBenchmark(package["dir"])
    for key in ("SYS-AL", "Al27", "al27", "AL"):
        assert len(b.measurements(key)) == 2


def test_ambiguous_or_unknown_system_lists_the_real_ones(package):
    b = sinbad.SinbadBenchmark(package["dir"])
    with pytest.raises(KeyError, match="SYS-AL"):
        b.measurements("SYS")          # ambiguous
    with pytest.raises(KeyError, match="SYS-AL"):
        b.measurements("Pu239")        # unknown


# -- objects and tables ----------------------------------------------------


def test_measurements(package):
    b = sinbad.SinbadBenchmark(package["dir"])
    m = b.measurements("Al27")[0]
    assert m.position == "A1"
    assert m.system == "SYS-AL"
    assert m.rel_unc == pytest.approx(_total("SYS-AL"))
    assert m.abs_unc == pytest.approx(2.0e-8 * _total("SYS-AL"))
    # The split that decides what an aggregate is worth.
    assert m.correlated_rel_unc == pytest.approx(math.hypot(CAL["SYS-AL"], SRC))
    assert m.independent_rel_unc == pytest.approx(STAT)


def test_to_dataframe(package):
    b = sinbad.SinbadBenchmark(package["dir"])
    df = b.to_dataframe()
    assert len(df) == 5
    assert set(df.columns) >= {"system", "nuclide", "value", "rel_unc", "abs_unc",
                               "correlated_unc", "independent_unc"}
    assert len(b.to_dataframe("Au197")) == 3


def test_ce_long_and_wide(package):
    b = sinbad.SinbadBenchmark(package["dir"])
    long = b.ce()
    assert len(long) == 10
    assert (long["C"] / long["E"]).round(6).equals(long["ce"].round(6))

    # Position labels repeat across systems, so the wide index has to as well.
    wide = b.ce(wide=True)
    assert list(wide.columns) == ["LIB-A", "LIB-B"]
    assert wide.index.names == ["system", "position"]
    assert wide.loc[("SYS-AL", "A1"), "LIB-A"] == pytest.approx(1.10)

    one = b.ce(system="Al27", wide=True)
    assert one.index.name == "position"


def test_ce_unknown_library_lists_the_real_ones(package):
    b = sinbad.SinbadBenchmark(package["dir"])
    with pytest.raises(KeyError, match="LIB-A"):
        b.ce(library="LIB-Z")


def test_covariance_block_structure(package):
    """Within a system: calibration + source. Across systems: source only."""
    b = sinbad.SinbadBenchmark(package["dir"])
    cov = b.covariance()
    assert cov.shape == (5, 5)

    within = CAL["SYS-AL"] ** 2 + SRC**2
    across = SRC**2
    assert cov.loc["EXP-AL-A1", "EXP-AL-A1"] == pytest.approx(_total("SYS-AL") ** 2)
    assert cov.loc["EXP-AL-A1", "EXP-AL-A2"] == pytest.approx(within)
    assert cov.loc["EXP-AL-A1", "EXP-AU-A1"] == pytest.approx(across)
    assert np.allclose(cov.values, cov.values.T)
    assert np.linalg.eigvalsh(cov.values).min() > -1e-12


def test_covariance_of_one_system_only(package):
    cov = sinbad.SinbadBenchmark(package["dir"]).covariance("Au197")
    assert cov.shape == (3, 3)
    assert cov.iloc[0, 1] == pytest.approx(CAL["SYS-AU"] ** 2 + SRC**2)


def test_covariance_absolute_scales_by_values(package):
    b = sinbad.SinbadBenchmark(package["dir"])
    rel = b.covariance(relative=True)
    absolute = b.covariance(relative=False)
    v = b.to_dataframe()["value"].values
    assert np.allclose(absolute.values, rel.values * np.outer(v, v))


def test_uncertainty_of_mean_quantifies_the_correlation(package):
    b = sinbad.SinbadBenchmark(package["dir"])
    u = b.uncertainty_of_mean("Al27")
    n = 2
    var_full = (_total("SYS-AL") ** 2 + CAL["SYS-AL"] ** 2 + SRC**2) / n
    assert u["full"] == pytest.approx(math.sqrt(var_full))
    assert u["diagonal_only"] == pytest.approx(_total("SYS-AL") / math.sqrt(n))
    assert u["factor"] > 1.0

    budget = b.uncertainty_budget()
    assert list(budget["scope"]) == ["whole entry", "SYS-AL", "SYS-AU"]
    assert (budget["factor"] > 1.0).all()


def test_covariance_ignores_undeclared_correlation():
    """A scope of ``unknown`` must stay on the diagonal, not be promoted."""
    model = _model()
    for m in model["experiment"]["measurements"]:
        for c in m["uncertainty"]["components"]:
            c["correlationScope"] = {"kind": "unknown"}
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "p"
        pkg.mkdir()
        (pkg / "benchmark.json").write_text(json.dumps(model))
        cov = sinbad.SinbadBenchmark(pkg).covariance()
        off = cov.values - np.diag(np.diag(cov.values))
        assert not off.any()


# -- calculation inputs (design report Sec. 3.11) --------------------------


def test_calculation_carries_its_own_inputs(package):
    b = sinbad.SinbadBenchmark(package["dir"])
    calc = b.calculations[0]
    assert [i.role for i in calc.inputs] == ["transportInput", "weightWindow"]
    assert calc.system == "SYS-AL"
    # The library as the code resolved it, not only as a human labelled it.
    assert calc.library_selector == "xsdir-L1"


def test_absent_weight_window_makes_a_calculation_irreproducible(package):
    b = sinbad.SinbadBenchmark(package["dir"])
    rep = b.reproducibility()
    assert len(rep) == 4
    assert not rep["reproducible"].any()
    assert (rep["available"] == 1).all()
    assert (rep["missing_roles"] == "weightWindow").all()


def test_modified_deck_declares_what_it_derives_from(package):
    arts = {a["id"]: a for a in sinbad.SinbadBenchmark(package["dir"]).artifacts}
    assert arts["ART-DECK-MOD"]["derivedFrom"] == "ART-DECK"
    assert "changed one material" in arts["ART-DECK-MOD"]["modification"]


def test_unresolved_reports_every_kind_of_gap(package):
    unres = sinbad.SinbadBenchmark(package["dir"]).unresolved()
    what = " | ".join(unres["what"])
    assert "relativeUncertainty" in what
    assert "input file" in what
    # A reconstructed component is neither missing nor reported, and is listed
    # once per type rather than once per point.
    assert "countingStatistics component" in what
    assert (unres["reason"] == "reconstructed, not reported").sum() == 1


def test_findings_are_carried_by_the_entry(package):
    f = sinbad.SinbadBenchmark(package["dir"]).findings()
    assert list(f["id"]) == ["DQ-TEST"]
    assert f.loc[0, "severity"] == "warning"


# -- presentation ----------------------------------------------------------


def test_summary_reports_systems_correlations_and_warnings(package):
    text = sinbad.SinbadBenchmark(package["dir"]).summary()
    assert BENCHMARK_ID in text
    assert "derived" in text          # the Al observable's unit confidence
    assert "DQ-TEST" in text          # data-quality warning surfaced
    assert "understates" in text      # the correlation factor is called out


def test_sensitivity_dataframe(package):
    s = sinbad.SinbadBenchmark(package["dir"]).sensitivity("SENS-1")
    df = s.to_dataframe()
    assert len(df) == 3
    assert list(df["reaction"].unique()) == ["(n,el)"]
    assert list(df["mt"].unique()) == [2]
    assert (df["e_low"] < df["e_high"]).all()


def test_unknown_sensitivity_lists_the_real_ones(package):
    b = sinbad.SinbadBenchmark(package["dir"])
    with pytest.raises(KeyError, match="SENS-1"):
        b.sensitivity("SENS-NOPE")


def test_plots_return_axes(package):
    import matplotlib

    matplotlib.use("Agg")
    b = sinbad.SinbadBenchmark(package["dir"])
    # Several systems -> small multiples, one panel each.
    axes = sinbad.plot_ce(b)
    assert axes.size >= len(b.systems)
    assert sinbad.plot_ce(b, system="Al27") is not None
    assert sinbad.plot_uncertainty_budget(b) is not None
    assert sinbad.plot_sensitivity(b.sensitivity("SENS-1")) is not None


def test_sensitivities_filter_by_system(package):
    b = sinbad.SinbadBenchmark(package["dir"])
    assert len(b.sensitivities()) == 1
    assert len(b.sensitivities("Al27")) == 1
    assert b.sensitivities("Au197") == []


def test_sensitivity_profile_found_by_position(package):
    b = sinbad.SinbadBenchmark(package["dir"])
    assert b.sensitivity_profile("Al27", "P1").id == "SENS-1"
    with pytest.raises(KeyError, match="P9"):
        b.sensitivity_profile("Al27", "P9")


def test_sensitivity_summary_integrates_each_reaction(package):
    b = sinbad.SinbadBenchmark(package["dir"])
    df = b.sensitivity_summary()
    s = b.sensitivity("SENS-1")
    assert df.loc[("SYS-AL", "P1"), "(n,el)"] == pytest.approx(
        s.coefficients.sum()
    )


def test_sensitivity_carries_its_library_and_measurement(package):
    s = sinbad.SinbadBenchmark(package["dir"]).sensitivity("SENS-1")
    # A profile belongs to one library; comparing it against another's C/E
    # would be a category error, so the object has to be able to say which.
    assert s.nuclear_data_library == "TEST-LIB"
    assert s.measurement == "M-AL-1"
    assert s.mts == [2]
