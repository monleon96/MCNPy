"""The kika-app contract: every name the desktop backend imports must exist.

`kika/tests/` holds invariants of the repository as a whole. This is the third
of them.

**Why kika owns this test.** ``kika-app``'s FastAPI backend imports 75 names
from 45 kika modules, and is shipped as a **PyInstaller-frozen binary** driven
by a hardcoded ``hiddenimports`` list in ``kika-app/kika-api.spec``. Two
consequences: a rename in this repository breaks a separate repository's
release build, and it does so *at run time inside a frozen binary*, which is
the worst place to find out. The app cannot defend itself — it pins
``kika-nd`` by commit, so it only meets the breakage when someone bumps the
pin. So the check lives here, on the side that does the renaming.

**Why the table is hardcoded rather than parsed.** ``kika-api.spec`` lives in
another repository that is not guaranteed to be checked out beside this one,
and kika must not grow a test dependency on kika-app. The mirror-image test —
``ast``-parsing the spec and importing every entry — lives on the app side, in
``kika-api/tests/test_kika_contract.py``. Together they make the spec
self-testing from both directions.

**Regenerating.** From the kika-app checkout::

    python - <<'PY'
    import ast, pathlib, collections
    spec = ast.parse(pathlib.Path("kika-api.spec").read_text())
    ...  # collect hiddenimports, then walk kika-api/app/**/*.py for ImportFrom
    PY

Harvested 2026-08-06 against kika-app ``develop`` at 6b3cb00.
"""
from __future__ import annotations

import importlib

import pytest

#: ``kika.*`` entries of ``hiddenimports`` in ``kika-app/kika-api.spec``.
#: PyInstaller bundles exactly these; a module that vanishes from the library
#: makes the frozen build fail to assemble or to run.
APP_MODULES = (
    "kika",
    "kika._utils",
    "kika._constants",
    "kika.utils",
    "kika.energy_grids",
    "kika.ace",
    "kika.ace.parsers",
    "kika.endf",
    "kika.endf.read_endf",
    "kika.endf.remote",
    "kika.endf.remote.exceptions",
    "kika.plotting",
    "kika.plotting.plot_builder",
    "kika.plotting.heatmap_builder",
    "kika.plotting.plot_data",
    "kika.plotting.comparison",
    "kika.sampling",
    "kika.sampling.utils",
    "kika.sampling.ace_perturbation",
    "kika.sampling.endf_perturbation",
    "kika.sampling.ace_perturbation_separate",
    "kika.mcnp",
    "kika.mcnp.material",
    "kika.mcnp.parse_materials",
    "kika.mcnp.parse_input",
    "kika.mcnp.parse_mctal",
    "kika.mcnp.pert_generator",
    "kika.njoy",
    "kika.njoy.modules",
    "kika.processing",
    "kika.processing.derived_covariance",
    "kika.processing.group_averaging",
    "kika.processing.linearization",
    "kika.processing.multigroup",
    "kika.processing.njoy_reconstruct",
    "kika.processing.penetration",
    "kika.processing.reconstruct",
    "kika.processing.resonance_formulas",
    "kika.processing.urr_formulas",
    "kika.exfor",
    "kika.exfor.database",
    "kika.sensitivities",
    "kika.sensitivities.sdf_parser",
    "kika.sensitivities.sdf",
    "kika.sensitivities.sensitivity_processing",
    "kika.UQ",
    "kika.UQ.sandwich",
)

#: Every ``from <module> import <name>`` in ``kika-api/app/**``, including the
#: ones written inside request handlers — which is most of the interesting
#: ones, and exactly what a grep over module headers misses.
APP_NAMES = (
    ("kika.UQ", "convergence_analysis"),
    ("kika.UQ", "fastTMC"),
    ("kika.UQ", "histogram_data"),
    ("kika.UQ", "normality_tests"),
    ("kika.UQ", "qq_plot_data"),
    ("kika.UQ.sandwich", "sandwich_uncertainty_propagation"),
    ("kika._constants", "ATOMIC_NUMBER_TO_SYMBOL"),
    ("kika._constants", "MT_COMPOSITE_ORDER"),
    ("kika._constants", "MT_TO_REACTION"),
    ("kika._constants", "NATURAL_ABUNDANCE"),
    ("kika._constants", "SYMBOL_TO_ATOMIC_NUMBER"),
    ("kika._utils", "MeV_to_kelvin"),
    ("kika._utils", "symbol_to_zaid"),
    ("kika._utils", "zaid_to_symbol"),
    ("kika.ace.parsers", "read_ace"),
    ("kika.cov.cross_section_covariance", "CrossSectionCovariance"),
    ("kika.endf.classes.mf2.mf2mt151", "RMatrixLimited"),
    ("kika.endf.classes.mf2.mf2mt151", "ResolvedResonanceRange"),
    ("kika.endf.classes.mf2.mf2mt151", "ScatteringRadiusOnly"),
    ("kika.endf.classes.mf2.mf2mt151", "UnresolvedCaseA"),
    ("kika.endf.classes.mf2.mf2mt151", "UnresolvedCaseB"),
    ("kika.endf.classes.mf2.mf2mt151", "UnresolvedCaseC"),
    ("kika.endf.processing", "detect_resonance_bounds"),
    ("kika.endf.read_endf", "read_endf"),
    ("kika.endf.remote", "download_endf"),
    ("kika.endf.remote", "parse_isotope"),
    ("kika.endf.remote.exceptions", "IsotopeNotFoundError"),
    ("kika.endf.remote.exceptions", "LibraryNotFoundError"),
    ("kika.endf.remote.exceptions", "NetworkError"),
    ("kika.endf.utils", "parse_endf_id"),
    ("kika.endf.writers.section_ops", "remove_sections"),
    ("kika.endf.writers.update_directory", "update_mf1_directory"),
    ("kika.energy_grids", "equal_lethargy_grid"),
    ("kika.energy_grids", "grids"),
    ("kika.energy_grids.utils", "_identify_energy_grid_or_subset"),
    ("kika.exfor.angular_distribution", "ExforAngularDistribution"),
    ("kika.exfor.database", "X4ProDatabase"),
    ("kika.materials", "Material"),
    ("kika.materials", "MaterialCollection"),
    ("kika.materials", "Nuclide"),
    ("kika.materials", "read_material"),
    ("kika.materials.parse_serpent", "read_serpent_materials"),
    ("kika.mcnp", "detect_mcnp_file_type"),
    ("kika.mcnp", "read_mcnp"),
    ("kika.mcnp.parse_input", "read_mcnp"),
    ("kika.mcnp.parse_mctal", "read_mctal"),
    ("kika.mcnp.pert_generator", "perturb_material"),
    ("kika.njoy", "read_deck"),
    ("kika.njoy.modules", "generate_module"),
    ("kika.plotting.comparison", "ComparisonBuilder"),
    ("kika.plotting.heatmap_builder", "HeatmapBuilder"),
    ("kika.plotting.plot_builder", "PlotBuilder"),
    ("kika.plotting.plot_data", "CovarianceHeatmapData"),
    ("kika.plotting.plot_data", "LegendreHeatmapData"),
    ("kika.plotting.plot_data", "PlotData"),
    ("kika.processing", "resonance_group_average"),
    ("kika.processing.njoy_reconstruct", "NjoyReconstructError"),
    ("kika.processing.njoy_reconstruct", "njoy_reconstruct"),
    ("kika.processing.njoy_reconstruct", "njoy_reconstruct_stream"),
    ("kika.sampling.mf31_sampling", "build_mf31_covariance"),
    ("kika.sampling.utils", "load_covariance"),
    ("kika.sensitivities.sdf", "SDFData"),
    ("kika.sensitivities.sdf_parser", "read_sdf"),
    ("kika.sensitivities.sensitivity_processing", "compute_sensitivity"),
    ("kika.sensitivities.sensitivity_processing", "compute_total_sensitivity"),
    ("kika.sensitivities.sensitivity_processing", "create_sdf_data"),
    ("kika.sensitivities.sensitivity_processing", "create_sdf_from_serpent"),
    ("kika.serpent.parse_dep", "parse_depletion_text"),
    ("kika.serpent.parse_det", "parse_detector_text"),
    ("kika.serpent.parse_his", "parse_history_text"),
    ("kika.serpent.parse_input", "read_serpent_input"),
    ("kika.serpent.parse_res", "parse_results_text"),
    ("kika.serpent.parse_sens", "parse_sensitivity_text"),
    ("kika.serpent.sens", "SensitivityFile"),
)


@pytest.mark.parametrize("module", APP_MODULES)
def test_every_bundled_module_imports(module):
    """Each ``hiddenimports`` entry must be a real, importable module."""
    importlib.import_module(module)


@pytest.mark.parametrize(
    "module,name", APP_NAMES, ids=[f"{m}:{n}" for m, n in APP_NAMES]
)
def test_every_name_the_app_imports_exists(module, name):
    mod = importlib.import_module(module)
    assert hasattr(mod, name), (
        f"kika-app imports {name} from {module} and it is gone. "
        f"This breaks the desktop backend, not this repository — see "
        f"kika-api/app/ and kika-api.spec."
    )


def test_kika_utils_nuclear_data_still_does_not_exist():
    """A negative the app asserts on every startup.

    ``kika-api/app/main.py:85`` and ``routers/covariance.py:58`` log
    ``"kika.utils.nuclear_data EXISTS - this is the problem!"`` if this module
    ever comes back — it once shadowed the real ``kika.nuclear_data``. Phase 3
    adds a subpackage under ``kika/nuclear_data/`` and is exactly the kind of
    work that could resurrect it by accident, so the app's runtime guard gets
    a compile-time twin here.
    """
    importlib.import_module("kika.utils")  # must exist
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("kika.utils.nuclear_data")
