"""Tier-3 golden: the thesis pipeline, on committed data.

``exfor_to_endf_sampling_v2.py`` is the evaluation pipeline the thesis rests
on. It runs on the ASNR cluster against 139 EXFOR datasets and a 27 MB Fe-56
tape, which is why nothing has ever tested it end to end. This runs it here,
on four committed datasets and the committed Fe-56 slice, with a fixed seed.

No production code was changed to make this possible.
``run_exfor_to_endf_sampling_v2`` already takes every knob as a keyword
argument with the module constant as its default, so a test just calls it. The
one exception is ``TOF_PARAMETERS_FILE``, read from module scope, which is
monkeypatched; and the manifest, which is already redirectable through
``$KIKA_UNCERTAINTY_MANIFEST_PATH``.

**Fixtures.** ``exfor_micro.db`` (94 KB) holds four real datasets chosen so
that between them they reach every branch of the uncertainty resolver;
``uncertainty_manifest_micro.yaml`` is written for them, not trimmed from
production, because the only production dataset with a ``sys_dep`` block is
Cierjacks 20743002 and its data blob alone is 1.6 MB. See the header of the
YAML for which dataset covers which branch.

**What is covered.** EXFOR loading, the per-point uncertainty resolution, the
nominal GLS Legendre fits, the Monte-Carlo sampling and the Legendre
covariance — everything the pipeline produces up to and including
``legendre_covariance.npy``. Two energy windows are frozen: one holding a
single experiment, one holding two, because the between-experiment Kish
weighting and discrepancy treatment only come into play in the second. The
denser covariance it produces (144 non-zero entries against 49) is that
machinery showing up.

**What is not.** The final ENDF-writing stage. It fails on this data, in every
window tried, on a duplicated point in the spliced energy grid. That is
recorded below as a pinned defect, not worked around: an untested stage that is
*known* untested is a smaller problem than one silently skipped.

**How thin it is.** Four datasets over two windows give two fitted energy bins
each, so the frozen covariance is 16x16. That is enough to catch a change in
the fit, the weighting or the sampling, and it is not enough to be called
coverage of the evaluation. Widening it means committing more EXFOR data.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).resolve().parent / "data"
REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MICRO_DB = DATA / "exfor_micro.db"
MICRO_TOF = DATA / "tof_parameters_micro.json"
MICRO_MANIFEST = DATA / "uncertainty_manifest_micro.yaml"

REGEN = bool(os.environ.get("REGEN_PIPELINE_GOLDEN"))

#: The four datasets in the fixture database.
FIXTURE_DATASETS = ["10886002", "11300004", "13606002", "23059003"]

#: Fixed run configuration, minus the energy window. Every one of these is part
#: of the golden: change any of them and the answer changes, legitimately.
RUN = dict(
    n_samples=4,
    base_seed=42,
    n_procs=1,
    target_zaid=[26000, 26056],
    exfor_source="database",
    generate_fitting_ace=False,
    save_covariance_files=True,
    verbose_diagnostics=False,
)

#: Two windows, for two different things.
#:
#: ``single`` spans all of dataset 10886002 (1.684-3.905 MeV, 47 energies) and
#: nothing else, so the fit sees one experiment and the between-experiment
#: machinery stays out of the way.
#:
#: ``pair`` puts 11300004 (one energy, 5 MeV) and 13606002 (4.5-9.99 MeV) in
#: the same window, which is what exercises the Kish weighting and the
#: between-experiment discrepancy — the part of the method that a
#: single-experiment window cannot reach.
WINDOWS = {
    "single": (1.7, 4.0),
    "pair": (4.4, 5.6),
}


# ---------------------------------------------------------------------------
# EXFOR loading and uncertainty resolution
# ---------------------------------------------------------------------------

def test_the_fixture_database_holds_what_the_manifest_describes():
    """Fixture and manifest must not drift apart."""
    import yaml

    from kika.exfor import read_all_exfor

    loaded = read_all_exfor(
        source="database", db_path=str(MICRO_DB),
        target=["Fe-0", "Fe-56"], group_by_energy=False,
    )
    assert sorted(loaded) == FIXTURE_DATASETS

    manifest = yaml.safe_load(MICRO_MANIFEST.read_text())
    described = set(manifest["datasets"])
    assert described < set(FIXTURE_DATASETS), (
        "the manifest describes a dataset the fixture database does not hold"
    )
    # 23059003 is absent on purpose: it is the defaults-fallback case.
    assert set(FIXTURE_DATASETS) - described == {"23059003"}


def test_fixture_datasets_keep_their_energy_coverage():
    """The windows in RUN only mean something if the data is where we think."""
    from kika.exfor import read_all_exfor

    loaded = read_all_exfor(
        source="database", db_path=str(MICRO_DB),
        target=["Fe-0", "Fe-56"], group_by_energy=False,
    )
    coverage = {
        did: (float(ds.energies().min()), float(ds.energies().max()))
        for did, ds in loaded.items()
    }
    assert coverage["10886002"] == pytest.approx((1.684, 3.905), rel=1e-6)
    assert coverage["11300004"] == pytest.approx((5.0, 5.0), rel=1e-6)
    assert coverage["13606002"] == pytest.approx((4.5, 9.99), rel=1e-6)
    assert coverage["23059003"] == pytest.approx((96.0, 96.0), rel=1e-6)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", params=sorted(WINDOWS), ids=sorted(WINDOWS))
def pipeline_run(request, tmp_path_factory, micro_tape):
    """Run the pipeline once per window, up to the covariance."""
    import scripts.exfor_to_endf_sampling_v2 as pipeline

    window = request.param
    energy_min_mev, energy_max_mev = WINDOWS[window]
    out = tmp_path_factory.mktemp(f"pipeline_{window}")
    (out / "empty_json").mkdir()

    previous_manifest = os.environ.get("KIKA_UNCERTAINTY_MANIFEST_PATH")
    previous_tof = pipeline.TOF_PARAMETERS_FILE
    os.environ["KIKA_UNCERTAINTY_MANIFEST_PATH"] = str(MICRO_MANIFEST)
    pipeline.TOF_PARAMETERS_FILE = str(MICRO_TOF)
    try:
        pipeline.run_exfor_to_endf_sampling_v2(
            endf_file=str(micro_tape),
            output_dir=str(out),
            # Required even though nothing reads it — see the pinned defect
            # test at the end of this file.
            exfor_directory=str(out / "empty_json"),
            exfor_db_path=str(MICRO_DB),
            # The ENDF-writing stage fails on this data; the golden stops at
            # the covariance. See test_writing_endf_samples_succeeds below.
            generate_fitting_samples=False,
            generate_nominal_endf=False,
            generate_mc_mean_endf=False,
            energy_min_mev=energy_min_mev,
            energy_max_mev=energy_max_mev,
            **RUN,
        )
    finally:
        pipeline.TOF_PARAMETERS_FILE = previous_tof
        if previous_manifest is None:
            os.environ.pop("KIKA_UNCERTAINTY_MANIFEST_PATH", None)
        else:
            os.environ["KIKA_UNCERTAINTY_MANIFEST_PATH"] = previous_manifest
    return window, out


@pytest.mark.slow
def test_pipeline_produces_its_expected_artifacts(pipeline_run):
    """The run wrote what a run is supposed to write."""
    _, output_dir = pipeline_run
    for name in (
        "nominal_fits.parquet",
        "legendre_samples_tmc.parquet",
        "legendre_covariance.npy",
        "run_metadata.json",
    ):
        assert (output_dir / name).is_file(), f"{name} was not produced"


@pytest.mark.slow
def test_pipeline_golden(pipeline_run):
    """Freeze the numbers: the nominal fits and the Legendre covariance.

    Same fixture data, same seed, same answer. This is the only test in the
    repository that would notice if a change to the fitting, the uncertainty
    resolution or the Monte-Carlo sampling moved the evaluation.
    """
    import pandas as pd

    window, output_dir = pipeline_run
    covariance = np.load(output_dir / "legendre_covariance.npy")
    fits = pd.read_parquet(output_dir / "nominal_fits.parquet")

    numeric = fits.select_dtypes(include=[np.number]).reindex(
        sorted(fits.select_dtypes(include=[np.number]).columns), axis=1
    )

    produced = {
        "covariance": covariance,
        "fits_numeric": numeric.to_numpy(dtype=float),
        "fits_columns": np.array(sorted(fits.columns)),
        "fits_shape": np.array(fits.shape),
    }

    golden_path = DATA / f"pipeline_golden_{window}.npz"

    if REGEN:
        np.savez_compressed(golden_path, **produced)
        pytest.skip("golden regenerated")

    if not golden_path.is_file():
        pytest.fail(
            f"{golden_path.name} is missing — generate it with "
            "REGEN_PIPELINE_GOLDEN=1 and commit it"
        )

    with np.load(golden_path, allow_pickle=False) as golden:
        assert sorted(golden.files) == sorted(produced)
        for key in sorted(produced):
            want, have = golden[key], np.asarray(produced[key])
            assert have.shape == want.shape, f"{key}: {want.shape} -> {have.shape}"
            if want.dtype.kind in "US":
                assert list(have) == list(want), f"{key} changed"
                continue
            np.testing.assert_allclose(
                have, want, rtol=1e-10, atol=0.0, err_msg=f"{key} moved"
            )


@pytest.mark.slow
def test_covariance_is_symmetric_and_positive_semidefinite(pipeline_run):
    """Structural facts, checked as well as frozen.

    A golden says "the same as last time". These say "and still meaningful".
    """
    _, output_dir = pipeline_run
    covariance = np.load(output_dir / "legendre_covariance.npy")
    assert covariance.ndim == 2 and covariance.shape[0] == covariance.shape[1]
    np.testing.assert_allclose(covariance, covariance.T, rtol=1e-10, atol=1e-30)

    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = max(abs(eigenvalues[-1]), 1e-300)
    assert eigenvalues[0] / scale > -1e-8, (
        f"covariance is not PSD: lambda_min/lambda_max = {eigenvalues[0] / scale:.3e}"
    )


# ---------------------------------------------------------------------------
# Pinned defects
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason=(
        "Pre-existing defect, pinned by GNDS phase 0. "
        "exfor_to_endf_sampling_v2.py:2557 runs os.path.isdir(exfor_directory) "
        "unconditionally, before the branch that decides whether JSON files are "
        "read at all. With exfor_source='database' the directory is never used, "
        "yet omitting it raises TypeError from genericpath rather than the "
        "graceful error the surrounding code is written to give. Production "
        "never hits it because the __main__ block always passes the module "
        "constant. Every test above has to create an empty directory to get "
        "past this line."
    ),
)
def test_database_source_does_not_need_a_json_directory(tmp_path, micro_tape):
    import scripts.exfor_to_endf_sampling_v2 as pipeline

    pipeline.run_exfor_to_endf_sampling_v2(
        endf_file=str(micro_tape),
        output_dir=str(tmp_path),
        exfor_db_path=str(MICRO_DB),
        exfor_source="database",
        target_zaid=[26000, 26056],
        n_samples=2,
        energy_min_mev=1.8,
        energy_max_mev=3.0,
        base_seed=42,
        n_procs=1,
        generate_fitting_samples=False,
        generate_nominal_endf=False,
        generate_mc_mean_endf=False,
    )


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Pre-existing defect, pinned by GNDS phase 0. Writing the perturbed "
        "ENDF samples fails with 'Spliced grid not sorted at index N: "
        "3905000.0 >= 3905000.0'. 3.905 MeV is the top energy of dataset "
        "10886002 and it coincides exactly with a point already on the Fe-56 "
        "MF4 grid, so the splice emits it twice and the sortedness assertion "
        "rejects it. Reproduced in two different energy windows (1.8-3.0 and "
        "4.5-5.5 MeV), and note that 3.905 MeV lies outside both — so the "
        "splice is also pulling in energies from beyond the requested window, "
        "which is worth understanding before fixing the duplicate. Until this "
        "is resolved the golden above covers the pipeline only as far as the "
        "Legendre covariance; the ENDF-writing stage is untested."
    ),
)
def test_writing_endf_samples_succeeds(tmp_path, micro_tape):
    import scripts.exfor_to_endf_sampling_v2 as pipeline

    (tmp_path / "empty_json").mkdir()
    previous_tof = pipeline.TOF_PARAMETERS_FILE
    os.environ["KIKA_UNCERTAINTY_MANIFEST_PATH"] = str(MICRO_MANIFEST)
    pipeline.TOF_PARAMETERS_FILE = str(MICRO_TOF)
    try:
        pipeline.run_exfor_to_endf_sampling_v2(
            endf_file=str(micro_tape),
            output_dir=str(tmp_path),
            exfor_directory=str(tmp_path / "empty_json"),
            exfor_db_path=str(MICRO_DB),
            generate_fitting_samples=True,
            energy_min_mev=WINDOWS["single"][0],
            energy_max_mev=WINDOWS["single"][1],
            **RUN,
        )
    finally:
        pipeline.TOF_PARAMETERS_FILE = previous_tof
        os.environ.pop("KIKA_UNCERTAINTY_MANIFEST_PATH", None)

    written = sorted((tmp_path / "endf").rglob("*.endf"))
    assert written, "no perturbed ENDF samples were written"
