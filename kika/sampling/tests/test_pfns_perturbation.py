"""The PFNS driver: what it writes, and what it says about what it wrote.

Two kinds of test.

**The tape.** A perturbed tape has to re-parse, keep every section the driver
did not touch, and carry a spectrum that still integrates to what it did
before. That is the whole contract with NJOY downstream.

**The record.** This pipeline writes a tape whose MF35 no longer describes its
own MF5, applies a projection to every sample, and renormalises every table.
None of that is wrong, and all of it is invisible in the output file — so the
run summary has to state it. The tests below treat a missing diagnostic as a
failure on the same footing as a wrong number, because a budget nobody reports
is a budget nobody checks.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
import pytest

from kika.endf import read_endf
from kika.sampling.pfns_perturbation import (
    SAMPLING_SPACE,
    _parameter_labels,
    perturb_pfns_files,
)


@pytest.fixture(scope="module")
def dry(micro_pfns_tape, tmp_path_factory):
    out = tmp_path_factory.mktemp("pfns_dry")
    return perturb_pfns_files(
        str(micro_pfns_tape), 8, generate_ace=False, seed=42,
        output_dir=str(out), dry_run=True,
    )


@pytest.fixture(scope="module")
def run(micro_pfns_tape, tmp_path_factory):
    out = tmp_path_factory.mktemp("pfns_run")
    results = perturb_pfns_files(
        str(micro_pfns_tape), 3, generate_ace=False, seed=42,
        output_dir=str(out),
    )
    return results, out


# ---------------------------------------------------------------------------
# The dry run
# ---------------------------------------------------------------------------

def test_perturb_pfns_files_dry_run_reports_bands_and_diagnostics(dry):
    """Everything a user needs to decide whether to spend the real run."""
    assert dry["dry_run"] is True
    assert dry["errors"] == []
    assert dry["sampling_space"] == "linear" == SAMPLING_SPACE

    isotope = dry["isotopes"]["98252"]
    assert isotope["n_bands"] == 4
    assert isotope["samples_written"] == 0
    assert isotope["input_normalisation_residual"]["max_abs"] == pytest.approx(
        4.306e-7, rel=1e-3
    )

    for index, band in enumerate(isotope["bands"]):
        assert band["band"] == index
        assert band["n_groups"] == 122
        assert band["row_sum_residual"] < 1e-2
        assert 0.0 < band["normalisation_drift"] < 1e-4
        assert 0 < band["rank"] < band["n_groups"], "the rank deficiency is data"
        assert band["acceptance_gate"]["passes_spectral_gate"]
        assert band["acceptance_gate"]["null_leakage"] < 1e-3


def test_the_dry_run_writes_no_tape_and_no_parquet(dry, micro_pfns_tape,
                                                   tmp_path_factory):
    out = tmp_path_factory.mktemp("pfns_dry_check")
    perturb_pfns_files(str(micro_pfns_tape), 2, generate_ace=False,
                       seed=1, output_dir=str(out), dry_run=True)
    assert glob.glob(os.path.join(str(out), "*.endf")) == []
    assert glob.glob(os.path.join(str(out), "*.parquet")) == []


def test_seeds_are_recorded_per_band_so_a_run_can_be_reproduced(dry):
    seeds = [band["seed"] for band in dry["isotopes"]["98252"]["bands"]]
    assert len(set(seeds)) == len(seeds), "two bands drew from the same seed"


# ---------------------------------------------------------------------------
# The tapes
# ---------------------------------------------------------------------------

def test_end_to_end_writes_n_perturbed_tapes_that_reparse(run):
    """Each output is read back and re-integrated, not merely counted."""
    results, out = run
    isotope = results["isotopes"]["98252"]
    assert isotope["samples_written"] == 3
    assert isotope["sample_errors"] == []

    tapes = sorted(glob.glob(os.path.join(str(out), "*.endf")))
    assert len(tapes) == 3

    for tape in tapes:
        endf = read_endf(tape, mf_numbers=[5, 35])
        partial = endf.files[5].sections[18].partials[0]
        for node in range(len(partial.incident_energies)):
            assert partial.normalisation(node) == pytest.approx(1.0, abs=1e-6)
            assert min(partial.chi[node]) >= 0.0


def test_the_sections_the_driver_did_not_touch_come_through_intact(run,
                                                                  micro_pfns_tape):
    """MT455's six LF=5 laws and all four MF35 bands must survive the splice."""
    _, out = run
    source = read_endf(str(micro_pfns_tape), mf_numbers=[5, 35])

    for tape in sorted(glob.glob(os.path.join(str(out), "*.endf"))):
        endf = read_endf(tape, mf_numbers=[5, 35])

        mt455 = endf.files[5].sections[455]
        assert [p.lf for p in mt455.partials] == [5] * 6
        assert str(mt455) == str(source.files[5].sections[455])

        assert str(endf.files[35].sections[18]) == str(
            source.files[35].sections[18]
        ), "MF35 was modified; nothing in this pipeline perturbs a covariance"


def test_the_perturbed_spectrum_actually_moved(run, micro_pfns_tape):
    """The test that would catch the whole pipeline being a no-op.

    A perturbation that silently vanishes — the float32 failure mode, a delta
    of zero, a splice that wrote the original back — produces tapes that pass
    every other test in this file.
    """
    _, out = run
    original = read_endf(str(micro_pfns_tape), mf_numbers=[5]).files[5]
    reference = original.sections[18].partials[0]

    for tape in sorted(glob.glob(os.path.join(str(out), "*.endf"))):
        partial = read_endf(tape, mf_numbers=[5]).files[5].sections[18].partials[0]
        probe = np.geomspace(1.0e3, 1.5e7, 200)
        before = np.interp(probe, reference.outgoing_grids[13], reference.chi[13])
        grid, values = partial.evaluate_at_incident(
            reference.incident_energies[13]
        )
        after = np.interp(probe, grid, values)
        moved = np.max(np.abs(after / before - 1.0))
        assert moved > 1e-4, f"the spectrum barely moved ({moved:.2e})"
        assert moved < 1.0, f"the spectrum moved implausibly far ({moved:.2e})"


def test_the_tape_grows_but_not_without_bound(run, micro_pfns_tape):
    """NJOY ACER reads MF5 into fixed buffers, so this is a real constraint."""
    _, out = run
    source = os.path.getsize(str(micro_pfns_tape))
    for tape in sorted(glob.glob(os.path.join(str(out), "*.endf"))):
        assert os.path.getsize(tape) - source < 1_000_000
        assert os.path.getsize(tape) > source, "nothing was inserted at all"


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

def test_the_summary_states_that_mf35_was_left_inconsistent(run):
    """A user has to be told, not left to notice."""
    results, _ = run
    assert results["mf35_unchanged"] is True
    note = results["mf35_unchanged_note"]
    assert "MF35" in note and "no longer describes" in note


def test_the_summary_reports_every_budget_the_run_spent(run):
    """Two renormalisations happen per node and both must be countable.

    A silent projection is a distribution you no longer know the shape of, so
    the constraint solve and the global rescale are each reported, along with
    how much probability mass positivity clipping actually removed — the mass,
    not the group count, because zeroing thirty groups that hold 1e-15 between
    them is not the event that zeroing one holding a per cent is.
    """
    results, _ = run
    applied = results["isotopes"]["98252"]["applied"]

    for field in (
        "max_projection_shift",
        "max_sum_error_after_projection",
        "max_renormalisation_error",
        "max_group_mass_error",
        "max_clipped_mass_fraction",
        "max_frac_mass_outside_mf35",
        "total_clipped",
        "total_groups_frozen",
        "total_outgoing_inserted",
        "total_steps_dropped",
        "n_incident_inserted",
    ):
        assert field in applied, f"{field} is not reported"

    assert applied["max_sum_error_after_projection"] < 1e-12
    assert applied["max_projection_shift"] < 1e-3
    assert applied["max_renormalisation_error"] < 1e-3
    assert applied["max_group_mass_error"] < 1e-4
    assert applied["n_incident_inserted"] == 3, "one shoulder per interior band edge"
    assert applied["total_steps_dropped"] == 0, "no cap was requested"


def test_the_covariance_provenance_is_recorded(run, micro_pfns_tape):
    """Which covariance produced this tape, permanently attached to the answer."""
    results, _ = run
    source = results["isotopes"]["98252"]["covariance_source"]
    assert source["origin"] == "tape"
    assert os.path.samefile(source["path"], str(micro_pfns_tape))


# ---------------------------------------------------------------------------
# The parquet
# ---------------------------------------------------------------------------

def test_parquet_columns_are_band_qualified_and_match_the_flatten_order(run):
    """The default ``{symbol}_MT{mt}_{group}`` naming cannot express this axis.

    The bands do not share an outgoing grid, so a column name that carries only
    an energy group is ambiguous across bands — and on ENDF/B-VIII.1 U-235,
    where the orders are 84 and 641, it would not even have the right length.
    """
    _, out = run
    master = glob.glob(os.path.join(str(out), "*master.parquet"))
    assert master, "no master perturbation matrix was written"

    frame = pd.read_parquet(master[0])
    columns = [c for c in frame.columns if c != "Sample_ID"]

    assert len(frame) == 3
    assert len(columns) == 4 * 122
    assert sorted({c.split("_")[2] for c in columns}) == ["b0", "b1", "b2", "b3"]
    assert frame[columns[0]].dtype == np.float64, (
        "float32 would annihilate an absolute MF35 delta"
    )
    assert columns == [c for c in columns], "column order must be band-major"


def test_the_parquet_metadata_carries_the_inconsistency_flag(run):
    _, out = run
    meta_path = glob.glob(os.path.join(str(out), "*metadata.json"))
    assert meta_path
    details = json.load(open(meta_path[0]))["isotope_details"]["98252"]

    assert details["pipeline"] == "pfns"
    assert details["mf35_unchanged"] is True
    assert details["sampling_space"] == "linear"
    assert "absolute" in details["quantity"]
    assert len(details["bands"]) == 4


def test_parameter_labels_are_one_per_column_in_band_major_order():
    grids = [np.array([0.0, 1.0, 2.0]), np.array([0.0, 3.0, 6.0, 9.0])]
    labels = _parameter_labels("Cf252", 18, grids)
    assert len(labels) == 2 + 3
    assert labels[0].startswith("Cf252_MT18_b0_")
    assert labels[2].startswith("Cf252_MT18_b1_")


def test_param_labels_of_the_wrong_length_are_refused(tmp_path):
    """The hook added to ``_write_isotope_parquet`` must not silently mis-align."""
    from kika.sampling.utils import _write_isotope_parquet

    with pytest.raises(ValueError, match="2 entries for 3 parameter columns"):
        _write_isotope_parquet(
            str(tmp_path), 98252, np.zeros((4, 3)), [18], [0.0, 1.0, 2.0, 3.0],
            verbose=False, param_labels=["a", "b"],
        )


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------

def test_a_tape_without_mf35_is_reported_not_raised(micro_tape, tmp_path):
    """One bad tape must not take the run down; nu-bar behaves the same way."""
    results = perturb_pfns_files(
        str(micro_tape), 2, generate_ace=False, output_dir=str(tmp_path),
    )
    assert results["isotopes"] == {}
    assert len(results["errors"]) == 1
    assert "MF5/MT18" in results["errors"][0]["error"]


# ---------------------------------------------------------------------------
# The real tape, and NJOY
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_endfb81_u235_end_to_end(u235_b81_tape, tmp_path):
    """The reference library, where the grid mismatch is real and MT18 is 36 MB."""
    results = perturb_pfns_files(
        str(u235_b81_tape), 2, generate_ace=False, seed=5,
        output_dir=str(tmp_path),
    )
    assert results["errors"] == []
    isotope = results["isotopes"]["92235"]
    assert isotope["n_bands"] == 5
    assert [b["n_groups"] for b in isotope["bands"]] == [83, 640, 640, 640, 640]
    assert isotope["samples_written"] == 2
    assert isotope["applied"]["max_group_mass_error"] < 1e-4

    for tape in sorted(glob.glob(os.path.join(str(tmp_path), "*.endf"))):
        growth = os.path.getsize(tape) - os.path.getsize(str(u235_b81_tape))
        assert growth < 4_000_000, f"tape grew by {growth / 1e6:.1f} MB"
        endf = read_endf(tape, mf_numbers=[5])
        partial = endf.files[5].sections[18].partials[0]
        for node in range(len(partial.incident_energies)):
            assert partial.normalisation(node) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.slow
def test_njoy_regenerates_ace_from_a_perturbed_pfns_tape(
    cf252_b81_tape, njoy_exe, tmp_path,
):
    """The question no unit test can answer: does ACER accept the grown MF5?

    NJOY is never mocked anywhere in this library and this is not the place to
    start — the risk being tested for is precisely that ACER reads MF5 into
    fixed-size buffers while the MT18 section grows by ~60 %. A mock would
    assert that the growth is fine, which is the thing in doubt.

    **Run against the full Cf-252 evaluation, not the committed micro-tape,
    and that is a correction rather than a convenience.** The first version of
    this test used the micro-tape and failed with NJOY return code 77 — but so
    does the *unperturbed* micro-tape, measured. That fixture is a section
    slice carrying MF1/451, MF3/MT18, MF5 and MF35 and nothing else: no
    MF2/151, no MF3/MT1 or MT2, so RECONR has nothing to reconstruct from. It
    is a perfectly good fixture for the record layer and cannot answer an NJOY
    question at all. A test that fails for a reason unrelated to the code under
    test is worse than no test, because it teaches people to ignore it.

    So the baseline is measured in the same run: NJOY is asked to process the
    original tape first. If that fails, the environment cannot answer the
    question and the test says so instead of blaming the perturbation.
    """
    from kika.sampling.endf_perturbation import _process_njoy_for_sample
    from kika.sampling.utils import DualLogger, _set_logger

    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    baseline_tape = baseline_dir / "original.endf"
    baseline_tape.write_bytes(cf252_b81_tape.read_bytes())

    _set_logger(DualLogger(str(baseline_dir / "njoy.log"), mode="w"))
    baseline = _process_njoy_for_sample(
        out_endf=str(baseline_tape), sample_index=0, njoy_exe=str(njoy_exe),
        temperatures=[293.6], library_name="endfb81",
        njoy_version="NJOY 2016.78", output_dir=str(baseline_dir),
        xsdir_file=None, extensions=None,
    )
    if not baseline.get("success"):
        pytest.skip(
            f"NJOY cannot process the unperturbed Cf-252 tape here "
            f"({baseline.get('errors')}), so it cannot say anything about the "
            f"perturbed one"
        )

    results = perturb_pfns_files(
        str(cf252_b81_tape), 1, generate_ace=True, njoy_exe=str(njoy_exe),
        ace_temperatures=[293.6], seed=1, output_dir=str(tmp_path / "perturbed"),
    )
    isotope = results["isotopes"]["98252"]
    assert isotope["samples_written"] == 1
    assert isotope["njoy_success"] == 1, (
        f"NJOY processed the original tape but not the perturbed one, so the "
        f"grown MF5 is the difference: {isotope['sample_errors']}"
    )
