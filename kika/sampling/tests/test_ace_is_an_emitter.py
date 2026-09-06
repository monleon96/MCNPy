"""``"ace"`` is an emitter: NJOY on each realisation's tape, per temperature.

What is pinned without NJOY, on the committed micro-tape, with the NJOY runner
replaced by a stub: the options are validated up front (dry run included),
the ENDF tape the ACE needs is written on demand when it was not asked for,
one NJOY call is made per sample and temperature with the right arguments,
the results land under ``NNNN/ace/`` and ``NNNN/njoy/``, a failed call is an
``error`` event that does not stop the run, and the metadata says which
failed.

What is pinned **with** NJOY, when ``NJOY_EXE`` points at one and a full tape
is at hand (``KIKA_FULL_TAPE``): a real ACE comes out of a real perturbed tape.
That gate is opt-in because it takes minutes and needs files the repository
does not carry; it was run on 2026-09-06 on U-235 JEFF-4.0.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import importlib
from kika.sampling.model_perturbation import (EMITTERS, TAPE_EMITTERS,
                                              AceOptions, perturbFromModel)

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
TAPE = str(DATA / "micro_fe56_xs_and_angular.endf")
REQUEST = {33: None}


@pytest.fixture
def fakeNjoy(tmp_path, monkeypatch):
    """A stand-in for ``run_njoy`` that writes what NJOY would and records the call."""
    exe = tmp_path / "njoy.exe"
    exe.write_text("not really")
    calls = []

    def stub(*, njoy_exe, endf_path, temperature, library_name, output_dir,
             njoy_version, additional_suffix, extension, ace_dir, xsdir_dir,
             njoy_files_dir, **_ignored):
        calls.append(dict(njoy_exe=njoy_exe, endf_path=Path(endf_path),
                          temperature=temperature, library_name=library_name,
                          extension=extension, ace_dir=Path(ace_dir),
                          njoy_files_dir=Path(njoy_files_dir)))
        ace_dir, njoy_files_dir = Path(ace_dir), Path(njoy_files_dir)
        ace_dir.mkdir(parents=True, exist_ok=True)
        njoy_files_dir.mkdir(parents=True, exist_ok=True)
        if temperature > 9000:  # the temperature that "fails"
            listing = njoy_files_dir / f"fail_{temperature:g}.output"
            listing.write_text("***error*** provoked by the test")
            return {"returncode": 77, "ace_file": None, "xsdir_file": None,
                    "njoy_output": str(listing)}
        name = f"26056.{extension}" if extension else f"260560_80{temperature:.0f}"
        ace = ace_dir / name
        ace.write_text(f"ACE for {Path(endf_path).name} at {temperature} K\n")
        xsdir = ace_dir / f"{name}.xsdir"
        xsdir.write_text(f"26056.{extension or '00c'} 55.454 {name} 0 1 1 100 0 0 {temperature*8.617e-11:.3e}\n")
        listing = njoy_files_dir / f"{name}.output"
        listing.write_text("njoy ok")
        return {"returncode": 0, "ace_file": str(ace), "xsdir_file": str(xsdir),
                "njoy_output": str(listing), "njoy_listing": str(listing)}

    # `kika.njoy` re-exports the function under the submodule's own name, so
    # attribute access lands on the function; the module itself is in sys.modules.
    monkeypatch.setattr(importlib.import_module("kika.njoy.run_njoy"), "run_njoy", stub)
    return exe, calls


# ----------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------

def test_ace_is_among_the_emitters_and_the_tape_ones_are_not_it():
    assert "ace" in EMITTERS and "ace" not in TAPE_EMITTERS
    assert set(TAPE_EMITTERS) | {"ace"} == set(EMITTERS)


def test_ace_without_options_is_refused_before_anything_is_read():
    with pytest.raises(ValueError, match="no AceOptions"):
        perturbFromModel(TAPE, REQUEST, 1, formats=("ace",))


def test_the_options_are_validated_up_front(tmp_path, monkeypatch):
    monkeypatch.delenv("NJOY_EXE", raising=False)
    with pytest.raises(ValueError, match="at least one temperature"):
        AceOptions().validate()
    with pytest.raises(ValueError, match="no NJOY executable"):
        AceOptions(temperatures=[293.6]).validate()
    with pytest.raises(FileNotFoundError, match="not found"):
        AceOptions(temperatures=[293.6], njoyExe=str(tmp_path / "nope")).validate()
    exe = tmp_path / "njoy"
    exe.write_text("")
    with pytest.raises(ValueError, match="one per temperature"):
        AceOptions(temperatures=[293.6, 600.0], extensions=["02c"],
                   njoyExe=str(exe)).validate()
    monkeypatch.setenv("NJOY_EXE", str(exe))
    options = AceOptions(temperatures=293.6)
    options.validate()
    assert options.temperatures == (293.6,) and options.njoyExe == str(exe)


def test_a_dry_run_validates_the_options_and_runs_no_njoy(fakeNjoy, tmp_path):
    exe, calls = fakeNjoy
    options = AceOptions(temperatures=[293.6], njoyExe=str(exe), libraryName="jeff40")
    run = perturbFromModel(TAPE, REQUEST, 2, seed=1, outputDir=tmp_path,
                           formats=("ace",), ace=options, dryRun=True)
    assert calls == []
    assert run.aceOptions is options
    metadata = json.loads(run.files["metadata"].read_text("utf-8"))
    assert metadata["ace"]["options"]["temperatures"] == [293.6]
    with pytest.raises(FileNotFoundError):
        perturbFromModel(TAPE, REQUEST, 1, formats=("ace",), dryRun=True,
                         ace=AceOptions(temperatures=[293.6],
                                        njoyExe=str(tmp_path / "missing")))


# ----------------------------------------------------------------------
# The emitter
# ----------------------------------------------------------------------

def test_one_njoy_call_per_sample_and_temperature_on_the_samples_tape(fakeNjoy, tmp_path):
    exe, calls = fakeNjoy
    options = AceOptions(temperatures=[293.6, 600.0], extensions=["02c", "06c"],
                         njoyExe=str(exe), libraryName="jeff40")
    run = perturbFromModel(TAPE, REQUEST, 2, seed=1, outputDir=tmp_path,
                           formats=("endf-delta", "ace"), ace=options)
    assert len(calls) == 4
    for number in range(2):
        mine = [c for c in calls if c["ace_dir"] == tmp_path / f"{number:04d}" / "ace"]
        assert [c["temperature"] for c in mine] == [293.6, 600.0]
        assert [c["extension"] for c in mine] == ["02c", "06c"]
        assert all(c["endf_path"] == run.samples[number]["files"]["endf-delta"] for c in mine)
        assert all(c["njoy_files_dir"] == tmp_path / f"{number:04d}" / "njoy" for c in mine)
        assert all(c["library_name"] == "jeff40" and c["njoy_exe"] == str(exe) for c in mine)
        aces = run.samples[number]["files"]["ace"]
        assert [p.name for p in aces] == ["26056.02c", "26056.06c"]
        assert all(p.exists() for p in aces)
        assert (tmp_path / f"{number:04d}" / "ace" / "26056.02c.xsdir").exists()
    assert len(run.paths("ace")) == 4
    assert run.aceFailures() == []
    emitted = [e for e in run.log.of("emitted") if e.subject == "ace"]
    assert len(emitted) == 4 and all(e.payload["returncode"] == 0 for e in emitted)


def test_the_tape_the_ace_needs_is_written_on_demand(fakeNjoy, tmp_path):
    """``formats=("ace",)`` alone still has to feed NJOY an ENDF tape."""
    exe, calls = fakeNjoy
    options = AceOptions(temperatures=[293.6], njoyExe=str(exe))
    run = perturbFromModel(TAPE, REQUEST, 1, seed=1, outputDir=tmp_path,
                           formats=("ace",), ace=options)
    files = run.samples[0]["files"]
    assert "endf-delta" in files and files["endf-delta"].exists()
    assert calls[0]["endf_path"] == files["endf-delta"]


def test_a_gnds_source_cannot_make_an_ace_yet_and_says_why(fakeNjoy, tmp_path):
    """ACE needs an ENDF tape, and GNDS->ENDF is the deferred §6.3 increment.

    Measured 2026-09-06 on the fixture round-tripped through GNDS: with the
    MF1/451 header and the AWR borrowed from the ENDF original, the writer
    still refuses on MT1's missing Q/LR. Three things, not one, so the
    refusal names all three and points at the increment; the MAT it would
    use comes from the library's table, not from the user.
    """
    import kika

    exe, calls = fakeNjoy
    gnds = tmp_path / "src" / "fe56.gnds.xml"
    gnds.parent.mkdir()
    kika.write(kika.read(TAPE), str(gnds), format="gnds")
    options = AceOptions(temperatures=[293.6], njoyExe=str(exe))
    with pytest.raises(ValueError, match="§6.3") as refused:
        perturbFromModel(str(gnds), REQUEST, 1, seed=1, outputDir=tmp_path / "run",
                         formats=("ace",), ace=options)
    assert "MAT would be 2631" in str(refused.value)
    assert calls == []
    # A dry run of the same request goes through: the options are checked,
    # nothing is written, and the note records the MAT the table gives.
    run = perturbFromModel(str(gnds), REQUEST, 1, seed=1, outputDir=tmp_path / "dry",
                           formats=("ace",), ace=options, dryRun=True)
    assert any("MAT 2631" in e.message for e in run.log.of("note"))


def test_a_failed_njoy_run_is_an_error_event_and_the_run_goes_on(fakeNjoy, tmp_path):
    exe, calls = fakeNjoy
    options = AceOptions(temperatures=[293.6, 99999.0], njoyExe=str(exe))
    run = perturbFromModel(TAPE, REQUEST, 2, seed=1, outputDir=tmp_path,
                           formats=("ace",), ace=options)
    assert run.nSamples == 2 and len(calls) == 4
    assert run.aceFailures() == [(0, 99999.0, 77), (1, 99999.0, 77)]
    errors = run.log.of("error")
    assert len(errors) == 2
    assert all("return code 77" in e.message and e.subject == "ace" for e in errors)
    assert [e.sample for e in errors] == [0, 1]
    assert all(len(sample["files"]["ace"]) == 1 for sample in run.samples)
    metadata = json.loads(run.files["metadata"].read_text("utf-8"))
    assert [f["temperature"] for f in metadata["ace"]["failures"]] == [99999.0, 99999.0]
    assert run.log.verdict().startswith("RUN FAILED")
    assert "NJOY failed" in run.files["log-text"].read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# The real thing, opt-in
# ----------------------------------------------------------------------

@pytest.mark.skipif(not (os.environ.get("NJOY_EXE") and os.environ.get("KIKA_FULL_TAPE")),
                    reason="needs NJOY_EXE and KIKA_FULL_TAPE (a full evaluation tape)")
def test_a_real_ace_comes_out_of_a_real_perturbed_tape(tmp_path):
    from kika.endf import read_endf

    tape = os.environ["KIKA_FULL_TAPE"]
    options = AceOptions(temperatures=[293.6], extensions=["02c"],
                         libraryName=os.environ.get("KIKA_LIBRARY_NAME", "jeff40"))
    run = perturbFromModel(tape, {33: None}, 1, seed=1, outputDir=tmp_path,
                           formats=("endf-delta", "ace"), ace=options)
    assert run.aceFailures() == []
    ace = run.samples[0]["files"]["ace"][0]
    assert ace.exists() and ace.stat().st_size > 100_000
    zaid = read_endf(tape).zaid
    assert ace.name == f"{zaid}.02c"
    assert (ace.parent / f"{zaid}.02c.xsdir").read_text().split()[0].startswith(str(zaid))
