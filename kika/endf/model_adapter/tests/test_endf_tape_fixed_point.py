"""``read → suite → write → read`` returns the same model. The §2.8 gate.

**This is the gate the whole-file writer was built against, and it is not byte
identity** (decided 2026-08-13, owner Juan; ``docs/gnds_endf_conflicts.md``
§2.8). Byte identity against the source tape would fail on choices that carry no
information — where a field is padded, whether ``1e-5`` is written ``1.0-5``,
how many LB=5 records one covariance block is split into — and chasing those is
a different job from being correct. What the fixed point catches is the thing
that matters: **a quantity that does not survive the trip.**

Its blind spot is stated so nobody has to find it: anything the model does not
carry is equally absent from both sides, so the comparison passes. MF5 and MF6
are exactly that today. The fixed point is necessary and not sufficient, and the
:class:`ConversionReport` is the other half — which is why
:func:`test_the_report_names_every_file_that_did_not_survive` is here and is not
decoration.

**Why the comparison is a generic walk rather than ``==``.** The model's nodes
are dataclasses holding numpy arrays, and ``==`` on those returns an array. The
walk is the same one ``test_generic_round_trip`` uses for the same reason, and
it is deliberately naive: turning each node into ``{field: value}`` compares
*everything the node carries*, including the fields a hand-written assertion
would forget.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from kika.endf.model_adapter import decodeCovarianceSuite, decodeReactionSuite
from kika.endf.read_endf import read_endf
from kika.endf.writers.assemble import (DEFAULT_TAPE_ID, MF_WRITE_ORDER,
                                        assembleTape, encodeTapeSections,
                                        writeEndfTape)


def _walk(node):
    """A model tree → nested plain data, arrays included. Compares by value."""
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        return {"__class__": type(node).__name__,
                **{f.name: _walk(getattr(node, f.name))
                   for f in dataclasses.fields(node)}}
    if isinstance(node, np.ndarray):
        return ["__array__"] + np.asarray(node, dtype=float).ravel().tolist()
    if isinstance(node, (list, tuple)):
        return [_walk(v) for v in node]
    if isinstance(node, dict):
        return {k: _walk(v) for k, v in sorted(node.items(), key=lambda kv: str(kv[0]))}
    if not isinstance(node, (str, bytes)) and hasattr(node, "__iter__"):
        # `Reactions`, `Products`, `Sums` and friends are plain iterable
        # containers, not dataclasses -- so the branch above walks straight past
        # them and the comparison degenerates to `container is container`, which
        # is True for two different objects only by accident. This is the branch
        # that makes the walk see three reactions instead of one repr.
        return ["__iterable__", type(node).__name__] + [_walk(v) for v in node]
    if isinstance(node, float):
        # ENDF's fixed-format floats do not round-trip bit for bit -- eleven
        # significant figures go out through six. Rounding here is what makes
        # this a fixed point of the *model* rather than of the text.
        return round(node, 8)
    return node


def _decode(path):
    """A tape → its reactionSuite with the covarianceSuite hung on it."""
    endf = read_endf(str(path))
    suite, report = decodeReactionSuite(endf)
    covariances, report = decodeCovarianceSuite(endf, report)
    suite.covarianceSuite = covariances
    return suite, report


@pytest.fixture(scope="module")
def roundTripped(micro_tape, tmp_path_factory):
    """The pair the whole file is about: the suite, and the suite written and read back."""
    suite, report = _decode(micro_tape)
    out = tmp_path_factory.mktemp("assemble") / "written.endf"
    report = writeEndfTape(suite, out, report=report)
    return suite, _decode(out)[0], report, out


def test_the_reaction_suite_is_a_fixed_point(roundTripped):
    """Every reaction, cross section, Q, resonance and angular form survives."""
    before, after, _report, _out = roundTripped
    assert _walk(before.reactions) == _walk(after.reactions)
    assert _walk(before.resonances) == _walk(after.resonances)
    assert _walk(before.PoPs) == _walk(after.PoPs)


def test_the_header_is_a_fixed_point(roundTripped):
    """MF1/451 -- the nineteen header fields and the descriptive block.

    The **directory** is deliberately excluded: its ``NC`` entries are line
    counts, and a tape whose MF34 came back with a different record split has
    different ones. That is the writer being honest, not a loss, and
    ``update_mf1_directory`` rebuilt them from the file it wrote.
    """
    before, after, _report, _out = roundTripped
    for name in ("headerFields", "evaluationInfo", "descriptiveText",
                 "mat", "za", "awr"):
        assert _walk(getattr(before.provenance, name)) == \
               _walk(getattr(after.provenance, name)), name


def test_the_covariances_are_a_fixed_point_in_their_numbers(roundTripped):
    """The matrices and grids come back; the per-record split need not.

    ``encodeMF34MT`` writes through ``kika/cov``, which collapses NI>1
    sub-subsections onto one LB=5/LB=6 record on the stored grid -- it says so
    in the report. So this asserts on what the covariance *is*, section by
    section, and not on how many records stated it.
    """
    before, after, _report, _out = roundTripped
    assert before.covarianceSuite is not None

    def byKey(suite):
        out = {}
        for section in suite.covarianceSections:
            row = section.rowData
            out[(row.ENDF_MF, row.ENDF_MT, section.label)] = section.form
        return out

    first, second = byKey(before.covarianceSuite), byKey(after.covarianceSuite)
    assert set(first) == set(second)
    for key, form in first.items():
        assert np.allclose(np.asarray(form.matrix, dtype=float),
                           np.asarray(second[key].matrix, dtype=float),
                           rtol=1e-6, atol=0), key
        assert np.allclose(np.asarray(form.rowGrid, dtype=float),
                           np.asarray(second[key].rowGrid, dtype=float),
                           rtol=1e-6, atol=0), key
        assert form.isRelative == second[key].isRelative, key


def test_the_written_tape_reads_back_with_the_same_sections(roundTripped):
    """No file gained, no file lost, no section quietly dropped."""
    _before, _after, _report, out = roundTripped
    written = read_endf(str(out))
    assert sorted(written.mf) == [1, 2, 3, 4, 34]
    assert sorted(written.mf[3].mt) == [1, 2, 102]


def test_the_report_names_every_file_that_did_not_survive(roundTripped):
    """The half the fixed point cannot see, and the reason it is load-bearing."""
    _before, _after, report, _out = roundTripped
    said = "\n".join(report.losses + report.approximations +
                     report.unsupported + report.warnings)

    # The tape label is not in the model at all -- `read_endf` drops the first
    # line -- so the writer says it invented one rather than letting the file
    # imply otherwise.
    assert "tape identification record" in said

    # And the one thing that genuinely did not survive the trip: MF34 came back
    # through `kika/cov`, which states the same numbers in a different record
    # split. The fixed point above passes *because* it compares the covariance
    # and not its layout, so the layout change has to be said out loud here or
    # it is said nowhere.
    assert "MF34/MT2" in said and "collapsed" in said
    assert not report.isClean, "a report that claims a clean trip is the failure mode"


def test_a_suite_with_no_mat_is_refused_rather_than_stamped_with_a_guess(micro_tape):
    """MAT is on every record of the file, and inventing one is not a default."""
    suite, _report = _decode(micro_tape)
    suite.provenance.mat = None
    with pytest.raises(ValueError, match="no MAT number"):
        encodeTapeSections(suite)


def test_the_assembly_puts_a_fend_after_every_file_and_ends_the_tape(micro_tape):
    """The bookkeeping no section can do for itself, because none knows what follows."""
    suite, _report = _decode(micro_tape)
    sections, _report, mat = encodeTapeSections(suite)
    lines = assembleTape(sections, mat).rstrip("\n").split("\n")

    assert lines[0].endswith("   1 0  0    0"), "TPID"
    assert lines[0].startswith(DEFAULT_TAPE_ID)
    assert lines[-1].strip().endswith("-1 0  0    0"), "TEND"
    assert lines[-2].strip().endswith("0 0  0    0"), "MEND"

    files = sorted({mf for mf, _mt, _s in sections})
    assert files == [mf for mf in MF_WRITE_ORDER if mf in files], "ascending MF"
    fends = [line for line in lines if line[70:75] == " 0  0" and line[66:70].strip()
             not in ("0", "-1", "1")]
    assert len(fends) == len(files), "one FEND per file written"


# ----------------------------------------------------------------------
# The same gate on the other committed tapes, because one tape covers one
# shape. The structural Fe-56 above is MF1/2/3/4/34; these bring the nu-bars
# and MF31, a real MF5 the model does not carry, and MF32.
# ----------------------------------------------------------------------

def _fixedPoint(path, tmp_path):
    suite, report = _decode(path)
    out = tmp_path / "written.endf"
    report = writeEndfTape(suite, out, report=report)
    return suite, _decode(out)[0], report


def test_the_nubars_and_mf31_survive(micro_nubar_tape, tmp_path):
    """MF1/452+455+456 and MF31, which no other fixture carries.

    The nu-bars are the case where a section's *placement* in the model is not
    where ENDF puts it — §17.3 hangs the multiplicity off the fission channel's
    neutron, ENDF states it in File 1 — so a round trip through the model tests
    ``nubarNode`` agreeing with ``attachNubar``, and not just the record layout.
    """
    before, after, report = _fixedPoint(micro_nubar_tape, tmp_path)
    assert _walk(before.reactions) == _walk(after.reactions)

    sections, _report, _mat = encodeTapeSections(before)
    assert [(mf, mt) for mf, mt, _ in sections if mf == 1] == \
        [(1, 451), (1, 452), (1, 455), (1, 456)]
    assert [(mf, mt) for mf, mt, _ in sections if mf == 31] == \
        [(31, 452), (31, 455), (31, 456)]


def test_a_tape_with_mf5_comes_back_without_it_and_says_so(micro_pfns_tape, tmp_path):
    """The blind spot, pinned. This is the test that keeps the report honest.

    Cf-252's MF5/MT18 and MF5/MT455 are parsed by kika and decoded by nothing,
    so the written tape has no MF5 and the fixed point above **passes anyway** —
    the distributions are absent from both sides. If the report ever stops
    saying so, this fails and the fixed point does not.
    """
    before, after, report = _fixedPoint(micro_pfns_tape, tmp_path)
    assert _walk(before.reactions) == _walk(after.reactions)

    said = "\n".join(report.losses + report.unsupported)
    assert "MF5 (energy distributions) is present and parsed by kika" in said

    sections, _report, _mat = encodeTapeSections(before)
    assert 5 not in {mf for mf, _mt, _s in sections}
    assert (35, 18) in {(mf, mt) for mf, mt, _s in sections}, "MF35 does survive"


def test_a_tape_with_mf32_comes_back_without_it_and_says_so(micro_mf32_tape, tmp_path):
    """§25.3 parameter covariances decode and have no ENDF encoder.

    They are not ``covarianceSections``, so the MF31/33/34/35 loop never sees
    them and said nothing at all until this was written. Same shape as the MF5
    case and a different container.
    """
    before, after, report = _fixedPoint(micro_mf32_tape, tmp_path)
    assert _walk(before.resonances) == _walk(after.resonances)

    said = "\n".join(report.unsupported)
    assert "parameter covariance section(s)" in said
    sections, _report, _mat = encodeTapeSections(before)
    assert 32 not in {mf for mf, _mt, _s in sections}
    assert (2, 151) in {(mf, mt) for mf, mt, _s in sections}, "MF2 does survive"
