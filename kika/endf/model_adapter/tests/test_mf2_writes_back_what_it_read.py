"""``MF2MT151.__str__`` reproduces the section it was parsed from, byte for byte.

**This is the prerequisite an MF2 encoder is gated against**, and it did not
hold when it was first measured. ``MF2MT151.__str__`` is the only MF2 writer
kika has, so any model→MF2 encoder ends in it: if it loses a field, the
encoder's byte-identity gate fails and the loss is attributed to the encoder
rather than to the writer underneath it. Hence a test that pins the writer on
its own, before the encoder exists.

**What it caught, 2026-08-07.** On Fe-57 JEFF-4.0 (LRU=1, LRF=7) exactly one
line of 163 differed: the particle-pair LIST header, whose L1 is **NPP** and was
written as a hardcoded ``0``. The counts either side of it — ``12*NPP`` and
``2*NPP`` — were correct, so the section stayed structurally valid and
internally consistent, and NPP is recoverable from ``2*NPP``. That is precisely
why nothing noticed: the parser derives NPP from the word count and never reads
the field the writer was not writing. A defect that only a comparison against
the *file* can see — the same blind spot P7b found for MF3 and MF4, where
comparing the model against the code it replaces made both sides agree on a
shared error.
"""
from __future__ import annotations

import pytest

from kika.endf.read_endf import read_endf

#: Every real evaluation the resolver knows. LRF=3 (Reich-Moore) and LRF=7 (RML)
#: are both represented; the Breit-Wigner path is not, because no tape to hand
#: uses it — see ``test_resonance_decode.py`` for the same caveat.
REAL_TAPES = ["fe56_host_tape", "fe57_host_tape", "fe56_jendl_tape",
              "u235_tape", "th232_tape", "pu241_tape"]


def _sourceLines(path):
    """The tape's own MF2 lines, SEND record included."""
    with open(path) as handle:
        return [line.rstrip("\n") for line in handle if line[70:72] == " 2"]


def _writtenLines(section):
    return [line for line in str(section).split("\n") if line.strip()]


def _assertByteIdentical(path, section, what):
    """Every record of the section, excluding the SEND that closes it.

    The SEND is excluded on purpose and tested separately below: it carries no
    data, and the tapes to hand disagree on how to spell it.
    """
    source, written = _sourceLines(path), _writtenLines(section)
    assert len(written) == len(source), (
        f"{what}: {len(source)} lines in the tape, {len(written)} written"
    )
    for index, (before, after) in enumerate(zip(source[:-1], written[:-1])):
        assert after == before, (
            f"{what}: MF2 line {index} differs\n  tape:    {before!r}\n"
            f"  written: {after!r}"
        )


@pytest.mark.parametrize("tape", REAL_TAPES)
def test_mf2_is_written_back_exactly_as_it_was_read(request, tape):
    path = request.getfixturevalue(tape)
    section = read_endf(str(path)).mf[2].mt[151]
    _assertByteIdentical(path, section, tape)


@pytest.mark.parametrize("tape", REAL_TAPES)
def test_the_section_is_closed_by_a_well_formed_send(request, tape):
    """The terminator is a SEND — MT 0, sequence 99999 — whatever its spelling."""
    path = request.getfixturevalue(tape)
    written = _writtenLines(read_endf(str(path)).mf[2].mt[151])[-1]
    assert written[70:72] == " 2" and written[72:75] == "  0"
    assert written[75:81] == "99999"[-6:].rjust(6) or written[75:].strip() == "99999"


@pytest.mark.parametrize("tape", REAL_TAPES)
def test_the_send_spelling_is_the_one_difference_left(request, tape):
    """Measured 2026-08-07, and recorded rather than normalised away.

    Three of the six tapes — Fe-56 JENDL-5, U-235 and Th-232 — write the SEND
    with **blank** data fields; Fe-56/Fe-57 JEFF-4.0 and Pu-241 write the
    ENDF-6-literal ``0.000000+0 0.000000+0 0 0 0 0``. Both are the same record
    and both parse identically; kika emits the literal form for every MF through
    the shared ``format_endf_send_record``.

    It is left alone deliberately. Preserving the source spelling would mean
    threading it through every section class in the library, and changing the
    shared helper would move the SEND of every MF3, MF33 and MF34 kika writes —
    including the ones the thesis pipeline splices into host tapes. A terminator
    that carries no data is not worth that blast radius. Pinned here so the
    difference is a known quantity rather than a surprise inside a future
    encoder's gate.
    """
    path = request.getfixturevalue(tape)
    source = _sourceLines(path)[-1]
    written = _writtenLines(read_endf(str(path)).mf[2].mt[151])[-1]

    blankInTape = not source[:66].strip()
    assert (written == source) != blankInTape, (
        f"{tape}: the SEND now agrees where it used to differ (or the reverse); "
        f"this test records which tapes spell it which way\n"
        f"  tape:    {source!r}\n  written: {written!r}"
    )


def test_an_rml_evaluation_keeps_its_particle_pair_count(fe57_host_tape):
    """The specific field the round trip lost, named so a regression is legible.

    Asserting the count is *written* rather than merely parsed: NPP can be
    derived from ``2*NPP`` in the same record, so a writer that emits zero there
    produces a file kika itself reads back correctly. Only the tape disagrees.
    """
    endf = read_endf(str(fe57_host_tape))
    ranges = endf.mf[2].mt[151].isotopes[0].energy_ranges
    rml = [r for r in ranges if getattr(r.parameters, "particle_pairs", None)]
    assert rml, "Fe-57 JEFF-4.0 is LRF=7; if this is empty the fixture changed"

    npp = len(rml[0].parameters.particle_pairs)
    assert npp > 0
    header = [line for line in _writtenLines(endf.mf[2].mt[151])
              if line[22:33].strip() == str(npp)
              and line[44:55].strip() == str(12 * npp)
              and line[55:66].strip() == str(2 * npp)]
    assert header, (
        f"no written LIST header carries NPP={npp} beside 12*NPP={12 * npp} "
        f"and 2*NPP={2 * npp}; the count is being written as something else"
    )


def test_a_reich_moore_evaluation_keeps_its_lad_flag(th232_tape):
    """The second field the same comparison caught, on a different formalism.

    LAD says whether MF4 angular distributions can be computed from the
    resonance parameters. It was neither parsed nor written — the CONT slot was
    a hardcoded 0 — so a tape declaring LAD=1 came back declaring the opposite.
    Th-232 JEFF-4.0 is the tape to hand that has it set; Fe-56 has LAD=0, which
    is why the defect survived every Fe-56 test in the suite.
    """
    endf = read_endf(str(th232_tape))
    ranges = endf.mf[2].mt[151].isotopes[0].energy_ranges
    resolved = [r for r in ranges if getattr(r.parameters, "nlsc", None) is not None]
    assert resolved, "Th-232 has no resolved range; the fixture changed"

    assert resolved[0].parameters.lad == 1, (
        "Th-232 JEFF-4.0 declares LAD=1; if this is 0 the field is being "
        "dropped on the way in again"
    )

