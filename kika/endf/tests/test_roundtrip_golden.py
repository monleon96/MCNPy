"""Tier-1 golden: the ENDF read-write spine, byte for byte.

Phase 0 of the GNDS roadmap needs one test that answers "does kika give back
what it was given". Everything later — the canonical model, the layer
inversion, moving the calculations onto it — may change internals only if this
stays put.

Measured, not assumed. On the real JEFF-4.0 Fe-56 tape and on the committed
slice of it:

**Byte-identical.** MF3/MT1, MF3/MT2 and MF3/MT102 come back exactly as
written, all 15 127 lines. That is the gate, and it is asserted exactly.

**One formatting defect, three symptoms.** kika writes an ENDF floating-point
field as an integer when its value is zero. It shows up in the SEND record that
closes every section (``0.000000+0`` six times becomes ``0`` six times), and in
the STA field of the second MF1/451 record. Consequence: splicing an
*unmodified* section back into a tape changes the tape, which is the operation
``kika.sampling`` and the thesis scripts perform on every sample.

**Two content divergences.** The l-dependent scattering radius APL of the
MF2/151 L=1 block is dropped, and MF4/MT2 re-emits one AWR where the source
carried two slightly different ones.

**One spec deviation.** ``update_mf1_directory`` writes its own line number in
the sequence-number field of the MF1/451 SEND record, where ENDF-6 requires the
literal 99999.

Each of the last three groups is pinned by a test written the way it *should*
pass, marked ``xfail(strict=True)``. When the defect is fixed the test starts
passing, the strict marker turns that into a failure, and whoever fixed it is
forced to come here and delete the marker. Phase 0 records defects; it does not
fix them.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from kika.endf import read_endf
from kika.endf.utils import parse_endf_id
from kika.endf.writers import ENDFWriter, update_mf1_directory

#: Sections whose round-trip is byte-identical, on the slice and on the real
#: tape alike. Any regression here fails the gate.
CLEAN_SECTIONS = [(3, 1), (3, 2), (3, 102)]

#: Sections that diverge, each pinned by its own test below. The two lists
#: together must cover the whole fixture, so a new section cannot slip through
#: unexamined.
DIVERGENT_SECTIONS = [(1, 451), (2, 151), (4, 2), (34, 2)]

#: The PFNS fixture gets its **own** pair, not extra entries in the two above.
#: The lists are asserted to cover their fixture exhaustively, and the Fe-56
#: slice carries neither MF5 nor MF35, so folding them together fails at once.
#:
#: **Measured at 75 columns, and that is not a weaker gate here — it is the
#: only one available.** ENDF/B-VIII.1 writes 75-character records with no
#: sequence numbers at all, while kika always emits the 5-digit field. So on
#: this fixture columns 76-80 cannot match by construction, exactly as they
#: cannot for MF34 on the Fe-56 slice. Columns 1-75 carry every value in the
#: file. The 80-column gate lives on JEFF-4.0 U-235 in
#: ``test_mf5_roundtrip.py``, where it does hold.
PFNS_CLEAN_SECTIONS = [(1, 451), (3, 18), (5, 18), (5, 455), (35, 18)]

#: Empty on purpose, and meaningful: every section of the PFNS fixture comes
#: back byte-for-byte in columns 1-75, including the LF=5 MT455 laws that MF5
#: keeps verbatim and the four LB=7 covariance bands.
PFNS_DIVERGENT_SECTIONS: list[tuple[int, int]] = []


def data_lines(text: str, mf: int, mt: int) -> list[str]:
    """The data lines of one (MF, MT) section, SEND and FEND excluded."""
    return [
        line for line in text.splitlines()
        if len(line) >= 75 and parse_endf_id(line)[1:] == (mf, mt)
    ]


def rendered(endf, mf: int, mt: int) -> list[str]:
    """What kika writes back for one section, as data lines."""
    return data_lines(str(endf.files[mf].sections[mt]), mf, mt)


def first_difference(expected: list[str], got: list[str]) -> str:
    """A short, readable report of where two line lists diverge.

    Comparing multi-thousand-line lists (or 27 MB byte strings) with a bare
    ``assert a == b`` makes pytest build a diff of the whole thing, which on
    the real Fe-56 tape takes longer than the rest of the suite put together.
    Every comparison here reports through this instead.
    """
    if expected == got:
        return ""
    if len(expected) != len(got):
        return f"line count {len(expected)} -> {len(got)}"
    for i, (want, have) in enumerate(zip(expected, got)):
        if want != have:
            n_diff = sum(1 for a, b in zip(expected, got) if a != b)
            return (
                f"{n_diff} of {len(expected)} lines differ; first at index {i}\n"
                f"  source: {want!r}\n"
                f"  kika  : {have!r}"
            )
    return "lists differ but no differing line was found"


def digest(path: Path) -> str:
    """SHA-256 of a file, so large comparisons never build a diff."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_two_section_lists_cover_the_whole_fixture(micro_tape):
    """Neither list may quietly lose a section as the fixture evolves."""
    endf = read_endf(str(micro_tape))
    present = {
        (mf, mt) for mf, file_ in endf.files.items() for mt in file_.sections
    }
    assert present == set(CLEAN_SECTIONS) | set(DIVERGENT_SECTIONS)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mf,mt", CLEAN_SECTIONS)
def test_section_roundtrip_is_byte_identical(micro_tape, mf, mt):
    """parse -> str() reproduces the source bytes exactly."""
    endf = read_endf(str(micro_tape))
    source = data_lines(micro_tape.read_text(), mf, mt)
    assert not first_difference(source, rendered(endf, mf, mt))


def test_the_two_pfns_section_lists_cover_the_whole_fixture(micro_pfns_tape):
    """Same rule as the Fe-56 pair: no section escapes examination."""
    endf = read_endf(str(micro_pfns_tape))
    present = {
        (mf, mt) for mf, file_ in endf.files.items() for mt in file_.sections
    }
    assert present == set(PFNS_CLEAN_SECTIONS) | set(PFNS_DIVERGENT_SECTIONS)


@pytest.mark.parametrize("mf,mt", PFNS_CLEAN_SECTIONS)
def test_pfns_section_roundtrip_is_byte_identical_to_column_75(
    micro_pfns_tape, mf, mt,
):
    """parse -> str() reproduces the source values exactly. The MF5/MF35 gate."""
    endf = read_endf(str(micro_pfns_tape))
    source = [line[:75] for line in data_lines(micro_pfns_tape.read_text(), mf, mt)]
    got = [line[:75] for line in rendered(endf, mf, mt)]
    assert not first_difference(source, got)


def test_replace_pfns_mf5_section_with_itself_changes_only_the_send_record(
    micro_pfns_tape, tmp_path,
):
    """The splice the PFNS pipeline performs, fed an unmodified MF5/MT18.

    ``perturb_pfns_files`` parses MF5/MT18, rewrites its tables and splices the
    section back. If that path is not value-preserving when the section did not
    change, no perturbation it produces can be trusted to be the perturbation
    it computed. Asserted on columns 1-75 of the **whole file**, not just the
    spliced section: the surrounding sections are what would reveal a splice
    that cut in the wrong place.

    **One line moves, and it is the tape that deviates rather than kika.**
    ENDF/B-VIII.1 leaves all six data fields of a SEND record blank;
    ENDF-6 §0.6.3 types them as two floating-point zeros and four integer
    zeros, which is what ``format_endf_send_record`` writes. So a spliced
    ENDF/B-VIII.1 tape gains a conformant SEND where it had a blank one. This
    is a **dialect difference, not the SEND defect** recorded in the module
    docstring, which is about kika writing a bare ``0`` for a float field.

    Consequence to keep in view: every perturbed PFNS tape written from an
    ENDF/B-VIII.1 source differs from its parent in the SEND record of each
    touched section. NJOY reads both. The test asserts the difference is
    exactly that one line and nothing else — and asserts it is still there, so
    that teaching the writer this dialect forces a visit here.
    """
    work = tmp_path / micro_pfns_tape.name
    shutil.copy2(micro_pfns_tape, work)
    before = [line[:75] for line in work.read_text().splitlines()]

    endf = read_endf(str(work))
    writer = ENDFWriter(str(work))
    assert writer.replace_mt_section(
        endf.files[5].sections[18], 5, str(work), update_directory=False
    ) is True

    after = [line[:75] for line in work.read_text().splitlines()]
    assert len(after) == len(before), "the splice changed the line count"

    moved = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert len(moved) == 1, first_difference(before, after)

    was, now = before[moved[0]], after[moved[0]]
    assert parse_endf_id(was)[1:] == (5, 0), f"not the MF5 SEND record: {was!r}"
    assert was[:66].strip() == "", "source SEND was not blank after all"
    assert now[:66].split() == ["0.000000+0", "0.000000+0", "0", "0", "0", "0"]


@pytest.mark.slow
@pytest.mark.parametrize("mf,mt", CLEAN_SECTIONS)
def test_real_fe56_section_roundtrip_is_byte_identical(fe56_host_tape, mf, mt):
    """The micro-tape is a slice; prove the same holds on the whole file."""
    source = data_lines(Path(fe56_host_tape).read_text(), mf, mt)
    endf = read_endf(str(fe56_host_tape))
    assert not first_difference(source, rendered(endf, mf, mt))


def test_mf34_roundtrip_matches_except_the_sequence_number_field(micro_tape):
    """Pin a purely cosmetic divergence: content identical, line numbers dropped.

    Not a defect to fix — this records that the difference is confined to
    columns 76-80. If it ever spreads into the data columns, this fails.
    """
    source = data_lines(micro_tape.read_text(), 34, 2)
    got = rendered(read_endf(str(micro_tape)), 34, 2)

    assert len(got) == len(source)
    assert [line[:75] for line in got] == [line[:75] for line in source]
    assert got != source, "MF34 now reproduces the sequence numbers — update this test"


# ---------------------------------------------------------------------------
# Pinned defects
# ---------------------------------------------------------------------------

def test_replace_mt_section_with_itself_changes_nothing(micro_tape, tmp_path):
    """The splice path the pipeline actually uses must be a no-op when fed back.

    ``kika.sampling`` and the thesis scripts never rewrite a whole tape; they
    parse one MT, modify it, and splice it back. Feeding an *unmodified*
    section through that path changes one line today — the SEND record that
    closes MF3/MT2.
    """
    work = tmp_path / micro_tape.name
    shutil.copy2(micro_tape, work)
    before = digest(work)

    endf = read_endf(str(work))
    writer = ENDFWriter(str(work))
    assert writer.replace_mt_section(
        endf.files[3].sections[2], 3, str(work), update_directory=False
    ) is True

    assert digest(work) == before


def test_mf1_451_roundtrip_is_byte_identical(fe56_host_tape):
    """The STA field of record 2 comes back as an integer zero.

    Deliberately run against the **real** tape. The committed slice cannot
    answer this: ``remove_sections`` rebuilt its MF1/451 directory with kika's
    own writer, so the slice already speaks kika's dialect and would round-trip
    cleanly for circular reasons.
    """
    source = data_lines(Path(fe56_host_tape).read_text(), 1, 451)
    endf = read_endf(str(fe56_host_tape))
    assert not first_difference(source, rendered(endf, 1, 451))


def test_rebuilding_the_directory_of_a_real_tape_changes_nothing(
    fe56_host_tape, tmp_path,
):
    work = tmp_path / "fe56.endf"
    shutil.copy2(fe56_host_tape, work)
    before = digest(work)
    assert update_mf1_directory(str(work)) is True
    assert digest(work) == before


def test_mf2_keeps_the_l_dependent_scattering_radius(micro_tape):
    source = data_lines(micro_tape.read_text(), 2, 151)
    got = rendered(read_endf(str(micro_tape)), 2, 151)
    assert not first_difference(source, got)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Deferred to phase 3 by decision, 2026-08-04, after the phase-1 defect "
        "burn-down cleared the other eight pins. JEFF-4.0 Fe-56 carries two "
        "slightly different AWR values on consecutive MF4/MT2 records "
        "(5.545443+1 then 5.545440+1). kika parses one and re-emits it on both, "
        "so the rewritten tape is self-consistent but is not the source byte "
        "for byte. Preserving the source's own inconsistency means carrying AWR "
        "per record, which is a model change, not a formatting fix — it belongs "
        "with the GNDS hierarchy in phase 3, where AWR lives on the model "
        "rather than on each ENDF record. Left pinned so that phase 3 has to "
        "decide rather than drift."
    ),
)
def test_mf4_reproduces_every_awr_field_as_written(micro_tape):
    source = data_lines(micro_tape.read_text(), 4, 2)
    got = rendered(read_endf(str(micro_tape)), 4, 2)
    assert not first_difference(source, got)
