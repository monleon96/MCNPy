"""MF32: the read-write gate, and what the sub-formats decode to.

The gate is one sentence — parse a section and re-emit it, and the bytes come
back — and for MF32 it carries more weight than usual. §32 is the corner of
ENDF-6 that evaluations most often depart from, and three departures show up in
the eleven evaluations reachable here:

* **Th-232** writes NM into both trailing fields of its INTG control record
  where §32.2.3 draws the second as 0, and puts a 3 in the L1 field of its
  parameter LIST where §32.2.3.2 draws a 0.
* **Cl-35** declares NJS=8 spin groups against NJSX=7 in the covariance matrix,
  so a reader that loops over NJSX mistakes the last group's records for the
  INTG control record.
* **All three R-Matrix Limited tapes** contradict §32.2.3.3's stated parameter
  count: NNN is ``Σ (NCH+1)·NRSA``, the formula §32.2.2.4 gives, not
  ``Σ NCH·NRSA``. The manual sentence is wrong, and the tapes agree with each
  other.

None of those is repaired on the way through. That is the point of the gate: a
parser that "fixed" any of them would pass a numerical test and silently hand
back a different file. See ``docs/library/mf32-notes.md`` in kika-workspace for the
survey these came from.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika.endf import read_endf
from kika.endf.classes.mf32.mf32mt151 import IntgMatrix, LCOMP2Body, UnresolvedBody
from kika.endf.parsers.parse_mf32 import parse_mf32_mt151
from kika.endf.utils import format_intg, intg_row_length, parse_endf_id, parse_intg

# The tests directory has no ``__init__.py``, so pytest's prepend import mode
# puts it on ``sys.path``. Shared rather than copied: ``first_difference``
# exists so a failure on Ta-181's 240 131-line section reports one line.
from test_mf5_roundtrip import first_difference  # noqa: E402


def mf32_lines(path) -> list[str]:
    """The MF32/MT151 data lines of a tape, in order, without terminators."""
    out = []
    for line in Path(path).read_text().splitlines():
        if len(line) < 75:
            continue
        _, mf, mt = parse_endf_id(line)
        if mf == 32 and mt == 151:
            out.append(line.rstrip())
    return out


def source_width(lines: list[str]) -> int:
    """75 or 80, whichever the tape itself uses.

    ENDF/B-VIII.1 is distributed **without** the five-column sequence number;
    JENDL-5 and the IAEA copies carry all 80. Comparing at a fixed width would
    make every ENDF/B-VIII.1 tape fail on a field it does not have, and the MF5
    and MF35 gates hit the same thing before this one.
    """
    return 80 if any(len(line) > 75 for line in lines) else 75


def reemit(lines: list[str]) -> list[str]:
    """Parse an MF32 section and re-emit it, at the source's own width.

    ``str`` closes the section with a SEND record, which carries MT=0 and so is
    not among the collected lines; drop it rather than teach the collector to
    keep it.
    """
    width = source_width(lines)
    section = parse_mf32_mt151(lines)
    return [line[:width].rstrip() for line in str(section).split("\n")[:-1]]


def roundtrip(path) -> str:
    """Parse the MF32 section of *path* and report how its bytes differ."""
    source = mf32_lines(path)
    width = source_width(source)
    return first_difference([line[:width].rstrip() for line in source],
                            reemit(source))


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def test_every_mf32_sub_format_round_trips(micro_mf32_tape):
    """The four committed slices: LCOMP=0, LCOMP=2 with and without INTG, RML."""
    assert not roundtrip(micro_mf32_tape)


def test_the_gate_holds_through_the_public_reader(micro_mf32_tape):
    """``read_endf`` reaches the same section the parser does.

    Separate from the test above because registering MF32 in ``MF_PARSERS`` is
    its own failure mode: the parser can be correct and still be unreachable.
    """
    source = mf32_lines(micro_mf32_tape)
    width = source_width(source)
    endf = read_endf(str(micro_mf32_tape), mf_numbers=[32])
    section = endf.mf[32].mt[151]
    got = [line[:width].rstrip() for line in str(section).split("\n")[:-1]]
    assert not first_difference([line[:width].rstrip() for line in source], got)


@pytest.mark.parametrize(
    "tape",
    ["mn55_b81", "mn55_jendl", "ta181_b81", "pu239_jendl",
     "am241_b81", "na23_b81", "th232_b81",
     "cl35_b81", "cu63_b81", "w186_b81", "cm244_b81"],
)
def test_every_real_evaluation_round_trips(tape, request):
    """All eleven evaluations on the shared tree, micro-tape sources included.

    LCOMP=1 exists only here — its smallest evaluation is 26 467 lines, too
    large to commit — so without this test that sub-format has no real coverage
    at all. Ta-181 and Pu-239 also carry the only sections long enough to wrap
    the five-digit sequence number.
    """
    assert not roundtrip(request.getfixturevalue(f"{tape}_tape"))


def test_the_sequence_number_wraps_rather_than_widening(pu239_jendl_tape):
    """Pu-239's MF32 is 190 445 lines. NS is five characters wide.

    The record after 99999 is numbered 1, not 100000, and the tape says so:
    ``f"{100000:5d}"`` is six characters, so getting this wrong writes an
    81-column line and everything downstream that assumes 80 breaks. JENDL-5
    rather than Ta-181 because ENDF/B-VIII.1 is distributed without the
    sequence number at all, so only a JENDL tape can show the wrap.

    Asserted on the lines themselves as well as through the gate: the gate
    reports this as an ordinary byte difference, and it is worth naming.
    """
    lines = mf32_lines(pu239_jendl_tape)
    assert len(lines) > 99999, "Pu-239's MF32 got smaller; pick another tape"
    assert {len(line) for line in lines} == {80}
    assert lines[99998][75:80] == "99999"
    assert lines[99999][75:80] == "    1"

    reemitted = reemit(lines)
    assert reemitted[99998][75:80] == "99999"
    assert reemitted[99999][75:80] == "    1"


# ---------------------------------------------------------------------------
# INTG records, which are the only genuinely new record type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ndigit,nrow", [(2, 18), (3, 13), (4, 11), (5, 9), (6, 8)])
def test_intg_round_trips_at_every_width(ndigit, nrow):
    """§32.2.3 defines five INTG widths; only NDIGIT=2 occurs on any tape here.

    So the other four are pinned synthetically or not at all. Each line must
    land on exactly 66 data columns — NDIGIT=6 is the one with no blank after
    JJ, and reading it with one shifts every field by a column.
    """
    assert intg_row_length(ndigit) == nrow

    values = [(-1) ** n * (n + 1) for n in range(nrow)]
    entries = [(nrow + 2, 1, values)]
    lines, next_line = format_intg(entries, ndigit, 1234, 32, 151, 7)

    assert next_line == 8
    assert len(lines[0]) == 80
    assert lines[0][66:80] == f"{1234:4d}{32:2d}{151:3d}{7:5d}"

    back, idx = parse_intg([line[:66] for line in lines], 0, ndigit, 1)
    assert idx == 1
    assert back[0][0] == nrow + 2 and back[0][1] == 1
    assert back[0][2] == values


def test_intg_reads_a_blank_field_as_zero():
    """§32.2.3 lets a zero coefficient be written blank or as an explicit 0.

    Both must read back as 0. This is also why the round-trip path re-emits the
    stored text instead of re-formatting: the two spellings are not
    distinguishable once decoded, so re-formatting would change bytes on any
    tape that spells them the other way.
    """
    written, _ = format_intg([(4, 1, [7, 0, -3])], 2, 1234, 32, 151, 1)
    assert written[0][:20] == "    4    1   7    -3", repr(written[0])

    explicit = written[0][:14] + "  0" + written[0][17:]
    both = parse_intg([written[0][:66], explicit[:66]], 0, 2, 2)[0]
    assert both[0][2][:3] == [7, 0, -3]
    assert both[1][2][:3] == [7, 0, -3]


def test_an_impossible_intg_width_is_refused():
    with pytest.raises(ValueError, match="NDIGIT=7"):
        intg_row_length(7)


# ---------------------------------------------------------------------------
# What the compact format decodes to
# ---------------------------------------------------------------------------

def test_the_compact_correlation_matrix_is_a_correlation_matrix(th232_b81_tape):
    """Unpacked from Th-232's 1940 INTG records: symmetric, unit diagonal, in [-1, 1].

    The reverse mapping takes each stored integer to the *centre* of its range,
    so with NDIGIT=2 the largest magnitude an off-diagonal coefficient can
    reach is 0.995 — a stored 99. A matrix whose off-diagonal touched 1.0 would
    mean the mapping was applied at the edge instead.
    """
    section = parse_mf32_mt151(mf32_lines(th232_b81_tape))
    body = section.energy_ranges()[0].body
    assert isinstance(body, LCOMP2Body)

    matrix = body.correlations.correlation_matrix()
    assert matrix.shape == (body.correlations.nnn, body.correlations.nnn)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0)

    offDiagonal = matrix[~np.eye(len(matrix), dtype=bool)]
    assert offDiagonal.min() >= -1.0 and offDiagonal.max() <= 1.0
    assert np.abs(offDiagonal).max() <= 0.995 + 1e-12
    assert np.any(offDiagonal != 0.0), "every correlation decoded to zero"


def test_the_packed_integers_map_to_the_centre_of_their_range():
    """§32.2.3's worked example: NDIGIT=2, stored 87, means 0.875.

    A reader that took the integer at face value would give 0.87, and one that
    took the top of the range 0.88. Both are within the format's own tolerance
    of the true value, which is exactly why this needs an assertion rather than
    an eyeball.
    """
    matrix = IntgMatrix(
        ndigit=2, nnn=3, nm=1, control_raw="",
        lines=[format_intg([(3, 1, [87, -12])], 2, 0, 32, 151, 1)[0][0][:66]],
    ).correlation_matrix()

    assert matrix[2, 0] == pytest.approx(0.875)
    assert matrix[2, 1] == pytest.approx(-0.125)
    assert matrix[0, 1] == 0.0


def test_a_diagonal_compact_covariance_declares_an_empty_intg_block(micro_tape_na23):
    """Na-23 states its diagonal covariance rather than implying it.

    The control record is there — NDIGIT=3, NNN=69 — with **NM=0**, so not one
    INTG record follows. It is the empty case written down, not the block left
    out, and the two are not the same thing to read: a parser that treated NM
    as "at least one" would consume the SEND record as a correlation line.

    NNN=69 is 3 x 23, the three varied parameters of each of 23 resonances,
    which is also the check that the uncertainties were counted right.
    """
    section = parse_mf32_mt151(mf32_lines(micro_tape_na23))
    body = section.energy_ranges()[0].body
    assert isinstance(body, LCOMP2Body)
    assert body.nrsa == 23
    assert len(body.parameters.values) == 12 * 23

    assert body.correlations is not None
    assert (body.correlations.nm, body.correlations.nnn) == (0, 3 * 23)
    assert body.correlations.entries == []
    assert np.array_equal(body.correlations.correlation_matrix(), np.eye(69))


# ---------------------------------------------------------------------------
# The two structures no other tape carries
# ---------------------------------------------------------------------------

def test_the_unresolved_range_decodes_its_own_covariance(th232_b81_tape):
    """Th-232's second range is LRU=2 — the only §32.2.4 body reachable here.

    Its matrix is over ``NPAR = MPAR × Σ NJS`` average parameters and is given
    as an upper triangle, so the count is the check: a body that mis-read NLS
    or MPAR gets a triangle of the wrong size and this fails.
    """
    section = parse_mf32_mt151(mf32_lines(th232_b81_tape))
    ranges = section.energy_ranges()
    assert [r.lru for r in ranges] == [1, 2]

    urr = ranges[1].body
    assert isinstance(urr, UnresolvedBody)
    assert urr.nls == len(urr.l_blocks) == 3

    njsTotal = sum(block.n2 for block in urr.l_blocks)
    npar = urr.mpar * njsTotal
    assert urr.matrix.n2 == npar
    assert len(urr.matrix.values) == npar * (npar + 1) // 2


@pytest.mark.parametrize("tape", ["cl35_b81", "cu63_b81", "w186_b81"])
def test_rml_parameter_count_follows_the_channels_plus_energy(tape, request):
    """NNN is ``Σ (NCH+1)·NRSA``, not §32.2.3.3's ``Σ NCH·NRSA``.

    The extra parameter per resonance is its energy, which §32.2.2.4 counts and
    §32.2.3.3 forgets. All three R-Matrix Limited evaluations agree against the
    manual, which is what makes this a manual erratum rather than a tape quirk.
    """
    section = parse_mf32_mt151(mf32_lines(request.getfixturevalue(f"{tape}_tape")))
    body = section.energy_ranges()[0].body
    groups = body.spin_groups

    assert len(groups) == body.njs, "spin groups must be counted by NJS"
    withEnergy = sum((g.nch + 1) * g.nrsa for g in groups)
    withoutEnergy = sum(g.nch * g.nrsa for g in groups)
    assert body.correlations.nnn == withEnergy
    assert body.correlations.nnn != withoutEnergy


def test_a_tape_whose_njs_exceeds_its_njsx_still_reads(cl35_b81_tape):
    """Cl-35: NJS=8 spin groups, NJSX=7 of them in the covariance matrix.

    Named separately because it is the single reason the spin-group loop reads
    NJS off the control record instead of NJSX off the particle-pair record,
    and nothing else on this machine distinguishes the two.
    """
    section = parse_mf32_mt151(mf32_lines(cl35_b81_tape))
    body = section.energy_ranges()[0].body
    assert body.njs == 8
    assert body.particle_pairs.l2 == 7
    assert len(body.spin_groups) == 8


# ---------------------------------------------------------------------------
# The gate has to be able to fail
# ---------------------------------------------------------------------------

#: Structural fields of Th-232's MF32 and what mis-reading each would cost.
#: ``(line index, column slice, replacement, what it is)``. Deliberately *not*
#: data fields — see :func:`test_only_a_structural_field_can_break_the_gate`.
TH232_STRUCTURAL_MUTATIONS = [
    (0, slice(44, 55), 0, "NIS, the isotope count"),
    (1, slice(44, 55), 1, "NER, the number of energy ranges"),
    (4, slice(44, 55), 11118, "NPL, the length of the compact parameter list"),
    (1859, slice(44, 55), 1939, "NM, the number of INTG records"),
]


@pytest.mark.parametrize(
    "index,field,value,what",
    TH232_STRUCTURAL_MUTATIONS,
    ids=[m[3].split(",")[0] for m in TH232_STRUCTURAL_MUTATIONS],
)
def test_a_mis_read_count_breaks_the_gate(micro_tape_th232, index, field, value, what):
    """Corrupt one structural count and the reconstruction must stop matching.

    Insurance against the failure mode a byte-identity test is most prone to:
    passing because both sides came from the same parse. Each of these changes
    how many records the walk consumes, so a parser that ignored the field
    would emit a section of the wrong length here and be caught.
    """
    source = mf32_lines(micro_tape_th232)
    width = source_width(source)
    assert not first_difference([line[:width].rstrip() for line in source],
                                reemit(source)), "the intact tape must pass first"

    corrupted = list(source)
    line = corrupted[index]
    corrupted[index] = line[:field.start] + f"{value:11d}" + line[field.stop:]
    assert first_difference([line[:width].rstrip() for line in corrupted],
                            reemit(corrupted)), f"mis-reading {what} went unnoticed"


def test_only_a_structural_field_can_break_the_gate(micro_tape_th232):
    """Changing a *number* cannot fail the gate, and that is by design.

    The parser stores each record's 66 data columns as text and re-emits them,
    so an altered datum comes back altered — faithfully. Only the fields that
    say how many records to read can desynchronise the walk. Worth an assertion
    rather than a comment, because it is the precise limit of what the gate
    proves: it proves kika does not *reshape* a section, not that it understood
    every number in it. The decode tests above are what cover the numbers.
    """
    source = mf32_lines(micro_tape_th232)
    width = source_width(source)

    corrupted = list(source)
    # A resonance energy in the middle of the LCOMP=2 parameter block.
    corrupted[6] = " 9.999999+9" + corrupted[6][11:]
    assert corrupted != source

    assert not first_difference([line[:width].rstrip() for line in corrupted],
                                reemit(corrupted))


@pytest.fixture(scope="session")
def micro_tape_na23() -> Path:
    return Path(__file__).resolve().parent / "data" / "micro_na23_mf32.endf"


@pytest.fixture(scope="session")
def micro_tape_th232() -> Path:
    return Path(__file__).resolve().parent / "data" / "micro_th232_mf32.endf"
