"""``NuclideInfo`` can write MF1/451 back, which it could not before.

``docs/library/library-gaps.md`` M2. Two of the four format-agnostic classes could be
read into and not written out, so the one way to change an evaluation's header
was to mutate the ``MF1MT451`` dataclass in place — bypassing the canonical
layer for exactly the case it exists for.

**The gate is equality against the file**, from the first commit, because there
is no older implementation to be measured against and no defect to inherit.
That is the criterion phase 3 arrived at the hard way: comparing a new path
against the code it replaces is blind whenever both are wrong the same way,
which is how the ZA truncation survived (D1).

**What made it possible was not the encoder.** ``MF1MT451.__str__`` already
reproduced the section; what was missing was that ``NuclideInfo`` threw away
the two things it needs. NWD descriptive records — up to 700 lines of an
evaluator's comment block, parsed into nothing — and the NXC directory were in
neither the fields nor ``metadata``. They are in ``metadata`` now, and
``test_metadata_contract.py``'s pinned key set says why.
"""
from __future__ import annotations

import pytest

from kika.endf.read_endf import read_endf
from kika.nuclear_data import NuclideInfo


@pytest.fixture(scope="module")
def section(micro_tape):
    return read_endf(str(micro_tape), mf_numbers=[1]).mf[1].mt[451]


def test_the_section_comes_back_byte_for_byte(section):
    original = str(section)
    rebuilt = str(NuclideInfo.from_endf(section).to_endf())

    assert rebuilt == original, (
        "MF1/451 did not survive the round trip; first difference at line "
        + str(next(
            (i for i, (a, b) in enumerate(
                zip(original.split("\n"), rebuilt.split("\n")), start=1) if a != b),
            "(only the length differs)",
        ))
    )


@pytest.mark.parametrize(
    "tape",
    ["fe56_host_tape", "fe57_host_tape", "fe56_jendl_tape",
     "u235_tape", "th232_tape", "pu241_tape"],
)
def test_real_evaluations_survive_too(request, tape):
    """Under ``--deep``. Six evaluations from three libraries.

    The committed slice has its directory rebuilt and its text is one lab's
    house style; these are 165 to 548 directory entries and 291 to 701 text
    records of whatever six evaluators actually wrote.
    """
    path = request.getfixturevalue(tape)
    mt451 = read_endf(str(path), mf_numbers=[1]).mf[1].mt[451]

    assert str(NuclideInfo.from_endf(mt451).to_endf()) == str(mt451)


def test_the_free_text_is_what_makes_it_possible(section):
    """The block that is parsed into nothing else, and is most of the section."""
    info = NuclideInfo.from_endf(section)

    assert len(info.metadata["text"]) == section._nwd
    assert len(info.metadata["directory"]) == section._nxc
    # `evaluation_info` parses seven fields out of the first two records. The
    # other ~600 are free text and have no other home.
    assert section._nwd > len(info.evaluation_info)


def test_nwd_and_nxc_are_counted_rather_than_stored(section):
    """A stored count that can disagree with what it counts is a defect waiting.

    Dropping a text record must change NWD in the written section, not leave it
    declaring a number the body no longer has — which is the shape of the MF34
    defect fixed in 98d7d23, where an under-declared count silently truncated a
    section on the way back in.
    """
    info = NuclideInfo.from_endf(section)
    info.metadata["text"] = info.metadata["text"][:-3]

    written = info.to_endf()
    assert written._nwd == section._nwd - 3
    assert len(written._text_lines) == 4 + written._nwd


def test_an_edited_field_reaches_the_file(section):
    """The flat fields are the source of truth, not a stashed model.

    If ``to_endf`` re-encoded a model kept from ``from_endf``, this edit would
    be silently dropped — the failure mode D2 records for ``AngularDistribution``
    and the reason that method builds from the fields too.
    """
    info = NuclideInfo.from_endf(section)
    info.temperature = 293.6

    assert info.to_endf()._temp == 293.6
    assert str(info.to_endf()) != str(section), "the edit did not reach the file"


def test_the_directory_is_written_back_as_read(section):
    """NC is a line count, so a directory is only true of one tape.

    This writes back the one it read; ``update_mf1_directory`` is what makes it
    true again after a section changes length, by scanning the written file.
    Recomputing here would mean guessing at lengths this object cannot see.
    """
    info = NuclideInfo.from_endf(section)

    assert info.to_endf()._directory == [tuple(e) for e in section._directory]


def test_an_ace_sourced_object_refuses_rather_than_inventing_a_header(fe56_ace):
    """ACE records four header fields out of nineteen and no descriptive text.

    Same refusal, and the same reason, as ``CrossSection.to_endf`` on a missing
    Q: a section that would be mostly invented is not written at all.
    """
    from kika.ace import read_ace

    info = NuclideInfo.from_ace(read_ace(str(fe56_ace)))

    with pytest.raises(ValueError) as excinfo:
        info.to_endf()
    message = str(excinfo.value)
    assert "ace" in message
    assert "lrp" in message, "the message should name what is missing"


def test_an_explicit_mat_beats_metadata(section):
    info = NuclideInfo.from_endf(section)

    assert info.to_endf(mat=9999)._mat == 9999
    assert info.to_endf()._mat == info.metadata["mat"]
