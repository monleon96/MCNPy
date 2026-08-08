"""MF1/451 through the model and back, byte for byte.

The same gate as ``test_endf_round_trip.py``, against the **file** rather than
against the flat path:

    str(encodeMF1MT451(*decodeMF1MT451(s)))  ==  str(s)

**Why this gate is available from the first commit.** ``NuclideInfo.to_endf``
(``docs/library-gaps.md`` M2) already passes exactly this comparison on six real
evaluations, so the model-side encoder has a working reference and a gate that
was proved reachable before it was written. What M2 did *not* close is the
model→MF1 direction: the façade builds the section from its flat fields by
design, so P6's "MF1 decode + encode" row was an over-claim for a day. This
closes it.

**What makes it work at all** is that the decoder puts the whole 451 header into
``EndfProvenance`` verbatim — including the NWD comment block, which is 291 to
701 lines across the tapes to hand and is parsed into *nothing* else. That was
M2's real finding: the write path was never the hard part, what the read path
discarded was.
"""
from __future__ import annotations

import pytest

from kika.endf.model_adapter import (decodeMF1MT451, decodeReactionSuite,
                                     encodeMF1MT451)
from kika.endf.read_endf import read_endf
from kika.nuclear_data import NuclideInfo

#: Every real evaluation the tape resolver knows, so the gate is not measured on
#: one library's house style. Th-232 and Pu-241 are here for the same reason
#: they are in ``test_endf_round_trip.py``: their ZA does not parse exactly.
REAL_TAPES = ["fe56_host_tape", "fe57_host_tape", "fe56_jendl_tape",
              "u235_tape", "th232_tape", "pu241_tape"]


def _roundTrip(section):
    """``str()`` of the section rebuilt from what the model decoded."""
    _pops, _style, provenance, _report = decodeMF1MT451(section)
    rebuilt, _ = encodeMF1MT451(provenance)
    return str(rebuilt)


@pytest.fixture(scope="module")
def microSection(micro_tape):
    return read_endf(str(micro_tape)).mf[1].mt[451]


def test_the_fixture_has_a_header_to_compare(microSection):
    """A 451 with no descriptive text would make the comparison vacuous."""
    assert microSection._nwd and microSection._nwd > 2
    assert microSection._nxc and microSection._nxc > 0


def test_the_header_encodes_byte_identically_to_the_file(microSection):
    assert _roundTrip(microSection) == str(microSection)


@pytest.mark.parametrize("tape", REAL_TAPES)
def test_the_same_holds_on_real_tapes(request, tape):
    """Under ``--deep``, on the real evaluations, not just the committed slice."""
    path = request.getfixturevalue(tape)
    section = read_endf(str(path)).mf[1].mt[451]
    assert _roundTrip(section) == str(section), f"{tape} MF1/451 differs from the file"


def test_the_model_and_the_flat_path_agree(microSection):
    """The façade and the model encoder must not become two sources of truth.

    This is the assertion the ZA truncation would have needed: two paths that
    build the same section from the same input have to agree, or one of them is
    quietly wrong and the disagreement only shows up on someone else's tape.
    """
    viaFlat = NuclideInfo.from_endf(microSection).to_endf()
    assert _roundTrip(microSection) == str(viaFlat)


def test_a_suite_can_be_encoded_directly(micro_tape):
    """The encoder takes the suite, not only the provenance — otherwise every
    caller has to know where ``decodeReactionSuite`` stashed the header."""
    endf = read_endf(str(micro_tape))
    suite, _ = decodeReactionSuite(endf)
    rebuilt, _ = encodeMF1MT451(suite)
    assert str(rebuilt) == str(endf.mf[1].mt[451])


def test_an_ace_sourced_header_refuses_rather_than_inventing_one(fe56_ace):
    """ACE records four header fields out of nineteen and no descriptive text.

    Writing the section anyway would mean inventing ``NLIB``, ``NSUB``, ``LREL``
    and the evaluator's comment block. Same refusal, and the same reasoning, as
    ``encodeMF3MT`` on a missing Q and ``CrossSection.to_endf`` on a missing QM.
    """
    from kika.ace import read_ace
    from kika.ace.model_adapter import decodeAce

    suite, _ = decodeAce(read_ace(str(fe56_ace)))
    with pytest.raises(ValueError, match="MF1/451"):
        encodeMF1MT451(suite)


def test_the_directory_is_written_back_as_read(microSection):
    """``NC`` is a line count, so the directory is only true of the tape it came
    from. Writing back what was read is right for a round trip; after a section
    changes length ``update_mf1_directory`` rebuilds it from the written file,
    which is the only place the true counts exist. A recomputation here would be
    a second source of truth for a number this object cannot see."""
    _pops, _style, provenance, _ = decodeMF1MT451(microSection)
    rebuilt, _ = encodeMF1MT451(provenance)

    assert rebuilt._directory == [tuple(e) for e in microSection._directory]
    assert rebuilt._nxc == microSection._nxc
