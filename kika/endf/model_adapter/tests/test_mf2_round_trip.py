"""``str(encodeMF2MT151(*decodeMF2MT151(s))) == str(s)``, on every real tape.

**This closes the last capability gap in the model.** Before it, ten decoders
faced five encoders and MF2/151 was the one hole that was not merely asymmetric
but *absent*: no path in kika could write resonance parameters from a
format-agnostic object at all, so the only way to change a resonance was to edit
ENDF records. MF1, MF33 and MF34 already had another way out, which is why
closing those was roadmap symmetry and closing this one is not.

**Why the equality is against ``str(s)`` and not against the tape.** Both sides
of this comparison are produced by ``MF2MT151.__str__``, so the SEND spelling —
the one place kika and three of the six tapes legitimately disagree, pinned in
``test_mf2_writes_back_what_it_read.py`` — is identical by construction and does
not have to be excluded. That makes this gate exact with no carve-outs. It
composes with the writer's own gate to give model → tape byte identity, and
``test_the_chain_reaches_the_tape_itself`` below asserts the composition rather
than leaving it to be inferred from two separate files.

**The gate is not the interesting half.** It passed on all six tapes the first
time it was run, which on its own is evidence of very little — a rebuild that
quietly returned its input would pass it too. What makes it mean something is
``test_an_edit_to_the_model_reaches_the_file``: physics is read from the model,
so changing a channel radius must change the written record. That is the
property an evaluation library actually needs, and byte identity alone is
perfectly compatible with not having it.
"""
from __future__ import annotations

import pytest

from kika.endf.model_adapter import decodeMF2MT151, encodeMF2MT151
from kika.endf.read_endf import read_endf

REAL_TAPES = ["fe56_host_tape", "fe57_host_tape", "fe56_jendl_tape",
              "u235_tape", "th232_tape", "pu241_tape"]


def _section(path):
    """MF2 only — a full parse of the Fe-56 tape costs minutes and buys nothing."""
    return read_endf(str(path), mf_numbers=[2]).mf[2].mt[151]


def _roundTrip(section):
    resonances, provenance, report = decodeMF2MT151(section)
    return encodeMF2MT151(resonances, provenance, report), resonances, provenance


def _lines(section):
    return [line for line in str(section).split("\n") if line.strip()]


@pytest.mark.parametrize("tape", REAL_TAPES)
def test_the_section_survives_the_model_byte_for_byte(request, tape):
    """Every record, SEND included. Reich-Moore, R-Matrix-Limited and URR case C."""
    section = _section(request.getfixturevalue(tape))
    written, _, _ = _roundTrip(section)

    before, after = _lines(section), _lines(written)
    assert len(after) == len(before), (
        f"{tape}: {len(before)} records in, {len(after)} out"
    )
    for index, (left, right) in enumerate(zip(before, after)):
        assert right == left, (
            f"{tape}: MF2 record {index} differs after the model\n"
            f"  read:    {left!r}\n  written: {right!r}"
        )


@pytest.mark.parametrize("tape", REAL_TAPES)
def test_the_chain_reaches_the_tape_itself(request, tape):
    """model → ``MF2MT151`` → **the file**, which is what the claim actually is.

    The SEND is excluded here and only here, for the reason
    ``test_mf2_writes_back_what_it_read.py`` records: three of these tapes spell
    it with blank data fields and kika emits the ENDF-6 literal form for every
    MF it writes. That difference belongs to the shared terminator helper, not
    to this encoder.
    """
    path = request.getfixturevalue(tape)
    written, _, _ = _roundTrip(_section(path))

    with open(path) as handle:
        source = [line.rstrip("\n") for line in handle if line[70:72] == " 2"]
    after = _lines(written)

    assert len(after) == len(source)
    for index, (left, right) in enumerate(zip(source[:-1], after[:-1])):
        assert right == left, (
            f"{tape}: MF2 line {index} of the tape is not reproduced\n"
            f"  tape:    {left!r}\n  written: {right!r}"
        )


@pytest.mark.parametrize("tape", REAL_TAPES)
def test_the_formalism_coverage_is_what_it_claims(request, tape):
    """Names what each tape exercises, so a fixture change cannot quietly narrow it.

    Between them these six cover LRF=3, LRF=7 and URR case C. **They cover
    nothing else**: Breit-Wigner (LRF=1/2), URR cases A and B, LRU=0, NRO=1,
    KBK/KPS and NIS>1 have no real evaluation here and are exercised by
    hand-built sections or not at all. The encoder handles all of them; only
    three are proven against a file.
    """
    section = _section(request.getfixturevalue(tape))
    seen = {(r.lru, r.lrf) for isotope in section.isotopes
            for r in isotope.energy_ranges}
    assert seen <= {(1, 3), (1, 7), (2, 2)}, (
        f"{tape} now carries {seen}, which includes a formalism no real tape "
        f"covered when this encoder was gated; re-read the coverage caveat"
    )


#: ``(tape, description, mutation, the record field it must move)``. Each entry
#: reads a value the *model* owns and requires it to reach the written record.
_EDITS = [
    ("fe56_host_tape", "a Reich-Moore per-l scattering radius",
     lambda r: setattr(r.resolved[0].formalism.spinGroups[1].channels[0],
                       "scatteringRadius", 0.4001)),
    ("fe56_host_tape", "a resonance energy",
     lambda r: r.resolved[0].formalism.spinGroups[0].energies.__setitem__(0, 1234.5)),
    ("fe56_host_tape", "a resonance width",
     lambda r: r.resolved[0].formalism.spinGroups[0].widths[0].__setitem__(1, 9.9)),
    ("fe57_host_tape", "an effective channel radius APE",
     lambda r: setattr(r.resolved[0].formalism.spinGroups[0].channels[1],
                       "hardSphereRadius", 0.71)),
    ("fe57_host_tape", "a channel boundary condition BND",
     lambda r: setattr(r.resolved[0].formalism.spinGroups[0].channels[1],
                       "boundaryConditionValue", -1.0)),
    ("fe57_host_tape", "the reduced-width-amplitude flag IFG",
     lambda r: setattr(r.resolved[0].formalism, "reducedWidthAmplitudes", True)),
    ("fe57_host_tape", "the relativistic-kinematics flag KRL",
     lambda r: setattr(r.resolved[0].formalism, "relativisticKinematics", True)),
    ("th232_tape", "an unresolved average width",
     lambda r: r.unresolved.tabulatedWidths.spinGroups[0]
                .channels[0].widths.__setitem__(0, 9.9)),
]


@pytest.mark.parametrize("tape,what,edit", _EDITS,
                         ids=[f"{what}" for _, what, _ in _EDITS])
def test_an_edit_to_the_model_reaches_the_file(request, tape, what, edit):
    """The property byte identity cannot demonstrate on its own.

    Every field here is one the model owns, so a value changed on the model has
    to appear in the written record. If any of these stops moving the output,
    the encoder has started preferring a copy in provenance — which is exactly
    the two-sources-of-truth failure the decoder was written to avoid, and it
    fails silently: the round trip still reproduces the file perfectly while
    ignoring the caller.
    """
    section = _section(request.getfixturevalue(tape))
    resonances, provenance, _ = decodeMF2MT151(section)
    before = str(encodeMF2MT151(resonances, provenance))

    edit(resonances)
    after = str(encodeMF2MT151(resonances, provenance))

    assert after != before, (
        f"{tape}: changing {what} on the model left the written section "
        f"unchanged, so the encoder is not reading it from the model"
    )


#: Fields the model has no node for, which must therefore come from provenance
#: and be written back untouched.
_BOOKKEEPING = [
    ("th232_tape", "LAD",
     lambda p: p.headerFields["regions"][0].__setitem__("lad", 0)),
    ("fe57_host_tape", "a particle-pair Q value",
     lambda p: p.headerFields["regions"][0]["particle_pairs"][2]
                .__setitem__("q", -14000.0)),
    ("th232_tape", "URR case C's interpolation code INT",
     lambda p: p.headerFields["regions"][1]["j_states"][0]
                .__setitem__("int_code", 5)),
]


@pytest.mark.parametrize("tape,what,edit", _BOOKKEEPING,
                         ids=[what for _, what, _ in _BOOKKEEPING])
def test_bookkeeping_is_written_back_from_provenance(request, tape, what, edit):
    """The other half of the rule: what the model cannot express still round-trips."""
    section = _section(request.getfixturevalue(tape))
    resonances, provenance, _ = decodeMF2MT151(section)
    before = str(encodeMF2MT151(resonances, provenance))

    edit(provenance)
    after = str(encodeMF2MT151(resonances, provenance))

    assert after != before, (
        f"{tape}: {what} is carried in provenance and changing it did not change "
        f"the written section, so it is being recomputed or ignored"
    )


#: Ways of arriving with something unwritable. Each must raise rather than emit
#: a structurally valid section with invented parameters — the hard rule phase 7
#: states and which applies to every encoder, not only the GNDS one.
_REFUSALS = [
    ("fe57_host_tape", "the particle-pair columns are gone",
     lambda r, p: p.headerFields["regions"][0].__setitem__("particle_pairs", None)),
    ("fe57_host_tape", "a channel names a reaction that does not exist",
     lambda r, p: setattr(r.resolved[0].formalism.spinGroups[0].channels[0],
                          "resonanceReaction", "MT999")),
    ("fe56_host_tape", "the model has fewer regions than provenance records",
     lambda r, p: r.resolved.clear()),
    ("th232_tape", "the unresolved region is missing",
     lambda r, p: setattr(r, "unresolved", None)),
    ("th232_tape", "provenance forgot which URR case it was",
     lambda r, p: p.headerFields["regions"][1].__setitem__("urr_case", None)),
    ("fe56_host_tape", "the provenance did not come from the decoder",
     lambda r, p: p.headerFields.clear()),
    ("fe56_host_tape", "an l-block record is missing",
     lambda r, p: p.headerFields["regions"][0].__setitem__(
         "l_blocks", p.headerFields["regions"][0]["l_blocks"][:1])),
]


@pytest.mark.parametrize("tape,what,break_", _REFUSALS,
                         ids=[what for _, what, _ in _REFUSALS])
def test_the_encoder_refuses_rather_than_inventing(request, tape, what, break_):
    section = _section(request.getfixturevalue(tape))
    resonances, provenance, _ = decodeMF2MT151(section)
    break_(resonances, provenance)

    with pytest.raises(ValueError):
        str(encodeMF2MT151(resonances, provenance))
