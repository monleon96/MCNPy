"""The real phase 3c gate: the model reproduces the flat path, byte for byte.

**Why not "ENDF → model → ENDF byte-identical".** That is what the roadmap
originally asked for, and it measures almost nothing. The writer is
patch-in-place (``kika/endf/writers/endf_writer.py:162-166``): whatever it is
not handed survives verbatim, so the assertion stays true even if the model
computes garbage, as long as the garbage is never given to the writer. It is
also already what the tier-1 golden asserts.

The gate here is narrower and much stronger — per MF/MT section,

    str(encodeMF3MT(decoded_reaction))  ==  str(s)

**It compared against the flat path until P7b, and that was too weak.** The
original formulation was ``== str(CrossSection.from_endf(s).to_endf())``: what
the model produced against what the code it replaces produces. That catches a
model that drops a field, but it cannot catch a model that inherits a *defect*
from the code it is copying — and one was there. ENDF writes ZA as a
fixed-format float, kika's reader rebuilds it as ``mantissa * 10**exponent``,
and Th-232's ``9.023200+4`` comes back as ``90231.99999999999``; both paths did
``int()`` on it and agreed on 90231. Comparing against the file instead makes
all 57 Th-232 sections and all 53 Pu-241 sections fail, which is the truth. The
flat path's version of that loss is pinned as an xfail below.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.endf.model_adapter import decodeMF3MT, decodeReactionSuite, encodeMF3MT
from kika.endf.read_endf import read_endf
from kika.nuclear_data import CrossSection


@pytest.fixture(scope="module")
def decoded(micro_tape):
    endf = read_endf(str(micro_tape))
    suite, report = decodeReactionSuite(endf)
    return endf, suite, report


def test_the_fixture_has_sections_to_compare(decoded):
    """A tape with no MF3 would make every comparison below vacuous."""
    endf, suite, _ = decoded
    assert len(endf.mf[3].mt) >= 3
    assert len(suite.reactions) == len(endf.mf[3].mt)


def test_every_mf3_section_encodes_byte_identically_to_the_file(decoded):
    endf, suite, _ = decoded

    for mt in sorted(endf.mf[3].mt):
        section = endf.mf[3].mt[mt]
        viaModel, _ = encodeMF3MT(suite.reactionByENDF_MT(mt))
        assert str(viaModel) == str(section), f"MT{mt} differs from the file"


def test_the_model_and_the_flat_path_still_agree_where_the_flat_path_is_right(decoded):
    """On Fe-56 the two coincide, which is what makes the swap of gate safe."""
    endf, suite, _ = decoded

    for mt in sorted(endf.mf[3].mt):
        section = endf.mf[3].mt[mt]
        viaModel, _ = encodeMF3MT(suite.reactionByENDF_MT(mt))
        viaFlat = CrossSection.from_endf(section).to_endf()
        assert str(viaModel) == str(viaFlat), f"MT{mt} differs from the flat path"


@pytest.mark.parametrize(
    "tape", ["fe56_host_tape", "fe56_jendl_tape", "u235_tape", "th232_tape"]
)
def test_the_same_holds_on_real_tapes(request, tape):
    """Under ``--deep``, on the real evaluations, not just the committed slice."""
    path = request.getfixturevalue(tape)
    endf = read_endf(str(path))
    suite, _ = decodeReactionSuite(endf)

    for mt in sorted(endf.mf[3].mt):
        section = endf.mf[3].mt[mt]
        viaModel, _ = encodeMF3MT(suite.reactionByENDF_MT(mt))
        assert str(viaModel) == str(section), f"{tape} MT{mt} differs from the file"


def test_the_flat_path_now_round_trips_a_tape_whose_za_does_not_parse_exactly(th232_tape):
    """This was a strict xfail until phase 3d, and the XPASS is why it changed.

    ``CrossSection.nuclide_id`` truncated ZA, so all 57 Th-232 MF3 sections came
    back naming ZA 90231 — Ac-231. The façade rounds. See
    ``docs/library-gaps.md`` D1.
    """
    endf = read_endf(str(th232_tape))
    assert int(float(endf.mf[3].mt[2].zaid)) != round(float(endf.mf[3].mt[2].zaid)), (
        "Th-232's ZA now parses exactly, so this test no longer measures anything"
    )
    for mt in sorted(endf.mf[3].mt):
        section = endf.mf[3].mt[mt]
        assert str(CrossSection.from_endf(section).to_endf()) == str(section)


# ---------------------------------------------------------------------------
# Provenance keeps what the metadata dict kept
# ---------------------------------------------------------------------------

def test_provenance_carries_every_key_the_metadata_contract_pins(decoded):
    """``kika/nuclear_data/tests/test_metadata_contract.py`` froze the ENDF key
    set. ``Provenance`` replaces that dict, and the replacement is only lossless
    if every key still has a home."""
    endf, suite, _ = decoded
    section = endf.mf[3].mt[2]
    flat = CrossSection.from_endf(section)
    reaction = suite.reactionByENDF_MT(2)
    provenance = reaction.provenance

    assert provenance.mat == flat.metadata["mat"]
    assert provenance.awr == pytest.approx(flat.metadata["awr"])
    assert provenance.qm == pytest.approx(flat.metadata["qm"])
    assert provenance.lr == flat.metadata["lr"]
    assert provenance.interpolationRegions == flat.metadata["interpolation_regions"]

    # `qi` is not provenance: it is the reaction's Q value, and it lives on the
    # output channel where §17.1.1 puts it.
    assert reaction.outputChannel.Q.value == pytest.approx(flat.metadata["qi"])


def test_a_missing_q_is_none_and_not_zero(decoded):
    """The distinction the untyped dict could not make.

    ``metadata.get('qi', 0.0)`` returns the same thing for "the Q is zero" and
    "there is no Q", which is half of the Q = 0 defect. ``Q.value is None`` says
    the second, and the encoder refuses rather than writing a zero.
    """
    from kika.nuclear_data.model import (EVAL_LABEL, EndfProvenance, Reaction,
                                          ReactionId, Regions1d)

    reaction = Reaction(id=ReactionId(label="MT102", ENDF_MT=102))
    assert reaction.outputChannel.Q.value is None
    assert reaction.outputChannel.Q.isKnown is False

    # Give it everything *except* a Q, so the failure is attributable to the Q
    # and not to some other missing piece.
    reaction.crossSection[EVAL_LABEL] = Regions1d.fromEndfRegions(
        np.array([1.0, 10.0]), np.array([2.0, 3.0]), [(2, 2)]
    )
    reaction.provenance = EndfProvenance(mat=2631, awr=55.0, za=26056, qm=0.0, lr=0)

    with pytest.raises(ValueError, match=r"carries no .*qi"):
        encodeMF3MT(reaction)

    # And with a Q it writes, which is what makes the failure above meaningful.
    reaction.outputChannel.Q.value = 0.0
    section, _ = encodeMF3MT(reaction)
    assert section.number == 102


def test_the_reconstructed_regions_agree_with_the_file_s_own_pairs(decoded):
    """The encoder prefers the file's ``(NBT, INT)`` pairs over the rebuilt ones.

    They agree — this asserts it — but preferring the original means a round
    trip does not depend on that agreement holding for every tape ever written.
    """
    endf, suite, _ = decoded
    for mt in sorted(endf.mf[3].mt):
        reaction = suite.reactionByENDF_MT(mt)
        _, _, rebuilt = reaction.crossSection["eval"].toEndfRegions()
        assert rebuilt == reaction.provenance.interpolationRegions, (
            f"MT{mt}: rebuilding the regions from regions1d does not reproduce "
            f"the file's own pairs"
        )


# ---------------------------------------------------------------------------
# The report declares what is missing
# ---------------------------------------------------------------------------

def test_the_report_names_every_mf_the_decoder_did_not_read(decoded):
    """The phase 7 hard rule, applied from the first decoder.

    *"A structurally valid, physically incomplete reactionSuite is worse than
    none, because it carries authority it has not earned."* So the gaps are
    declared, by MF number, rather than left for a reader to notice.
    """
    endf, _, report = decoded

    assert report.unsupported, "the micro-tape has MF34, which is not a reactionSuite node"
    text = " ".join(report.unsupported)
    # MF1, MF2, MF3 and MF4 are read as of P7b. What remains is the covariance
    # files, which §25.1.1 puts in a root node of their own.
    for mf in sorted(set(endf.mf) - {1, 2, 3, 4}):
        assert f"MF{mf}" in text, f"MF{mf} is in the file and the report does not mention it"


def test_a_report_is_always_present_even_when_it_says_nothing():
    """``if report:`` must not read "nothing to report" as "no report"."""
    from kika.nuclear_data.model import ConversionReport

    report = ConversionReport()
    assert bool(report) is True
    assert report.isClean and report.isEmpty
    assert report.summary() == "clean"


def test_decoding_one_section_reports_a_missing_interpolation_region():
    """An approximation is declared, because an approximation looks like data."""
    from kika.nuclear_data.model import ConversionReport

    class _Section:
        number = 2
        energies = [1.0, 10.0]
        cross_sections = [1.0, 2.0]
        energy_interpolation: list = []
        atomic_weight_ratio = 55.0
        q_mass_difference = 0.0
        q_reaction = 0.0
        breakup_flag = 0
        zaid = 26056
        _mat = 2631

    report = ConversionReport()
    _, report = decodeMF3MT(_Section(), report)

    assert report.approximations
    assert "lin-lin" in report.approximations[0]
