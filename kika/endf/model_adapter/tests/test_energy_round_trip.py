"""MF5 → the ``energy`` half of ``uncorrelated`` → MF5, byte for byte.

The same gate as ``test_angular_round_trip``, and against the file rather than
against any flat class: ``str(encodeMF5MT(*decodeMF5MT(section))) == str(section)``.
It is the strictly stronger statement — a decoder that dropped a trailing point
or re-derived an interpolation code would still be its own fixed point.

Two paths are exercised and they are not variations of one another. **LF=1 goes
through the model**: the TAB2 becomes an ``XYs2d`` (or a ``Regions2d``), every
node a ``XYs1d`` or a ``Regions1d``, and the bytes are rebuilt from those.
Everything else **goes through the provenance** as the records the evaluator
wrote, because the six analytic spectra of §18.3 have no model node to be
rebuilt from. Cf-252 carries one of each — a real LF=1 MT18 and a real NK=6 of
LF=5 in MT455 — so the committed fixture alone covers both.
"""
from __future__ import annotations

import pytest

from kika.endf.model_adapter import decodeMF5MT, encodeMF5MT
from kika.endf.read_endf import read_endf
from kika.nuclear_data.model import Regions2d, XYs2d

#: Under ``--deep``. Between them: U-235's per-incident-node outgoing grids,
#: which is the shape a one-shared-grid reader gets wrong on the reference
#: library, and three more fissile evaluations.
REAL_TAPES = ["u235_tape", "pu241_tape", "th232_tape", "cf252_b81_tape"]


@pytest.fixture(scope="module")
def mf5(micro_pfns_tape):
    return read_endf(str(micro_pfns_tape)).mf[5]


def _roundTrip(section, mt):
    form, provenance, report = decodeMF5MT(section)
    encoded, report = encodeMF5MT(form, provenance, mt, report)
    return encoded, form, report


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def test_the_fixture_carries_both_paths(mf5):
    """One LF=1 and one NK>1 of a law with no node. A tape with only the first
    would leave the half that round-trips out of the provenance untested."""
    assert [p.lf for p in mf5.mt[18].partials] == [1]
    assert [p.lf for p in mf5.mt[455].partials] == [5] * 6


@pytest.mark.parametrize("mt", [18, 455])
def test_the_section_encodes_byte_identically_to_the_file(mf5, mt):
    encoded, _, _ = _roundTrip(mf5.mt[mt], mt)
    assert str(encoded) == str(mf5.mt[mt])


@pytest.mark.parametrize("tape", REAL_TAPES)
def test_every_mf5_section_of_a_real_tape_encodes_byte_identically(request, tape):
    endf = read_endf(str(request.getfixturevalue(tape)))
    if 5 not in endf.mf:
        pytest.skip(f"{tape} carries no MF5")

    laws = set()
    for mt in sorted(endf.mf[5].mt):
        section = endf.mf[5].mt[mt]
        laws.update(p.lf for p in section.partials)
        encoded, _, _ = _roundTrip(section, mt)
        assert str(encoded) == str(section), f"{tape} MT{mt}"
    assert laws, f"{tape} MF5 is empty"


def test_a_weighted_sum_of_tabulated_laws_still_comes_back(u235_tape):
    """The case ENDF/B-VIII.1 does not contain, and JEFF-4.0 does.

    Measured 2026-08-24 over ENDF/B-VIII.1's 595 MF5 sections: **zero** NK>1
    sections hold an LF=1 — 487 are NK=1/LF=1 and the other 108 are homogeneous
    LF=5/7/9. A reader tested only against that library would never meet a
    tabulated partial it is not allowed to model, and would lose it silently:
    the first version of this adapter kept verbatim records only for the laws
    it could not *parse*, so a `weightedFunctionals` of eight tabulated
    subsections came back as eight empty ones — 2 354 lines down to 34.

    **JEFF-4.0's U-235 is that section**, MT455 with NK=8 and all eight LF=1,
    the per-precursor delayed spectra. It is why the provenance keeps the bytes
    of every partial that does not reach the model, whatever its law, and not
    only of the analytic ones.
    """
    section = read_endf(str(u235_tape)).mf[5].mt[455]
    assert [p.lf for p in section.partials] == [1] * 8

    encoded, form, report = _roundTrip(section, 455)
    assert form is None, "one partial of a weighted sum is not the distribution"
    assert any("weightedFunctionals" in line for line in report.unsupported)
    assert str(encoded) == str(section)


# ---------------------------------------------------------------------------
# What reaches the model, and what deliberately does not
# ---------------------------------------------------------------------------

def test_an_lf1_reaches_the_model(mf5):
    _, form, report = _roundTrip(mf5.mt[18], 18)
    assert isinstance(form, (XYs2d, Regions2d))
    assert report.isClean, report.summary()
    assert len(form.function1ds) == len(mf5.mt[18].partials[0].incident_energies)


def test_the_energy_axes_come_off_the_node_and_are_shared(mf5):
    """``kika/gnds/encode.py:_axesUnlessNested`` decides "this child inherits"
    by object **identity**, so one factory call per region would make every
    nested form look like it carried axes of its own."""
    _, form, _ = _roundTrip(mf5.mt[18], 18)
    assert [(a.index, a.label, a.unit) for a in form.axes.axes] == [
        (2, "energy_in", "eV"),
        (1, "energy_out", "eV"),
        (0, "P(energy_out|energy_in)", "1/eV"),
    ]
    if isinstance(form, Regions2d):
        assert all(region.axes is form.axes for region in form.function2ds)


def test_nk_greater_than_one_is_declared_and_not_half_read(mf5):
    """A partial of a weighted sum is not the distribution.

    MT455 is NK=6. Even had one of the six been an LF=1, hanging it on the
    product as *the* energy distribution would be a statement no schema can
    catch — §18.3's node for a weighted sum is ``weightedFunctionals``, which
    kika does not model. So the whole section stays out of the reactionSuite.
    """
    _, form, report = _roundTrip(mf5.mt[455], 455)
    assert form is None
    assert any("weightedFunctionals" in line for line in report.unsupported)


def test_the_encoder_refuses_to_work_without_the_provenance(mf5):
    """NK, LF, U and p(E) are not in the model, and neither are the laws it
    does not model — so this is not a lossy encode, it is an impossible one."""
    form, _provenance, _report = decodeMF5MT(mf5.mt[18])
    with pytest.raises(ValueError, match="needs the EndfProvenance"):
        encodeMF5MT(form, None, 18)


def test_the_encoder_refuses_a_form_the_provenance_does_not_expect(mf5):
    """MT455 modelled nothing; handing it a form would silently write the
    section as an LF=1 it never was."""
    form, _p, _r = decodeMF5MT(mf5.mt[18])
    _f, provenance, _r = decodeMF5MT(mf5.mt[455])
    with pytest.raises(ValueError, match="carried the model form"):
        encodeMF5MT(form, provenance, 455)


# ---------------------------------------------------------------------------
# The interpolation qualifier — §0.5.2.1's decade, which MF4 never needed
# ---------------------------------------------------------------------------

def test_a_unit_base_tab2_becomes_a_qualifier_and_comes_back_as_22():
    """The defect this increment had to fix before it could read the library.

    ENDF-6 §0.5.2.1 puts the *qualifier* of a two-dimensional interpolation in
    the tens digit of the INT code, and MF5's TAB2 uses ``INT=22`` — unit base,
    lin-lin — in 44 of ENDF/B-VIII.1's 487 LF=1 sections. MF4's TAB2 never
    does, so ``fromEndfTab2`` knew codes 1-6 only and raised ``KeyError: 22``
    on N-15, Mg-24 and forty others. GNDS states the same thing as a second
    attribute, which the writer already had; only the ENDF mapping was absent.
    """
    from kika.endf.classes.mf5.partials import MF5PartialTabulated
    from kika.nuclear_data.model import InterpolationQualifier

    partial = MF5PartialTabulated(
        u=0.0, lf=1, p_interp=[(2, 2)], p_energies=[1.0e5, 2.0e7],
        p_values=[1.0, 1.0],
        tab2_interp=[(2, 22)],
        incident_energies=[1.0e5, 2.0e7],
        outgoing_grids=[[0.0, 1.0e5], [0.0, 2.0e6]],
        chi=[[0.0, 1.0e-5], [0.0, 5.0e-7]],
        outgoing_interp=[[(2, 2)], [(2, 2)]],
    )
    section = _section(16, [partial])

    form, provenance, report = decodeMF5MT(section)
    assert form.interpolationQualifier is InterpolationQualifier.unitBase
    assert form.interpolation == "lin-lin"
    assert report.isClean, report.summary()

    encoded, _ = encodeMF5MT(form, provenance, 16)
    assert encoded.partials[0].tab2_interp == [(2, 22)]
    assert str(encoded) == str(section)


def _section(mt, partials):
    from kika.endf.classes.mf5.base import MF5MT

    section = MF5MT(number=mt)
    section._za, section._awr, section._mat = 26056.0, 55.454, 2631
    section._nk = len(partials)
    section.partials = partials
    return section


def test_mf5_keeps_its_own_header_and_not_mf4s(u235_tape):
    """One product, one provenance — and two files that can disagree.

    MF4's MAT/ZA/AWR sit at the top level of ``EndfProvenance`` and MF5's live
    inside its own ``headerFields["mf5"]`` block. Merging them would let
    whichever file was decoded first decide the other's header, and **two
    sections of ENDF/B-VIII.1 disagree with themselves**: Ce-140 writes
    ``AWR=1.387036+2`` in MF5/MT91 against ``1.387030+2`` in MF4, and Am-243
    does the same in MT18. Six significant figures apart is still a different
    byte, and only writing whole tapes back finds it — the section round trip
    cannot, because on its own there is no MF4 to be contaminated by.
    """
    section = read_endf(str(u235_tape)).mf[5].mt[18]
    _encoded, provenance, _report = _roundTrip(section, 18)
    block = provenance.headerFields["mf5"]
    assert block["awr"] == section._awr
    assert block["mat"] == section._mat
    assert "ltt" not in block, "MF4's fields stay flat and out of this block"
