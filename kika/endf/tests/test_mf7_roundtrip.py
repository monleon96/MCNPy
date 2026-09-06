"""MF7: the read-write gate for the thermal scattering law, and what it decodes.

**The gate.** Parse an MF7 section, re-emit it, get the evaluator's bytes back.
Asserted on four committed micro-tapes covering all three ``LTHR`` branches,
both padding dialects and both libraries, and — under ``--deep`` — on the
full-size tapes including ``tsl-HinH2O.endf``, whose single MF7/MT4 is 1 144 410
records.

**What the gate would miss on its own.** Two record layouts in File 7 can be
walked wrongly and still round-trip, because getting them wrong is
self-consistent:

* the extra-temperature records of a :class:`TemperatureTable` carry S values
  with **no x column**, so reading them as x/y pairs consumes twice the records
  and re-writes them the same wrong way;
* the number of effective-temperature tables after MT4's β loop is written
  nowhere and has to be derived from the B array.

So the structural assertions below are not decoration. They check the counts
against the physics the file states elsewhere — LT against the temperature list,
NI against 6(NS+1), the Teff count against the secondary scatterers' laws.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kika.endf import read_endf, thermal_scatterer
from kika.endf.classes.mf7.elastic import MF7MT2
from kika.endf.classes.mf7.inelastic import FREE_GAS, SCT_APPROXIMATION, MF7MT4
from kika.endf.classes.mf7.composition import MF7MT451
from kika.endf.parsers.parse_mf7 import parse_mf7, parse_mf7_mt
from kika.endf.utils import PAD_BLANK, PAD_ZERO, parse_endf_id

# The tests directory has no ``__init__.py``, so pytest's prepend import mode
# puts it on ``sys.path``. Shared rather than copied, as MF32 already does.
from test_mf5_roundtrip import data_lines, first_difference  # noqa: E402


def source_width(lines: list[str]) -> int:
    """75 or 80, whichever the tape itself uses.

    ENDF/B-VIII.1's TSL sublibrary is 75 columns throughout — no sequence
    numbers. JEFF-4.0's ``tsl/`` is **mixed**: ``tsl_4-Be.txt`` carries all 80
    while ``tsl_Be_BeO.txt``, an adopted ENDF/B evaluation, does not. So the
    width is a property of the file and cannot be a constant here.
    """
    return 80 if any(len(line) > 75 for line in lines) else 75


def roundtrip(path, mt: int) -> str:
    """Parse MF7/MT*mt* out of *path* and report how its bytes differ."""
    text = Path(path).read_text()
    endf = read_endf(str(path), mf_numbers=[7])
    section = endf.files[7].sections[mt]
    raw = data_lines(text, 7, mt)
    width = source_width(raw)
    source = [line[:width] for line in raw]
    got = [line[:width] for line in data_lines(str(section), 7, mt)]
    return first_difference(source, got)


def sections_of(path) -> dict:
    return read_endf(str(path), mf_numbers=[7]).files[7].sections


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def test_every_committed_tsl_section_round_trips(micro_tsl_tape):
    """All four fixtures, every MF7 section in each, byte for byte."""
    sections = sections_of(micro_tsl_tape)
    assert sections, f"no MF7 parsed out of {Path(micro_tsl_tape).name}"
    for mt in sections:
        assert not roundtrip(micro_tsl_tape, mt), f"MF7/MT{mt}"


def test_the_gate_holds_through_the_public_reader(micro_tsl_sch4_tape):
    """``read_endf`` reaches the section the parser produces.

    Separate from the test above because registering MF7 in ``MF_PARSERS`` is
    its own failure mode: the parser can be correct and still be unreachable,
    and every test here that goes through ``read_endf`` would then skip its
    subject rather than fail.
    """
    endf = read_endf(str(micro_tsl_sch4_tape))
    assert 7 in endf.files, "MF7 is not registered in MF_PARSERS"
    assert sorted(endf.files[7].sections) == [2, 4]
    assert isinstance(endf.files[7].sections[2], MF7MT2)
    assert isinstance(endf.files[7].sections[4], MF7MT4)


def test_read_mf7_mt_returns_the_same_section(micro_tsl_sch4_tape):
    from kika.endf import read_mf7_mt

    section = read_mf7_mt(str(micro_tsl_sch4_tape), 4)
    assert isinstance(section, MF7MT4)
    assert read_mf7_mt(str(micro_tsl_sch4_tape), 451) is None


# ---------------------------------------------------------------------------
# MT2: the three LTHR branches
# ---------------------------------------------------------------------------

def test_lthr2_is_a_single_debye_waller_table(micro_tsl_sch4_tape):
    """s-CH₄: incoherent elastic only, so no coherent block exists."""
    mt2 = sections_of(micro_tsl_sch4_tape)[2]
    assert mt2.lthr == 2
    assert not mt2.has_coherent
    assert mt2.has_incoherent
    assert mt2.incoherent.sb == pytest.approx(817.44)
    assert mt2.incoherent.temperatures == [22.0, 22.0]


def test_lthr1_stacks_bragg_edges_over_temperature(micro_tsl_bemetal_elastic_tape):
    """Be metal: 2306 edges at 11 temperatures, on a histogram grid.

    ``LT + 1 == len(temperatures)`` is the assertion that catches the
    extra-temperature records being read as x/y pairs: doing so would halve the
    rows read while leaving LT untouched.
    """
    mt2 = sections_of(micro_tsl_bemetal_elastic_tape)[2]
    assert mt2.lthr == 1
    assert mt2.has_coherent and not mt2.has_incoherent

    coherent = mt2.coherent
    assert len(coherent.energies) == 2306
    assert coherent.table.lt == 10
    assert len(coherent.temperatures) == coherent.table.lt + 1 == 11
    assert coherent.temperatures[0] == 77.0
    assert coherent.temperatures[-1] == 1200.0
    assert coherent.is_histogram, "Bragg edges must interpolate as INT=1"
    for index in range(len(coherent.temperatures)):
        assert len(coherent.s_at(index)) == 2306


def test_lthr3_carries_both_blocks_in_order(micro_tsl_un_elastic_tape):
    """N in UN: coherent *then* incoherent, the branch ``lthr == 1`` misses."""
    mt2 = sections_of(micro_tsl_un_elastic_tape)[2]
    assert mt2.lthr == 3
    assert mt2.has_coherent and mt2.has_incoherent
    assert len(mt2.coherent.energies) == 911
    assert len(mt2.coherent.temperatures) == 8
    assert len(mt2.incoherent.temperatures) == 8
    assert mt2.incoherent.sb == pytest.approx(0.4981802)


def test_an_unknown_lthr_is_refused():
    """LTHR=4 has no record layout, so guessing one would misread the rest."""
    lines = [
        " 1.260000+2 8.934780+0          4          0          0          0  26 7  2",
    ]
    with pytest.raises(ValueError, match=r"LTHR=4"):
        parse_mf7_mt(lines, 2)


# ---------------------------------------------------------------------------
# MT4: the B array, the beta loop, the Teff tables
# ---------------------------------------------------------------------------

def test_the_b_array_decodes_to_the_documented_quantities(micro_tsl_sch4_tape):
    """s-CH₄ has one secondary scatterer treated as a free gas."""
    mt4 = sections_of(micro_tsl_sch4_tape)[4]
    assert mt4.lat == 0
    assert mt4.lasym == 0
    assert mt4.lln == 0
    assert mt4.ns == 1
    assert len(mt4.b) == 6 * (mt4.ns + 1) == 12

    assert mt4.free_atom_xs == pytest.approx(20.43589)
    assert mt4.principal_awr == pytest.approx(0.99917)
    assert mt4.n_principal_atoms == 1

    secondary = mt4.secondary_scatterers()
    assert len(secondary) == 1
    assert secondary[0].analytic_flag == FREE_GAS
    assert secondary[0].awr == pytest.approx(11.898), "carbon"
    assert not secondary[0].needs_teff
    assert mt4.expected_teff_records == 1
    assert len(mt4.teff) == 1


def test_free_atom_cross_section_is_sigma_times_the_atom_count(
        micro_tsl_bemetal_elastic_tape, micro_tsl_sch4_tape):
    """``B(1) == σ_free × B(6)``, which is why B(1) alone is not σ_free.

    Measured, and the reason :attr:`MF7MT4.free_atom_xs` says so in its name's
    docstring rather than being called ``sigma_free``: H-in-H₂O and H-in-CH₂
    both write 40.87 for a σ_free of 20.436, because both have two principal
    atoms. Reading B(1) as a per-atom cross section is wrong by exactly B(6).
    """
    mt4 = sections_of(micro_tsl_sch4_tape)[4]
    assert mt4.free_atom_xs / mt4.n_principal_atoms == pytest.approx(20.43589)


def test_the_beta_loop_shares_one_alpha_grid_and_one_temperature_grid(
        micro_tsl_sch4_tape):
    mt4 = sections_of(micro_tsl_sch4_tape)[4]
    assert len(mt4.blocks) == 80
    assert mt4.betas[0] == 0.0
    assert len(mt4.temperatures) == 1

    first = mt4.blocks[0]
    assert len(first.alphas) == 70
    assert len(first.s_at(0)) == len(first.alphas)
    for block in mt4.blocks:
        assert block.temperatures == mt4.temperatures


def test_ni_must_be_six_times_ns_plus_one():
    """A B array whose length contradicts NS is refused, not truncated."""
    lines = [
        " 1.340000+2 1.589400+1          0          0          0          0  34 7  4",
        " 0.000000+0 0.000000+0          0          0          6          1  34 7  4",
        " 2.043589+1 4.500000+2 9.991700-1 1.138500+1 0.000000+0 1.000000+0  34 7  4",
    ]
    with pytest.raises(ValueError, match=r"NI = 6\(NS\+1\) = 12"):
        parse_mf7_mt(lines, 4)


def test_a_short_temperature_stack_is_refused():
    """LT promises more records than the section holds."""
    lines = [
        " 1.260000+2 8.934780+0          1          0          0          0  26 7  2",
        " 7.700000+1 0.000000+0          3          0          1          2  26 7  2",
        "          2          1                                              26 7  2",
        " 1.000000+0 2.000000+0 3.000000+0 4.000000+0                        26 7  2",
    ]
    with pytest.raises(ValueError, match=r"LT=3 but the section ended after 0"):
        parse_mf7_mt(lines, 2)


# ---------------------------------------------------------------------------
# MT451: composition
# ---------------------------------------------------------------------------

def test_mt451_maps_the_pseudo_za_back_to_nuclides(micro_tsl_bemetal_elastic_tape):
    """ZA 126 names nothing; MT451 says the scatterer is Be-9."""
    mt451 = sections_of(micro_tsl_bemetal_elastic_tape)[451]
    assert isinstance(mt451, MF7MT451)
    assert mt451.za == 126, "the HEAD's pseudo-ZA, not 4009"

    isotopes = mt451.isotopes()
    assert len(isotopes) == 1
    assert isotopes[0].zai == 4009
    assert (isotopes[0].z, isotopes[0].a) == (4, 9)
    assert isotopes[0].atom_fraction == pytest.approx(1.0)
    assert isotopes[0].sigma_free == pytest.approx(6.153875)


def test_mt451_isotope_count_comes_from_nw_not_nas(micro_tsl_un_elastic_tape):
    """N in UN lists two nitrogen isotopes in one element record.

    NW is the only field that cannot disagree with the body under it. NAS is
    documented as the atom count and sometimes is not — ``tsl-BeinBe2C`` writes
    1 for Be₂C — so nothing here is sized from it.
    """
    mt451 = sections_of(micro_tsl_un_elastic_tape)[451]
    assert mt451.na == 1
    isotopes = mt451.isotopes()
    assert len(isotopes) == 2
    assert [int(i.zai) for i in isotopes] == [7014, 7015]
    assert sum(i.atom_fraction for i in isotopes) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Dialects
# ---------------------------------------------------------------------------

def test_padding_is_read_from_the_tape_not_assumed(
        micro_tsl_jeff_be_elastic_tape, micro_tsl_bemetal_elastic_tape):
    """JEFF-4.0's Be zero-fills LIST bodies and blank-fills TAB1 bodies.

    Inside one section. This is why :class:`~kika.endf.utils.PadStyle` has two
    fields: a single flag would write 5 994 lines of ``tsl_4-Be.txt`` back
    wrongly whichever value it took.
    """
    jeff = sections_of(micro_tsl_jeff_be_elastic_tape)[2]
    assert jeff.pad.pairs == PAD_BLANK
    assert jeff.pad.values == PAD_ZERO

    endfb = sections_of(micro_tsl_bemetal_elastic_tape)[2]
    assert endfb.pad.pairs == PAD_BLANK
    assert endfb.pad.values == PAD_BLANK


def test_the_two_libraries_produce_the_same_classes(
        micro_tsl_bemetal_elastic_tape, micro_tsl_jeff_be_elastic_tape):
    """Both dialects land in the same objects, read through the same names."""
    for tape in (micro_tsl_bemetal_elastic_tape, micro_tsl_jeff_be_elastic_tape):
        mt2 = sections_of(tape)[2]
        assert isinstance(mt2, MF7MT2)
        assert mt2.lthr == 1
        assert mt2.coherent.is_histogram
        assert len(mt2.coherent.temperatures) == mt2.coherent.table.lt + 1


# ---------------------------------------------------------------------------
# Parser robustness
# ---------------------------------------------------------------------------

def test_an_mt_that_is_not_a_file_7_section_is_refused():
    lines = [" 1.260000+2 8.934780+0          0          0          0          0  26 7 18"]
    with pytest.raises(ValueError, match=r"MF7/MT18 is not a section"):
        parse_mf7_mt(lines, 18)


def test_a_bad_section_loses_only_itself(micro_tsl_bemetal_elastic_tape, caplog):
    """The file-level loop logs and continues, as MF4's and MF5's do."""
    text = Path(micro_tsl_bemetal_elastic_tape).read_text().splitlines()
    mf7_lines = [line for line in text
                 if len(line) >= 75 and parse_endf_id(line)[1] == 7]
    broken = [
        line.replace("          1          0          0          0",
                     "          9          0          0          0", 1)
        if parse_endf_id(line)[2] == 2 and line.startswith(" 1.260000+2")
        else line
        for line in mf7_lines
    ]

    mf = parse_mf7(broken)
    assert 2 not in mf.sections, "the LTHR=9 section should have been refused"
    assert 451 in mf.sections, "a bad MT2 took MT451 down with it"


def test_a_section_that_does_not_consume_its_own_records_is_refused():
    """Trailing records mean the walk was wrong somewhere above them."""
    lines = [
        " 1.070000+2 1.000000+0          2          0          0          0   7 7  2",
        " 8.198006+1 0.000000+0          0          0          1          2   7 7  2",
        "          2          2                                              7 7  2",
        " 2.960000+2 8.486993+0 4.000000+2 9.093191+0                         7 7  2",
        " 9.999999+9 9.999999+9                                               7 7  2",
    ]
    with pytest.raises(ValueError, match=r"1 left over"):
        parse_mf7_mt(lines, 2)


# ---------------------------------------------------------------------------
# Writing back
# ---------------------------------------------------------------------------

def write_back(tape, mt: int, tmp_path) -> tuple[list[str], list[str]]:
    """Splice MF7/MT*mt* back into its own tape unchanged; return both texts."""
    from kika.endf.writers._section_writer import write_mf_section_to_file

    section = sections_of(tape)[mt]
    out = tmp_path / "written.endf"
    write_mf_section_to_file(str(tape), section, str(out))
    return (Path(tape).read_text().splitlines(), out.read_text().splitlines())


def test_writing_a_section_back_unchanged_changes_nothing(
        micro_tsl_un_elastic_tape, tmp_path):
    """The whole-file check the per-section gate cannot make.

    ``str(section)`` is only half of writing: the other half is the splice and
    the MF1/451 directory rebuild, and either can corrupt a file the section
    itself renders perfectly. LTHR=3 is the case to run it on, because its MT2
    is two blocks and a writer that emitted only the first would still produce
    a plausible file.
    """
    source, written = write_back(micro_tsl_un_elastic_tape, 2, tmp_path)
    assert len(source) == len(written)
    differing = [i for i, (a, b) in enumerate(zip(source, written)) if a != b]
    assert not differing, f"lines {differing[:5]} changed"


def test_a_seventy_five_column_tape_does_not_gain_sequence_numbers(
        micro_tsl_bemetal_elastic_tape, tmp_path):
    """ENDF/B-VIII.1's TSL sublibrary omits the sequence number; kika writes one.

    Without ``match_source_width`` the spliced section came back 80 columns wide
    in a 75-column file — a tape whose record width changes partway through,
    which NJOY reads without complaint and processes wrongly.
    """
    source, written = write_back(micro_tsl_bemetal_elastic_tape, 2, tmp_path)
    assert max(len(line) for line in source) == 75
    assert max(len(line) for line in written) == 75


def test_an_eighty_column_tape_keeps_its_sequence_numbers(
        micro_tsl_jeff_be_elastic_tape, tmp_path):
    """The other direction: JEFF-4.0's own TSL carries all 80 columns.

    The same test as above and not a duplicate of it — a width rule that simply
    truncated everything to 75 would pass that one and destroy this file.

    One line changes, and it is a measured divergence rather than a defect: the
    SEND record. ENDF-6 lets a section terminator leave columns 1-66 blank or
    write six explicit zeros; this tape blanks them, ``tsl-Be-metal.endf``
    zero-fills them (which is why the two tests above see no difference at all),
    and kika's shared ``format_endf_send_record`` always zero-fills. Both forms
    parse to the same terminator. Preserving the blank one would mean storing
    the source text of a record that carries no data.
    """
    source, written = write_back(micro_tsl_jeff_be_elastic_tape, 2, tmp_path)
    assert max(len(line) for line in source) == 80
    assert max(len(line) for line in written) == 80

    differing = [i for i, (a, b) in enumerate(zip(source, written)) if a != b]
    assert len(differing) == 1, f"expected only the SEND to change, got {differing[:5]}"

    index = differing[0]
    assert parse_endf_id(source[index])[1:] == (7, 0), "the changed line is a SEND"
    assert not source[index][:66].strip()
    assert written[index][:66] == " 0.000000+0" * 2 + f"{0:11d}" * 4


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_a_tsl_tape_has_no_nuclide_and_says_so(micro_tsl_sch4_tape):
    """``zaid``/``isotope`` stay None; ``is_thermal_scattering`` explains why."""
    endf = read_endf(str(micro_tsl_sch4_tape), mf_numbers=[1, 7])
    assert endf.mat == 34
    assert endf.zaid is None
    assert endf.isotope is None
    assert endf.is_thermal_scattering


def test_a_neutron_tape_is_not_thermal_scattering(micro_tape):
    endf = read_endf(str(micro_tape), mf_numbers=[1])
    assert not endf.is_thermal_scattering
    assert endf.zaid == 26056


@pytest.mark.parametrize("key,source,name,name_source,principal,compound", [
    ("sch4", "tsl-s-CH4.endf", "s-CH4", "filename", None, None),
    ("bemetal_elastic", "tsl-Be-metal.endf", "Be-metal", "filename", None, None),
    ("un_elastic", "tsl-NinUN.endf", "NinUN", "zsymam", "N", "UN"),
    ("jeff_be_elastic", "tsl_4-Be.txt", "4-Be", "filename", None, None),
])
def test_the_name_is_inferred_and_says_which_rule_produced_it(
        key, source, name, name_source, principal, compound, request):
    """ZSYMAM first, filename second — and ``name_source`` records which.

    The four cases are the four conventions really in use. ``N(UN)`` parses;
    ``s-CH4``, ``Be-Metal`` and JEFF's `` 4-Be-`` do not name a compound at all,
    so the filename is the only thing that can name them. An inferred name must
    never be mistaken for a stated one, which is what ``name_source`` is for.
    """
    tape = request.getfixturevalue(f"micro_tsl_{key}_tape")
    endf = read_endf(str(tape), mf_numbers=[1, 7])
    scatterer = thermal_scatterer(endf, source=source)

    assert scatterer.name == name
    assert scatterer.name_source == name_source
    assert scatterer.principal == principal
    assert scatterer.compound == compound


def test_the_scatterer_carries_the_composition_when_the_tape_has_one(
        micro_tsl_bemetal_elastic_tape, micro_tsl_sch4_tape):
    """28 of ENDF/B-VIII.1's 114 TSL tapes have no MF7/451; that is not an error."""
    endf = read_endf(str(micro_tsl_bemetal_elastic_tape), mf_numbers=[1, 7])
    scatterer = thermal_scatterer(endf, source="tsl-Be-metal.endf")
    assert scatterer.mat == 26
    assert scatterer.za == 126
    assert scatterer.zsymam == "Be-Metal"
    assert [int(i.zai) for i in scatterer.composition] == [4009]

    bare = thermal_scatterer(
        read_endf(str(micro_tsl_sch4_tape), mf_numbers=[1, 7]),
        source="tsl-s-CH4.endf",
    )
    assert bare.composition == []


def test_a_scatterer_can_be_named_without_the_filename(micro_tsl_un_elastic_tape):
    """ZSYMAM alone is enough when it names both parts."""
    endf = read_endf(str(micro_tsl_un_elastic_tape), mf_numbers=[1, 7])
    assert thermal_scatterer(endf).name == "NinUN"


def test_za_and_awr_are_read_from_mf1_when_it_is_there(micro_tsl_bemetal_elastic_tape):
    """And from MF7's HEAD when only MF7 was parsed.

    The two files spell the same two fields differently — ``MF1MT451.zaid`` and
    ``atomic_weight_ratio`` against MF7's ``za`` and ``awr`` — so this pins that
    both paths are actually wired rather than one quietly falling through to
    the other.
    """
    with_mf1 = thermal_scatterer(
        read_endf(str(micro_tsl_bemetal_elastic_tape), mf_numbers=[1, 7]))
    mf7_only = thermal_scatterer(
        read_endf(str(micro_tsl_bemetal_elastic_tape), mf_numbers=[7]))

    assert with_mf1.za == mf7_only.za == 126
    assert with_mf1.awr == mf7_only.awr == pytest.approx(8.93478)
    assert with_mf1.zsymam == "Be-Metal"
    assert mf7_only.zsymam is None, "MF1 was not read, so there is no label"


# ---------------------------------------------------------------------------
# The full-size tapes
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_the_largest_section_in_the_library_round_trips(tsl_h_h2o_tape):
    """H in H₂O: 1 144 410 records in one MF7/MT4, 317 β × 94 temperatures.

    The size is the point. It is the only tape where the temperature stack is
    deep enough (94) for an off-by-one in the LIST walk to stay inside the
    section rather than running off its end, and the only one with a
    secondary scatterer *and* a full β loop. Roughly 35 s to parse.
    """
    endf = read_endf(str(tsl_h_h2o_tape), mf_numbers=[7])
    mt4 = endf.files[7].sections[4]
    assert len(mt4.blocks) == 317
    assert len(mt4.temperatures) == 94
    assert mt4.n_principal_atoms == 2, "two hydrogens per water molecule"
    assert mt4.ns == 1
    assert mt4.secondary_scatterers()[0].analytic_flag == FREE_GAS
    assert mt4.expected_teff_records == 1
    assert not roundtrip(tsl_h_h2o_tape, 4)


@pytest.mark.slow
def test_lasym_round_trips_and_carries_negative_beta(tsl_ortho_h_tape):
    """ortho-H: the only ``LASYM=1`` branch, and the only sub-1e-100 values.

    Both matter. ``LASYM=1`` means S(α, β) is asymmetric so negative β are
    tabulated explicitly, and this tape reaches down to 1.5963e-100 — which
    ``format_endf_number`` used to write as ``0.000000+0``, silently, on 1 403
    of these records.
    """
    endf = read_endf(str(tsl_ortho_h_tape), mf_numbers=[7])
    mt4 = endf.files[7].sections[4]
    assert mt4.lasym == 1
    assert len(mt4.blocks) == 595
    assert min(mt4.betas) < 0, "an asymmetric table must tabulate negative beta"
    assert not roundtrip(tsl_ortho_h_tape, 4)


@pytest.mark.slow
def test_the_jeff_dialect_round_trips_at_full_size(tsl_jeff_be_tape):
    """80-column records with sequence numbers, CRLF, zero-padded LIST bodies."""
    for mt in sections_of(tsl_jeff_be_tape):
        assert not roundtrip(tsl_jeff_be_tape, mt), f"MF7/MT{mt}"


def test_the_same_evaluation_reads_identically_from_both_libraries(
        tsl_be_metal_tape, tsl_jeff_be_beo_tape, tsl_un_tape):
    """JEFF-4.0 adopted ENDF/B's TSL evaluations; the reader must not care.

    ``tsl_Be_BeO.txt`` and ``tsl-BeinBeO.endf`` are the same bytes under
    different names. What this pins is the other half of the user-facing
    promise: whichever library a TSL tape comes from, it lands in the same
    ``endf.files[7]`` objects, read through the same attribute names.
    """
    from_jeff = sections_of(tsl_jeff_be_beo_tape)
    assert sorted(from_jeff) == [2, 4, 451]
    assert from_jeff[2].lthr == 1
    assert from_jeff[4].n_principal_atoms == 1

    from_endfb = sections_of(tsl_be_metal_tape)
    assert type(from_jeff[2]) is type(from_endfb[2])
    assert type(from_jeff[4]) is type(from_endfb[4])


@pytest.mark.slow
def test_a_plain_decimal_field_is_re_emitted_canonically(tsl_jeff_si_tape):
    """A measured divergence, not a defect: same value, different bytes.

    JEFF-4.0's Si writes ``2.16827944`` as the SB field of MF7/MT2's incoherent
    TAB1, in plain Fortran decimal rather than ENDF's exponential form.
    ``parse_number`` tries ``float()`` first and reads it without complaint, and
    kika re-emits the canonical ``2.168279+0``. The values are equal; the bytes
    are not.

    Recorded as a measurement rather than pinned as an xfail because there is
    nothing to fix — re-emitting it verbatim would mean storing the source text
    of every number on the tape. Identical in kind to
    ``test_endfb81_pu239_mf5_head_roundtrip``, and the reason this is *one* line
    rather than a class of them: it is the only field on the tape written that
    way.
    """
    text = Path(tsl_jeff_si_tape).read_text()
    section = sections_of(tsl_jeff_si_tape)[2]

    source = data_lines(text, 7, 2)
    got = data_lines(str(section), 7, 2)
    assert source[1][:11].strip() == "2.16827944"
    assert got[1][:11].strip() == "2.168279+0"
    assert section.incoherent.sb == pytest.approx(2.16827944)

    width = source_width(source)
    differing = [i for i, (a, b)
                 in enumerate(zip(source, got)) if a[:width] != b[:width]]
    assert differing == [1], f"only the SB field should differ, got {differing}"
