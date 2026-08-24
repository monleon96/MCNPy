"""MF6 → model → MF6, byte for byte, and what the model got out of it.

The gate is the same one MF5's adapter answers to and for the same reason: a
section decoded and re-encoded must be the bytes it came in as. It is a
stronger claim here than the shape of the code suggests, because MF6 is where
the slicing of this adapter happens — a law that has not been mapped yet is
kept verbatim in the provenance, so *this file passes at every commit* and each
step moves laws from the bytes to the model without moving the result.

The four neutron fixtures carry every LAW a neutron sublibrary has and the five
charged-particle ones carry LAW=5, so ``micro_mf6_any_tape`` is the whole of
what is reachable offline. Nothing here reads ``/share_snc``.
"""
from __future__ import annotations

import pytest

from kika.endf import read_endf
from kika.endf.model_adapter import decodeMF6MT, encodeMF6MT
from kika.endf.model_adapter.decode import decodeReactionSuite
from kika.endf.model_adapter.energy_angle import frameForProduct, recoilAngularHref
from kika.nuclear_data.model import (
    AngularEnergy,
    AngularTwoBody,
    EnergyAngular,
    Frame,
    Interpolation,
    Isotropic2d,
    KalbachMann,
    Uncorrelated,
    Unspecified,
    XYs1d,
    pidFromZA,
    zaFromPid,
)


def _roundTrip(section, mt):
    """Decode and encode in one breath — they are never tested apart."""
    entries, provenance, report = decodeMF6MT(section)
    forms = {label: distribution
             for label, _pid, _multiplicity, distribution in entries}
    encoded, report = encodeMF6MT(forms, provenance, mt, report)
    return encoded, entries, provenance, report


# ---------------------------------------------------------------------------
# The byte gate
# ---------------------------------------------------------------------------

def test_every_section_encodes_byte_identically(micro_mf6_any_tape):
    endf = read_endf(str(micro_mf6_any_tape), mf_numbers=[6])
    sections = endf.mf[6].mt
    assert sections, f"{micro_mf6_any_tape.name} carries no MF6"
    for mt, section in sorted(sections.items()):
        encoded, *_ = _roundTrip(section, mt)
        assert str(encoded) == str(section), f"MT{mt} of {micro_mf6_any_tape.name}"


def test_the_whole_tape_comes_back_with_its_mf6(tmp_path, micro_mf6_tape):
    """``read -> suite -> write -> read`` leaves every MF6 section untouched.

    The section gate above cannot see the assembly: MF6 could encode perfectly
    and still be dropped on the way to a tape, which is exactly what
    ``MF_WRITE_ORDER`` did until this adapter landed.
    """
    import kika

    suite, _ = decodeReactionSuite(read_endf(str(micro_mf6_tape)))
    written = tmp_path / "written.endf"
    kika.write(suite, written, format="endf")

    before = read_endf(str(micro_mf6_tape), mf_numbers=[6]).mf[6].mt
    after = read_endf(str(written), mf_numbers=[6]).mf[6].mt
    assert sorted(after) == sorted(before)
    for mt in sorted(before):
        assert str(after[mt]) == str(before[mt]), f"MT{mt}"


# ---------------------------------------------------------------------------
# The two refusals
# ---------------------------------------------------------------------------

def test_encoding_without_the_provenance_is_refused(micro_mf6_tape):
    endf = read_endf(str(micro_mf6_tape), mf_numbers=[6])
    mt = sorted(endf.mf[6].mt)[0]
    with pytest.raises(ValueError, match="needs the EndfProvenance"):
        encodeMF6MT({}, None, mt)


def test_a_form_for_a_product_the_section_does_not_have_is_refused(micro_mf6_tape):
    endf = read_endf(str(micro_mf6_tape), mf_numbers=[6])
    mt = sorted(endf.mf[6].mt)[0]
    _entries, provenance, report = decodeMF6MT(endf.mf[6].mt[mt])
    with pytest.raises(ValueError, match="does not list as products"):
        encodeMF6MT({"Xx999": object()}, provenance, mt, report)


# ---------------------------------------------------------------------------
# The spine: products, pids, multiplicities
# ---------------------------------------------------------------------------

def test_a_product_is_built_for_every_subsection_that_states_a_law(micro_mf6_any_tape):
    endf = read_endf(str(micro_mf6_any_tape), mf_numbers=[6])
    for mt, section in sorted(endf.mf[6].mt.items()):
        _encoded, entries, _provenance, _report = _roundTrip(section, mt)
        stating = [p for p in section.products if p.law >= 0]
        assert len(entries) == len(stating), f"MT{mt}"


def test_every_product_of_a_section_gets_its_own_label(micro_mf6_any_tape):
    """Two subsections may name one particle, and §17.2.1 tells them apart.

    It does not happen in a neutron sublibrary — 296 sections and 808
    non-deferring products of eight ENDF/B-VIII.1 tapes, not one repeat — which
    is exactly why the charged-particle fixtures are in this parametrisation:
    ``a+He4`` and ``d+H2`` MT2 make the ejectile and its recoil the same
    nuclide, and ``t+Li7`` MT24 emits two He4.
    """
    endf = read_endf(str(micro_mf6_any_tape), mf_numbers=[6])
    for mt, section in sorted(endf.mf[6].mt.items()):
        _encoded, entries, _provenance, _report = _roundTrip(section, mt)
        labels = [label for label, _pid, _m, _d in entries]
        assert len(labels) == len(set(labels)), f"MT{mt} repeats a label"


def test_a_repeated_particle_keeps_its_pid_and_gains_an_ordinal():
    """``a+He4`` MT2: elastic off an identical particle, twice the same ZAP."""
    endf = read_endf("kika/endf/tests/data/micro_a_he4_mf6.endf", mf_numbers=[6])
    _encoded, entries, _provenance, report = _roundTrip(endf.mf[6].mt[2], 2)
    byLabel = {label: pid for label, pid, _m, _d in entries}
    assert byLabel == {"He4": "He4", "He4__1": "He4"}
    assert any("two nodes for one particle" in m for m in report.approximations)


def test_the_yield_becomes_the_multiplicity(micro_mf6_tape):
    endf = read_endf(str(micro_mf6_tape), mf_numbers=[6])
    for mt, section in sorted(endf.mf[6].mt.items()):
        _encoded, entries, _provenance, _report = _roundTrip(section, mt)
        stating = [p for p in section.products if p.law >= 0]
        for (label, _pid, multiplicity, _d), product in zip(entries, stating):
            assert multiplicity is not None, f"MT{mt} {label}"
            assert multiplicity.isEvaluable
            values = list(multiplicity.form.toEndfRegions()[1])
            assert values == pytest.approx(list(product.y_values))


def test_a_single_region_yield_is_an_xys1d_and_not_a_one_region_regions1d(
        micro_mf6_tape):
    """``gnds.xsd`` puts ``minOccurs="2"`` on the children of ``regions1d``."""
    endf = read_endf(str(micro_mf6_tape), mf_numbers=[6])
    seen = 0
    for mt, section in sorted(endf.mf[6].mt.items()):
        _encoded, entries, _p, _r = _roundTrip(section, mt)
        stating = [p for p in section.products if p.law >= 0]
        for (_label, _pid, multiplicity, _d), product in zip(entries, stating):
            if len(product.y_interp) <= 1:
                assert isinstance(multiplicity.form, XYs1d)
                seen += 1
    assert seen, "no single-region yield in this fixture"


# ---------------------------------------------------------------------------
# The laws with no body
# ---------------------------------------------------------------------------

def test_law_zero_is_unspecified_and_law_three_is_an_isotropic_two_body():
    endf = read_endf("kika/endf/tests/data/micro_u235_mf6.endf", mf_numbers=[6])

    _e, entries, _p, _r = _roundTrip(endf.mf[6].mt[18], 18)
    assert [pid for _label, pid, _m, _d in entries] == ["n", "photon"]
    assert all(isinstance(d, Unspecified) for _l, _pid, _m, d in entries)

    _e, entries, _p, _r = _roundTrip(endf.mf[6].mt[800], 800)
    (_label, _pid, _multiplicity, form), = entries
    assert isinstance(form, AngularTwoBody)
    assert isinstance(form.angular, Isotropic2d)


def test_a_negative_law_gets_no_product_and_is_declared():
    """U-235's MT18 is NK=56 and two products, and the 54 are the point.

    Forty are ``ZAP=0, LIP=1..40, LAW=-15`` and fourteen ``ZAP=1, LIP=1..14,
    LAW=-5``: pointers into MF15 and MF5, not products. The distributed GNDS
    translation of the same evaluation carries the same two.
    """
    endf = read_endf("kika/endf/tests/data/micro_u235_mf6.endf", mf_numbers=[6])
    section = endf.mf[6].mt[18]
    assert section.num_products == 56

    _encoded, entries, provenance, report = _roundTrip(section, 18)
    assert len(entries) == 2
    records = provenance.headerFields["mf6"]["products"]
    assert len(records) == 56
    assert sum(1 for r in records if r["label"] is None) == 54
    assert sum(1 for m in report.unsupported if "defers to" in m) == 54


def test_a_recoil_points_at_the_two_body_product_above_it():
    """LAW=4 is "the recoil of the product above", so the href is read, not guessed."""
    endf = read_endf("kika/endf/tests/data/micro_be9_mf6.endf", mf_numbers=[6])
    _encoded, entries, _provenance, _report = _roundTrip(endf.mf[6].mt[800], 800)
    byLabel = {label: form for label, _pid, _m, form in entries}
    ejectile = [label for label, form in byLabel.items()
                if isinstance(form, AngularTwoBody) and form.angular is not None]
    recoils = [form for form in byLabel.values()
               if isinstance(form, AngularTwoBody) and form.isRecoil]
    assert len(ejectile) == 1 and len(recoils) == 1
    assert recoils[0].recoilHref == recoilAngularHref(ejectile[0])


# ---------------------------------------------------------------------------
# The frame, which LCT=3 makes a per-product question
# ---------------------------------------------------------------------------

def test_lct_three_splits_the_frame_on_the_product_mass():
    """C-12's MT5: n/H1/H2/He4 in the centre of mass, Li6 upwards in the lab.

    The split is ``A <= 4`` and the numbers are the distributed GNDS file's —
    ``n-006_C_012.endf.gnds.xml`` writes ``productFrame`` exactly this way,
    photon included, which is why the photon is not special-cased.
    """
    endf = read_endf("kika/endf/tests/data/micro_c12_mf6.endf", mf_numbers=[6])
    section = endf.mf[6].mt[5]
    assert section._lct == 3

    frames = {pidFromZA(p.za, p.lip): frameForProduct(3, p.za)
              for p in section.products}
    assert frames["n"] is Frame.centerOfMass
    assert frames["H1"] is Frame.centerOfMass
    assert frames["He4"] is Frame.centerOfMass
    assert frames["photon"] is Frame.centerOfMass
    assert frames["Li6"] is Frame.lab
    assert frames["C12"] is Frame.lab


@pytest.mark.parametrize("lct,expected", [(1, Frame.lab), (2, Frame.centerOfMass)])
def test_lct_one_and_two_are_the_whole_sections_answer(lct, expected):
    for zap in (0, 1, 1001, 26056):
        assert frameForProduct(lct, zap) is expected


# ---------------------------------------------------------------------------
# The provenance
# ---------------------------------------------------------------------------

def test_the_provenance_keeps_the_header_fields_that_have_no_gnds_node():
    endf = read_endf("kika/endf/tests/data/micro_u235_mf6.endf", mf_numbers=[6])
    _e, _entries, provenance, _r = _roundTrip(endf.mf[6].mt[18], 18)
    block = provenance.headerFields["mf6"]
    # JP is 0 on 12 387 of the library's 12 388 sections and 11 here.
    assert block["jp"] == 11
    assert block["lct"] == 1
    assert block["nk"] == 56
    assert block["mat"] == 9228


def test_mf6_keeps_its_own_header_and_does_not_share_mf3s():
    """One reaction, one provenance, and the two files under separate keys.

    Two ENDF/B-VIII.1 tapes state a different AWR in MF5 than in MF4, so "they
    describe the same material, it cannot matter which copy is kept" is false
    and a byte gate finds it. MF6's copy lives inside its own block.
    """
    suite, _ = decodeReactionSuite(
        read_endf("kika/endf/tests/data/micro_c12_mf6.endf"))
    reaction = suite.findReactionByENDF_MT(5)
    header = reaction.provenance.headerFields
    assert "mf6" in header
    assert set(header["mf6"]) >= {"mat", "za", "awr", "jp", "lct", "nk",
                                  "pad", "products"}
    # MF3's own fields are on the provenance itself, not inside the MF6 block.
    assert reaction.provenance.qm is not None


def test_the_verbatim_body_is_text_and_not_a_parsed_endf_object():
    """The model must not grow an accidental format surface.

    LAW=5 is the one law with a body that does not reach a §18 node, so the
    charged-particle fixture is the only place left where this can be asked.
    """
    endf = read_endf("kika/endf/tests/data/micro_p_he3_mf6.endf", mf_numbers=[6])
    _e, _entries, provenance, _r = _roundTrip(endf.mf[6].mt[2], 2)
    records = provenance.headerFields["mf6"]["products"]
    kept = [r for r in records if not r["modelled"]]
    assert kept, "this fixture has a law the model does not carry yet"
    for record in kept:
        assert record["raw_lines"]
        assert all(isinstance(line, str) for line in record["raw_lines"])
        assert all(len(line) <= 66 for line in record["raw_lines"])


# ---------------------------------------------------------------------------
# The suite the pass builds
# ---------------------------------------------------------------------------

def test_the_channel_gains_a_product_per_subsection():
    suite, _ = decodeReactionSuite(
        read_endf("kika/endf/tests/data/micro_c12_mf6.endf"))
    channel = suite.findReactionByENDF_MT(5).outputChannel
    assert len(channel.products) == 21
    assert channel.genre == "sumOfRemainingOutputChannels"
    assert [p.pid for p in channel.products][:4] == ["n", "H1", "H2", "He4"]
    assert channel.products.byPid("photon")


def test_a_two_body_section_says_so_and_an_nbody_one_does_not():
    suite, _ = decodeReactionSuite(
        read_endf("kika/endf/tests/data/micro_li6_mf6.endf"))
    assert suite.findReactionByENDF_MT(52).outputChannel.genre == "twoBody"
    assert suite.findReactionByENDF_MT(103).outputChannel.genre == "twoBody"
    # MT41 is LAW=6, an n-body phase space, and its products' energies are not
    # fixed by the angle.
    assert suite.findReactionByENDF_MT(41).outputChannel.genre == "NBody"


def test_an_mf6_section_with_no_mf3_is_declared_and_not_invented():
    """GNDS attaches a distribution to a product of a *reaction*."""
    from kika.endf.model_adapter.decode import _attachEnergyAngleDistributions
    from kika.nuclear_data.model import ConversionReport, ReactionSuite

    endf = read_endf("kika/endf/tests/data/micro_c12_mf6.endf", mf_numbers=[6])
    suite = ReactionSuite(evaluation="", projectile="n", target="C12")
    report = _attachEnergyAngleDistributions(suite, endf.mf[6].mt[5], 5,
                                             ConversionReport())
    assert len(suite.reactions) == 0
    assert any("has no MF3/MT5 to hang from" in m for m in report.losses)


# ---------------------------------------------------------------------------
# ZAP -> pid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("za,lip,pid", [
    (0, 0, "photon"), (1, 0, "n"), (1001, 0, "H1"), (2004, 0, "He4"),
    (26056, 0, "Fe56"), (3006, 2, "Li6_e2"),
    # LIP on a photon or a neutron is a line index, not an excited state.
    (0, 37, "photon"), (1, 7, "n"),
    # A=0 is ENDF's natural element; GNDS spells it with the symbol alone.
    (26000, 0, "Fe"),
])
def test_pid_from_za(za, lip, pid):
    assert pidFromZA(za, lip) == pid
    assert zaFromPid(pid) == za


def test_a_za_outside_the_element_table_is_spelled_rather_than_refused():
    assert pidFromZA(150000) == "ZA150000"
    assert zaFromPid("ZA150000") == 150000


def test_the_target_is_named_the_way_a_gnds_file_names_it():
    """``Fe56``, not ``ZA26056`` — one naming authority for the two roads in."""
    suite, _ = decodeReactionSuite(
        read_endf("kika/endf/tests/data/micro_c12_mf6.endf"))
    assert suite.target == "C12"
    assert "C12" in suite.PoPs


# ---------------------------------------------------------------------------
# LAW=1 LANG=2 — KalbachMann
# ---------------------------------------------------------------------------

def test_kalbach_mann_is_f_and_r_and_no_a():
    """C-12's MT5: four LANG=2 products, and the library has zero ``<a>``.

    ENDF-6 lets an evaluator write only the pre-compound fraction and leave the
    slope to Kalbach's systematics, which is what all 3 730 nodes in
    ENDF/B-VIII.1 do. ``kalbach()`` returns NaN rather than 0.0 there, and this
    is the test that those NaNs do not reach the model.
    """
    endf = read_endf("kika/endf/tests/data/micro_c12_mf6.endf", mf_numbers=[6])
    section = endf.mf[6].mt[5]
    _encoded, entries, _provenance, _report = _roundTrip(section, 5)

    kalbach = [form for _l, _p, _m, form in entries
               if isinstance(form, KalbachMann)]
    assert len(kalbach) == 4
    for form in kalbach:
        assert form.isComplete
        assert form.a is None
        assert form.productFrame is Frame.centerOfMass
        assert len(form.f.function1ds) == len(form.r.function1ds) == 19
        # LEP=1 on this tape: the outgoing spectrum is a histogram, which is
        # what the distributed GNDS file spells interpolation="flat".
        assert form.f.function1ds[0].interpolation is Interpolation.flat


def test_the_kalbach_columns_are_the_records_own():
    endf = read_endf("kika/endf/tests/data/micro_c12_mf6.endf", mf_numbers=[6])
    section = endf.mf[6].mt[5]
    body = section.products[0].law_data
    assert body.lang == 2

    _encoded, entries, _provenance, _report = _roundTrip(section, 5)
    form = entries[0][3]
    for k in range(len(body.incident_energies)):
        e_out, r, _a = body.kalbach(k)
        assert form.f.function1ds[k].xs == pytest.approx(e_out)
        assert form.r.function1ds[k].ys == pytest.approx(r)


def test_the_kalbach_axes_are_shared_by_identity_within_each_half():
    """``kika/gnds/encode.py:_axesUnlessNested`` decides inheritance by identity."""
    endf = read_endf("kika/endf/tests/data/micro_c12_mf6.endf", mf_numbers=[6])
    _e, entries, _p, _r = _roundTrip(endf.mf[6].mt[5], 5)
    form = entries[0][3]
    for half, unit in (("f", "1/eV"), ("r", "")):
        container = getattr(form, half)
        assert container.axes is not None
        dependent = container.axes.axes[-1]
        assert (dependent.index, dependent.label, dependent.unit) == (0, half, unit)
        for child in container.function1ds:
            assert child.axes is container.axes


# ---------------------------------------------------------------------------
# LAW=1 LANG=1 — the NA split
# ---------------------------------------------------------------------------

def test_a_lang_one_product_with_no_angular_coefficients_is_uncorrelated():
    """The correction the counts forced: LANG=1 is not ``energyAngular``.

    18 738 LANG=1 products in ENDF/B-VIII.1 against 2 948 ``energyAngular`` in
    its GNDS translation, and ``NA`` is the difference. C-12's MT5 states
    seventeen LANG=1 products, every one ``NA=0``, and the distributed
    ``n-006_C_012.endf.gnds.xml`` carries no ``energyAngular`` at all.
    """
    endf = read_endf("kika/endf/tests/data/micro_c12_mf6.endf", mf_numbers=[6])
    section = endf.mf[6].mt[5]
    lang1 = [p for p in section.products if p.law_data.lang == 1]
    assert len(lang1) == 17
    assert all(max(p.law_data.na) == 0 for p in lang1)

    _e, entries, _p, _r = _roundTrip(section, 5)
    uncorrelated = [f for _l, _pid, _m, f in entries if isinstance(f, Uncorrelated)]
    assert len(uncorrelated) == 17
    assert not [f for _l, _pid, _m, f in entries if isinstance(f, EnergyAngular)]
    for form in uncorrelated:
        assert isinstance(form.angular, Isotropic2d)
        assert form.isComplete


def test_the_energy_half_of_an_na_zero_product_is_the_f0_column():
    endf = read_endf("kika/endf/tests/data/micro_c12_mf6.endf", mf_numbers=[6])
    section = endf.mf[6].mt[5]
    _e, entries, _p, _r = _roundTrip(section, 5)
    index = next(i for i, p in enumerate(section.products)
                 if p.law_data.lang == 1)
    body = section.products[index].law_data
    form = entries[index][3]
    for k in range(len(body.incident_energies)):
        e_out, f0 = body.spectrum(k)
        assert form.energy.function1ds[k].xs == pytest.approx(e_out)
        assert form.energy.function1ds[k].ys == pytest.approx(f0)


def test_the_three_dimensional_tab2_refuses_more_than_one_region():
    """``gnds.xsd`` declares no element of type ``xData_regions_3d_primary``.

    So an ENDF TAB2 with NR>1 at that level has no legal node, and flattening
    the regions would silently drop where the interpolation law changes. The
    branch is unreachable on the data measured — every LAW=1 and LAW=7 TAB2 in
    the library writes one plain ``INT=2`` — and this is what says so if that
    stops being true.
    """
    from kika.nuclear_data.model import XYs2d, fromEndfTab3, toEndfTab3

    nodes = [XYs2d(function1ds=[], outerDomainValue=float(i)) for i in range(4)]
    surface = fromEndfTab3(nodes, [(4, 2)])
    assert len(surface.function2ds) == 4
    assert toEndfTab3(surface) == (nodes, [(4, 2)])

    with pytest.raises(ValueError, match="no regions3d"):
        fromEndfTab3(nodes, [(2, 1), (4, 2)])


# ---------------------------------------------------------------------------
# LAW=7 — angularEnergy
# ---------------------------------------------------------------------------

def test_law_seven_is_an_angular_energy_and_not_its_mirror():
    """Be-9's MT16 carries the only two ``angularEnergy`` in the library.

    ``energyAngular`` and ``angularEnergy`` share a complexType exactly, so the
    element name is the only thing in a written file that says which variable is
    outermost. LAW=7 puts mu outside E', and the axes have to say so too.
    """
    endf = read_endf("kika/endf/tests/data/micro_be9_mf6.endf", mf_numbers=[6])
    section = endf.mf[6].mt[16]
    body = section.products[0].law_data
    assert body.law == 7

    _e, entries, _p, _r = _roundTrip(section, 16)
    forms = [f for _l, _pid, _m, f in entries]
    assert len(forms) == 2
    for form in forms:
        assert isinstance(form, AngularEnergy)
        assert not isinstance(form, EnergyAngular)
        assert form.isComplete
        labels = [axis.label for axis in form.xys3d.axes.axes]
        assert labels == ["energy_in", "mu", "energy_out",
                          "P(mu,energy_out|energy_in)"]
        # LCT=1 on this section, so everything is in the laboratory.
        assert form.productFrame is Frame.lab


def test_the_law_seven_surface_keeps_all_three_levels():
    endf = read_endf("kika/endf/tests/data/micro_be9_mf6.endf", mf_numbers=[6])
    section = endf.mf[6].mt[16]
    body = section.products[0].law_data
    _e, entries, _p, _r = _roundTrip(section, 16)
    surface = entries[0][3].xys3d

    assert len(surface.function2ds) == len(body.blocks)
    for k, block in enumerate(body.blocks):
        node = surface.function2ds[k]
        assert node.outerDomainValue == pytest.approx(block.energy)
        assert len(node.function1ds) == len(block.angles)
        for m, angle in enumerate(block.angles):
            e_out, f = body.table(k, m)
            child = node.function1ds[m]
            assert child.outerDomainValue == pytest.approx(angle.mu)
            assert child.toEndfRegions()[0] == pytest.approx(e_out)
            assert child.toEndfRegions()[1] == pytest.approx(f)


# ---------------------------------------------------------------------------
# LAW=2 and LAW=6
# ---------------------------------------------------------------------------

def test_law_two_is_an_angular_two_body_with_the_implicit_a0_restored():
    endf = read_endf("kika/endf/tests/data/micro_li6_mf6.endf", mf_numbers=[6])
    section = endf.mf[6].mt[52]
    _e, entries, _p, _r = _roundTrip(section, 52)

    ejectile = entries[0][3]
    assert isinstance(ejectile, AngularTwoBody)
    body = section.products[0].law_data
    assert body.law == 2 and set(body.lang) == {0}
    for k in range(len(body.incident_energies)):
        child = ejectile.angular.function1ds[k]
        # ENDF writes A_1..A_NL; A_0 is 1 by normalisation and GNDS states it.
        assert child.coefficients[0] == 1.0
        assert child.coefficients[1:] == pytest.approx(body.legendre(k))


def test_law_two_lang_twelve_is_a_tabulated_cosine():
    """The only LANG=12 witness on this machine: ``t+Li7`` MT50."""
    endf = read_endf("kika/endf/tests/data/micro_t_li7_mf6.endf", mf_numbers=[6])
    section = endf.mf[6].mt[50]
    body = section.products[0].law_data
    assert body.law == 2 and set(body.lang) == {12}

    _e, entries, _p, _r = _roundTrip(section, 50)
    angular = entries[0][3].angular
    for k in range(len(body.incident_energies)):
        mu, f = body.tabulated(k)
        child = angular.function1ds[k]
        # LANG = 10 + INT, so 12 is lin-lin.
        assert child.interpolation is Interpolation.linlin
        assert child.xs == pytest.approx(mu)
        assert child.ys == pytest.approx(f)


def test_law_six_is_an_uncorrelated_holding_an_n_body_phase_space():
    """Li-6's MT41, three of the library's five LAW=6 products.

    The distributed GNDS translation writes an ``isotropic2d`` angular half and
    an ``NBodyPhaseSpace`` energy half with ``numberOfProducts`` and no mass.
    """
    from kika.nuclear_data.model import NBodyPhaseSpace

    endf = read_endf("kika/endf/tests/data/micro_li6_mf6.endf", mf_numbers=[6])
    section = endf.mf[6].mt[41]
    _e, entries, _p, _r = _roundTrip(section, 41)
    assert len(entries) == 3
    for index, (_label, _pid, _m, form) in enumerate(entries):
        assert isinstance(form, Uncorrelated)
        assert isinstance(form.angular, Isotropic2d)
        assert isinstance(form.energy, NBodyPhaseSpace)
        assert form.energy.numberOfProducts == section.products[index].law_data.npsx
        # APSX is in units of the neutron mass; putting it in `mass` would mean
        # choosing a neutron mass and writing a number the evaluator did not.
        assert form.energy.mass is None


def test_apsx_survives_in_the_provenance_and_not_in_the_model():
    endf = read_endf("kika/endf/tests/data/micro_li6_mf6.endf", mf_numbers=[6])
    section = endf.mf[6].mt[41]
    _e, _entries, provenance, _r = _roundTrip(section, 41)
    records = provenance.headerFields["mf6"]["products"]
    assert all(r["law_fields"]["apsx"] == pytest.approx(section.products[i].law_data.apsx)
               for i, r in enumerate(records))


# ---------------------------------------------------------------------------
# LAW=5 stays outside the model, and says so
# ---------------------------------------------------------------------------

def test_law_five_is_kept_verbatim_and_declared():
    """0 of 558 GNDS evaluations carry a ``CoulombPlusNuclearElastic``.

    So there is nothing to map it onto that any file has ever used, and the
    tape comes back through the provenance instead.
    """
    endf = read_endf("kika/endf/tests/data/micro_p_he3_mf6.endf", mf_numbers=[6])
    section = endf.mf[6].mt[2]
    _e, entries, provenance, report = _roundTrip(section, 2)

    records = provenance.headerFields["mf6"]["products"]
    law5 = [r for r in records if r["law"] == 5]
    assert len(law5) == 1
    assert not law5[0]["modelled"] and law5[0]["raw_lines"]
    assert any("LAW=5" in m for m in report.unsupported)


def test_ti50_states_both_halves_of_the_na_split_in_one_section():
    """The only ``energyAngular`` reachable offline, and the mixed case with it.

    Ti-50's MT17 is 27 lines and two products. The neutron has ``NA=0`` at the
    first incident energy and ``NA=4`` at the second — so it is one
    ``energyAngular`` whose first node carries a single Legendre coefficient
    and whose second carries five. The photon is ``NA=0`` throughout and is an
    ``uncorrelated``.

    The distributed ``n-022_Ti_050.endf.gnds.xml`` says exactly that, node for
    node, which is what makes this a check and not a restatement of the code.
    """
    endf = read_endf("kika/endf/tests/data/micro_ti50_mf6.endf", mf_numbers=[6])
    section = endf.mf[6].mt[17]
    assert [p.law_data.na for p in section.products] == [[0, 4], [0, 0]]

    _encoded, entries, _provenance, _report = _roundTrip(section, 17)
    byLabel = {label: form for label, _pid, _m, form in entries}
    assert isinstance(byLabel["n"], EnergyAngular)
    assert isinstance(byLabel["photon"], Uncorrelated)

    nodes = byLabel["n"].xys3d.function2ds
    assert len(nodes) == 2
    assert [len(node.function1ds) for node in nodes] == [3, 8]
    assert [len(node.function1ds[0].coefficients) for node in nodes] == [1, 5]


def test_the_energy_angular_axes_say_which_variable_is_outermost():
    endf = read_endf("kika/endf/tests/data/micro_ti50_mf6.endf", mf_numbers=[6])
    _e, entries, _p, _r = _roundTrip(endf.mf[6].mt[17], 17)
    surface = next(f for _l, _pid, _m, f in entries
                   if isinstance(f, EnergyAngular)).xys3d
    assert [axis.label for axis in surface.axes.axes] == [
        "energy_in", "energy_out", "mu", "P(energy_out,mu|energy_in)"]
    # And the qualifier is **read**, not added. This section's TAB2 writes
    # `INT=22` — unit base, lin-lin — so the evaluation does say how to
    # interpolate between incident nodes, and the distributed GNDS file carries
    # `interpolationQualifier="unitbase"` because kika's source did too. That
    # corrects `mf6_notes.md`, which said no measured LAW=1 TAB2 used one.
    from kika.nuclear_data.model import InterpolationQualifier

    assert surface.interpolationQualifier is InterpolationQualifier.unitBase
    assert endf.mf[6].mt[17].products[0].law_data.tab2_interp == [(2, 22)]


def test_a_qualified_tab2_code_survives_the_round_trip():
    """``INT=22`` is the decade, and losing it would be silent.

    ``fromEndfTab2``/``fromEndfTab3`` split the code into a rule and a
    qualifier and ``endfInterpolationCode`` puts them back. Six of the 28 LAW=1
    TAB2s in the committed fixtures carry one, so a decade dropped on the way
    in would come back out as a plain ``2`` and the section would still look
    right everywhere except this line.
    """
    for name, mt in (("micro_ti50_mf6", 17), ("micro_t_li7_mf6", 24)):
        endf = read_endf(f"kika/endf/tests/data/{name}.endf", mf_numbers=[6])
        section = endf.mf[6].mt[mt]
        qualified = [p for p in section.products
                     if any(code > 10 for _nbt, code in p.law_data.tab2_interp)]
        assert qualified, f"{name} MT{mt} was chosen for its qualified code"
        encoded, *_ = _roundTrip(section, mt)
        assert str(encoded) == str(section)


@pytest.mark.tape
@pytest.mark.parametrize("tape,mts", [
    ("n-006_C_012", (5,)),
    ("n-003_Li_006", (41, 52, 103)),
    ("n-022_Ti_050", (17,)),
])
def test_the_form_census_matches_the_distributed_gnds_translation(tape, mts):
    """The oracle: count the same nodes FUDGE counted, off the same evaluation.

    ENDF-B-VIII.1-GNDS is the distributed translation of the library this
    adapter reads, produced by a second implementation. Comparing *numbers* per
    form per MT is what catches the mapping being inverted — an
    ``energyAngular`` written where an ``uncorrelated`` belongs still validates,
    still round-trips to ENDF, and is a different statement about the physics.

    Counts, not values: the three numeric oracles already compare those to
    1e-14, and FUDGE reshapes data on the way in (it adds a residual product to
    Ti-50's MT17 that MF6 does not state, and resolves Be-9's MT701 residual to
    an excited level from the MT number). What must agree is which §18 form each
    subsection became.
    """
    import os
    from collections import Counter
    from pathlib import Path
    import xml.etree.ElementTree as ET

    root = Path(os.environ.get("KIKA_TAPES", "/share_snc/snc/JuanMonleon"))
    source = root / "ENDF-B-VIII.1-GNDS" / "ENDF-B-VIII.1-GNDS" / "neutrons"
    endfPath = Path("/share_snc/lib/endf/endfb81") / f"{tape}.endf"
    gndsPath = source / f"{tape}.endf.gnds.xml"
    if not (endfPath.is_file() and gndsPath.is_file()):
        pytest.skip(f"needs {endfPath} and {gndsPath}")

    endf = read_endf(str(endfPath), mf_numbers=[6])
    suiteRoot = ET.parse(gndsPath).getroot()

    for mt in mts:
        _encoded, entries, _provenance, _report = _roundTrip(endf.mf[6].mt[mt], mt)
        ours = Counter(type(form).__name__ for _l, _p, _m, form in entries
                       if form is not None)

        theirs = Counter()
        for reaction in suiteRoot.iter("reaction"):
            if reaction.get("ENDF_MT") != str(mt):
                continue
            for product in reaction.find("outputChannel").find("products"):
                distribution = product.find("distribution")
                for form in (distribution if distribution is not None else ()):
                    theirs[form.tag] = theirs[form.tag] + 1

        for ourName, theirName in (("Uncorrelated", "uncorrelated"),
                                   ("EnergyAngular", "energyAngular"),
                                   ("AngularEnergy", "angularEnergy"),
                                   ("KalbachMann", "KalbachMann"),
                                   ("AngularTwoBody", "angularTwoBody")):
            assert ours[ourName] == theirs[theirName], (
                f"{tape} MT{mt} {theirName}: kika {ours[ourName]}, "
                f"FUDGE {theirs[theirName]}")


def test_two_products_of_one_particle_get_two_nodes():
    """``ensureProduct`` matches on the label, and it did not until MF6 arrived.

    Alpha + He4 MT2 is elastic scattering off an identical particle, so the
    ejectile (LAW=5) and its recoil (LAW=4) are the same nuclide with the same
    ``LIP``. Matching on ``pid`` alone handed both subsections the first
    product: the second's multiplicity was refused as a collision with a nu-bar
    that was not there, and its distribution would have overwritten the first's.
    """
    import kika

    for name, mt, labels in (
            ("micro_a_he4_mf6", 2, ["He4", "He4__1"]),
            ("micro_d_h2_mf6", 2, ["H2", "H2__1"]),
            ("micro_t_li7_mf6", 24, ["n", "He4", "He4__2"]),
    ):
        suite = kika.read(f"kika/endf/tests/data/{name}.endf")
        products = list(suite.findReactionByENDF_MT(mt).outputChannel.products)
        assert [p.label for p in products] == labels, name
        assert all(p.multiplicity is not None for p in products), name
        # And no product was mistaken for one another file had already filled.
        assert not suite.report.warnings, name


def test_ensure_product_still_finds_the_nubars_neutron():
    """The fallback the label match must not break.

    MF1's nu-bar puts a product on the fission channel before MF6 runs, and
    both call it ``n`` — so MF6's neutron has to *be* that product and not a
    second one. It is the case ``ensureProduct`` was written for.
    """
    from kika.nuclear_data.model import Multiplicity, OutputChannel

    channel = OutputChannel()
    nubar = channel.ensureProduct("n")
    nubar.multiplicity = Multiplicity(form=None)

    assert channel.ensureProduct("n", "n") is nubar
    assert channel.ensureProduct("n") is nubar
    # A different label is a different node, even for the same particle.
    second = channel.ensureProduct("n", "n__3")
    assert second is not nubar
    assert (second.pid, second.label) == ("n", "n__3")
    assert len(channel.products) == 2


def test_an_mf6_tape_writes_a_gnds_file_with_no_empty_distribution(
        micro_mf6_tape, tmp_path):
    """The gate §6.1 named, reached from the ENDF side.

    A product whose law kika cannot read is written with an **empty
    `<distribution/>`** rather than an ``<unspecified/>``, so the file announces
    its own incompleteness and fails schema validation. Phase 7b took that count
    to zero for suites decoded from GNDS; this is the same count for a suite
    decoded from a tape whose distributions are all in File 6.
    """
    import xml.etree.ElementTree as ET

    import kika

    suite = kika.read(str(micro_mf6_tape))
    out = tmp_path / "written.gnds.xml"
    kika.write(suite, out, format="gnds")

    root = ET.parse(out).getroot()
    distributions = list(root.iter("distribution"))
    assert distributions, "this tape's products carry no distribution at all"
    assert not [d for d in distributions if len(d) == 0]


def test_the_forms_survive_endf_to_gnds_and_back(micro_mf6_tape, tmp_path):
    """`kika/gnds` already read and wrote all five of these before MF6 arrived.

    That is a claim worth checking rather than repeating: the registry says
    ``energyAngular``, ``angularEnergy``, ``KalbachMann``, ``uncorrelated`` and
    ``angularTwoBody`` are all ``PAIRED``, and until this adapter landed nothing
    could put an ENDF-decoded one of the first three in front of it.
    """
    from collections import Counter

    import kika

    def formsOf(suite):
        counted = Counter()
        for reaction in suite.reactions:
            for product in reaction.outputChannel.products:
                for label in (product.distribution or ()):
                    counted[type(product.distribution[label]).__name__] += 1
        return counted

    fromTape = kika.read(str(micro_mf6_tape))
    out = tmp_path / "written.gnds.xml"
    kika.write(fromTape, out, format="gnds")
    fromXml = kika.read(out)

    assert formsOf(fromXml) == formsOf(fromTape)


@pytest.mark.tape
@pytest.mark.parametrize("fixture", ["be9_b81_tape", "c12_b81_tape",
                                     "li6_b81_tape", "ti50_b81_tape"])
def test_every_mf6_section_of_a_real_tape_encodes_byte_identically(
        fixture, request):
    """The fixtures are cuts; these are the whole evaluations they were cut from.

    Measured wider than this at the time it landed — 254 MF6 sections and 405
    products over 33 ENDF/B-VIII.1 tapes under 900 kB, zero differing — and four
    tapes is what is worth paying for on every ``-m tape`` run.
    """
    path = request.getfixturevalue(fixture)
    sections = read_endf(str(path), mf_numbers=[6]).mf[6].mt
    assert sections
    for mt, section in sorted(sections.items()):
        encoded, *_ = _roundTrip(section, mt)
        assert str(encoded) == str(section), f"{fixture} MT{mt}"
