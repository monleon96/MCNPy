"""The ``reactionSuite`` reader, against the two committed evaluations.

The pair is chosen so that between them they carry every node
:mod:`kika.gnds.decode` handles. H-2 has ``externalFiles``, ``sums``,
``productions``, ``applicationData``, a ``regions1d`` cross section and a
``reference`` one. Fe-56's trim has ``resonancesWithBackground`` with two
background regions, a second style, a ``recoil`` angular distribution and a
``regions2d`` one. Neither has ``fissionComponents``, ``orphanProducts`` or
``incompleteReactions``; those paths are the same three lines of
:data:`~kika.gnds.decode.REACTION_LISTS` and are exercised by the library sweep
in :mod:`kika.gnds.tests.test_cross_section_oracle`.

**What is deliberately not asserted here.** That the numbers are *right*. These
tests say a file was read the way it was written; saying it was read correctly
needs a second, independent encoding of the same evaluation, which is what the
oracle module is for.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from kika.gnds.decode import readReactionSuite
from kika.gnds.xpath import Document
from kika.nuclear_data import model as m
from kika.nuclear_data.model import (AngularTwoBody, Background, Constant1d,
                                     CrossSectionSum, Evaluated, Isotropic2d,
                                     Nuclide, Reference, Regions1d, Regions2d,
                                     ResonancesWithBackground, Unspecified,
                                     XYs1d)


@pytest.fixture(scope="module")
def h2(h2_gnds):
    return readReactionSuite(Document.parse(h2_gnds))


@pytest.fixture(scope="module")
def fe56(micro_fe56_gnds):
    return readReactionSuite(Document.parse(micro_fe56_gnds))


# ---------------------------------------------------------------------------
# the suite
# ---------------------------------------------------------------------------

def test_the_root_attributes_are_read(h2):
    suite, _ = h2
    assert (suite.projectile, suite.target) == ("n", "H2")
    assert suite.evaluation == "ENDF/B-8.1"
    assert suite.format == "2.0"
    assert suite.projectileFrame == "lab"
    assert suite.interaction == "nuclear"


def test_a_covariance_suite_is_refused_by_this_reader(h2_gnds_cov):
    """The two roots are different documents and the error says which is which."""
    with pytest.raises(ValueError, match="is a <covarianceSuite>, not a <reactionSuite>"):
        readReactionSuite(Document.parse(h2_gnds_cov))


def test_the_report_is_hung_on_the_suite_as_well_as_returned(h2):
    """`suite.report` is how a caller who dropped the tuple still finds out."""
    suite, report = h2
    assert suite.report is report
    assert not report.isClean  # §18's uncorrelated laws are phase 7b


# ---------------------------------------------------------------------------
# externalFiles and styles
# ---------------------------------------------------------------------------

def test_the_external_file_and_its_digest_are_read(h2):
    suite, report = h2
    entry = suite.externalFiles.byLabel("covariances")
    assert entry.path.endswith("n-001_H_002.endf.gnds-covar.xml")
    assert entry.algorithm == "sha1" and len(entry.checksum) == 40
    # The sibling is committed beside it, so the digest was actually checked.
    assert not report.warnings, report.warnings


def test_a_broken_digest_is_a_warning_and_not_a_refusal(tmp_path, h2_gnds,
                                                        h2_gnds_cov):
    """A mismatched pair still parses; what it has lost is the guarantee.

    Refusing here would make a hand-trimmed fixture unreadable, and a trim is a
    legitimate thing to do to a file. The warning says the two were not written
    together, which is the whole content of a checksum.
    """
    covariances = tmp_path / "Covariances"
    covariances.mkdir()
    (covariances / h2_gnds_cov.name).write_bytes(
        h2_gnds_cov.read_bytes() + b"<!-- edited -->"
    )
    copy = tmp_path / h2_gnds.name
    copy.write_bytes(h2_gnds.read_bytes())

    suite, report = readReactionSuite(Document.parse(copy))
    assert len(suite.reactions) == 3
    assert any("has been edited" in w for w in report.warnings), report.warnings


def test_the_evaluated_style_keeps_its_temperature_and_domain(fe56):
    suite, _ = fe56
    evaluated = suite.styles["eval"]
    assert isinstance(evaluated, Evaluated)
    assert (evaluated.library, evaluated.version) == ("ENDF/B", "8.1.1")
    assert evaluated.temperature.value == 0.0
    assert evaluated.temperature.unit == "K"
    assert evaluated.projectileEnergyDomain.max == 1.5e8
    assert 1.0 in evaluated.projectileEnergyDomain


def test_the_reconstructed_style_names_what_it_derives_from(fe56):
    """§9.1's own example, read off a real file: two styles, one chain."""
    suite, _ = fe56
    assert suite.styles.labels == ["eval", "recon"]
    assert suite.styles["recon"].derivedFrom == "eval"
    assert suite.styles.evaluatedFor("recon").label == "eval"


def test_documentation_is_declared_lost_rather_than_dropped(fe56):
    _, report = fe56
    assert any("style <documentation>" in loss for loss in report.losses)


# ---------------------------------------------------------------------------
# PoPs
# ---------------------------------------------------------------------------

def test_the_particles_come_out_with_their_masses_and_spins(fe56):
    suite, _ = fe56
    assert set(suite.PoPs) == {"photon", "n", "p", "Fe56", "Fe57"}
    assert suite.PoPs["n"].spin.value == 0.5      # written "1/2"
    assert suite.PoPs["n"].spin.unit == "hbar"
    assert suite.PoPs["n"].mass.value == pytest.approx(1.00866491574)


def test_a_nuclide_carries_its_Z_and_A_from_the_nodes_that_group_it(fe56):
    """Z is on ``chemicalElement`` and A on ``isotope``, six levels up."""
    suite, _ = fe56
    fe56Nuclide = suite.PoPs["Fe56"]
    assert isinstance(fe56Nuclide, Nuclide)
    assert (fe56Nuclide.Z, fe56Nuclide.A, fe56Nuclide.ZA) == (26, 56, 26056)
    assert fe56Nuclide.nuclearLevel == 0
    assert fe56Nuclide.spin.value == 0.0   # from <nucleus>, not from <nuclide>


def test_what_the_minimal_pops_drops_is_counted_not_listed(fe56):
    """One aggregated entry per kind. A report with 171 lines declares nothing."""
    _, report = fe56
    aliases = [loss for loss in report.losses if "PoPs <alias>" in loss]
    assert len(aliases) == 1 and aliases[0].startswith("4 x ")


# ---------------------------------------------------------------------------
# reactions and cross sections
# ---------------------------------------------------------------------------

def test_reactions_are_keyed_by_their_gnds_label_and_by_MT(fe56):
    suite, _ = fe56
    assert suite.reactions.ENDF_MTs == [2, 102]
    assert suite.reactions["n + Fe56"] is suite.reactions[2]
    # The label is the file's, not "MT2" — that is the ENDF adapter's invention
    # for a format that has no semantic label.
    assert suite.reactions[102].label == "Fe57 + photon [inclusive]"


def test_a_reactions_identity_carries_the_products_it_actually_has(fe56):
    suite, _ = fe56
    assert suite.reactions[102].id.products == ("Fe57", "photon")


def test_each_cross_section_form_lands_under_its_style_label(fe56):
    suite, _ = fe56
    forms = suite.reactions[2].crossSection
    assert sorted(forms) == ["eval", "recon"]
    assert isinstance(forms["eval"], ResonancesWithBackground)
    assert isinstance(forms["recon"], XYs1d)


def test_a_regions1d_cross_section_keeps_its_regions(h2):
    suite, _ = h2
    form = suite.reactions[102].crossSection["eval"]
    assert isinstance(form, Regions1d)
    assert len(form.function1ds) == 2
    assert form.function1ds[0].interpolation == "log-log"


def test_a_reference_form_is_kept_as_the_link_it_is(h2):
    """§16.1.1's ``reference``: kika does not follow it and does not pretend to.

    Dereferencing here would copy a cross section into a second reaction and
    lose the statement that the two are the *same* one — which is the only
    thing the node says.
    """
    suite, _ = h2
    form = suite.productions[102].crossSection["eval"]
    assert isinstance(form, Reference)
    assert form.href.startswith("/reactionSuite/reactions/reaction[@label=")


def test_the_background_is_three_regions_and_not_one_curve(fe56):
    suite, _ = fe56
    background = suite.reactions[2].crossSection["eval"].background
    assert isinstance(background, Background)
    assert len(background) == 2
    assert isinstance(background.resolvedRegion, XYs1d)
    assert isinstance(background.fastRegion, XYs1d)
    assert background.unresolvedRegion is None      # Fe-56 has no URR
    # The two regions are separate curves over separate domains, which is the
    # whole reason `background` is not one function. The trim keeps 20 points of
    # each, so the resolved one stops short of the 850 keV boundary; the full
    # file's boundary is checked against ENDF in the oracle module.
    assert background.resolvedRegion.xs[0] == 1e-5
    assert background.fastRegion.xs[0] == 850000.0
    assert background.resolvedRegion.xs[-1] < background.fastRegion.xs[0]


def test_the_resonance_link_is_kept_so_the_background_stays_attached(fe56):
    suite, _ = fe56
    form = suite.reactions[2].crossSection["eval"]
    assert form.resonanceRegionHref == "/reactionSuite/resonances"


def test_the_covariance_back_link_on_a_form_is_declared_lost(h2):
    """§7's ``uncertainty`` hangs on the form, and the model has no slot."""
    _, report = h2
    assert any("<XYs1d><uncertainty>" in loss for loss in report.losses)


# ---------------------------------------------------------------------------
# output channels, Q and products
# ---------------------------------------------------------------------------

def test_the_channel_genre_and_process_are_read(fe56):
    suite, _ = fe56
    assert suite.reactions[2].outputChannel.isTwoBody
    channel = suite.reactions[102].outputChannel
    assert (channel.genre, channel.process) == ("NBody", "inclusive")


def test_q_takes_its_unit_from_the_axis_and_not_from_a_guess(fe56):
    """§2.3.3: a unit lives with an axis or a physicalQuantity, never on a node."""
    suite, _ = fe56
    q = suite.reactions[102].outputChannel.Q
    assert q.isKnown and q.value == 7646430.0 and q.unit == "eV"


def test_a_products_multiplicity_reads_as_a_constant(fe56):
    suite, _ = fe56
    product = suite.reactions[2].outputChannel.products.byPid("n")[0]
    assert product.multiplicity.constant == 1.0
    assert product.multiplicity.evaluate(1.0e6) == 1.0


def test_an_angular_distribution_lands_under_its_style_label(fe56):
    suite, _ = fe56
    product = suite.reactions[2].outputChannel.products.byPid("n")[0]
    form = product.distribution["eval"]
    assert isinstance(form, AngularTwoBody)
    assert isinstance(form.angular, Regions2d)
    assert form.productFrame == "centerOfMass"
    assert form.energies[0] == 1e-5


def test_the_recoil_product_is_a_link_and_tabulates_nothing(fe56, micro_fe56_gnds):
    """22 540 of the library's 45 080 ``angularTwoBody`` nodes are this shape.

    The href is kept unfollowed, and it is checked here by *following it*, with
    the resolver rather than with a string comparison: the link is relative
    (``../../../../product[@label='n']/…``) and a test that only matched its
    text would pass on a link that reaches the wrong product.
    """
    from kika.gnds.xpath import Resolver

    suite, _ = fe56
    product = suite.reactions[2].outputChannel.products.byPid("Fe56")[0]
    form = product.distribution["eval"]
    assert form.isRecoil
    # And the two accessors say "nothing tabulated" rather than raising.
    assert form.energies == [] and form.function1ds == []

    document = Document.parse(micro_fe56_gnds)
    source = document.root.find(
        ".//reaction[@ENDF_MT='2']/outputChannel/products/product[@pid='Fe56']"
        "/distribution/angularTwoBody/recoil"
    )
    reached = Resolver(document).resolve(form.recoilHref, context=source)
    assert reached, reached.problem
    assert reached.element.tag == "angularTwoBody"
    assert document.parents[document.parents[reached.element]].attrib["pid"] == "n"


def test_an_unspecified_distribution_is_read_as_a_statement(h2):
    """§18's ``unspecified`` is the evaluator saying so, not kika failing to."""
    suite, _ = h2
    product = suite.productions[102].outputChannel.products.byPid("H3")[0]
    assert isinstance(product.distribution["eval"], Unspecified)
    assert product.distribution["eval"].productFrame == "lab"


def test_the_three_uncorrelated_laws_are_read_whole(h2):
    """Phase 7b's first landing, on the fixture that used to report all three.

    H-2's three ``uncorrelated`` distributions were the subject of
    ``test_an_unimplemented_law_is_named_with_its_xpath`` until the law was
    implemented: the report named them and the model held nothing. Now both
    halves arrive, so the assertion is what they *are* rather than that they
    are missing. The naming doctrine keeps its own subject in
    ``test_an_unread_energy_form_is_named_with_its_xpath`` below.
    """
    suite, report = h2
    forms = [form
             for reaction in suite.reactions
             for product in reaction.outputChannel.products
             for form in product.distribution.forms.values()
             if isinstance(form, m.Uncorrelated)]
    assert len(forms) == 3
    assert all(isinstance(f.angular, m.Isotropic2d) for f in forms)

    phaseSpace = [f.energy for f in forms
                  if isinstance(f.energy, m.NBodyPhaseSpace)]
    assert len(phaseSpace) == 2
    assert phaseSpace[0].numberOfProducts == 3
    assert phaseSpace[0].mass.value == pytest.approx(3.02460278964)
    assert phaseSpace[0].mass.unit == "amu"

    gamma = [f.energy for f in forms if isinstance(f.energy, m.PrimaryGamma)]
    assert len(gamma) == 1
    assert gamma[0].value == pytest.approx(6251002.0)
    assert gamma[0].domainMax == pytest.approx(1.5e8)
    # `<axes>` is a required child of the node, so losing it would write an
    # invalid file back out.
    assert gamma[0].axes is not None

    assert not [e for e in report.unsupported if "/uncorrelated:" in e]


def test_an_unread_energy_form_is_named_with_its_xpath(h2_gnds, tmp_path):
    """The naming doctrine, on a subject that survives phase 7b.

    Six of ``uncorrelated/energy``'s eleven choices are analytic spectra
    (``gnds.xsd:1697-1709``) that kika reports rather than tabulates. This
    plants one into H-2 and checks that the report says which node, where, and
    that the angular half is still read — losing that as well would throw away
    something the file did state.
    """
    tree = ET.parse(h2_gnds)
    energy = tree.getroot().find(".//uncorrelated/energy")
    for child in list(energy):
        energy.remove(child)
    ET.SubElement(energy, "evaporation")
    path = tmp_path / "evaporation.gnds.xml"
    tree.write(path)

    suite, report = readReactionSuite(Document.parse(path))
    entries = [e for e in report.unsupported if "/evaporation:" in e]
    assert len(entries) == 1
    assert "analytic §18.3 spectrum" in entries[0]
    assert "gnds.xsd:" in entries[0]

    planted = [form
               for reaction in suite.reactions
               for product in reaction.outputChannel.products
               for form in product.distribution.forms.values()
               if isinstance(form, m.Uncorrelated) and form.energy is None]
    assert len(planted) == 1
    assert isinstance(planted[0].angular, m.Isotropic2d)
    assert planted[0].isComplete is False


def test_a_product_that_loses_its_distribution_keeps_everything_else(h2):
    """The photon of MT102, which used to lose its distribution entirely."""
    suite, _ = h2
    product = suite.reactions[102].outputChannel.products.byPid("photon")[0]
    form = product.distribution["eval"]
    assert isinstance(form, m.Uncorrelated)
    assert isinstance(form.energy, m.PrimaryGamma)
    assert product.multiplicity.constant == 1.0


# ---------------------------------------------------------------------------
# sums
# ---------------------------------------------------------------------------

def test_a_cross_section_sum_keeps_both_its_values_and_its_summands(h2):
    """The two are different statements and neither is derivable from the other."""
    suite, _ = h2
    total = suite.sums[1]
    assert isinstance(total, CrossSectionSum)
    assert total.label == "total"
    assert len(total.summands) == 3
    assert all(add.href.startswith("/reactionSuite/reactions/")
               for add in total.summands)
    assert isinstance(total.crossSection["eval"], XYs1d)
    assert total.outputChannel.Q.value == 0.0


def test_a_sum_is_reachable_through_the_suite_wide_mt_lookup(h2):
    """MT1 is not in ``reactions`` and asking the suite for it still works."""
    suite, _ = h2
    assert suite.reactionByENDF_MT(1) is suite.sums[1]
    assert 1 not in suite.reactions
    assert suite.findReactionByENDF_MT(999) is None


def test_application_data_is_declared_lost_rather_than_kept_as_raw_xml(h2):
    _, report = h2
    assert any("applicationData holds ['LLNL']" in loss for loss in report.losses)


# ---------------------------------------------------------------------------
# resonances — read by kika.gnds.resonances, hung on the suite here
# ---------------------------------------------------------------------------

def test_the_resonance_block_is_attached_to_the_suite(fe56):
    """§19 has its own module; what this checks is that it is wired in and that
    the cross section form's href points at what was actually read."""
    suite, _ = fe56
    assert suite.hasResonances
    assert suite.resonances.domain == (1e-5, 8.5e5)
    assert suite.reactions[2].crossSection["eval"].resonanceRegionHref \
        == "/reactionSuite/resonances"


def test_a_suite_with_no_resonances_leaves_the_one_child_that_may_be_none(h2):
    """H-2 has a ``scatteringRadius`` and no region at all — §14.1.1's one
    legitimate ``None`` is the *region*, not the block."""
    suite, _ = h2
    assert suite.hasResonances
    assert suite.resonances.domain is None
    assert suite.resonances.scatteringRadius.constant == 5.1977


# ---------------------------------------------------------------------------
# the pieces that no committed fixture reaches
# ---------------------------------------------------------------------------

def test_a_second_form_in_a_container_that_holds_one_is_reported(h2_gnds):
    """``Q`` and ``multiplicity`` are single-valued in kika and multi-valued in
    GNDS, so a file with two forms forces a choice that has to be said out loud.

    No distributed evaluation does this, which is exactly why it is planted:
    the branch would otherwise be code no test walks.
    """
    tree = ET.parse(h2_gnds)
    q = tree.getroot().find(".//reaction[@ENDF_MT='2']/outputChannel/Q")
    second = ET.SubElement(q, "constant1d")
    second.attrib.update({"label": "recon", "value": "1", "domainMin": "1e-5",
                          "domainMax": "1.5e8"})

    suite, report = readReactionSuite(Document(root=tree.getroot()))
    assert suite.reactions[2].outputChannel.Q.value == 0.0     # the eval one
    assert any("this <Q> carries ['eval', 'recon']" in loss
               for loss in report.losses), report.losses


def test_an_energy_dependent_q_is_refused_rather_than_averaged(h2_gnds):
    """Every Q in ENDF/B-VIII.1-GNDS is a ``constant1d``; the model holds a
    number. An ``XYs1d`` Q would need a node kika does not have, and picking
    one of its values would be an invention that looks like data."""
    tree = ET.parse(h2_gnds)
    q = tree.getroot().find(".//reaction[@ENDF_MT='2']/outputChannel/Q")
    q.remove(q.find("constant1d"))
    ET.SubElement(q, "XYs1d").attrib["label"] = "eval"

    suite, report = readReactionSuite(Document(root=tree.getroot()))
    assert not suite.reactions[2].outputChannel.Q.isKnown
    assert any("energy-dependent Q" in entry for entry in report.unsupported)


def test_an_isotropic_two_body_angular_form_is_read_as_the_distribution(h2_gnds):
    """780 ``angularTwoBody`` nodes in the library hold an ``isotropic2d``, and
    neither committed fixture is one of them."""
    tree = ET.parse(h2_gnds)
    twoBody = tree.getroot().find(
        ".//reaction[@ENDF_MT='2']/outputChannel/products/product[@pid='n']"
        "/distribution/angularTwoBody"
    )
    for child in list(twoBody):
        twoBody.remove(child)
    ET.SubElement(twoBody, "isotropic2d")

    suite, _ = readReactionSuite(Document(root=tree.getroot()))
    product = suite.reactions[2].outputChannel.products.byPid("n")[0]
    form = product.distribution["eval"]
    assert isinstance(form.angular, Isotropic2d)
    assert not form.isRecoil
    assert form.energies == []


def test_a_subtract_summand_is_reported(h2_gnds):
    """§21.3 admits it and no neutron evaluation uses one, so a file that did
    would otherwise have a term silently added instead of subtracted."""
    tree = ET.parse(h2_gnds)
    summands = tree.getroot().find(".//crossSectionSum[@ENDF_MT='1']/summands")
    ET.SubElement(summands, "subtract").attrib["href"] = "/reactionSuite/reactions"

    suite, report = readReactionSuite(Document(root=tree.getroot()))
    assert len(suite.sums[1].summands) == 3      # the adds, not the subtract
    assert any("<subtract>" in entry for entry in report.unsupported)


def test_an_unmodelled_style_is_reported_rather_than_dropped(h2_gnds):
    """``heated`` is modelled and occurs in no distributed neutron evaluation."""
    tree = ET.parse(h2_gnds)
    styles = tree.getroot().find("styles")
    ET.SubElement(styles, "heated").attrib.update(
        {"label": "heated300", "derivedFrom": "eval"}
    )

    suite, report = readReactionSuite(Document(root=tree.getroot()))
    assert suite.styles.labels == ["eval"]
    assert any("/styles/heated:" in entry for entry in report.unsupported)
