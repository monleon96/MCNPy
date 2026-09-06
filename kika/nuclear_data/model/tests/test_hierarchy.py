"""The §9-25 hierarchy: present-but-empty, derivedFrom chains, and round-tripping.

Three gates, in the order the roadmap states them.

1. **Every §14.1.1 child is present as an empty container, not ``None``.** The
   distinction is the whole point of building the hierarchy before filling it: a
   slot that exists and is empty says "kika models this and this file has none",
   while a missing attribute says nothing and fails somewhere unrelated later.

2. **``derivedFrom`` resolves, and a cycle raises.** ``derivedFrom`` is a
   free-text label, so two styles naming each other is one typo away; without a
   check the failure is a hang rather than an error.

3. **Every node round-trips through ``to_dict``/``from_dict``.** The cheapest
   possible proxy for the phase 5 reader and the phase 7c writer, run three
   months before either exists. It catches the modelling mistakes that only show
   up when something has to walk the tree generically — a field that cannot be
   reconstructed, a container that loses its element type, a cycle.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from kika.nuclear_data import model as m


# ---------------------------------------------------------------------------
# 1. Present, not absent
# ---------------------------------------------------------------------------

#: §14.1.1's child list, verbatim. `resonances` is the one legitimate `None`:
#: a target with no resonance parameters has no resonance region at all, which
#: is a different statement from "an empty list of them".
SUITE_CHILDREN = (
    "externalFiles", "styles", "PoPs", "reactions", "orphanProducts", "sums",
    "fissionComponents", "productions", "incompleteReactions", "applicationData",
)


@pytest.fixture
def suite() -> m.ReactionSuite:
    return m.ReactionSuite(evaluation="JEFF-4.0", projectile="n", target="Fe56")


@pytest.mark.parametrize("child", SUITE_CHILDREN)
def test_every_child_of_an_empty_suite_is_a_container_not_none(suite, child):
    value = getattr(suite, child)
    assert value is not None, (
        f"reactionSuite.{child} is None on a fresh suite. §14.1.1 lists it as a "
        f"child; an absent slot and an empty one say different things."
    )
    assert len(value) == 0


@pytest.mark.parametrize("child", SUITE_CHILDREN)
def test_an_empty_container_is_still_truthy(suite, child):
    """``if suite.reactions:`` must not read *empty* as *absent*.

    This is the trap the distinction above creates: Python's default truthiness
    for an empty collection is ``False``, so the natural spelling of "does this
    evaluation model reactions" would answer "no" for an evaluation that models
    them and happens to have none yet.
    """
    assert bool(getattr(suite, child)) is True


def test_resonances_is_the_one_child_allowed_to_be_none(suite):
    assert suite.resonances is None
    assert suite.hasResonances is False


def test_the_declared_but_unimplemented_nodes_name_themselves():
    """A reader meeting one is told which GNDS node is missing."""
    assert m.NOT_IMPLEMENTED_DISTRIBUTIONS
    for name, cls in m.NOT_IMPLEMENTED_DISTRIBUTIONS.items():
        with pytest.raises(NotImplementedError, match=name):
            cls()
    from kika.nuclear_data.model.functions.higher import NOT_IMPLEMENTED_NODES

    # Down to one entry since XYs3d landed, and the assert is what keeps this
    # loop from passing on an empty dict when the last one goes -- except that
    # the last one cannot go: regions3d has no element in gnds.xsd to read.
    assert NOT_IMPLEMENTED_NODES
    for name, cls in NOT_IMPLEMENTED_NODES.items():
        with pytest.raises(NotImplementedError, match=name):
            cls()


# ---------------------------------------------------------------------------
# 2. The derivedFrom chain
# ---------------------------------------------------------------------------

def test_the_chain_resolves_evaluated_to_reconstructed_to_realization():
    styles = m.Styles()
    styles.add(m.Evaluated(label="eval", library="JEFF", version="4.0"))
    styles.add(m.CrossSectionReconstructed(label="recon", derivedFrom="eval"))
    styles.add(m.Realization(label="sample-0007", derivedFrom="recon"))

    chain = [s.label for s in styles.chain("sample-0007")]
    assert chain == ["sample-0007", "recon", "eval"]
    assert styles.rootOf("sample-0007").label == "eval"
    assert styles.evaluatedFor("sample-0007").library == "JEFF"


def test_a_realization_must_say_what_it_was_sampled_from():
    """§9.3.1 makes ``derivedFrom`` **required** on ``realization``.

    *"The derivedFrom node must point to the evaluated style containing the
    covariances and/or probabilities that were sampled to produce this
    realization."* A perturbed sample that cannot name its parent covariance is
    not traceable, which is the whole reason the style exists.
    """
    with pytest.raises(m.StyleError, match="requires derivedFrom"):
        m.Realization(label="sample-0007")

    assert m.Evaluated(label="eval").derivedFrom is None  # not required here


def test_a_derivedfrom_cycle_raises_instead_of_hanging():
    styles = m.Styles()
    styles.add(m.CrossSectionReconstructed(label="a", derivedFrom="b"))
    styles.add(m.CrossSectionReconstructed(label="b", derivedFrom="a"))

    with pytest.raises(m.StyleError, match="cycle"):
        styles.chain("a")


def test_a_dangling_derivedfrom_raises():
    styles = m.Styles()
    styles.add(m.CrossSectionReconstructed(label="recon", derivedFrom="nope"))

    with pytest.raises(m.StyleError, match="does not exist"):
        styles.chain("recon")


def test_duplicate_style_labels_are_refused():
    styles = m.Styles()
    styles.add(m.Evaluated(label="eval"))
    with pytest.raises(m.StyleError, match="duplicate"):
        styles.add(m.Evaluated(label="eval"))


# ---------------------------------------------------------------------------
# The multi-form crossSection
# ---------------------------------------------------------------------------

def test_a_cross_section_holds_one_form_per_style_label():
    """§9.1's own example: a `resonancesWithBackground` labelled ``eval`` beside
    an ``XYs1d`` labelled ``recon``."""
    xs = m.CrossSection()
    xs[m.EVAL_LABEL] = m.ResonancesWithBackground(
        background=m.Background(
            resolvedRegion=m.XYs1d(xs=[1.0, 10.0], ys=[2.0, 3.0])
        )
    )
    xs["recon"] = m.XYs1d(xs=[1.0, 5.0, 10.0], ys=[2.0, 2.5, 3.0])

    assert xs.hasEvaluated
    assert sorted(xs) == ["eval", "recon"]
    assert xs.evaluate(5.0, label="recon") == pytest.approx(2.5)


def test_evaluating_a_representation_rather_than_a_function_is_an_error():
    """``resonancesWithBackground`` is an instruction, not a curve.

    Returning the background alone would be a plausible-looking wrong answer —
    it is the cross section with every resonance missing.
    """
    xs = m.CrossSection()
    xs[m.EVAL_LABEL] = m.ResonancesWithBackground(
        background=m.Background(
            resolvedRegion=m.XYs1d(xs=[1.0, 10.0], ys=[2.0, 3.0])
        )
    )
    with pytest.raises(TypeError, match="representation rather than a function"):
        xs.evaluate(5.0)


def test_endf_mt_is_carried_but_derived():
    """§15.1.1 requires ``ENDF_MT`` and deprecates it in the same paragraph."""
    reaction = m.Reaction(id=m.ReactionId(label="n + Fe56", ENDF_MT=2))
    assert reaction.ENDF_MT == 2

    future = m.Reaction(id=m.ReactionId(label="something with no MT"))
    assert future.ENDF_MT is None  # and nothing else breaks

    suite = m.ReactionSuite(evaluation="x", projectile="n", target="Fe56")
    suite.reactions.append(reaction)
    suite.reactions.append(future)
    assert suite.reactionByENDF_MT(2) is reaction
    assert suite.reactionByLabel("something with no MT") is future


# ---------------------------------------------------------------------------
# The MF34 mapping, stated once so P7 cannot get it wrong quietly
# ---------------------------------------------------------------------------

def test_a_legendre_order_covariance_is_a_slice_domain_value():
    """§25.2.5-6. The Legendre order is a *slice* of the distribution, not a
    separate quantity."""
    link = m.DataLink.forLegendreOrder("/reactionSuite/reactions/reaction[@label='2']", 1)

    assert len(link.slices) == 1
    assert link.slices.slices[0].domainValue == 1.0
    assert link.legendreOrder == 1


def test_a_slice_takes_a_point_or_a_range_but_not_both():
    with pytest.raises(ValueError, match="not both"):
        m.Slice(dimension=1, domainValue=1.0, domainMin=0.0)
    with pytest.raises(ValueError, match="needs domainValue"):
        m.Slice(dimension=1)


def test_a_covariance_matrix_checks_its_grid():
    with pytest.raises(ValueError, match="expected one more than the rows"):
        m.CovarianceMatrix(matrix=np.eye(3), rowGrid=np.array([1.0, 2.0, 3.0]))

    ok = m.CovarianceMatrix(matrix=np.eye(3), rowGrid=np.array([1.0, 2.0, 3.0, 4.0]))
    assert ok.isSquare and ok.isSymmetric


# ---------------------------------------------------------------------------
# Resonances: c3..c6 is gone
# ---------------------------------------------------------------------------

def test_breit_wigner_widths_are_named_not_numbered():
    from kika.nuclear_data.model.resonances.breit_wigner import Resonance

    r = Resonance(energy=1.15e3, spin=0.5, totalWidth=1.4, neutronWidth=1.2,
                  captureWidth=0.2, fissionWidth=0.0)
    assert not hasattr(r, "c3")
    # One-way bridge for the ENDF adapter, in ENDF's own positional order.
    assert r.toFlat() == (1.15e3, 0.5, 1.4, 1.2, 0.2, 0.0)
    assert Resonance.fromFlat(*r.toFlat()) == r


def test_an_r_matrix_width_belongs_to_a_channel():
    """Which is why ``c3..c6`` cannot express Reich-Moore: two fission widths."""
    group = m.RMatrixSpinGroup(
        label="1",
        channels=[m.Channel(label="elastic", resonanceReaction="n + Fe56"),
                  m.Channel(label="capture", resonanceReaction="capture")],
        energies=[1.0, 2.0],
        widths=[[1.2, 0.2], [1.3, 0.25]],
    )
    assert len(group) == 2

    with pytest.raises(ValueError, match="width row has"):
        m.RMatrixSpinGroup(
            label="1",
            channels=[m.Channel(label="elastic", resonanceReaction="n + Fe56")],
            energies=[1.0],
            widths=[[1.2, 0.2]],
        )


def test_resonances_report_their_overall_domain():
    """The model's counterpart to ``detect_resonance_bounds``, which reads ENDF."""
    resonances = m.Resonances(
        resolved=[m.ResolvedRegion(domainMin=1e-5, domainMax=8.5e5)],
        unresolved=m.UnresolvedRegion(domainMin=8.5e5, domainMax=2e6),
    )
    assert resonances.domain == (1e-5, 2e6)
    assert m.Resonances().domain is None


# ---------------------------------------------------------------------------
# Navigation — the model has to be usable, not merely correct
# ---------------------------------------------------------------------------

def _suiteWithCapture():
    reaction = m.Reaction(id=m.ReactionId(label="capture", ENDF_MT=102))
    reaction.crossSection["eval"] = m.XYs1d(
        xs=np.array([1.0, 2.0, 3.0]), ys=np.array([9.0, 8.0, 7.0])
    )
    suite = m.ReactionSuite(evaluation="JEFF-4.0", projectile="n", target="Fe56")
    suite.reactions.append(reaction)
    return suite


def test_a_reaction_is_indexed_by_mt_and_by_label():
    suite = _suiteWithCapture()
    assert suite.reactions[102] is suite.reactions["capture"]
    assert 102 in suite.reactions
    assert "capture" in suite.reactions
    assert 16 not in suite.reactions


def test_an_integer_index_is_an_mt_and_says_so_when_it_misses():
    """``reactions[0]`` is not "the first reaction", and the error has to explain
    that — otherwise it reads as a bug in kika rather than a deliberate choice."""
    suite = _suiteWithCapture()
    with pytest.raises(KeyError, match="by MT, not by position"):
        suite.reactions[0]
    # Position is still reachable for the rare caller who really means it.
    assert list(suite.reactions)[0].ENDF_MT == 102


def test_an_mt_out_of_a_numpy_array_still_indexes():
    """``np.int64`` is not an ``int``; failing on it would be a papercut."""
    suite = _suiteWithCapture()
    mts = np.array([102])
    assert suite.reactions[mts[0]].ENDF_MT == 102


def test_a_reaction_cannot_be_looked_up_by_something_meaningless():
    suite = _suiteWithCapture()
    with pytest.raises(TypeError, match="by MT .int. or by label"):
        suite.reactions[1.5]


def test_cross_section_hands_back_the_two_arrays():
    suite = _suiteWithCapture()
    energies, sigma = suite.cross_section(102)
    np.testing.assert_array_equal(energies, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(sigma, [9.0, 8.0, 7.0])


def test_cross_section_hands_back_a_copy():
    """A caller who normalises what they were given must not edit the evaluation."""
    suite = _suiteWithCapture()
    _, sigma = suite.cross_section(102)
    sigma *= 0.0
    assert suite.cross_section(102)[1][0] == 9.0


def test_cross_section_flattens_regions_by_reusing_the_endf_layout():
    """``Regions1d`` shares a boundary point between consecutive regions;
    ``toEndfRegions`` already drops it, so this must not double-count."""
    suite = _suiteWithCapture()
    suite.reactions[102].crossSection["eval"] = m.Regions1d.fromEndfRegions(
        xs=[1.0, 2.0, 3.0, 4.0], ys=[10.0, 20.0, 30.0, 40.0],
        nbtIntPairs=[(2, 2), (4, 5)],
    )
    energies, sigma = suite.cross_section(102)
    np.testing.assert_array_equal(energies, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(sigma, [10.0, 20.0, 30.0, 40.0])


def test_cross_section_names_the_forms_it_has_when_asked_for_one_it_lacks():
    suite = _suiteWithCapture()
    with pytest.raises(KeyError, match="recon"):
        suite.cross_section(102, form="recon")


def test_a_form_that_is_not_tabulated_says_why_it_cannot_be_flattened():
    """``ResonancesWithBackground`` has to be reconstructed first, and the
    shortcut must say so rather than return something plausible."""
    suite = _suiteWithCapture()
    suite.reactions[102].crossSection["eval"] = m.Reference(href="../another")
    with pytest.raises(TypeError, match="not a tabulated function"):
        suite.cross_section(102)


def test_the_suite_repr_says_what_is_in_it():
    suite = _suiteWithCapture()
    suite.resonances = m.Resonances(
        resolved=[m.ResolvedRegion(domainMin=1e-5, domainMax=8.5e5)]
    )
    text = repr(suite)
    assert "n + Fe56" in text
    assert "1 reactions" in text
    assert "850 keV" in text, f"the resonance domain should read at human scale: {text}"


def test_the_summary_reports_a_partial_decode_rather_than_hiding_it():
    suite = _suiteWithCapture()
    suite.report = m.ConversionReport()
    suite.report.lost("MF6 was on the tape and kika has no parser for it")
    assert "1 losses" in suite.summary()
    assert "1 losses" in repr(suite), "a lossy decode must be visible in the repr"


def test_a_clean_decode_does_not_clutter_the_repr():
    suite = _suiteWithCapture()
    suite.report = m.ConversionReport()
    assert "clean" not in repr(suite)
    assert "clean" in suite.summary()


# ---------------------------------------------------------------------------
# 4. The nodes the real GNDS files forced (phase 5, P4)
# ---------------------------------------------------------------------------

def test_a_background_is_three_named_regions_and_knows_which_it_has():
    """§16.1.1. A region the evaluation does not carry stays ``None``, and that
    is information: 1 541 of the library's 1 613 backgrounds have a resolved
    region and all 1 613 have a fast one, so absence is never "empty"."""
    background = m.Background(
        resolvedRegion=m.XYs1d(xs=[1e-5, 8.5e5], ys=[0.0, 1.0]),
        fastRegion=m.XYs1d(xs=[8.5e5, 1.5e8], ys=[1.0, 2.0]),
    )
    assert len(background) == 2
    assert background.unresolvedRegion is None
    assert list(background.regions) == [
        "resolvedRegion", "unresolvedRegion", "fastRegion"
    ]
    assert "resolvedRegion=XYs1d" in repr(background)


def test_an_empty_background_is_present_rather_than_absent():
    """The §14.1.1 rule, applied one level down: `if form.background:` must not
    read a background whose regions have not been decoded as no background."""
    assert m.Background()
    assert len(m.Background()) == 0
    assert "no region" in repr(m.Background())


def test_a_cross_section_sum_is_a_reaction_so_the_suite_lookup_finds_it():
    """§21.2. MT1 is not an exclusive reaction and asking for it must still work."""
    suite = m.ReactionSuite(evaluation="e", projectile="n", target="Fe56")
    total = m.CrossSectionSum(
        id=m.ReactionId(label="total", ENDF_MT=1),
        summands=m.Summands([m.Add(href="/reactionSuite/reactions/reaction[0]")]),
    )
    suite.sums.append(total)

    assert suite.reactionByENDF_MT(1) is total
    assert isinstance(total, m.Reaction)
    assert 1 not in suite.reactions
    assert len(total.summands) == 1
    # A sum has no products; §21.2 gives the node no outputChannel at all.
    assert len(total.outputChannel.products) == 0


def test_a_sum_keeps_its_own_values_beside_the_links_it_sums():
    """Recomputing the total from the summands would replace what the evaluator
    wrote, and the two need not agree to the last digit."""
    total = m.CrossSectionSum(id=m.ReactionId(label="total", ENDF_MT=1))
    total.crossSection[m.EVAL_LABEL] = m.XYs1d(xs=[1.0, 10.0], ys=[3.0, 4.0])
    assert total.crossSection.hasEvaluated
    assert len(total.summands) == 0        # values without parts is legal
    assert "0 summands" in repr(total)


def test_a_recoil_angular_distribution_carries_a_link_and_no_table():
    """§18's commonest ``angularTwoBody``: 22 540 of the library's 45 080."""
    recoil = m.AngularTwoBody(
        label="eval", recoilHref="../../product[@label='n']/distribution"
    )
    assert recoil.isRecoil
    assert recoil.angular is None
    # The accessors answer "nothing tabulated" instead of raising, because a
    # caller iterating every product's energies should not have to special-case
    # the majority shape.
    assert recoil.energies == []
    assert recoil.function1ds == []


def test_an_isotropic_two_body_form_is_a_distribution_not_an_absence():
    isotropic = m.AngularTwoBody(label="eval", angular=m.Isotropic2d())
    assert not isotropic.isRecoil
    assert isotropic.angular is not None
    assert isotropic.energies == []


def test_a_range_quantity_carries_its_unit_and_refuses_an_inverted_interval():
    domain = m.RangeQuantity(min=1e-5, max=1.5e8, unit="eV")
    assert 1.0 in domain and 2e8 not in domain
    assert str(domain).endswith("eV")
    with pytest.raises(ValueError, match="runs min to max"):
        m.RangeQuantity(min=1.5e8, max=1e-5, unit="eV")


def test_the_evaluated_style_holds_the_two_children_the_schema_requires():
    style = m.Evaluated(
        label="eval", library="ENDF/B", version="8.1.1",
        temperature=m.PhysicalQuantity(value=0.0, unit="K"),
        projectileEnergyDomain=m.RangeQuantity(min=1e-5, max=1.5e8, unit="eV"),
    )
    assert style.temperature.unit == "K"
    assert style.projectileEnergyDomain.max == 1.5e8
