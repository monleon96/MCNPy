"""MF1's nu-bars and MF31's covariances of them, both directions.

Two gates, and they are deliberately of different strengths.

**MF1/452, /455, /456 are held to byte identity**, like MF1/451, MF3 and MF4
before them. A nu-bar is a TAB1 or a short LIST; nothing in the trip through the
model has any licence to move a digit, and if it does, the file kika writes is
not the file it read.

**MF31 is held to a numerical fixed point**, like MF33 and MF34, for the reason
``test_covariance_round_trip.py`` states: the decoders read through
``to_xs_covmat``, which has already collapsed the file's NC/NI sub-subsection
structure into dense matrices, so the LB layout is not recoverable from the
model and byte identity is not on the table.

**The one that is worth the fixture's existence** is neither: it is *where* the
three MTs land. §17.3 puts the prompt nu-bar on the fission product, §21.3 puts
the total and the delayed on ``multiplicitySums``, §18.4 puts the precursor
rates on ``fissionFragmentData`` — three different nodes for three sections of
one ENDF file. A model that hung all three off one node would decode, encode,
round-trip and be wrong, and only the hrefs would show it.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.endf.model_adapter import (attachNubar, decodeMF31MT, decodeMF1Nubar,
                                     decodeCovarianceSuite, decodeReactionSuite,
                                     encodeMF1MT452, encodeMF1MT455,
                                     encodeMF1MT456, encodeMF31MT, nubarHref)
from kika.endf.model_adapter.multiplicity import (DELAYED_NUBAR_LABEL,
                                                  TOTAL_NUBAR_LABEL,
                                                  fissionProductMultiplicityHref,
                                                  multiplicitySumHref, nubarNode)
from kika.endf.read_endf import read_endf


@pytest.fixture(scope="module")
def nubarEndf(micro_nubar_tape):
    return read_endf(str(micro_nubar_tape))


@pytest.fixture(scope="module")
def decoded(nubarEndf):
    suite, report = decodeReactionSuite(nubarEndf)
    covariances, report = decodeCovarianceSuite(nubarEndf, report,
                                                evaluation=suite.evaluation)
    suite.covarianceSuite = covariances
    return suite, report


def test_the_fixture_carries_all_three_nubars_and_their_covariances(nubarEndf):
    """A tape with only MT452 would make most of this file vacuous."""
    assert set(nubarEndf.mf[1].mt) >= {451, 452, 455, 456}
    assert set(nubarEndf.mf[31].mt) == {452, 455, 456}
    assert 18 in nubarEndf.mf[3].mt, "no MF3/MT18: the nu-bars would have no home"


# ---------------------------------------------------------------------------
# Where the three MTs land
# ---------------------------------------------------------------------------

def test_the_prompt_nubar_is_the_fission_products_multiplicity(decoded):
    """§17.3. MT456 is not derived from anything; it *is* the multiplicity."""
    suite, _ = decoded
    fission = suite.findReactionByENDF_MT(18)
    neutrons = suite.reactions and fission.outputChannel.products.byPid("n")

    assert len(neutrons) == 1
    multiplicity = neutrons[0].multiplicity
    assert multiplicity is not None
    # ENDF/B-VIII.1 U-235: nu_p(1e-5 eV) = 2.414
    assert multiplicity.evaluate(1e-5) == pytest.approx(2.414)


def test_the_total_and_delayed_nubars_are_multiplicity_sums(decoded):
    """§21.3. Both are derived, and neither belongs on the product."""
    suite, _ = decoded
    sums = suite.sums.multiplicitySums

    total = sums.byENDF_MT(452)
    delayed = sums.byENDF_MT(455)
    assert total is not None and total.label == TOTAL_NUBAR_LABEL
    assert delayed is not None and delayed.label == DELAYED_NUBAR_LABEL

    # nu_total = nu_prompt + nu_delayed, to the precision the file states them.
    fission = suite.findReactionByENDF_MT(18)
    prompt = fission.outputChannel.products.byPid("n")[0].multiplicity
    assert total.multiplicity.evaluate(1e-5) == pytest.approx(
        prompt.evaluate(1e-5) + delayed.multiplicity.evaluate(1e-5), rel=1e-6
    )


def test_the_total_links_to_what_it_is_a_sum_of(decoded):
    """An empty ``summands`` on the total would lose the sum rule itself."""
    suite, _ = decoded
    total = suite.sums.multiplicitySums.byENDF_MT(452)
    hrefs = [summand.href for summand in total.summands]

    assert fissionProductMultiplicityHref() in hrefs
    assert multiplicitySumHref(DELAYED_NUBAR_LABEL) in hrefs


def test_the_delayed_sum_has_no_summands_and_says_why(decoded):
    """The honest empty list, and the report entry that keeps it honest.

    MF1/455 gives the aggregate; the per-family split is MF5/455's weights,
    which nothing decodes. Filling ``summands`` with links to the six empty
    family multiplicities would make the model look complete.
    """
    suite, report = decoded
    delayed = suite.sums.multiplicitySums.byENDF_MT(455)

    assert len(delayed.summands) == 0
    assert any("MF5/455" in entry for entry in report.losses)


def test_the_precursor_families_carry_their_rates(decoded):
    """§18.4. Six families for U-235, each with lambda in 1/s."""
    suite, _ = decoded
    fission = suite.findReactionByENDF_MT(18)
    families = list(fission.outputChannel.fissionFragmentData.delayedNeutrons)

    assert [family.label for family in families] == ["1", "2", "3", "4", "5", "6"]
    assert all(family.rate.unit == "1/s" for family in families)
    # ENDF/B-VIII.1 U-235's shortest-lived group.
    assert families[-1].rate.value == pytest.approx(2.853)
    # And they are strictly increasing, which is how ENDF orders them.
    rates = [family.rate.value for family in families]
    assert rates == sorted(rates)


def test_a_non_fissile_evaluation_grows_nothing(micro_tape):
    """Fe-56 has no MF1 nu-bar, so the whole path must be inert on it.

    Stated as a test because the cost of this landing on the thesis pipeline is
    the thing that would matter, and "it does nothing" is exactly the claim that
    rots without one.
    """
    endf = read_endf(str(micro_tape))
    suite, report = decodeReactionSuite(endf)

    assert len(suite.sums.multiplicitySums) == 0
    assert not any("nu-bar" in entry for entry in report.losses)


def test_a_total_only_evaluation_puts_the_total_on_the_product():
    """The exception the ENDF files force, and the one easy to get wrong.

    With no MT456 the total *is* the primitive. Pointing its covariance at a
    ``multiplicitySum`` that the decoder never created would produce a dangling
    href — valid-looking, and resolving to nothing.
    """
    assert nubarHref(452, separatePrompt=False) == fissionProductMultiplicityHref()
    assert nubarHref(452, separatePrompt=True) == multiplicitySumHref(TOTAL_NUBAR_LABEL)
    assert nubarHref(456) == fissionProductMultiplicityHref()
    assert nubarHref(455) == multiplicitySumHref(DELAYED_NUBAR_LABEL)


def test_a_total_only_evaluation_lands_on_the_product_end_to_end(micro_nubar_tape):
    """The same exception, exercised through the decoders rather than the href.

    :func:`test_a_total_only_evaluation_puts_the_total_on_the_product` pins the
    string; this pins the *behaviour*, because the two can disagree. Built by
    cutting MT455 and MT456 out of the parsed U-235 tape, which is the only way
    to get a total-only evaluation here — every fissile tape on this machine
    carries all three.

    The failure this guards against is silent by construction: a covariance
    whose ``href`` names a ``multiplicitySum`` the decoder never created is a
    dangling link, and nothing in a round trip resolves links.

    **The MF31 half has to be written rather than cut**, and the first version
    of this test got that wrong. U-235's MF31/452 is NC-type — it *is* the
    declaration "total = MT455 + MT456" — so deleting the contributors does not
    leave a total-only covariance, it leaves an unresolvable one (that case has
    its own test). A real total-only evaluation states an explicit matrix, so
    one is written here with kika's own encoder. The numbers are beside the
    point; the href is the assertion.
    """
    full = read_endf(str(micro_nubar_tape))
    fullSuite, _ = decodeReactionSuite(full)
    fullCov, _ = decodeCovarianceSuite(full)
    fullSuite.covarianceSuite = fullCov
    explicit, _ = encodeMF31MT(fullCov, 456, mat=full.mat)
    explicit.number = 452

    endf = read_endf(str(micro_nubar_tape))
    for mt in (455, 456):
        del endf.mf[1].mt[mt]
    endf.mf[31].mt.clear()
    endf.mf[31].mt[452] = explicit

    suite, report = decodeReactionSuite(endf)
    covariances, report = decodeCovarianceSuite(endf, report)

    fission = suite.findReactionByENDF_MT(18)
    product = fission.outputChannel.products.byPid("n")[0]
    assert product.multiplicity is not None, "the total is the only multiplicity"
    assert len(suite.sums.multiplicitySums) == 0, "nothing is derived from anything"
    assert fission.outputChannel.fissionFragmentData is None

    # And the covariance points at the node that exists, not at the one the
    # three-MT layout would have created.
    assert len(covariances.covarianceSections) == 1
    section = covariances.covarianceSections[0]
    assert section.rowData.ENDF_MFMT == "31/452"
    assert section.rowData.href == fissionProductMultiplicityHref()


def test_a_nubar_with_no_fission_reaction_is_reported_not_invented(nubarEndf):
    """MF3/MT18 is what the nu-bars hang from; without it there is no node."""
    from kika.nuclear_data.model import ConversionReport, ReactionSuite

    empty = ReactionSuite(evaluation="none", projectile="n", target="U235")
    report = attachNubar(empty, nubarEndf.mf[1], ConversionReport())

    assert len(empty.sums.multiplicitySums) == 0
    assert any("no MF3/MT18" in entry for entry in report.losses)


def test_the_nubar_and_the_angular_distribution_share_one_neutron(micro_nubar_tape,
                                                                 micro_tape):
    """§17.2.1 gives one product one multiplicity *and* one distribution.

    ENDF states them in different files, and the decoder reads them in separate
    passes. Both passes used to append a product of their own, so a fissile tape
    carrying MF1/456 and MF4/MT18 came out with **two** neutrons on the fission
    channel — one holding the multiplicity, one holding the distribution — and
    ``byPid('n')[0]`` returned whichever ran first. Nothing failed; the model
    was simply describing a fission that emits two kinds of neutron.

    The nu-bar tape has no MF4/MT18 (the cut drops it), so the second pass is
    driven here with a real MF4 section from the Fe-56 fixture. The section's
    physics is irrelevant to the assertion; its arrival is the point.
    """
    from kika.endf.model_adapter.decode import _attachAngularDistribution

    endf = read_endf(str(micro_nubar_tape))
    suite, report = decodeReactionSuite(endf)
    fission = suite.findReactionByENDF_MT(18)
    assert len(fission.outputChannel.products) == 1

    mf4mt = read_endf(str(micro_tape)).mf[4].mt[2]
    _attachAngularDistribution(suite, mf4mt, 18, report)

    neutrons = fission.outputChannel.products.byPid("n")
    assert len(neutrons) == 1, "MF4 appended a second neutron instead of reusing"
    assert neutrons[0].multiplicity is not None
    assert neutrons[0].distribution is not None


# ---------------------------------------------------------------------------
# MF1 → model → MF1, byte for byte
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mt, encoder", [
    (452, encodeMF1MT452), (455, encodeMF1MT455), (456, encodeMF1MT456),
])
def test_the_nubar_sections_are_written_back_byte_identical(decoded, nubarEndf,
                                                            mt, encoder):
    suite, _ = decoded
    original = nubarEndf.mf[1].mt[mt]
    built, _ = encoder(suite, mat=original._mat)

    assert str(built) == str(original)


def test_the_written_header_is_the_files_and_not_a_default(decoded, nubarEndf):
    """ZA and AWR come off the section, not off the model's guess at them."""
    suite, _ = decoded
    built, _ = encodeMF1MT456(suite)

    assert built._za == pytest.approx(nubarEndf.mf[1].mt[456]._za)
    assert built._awr == pytest.approx(nubarEndf.mf[1].mt[456]._awr)
    assert built._lnu == nubarEndf.mf[1].mt[456]._lnu


def test_encoding_a_nubar_the_suite_does_not_carry_raises(micro_tape):
    """Fe-56 has none; writing MF1/452 from it must fail loudly."""
    endf = read_endf(str(micro_tape))
    suite, _ = decodeReactionSuite(endf)

    with pytest.raises(ValueError, match="no MT452 nu-bar"):
        encodeMF1MT452(suite)


def test_a_polynomial_nubar_declares_the_domain_it_was_given(nubarEndf):
    """LNU=1 states coefficients and no energy range, so the domain is invented.

    Reported rather than silent: the coefficients are the file's, the domain is
    not, and a reader comparing two evaluations' domains would otherwise be
    comparing kika's defaults.
    """
    from kika.nuclear_data.model import ConversionReport

    class _Polynomial:
        number = 452
        lnu = 1
        coefficients = [2.4, 0.1]
        zaid = 92235.0
        atomic_weight_ratio = 233.0248
        _mat = 9228

    multiplicity, _, report = decodeMF1Nubar(_Polynomial(), ConversionReport())

    assert multiplicity.evaluate(0.0) == pytest.approx(2.4)
    assert any("LNU=1" in entry for entry in report.approximations)


# ---------------------------------------------------------------------------
# MF31
# ---------------------------------------------------------------------------

def _sectionFor(suite, mt):
    for section in suite.covarianceSuite.covarianceSections:
        if section.rowData.ENDF_MFMT == f"31/{mt}":
            return section
    raise AssertionError(f"no MF31/MT{mt} section in the suite")


def test_each_nubar_covariance_points_at_its_own_node(decoded):
    """The assertion the whole module exists for.

    Three covariances, three different hrefs. If this ever collapses to one
    node, every downstream consumer will read the delayed covariance as if it
    were about the total, and nothing else in the suite will look wrong.
    """
    suite, _ = decoded

    assert _sectionFor(suite, 456).rowData.href == fissionProductMultiplicityHref()
    assert _sectionFor(suite, 452).rowData.href == multiplicitySumHref(TOTAL_NUBAR_LABEL)
    assert _sectionFor(suite, 455).rowData.href == multiplicitySumHref(DELAYED_NUBAR_LABEL)
    assert len({_sectionFor(suite, mt).rowData.href for mt in (452, 455, 456)}) == 3


def test_the_matrices_are_the_arrays_kika_cov_already_produced(nubarEndf, decoded):
    """The adapter re-expresses; it must not recompute."""
    suite, _ = decoded
    siblings = {mt: nubarEndf.mf[31].mt[mt] for mt in nubarEndf.mf[31].mt}
    from kika.endf.model_adapter.covariances import nubarShims

    values = nubarShims(nubarEndf.mf[1])
    for mt in sorted(siblings):
        expected = nubarEndf.mf[31].mt[mt].to_xs_covmat(
            sibling_sections=siblings, mf3_sections=values
        )
        decodedSections, _ = decodeMF31MT(
            nubarEndf.mf[31].mt[mt], siblingSections=siblings, nubarValues=values
        )
        assert len(decodedSections) == len(expected.matrices)
        for entry, matrix in zip(decodedSections, expected.matrices):
            np.testing.assert_array_equal(
                entry.form.matrix, np.asarray(matrix, dtype=float)
            )


def test_the_derived_total_is_resolved_and_weighted_by_the_nubars(decoded):
    """MF31/452 is NC-type: the file declares it rather than storing it.

    U-235's total nu-bar covariance is ``LTY=0``, a sum of MT455 and MT456. The
    check that it was resolved *correctly* rather than merely resolved is the
    weighting: delayed nu-bar is ~0.65 % of total, so the total's relative
    variance has to sit just below prompt's — a straight unweighted sum of the
    two relative matrices would land far above it.
    """
    suite, _ = decoded
    total = _sectionFor(suite, 452).form
    prompt = _sectionFor(suite, 456).form

    assert total.matrix.shape[0] > 1
    ratio = np.median(np.diag(total.matrix)) / np.median(np.diag(prompt.matrix))
    assert 0.95 < ratio < 1.0, (
        f"total/prompt relative variance ratio is {ratio}; ~0.99 is the "
        f"weighted sum, ~2 would be an unweighted one"
    )


def test_an_unresolvable_derived_total_is_reported_rather_than_dropped(nubarEndf):
    """Without the siblings the sum cannot be resolved — and must say so."""
    sections, report = decodeMF31MT(nubarEndf.mf[31].mt[452])

    assert sections == []
    assert any("NC-type" in entry for entry in report.losses)


def test_the_covariances_are_relative_as_endf_states_them(decoded):
    suite, _ = decoded
    for mt in (452, 455, 456):
        assert _sectionFor(suite, mt).form.isRelative


@pytest.mark.parametrize("mt", [452, 455, 456])
def test_mf31_survives_a_round_trip_through_the_model(decoded, nubarEndf, mt):
    suite, _ = decoded
    siblings = {mt: nubarEndf.mf[31].mt[mt] for mt in nubarEndf.mf[31].mt}

    built, _ = encodeMF31MT(suite.covarianceSuite, mt, mat=nubarEndf.mat)
    rebuilt, _ = decodeMF31MT(built, siblingSections=siblings)

    original = _sectionFor(suite, mt).form
    assert len(rebuilt) == 1
    np.testing.assert_allclose(rebuilt[0].form.matrix, original.matrix, rtol=1e-4)
    np.testing.assert_allclose(rebuilt[0].form.rowGrid, original.rowGrid, rtol=1e-9)


@pytest.mark.parametrize("mt", [452, 455, 456])
def test_the_written_section_says_it_is_mf31(decoded, nubarEndf, mt):
    """Columns 71-72, not just the block the writer drops it into.

    ``MF33MT.__str__`` wrote the literal ``33`` until this landed, so an MF31
    section came out stamped MF33 — correct in every number and wrong in the
    two columns a parser reads to know what it is looking at.
    """
    suite, _ = decoded
    built, _ = encodeMF31MT(suite.covarianceSuite, mt, mat=nubarEndf.mat)

    assert built._mf == 31
    identifiers = {
        line[70:72] for line in str(built).splitlines() if len(line) >= 75
    }
    assert identifiers == {"31"}


def test_the_encoder_declares_that_a_derived_covariance_became_explicit(decoded,
                                                                        nubarEndf):
    """Writing back the resolved matrix loses the *declaration* that it is a sum."""
    suite, _ = decoded
    _, report = encodeMF31MT(suite.covarianceSuite, 452, mat=nubarEndf.mat)

    assert any("NC-type" in entry for entry in report.approximations)


def test_encoding_an_mt_the_suite_does_not_carry_raises(decoded):
    suite, _ = decoded
    with pytest.raises(ValueError, match="no MF31 covariance sections for MT18"):
        encodeMF31MT(suite.covarianceSuite, 18)


def test_the_front_door_reads_the_whole_thing(micro_nubar_tape):
    """``kika.read()`` — the path a user actually takes."""
    import kika

    evaluation = kika.read(str(micro_nubar_tape))

    assert len(evaluation.covarianceSuite.covarianceSections) == 3
    assert nubarNode(evaluation, 456) is not None
    assert nubarNode(evaluation, 452) is not None
    # The redirect notice must be gone: the door decoded MF31, so telling the
    # user to call `decodeCovarianceSuite` for it would be false.
    assert not any("MF31" in entry for entry in evaluation.report.unsupported)
