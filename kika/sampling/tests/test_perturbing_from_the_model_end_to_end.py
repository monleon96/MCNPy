"""One request, several quantities, three formats -- and nothing else moved.

This is the pipeline gate. What it has to establish is not that files appear but
that the numbers in them are the ones drawn, that a quantity the request did not
name is untouched, and that the three emitters describe the *same* realisation.

The fixture is ``micro_fe56_xs_and_angular.endf``, built for this: real JEFF-4.0
Fe-56 MF3 (MT1, MT2, MT102), MF4/MT2 in LTT=3 form, MF34/MT2 for L=1..6, and the
MF33/MT2 block of ``micro_fe56_cov.endf`` spliced in beside them. It is the one
committed tape where a single evaluation states both a cross-section covariance
and an angular one for the same reaction, which is the situation this whole
pipeline exists for -- and the situation none of the single-quantity pipelines
can express, since each of them reads one file and writes one file.

**What is deliberately not asserted here.** Byte identity against
``perturb_PENDF_files``/``perturb_ENDF_files``: those run on a PENDF and on a
full tape respectively, and the arithmetic comparison against both appliers is
made where it belongs, in
``kika/nuclear_data/model/tests/test_applying_a_perturbation_to_{a_node,an_angular_distribution}.py``.
Repeating it here would measure the appliers twice and the pipeline not at all.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from kika.endf import read_endf
from kika.sampling.model_perturbation import EMITTERS, perturbFromModel

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
TAPE = str(DATA / "micro_fe56_xs_and_angular.endf")
REQUEST = {33: None, 34: {"mt": [2], "index": [1, 2, 3]}}


def _blocks(path):
    """``(MF, MT) -> lines``, ignoring the section-end records.

    FEND and MEND are excluded because the fixture carries 78 copies of MF3's
    FEND -- an artefact of the tool that sliced it out of the full tape -- and
    the writer emits the one ENDF-6 asks for. That is the writer being right,
    and it is not what a fidelity check is about.
    """
    grouped = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        try:
            mf, mt = int(line[70:72]), int(line[72:75])
        except ValueError:
            continue
        if mt == 0:
            continue
        grouped[(mf, mt)].append(line)
    return grouped


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return perturbFromModel(TAPE, REQUEST, 2, seed=20260906,
                            outputDir=tmp_path_factory.mktemp("run"),
                            formats=EMITTERS)


# ----------------------------------------------------------------------
# What the request became
# ----------------------------------------------------------------------

def test_the_run_says_what_it_drew_together_and_what_apart(run):
    """The claim a mixed run has to make, and the one no output file carries."""
    assert "2 independent draw(s)" in run.description
    assert "no section states a block across two files" in run.description
    assert run.nSamples == 2

    metadata = (run.outputDir / "run_metadata.json").read_text(encoding="utf-8")
    assert "independent draw(s)" in metadata
    assert '"grouping": "mf"' in metadata


def test_one_realisation_covers_both_quantities(run):
    pset = run.samples[0]["set"]
    assert pset.quantities() == ("angularDistribution", "crossSection")
    assert {component.mf for component in pset.components()} == {33, 34}
    assert len(pset.groups) == 2, "the file correlates neither with the other"


def test_two_samples_are_two_different_realisations(run):
    first, second = (sample["set"] for sample in run.samples)
    assert first.label != second.label
    assert not np.array_equal(first.block(2, mf=33)[0],
                              second.block(2, mf=33)[0])
    assert not np.array_equal(first.block(2, mf=34, order=1)[0],
                              second.block(2, mf=34, order=1)[0])


# ----------------------------------------------------------------------
# The delta emitter: the numbers are the drawn ones, and nothing else moved
# ----------------------------------------------------------------------

def test_the_cross_section_carries_the_factors_that_were_drawn(run):
    """Spot-checked against the block, at the middle of each of its bins.

    The tolerance is ENDF's, not the arithmetic's: a tape stores six significant
    digits, so a factor applied exactly and then written is recoverable to about
    1e-6 and no better. Asserting more would be asserting about the file format.
    """
    sample = run.samples[0]
    factors, edges = sample["set"].block(2, mf=33)

    original = read_endf(TAPE).get_file(3).sections[2]
    perturbed = read_endf(str(sample["files"]["endf-delta"])).get_file(3).sections[2]
    baseE = np.asarray(original.energies, dtype=float)
    baseX = np.asarray(original.cross_sections, dtype=float)
    newE = np.asarray(perturbed.energies, dtype=float)
    newX = np.asarray(perturbed.cross_sections, dtype=float)

    assert newE.size > baseE.size, "no step duplicate was inserted at a bin edge"
    for at in range(len(factors)):
        probe = 0.5 * (edges[at] + edges[at + 1])
        expected = np.interp(probe, baseE, baseX) * factors[at]
        assert np.interp(probe, newE, newX) == pytest.approx(expected, rel=1e-5)


def test_the_angular_distribution_carries_its_own_factors(run):
    """And ``a_0`` is still 1: the magnitude is not in MF4 to be scaled."""
    from kika.endf.model_adapter import decodeMF4MT

    sample = run.samples[0]
    factors, edges = sample["set"].block(2, mf=34, order=1)

    def coefficients(path):
        distribution, _provenance, _report = decodeMF4MT(
            read_endf(path).get_file(4).sections[2])
        region = distribution.angular.function2ds[0]
        return (np.array([float(f.outerDomainValue) for f in region.function1ds]),
                [np.asarray(f.coefficients, dtype=float)
                 for f in region.function1ds])

    baseE, baseA = coefficients(TAPE)
    newE, newA = coefficients(str(sample["files"]["endf-delta"]))
    assert newE.size > baseE.size

    for at in range(len(factors)):
        probe = 0.5 * (edges[at] + edges[at + 1])
        if probe < baseE[0] or probe > baseE[-1]:
            continue
        expected = np.interp(probe, baseE, [a[1] for a in baseA]) * factors[at]
        got = np.interp(probe, newE, [a[1] for a in newA])
        assert got == pytest.approx(expected, rel=1e-3), (
            f"a_1 at {probe:.3e} eV is not the drawn factor times the baseline")

    assert all(vector[0] == 1.0 for vector in newA)


def test_a_reaction_the_request_did_not_name_is_untouched(run):
    """Byte for byte, which is what the delta emitter is for.

    MT1 and MT102 are in the same MF3 as the perturbed MT2, so they are
    re-rendered by the writer even though the model did not change them -- this
    asserts that the re-render is faithful, which is the property that makes
    ``cmp`` between two samples show only the perturbation.
    """
    source = _blocks(TAPE)
    delta = _blocks(str(run.samples[0]["files"]["endf-delta"]))

    for key in ((3, 1), (3, 102), (2, 151), (33, 2), (34, 2)):
        assert delta[key] == source[key], f"MF{key[0]}/MT{key[1]} was rewritten"
    assert delta[(3, 2)] != source[(3, 2)]
    assert delta[(4, 2)] != source[(4, 2)]


def test_two_samples_differ_only_where_they_were_perturbed(run):
    first = _blocks(str(run.samples[0]["files"]["endf-delta"]))
    second = _blocks(str(run.samples[1]["files"]["endf-delta"]))
    differing = {key for key in set(first) | set(second)
                 if first.get(key) != second.get(key)}
    assert differing == {(3, 2), (4, 2)}, (
        f"two samples of the same request differ in {sorted(differing)}")


# ----------------------------------------------------------------------
# The other two emitters
# ----------------------------------------------------------------------

def test_the_gnds_file_carries_the_evaluation_and_the_realisation(run):
    """§9.3: a realisation is a labelled form beside the evaluated one.

    This is what the ENDF side cannot say. A tape has one cross section per MT,
    so a perturbed tape *replaces* the evaluation; a GNDS reactionSuite holds
    both and says which is which, which is the whole reason the model path was
    worth building.
    """
    text = Path(run.samples[0]["files"]["gnds"]).read_text(encoding="utf-8")
    assert 'label="eval"' in text
    assert f'label="{run.samples[0]["label"]}"' in text
    assert text.count(f'label="{run.samples[0]["label"]}"') >= 2, (
        "both the perturbed cross section and the perturbed distribution should "
        "be there under the realisation's label")


def test_the_whole_tape_emitter_writes_a_readable_tape(run):
    """And it writes the *realisation*, not the evaluation it sits beside."""
    tape = read_endf(str(run.samples[0]["files"]["endf-tape"]))
    assert 3 in tape.files and 2 in tape.files[3].sections

    factors, edges = run.samples[0]["set"].block(2, mf=33)
    section = tape.get_file(3).sections[2]
    energies = np.asarray(section.energies, dtype=float)
    values = np.asarray(section.cross_sections, dtype=float)

    original = read_endf(TAPE).get_file(3).sections[2]
    baseE = np.asarray(original.energies, dtype=float)
    baseX = np.asarray(original.cross_sections, dtype=float)

    probe = 0.5 * (edges[0] + edges[1])
    assert np.interp(probe, energies, values) == pytest.approx(
        np.interp(probe, baseE, baseX) * factors[0], rel=1e-5)


def test_the_whole_tape_emitter_is_not_the_delta_emitter(run):
    """It writes what the model carries, which is less than the tape held.

    Stated as a test rather than only in a docstring because the difference is
    invisible in the file: MF7, MF12-15 and MF32 are simply absent, and here MF33
    and MF34 are too, since a realisation does not restate the covariance it was
    drawn from.
    """
    whole = _blocks(str(run.samples[0]["files"]["endf-tape"]))
    delta = _blocks(str(run.samples[0]["files"]["endf-delta"]))
    assert (33, 2) in delta and (34, 2) in delta
    assert set(whole) < set(delta), (
        "the whole-tape emitter should carry fewer sections, not different ones")


# ----------------------------------------------------------------------
# Refusals
# ----------------------------------------------------------------------

def test_an_unknown_emitter_is_refused_before_anything_is_drawn():
    with pytest.raises(ValueError, match="unknown emitter"):
        perturbFromModel(TAPE, REQUEST, 1, formats=("hdf5",))


def test_the_delta_emitter_needs_the_tape_it_patches(tmp_path):
    """A parsed object has no path, and the delta is a patch of a file."""
    parsed = read_endf(TAPE)
    with pytest.raises(ValueError, match="needs the path"):
        perturbFromModel(parsed, REQUEST, 1, seed=1, outputDir=tmp_path,
                         formats=("endf-delta",))


def test_the_realisation_is_removed_from_the_suite_after_it_is_written(run):
    """The other half of "labelled form instead of deepcopy".

    Leaving the forms in place would put the whole ensemble in memory one form
    at a time, which is exactly the cost the copy was rejected for. Checked
    through the second sample's diagnostics being complete: if the first
    sample's label had stayed, the second would have been applied on top of it.
    """
    first, second = (sample["set"] for sample in run.samples)
    assert set(first.components()) == set(second.components())
    for sample in run.samples:
        assert set(sample["applied"]) == set(sample["set"].components())


# ----------------------------------------------------------------------
# What the run says about what it did not do
# ----------------------------------------------------------------------

def test_a_run_that_leaves_a_sum_stale_says_so():
    """MT1 perturbed beside its partials makes a total that is not their sum.

    ENDF states MT1 and MT4 as ordinary sections, so a request for "every MT the
    file states" routinely names a sum and its partials, and each is scaled by
    its own block. Re-deriving is decision 3 of the roadmap and moves numbers, so
    the run records the fact rather than repairing it quietly or leaving it out.

    Built by hand rather than run, because the committed tape that carries a
    summed MT (``micro_fe56_structural.endf``: MT1 goes to ``sums`` because its
    partials MT2 and MT102 are given beside it) has no MF33 to draw from. What is
    under test is the note, and the note reads the suite.
    """
    from kika.endf.model_adapter import decodeReactionSuite
    from kika.sampling.joint_blocks import ComponentKey
    from kika.sampling.model_perturbation import _redundancyNote
    from kika.sampling.perturbation_set import PerturbationSet

    suite, _report = decodeReactionSuite(
        read_endf(str(DATA / "micro_fe56_structural.endf")))
    assert 1 in {int(r.ENDF_MT) for r in suite.sums.reactions}, (
        "this fixture is meant to carry MT1 as a sum; it no longer does")

    edges = np.array([1.0e-5, 1.0e6, 2.0e7])
    components = [ComponentKey(26056, 33, mt) for mt in (1, 2)]
    pset = PerturbationSet(
        label="realization-0000",
        factors={component: np.array([1.1, 0.9]) for component in components},
        binEdges={component: edges for component in components})

    note = _redundancyNote(suite, pset)
    assert note is not None
    assert "not the sum of its parts" in note
    assert "decision 3" in note

    # And a realisation that touches only the partials has nothing to say.
    partialsOnly = PerturbationSet(
        label="realization-0000",
        factors={components[1]: np.array([1.1, 0.9])},
        binEdges={components[1]: edges})
    assert _redundancyNote(suite, partialsOnly) is None


def test_a_reaction_whose_partials_the_file_omits_is_not_a_sum():
    """MT4 with no MT51-91 beside it is a reaction, and the note must not fire.

    ``f982268`` derives sum-hood rather than reading it off a list: ENDF-6 does
    not mark a summed MT, and MT103 is a sum in an evaluation that writes
    MT600-649 and a reaction in one that does not. ``micro_fe56_mf33.endf`` is
    the second case for MT4, and a note there would be noise -- which is how a
    run's warnings stop being read.
    """
    from kika.sampling.model_perturbation import perturbFromModel

    run = perturbFromModel(str(DATA / "micro_fe56_mf33.endf"), {33: None}, 1,
                           seed=5)
    assert run.samples[0]["set"].reactions() == (4, 16)
    assert run.notes == []


# ----------------------------------------------------------------------
# The multiplicity, end to end
# ----------------------------------------------------------------------

NUBAR_TAPE = str(DATA / "micro_u235_nubar.endf")


@pytest.fixture(scope="module")
def nubarRun(tmp_path_factory):
    return perturbFromModel(NUBAR_TAPE, {31: [455, 456]}, 2, seed=20260906,
                            outputDir=tmp_path_factory.mktemp("nubar"),
                            formats=("endf-delta",))


def test_the_written_nubar_tape_satisfies_the_sum_rule(nubarRun):
    """The whole reason the family rule had to be ported before this shipped.

    Perturbing MT455 and MT456 and writing them is not enough: MT452 is the sum
    of the two, so a delta that wrote only what the request named would leave a
    tape stating a total that is not the sum of the parts it had just changed.
    Measured before the rule existed: MT452 at 2.5178 at 1 MeV with a prompt
    already moved to 2.4967.

    The tolerance is the file's. ENDF stores six significant digits, so the
    identity that holds to 1e-16 in the model is recoverable to about 1e-6 once
    written.
    """
    tape = read_endf(str(nubarRun.samples[0]["files"]["endf-delta"]))
    tables = {mt: (np.asarray(tape.get_file(1).sections[mt].energies, dtype=float),
                   np.asarray(tape.get_file(1).sections[mt].nubar_values,
                              dtype=float))
              for mt in (452, 455, 456)}

    probe = np.geomspace(1.0e-3, 1.0e7, 25)
    total = np.interp(probe, *tables[452])
    parts = (np.interp(probe, *tables[455]) + np.interp(probe, *tables[456]))
    assert np.max(np.abs(total - parts) / np.abs(total)) < 5e-6


def test_the_derived_total_is_rewritten_although_it_was_not_requested(nubarRun):
    """A realisation writes what it changed, not what it was asked for.

    The request names MT455 and MT456; the applier also rebuilds MT452, and the
    emitter has to know. It learns it from the forms the applier displaced, not
    from the request -- inferring it from the request is what left the total
    stale in the first version of this.
    """
    before = read_endf(NUBAR_TAPE).get_file(1).sections[452]
    after = read_endf(str(nubarRun.samples[0]["files"]["endf-delta"])
                      ).get_file(1).sections[452]

    assert len(after.energies) > len(before.energies), (
        "MT452 was not rebuilt on the union of the perturbed components' grids")
    baseline = np.interp(1.0e6, np.asarray(before.energies, dtype=float),
                         np.asarray(before.nubar_values, dtype=float))
    realised = np.interp(1.0e6, np.asarray(after.energies, dtype=float),
                         np.asarray(after.nubar_values, dtype=float))
    assert realised != baseline


def test_a_nubar_run_leaves_the_rest_of_the_tape_alone(nubarRun):
    source = _blocks(NUBAR_TAPE)
    delta = _blocks(str(nubarRun.samples[0]["files"]["endf-delta"]))

    for key in ((3, 18), (31, 452), (31, 455), (31, 456)):
        assert delta[key] == source[key], f"MF{key[0]}/MT{key[1]} was rewritten"
    for key in ((1, 452), (1, 455), (1, 456)):
        assert delta[key] != source[key]


def test_two_nubar_samples_differ_only_in_the_multiplicity(nubarRun):
    first = _blocks(str(nubarRun.samples[0]["files"]["endf-delta"]))
    second = _blocks(str(nubarRun.samples[1]["files"]["endf-delta"]))
    differing = {key for key in set(first) | set(second)
                 if first.get(key) != second.get(key)}
    assert differing == {(1, 452), (1, 455), (1, 456)}, (
        f"two samples of the same request differ in {sorted(differing)}")


def test_the_evaluation_is_back_on_the_suite_after_each_sample(nubarRun):
    """A nu-bar realisation replaces the evaluated form, so it has to be put back.

    If it were not, sample 1 would be drawn on top of sample 0's nu-bar and the
    ensemble would drift with the sample index -- and nothing in the output would
    say so. The check is that the two samples' totals differ from each other but
    both stay near the baseline, which a compounding error would not.
    """
    baseline = read_endf(NUBAR_TAPE).get_file(1).sections[452]
    baselineNu = np.interp(1.0e6, np.asarray(baseline.energies, dtype=float),
                           np.asarray(baseline.nubar_values, dtype=float))

    realised = []
    for sample in nubarRun.samples:
        section = read_endf(str(sample["files"]["endf-delta"])).get_file(1).sections[452]
        realised.append(float(np.interp(
            1.0e6, np.asarray(section.energies, dtype=float),
            np.asarray(section.nubar_values, dtype=float))))

    assert realised[0] != realised[1]
    for value in realised:
        assert abs(value - baselineNu) / baselineNu < 0.10


def test_a_gnds_nubar_realisation_sits_beside_its_evaluation(tmp_path):
    """What ``Multiplicity`` becoming a ``Component`` bought, stated as a file.

    Before it, a nu-bar realisation had to **replace** the evaluated form, so a
    GNDS document written from that suite carried the perturbed nu-bar instead of
    the one it was drawn from -- while the same document's cross sections carried
    both. §9.3's ``realization`` style says "this is a draw from that
    evaluation", and it cannot say that about a form that displaced the
    evaluation.
    """
    import re

    run = perturbFromModel(NUBAR_TAPE, {31: [455, 456]}, 1, seed=4,
                           outputDir=tmp_path, formats=("gnds",))
    text = Path(run.paths("gnds")[0]).read_text(encoding="utf-8")

    blocks = re.findall(r"<multiplicity>(.*?)</multiplicity>", text, re.S)
    assert len(blocks) >= 3, "the three family members should all be written"
    withBoth = [b for b in blocks
                if 'label="eval"' in b and 'label="realization-0000"' in b]
    assert len(withBoth) == 3, (
        f"{len(withBoth)} of {len(blocks)} multiplicities carry both forms; the "
        f"evaluation and the realisation have to be in the file together")


def test_the_whole_tape_emitter_writes_the_perturbed_nubar(tmp_path):
    """``label=`` reaches MF1 now, so the tape carries the realisation.

    The chain is the one M0 built for MF3, extended: ``kika.write(..., label=)``
    -> ``_mf1Sections`` -> ``encodeMF1MT452/455/456(..., label=)``. Without it
    the whole-tape emitter would write the *evaluated* nu-bar into a file whose
    cross sections were perturbed -- a tape that is neither the evaluation nor
    the realisation.
    """
    run = perturbFromModel(NUBAR_TAPE, {31: [455, 456]}, 1, seed=4,
                           outputDir=tmp_path, formats=("endf-tape",))
    written = read_endf(str(run.paths("endf-tape")[0]))
    original = read_endf(NUBAR_TAPE)

    for mt in (452, 455, 456):
        before = original.get_file(1).sections[mt]
        after = written.get_file(1).sections[mt]
        probe = 1.0e6
        baseline = np.interp(probe, np.asarray(before.energies, dtype=float),
                             np.asarray(before.nubar_values, dtype=float))
        realised = np.interp(probe, np.asarray(after.energies, dtype=float),
                             np.asarray(after.nubar_values, dtype=float))
        assert realised != baseline, f"MT{mt} came out unperturbed"


def test_a_member_the_draw_did_not_touch_falls_back_to_the_evaluation(tmp_path):
    """A partially perturbed family still writes a whole tape, and says so.

    The rule ``encodeMF3MT`` set: writing only the perturbed members would leave
    holes, refusing would leave the file unwritable, so a member with no form
    under the realisation's label is written from ``eval`` and the report states
    which. Here the family rule perturbs all three anyway -- what is checked is
    that the machinery is the same one, by asking for a tape under a label
    nothing carries.
    """
    from kika import write as kikaWrite
    from kika.endf import read_endf as _read
    from kika.endf.model_adapter import decodeReactionSuite

    suite, _report = decodeReactionSuite(_read(NUBAR_TAPE))
    out = tmp_path / "unlabelled.endf"
    report = kikaWrite(suite, str(out), format="endf", label="realization-0999")

    text = "\n".join(str(w) for w in getattr(report, "warnings", []) or [])
    assert "nu-bar written with the 'realization-0999' form" in text
    assert "fell back" in text
    written = _read(str(out))
    original = _read(NUBAR_TAPE)
    assert (list(written.get_file(1).sections[452].nubar_values)
            == list(original.get_file(1).sections[452].nubar_values)), (
        "nothing carried that label, so the evaluated nu-bar is what should "
        "have been written")


def test_a_gnds_nubar_realisation_comes_back_when_the_file_is_read(tmp_path):
    """Writing both forms is only half of it; the reader had to learn them too.

    ``readMultiplicity`` picked one form and reported the rest -- correct while
    the model held one, and lossy in exactly the case ``Multiplicity`` became a
    ``Component`` for. A document carrying a realisation beside its evaluation
    would have come back with one of the two, and which one depended on
    ``pickOneForm``.
    """
    import kika

    run = perturbFromModel(NUBAR_TAPE, {31: [455, 456]}, 1, seed=4,
                           outputDir=tmp_path, formats=("gnds",))
    suite = kika.read(str(run.paths("gnds")[0]))
    if isinstance(suite, tuple):
        suite = suite[0]

    from kika.nuclear_data.model import EVAL_LABEL

    # Navigated GNDS-side rather than through `nubarNode`: that helper gates on
    # the ENDF provenance -- deliberately, so an MF6 yield is never mistaken for
    # a nu-bar -- and a suite read from GNDS has none. So the prompt nu-bar on
    # the fission product is found the way a GNDS reader finds it.
    nodes = [entry.multiplicity for entry in suite.sums.multiplicitySums]
    for reaction in suite.reactions:
        for product in (getattr(reaction.outputChannel, "products", None) or ()):
            if product.multiplicity is not None and len(product.multiplicity) > 1:
                nodes.append(product.multiplicity)

    assert len(nodes) >= 3, f"only {len(nodes)} multiplicities came back"
    for node in nodes:
        assert sorted(node.keys()) == [EVAL_LABEL, "realization-0000"], (
            f"a multiplicity came back with {sorted(node.keys())}")
        assert node.form is node[EVAL_LABEL]
