"""The writer: the round trip, the schema, and the gaps it refuses to fill.

Three gates, and the third is the one that matters.

1. **The round trip is a fixed point.** ``read → write → read → write`` gives
   two canonically identical trees, and the second model holds the same numbers
   as the first. Byte identity with the *distributed* file is explicitly not
   asserted — FUDGE writes ``1e-5`` where Python's shortest round-tripping repr
   is ``1e-05`` — and the test says so rather than leaving a reader to discover
   it from a failure.

2. **The output validates**, against FUDGE's own GNDS 2.0 schema.

3. **Except where it deliberately does not, and the exception is pinned.** A
   product whose §18 law kika does not read comes out with an empty
   ``<distribution/>``, which the schema rejects. That is the design: writing
   ``<unspecified/>`` there would make the file valid by asserting, on the
   evaluator's behalf, that no distribution was given. So
   :func:`test_the_only_schema_errors_are_the_distributions_phase_7b_will_fill`
   asserts the *count* and *kind* of the remaining errors. When phase 7b lands,
   that test fails and is the place the change gets recorded.
"""
from __future__ import annotations

import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import kika
from kika.gnds.decode import readReactionSuite
from kika.gnds.encode import (_function, chooseFormat, serialise, sha1,
                              writeReactionSuite)
from kika.gnds.version import UnsupportedGndsVersion
from kika.gnds.xpath import Document

#: FUDGE's GNDS 2.0 schemas. Absent on a machine without FUDGE, and the tests
#: that need them skip rather than pass silently. **There are two**: a
#: `covarianceSuite` has no global declaration in `gnds.xsd` at all — §25.1.1
#: makes it a root in its own right and FUDGE ships it its own schema — so
#: validating the sibling against the reaction schema fails with "no matching
#: global declaration" and says nothing about the file.
SCHEMA = Path("/soft_snc/FUDGE/6.10.0/fudge/fudge/gnds.xsd")
COVARIANCE_SCHEMA = Path(
    "/soft_snc/FUDGE/6.10.0/fudge/fudge/covariances/covariances.xsd")

#: The committed evaluations, by conftest fixture name. Between them they carry
#: every node the writer emits. It was three until phase 7b needed witnesses:
#: ``h3`` for ``uncorrelated/angular``'s ``XYs2d``, ``micro_be9`` for
#: ``angularEnergy``, ``s36`` for ``KalbachMann``.
FIXTURES = ("h2_gnds", "micro_fe56_gnds", "micro_ta182_gnds", "h3_gnds",
            "micro_be9_gnds", "s36_gnds")

#: The ones that still write an empty ``<distribution/>``, and the law that
#: makes them. **Named rather than left out of** :data:`FIXTURES`, so that "not
#: yet complete" is a statement in the file instead of an absence nobody
#: notices; the invariant test below runs on all of them and the *fact* test
#: skips exactly these.
#:
#: ``s36`` carries five ``branching3d``, the last unread §18 node. When it
#: lands this becomes empty and both tests run on everything — which is the
#: acceptance gate phase 7b was written around.
INCOMPLETE_FIXTURES = {"s36_gnds": "branching3d"}


@pytest.fixture(params=FIXTURES)
def evaluation(request):
    """Each committed reactionSuite in turn, as a path."""
    return request.getfixturevalue(request.param)


@pytest.fixture(params=FIXTURES)
def completeEvaluation(request):
    """The same, minus the ones :data:`INCOMPLETE_FIXTURES` names.

    A skip and not a filtered list: the skip says *which* file and *which* law
    on the run, so an entry that outlives its reason is visible rather than
    quietly absent from the parametrisation.
    """
    law = INCOMPLETE_FIXTURES.get(request.param)
    if law is not None:
        pytest.skip(f"{request.param} still carries an unread {law}")
    return request.getfixturevalue(request.param)


def _canonical(path: Path) -> str:
    """One file as a canonical string — attribute order and whitespace removed.

    ``ET.canonicalize`` is XML C14N, so it settles attribute order and the
    insignificant whitespace ``ET.indent`` produced. What it does *not* settle
    is number spelling, which is why :func:`kika.gnds.encode._number` has to be
    exactly round-tripping rather than merely close.
    """
    return ET.canonicalize(from_file=str(path), strip_text=True)


def _write(suite, directory: Path, name: str = "out.gnds.xml"):
    path = directory / name
    report = kika.write(suite, path)
    return path, report


# ---------------------------------------------------------------------------
# 1. the fixed point
# ---------------------------------------------------------------------------

def test_writing_twice_gives_the_same_file(evaluation, tmp_path):
    """``write → read → write`` is a fixed point on the tree.

    Not compared against the *distributed* file: kika drops ``documentation``
    and ``applicationData``, and FUDGE's number spelling is its own. What must
    hold is that the writer adds nothing and loses nothing on a second pass —
    if it did, every round trip would drift.
    """
    first, _ = _write(kika.read(evaluation, covariances=False),
                      tmp_path, "first.xml")
    second, _ = _write(kika.read(first, covariances=False),
                       tmp_path, "second.xml")
    assert _canonical(first) == _canonical(second)


def test_every_cross_section_survives_the_round_trip(evaluation, tmp_path):
    """The numbers, not the tree. Compared with ``assert_array_equal``: both
    sides are the same decimal text read into the same doubles, so anything
    short of exact means the writer changed a value."""
    before = kika.read(evaluation, covariances=False)
    path, _ = _write(before, tmp_path)
    after = kika.read(path, covariances=False)

    assert after.reactions.ENDF_MTs == before.reactions.ENDF_MTs
    assert after.sums.ENDF_MTs == before.sums.ENDF_MTs
    for mt in before.reactions.ENDF_MTs:
        for label in before.reactions[mt].crossSection:
            original = before.reactions[mt].crossSection[label]
            written = after.reactions[mt].crossSection[label]
            assert type(written) is type(original), f"MT{mt} form {label!r}"
            for xs, ys in _pointwisePairs(original, written):
                np.testing.assert_array_equal(xs, ys, err_msg=f"MT{mt} {label!r}")


def _pointwisePairs(original, written):
    """``(array, array)`` for every tabulated curve inside two matching forms."""
    from kika.nuclear_data.model import (Regions1d, ResonancesWithBackground,
                                         XYs1d)

    if isinstance(original, ResonancesWithBackground):
        assert original.resonanceRegionHref == written.resonanceRegionHref
        for name in original.background.regions:
            left = getattr(original.background, name)
            right = getattr(written.background, name)
            assert (left is None) == (right is None), name
            if left is not None:
                yield from _pointwisePairs(left, right)
        return
    if isinstance(original, XYs1d):
        yield original.xs, written.xs
        yield original.ys, written.ys
        return
    if isinstance(original, Regions1d):
        assert len(original.function1ds) == len(written.function1ds)
        for a, b in zip(original.function1ds, written.function1ds):
            yield from _pointwisePairs(a, b)


def test_the_resonance_parameters_survive_the_round_trip(micro_fe56_gnds,
                                                         tmp_path):
    """312 resonances, two channels each, through the table and back.

    The table is the writer's most dangerous node: a wrong ``columnIndex`` or a
    wrong header order produces a file holding every number the model held, in
    the wrong columns, which reads back as a *different evaluation* rather than
    as an error.
    """
    before = kika.read(micro_fe56_gnds, covariances=False)
    path, _ = _write(before, tmp_path)
    after = kika.read(path, covariances=False)

    left = before.resonances.resolved[0].formalism
    right = after.resonances.resolved[0].formalism
    assert right.approximation == left.approximation
    assert right.boundaryCondition == left.boundaryCondition
    assert right.numberOfResonances == left.numberOfResonances == 312
    for a, b in zip(left.spinGroups, right.spinGroups):
        assert (b.spin, b.parity, b.label) == (a.spin, a.parity, a.label)
        assert [(c.resonanceReaction, c.L, c.channelSpin, c.columnIndex)
                for c in b.channels] == \
               [(c.resonanceReaction, c.L, c.channelSpin, c.columnIndex)
                for c in a.channels]
        np.testing.assert_array_equal(b.energies, a.energies)
        np.testing.assert_array_equal(b.widths, a.widths)


def test_the_unresolved_averages_survive_the_round_trip(micro_ta182_gnds,
                                                        tmp_path):
    before = kika.read(micro_ta182_gnds, covariances=False)
    path, _ = _write(before, tmp_path)
    after = kika.read(path, covariances=False)

    left = before.resonances.unresolved.tabulatedWidths
    right = after.resonances.unresolved.tabulatedWidths
    assert right.selfShieldingOnly == left.selfShieldingOnly
    np.testing.assert_array_equal(right.energyGrid, left.energyGrid)
    assert [(g.L, g.J) for g in right.spinGroups] == \
           [(g.L, g.J) for g in left.spinGroups]
    for a, b in zip(left.spinGroups, right.spinGroups):
        np.testing.assert_array_equal(b.levelSpacing, a.levelSpacing)
        for x, y in zip(a.channels, b.channels):
            assert (y.label, y.degreesOfFreedom) == (x.label, x.degreesOfFreedom)
            np.testing.assert_array_equal(y.widths, x.widths)


def test_the_breit_wigner_table_keeps_its_l_blocks_and_their_order(
        micro_ta182_gnds, tmp_path):
    before = kika.read(micro_ta182_gnds, covariances=False)
    path, _ = _write(before, tmp_path)
    after = kika.read(path, covariances=False)

    left = before.resonances.resolved[0].formalism.resonanceParameters
    right = after.resonances.resolved[0].formalism.resonanceParameters
    assert [g.L for g in right.spinGroups] == [g.L for g in left.spinGroups]
    for a, b in zip(left.spinGroups, right.spinGroups):
        assert [r.toFlat() for r in b.resonances] == \
               [r.toFlat() for r in a.resonances]


def test_the_covariance_matrices_survive_the_round_trip(h2_gnds, tmp_path):
    """Including the lower-triangular packing, which the writer chooses and the
    reader has to mirror back."""
    before = kika.read(h2_gnds)
    path, _ = _write(before, tmp_path)
    after = kika.read(path)

    assert after.covarianceSuite is not None
    assert len(after.covarianceSuite) == len(before.covarianceSuite)
    for a, b in zip(before.covarianceSuite.covarianceSections,
                    after.covarianceSuite.covarianceSections):
        assert b.label == a.label
        assert b.rowData.ENDF_MFMT == a.rowData.ENDF_MFMT
        for left, right in zip(_matrices(a.form), _matrices(b.form)):
            np.testing.assert_array_equal(right.matrix, left.matrix)
            np.testing.assert_array_equal(right.rowGrid, left.rowGrid)


def _radii(path):
    """Every radius in a written file, as ``(tag, value, unit of axis 0)``."""
    found = []
    for element in ET.parse(path).getroot().iter():
        if element.tag not in ("scatteringRadius", "hardSphereRadius"):
            continue
        for form in element:
            units = [axis.attrib.get("unit") for axis in form.iter("axis")
                     if axis.attrib.get("index") == "0"]
            found.append((element.tag, form.attrib.get("value"),
                          units[0] if units else None))
    return found


def test_an_endf_radius_is_never_labelled_fm(micro_tape, tmp_path):
    """ENDF's AP is in units of 10^-12 cm and GNDS's radius is in fm.

    The writer's rule has always been "the unit as the model holds it, or not
    at all" — but two call sites wrote ``unit="fm"`` outright, and an
    ENDF-sourced radius reaching them came out a factor of ten wrong *as a
    statement of the file*, which no schema and no round trip can see. Measured
    on this tape before the fix: ``0.5002`` labelled ``fm``, where FUDGE's own
    conversion of the same evaluation writes ``5.002 fm``.

    The numbers are asserted too, so that a later decision to canonicalise to
    fm has to come here and say so rather than arrive as a silent rescale.
    """
    path, _ = _write(kika.read(micro_tape, covariances=False), tmp_path)
    radii = _radii(path)

    assert radii, "the tape's resonance block carries radii; none were written"
    assert {unit for _, _, unit in radii} == {""}, radii
    assert {value for _, value, _ in radii} == {"0.5444", "0.5002"}, radii


def test_a_gnds_radius_keeps_the_unit_it_came_with(micro_fe56_gnds, tmp_path):
    """The other half: the model carries the unit, so a file that declares one
    still declares it after a round trip.

    Without this the fix above would be a regression dressed as honesty — every
    radius written with an empty unit, including the ones whose unit was
    *stated* in the file kika had just read.
    """
    before = _radii(micro_fe56_gnds)
    path, _ = _write(kika.read(micro_fe56_gnds, covariances=False), tmp_path)

    assert {unit for _, _, unit in before} == {"fm"}, before
    assert _radii(path) == before


def _matrices(form):
    from kika.nuclear_data.model import CovarianceMatrix, Mixed

    if isinstance(form, CovarianceMatrix):
        yield form
    elif isinstance(form, Mixed):
        for component in form.components:
            yield from _matrices(component)


# ---------------------------------------------------------------------------
# 2 and 3. the schema, and the one gap left in it
# ---------------------------------------------------------------------------

def _schemaErrors(path: Path, schema: Path = SCHEMA):
    if not schema.exists() or shutil.which("xmllint") is None:
        pytest.skip(f"{schema} or xmllint is not on this machine")
    result = subprocess.run(
        ["xmllint", "--noout", "--huge", "--schema", str(schema), str(path)],
        capture_output=True, text=True,
    )
    return [line.split("Schemas validity error : ")[1]
            for line in result.stderr.splitlines()
            if "Schemas validity error : " in line]


def test_the_only_schema_errors_are_the_nodes_phase_7b_will_fill(
        evaluation, tmp_path):
    """Every node the writer emits validates, except the ones it leaves empty.

    A product whose §18 law kika does not read gets an empty
    ``<distribution/>``, and one whose §17.3 multiplicity form it does not
    model gets an empty ``<multiplicity/>``. Both are deliberate and both are
    invalid, which is the point: the file announces its own incompleteness to
    any validator rather than asserting, on the evaluator's behalf, that
    nothing was given. This test says those are the *only* things wrong with
    the file, so the day the last one lands the count goes to zero and this is
    what notices.

    **The multiplicity half was not here until ``s36`` was committed**, and its
    absence was not a decision — no committed fixture carried a ``branching1d``,
    ``reference`` or ``unspecified`` multiplicity, so a whole second family of
    deliberate invalidity had never been written by a test. 14 032 + 3 539 + 178
    occurrences across the distribution say it is not the rare half.
    """
    suite = kika.read(evaluation, covariances=False)
    path, report = _write(suite, tmp_path)

    errors = _schemaErrors(path)
    assert all("Element 'distribution': Missing child element" in error
               or "Element 'multiplicity': Missing child element" in error
               for error in errors), errors

    products = [product for reaction in _everyReaction(suite)
                for product in _everyProduct(reaction.outputChannel)]
    withoutDistribution = sum(
        1 for product in products
        if product.distribution is None or len(product.distribution) == 0
    )
    withoutMultiplicity = sum(
        1 for product in products
        if product.multiplicity is not None
        and product.multiplicity.constant is None
        and product.multiplicity.function is None
    )
    assert len(errors) == withoutDistribution + withoutMultiplicity, errors

    if withoutDistribution:
        assert any(str(withoutDistribution) in entry
                   for entry in report.unsupported), report.unsupported


def _everyReaction(suite):
    for container in (suite.reactions, suite.orphanProducts, suite.sums,
                      suite.fissionComponents, suite.productions,
                      suite.incompleteReactions):
        yield from container


def _everyProduct(channel):
    for product in channel.products:
        yield product
        if product.outputChannel is not None:
            yield from _everyProduct(product.outputChannel)


def test_every_committed_gnds_fixture_now_validates_completely(
        completeEvaluation, tmp_path):
    """Phase 7b's acceptance gate, and the number it moves.

    Before ``uncorrelated`` landed the first three fixtures wrote 3, 1 and 21
    empty ``<distribution/>`` elements — 25 schema errors that were deliberate,
    one per product whose law kika could not read. All 25 were the same law. The
    sibling test above still holds the *invariant* (whatever errors there are,
    they are only that kind, and their count matches the model); this one holds
    the **fact**, so that a law regressing into unread shows up as a failure
    here rather than as a silently-true invariant.

    It runs on every fixture except those :data:`INCOMPLETE_FIXTURES` names,
    which is one: ``s36`` and its five ``branching3d``.
    """
    suite = kika.read(completeEvaluation, covariances=False)
    path, _ = _write(suite, tmp_path)
    assert _schemaErrors(path) == []


def test_an_unreadable_distribution_is_left_empty_and_never_called_unspecified(
        h2_gnds, tmp_path):
    """The one substitution that would make the file valid, and is a forgery.

    ``<unspecified/>`` is the **evaluator** stating that no distribution is
    given. Writing it where kika merely cannot read the law yet would put that
    statement in their mouth, and nothing downstream could tell the difference.

    **The subject moved when phase 7b landed ``uncorrelated``.** H-2's three
    empty ``<distribution/>`` elements were its three ``uncorrelated`` laws;
    they are read now, so the doctrine needs a law that is still unread. One is
    planted here — an ``<evaporation>``, one of the six analytic §18.3 spectra
    (``gnds.xsd:1697-1709``) — and it exercises a second rule at the same time:
    the reader keeps the angular half it *could* read, and the writer refuses
    to emit a one-child ``<uncorrelated>`` because ``gnds.xsd:1677-1680`` is an
    ``xs:sequence``. A half node would validate against nothing and read as a
    complete statement; the empty element does not, and says so.
    """
    tree = ET.parse(h2_gnds)
    energy = tree.getroot().find(".//uncorrelated/energy")
    for child in list(energy):
        energy.remove(child)
    ET.SubElement(energy, "evaporation")
    planted = tmp_path / "evaporation.gnds.xml"
    tree.write(planted)

    suite = kika.read(planted, covariances=False)
    path, report = _write(suite, tmp_path)

    root = ET.parse(path).getroot()
    empty = [d for d in root.iter("distribution") if len(d) == 0]
    assert len(empty) == 1
    assert list(root.iter("uncorrelated")) != [], (
        "the other two uncorrelated laws must still be written; this test is "
        "about the one that could not be read, not about the law"
    )
    # H-2 *does* have a genuine `unspecified`, on its production channel, and
    # it survives — so the absence above is not the writer dropping the node.
    assert len(list(root.iter("unspecified"))) == 2
    assert any("does not validate" in entry for entry in report.unsupported)
    assert any("requires both children" in entry
               for entry in report.unsupported)
    errors = _schemaErrors(path)
    assert len(errors) == 1
    assert "Element 'distribution': Missing child element" in errors[0]


def test_a_form_the_writer_cannot_serialise_is_counted_as_incomplete(
        micro_fe56_gnds, tmp_path):
    """The half of the doctrine that was not being announced.

    ``declareWhatIsMissing`` puts the file's own "this does not validate" line
    in the report, and it fires off ``incompleteProducts``. That list was built
    from ``len(distribution) == 0`` — the *model* dict — so a product whose one
    form the writer has no serialisation for produced a childless
    ``<distribution/>`` and **no announcement**: an invalid file that claimed
    to be complete. Counting the written element instead is what closes it.
    """
    class _UnknownLaw:
        productFrame = "lab"

    suite = kika.read(micro_fe56_gnds, covariances=False)
    product = next(p for reaction in _everyReaction(suite)
                   for p in _everyProduct(reaction.outputChannel)
                   if p.distribution is not None and len(p.distribution))
    product.distribution.forms = {"eval": _UnknownLaw()}

    path, report = _write(suite, tmp_path)
    assert any("no serialisation for a _UnknownLaw" in entry
               for entry in report.unsupported)
    assert any("does not validate" in entry for entry in report.unsupported)
    root = ET.parse(path).getroot()
    assert [d for d in root.iter("distribution") if len(d) == 0]


def test_the_covariance_sibling_validates_too(h2_gnds, tmp_path):
    suite = kika.read(h2_gnds)
    path, _ = _write(suite, tmp_path)
    sibling = tmp_path / "Covariances" / "out.gnds-covar.xml"
    assert sibling.is_file()
    assert _schemaErrors(sibling, COVARIANCE_SCHEMA) == []


def test_a_suite_holding_a_mixed_can_be_written_at_all(gnds_data_dir, tmp_path):
    """Writing **any** ``mixed`` raised ``AttributeError`` until 2026-08-17.

    ``_covarianceForm`` asked the form for a ``label`` that the model class did
    not carry, and ``covariances.xsd:135-142`` makes that attribute
    ``use="required"`` — so the field was missing from the model, dropped by the
    reader and demanded by the writer, three ways of getting one attribute
    wrong. Nothing saw it because ``h2_gnds`` is the only covariance file in the
    round trip and it holds no ``mixed``; F-19 holds four and La-139 thirteen,
    and both have been committed since phase 5.

    A ``covarianceSuite`` is a root in its own right (§25.1.1), so this goes
    through ``writeCovarianceSuite`` directly rather than through ``kika.write``
    — the committed covariance fixtures have no ``reactionSuite`` beside them.
    """
    from kika.gnds.covariances import readCovarianceSuite
    from kika.gnds.encode import writeCovarianceSuite

    source = gnds_data_dir / "Covariances/n-009_F_019.endf.gnds-covar.xml"
    suite, _ = readCovarianceSuite(Document.parse(source))

    tree, _report = writeCovarianceSuite(suite, "2.0")
    path = tmp_path / "f19.gnds-covar.xml"
    path.write_bytes(serialise(tree))

    written = [element.attrib.get("label")
               for element in ET.parse(path).getroot().iter("mixed")]
    original = [element.attrib.get("label")
                for element in ET.parse(source).getroot().iter("mixed")]
    assert written == original, "the mixed labels did not survive the round trip"
    assert all(label for label in written), "a required attribute went missing"

    assert _schemaErrors(path, COVARIANCE_SCHEMA) == []


#: The schema errors an **ENDF-decoded** suite still has, by kind. Measured
#: 2026-08-17 at twelve and written down in `docs/library-gaps.md` D20; seven
#: since the MF4 angular `axes`, the single-region containers and the evaluated
#: style's domain landed. None of them is the deliberate empty
#: `<distribution/>` the GNDS fixtures produce -- these are gaps in the
#: ENDF -> model -> GNDS path itself.
#:
#: **Every survivor is now a decision rather than a defect**, which is what
#: makes the count worth pinning instead of chasing to zero: what an ENDF
#: `AUG-2018` becomes, what `genre` an ENDF output channel has, where a
#: `scatteringRadius` goes when `RMatrixType` admits no such child (§4.1), and
#: what an output channel with no products should say. The five that were
#: writer defects are gone.
ENDF_SCHEMA_GAPS = (
    "Element 'evaluated', attribute 'date'",
    "Element 'scatteringRadius': This element is not expected",
    "Element 'outputChannel': The attribute 'genre' is required but missing",
    "Element 'products': Missing child element",
    "Element 'multiplicity': Missing child element",
)


def _endfSchemaErrors(suite, tmp_path, name="out.xml"):
    path, _ = _write(suite, tmp_path, name)
    return path, _schemaErrors(path)


def test_the_endf_decoded_suites_schema_errors_are_pinned(micro_tape, tmp_path):
    """The gate that was missing, and it is why D19 lived for a phase.

    ``test_the_only_schema_errors_are_the_distributions_phase_7b_will_fill``
    walks the three **GNDS** fixtures, and the GNDS reader never produces a bare
    ``Isotropic2d`` -- so nothing ever validated a file written from an ENDF
    decode, which is the direction that had an invalid node in it.

    Seven errors survive today and every one is a known ENDF->GNDS gap
    (:data:`ENDF_SCHEMA_GAPS`). Pinned by count and kind, the same shape as the
    GNDS half: fixing one lowers the number and this test is what notices.

    **The count is not monotone downwards, and that is the interesting part.**
    An absent required child makes everything below it unreachable, so a fix can
    raise the number by exposing what it was hiding. The MF4 angular ``axes``
    did exactly that: one error was standing in front of 3 999.
    """
    _path, errors = _endfSchemaErrors(kika.read(micro_tape, covariances=False),
                                      tmp_path)
    unknown = [e for e in errors
               if not any(kind in e for kind in ENDF_SCHEMA_GAPS)]
    assert unknown == [], unknown
    assert len(errors) == 7


#: D25. What is wrong with a ``covarianceSuite`` **written from an ENDF decode**,
#: by kind. **Empty, and it was two.** Both were the same shape of hole — an
#: attribute §25 makes required that the ENDF side has no concept of and
#: therefore never set — and both are closed:
#:
#: ``covarianceMatrix/@label``, 4 on Fe-56 and 2 on the PFNS tape, was the hole
#: §25.3 had already patched one level down for ``parameterCovarianceMatrix``.
#: ``covariances.xsd`` requires the attribute on **all four** forms, so
#: ``encode._formLabel`` now fills it from the style ``writeCovarianceSuite``
#: synthesises, for all four rather than for the one that was measured.
#: ``covarianceSuite/@target``, 1 on each, was not a matrix attribute at all:
#: the root saying what nuclide the file is about. It comes from the
#: ``reactionSuite`` through ``decodeCovarianceSuite(target=…)``, the same route
#: ``evaluation`` already took, because ENDF has no such string and the two
#: documents of one pair must not derive it twice.
#:
#: The tuple stays rather than being deleted with the last entry: it is what a
#: newly-introduced gap would be measured against, and an empty tuple asserts
#: something an absent one does not.
ENDF_COVARIANCE_SCHEMA_GAPS = ()

#: ``(fixture name, error count)``. Two tapes rather than one because MF35 is a
#: different route into the same writer — ``micro_pfns_cov`` reaches
#: ``_covarianceMatrix`` through ``decodeMF35MT`` where ``micro_fe56_cov``
#: reaches it through MF33 and MF34 — and a gap that only one of them shows is
#: worth telling apart from one both do. Today they show the same two kinds.
ENDF_COVARIANCE_TAPES = (("micro_cov_tape", 0), ("micro_pfns_cov_tape", 0))


def _endfCovarianceSchemaErrors(tape, tmp_path):
    """Write the pair and validate **the sibling**, which is the whole point.

    ``kika.write`` emits two documents (``_write.py:_writeLinkedPair``): the
    evaluation, and ``Covariances/<name>-covar.xml`` beside it. Everything that
    validated an ENDF-decoded file so far validated the first one — and read it
    with ``covariances=False``, so the second was never even built.
    """
    suite = kika.read(tape)
    path = tmp_path / "out.gnds.xml"
    kika.write(suite, path)
    sibling = path.parent / "Covariances" / "out.gnds-covar.xml"
    assert sibling.is_file(), f"kika.write emitted no covariance sibling at {sibling}"
    return sibling, _schemaErrors(sibling, COVARIANCE_SCHEMA)


@pytest.mark.parametrize("fixture,expected", ENDF_COVARIANCE_TAPES)
def test_the_endf_decoded_covariance_sibling_is_pinned(fixture, expected,
                                                       request, tmp_path):
    """D25. The gate that was missing one level below the gate that was missing.

    ``test_the_endf_decoded_suites_schema_errors_are_pinned`` closed D20 — "no
    test validates a suite decoded from ENDF" — and it closed it for the
    ``reactionSuite`` only: it reads with ``covariances=False`` and validates
    against ``gnds.xsd``. A ``covarianceSuite`` is a **root in its own right**
    (§25.1.1) with **its own schema**, so the sibling ``kika.write`` puts beside
    every evaluation that has covariances was never looked at by anything. The
    same hole, one level further out, surviving the fix for itself.

    This test landed **declaring the damage rather than repairing it**, which is
    the opposite order from D20 and is why it went first: a number that is
    pinned can be lowered by anyone, and a number nobody measures is what let
    D19 live for a phase. It was pinned at 5 and 3 for exactly one commit; both
    kinds are described in :data:`ENDF_COVARIANCE_SCHEMA_GAPS`, which is now
    empty.

    **The count is not monotone downwards.** An absent required *child* makes
    its whole subtree unreachable to the validator, so repairing one can raise
    the count by exposing what it was standing in front of — the MF4 angular
    ``axes`` hid 3 999 errors behind one. Both gaps here were *attributes*, which
    do not hide a subtree, and they did fall cleanly — 5 → 0 and 3 → 0. That was
    a property of those two and is not a rule: the next entry here may well go
    up before it goes down.
    """
    tape = request.getfixturevalue(fixture)
    _sibling, errors = _endfCovarianceSchemaErrors(tape, tmp_path)

    unknown = [e for e in errors
               if not any(kind in e for kind in ENDF_COVARIANCE_SCHEMA_GAPS)]
    assert unknown == [], unknown
    assert len(errors) == expected


def test_the_covariance_sibling_says_the_same_target_as_the_evaluation(
        micro_cov_tape, tmp_path):
    """D25, half one. The pair must not disagree about what it is about.

    ENDF has no target *string* anywhere: the nuclide name is derived from
    MF1/451's ZA through PoPs, and ``decodeReactionSuite`` has already done that
    work. So the covariance suite is handed the answer instead of re-deriving
    it — a second derivation is a second chance for the two documents of one
    pair to name different nuclides, and a reader following the ``externalFile``
    from one to the other would have no way to tell which was right.

    Asserted **against the evaluation's own value**, not against a literal:
    ``target`` is ``'unknown'`` on this micro-tape, which is what
    ``decode.py:235`` says when PoPs comes back empty. Mirroring that is
    correct; inventing a nuclide name to make the file look finished would not
    be.
    """
    suite = kika.read(micro_cov_tape)
    path = tmp_path / "out.gnds.xml"
    kika.write(suite, path)

    evaluation = ET.parse(path).getroot()
    sibling = ET.parse(path.parent / "Covariances" / "out.gnds-covar.xml").getroot()

    assert sibling.attrib["target"] == evaluation.attrib["target"]
    assert sibling.attrib["target"] == suite.target
    assert sibling.attrib["projectile"] == evaluation.attrib["projectile"]


def test_a_covariance_suite_decoded_without_a_target_says_so(micro_cov_tape):
    """The consequence is announced where the absence is, not where it bites.

    ``decodeCovarianceSuite`` is reachable without a ``reactionSuite`` — the
    samplers at ``sampling/endf_perturbation.py:663`` and
    ``sampling/mf35_sampling.py:177`` call it that way, and they are right to:
    they want the matrices, not a document. But the suite they get back cannot
    be written as valid GNDS, and without this line the next caller learns that
    from ``xmllint``, three steps later and in another session.

    The signature keeps its default rather than making ``target`` required: the
    samplers have no reaction suite to take one from, and breaking them to
    protect a writer they never call would be the wrong trade.
    """
    from kika.endf.model_adapter.covariances import decodeCovarianceSuite
    from kika.endf.read_endf import read_endf

    endf = read_endf(str(micro_cov_tape))

    _suite, report = decodeCovarianceSuite(endf)
    assert any("cannot be written as valid GNDS" in entry
               for entry in report.warnings), report.warnings

    _suite, report = decodeCovarianceSuite(endf, target="Fe56")
    assert not [entry for entry in report.warnings
                if "cannot be written as valid GNDS" in entry], report.warnings


def test_every_endf_built_covariance_form_gets_the_label_the_schema_requires(
        micro_cov_tape, tmp_path):
    """D25, half two — and the fix is for four forms, not for the one measured.

    ``covariances.xsd`` marks ``label`` required on ``covarianceMatrix``
    (:118), ``shortRangeSelfScalingVariance`` (:126), ``mixed`` (:135) and
    ``sum`` (:143). Only the first occurs in an ENDF decode today, so only the
    first was in the error count — patching just that one would leave three
    forms one MF away from reintroducing D25 with nothing to notice.

    The approximation is **reported**, which is the part that matters more than
    the attribute: a reader of the written file cannot tell a style label kika
    invented from one an evaluator chose, and the report is the only place that
    difference exists.
    """
    from kika.gnds.encode import SYNTHESISED_STYLE_LABEL

    suite = kika.read(micro_cov_tape)
    path = tmp_path / "out.gnds.xml"
    report = kika.write(suite, path)

    sibling = ET.parse(path.parent / "Covariances" / "out.gnds-covar.xml").getroot()
    forms = [element for element in sibling.iter()
             if element.tag in ("covarianceMatrix", "mixed", "sum",
                                "shortRangeSelfScalingVariance")]
    assert forms, "the tape has covariances and none of them reached the file"
    for form in forms:
        assert form.attrib.get("label") == SYNTHESISED_STYLE_LABEL, form.attrib

    assert sum("carried no label" in entry
               for entry in report.approximations) == len(forms), (
        report.approximations)


def test_a_bare_isotropic2d_goes_where_the_schema_admits_it(micro_tape, tmp_path):
    """D19. ``<isotropic2d>`` is not a child of ``<distribution>``.

    ``kika/endf/model_adapter/angular.py:146`` returns a **bare**
    ``Isotropic2d`` for every MF4 with LTT=0, and the writer used to emit it
    straight under ``<distribution>`` with a ``label`` and a ``productFrame``.
    ``DistributionType`` (``gnds.xsd:1647-1662``) has no such child and
    ``DistributionIsotropic2dType`` (``:1693``) has no attributes at all, so the
    node was invalid twice over -- and kika's own reader dropped it into the
    report, so an isotropic ENDF distribution vanished between the two halves.

    Built by hand rather than from a fixture: no committed micro-tape carries an
    LTT=0 section, and minting one to reach a single branch is worse than
    stating the shape here.

    **Compared against the same suite unmodified**, not against zero: the ENDF
    path still has the twelve gaps above, and the claim being made is the narrow
    one -- putting the isotropic distribution in adds no schema error and comes
    back on a re-read.
    """
    from kika.nuclear_data.model import Frame, Isotropic2d

    before = kika.read(micro_tape, covariances=False)
    _path, baseline = _endfSchemaErrors(before, tmp_path, "baseline.xml")

    suite = kika.read(micro_tape, covariances=False)
    product = suite.reactions[2].outputChannel.products[0]
    product.distribution.forms.clear()
    product.distribution["eval"] = Isotropic2d(label="eval",
                                               productFrame=Frame.centerOfMass)
    path, errors = _endfSchemaErrors(suite, tmp_path, "isotropic.xml")

    root = ET.parse(path).getroot()
    stray = [d for d in root.iter("distribution")
             if d.find("isotropic2d") is not None]
    assert not stray, "an <isotropic2d> is still a direct child of <distribution>"
    wrapped = [b for b in root.iter("angularTwoBody")
               if b.find("isotropic2d") is not None]
    assert wrapped, "the isotropic distribution was not written at all"
    assert wrapped[0].find("isotropic2d").attrib == {}, (
        "DistributionIsotropic2dType has no attributes"
    )
    assert set(errors) - set(baseline) == set(), (
        "writing the isotropic distribution added a schema error"
    )
    # It used to remove one as well, because the LTT=3 section it replaces was
    # written with no `axes` at all -- and an absent required first child is
    # what xmllint reports as "this element is not expected. Expected is
    # ( axes )", which is where the D20 note's "written after the function1ds"
    # came from. The writer always emitted axes before the container; the model
    # simply had none to give. Now that it does, replacing that section removes
    # nothing, and the delta is zero.
    assert len(errors) == len(baseline)

    # And the half that says it is not merely valid: kika reads it back.
    after = kika.read(path, covariances=False)
    form = after.reactions[2].outputChannel.products[0].distribution["eval"]
    assert isinstance(form.angular, Isotropic2d), form


def test_where_a_functional_sits_decides_which_attribute_it_declares(tmp_path):
    """§5-6: ``index``, ``outerDomainValue`` and ``label`` are exclusive.

    ``gnds.xsd:2109-2245`` gives a functional three types by position and each
    admits exactly one of the three attributes: a child of a *regions*
    container must declare ``index`` and may not declare ``outerDomainValue``,
    a child of an ``XYs2d`` the other way round, and the head of a container
    carries ``label`` and the ``axes``.

    The writer used to emit whichever of the three the model happened to hold,
    which put an ``index`` on all 3 960 ``Legendre`` children of the Fe-56 LTT=3
    section. Nothing said so because the container above them had no ``axes``,
    and an absent required child makes everything under it unreachable.

    Built by hand rather than from a tape: the rule is about *position*, so a
    synthetic two-by-two — a ``Regions2d`` of two ``XYs2d`` of two ``Legendre``
    — states it in one place and survives any fixture changing underneath it.
    The model deliberately carries **both** attributes on every node, so the
    test fails if the writer copies rather than decides.
    """
    from kika.nuclear_data.model import (ConversionReport, Legendre, Regions2d,
                                         XYs2d, angularAxes)

    axes = angularAxes()

    def series(outer, index):
        return Legendre(coefficients=np.array([1.0, 0.1]),
                        outerDomainValue=outer, index=index, axes=axes)

    regions = Regions2d(axes=axes, function2ds=[
        XYs2d(axes=axes, outerDomainValue=float(n), index=n,
              function1ds=[series(1.0 + n, 0), series(2.0 + n, 1)])
        for n in (0, 1)
    ])
    root = ET.Element("root")
    report = ConversionReport()
    _function(root, regions, report, "test")

    written = root.find("regions2d")
    assert written.find("axes") is not None, "the container head carries the axes"

    for position, child in enumerate(written.find("function2ds")):
        assert child.attrib.get("index") == str(position), child.attrib
        assert "outerDomainValue" not in child.attrib, (
            "a child of a regions container is an xData_XYs2d_inRegions")
        assert child.find("axes") is None, "§5.1.1 has a child inherit its axes"

        for grandchild in child.find("function1ds"):
            assert "outerDomainValue" in grandchild.attrib, grandchild.attrib
            assert "index" not in grandchild.attrib, (
                "a child of an XYs2d is an xData_Legendre_1d, which has no index")

    # The axes are one shared object all the way down, so nothing is reported
    # lost -- `_axesUnlessNested` decides inheritance by identity, and a second
    # equal-but-distinct Axes would be flagged on every region instead.
    assert not [entry for entry in report.losses if "axes of its own" in entry], (
        report.losses)


def _xys3d():
    """A hand-built ``XYs3d``, axes shared by identity all the way down.

    Hand-built because **no file within reach carries one**: no committed
    fixture has an ``XYs3d``, and the only registered tape that holds an
    ``energyAngular`` is ``fe56_gnds``, which is 18.8 MB on the share and
    marked ``tape``. So this pairs kika against kika's own emitter and against
    the schema, and against no witness in distributed data.
    """
    from kika.nuclear_data.model import Axes, Axis, XYs1d, XYs2d, XYs3d

    axes = Axes([
        Axis(index=3, label="energy_in", unit="eV"),
        Axis(index=2, label="energy_out", unit="eV"),
        Axis(index=1, label="mu", unit=""),
        Axis(index=0, label="P(mu,energy_out|energy_in)", unit=""),
    ])

    def spectrum(outer, first):
        return XYs2d(axes=axes, outerDomainValue=outer, function1ds=[
            XYs1d(xs=np.array([-1.0, 1.0]), ys=np.array([first, first + 1.0]),
                  outerDomainValue=inner, axes=axes)
            for inner in (1.0e4, 2.0e4)
        ])

    return XYs3d(
        axes=axes,
        interpolationQualifier="unitBase",
        # 1 MeV twice: the outermost axis may repeat a value for the same
        # reason the 2-d one may, and the list must survive the round trip.
        function2ds=[spectrum(1.0e6, 0.0), spectrum(1.0e6, 2.0),
                     spectrum(2.0e6, 4.0)],
    )


def test_an_xys3d_writes_the_shape_the_schema_asks_for():
    """§6.5's container, and the ``axes`` identity decided over **two** levels.

    ``_axesUnlessNested`` asks ``form.axes is parentAxes``. For a 3-d form the
    same object has to reach the 1-d grandchildren, so a fresh ``Axes`` built
    anywhere on the way down would put "carries axes of its own" in the report
    for every nested node and write a file the schema rejects. This is the 3-d
    twin of :func:`test_a_regions_container_indexes_its_children`.
    """
    from kika.nuclear_data.model import ConversionReport

    root = ET.Element("root")
    report = ConversionReport()
    _function(root, _xys3d(), report, "test")

    written = root.find("XYs3d")
    assert written.find("axes") is not None, "the container head carries the axes"
    assert written.attrib["interpolationQualifier"] == "unitbase", (
        "FUDGE's spelling, as for XYs2d")

    children = list(written.find("function2ds"))
    assert [float(c.attrib["outerDomainValue"]) for c in children] == [1e6, 1e6, 2e6]
    for child in children:
        # `function2ds` (gnds.xsd:2253) is a choice of xData_XYs2d and
        # xData_regions_2d; both require `outerDomainValue` and neither has an
        # `index` attribute at all.
        assert "index" not in child.attrib, child.attrib
        assert child.find("axes") is None, "§5.1.1 has a child inherit its axes"
        for grandchild in child.find("function1ds"):
            assert "outerDomainValue" in grandchild.attrib, grandchild.attrib
            assert grandchild.find("axes") is None

    assert not [entry for entry in report.losses if "axes of its own" in entry], (
        report.losses)


def test_an_xys3d_survives_model_to_xml_to_model():
    """The honest gate for this node: our reader against our writer.

    What must survive is what a 3-d container is *for* — the outermost grid
    including its repeat, the qualifier, and the one shared ``Axes`` object the
    writer decides inheritance by.
    """
    from kika.gnds.primitives import readForm
    from kika.nuclear_data.model import ConversionReport, InterpolationQualifier

    before = _xys3d()
    root = ET.Element("root")
    _function(root, before, ConversionReport(), "test")
    after = readForm(root.find("XYs3d"))

    assert after.outerDomainValues == before.outerDomainValues == [1e6, 1e6, 2e6]
    assert after.interpolationQualifier is InterpolationQualifier.unitBase
    assert after.axes == before.axes
    for child in after:
        assert child.axes is after.axes
        assert [f.outerDomainValue for f in child] == [1.0e4, 2.0e4]
        for grandchild in child:
            assert grandchild.axes is after.axes
    np.testing.assert_array_equal(after[2][1].ys, before[2][1].ys)


def test_an_xys3d_validates_against_the_schema(tmp_path):
    """``xData_XYs3d_primary`` has no global element, so the root is declared here.

    ``XYs3d`` occurs in ``gnds.xsd`` only inside ``DistributionAEType``
    (``:1797``), which is ``energyAngular``/``angularEnergy``, and neither has a
    global declaration either. Validating the fragment needs one of its type,
    and a two-line schema that ``xs:include``s FUDGE's is the least that
    provides one. ``gnds.xsd`` carries no ``targetNamespace``, so the include
    is a plain textual merge and the type is FUDGE's own, unmodified.
    """
    if not SCHEMA.exists():
        pytest.skip(f"{SCHEMA} is not on this machine")
    wrapper = tmp_path / "xys3d.xsd"
    wrapper.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" '
        'elementFormDefault="qualified">\n'
        f'  <xs:include schemaLocation="{SCHEMA}"/>\n'
        '  <xs:element name="XYs3d" type="xData_XYs3d_primary"/>\n'
        "</xs:schema>\n"
    )

    root = ET.Element("root")
    from kika.nuclear_data.model import ConversionReport
    _function(root, _xys3d(), ConversionReport(), "test")

    fragment = tmp_path / "fragment.xml"
    ET.ElementTree(root.find("XYs3d")).write(fragment)
    assert _schemaErrors(fragment, wrapper) == []


def test_a_non_linlin_xys3d_cannot_be_written_valid_and_that_is_the_schema(tmp_path):
    """The one place ``XYs3d`` is *not* ``XYs2d`` a floor up, and it is a gap.

    ``xData_XYs2d_primary`` (``gnds.xsd:2193``), ``xData_XYs2d`` (``:2219``) and
    ``xData_XYs2d_inRegions`` (``:2210``) each declare an ``interpolation``
    attribute. ``xData_XYs3d_primary`` (``:2260``) declares only ``label`` and
    ``interpolationQualifier`` — it omits ``interpolation`` — so an ``XYs3d``
    whose outermost axis is anything but §6.1.1's lin-lin default writes an
    attribute the schema rejects.

    kika keeps the field rather than dropping it: the outermost axis of a real
    energy-angular distribution has an interpolation law, and losing it on the
    way in would be worse than writing a file this schema refuses. The lin-lin
    case — every one written so far — validates, which
    :func:`test_an_xys3d_validates_against_the_schema` asserts. This test pins
    the other case so it is a recorded gap and not a later surprise.
    """
    if not SCHEMA.exists():
        pytest.skip(f"{SCHEMA} is not on this machine")
    from kika.nuclear_data.model import ConversionReport

    wrapper = tmp_path / "xys3d.xsd"
    wrapper.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" '
        'elementFormDefault="qualified">\n'
        f'  <xs:include schemaLocation="{SCHEMA}"/>\n'
        '  <xs:element name="XYs3d" type="xData_XYs3d_primary"/>\n'
        "</xs:schema>\n"
    )

    form = _xys3d()
    form.interpolation = "log-log"
    root = ET.Element("root")
    _function(root, form, ConversionReport(), "test")
    fragment = tmp_path / "loglog.xml"
    ET.ElementTree(root.find("XYs3d")).write(fragment)

    errors = _schemaErrors(fragment, wrapper)
    assert len(errors) == 1, errors
    assert "attribute 'interpolation': The attribute 'interpolation' is not " \
           "allowed" in errors[0], errors[0]


def test_a_grafted_energy_angular_validates_inside_a_whole_file(
        h2_gnds, tmp_path):
    """§18.4 written back into a file that validates end to end.

    The fragment tests above pin the ``XYs3d`` against a declaration this test
    file makes up. This one is the real shape: an ``<energyAngular>`` inside a
    ``<distribution>`` inside a product of a committed evaluation, validated as
    a whole document against FUDGE's schema with nothing declared by us.

    ``h2_gnds`` is the fixture that round-trips clean today, so any error here
    is the grafted node's. Grafted because **no committed fixture carries an
    ``energyAngular``** — the witness is ``fe56_gnds`` on the share, and the
    gate that uses it is ``test_cross_section_oracle``'s report set.
    """
    from kika.gnds.tests.test_distributions import (ENERGY_ANGULAR_XML,
                                                    graftDistributionForm)

    source = graftDistributionForm(h2_gnds, tmp_path / "grafted.gnds.xml",
                                   ENERGY_ANGULAR_XML)
    written, report = _write(kika.read(source, covariances=False), tmp_path)

    assert list(ET.parse(written).getroot().iter("energyAngular")), (
        "the graft has to survive to the output for this to be a gate")
    assert _schemaErrors(written) == []
    assert not [entry for entry in report.losses if "axes of its own" in entry]


def test_the_angular_energy_witness_is_written_back_valid(micro_be9_gnds,
                                                          tmp_path):
    """§18.5 through the whole pipeline on the file that actually has one.

    Not a graft: ``micro_be9`` is trimmed from ``n-004_Be_009``, the only
    evaluation in the distribution carrying an ``angularEnergy``, and it holds
    both of the two that exist. Writing it back and validating the whole
    document is the strongest gate §18.5 can have, because there is no tape to
    put behind it the way ``fe56_gnds`` stands behind §18.4.

    The tag assertion is not decoration. ``DistributionAEType`` is shared, so a
    writer that emitted ``<energyAngular>`` here would produce a file that
    **validates** — ``_schemaErrors == []`` would pass on it — and states the
    wrong physics. The two asserts together are what close that.
    """
    written, _report = _write(kika.read(micro_be9_gnds, covariances=False),
                              tmp_path)
    root = ET.parse(written).getroot()

    assert len(list(root.iter("angularEnergy"))) == 2
    assert not list(root.iter("energyAngular")), (
        "an <angularEnergy> was written as its mirror; the file would validate"
    )
    assert _schemaErrors(written) == []


def test_every_committed_covariance_fixture_can_be_written(
        gnds_covariance_fixture, tmp_path):
    """The gate whose absence let two writer defects live in a shipped module.

    Only ``h2_gnds``'s sibling was ever written, and it is the *simplest*
    covariance file in the library: one ``covarianceMatrix`` per section, no
    ``mixed``, no ``shortRangeSelfScalingVariance``, no parameter covariances.
    The four fixtures here were committed in phase 5 precisely because between
    them they carry every §25 construct kika reads — and nothing wrote them.

    Both defects this catches were of the same kind: an attribute asked of a
    class that does not have it. ``mixed`` was asked for a ``label`` the model
    lacked, and ``shortRangeSelfScalingVariance`` for an ``isRelative`` that
    lives on the matrix it wraps rather than on itself. Reading was fine and
    tested; writing raised ``AttributeError`` on any real file.
    """
    from kika.gnds.covariances import readCovarianceSuite
    from kika.gnds.encode import writeCovarianceSuite

    suite, _ = readCovarianceSuite(Document.parse(gnds_covariance_fixture))
    tree, _report = writeCovarianceSuite(suite, "2.0")
    path = tmp_path / "out.gnds-covar.xml"
    path.write_bytes(serialise(tree))

    assert _schemaErrors(path, COVARIANCE_SCHEMA) == []


#: The two committed covariance files whose whole content is §25.3. Si-32 has
#: one ``parameterCovariance`` and **no ``covarianceSections`` at all**; Tm-171
#: has one plus the ten ``averageParameterCovariance`` of its URR. They are the
#: only witnesses either node has, and until §25.3's writer landed the gate
#: above passed on both by writing files with the nodes silently missing.
PARAMETER_COVARIANCE_FIXTURES = ("n-014_Si_032", "n-069_Tm_171")


def _covarianceRoundTrip(name, tmp_path):
    """One committed covariance fixture, out and back. Returns (before, after, root)."""
    from kika.gnds.covariances import readCovarianceSuite
    from kika.gnds.encode import writeCovarianceSuite

    source = (Path(__file__).parent / "data" / "Covariances"
              / f"{name}.endf.gnds-covar.xml")
    before, _ = readCovarianceSuite(Document.parse(source))
    tree, report = writeCovarianceSuite(before, "2.0")
    path = tmp_path / "out.gnds-covar.xml"
    path.write_bytes(serialise(tree))
    after, _ = readCovarianceSuite(Document.parse(path))
    return before, after, ET.parse(path).getroot(), path, report


@pytest.mark.parametrize("name", PARAMETER_COVARIANCE_FIXTURES)
def test_the_parameter_covariances_reach_the_written_file(name, tmp_path):
    """The gate the schema test could not be: **the node is there**.

    ``test_every_committed_covariance_fixture_can_be_written`` validated Si-32
    and Tm-171 while the writer dropped every §25.3 node on the floor, because a
    file with nothing in it is a valid file. So the first assertion is presence
    and the count, not validity.
    """
    before, _after, root, _path, _report = _covarianceRoundTrip(name, tmp_path)

    container = root.find("parameterCovariances")
    assert container is not None, (
        f"{name}'s whole content is §25.3 and the written file has no "
        f"<parameterCovariances> at all")
    assert len(container) == len(before.parameterCovariances)

    # covariances.xsd:27-34 is an xs:sequence, not a bag: every
    # `parameterCovariance` before every `averageParameterCovariance`.
    tags = [child.tag for child in container]
    assert tags == sorted(tags, key=lambda tag: tag != "parameterCovariance"), tags


@pytest.mark.parametrize("name", PARAMETER_COVARIANCE_FIXTURES)
def test_the_parameter_covariance_matrices_survive_the_round_trip(name, tmp_path):
    """Matrices element for element, and every link attribute.

    ``matrixStartIndex`` is in here on purpose. It is already zero-based —
    Si-32's 1 + 18 = 19 is the arithmetic that says so, and
    ``covariances.py:_readParameterCovarianceMatrix`` carries the note — so a
    writer that "converted back to one-based" would shift every link by a row
    and leave the counts still summing to the order, which is all the model
    checks.
    """
    before, after, _root, _path, _report = _covarianceRoundTrip(name, tmp_path)

    assert len(after.parameterCovariances) == len(before.parameterCovariances)
    for first, second in zip(before.parameterCovariances,
                             after.parameterCovariances):
        assert type(first) is type(second)
        assert first.label == second.label
        np.testing.assert_array_equal(first.form.matrix, second.form.matrix)
        assert first.form.isRelative == second.form.isRelative
        assert (first.rowData is None) == (second.rowData is None)
        if first.rowData is not None:
            assert first.rowData.href == second.rowData.href

        links = getattr(first.form, "parameters", None)
        if links is None:
            continue
        assert [(link.label, link.href, link.nParameters, link.matrixStartIndex)
                for link in links] == [
            (link.label, link.href, link.nParameters, link.matrixStartIndex)
            for link in second.form.parameters]


def test_neither_parameter_covariance_fixture_loses_its_links(tmp_path):
    """``_readParameterCovarianceMatrix`` drops the links when they do not
    account for every row, and that would make the round trip above assert an
    equality of two empty lists. Both fixtures cover all their rows — Si-32's
    1 + 18 = 19, Tm-171's 1 + 1200 = 1201 — so the comparison has teeth."""
    from kika.gnds.covariances import readCovarianceSuite

    for name in PARAMETER_COVARIANCE_FIXTURES:
        source = (Path(__file__).parent / "data" / "Covariances"
                  / f"{name}.endf.gnds-covar.xml")
        suite, report = readCovarianceSuite(Document.parse(source))
        assert not [entry for entry in report.losses
                    if "cannot be named" in entry], report.losses
        for covariance in suite.parameterCovariances:
            if hasattr(covariance.form, "parameters"):
                assert covariance.form.parameters, covariance.label


def test_si_32_is_still_written_with_no_covariance_sections(tmp_path):
    """The fixture exists for the case "a suite that is nothing but §25.3", and
    a writer that invented an empty ``<covarianceSections/>`` would say the
    evaluation has cross-section covariances it does not have."""
    _before, after, root, _path, _report = _covarianceRoundTrip("n-014_Si_032",
                                                                tmp_path)
    assert root.find("covarianceSections") is None
    assert after.covarianceSections == []


@pytest.mark.parametrize("name", PARAMETER_COVARIANCE_FIXTURES)
def test_the_written_parameter_covariances_validate(name, tmp_path):
    """Same schema gate as the fixture sweep, but on a file that has the nodes.

    Skipped on any machine without FUDGE, **which includes every GitHub
    runner** — the green tick upstream says nothing about this one.
    """
    _before, _after, _root, path, _report = _covarianceRoundTrip(name, tmp_path)
    assert _schemaErrors(path, COVARIANCE_SCHEMA) == []


def test_a_cross_term_parameter_covariance_is_reported_not_emitted(tmp_path):
    """§25.3.1 has no ``columnData``, and the model has one.

    ``covariances.xsd:160-168`` gives ``parameterCovariance`` exactly
    ``rowData`` and ``parameterCovarianceMatrix``. The model carries a
    ``columnData`` and an ``isCrossTerm`` built off it, and the reader fills it
    from a file that has one — so this is a state the model reaches and the
    format cannot express. Writing the attribute anyway would invalidate the
    file; dropping it silently would lose the only statement that the two axes
    are different parameters.
    """
    from kika.gnds.encode import writeCovarianceSuite
    from kika.nuclear_data.model import (CovarianceSuite, DataLink,
                                         ParameterCovariance,
                                         ParameterCovarianceMatrix,
                                         ParameterLink)

    suite = CovarianceSuite(
        evaluation="test", projectile="n", target="Si32",
        parameterCovariances=[ParameterCovariance(
            label="cross",
            rowData=DataLink(href="#rows"),
            columnData=DataLink(href="#columns"),
            form=ParameterCovarianceMatrix(
                matrix=np.eye(3), label="eval",
                parameters=[ParameterLink(label="l", href="#p", nParameters=3)],
            ),
        )],
    )
    tree, report = writeCovarianceSuite(suite, "2.0")
    written = tree.getroot().find("parameterCovariances/parameterCovariance")

    assert written.find("columnData") is None
    assert "crossTerm" not in written.attrib
    assert any("#columns" in entry for entry in report.losses), report.losses


def test_a_parameter_covariance_whose_rows_nothing_names_is_left_out(tmp_path):
    """The other half of the dropped-link case, and it is a real one.

    When ``_readParameterCovarianceMatrix`` drops links that do not cover every
    row, what reaches the writer is a matrix with no ``parameters``. §25.3.2
    makes ``<parameters>`` mandatory with at least one ``parameterLink``, so
    there is no valid file to write: the choice is an invalid one or an absent
    covariance, and the absent one is the only version that does not claim
    something. Neither committed fixture reaches this — see
    ``test_neither_parameter_covariance_fixture_loses_its_links``.
    """
    from kika.gnds.encode import writeCovarianceSuite
    from kika.nuclear_data.model import (CovarianceSuite, DataLink,
                                         ParameterCovariance,
                                         ParameterCovarianceMatrix)

    suite = CovarianceSuite(
        evaluation="test", projectile="n", target="Si32",
        parameterCovariances=[ParameterCovariance(
            label="unnamed rows",
            rowData=DataLink(href="#rows"),
            form=ParameterCovarianceMatrix(matrix=np.eye(3), label="eval"),
        )],
    )
    tree, report = writeCovarianceSuite(suite, "2.0")

    # Not an empty container either: `<parameterCovariances/>` is valid and
    # says the evaluation has none, which is not what happened.
    assert tree.getroot().find("parameterCovariances") is None
    assert any("unnamed rows" in entry for entry in report.losses), report.losses


def test_an_endf_built_parameter_matrix_gets_the_label_the_schema_requires():
    """§25.3.2 makes ``label`` required and MF32 has no such concept.

    ``kika/endf/model_adapter/parameter_covariances.py`` builds every
    ``ParameterCovarianceMatrix`` with ``label=None``, so the one attribute the
    schema will not do without is exactly the one an ENDF-sourced suite never
    has. It is filled with the style label the suite writer synthesises, and
    reported — the source said nothing about which style the matrix belongs to.
    """
    from kika.gnds.encode import writeCovarianceSuite
    from kika.nuclear_data.model import (CovarianceSuite, DataLink,
                                         ParameterCovariance,
                                         ParameterCovarianceMatrix,
                                         ParameterLink)

    suite = CovarianceSuite(
        evaluation="test", projectile="n", target="Si32",
        parameterCovariances=[ParameterCovariance(
            label="MF32", rowData=DataLink(href="#rows"),
            form=ParameterCovarianceMatrix(
                matrix=np.eye(2), label=None,
                parameters=[ParameterLink(label="l", href="#p", nParameters=2)],
            ),
        )],
    )
    tree, report = writeCovarianceSuite(suite, "2.0")
    matrix = tree.getroot().find(
        "parameterCovariances/parameterCovariance/parameterCovarianceMatrix")

    assert matrix.attrib["label"] == "eval"
    assert matrix.attrib["type"] == "absolute"
    assert any("no label" in entry for entry in report.approximations), (
        report.approximations)


# ---------------------------------------------------------------------------
# the version policy
# ---------------------------------------------------------------------------

def test_a_file_read_as_2_0_is_written_as_2_0(h2_gnds, tmp_path):
    """A round trip that upgraded the declared version would make every diff
    against the source start with a changed root attribute."""
    suite = kika.read(h2_gnds, covariances=False)
    assert suite.format == "2.0"
    path, _ = _write(suite, tmp_path)
    assert ET.parse(path).getroot().attrib["format"] == "2.0"


def test_the_version_can_be_forced(h2_gnds, tmp_path):
    suite = kika.read(h2_gnds, covariances=False)
    path = tmp_path / "forced.xml"
    kika.write(suite, path, gnds="2.1")
    assert ET.parse(path).getroot().attrib["format"] == "2.1"


def test_a_version_kika_cannot_read_back_is_refused(h2_gnds, tmp_path):
    suite = kika.read(h2_gnds, covariances=False)
    with pytest.raises(UnsupportedGndsVersion, match="1.9"):
        kika.write(suite, tmp_path / "old.xml", gnds="1.9")


def test_a_suite_with_no_gnds_origin_is_written_as_the_default(micro_cov_tape,
                                                              h2_gnds):
    """An ENDF-sourced suite has no version to mirror, so it gets the default.

    **"Has no provenance" is not the test, and was the first attempt.** The ENDF
    adapter fills ``suite.provenance`` only when MF1/451 parses, so a tape with
    a damaged header — this micro fixture is one — decoded to a suite with none,
    and the writer read that as "came from GNDS" and mirrored a version nothing
    had declared. The signal is the positive one: a ``GndsProvenance``, which
    only the GNDS reader sets.
    """
    from kika.gnds.version import DEFAULT_WRITE_FORMAT
    from kika.nuclear_data.model import GndsProvenance

    fromEndf = kika.read(micro_cov_tape)
    assert not isinstance(fromEndf.provenance, GndsProvenance)
    assert chooseFormat(fromEndf) == DEFAULT_WRITE_FORMAT == "2.0"

    fromGnds = kika.read(h2_gnds, covariances=False)
    assert isinstance(fromGnds.provenance, GndsProvenance)
    assert fromGnds.provenance.formatVersion == "2.0"


# ---------------------------------------------------------------------------
# the two files, and the door
# ---------------------------------------------------------------------------

def test_the_covariances_go_in_their_own_document_with_a_matching_digest(
        h2_gnds, tmp_path):
    """§25.1.1. And the ``externalFile`` is rewritten to name what was actually
    put on disk — carrying the digest the suite was *read* with would point at a
    file kika has just changed."""
    suite = kika.read(h2_gnds)
    path, report = _write(suite, tmp_path)

    sibling = tmp_path / "Covariances" / "out.gnds-covar.xml"
    assert ET.parse(sibling).getroot().tag == "covarianceSuite"

    entry = ET.parse(path).getroot().find("externalFiles/externalFile")
    assert entry.attrib["path"] == "Covariances/out.gnds-covar.xml"
    assert entry.attrib["algorithm"] == "sha1"
    assert entry.attrib["checksum"] == sha1(sibling.read_bytes())
    assert any("separate document" in w for w in report.warnings)


def test_the_pair_reads_back_through_the_door_with_its_links_intact(h2_gnds,
                                                                   tmp_path):
    """The written pair is a pair: the door follows the link and the covariance
    reader resolves every ``rowData`` href through it."""
    path, _ = _write(kika.read(h2_gnds), tmp_path)
    reread = kika.read(path)
    assert reread.covarianceSuite is not None
    assert not [loss for loss in reread.report.losses
                if "The covariance itself is read" in loss]


def test_writing_endf_says_why_it_cannot(h2_gnds, tmp_path):
    """Not "not implemented": kika's ENDF writer patches a tape it already read,
    and a whole-file model → ENDF writer is a project, not a missing function."""
    suite = kika.read(h2_gnds, covariances=False)
    with pytest.raises(NotImplementedError, match="patches sections into a tape"):
        kika.write(suite, tmp_path / "out.endf", format="endf")


def test_an_unknown_format_is_refused_by_name(h2_gnds, tmp_path):
    suite = kika.read(h2_gnds, covariances=False)
    with pytest.raises(ValueError, match="format must be one of"):
        kika.write(suite, tmp_path / "out.ace", format="ace")


# ---------------------------------------------------------------------------
# the report is the only place the gaps are visible
# ---------------------------------------------------------------------------

def test_the_minimal_pops_is_declared_in_every_written_file(evaluation,
                                                            tmp_path):
    """A reader of the written file cannot tell kika's minimal §12 from a
    database that genuinely says this much. The report can."""
    _, report = _write(kika.read(evaluation, covariances=False), tmp_path)
    assert any("minimal §12 model" in loss for loss in report.losses)


def test_the_writer_returns_a_report_and_not_the_file(h2_gnds, tmp_path):
    from kika.nuclear_data.model import ConversionReport

    result = kika.write(kika.read(h2_gnds, covariances=False),
                        tmp_path / "out.xml")
    assert isinstance(result, ConversionReport)


def test_the_suite_children_come_out_in_the_order_the_schema_declares(h2_gnds,
                                                                     tmp_path):
    """``ReactionSuiteType`` is an ``xs:sequence``. A file with ``sums`` after
    ``productions`` is one every reader still reads and no validator accepts —
    which is how it would survive unnoticed."""
    from kika.gnds.encode import SUITE_ORDER

    path, _ = _write(kika.read(h2_gnds, covariances=False), tmp_path)
    written = [child.tag for child in ET.parse(path).getroot()]
    positions = [SUITE_ORDER.index(tag) for tag in written]
    assert positions == sorted(positions), written


def test_the_endf_mfmt_is_written_with_the_comma_the_spec_defines(h2_gnds,
                                                                  tmp_path):
    """§25.2.3 p. 363 makes it a comma-separated pair, and kika's ENDF adapter
    writes a slash.

    Not fixed at the adapter: nine sites across ``kika/cov`` and ``kika/sampling``
    split ``ENDF_MFMT`` on that slash, two of them in the deployed thesis
    pipeline. It is fixed at the writer, which is the one place the string is
    handed to somebody else's tool.
    """
    from kika.gnds.encode import _mfmt
    from kika.nuclear_data.model import DataLink

    suite = kika.read(h2_gnds)
    path, _ = _write(suite, tmp_path)
    written = ET.parse(tmp_path / "Covariances" / "out.gnds-covar.xml").getroot()
    values = {node.attrib["ENDF_MFMT"] for node in written.iter("rowData")}
    assert values == {"33,1", "33,2", "33,16", "33,102"}

    # The ENDF adapter's spelling goes out as the spec's...
    assert _mfmt(DataLink(href="/x", ENDF_MFMT="33/2")) == "33,2"
    # ...and something that parses as neither is passed through, not mangled.
    assert _mfmt(DataLink(href="/x", ENDF_MFMT="whatever")) == "whatever"


def test_writing_does_not_edit_the_suite_it_was_given(h2_gnds, tmp_path):
    """Writing a copy out for inspection must not change the evaluation.

    Both ``externalFile`` links are rewritten during a write — the forward one
    to name the file actually produced, the back one so the pair's hrefs resolve
    — and both are rewritten **on the caller's objects**. They are put back.
    """
    suite = kika.read(h2_gnds)
    before = [(e.label, e.path, e.checksum) for e in suite.externalFiles]
    backBefore = [(e.label, e.path) for e in suite.covarianceSuite.externalFiles]

    kika.write(suite, tmp_path / "a.gnds.xml")
    kika.write(suite, tmp_path / "b.gnds.xml")

    assert [(e.label, e.path, e.checksum) for e in suite.externalFiles] == before
    assert [(e.label, e.path)
            for e in suite.covarianceSuite.externalFiles] == backBefore
    # And each written pair is internally consistent, not cross-linked.
    for stem in ("a", "b"):
        entry = ET.parse(tmp_path / f"{stem}.gnds.xml").getroot().find(
            "externalFiles/externalFile")
        assert entry.attrib["path"] == f"Covariances/{stem}.gnds-covar.xml"
        assert entry.attrib["checksum"] == sha1(
            (tmp_path / "Covariances" / f"{stem}.gnds-covar.xml").read_bytes())
