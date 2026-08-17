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
from kika.gnds.encode import (chooseFormat, serialise, sha1,
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

#: The three committed evaluations, by conftest fixture name. Between them they
#: carry every node the writer emits.
FIXTURES = ("h2_gnds", "micro_fe56_gnds", "micro_ta182_gnds")


@pytest.fixture(params=FIXTURES)
def evaluation(request):
    """Each committed reactionSuite in turn, as a path."""
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


def test_the_only_schema_errors_are_the_distributions_phase_7b_will_fill(
        evaluation, tmp_path):
    """Every node the writer emits validates, except the ones it leaves empty.

    A product whose §18 law kika does not read gets an empty
    ``<distribution/>``. That is deliberate and it is the *only* thing wrong
    with the file — this test says so by counting, so the day §18 lands the
    number goes to zero and this test is what notices.
    """
    suite = kika.read(evaluation, covariances=False)
    path, report = _write(suite, tmp_path)

    errors = _schemaErrors(path)
    assert all("Element 'distribution': Missing child element" in error
               for error in errors), errors

    withoutDistribution = sum(
        1 for reaction in _everyReaction(suite)
        for product in _everyProduct(reaction.outputChannel)
        if product.distribution is None or len(product.distribution) == 0
    )
    assert len(errors) == withoutDistribution
    if errors:
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


def test_an_unreadable_distribution_is_left_empty_and_never_called_unspecified(
        h2_gnds, tmp_path):
    """The one substitution that would make the file valid, and is a forgery.

    ``<unspecified/>`` is the **evaluator** stating that no distribution is
    given. Writing it where kika merely cannot read the law yet would put that
    statement in their mouth, and nothing downstream could tell the difference.
    """
    suite = kika.read(h2_gnds, covariances=False)
    path, report = _write(suite, tmp_path)

    root = ET.parse(path).getroot()
    empty = [d for d in root.iter("distribution") if len(d) == 0]
    assert len(empty) == 3
    # H-2 *does* have a genuine `unspecified`, on its production channel, and
    # it survives — so the absence above is not the writer dropping the node.
    assert len(list(root.iter("unspecified"))) == 2
    assert any("does not validate" in entry for entry in report.unsupported)


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


#: The schema errors an **ENDF-decoded** suite still has, by kind. Twelve, and
#: none of them is the deliberate empty `<distribution/>` that the GNDS
#: fixtures produce -- these are gaps in the ENDF -> model -> GNDS path itself,
#: measured 2026-08-17 and written down in `docs/library-gaps.md` D20. Pinned
#: rather than fixed here: each one is its own decision (what date format does
#: an ENDF `AUG-2018` become? what `genre` does an ENDF output channel have?),
#: and a gate that exists is worth more than a gate that waits for all twelve.
ENDF_SCHEMA_GAPS = (
    "Element 'evaluated', attribute 'date'",
    "Element 'evaluated': Missing child element",
    "Element 'scatteringRadius': This element is not expected",
    "Element 'function1ds': Missing child element",
    "Element 'outputChannel': The attribute 'genre' is required but missing",
    "Element 'products': Missing child element",
    "Element 'multiplicity': Missing child element",
    "Element 'function2ds': This element is not expected",
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

    Twelve errors survive today and every one is a known ENDF->GNDS gap
    (:data:`ENDF_SCHEMA_GAPS`). Pinned by count and kind, the same shape as the
    GNDS half: fixing one lowers the number and this test is what notices.
    """
    _path, errors = _endfSchemaErrors(kika.read(micro_tape, covariances=False),
                                      tmp_path)
    unknown = [e for e in errors
               if not any(kind in e for kind in ENDF_SCHEMA_GAPS)]
    assert unknown == [], unknown
    assert len(errors) == 12


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
    # It removes one, and that is the point rather than an aside: the LTT=3
    # section it replaced is written as an `<XYs2d>` whose `function2ds` comes
    # before its `axes`, which the schema rejects (ENDF_SCHEMA_GAPS again).
    assert len(errors) == len(baseline) - 1

    # And the half that says it is not merely valid: kika reads it back.
    after = kika.read(path, covariances=False)
    form = after.reactions[2].outputChannel.products[0].distribution["eval"]
    assert isinstance(form.angular, Isotropic2d), form


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
