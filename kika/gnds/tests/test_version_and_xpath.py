"""Phase 5 P1: the format gate, and href resolution against real files.

The interesting test in this module is
:func:`test_every_href_in_the_paired_fixture_resolves`. Resolution is the part
of a GNDS reader that fails *quietly* — an href that reaches nothing yields a
missing cross section, not an exception — so it is gated on every link in the
committed files rather than on hand-written examples.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from kika.gnds.version import (ACCEPTED, DEFAULT_WRITE_FORMAT, MODEL_FORMAT,
                               REFUSED, UnsupportedGndsVersion, checkFormat)
from kika.gnds.xpath import (Document, Resolver, parseStep, readExternalFiles,
                             splitSteps, verifyChecksum)


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("declared", ACCEPTED)
def test_both_accepted_versions_pass_through_unchanged(declared):
    assert checkFormat(declared) == declared


@pytest.mark.parametrize("declared", sorted(REFUSED))
def test_the_pre_2_0_versions_are_refused_by_name(declared):
    """1.9 and 1.10 are where the 149 change requests fell. Refused, not attempted."""
    with pytest.raises(UnsupportedGndsVersion) as raised:
        checkFormat(declared, source="Fe56.gnds.xml")
    message = str(raised.value)
    assert "Fe56.gnds.xml" in message
    assert declared in message
    assert "2.0, 2.1" in message


def test_an_unknown_version_is_refused_rather_than_attempted():
    """A future 2.2 must not be read with 2.1 rules and no complaint."""
    with pytest.raises(UnsupportedGndsVersion) as raised:
        checkFormat("2.2")
    assert "would not raise" in str(raised.value)


def test_a_missing_format_attribute_is_refused():
    with pytest.raises(UnsupportedGndsVersion) as raised:
        checkFormat(None)
    assert "§14.1.1" in str(raised.value)


def test_the_three_version_constants_are_deliberately_different():
    """Model target, accepted set and write default are three separate answers.

    The model is built to 2.1, every published library is written in 2.0, and
    the writer emits 2.0 when it has no origin to preserve. Collapsing any two
    of these into one constant is the mistake this asserts against.
    """
    assert MODEL_FORMAT == "2.1"
    assert DEFAULT_WRITE_FORMAT == "2.0"
    assert MODEL_FORMAT in ACCEPTED and DEFAULT_WRITE_FORMAT in ACCEPTED


def test_the_committed_fixtures_pass_the_gate(gnds_data_dir):
    for path in sorted(gnds_data_dir.rglob("*.xml")):
        root = ET.parse(path).getroot()
        assert checkFormat(root.attrib.get("format"), path.name) == "2.0"


# ---------------------------------------------------------------------------
# path syntax
# ---------------------------------------------------------------------------

def test_a_label_containing_brackets_survives_the_split():
    """``Fe57 + photon [inclusive]`` is a real label, and it ends in ``]``.

    Any bracket matching that is not quote-aware truncates it at the wrong
    ``]`` and the link silently resolves to nothing. This is the first href in
    Fe-56, so it is the first thing tested.
    """
    href = "/reactionSuite/reactions/reaction[@label='Fe57 + photon [inclusive]']"
    steps = splitSteps(href)
    assert steps == ["", "reactionSuite", "reactions",
                     "reaction[@label='Fe57 + photon [inclusive]']"]
    tag, predicates = parseStep(steps[-1])
    assert tag == "reaction"
    assert predicates == {"label": "Fe57 + photon [inclusive]"}


def test_a_label_containing_a_slash_would_survive_too():
    """The separator itself may appear inside a quoted predicate value."""
    tag, predicates = parseStep("reaction[@label='a/b']")
    assert (tag, predicates) == ("reaction", {"label": "a/b"})
    assert splitSteps("x/reaction[@label='a/b']/y") == [
        "x", "reaction[@label='a/b']", "y"
    ]


def test_two_predicates_on_one_step():
    tag, predicates = parseStep("channel[@L='0'][@channelSpin='1/2']")
    assert tag == "channel"
    assert predicates == {"L": "0", "channelSpin": "1/2"}


def test_a_positional_predicate_is_refused_not_ignored():
    """Following the wrong element is worse than not following it."""
    with pytest.raises(ValueError, match="does not implement"):
        parseStep("reaction[1]")


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------

def _pairedResolver(h2_gnds, h2_gnds_cov):
    """The H-2 pair, wired both ways, as a reader with both files would have it."""
    reactions = Document.parse(h2_gnds)
    covariances = Document.parse(h2_gnds_cov)
    return (
        Resolver(reactions, {"covariances": covariances}),
        Resolver(covariances, {"reactions": reactions}),
    )


def _hrefs(document):
    for node in document.root.iter():
        href = node.attrib.get("href")
        if href:
            yield node, href


def test_every_href_in_the_paired_fixture_resolves(h2_gnds, h2_gnds_cov):
    """Both directions, every link, on an unmodified published evaluation.

    H-2 is committed as a pair precisely so this can be asserted: with both
    files in hand nothing in either should point at nothing.
    """
    forward, backward = _pairedResolver(h2_gnds, h2_gnds_cov)
    checked = 0
    for resolver in (forward, backward):
        for node, href in _hrefs(resolver.primary):
            resolution = resolver.resolve(href, context=node)
            assert resolution, f"{href!r} did not resolve: {resolution.problem}"
            checked += 1
    assert checked > 10, f"only {checked} hrefs exercised; expected the pair's full set"


def test_the_covariance_grid_link_walks_up_and_across(h2_gnds, h2_gnds_cov):
    """``../../grid[@index='2']/values`` — the parent axis, which ``find`` lacks.

    A covariance's column grid is stored as a link to its row grid rather than
    as 124 repeated numbers. Getting this wrong does not raise; it produces a
    covariance with no column grid.
    """
    _, backward = _pairedResolver(h2_gnds, h2_gnds_cov)
    links = [
        (node, href) for node, href in _hrefs(backward.primary)
        if href.startswith("..")
    ]
    assert links, "the H-2 covariance file has no relative grid link to test"
    for node, href in links:
        resolution = backward.resolve(href, context=node)
        assert resolution, resolution.problem
        assert resolution.element.tag == "values"
        assert len(resolution.element.text.split()) > 1


def test_a_relative_href_without_context_is_reported_not_raised(h2_gnds, h2_gnds_cov):
    _, backward = _pairedResolver(h2_gnds, h2_gnds_cov)
    resolution = backward.resolve("../../grid[@index='2']/values")
    assert not resolution
    assert "no context element" in resolution.problem


def test_a_covariance_file_read_alone_reports_the_links_it_cannot_follow(
    gnds_covariance_fixture,
):
    """The normal case for four of the committed fixtures, and it must not raise.

    A ``covarianceSuite`` handed over without its ``reactionSuite`` still has to
    be readable; what it cannot do is say which cross section each section is
    about. So every ``$reactions#`` href reports, every internal one resolves,
    and nothing raises.
    """
    document = Document.parse(gnds_covariance_fixture)
    resolver = Resolver(document)          # deliberately no external documents

    external = internal = 0
    for node, href in _hrefs(document):
        resolution = resolver.resolve(href, context=node)
        if href.startswith("$"):
            external += 1
            assert not resolution
            assert "was not supplied" in resolution.problem
        else:
            internal += 1
            assert resolution, f"{href!r} did not resolve: {resolution.problem}"
    assert external, f"{gnds_covariance_fixture.name} has no cross-file href"


def test_every_internal_href_in_the_fe56_trim_resolves(micro_fe56_gnds):
    """The trim promises no dangling links; this is that promise, dereferenced."""
    document = Document.parse(micro_fe56_gnds)
    resolver = Resolver(document)
    for node, href in _hrefs(document):
        resolution = resolver.resolve(href, context=node)
        assert resolution, f"{href!r} did not resolve: {resolution.problem}"


def test_the_bracketed_label_resolves_in_the_real_file(micro_fe56_gnds):
    """Not just parsed — followed, to the reaction it names, in Fe-56 itself."""
    document = Document.parse(micro_fe56_gnds)
    resolver = Resolver(document)
    resolution = resolver.resolve(
        "/reactionSuite/reactions/reaction[@label='Fe57 + photon [inclusive]']"
    )
    assert resolution, resolution.problem
    assert resolution.element.attrib["ENDF_MT"] == "102"


def test_walking_above_the_root_is_reported(micro_fe56_gnds):
    document = Document.parse(micro_fe56_gnds)
    resolver = Resolver(document)
    resolution = resolver.resolve("../..", context=document.root)
    assert not resolution
    assert "above the root" in resolution.problem


def test_an_absolute_href_into_the_wrong_root_is_reported(h2_gnds, h2_gnds_cov):
    forward, _ = _pairedResolver(h2_gnds, h2_gnds_cov)
    resolution = forward.resolve("/covarianceSuite/covarianceSections")
    assert not resolution
    assert "is a <reactionSuite>" in resolution.problem


# ---------------------------------------------------------------------------
# externalFiles and checksums
# ---------------------------------------------------------------------------

def test_the_declared_checksum_verifies(h2_gnds, h2_gnds_cov):
    entries = readExternalFiles(ET.parse(h2_gnds).getroot())
    assert [e.label for e in entries] == ["covariances"]
    entry = entries[0]
    assert entry.algorithm == "sha1"
    assert entry.resolvedAgainst(h2_gnds) == h2_gnds_cov
    assert verifyChecksum(entry, h2_gnds_cov) is None


def test_an_edited_sibling_fails_the_checksum(h2_gnds, h2_gnds_cov, tmp_path):
    """The half that makes the half above mean something."""
    entry = readExternalFiles(ET.parse(h2_gnds).getroot())[0]
    tampered = tmp_path / h2_gnds_cov.name
    tampered.write_bytes(h2_gnds_cov.read_bytes().replace(b"1e-5", b"2e-5", 1))

    problem = verifyChecksum(entry, tampered)
    assert problem is not None
    assert "has been edited" in problem


def test_an_entry_with_no_checksum_is_not_a_failure(h2_gnds_cov):
    """The distribution's covariance files point back with no digest at all."""
    entries = readExternalFiles(ET.parse(h2_gnds_cov).getroot())
    assert [e.label for e in entries] == ["reactions"]
    assert entries[0].checksum is None
    assert verifyChecksum(entries[0], h2_gnds_cov) is None


def test_an_unknown_digest_algorithm_is_reported(h2_gnds, h2_gnds_cov):
    entry = readExternalFiles(ET.parse(h2_gnds).getroot())[0]
    entry.algorithm = "sha256"
    problem = verifyChecksum(entry, h2_gnds_cov)
    assert problem is not None and "§3.4.3" in problem
