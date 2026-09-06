"""The committed GNDS fixtures are what they claim to be.

This is phase 5's P0: no reader exists yet, and these tests are what the reader
will be written against. Every claim a fixture's docstring makes — "this one
carries ``slices``", "this pair's checksum is real", "the trim leaves no
dangling href" — is asserted here, because a fixture whose description has
drifted from its content is worse than no fixture: it makes a reader look
correct on a construct it never met.

**Provenance.** Five files are committed **byte for byte** as the NNDC
distributes them in ENDF/B-VIII.1-GNDS; only ``micro_fe56.gnds.xml`` is cut
down, by ``data/build_micro_fe56_gnds.py``, which validates its own output
against FUDGE's GNDS 2.0 schema before writing it.
"""
from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

#: The two root nodes GNDS defines. §14.1.1 and §25.1.1.
ROOTS = ("reactionSuite", "covarianceSuite")

#: What the published library is written in. Not what kika's model targets —
#: the model is 2.1 — and the gap is the whole reason `kika/gnds/version.py`
#: accepts both. If this ever fails, a library has moved and that is news.
PUBLISHED_FORMAT = "2.0"


def _roots(directory: Path):
    # rglob, not glob: the covariances sit under ``Covariances/`` because that
    # is where the distribution puts them and because their externalFile paths
    # are written relative to it.
    for path in sorted(directory.rglob("*.xml")):
        yield path, ET.parse(path).getroot()


def test_every_committed_fixture_is_a_gnds_root(gnds_data_dir):
    """Each fixture parses, and is one of the two GNDS roots at format 2.0."""
    found = list(_roots(gnds_data_dir))
    assert found, f"no GNDS fixtures under {gnds_data_dir}"
    for path, root in found:
        assert root.tag in ROOTS, f"{path.name} has root <{root.tag}>"
        assert root.attrib.get("format") == PUBLISHED_FORMAT, (
            f"{path.name} declares format={root.attrib.get('format')!r}; the "
            f"whole fixture set is meant to be the published 2.0 encoding"
        )


def test_the_h2_pair_checksum_is_real(h2_gnds, h2_gnds_cov):
    """The SHA-1 in ``externalFiles`` matches the sibling actually committed.

    This is the property that makes H-2 the pair worth having: the reader's
    checksum verification can be tested against a file whose checksum was
    written by the evaluation's publisher, not by this repository.
    """
    root = ET.parse(h2_gnds).getroot()
    entry = root.find("externalFiles/externalFile")
    assert entry is not None, "H-2 has no externalFiles entry to check"
    assert entry.attrib.get("algorithm") == "sha1"
    assert entry.attrib["path"].endswith(h2_gnds_cov.name)

    digest = hashlib.sha1(h2_gnds_cov.read_bytes()).hexdigest()
    assert digest == entry.attrib["checksum"], (
        "the committed covariance sibling does not match the checksum its "
        "reactionSuite declares — the pair was not copied together"
    )


def test_a_changed_byte_breaks_the_checksum(h2_gnds, h2_gnds_cov, tmp_path):
    """The negative half. Without it the test above passes on a stub verifier.

    §14.1.1's checksum only earns its keep if a corrupted sibling fails it, so
    the corruption is exercised here rather than assumed.
    """
    root = ET.parse(h2_gnds).getroot()
    declared = root.find("externalFiles/externalFile").attrib["checksum"]

    tampered = tmp_path / h2_gnds_cov.name
    original = h2_gnds_cov.read_bytes()
    # Flip one digit inside a number, not a structural character: the point is
    # that a change no XML parser would notice still fails the checksum.
    tampered.write_bytes(original.replace(b"1e-5", b"2e-5", 1))
    assert tampered.read_bytes() != original, "the tamper found nothing to change"
    assert hashlib.sha1(tampered.read_bytes()).hexdigest() != declared


def test_each_covariance_fixture_carries_what_it_was_committed_for(
    gnds_covariance_fixture,
):
    """The construct named in ``GNDS_COVARIANCE_FIXTURES`` is really in the file.

    Each of these was picked as the *smallest* file in the 270-file covariance
    distribution carrying its construct. That selection is only worth anything
    if it stays true of the bytes committed, so it is checked rather than
    trusted.
    """
    path = Path(gnds_covariance_fixture)
    stem = path.name.split(".")[0]
    root = ET.parse(path).getroot()
    tags = {node.tag for node in root.iter()}
    compressions = {
        node.attrib["compression"]
        for node in root.iter("array") if "compression" in node.attrib
    }
    crossTerms = [n for n in root.iter("covarianceSection") if "crossTerm" in n.attrib]

    if stem == "n-014_Si_032":
        assert "parameterCovariances" in tags
        assert root.find("covarianceSections") is None, (
            "Si-32 is committed for the case of a covarianceSuite with no "
            "covarianceSections at all; it now has some"
        )
    elif stem == "n-069_Tm_171":
        assert "averageParameterCovariance" in tags
        assert "flattened" in compressions
    elif stem == "n-009_F_019":
        assert {"sum", "summand", "columnData", "mixed",
                "shortRangeSelfScalingVariance"} <= tags
        assert crossTerms, "F-19 is committed for its crossTerm sections"
    elif stem == "n-057_La_139":
        assert {"slices", "slice"} <= tags
        mfmt = {n.attrib.get("ENDF_MFMT") for n in root.iter("rowData")}
        assert any((m or "").startswith("34,") for m in mfmt), (
            "La-139 is the fixture for the MF34 Legendre-order slice; its "
            "rowData no longer names an MF34 quantity"
        )
    else:  # pragma: no cover - a new fixture with no assertion is a gap
        pytest.fail(f"{stem} is in GNDS_COVARIANCE_FIXTURES with nothing asserted")


def test_the_fe56_trim_left_no_dangling_href(micro_fe56_gnds):
    """Nothing in the trim points into a file the trim dropped.

    ``build_micro_fe56_gnds.py`` drops ``externalFiles`` and everything that
    referenced it. A surviving ``$covariances#`` href would be a *silent* defect:
    every later phase would read the unresolvable link as a reader bug and go
    looking in the wrong place.
    """
    root = ET.parse(micro_fe56_gnds).getroot()
    assert root.find("externalFiles") is None
    dangling = [
        node.attrib["href"] for node in root.iter()
        if "$" in (node.attrib.get("href") or "")
    ]
    assert not dangling, f"the trim kept {len(dangling)} cross-file hrefs: {dangling[:3]}"


def test_the_fe56_trim_kept_its_resonances_whole(micro_fe56_gnds):
    """The resonance block is the only reason this fixture exists.

    The row counts are pinned deliberately. They are the trim's contract with
    every later phase — P5 asserts the decoded spin groups against these — and
    if a re-cut from a re-issued evaluation changes them, that must surface here
    and not as an unexplained failure in the resonance decoder.
    """
    root = ET.parse(micro_fe56_gnds).getroot()
    rmatrix = root.find("resonances/resolved/RMatrix")
    assert rmatrix is not None, "the trim lost its RMatrix"
    assert rmatrix.attrib.get("approximation") == "ReichMoore"

    groups = rmatrix.findall("spinGroups/spinGroup")
    assert [g.attrib["label"] for g in groups] == ["0", "1", "2", "3", "4"]
    assert [
        int(g.find("resonanceParameters/table").attrib["rows"]) for g in groups
    ] == [40, 63, 78, 75, 56]

    # The links the trim was built around: dropping the other 73 reactions is
    # only safe because these two survived to be linked to.
    #
    # The label is pulled out with a regex rather than by stripping the tail of
    # the href, because a GNDS label may itself end in ``]`` — this evaluation's
    # capture channel is literally ``Fe57 + photon [inclusive]`` — and a
    # ``strip("']")`` eats the label's own bracket. That is the first thing the
    # phase 1 xPath resolver has to get right, so it is pinned here first.
    linked = {
        match.group(1)
        for node in rmatrix.iter("link")
        for match in [re.search(r"\[@label='(.*)'\]$", node.attrib["href"])]
        if match
    }
    assert len(linked) == len(list(rmatrix.iter("link"))), (
        "a resonanceReaction link did not end in [@label='...']"
    )
    labels = {r.attrib["label"] for r in root.iter("reaction")}
    assert linked <= labels, f"resonanceReactions link to missing reactions: {linked - labels}"


def test_the_ta182_trim_kept_the_other_two_formalisms_whole(micro_ta182_gnds):
    """Fe-56 covers ``RMatrix``; this covers the rest of §19.

    Ta-182 is the smallest distributed file carrying a ``BreitWigner`` **and** a
    ``tabulatedWidths``, and its whole ``resonances`` subtree is 11.7 kB — the
    373 kB of the source is ``reactions`` and ``sums``, which is what the trim
    removes. The counts are pinned for the same reason Fe-56's are: they are the
    trim's contract with the resonance decoder.
    """
    root = ET.parse(micro_ta182_gnds).getroot()

    breitWigner = root.find("resonances/resolved/BreitWigner")
    assert breitWigner is not None, "the trim lost its BreitWigner"
    assert breitWigner.attrib["approximation"] == "MultiLevel"
    assert breitWigner.attrib.get("calculateChannelRadius") == "true"
    table = breitWigner.find("resonanceParameters/table")
    assert int(table.attrib["rows"]) == 10
    assert [c.attrib["name"] for c in table.find("columnHeaders")] == [
        "energy", "L", "J", "totalWidth", "neutronWidth", "captureWidth"
    ]

    unresolved = root.find("resonances/unresolved/tabulatedWidths")
    assert unresolved is not None, "the trim lost its tabulatedWidths"
    assert len(unresolved.findall("Ls/L")) == 2
    assert len(unresolved.findall("Ls/L/Js/J")) == 6
    # Every URR channel names a resonanceReaction, and every resonanceReaction
    # carries the <link> §19.4.1 requires — which is why the writer needs that
    # list on the model rather than reconstructing it from the width labels.
    named = {r.attrib["label"] for r in unresolved.iter("resonanceReaction")}
    assert named == {w.attrib["resonanceReaction"] for w in unresolved.iter("width")}
    assert all(r.find("link") is not None
               for r in unresolved.iter("resonanceReaction"))

    linked = {
        match.group(1)
        for node in unresolved.iter("link")
        for match in [re.search(r"\[@label='(.*)'\]$", node.attrib["href"])]
        if match
    }
    assert linked <= {r.attrib["label"] for r in root.iter("reaction")}


def test_the_trim_keeps_both_cross_section_forms(micro_fe56_gnds):
    """§9.1's own example, in a real file: ``eval`` and ``recon`` side by side.

    A ``crossSection`` holding a ``resonancesWithBackground`` labelled ``eval``
    *and* an ``XYs1d`` labelled ``recon`` is the case ``CrossSection.forms``
    exists for, and it is the one a reader that assumes one form per quantity
    gets wrong.
    """
    root = ET.parse(micro_fe56_gnds).getroot()
    elastic = next(r for r in root.iter("reaction") if r.attrib["ENDF_MT"] == "2")
    forms = {child.attrib.get("label"): child.tag
             for child in elastic.find("crossSection")}
    assert forms == {"eval": "resonancesWithBackground", "recon": "XYs1d"}


def test_the_gnds_marker_is_applied(request, gnds_data_dir):
    """``-m gnds`` selects this test, because it asked for a GNDS fixture.

    ``conftest.py`` has claimed since it was written that ``gnds`` is applied
    automatically from fixtures; nothing applied it until this phase, so
    ``-m gnds`` quietly selected nothing. This is the test that keeps the
    docstring honest.
    """
    assert request.node.get_closest_marker("gnds") is not None
