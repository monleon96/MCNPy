"""The support profile, and the half ``nodes.check()`` cannot cover.

``kika/gnds/tests/test_nodes.py`` tests that what the registry *declares*
matches what the code *does*. Nothing tested that the declaration **covers the
format** — and it does not: the registry names 58 forms at 12 choice points and
the schema declares 300 nodes. A profile built on the registry would not say
``thermalNeutronScatteringLaw`` is unsupported, it would say nothing about it,
and "nothing" is the one answer a support profile may not give.

So the tests here are chosen for what they would catch:

1. a node the schema declares and the table forgot — **the gate**, and its
   mirror for the twelve names kika uses that the schema does not have;
2. a reason that cites nothing, or cites a line that has since rotted away;
3. the registry and the table disagreeing about a node both name — with the
   join written as a **bound**, because ``NODES`` bounds a capability and does
   not fix it;
4. the guards that stop the rest passing in an empty room: a table of one
   status, a bridge that joined nothing, a ratchet nobody has seen fail;
5. the tempting refactor — deriving the statuses by importing ``NODES`` —
   which would wake the model for every consumer of the library *and* make
   test 3 tautological, in one commit.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from kika.gnds import _capabilities as cap
from kika.gnds._capabilities import CAPABILITIES, NOT_IN_SCHEMA, Coverage
from kika.gnds.nodes import NODES, Status

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CENSUS = HERE / "data" / "gnds_schema_nodes.json"
BUILDER = HERE / "data" / "build_schema_census.py"

#: What the census says the schema declares. Read once; it is the fixed point
#: every gate below measures against, and it is committed precisely so these
#: run on a machine with no FUDGE install.
SCHEMA = json.loads(CENSUS.read_text(encoding="utf-8"))
SCHEMA_NODES = SCHEMA["nodes"]

#: ``NODES`` keys the schema also declares, and the ones it does not.
_TAGS = {tag for _, tag in NODES}
BRIDGED = sorted(_TAGS & set(SCHEMA_NODES))
UNBRIDGED = sorted(_TAGS - set(SCHEMA_NODES))

#: The lattice the bridge is a bound on.
_ORDER = {Coverage.UNSUPPORTED: 0, Coverage.PARTIAL: 1, Coverage.FULL: 2}
_CEILING = {
    Status.PAIRED: Coverage.FULL,
    Status.READ_ONLY: Coverage.PARTIAL,
    Status.WRITE_ONLY: Coverage.PARTIAL,
    Status.NEITHER: Coverage.UNSUPPORTED,
}


def _ceiling(tag):
    """The best a row for *tag* may claim, given every family that declares it.

    ``min`` and not ``max``: a tag ``PAIRED`` at one choice point and
    ``NEITHER`` at another is not fully supported, it is supported *somewhere*,
    and the word for that is ``partial``.
    """
    return min((_CEILING[spec.status] for key, spec in NODES.items()
                if key[1] == tag), key=_ORDER.__getitem__)


# ---------------------------------------------------------------------------
# 1. the gate, and its mirror
# ---------------------------------------------------------------------------

def test_every_schema_node_appears_with_a_status():
    """**The gate.** A node ``gnds.xsd`` or ``covariances.xsd`` declares and
    this table does not is exactly the under-declaration the whole module
    exists to prevent — the failure of saying *nothing* about a node rather
    than saying no."""
    missing = sorted(set(SCHEMA_NODES) - set(CAPABILITIES))
    assert not missing, "\n".join(
        [f"{len(missing)} node(s) the schema declares have no capability row; "
         f"add one, or the profile under-declares by exactly this much:"] +
        [f"  <{node}>  {SCHEMA_NODES[node][0]}:{SCHEMA_NODES[node][1]}"
         for node in missing]
    )


def test_every_node_kika_names_appears_with_a_status():
    """The mirror, and the one the gate above cannot see.

    Twelve of the names ``NODES`` declares are not ``xs:element``\\ s of this
    schema at all — eight styles ``RS_StylesType`` does not admit, and four
    functional forms. The schema is not a superset of what kika writes, so a
    table keyed only on it would drop, in silence, precisely the nodes where
    kika is most likely to hand somebody a file their reader rejects.
    """
    missing = sorted(_TAGS - set(CAPABILITIES))
    assert not missing, (
        f"kika/gnds/nodes.py names {missing} and the capability table does "
        f"not. If a node is not in the schema that is a row to write, not a "
        f"row to skip"
    )


def test_no_row_names_a_node_from_neither_source():
    """The other direction: a typo, or a node a schema update withdrew.

    Without this the table can grow names nothing has ever declared and both
    gates above stay green — a profile is only worth its left-hand column.
    """
    legal = set(SCHEMA_NODES) | _TAGS
    invented = sorted(set(CAPABILITIES) - legal)
    assert not invented, (
        f"{invented} are declared by neither schema nor kika. The legal set "
        f"is the union of the two and nothing else"
    )


def test_the_twelve_unbridged_names_are_marked_as_such():
    """They are in the table and they are **not** counted as GNDS nodes.

    A row called ``realization`` inside a count of "nodes GNDS has" would
    mis-state the format. The count and the rendering both key on ``where``,
    so this is what keeps the two facts apart.
    """
    assert sorted(NOT_IN_SCHEMA) == UNBRIDGED, (
        f"NOT_IN_SCHEMA is {sorted(NOT_IN_SCHEMA)} and the registry tags the "
        f"schema does not declare are {UNBRIDGED}"
    )
    assert cap.SCHEMA_NODE_COUNT == len(SCHEMA_NODES) == 300
    for node in NOT_IN_SCHEMA:
        assert CAPABILITIES[node].where == "not in gnds.xsd"


def test_the_census_was_not_extracted_with_a_regex():
    """These twelve are what a naive ``xs:element name="…"`` scan misses,
    every one of them because ``minOccurs`` precedes ``name`` on its line.

    Four are load-bearing §19 nodes, so the cheap route would have
    under-declared exactly the chapter kika reads most carefully. This is the
    mistake written down, so simplifying the generator back to a grep fails
    here instead of shrinking the format by twelve nodes in silence.
    """
    for node in ("resolved", "unresolved", "spinGroup", "isotope", "isotopes",
                 "atomic", "J", "L", "configuration", "averageProductEnergy",
                 "resonanceReaction", "weighted"):
        assert node in SCHEMA_NODES, (
            f"<{node}> is missing from the census. It is one of the twelve a "
            f"regex over the schema drops; build_schema_census.py parses XML "
            f"for this reason"
        )


# ---------------------------------------------------------------------------
# 2. the reasons
# ---------------------------------------------------------------------------

def test_every_row_says_why_and_cites_something():
    """``test_nodes.py:64``'s rule, applied to every row and not only to the
    one-sided ones. A reason that cites nothing is how a row becomes an
    excuse, and a ``full`` claim with no citation is the most expensive kind.
    """
    for entry in CAPABILITIES.values():
        for label, text in (("why", entry.why), ("caveat", entry.caveat)):
            if text is None:
                continue
            assert text.strip(), f"<{entry.node}>'s {label} is empty"
            assert "§" in text or ".py:" in text or ".xsd:" in text, (
                f"<{entry.node}>'s {label} cites neither a § nor a "
                f"file:line: {text!r}"
            )


def test_every_citation_points_at_a_line_that_exists():
    """A ``file.py:NNN`` that rotted when the file was edited.

    Prose citations are the one thing in this repository nothing checks, and
    ``nodes.py``'s reasons carry dozens of them. This checks the table's own,
    which is as far as this file's business reaches.
    """
    pattern = re.compile(r"\b([A-Za-z_][\w/]*\.py):(\d+)")
    checked = 0
    for entry in CAPABILITIES.values():
        for text in (entry.why, entry.caveat):
            for name, line in pattern.findall(text or ""):
                candidates = list(REPO.glob(f"kika/**/{Path(name).name}"))
                assert candidates, (
                    f"<{entry.node}> cites {name}:{line} and no such file is "
                    f"in the tree"
                )
                longest = max(len(p.read_text(encoding='utf-8').splitlines())
                              for p in candidates)
                assert int(line) <= longest, (
                    f"<{entry.node}> cites {name}:{line} and the longest file "
                    f"of that name has {longest} lines"
                )
                checked += 1
    assert checked > 10, "no citation was resolved; the pattern went stale"


# ---------------------------------------------------------------------------
# 3. the bridge: NODES bounds a capability, it does not fix it
# ---------------------------------------------------------------------------

def test_the_families_column_is_what_the_registry_declares():
    """The hand-copied ``families`` against the registry, both directions.

    It is copied rather than imported because importing ``nodes`` here would
    be fine and importing it *there* would wake the model — see the last test
    in this file. Copied and checked beats derived and vacuous.
    """
    for tag in BRIDGED:
        declared = tuple(sorted(key[0] for key in NODES if key[1] == tag))
        assert tuple(sorted(CAPABILITIES[tag].families)) == declared, (
            f"<{tag}> is declared by families {declared} in nodes.py and the "
            f"capability row copies {CAPABILITIES[tag].families}"
        )
    for entry in CAPABILITIES.values():
        if entry.node not in _TAGS:
            assert not entry.families, (
                f"<{entry.node}> claims nodes.py families and nodes.py does "
                f"not declare it at all"
            )


def test_the_registry_bounds_the_capability():
    """A ``PAIRED`` entry whose writer was deleted drops the ceiling, and any
    ``full`` row above it fails here.

    Stated as an inequality on purpose. ``reference`` is ``PAIRED`` in both the
    families that declare it and is ``partial``, because §18.1.1 admits it a
    third time and kika has no entry there — a node is judged at every place
    the schema admits it, and the registry only knows about the places it
    keys. The equality half is recovered by the caveat: a row at the ceiling's
    value needs no excuse, and a row below it needs one that cites something.
    """
    for tag in BRIDGED:
        entry = CAPABILITIES[tag]
        ceiling = _ceiling(tag)
        assert _ORDER[entry.coverage] <= _ORDER[ceiling], (
            f"<{tag}> claims {entry.coverage.value} and nodes.py allows at "
            f"best {ceiling.value}"
        )
        if _ORDER[entry.coverage] < _ORDER[ceiling]:
            assert entry.caveat, (
                f"<{tag}> is {entry.coverage.value} where nodes.py allows "
                f"{ceiling.value}, and gives no caveat. Without one the table "
                f"can be quietly downgraded until the bound above passes"
            )


def test_the_bridge_joined_something():
    """The empty-room guard for the bridge itself. Every assertion above is
    worthless if ``BRIDGED`` came out empty because a key shape changed."""
    assert len(BRIDGED) == 42, (
        f"{len(BRIDGED)} registry tags are schema nodes; the two tables were "
        f"written against 42"
    )
    assert len(UNBRIDGED) == 12


def test_a_paired_entry_turned_neither_fails_the_bridge():
    """Plant one and watch it fail. Without this the bound could hold
    vacuously and the three tests above would still pass.

    The victim is ``("covarianceForm", "covarianceMatrix")`` because it can
    never legitimately become ``NEITHER``: §25.2 makes the matrix the
    mandatory form, and ``test_covariance_oracle`` collapses long before this
    would. A victim that might one day be implemented is a test that expires
    quietly, which is the mistake ``test_nodes.py:237-240`` records having
    made once already.
    """
    from kika.gnds.nodes import NodeSpec

    key = ("covarianceForm", "covarianceMatrix")
    original = NODES[key]
    assert original.status is Status.PAIRED, (
        f"{key} was chosen because it cannot stop being paired; it is now "
        f"{original.status.value}, so this test measures nothing"
    )
    NODES[key] = NodeSpec(tag=original.tag, family=original.family,
                          section=original.section, cls=original.cls,
                          status=Status.NEITHER, reason="planted by a test")
    try:
        assert _ceiling("covarianceMatrix") is Coverage.UNSUPPORTED
        with pytest.raises(AssertionError, match="covarianceMatrix"):
            test_the_registry_bounds_the_capability()
    finally:
        NODES[key] = original
    test_the_registry_bounds_the_capability()


# ---------------------------------------------------------------------------
# 4. the empty-room guards
# ---------------------------------------------------------------------------

def test_the_table_is_not_a_room_of_one_status():
    """A table of 300 ``unsupported`` rows passes every gate above and asserts
    nothing. Pinned by **bounds** rather than by counts, so this stays a guard
    and does not become a second copy of the table."""
    counted = [e for e in CAPABILITIES.values()
               if e.where != "not in gnds.xsd"]
    kinds = {e.coverage for e in counted}
    assert kinds == set(Coverage), f"only {kinds} occur"
    full = sum(1 for e in counted if e.coverage is Coverage.FULL)
    assert 100 < full < 200, f"{full} full rows; the table was written at 134"
    assert len({e.group for e in CAPABILITIES.values()}) > 25


def test_a_group_is_one_coverage():
    """A group is a shared sentence, and a sentence that covers two different
    answers is not honest about either. The rendering leans on this too."""
    kinds = {}
    for entry in CAPABILITIES.values():
        kinds.setdefault(entry.group, set()).add(entry.coverage)
    mixed = {group: sorted(c.value for c in seen)
             for group, seen in kinds.items() if len(seen) > 1}
    assert not mixed, f"these groups claim two coverages at once: {mixed}"


def test_the_silent_drops_are_pinned_by_count():
    """Twenty-one nodes vanish with nothing said. **Each repair lowers this
    number and this test is what notices** — the shape ``KNOWN_DEFECTS`` has
    next door.

    Most are honest: a child of a container already reported, a branch the
    reader never enters, a node no valid document can contain. One is not.
    ``targetInfo`` hangs off ``<evaluated>``, which kika reads, and the style
    reader takes three children and never looks at the rest — so it is dropped
    under a node kika claims to support. That is the entry this number should
    lose first.
    """
    silent = sorted(e.node for e in cap.capabilities().silent)
    assert len(silent) == 21, silent
    assert "targetInfo" in silent and "isotopicAbundances" in silent
    for node in silent:
        assert CAPABILITIES[node].why


def test_reportedVia_names_a_node_the_table_has():
    """A ``reportedVia`` pointing at a node that was renamed, or at one that
    has since become ``full`` and no longer reports anything."""
    for entry in CAPABILITIES.values():
        via = entry.reportedVia
        if via is None:
            continue
        assert via in CAPABILITIES, (
            f"<{entry.node}> says it is reported via <{via}>, which is not a "
            f"node this table has"
        )
        assert entry.coverage is not Coverage.FULL, (
            f"<{entry.node}> is full and names a report line; a full node "
            f"loses nothing to report"
        )


# ---------------------------------------------------------------------------
# 5. the API
# ---------------------------------------------------------------------------

def test_filters_compose_and_an_unknown_word_raises():
    """"I do not know that word" and "there is nothing of that kind" are
    different answers, and a support profile is the last place to conflate
    them — an empty view would read as "kika supports none of that"."""
    everything = cap.capabilities()
    assert len(everything) == len(CAPABILITIES)
    assert 0 < len(cap.capabilities(coverage="partial")) < len(everything)
    assert len(cap.capabilities(group="thermalScattering")) == 24
    assert cap.capabilities(node="XYs1d")[  # the exact-node form
        "XYs1d"].coverage is Coverage.FULL
    with pytest.raises(ValueError, match="not a coverage"):
        cap.capabilities(coverage="mostly")
    with pytest.raises(ValueError, match="not a group"):
        cap.capabilities(group="thermal")
    with pytest.raises(KeyError):
        cap.capabilities(node="notANode")


def test_the_rendering_names_every_node_and_sums_correctly():
    """A renderer that truncates, or a summary whose counts drifted from the
    rows they claim to count."""
    text = cap.capabilities().text()
    for node in CAPABILITIES:
        assert re.search(rf"(?<![\w-]){re.escape(node)}(?![\w-])", text), (
            f"<{node}> has a row and does not appear in the rendering"
        )
    counted = [e for e in CAPABILITIES.values()
               if e.where != "not in gnds.xsd"]
    summary = cap.capabilities().summary()
    for coverage in Coverage:
        n = sum(1 for e in counted if e.coverage is coverage)
        assert f"{n} {coverage.value}" in summary, summary
    # every reason survives the render, including the ones a group-keyed
    # renderer would drop by showing only its first member's
    for entry in CAPABILITIES.values():
        assert entry.why.split()[0] in text


def test_asking_what_kika_supports_does_not_wake_the_model():
    """The tempting refactor, which would cost twice.

    Deriving the statuses by importing ``NODES`` into ``_capabilities`` would
    (a) make ``import kika.gnds`` pull in ``kika.nuclear_data.model`` for the
    cluster pipeline, the desktop app and every notebook, and (b) make
    ``test_the_registry_bounds_the_capability`` true by construction. This
    catches the first, which is the one a reviewer would not see.

    A subprocess for ``test_dormancy``'s reason: by the time this module is
    collected the model is already in ``sys.modules``.
    """
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import sys
            import kika.gnds
            kika.gnds.capabilities().text()
            leaked = sorted(m for m in sys.modules
                            if m.startswith("kika.nuclear_data.model"))
            assert not leaked, (
                "asking what kika supports woke the model: " + ", ".join(leaked)
            )
            print("dormant")
        """)],
        capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == "dormant"


# ---------------------------------------------------------------------------
# 6. drift, where the schema is installed
# ---------------------------------------------------------------------------

@pytest.mark.tape
def test_the_census_matches_the_live_schema():
    """The committed census against FUDGE's own files.

    This is the only test here that can be skipped, and skipping it leaves
    nothing unchecked — the gate above reads the committed census and runs
    everywhere. That is the whole reason the census is committed rather than
    parsed at test time: ``build_micro_fe56_gnds.py`` and ``test_encode.py``
    both go quiet on a machine without FUDGE, and a gate that goes quiet is
    not a gate.
    """
    from kika.gnds.tests.data import build_schema_census as builder

    if not all(path.exists() for _, path in builder.SCHEMAS):
        pytest.skip("FUDGE's schemas are not on this machine")
    live = builder.extract()
    assert set(live["nodes"]) == set(SCHEMA_NODES), (
        "the schemas no longer declare what the census says; re-run "
        "build_schema_census.py and give every new node a capability row"
    )
    for label, meta in live["schemas"].items():
        assert meta["sha256"] == SCHEMA["schemas"][label]["sha256"], (
            f"{label} changed under the census"
        )
