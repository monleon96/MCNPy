"""Build the trimmed GNDS resonance fixtures from the distributed library.

Run by hand, not by the test suite::

    poetry run python kika/gnds/tests/data/build_micro_fe56_gnds.py

**Why a trim exists at all when every other GNDS fixture here is unmodified.**
The five files copied verbatim from the distribution (``n-001_H_002`` and its
covariance sibling, ``n-009_F_019``, ``n-014_Si_032``, ``n-057_La_139``,
``n-069_Tm_171``) between them carry every covariance construct the library
uses, and H-2 carries the cross-file ``href`` and its SHA-1 checksum. What none
of them carries is a **resonance region of any kind**, and §19 is where the two
formalisms and the unresolved region live. The smallest distributed file with an
``RMatrix`` is Ce-142 at 1.6 MB, and the material this project is actually about
— Fe-56 — is 18.8 MB.

So two files are trimmed, between them covering §19 completely:

``micro_fe56``    ``RMatrix`` (Reich-Moore), 5 spin groups, and the
                  ``resonancesWithBackground`` cross section form that points at
                  it. The material the thesis is about.
``micro_ta182``   ``BreitWigner`` (MultiLevel) **and** ``tabulatedWidths``, the
                  other resolved formalism and the unresolved region, in one
                  file. Ta-182's whole ``resonances`` subtree is 11.7 kB; the
                  373 kB is all ``reactions`` and ``sums``, which is exactly
                  what this trim removes.
``micro_sr88``    §19.3.4's ``externalRMatrix``. Added 2026-08-24 with the row
                  of ``gnds_endf_conflicts.md`` §6.4 it closes: the node exists
                  **7 times in the whole library and all 7 are in this file**,
                  so there was no fixture that could gate reading or writing it
                  and the reader reported it instead. Its ``resonances`` is
                  41 kB.

**What is kept, whole and unedited:** ``resonances`` — the ``scatteringRadius``,
the ``resolved`` region with its formalism, the ``resonanceReactions``, the
``spinGroups`` (or ``Ls``/``Js``), their ``channels`` and every
``resonanceParameters/table``. Not one number in a resonance block is touched,
because that block is the reason these files exist.

**What is dropped, and why each drop is safe:**

``externalFiles``      the covariance sibling is 4.8 MB and is not shipped. Every
                       ``uncertainty``/``covariance`` node pointing into it is
                       dropped with it, so the trim leaves **no dangling href**.
                       Cross-file href resolution is tested on H-2 instead,
                       where the file and its checksum are both real.
``sums``               ``crossSectionSums/total`` names every reaction in its
                       ``summands``; keeping it after dropping most of them
                       would leave that many dangling hrefs. H-2 carries
                       ``crossSectionSum`` with resolvable summands.
``applicationData``    FUDGE's ENDF round-trip hints. kika does not read them.
most ``reaction``      all but the ones the resonance block's own
                       ``resonanceReactions`` link to, so dropping the rest
                       leaves those links intact. Named per file in
                       :data:`TARGETS`; the build refuses if one is absent.
unreferenced ``PoPs``  the source ships 172 nuclides across 12 chemical
                       elements, **262 kB of the 310 kB** the trim would
                       otherwise weigh, describing particles no kept reaction
                       mentions. Only the particles named by a surviving
                       ``pid``/``target``/``projectile``/``ejectile`` are kept.
                       ``gaugeBosons``, ``baryons`` and ``aliases`` stay whole:
                       they are a few hundred bytes and the aliases are how a
                       reader resolves ``photon`` onto ``gamma``.

**What is truncated rather than dropped:** tabulated payloads outside the
resonance block. An ``XYs1d`` keeps its first :data:`MAX_XY_PAIRS` pairs (an
even count, so the interleaved ``x y x y`` layout stays coherent), an ``XYs2d``
its first :data:`MAX_SUBFUNCTIONS` children, and the ``documentation`` text
blocks their first :data:`MAX_DOC_LINES` lines. The result is **structurally
faithful and numerically abridged**: it is a fixture for a reader, not an
evaluation, and no test may treat its cross sections as physics. The oracle
tests that compare numbers against the ENDF path read the full file from
``$KIKA_TAPES`` under the ``gnds`` marker instead.

The output is validated against FUDGE's own GNDS 2.0 schema before it is
written, so a trim that produces a structurally invalid file fails here rather
than three phases later::

    xmllint --noout --schema /soft_snc/FUDGE/6.10.0/fudge/fudge/gnds.xsd micro_fe56.gnds.xml
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: The root the distributed library sits under. Overridable for a machine whose
#: share is mounted elsewhere; the default is the one `conftest.py` looks under.
TAPES = Path(os.environ.get("KIKA_TAPES", "/share_snc/snc/JuanMonleon"))
NEUTRONS = TAPES / "ENDF-B-VIII.1-GNDS/ENDF-B-VIII.1-GNDS/neutrons"

#: FUDGE's GNDS 2.0 schema. Absent on a machine without FUDGE, in which case the
#: validation step is skipped and says so.
SCHEMA = Path("/soft_snc/FUDGE/6.10.0/fudge/fudge/gnds.xsd")

#: ``output name -> (source file, the reactions to keep)``. For the two §19
#: fixtures the kept reactions are exactly the ones the file's own
#: ``resonanceReactions`` link to; keeping those and no others is what makes the
#: resonance block's links resolve in the trim, and the build refuses rather
#: than produce a dangling one.
#:
#: ``micro_be9`` is here for a different reason and keeps a different reaction.
#: Be-9's ``resonances`` is a lone ``scatteringRadius`` — 295 bytes, no
#: ``resonanceReactions``, nothing to link — so the reaction it keeps is chosen
#: by what it carries instead: ``2n + 2He4`` holds **both** ``angularEnergy``
#: nodes in the file, and that file is the whole population of §18.5 across the
#: 558 distributed neutron evaluations. Two occurrences, one file, 0.92 MB —
#: which is exactly the condition this script exists for.
TARGETS = {
    "micro_fe56.gnds.xml": (
        "n-026_Fe_056.endf.gnds.xml",
        ("n + Fe56", "Fe57 + photon [inclusive]"),
    ),
    "micro_ta182.gnds.xml": (
        "n-073_Ta_182.endf.gnds.xml",
        ("n + Ta182", "Ta183 + photon [inclusive]"),
    ),
    "micro_be9.gnds.xml": (
        "n-004_Be_009.endf.gnds.xml",
        ("2n + 2He4",),
    ),
    # Same condition as `micro_be9`, one section further on: Sr-88 is the whole
    # population of §19.3.4's `externalRMatrix` across the 558 distributed
    # neutron evaluations -- 7 nodes, one file, all `type="SAMMY"`, measured
    # 2026-08-24. Its `resonances` block is 41 kB of the file's 5.2 MB, so the
    # trim is almost entirely `reactions` and `sums`, exactly as Ta-182's is.
    "micro_sr88.gnds.xml": (
        "n-038_Sr_088.endf.gnds.xml",
        ("n + Sr88", "Sr89 + photon [inclusive]"),
    ),
}

#: reactionSuite children dropped wholesale. See the module docstring for why
#: each one cannot simply be kept.
DROP_CHILDREN = ("externalFiles", "sums", "applicationData")

#: Truncation limits. Chosen to keep the file readable by eye; nothing depends
#: on the exact values.
MAX_XY_PAIRS = 20
MAX_SUBFUNCTIONS = 6
MAX_DOC_LINES = 8

#: The subtree that is never touched: no truncation, no pruning, nothing.
PROTECTED = "resonances"


def _truncateValues(element: ET.Element, pairs: int) -> None:
    """Keep the first ``pairs`` ``(x, y)`` of an interleaved ``values`` node.

    §6.1.1 serialises an ``XYs1d`` as ``x0 y0 x1 y1 ...`` in one ``values``
    node, so the kept count must be **even** or the last point loses its
    ordinate and the file stops being an XYs1d at all.
    """
    text = (element.text or "").split()
    if len(text) <= 2 * pairs:
        return
    element.text = " ".join(text[: 2 * pairs])


def _truncateDocumentation(element: ET.Element) -> None:
    """Shorten the evaluator's comment block, keeping it a valid CDATA body."""
    for node in element.iter():
        if node.tag in ("body", "endfCompatible") and node.text:
            lines = node.text.splitlines()
            if len(lines) > MAX_DOC_LINES:
                node.text = "\n".join(lines[:MAX_DOC_LINES] + ["[trimmed]"])


def _hasCovarianceHref(element: ET.Element) -> bool:
    """Whether this subtree points into the covariance file the trim drops."""
    for node in element.iter():
        if "$covariances#" in (node.attrib.get("href") or ""):
            return True
    return False


def _referencedParticles(root: ET.Element) -> set:
    """Every particle id the surviving tree names, PoPs itself excluded.

    Read off the attributes that hold one — ``pid`` on a product, ``target``
    and ``projectile`` on the root, ``ejectile`` on a ``resonanceReaction`` —
    rather than from a hardcoded list, so a relabelled evaluation trims to what
    it actually references instead of to what this script remembers.
    """
    referenced = {root.attrib.get("projectile"), root.attrib.get("target")}
    for child in root:
        if child.tag == "PoPs":
            continue
        for node in child.iter():
            for name in ("pid", "ejectile", "id"):
                if name in node.attrib and node.tag != "nucleus":
                    referenced.add(node.attrib[name])
    return {pid for pid in referenced if pid}


def _prunePoPs(root: ET.Element) -> None:
    """Drop every nuclide the surviving tree never names."""
    pops = root.find("PoPs")
    if pops is None:
        return
    keep = _referencedParticles(root)
    # An alias redirects one id onto another (`photon` -> `gamma`), so a
    # referenced alias keeps its target alive too.
    for alias in pops.iter("alias"):
        if alias.attrib.get("id") in keep and alias.attrib.get("pid"):
            keep.add(alias.attrib["pid"])
    # A nuclide is named by its own id and by the isotope it belongs to; both
    # spellings appear in the wild, so match on either.
    for elements in pops.findall("chemicalElements"):
        for element in list(elements):
            for isotopes in element.findall("isotopes"):
                for isotope in list(isotopes):
                    for nuclides in isotope.findall("nuclides"):
                        for nuclide in list(nuclides):
                            if nuclide.attrib.get("id") not in keep:
                                nuclides.remove(nuclide)
                    if not any(len(n) for n in isotope.findall("nuclides")):
                        isotopes.remove(isotope)
                if len(isotopes) == 0:
                    element.remove(isotopes)
            if element.find("isotopes") is None:
                elements.remove(element)


def _prune(element: ET.Element) -> None:
    """Truncate and prune one subtree in place, leaving ``resonances`` alone."""
    if element.tag == PROTECTED:
        return

    for child in list(element):
        # An `uncertainty` whose `covariance` points at the dropped external
        # file would be a dangling href, which is the one defect a fixture must
        # not have: every later phase would read it as a reader bug.
        if child.tag == "uncertainty" and _hasCovarianceHref(child):
            element.remove(child)
            continue
        _prune(child)

    if element.tag == "values":
        _truncateValues(element, MAX_XY_PAIRS)
    elif element.tag in ("XYs2d", "regions2d", "function1ds", "function2ds"):
        for extra in list(element)[MAX_SUBFUNCTIONS:]:
            if extra.tag != "axes":
                element.remove(extra)


def build(source: Path, keepReactions) -> ET.ElementTree:
    if not source.exists():
        raise SystemExit(
            f"{source} is not there. This script cuts the committed fixtures out "
            f"of the distributed ENDF/B-VIII.1 GNDS library; set KIKA_TAPES to "
            f"the root that holds ENDF-B-VIII.1-GNDS/."
        )

    tree = ET.parse(source)
    root = tree.getroot()

    for name in DROP_CHILDREN:
        for child in root.findall(name):
            root.remove(child)

    reactions = root.find("reactions")
    for reaction in list(reactions):
        if reaction.attrib.get("label") not in keepReactions:
            reactions.remove(reaction)
    kept = [r.attrib.get("label") for r in reactions]
    missing = [label for label in keepReactions if label not in kept]
    if missing:
        raise SystemExit(
            f"{source.name} has no reaction labelled {missing}; the resonance "
            f"block's resonanceReactions link to them, so the trim would leave "
            f"dangling hrefs. Has the evaluation been relabelled?"
        )

    for child in root:
        if child.tag == "styles":
            _truncateDocumentation(child)
        _prune(child)

    # After the reactions are pruned, not before: what PoPs may drop depends on
    # which reactions survived to name a particle.
    _prunePoPs(root)
    return tree


def validate(path: Path) -> None:
    if not SCHEMA.exists() or shutil.which("xmllint") is None:
        # Skipped rather than fatal, and **said out loud**, because a fixture
        # built without validation is the exact failure this script exists to
        # prevent. It used to skip only on a missing schema and then die with
        # `FileNotFoundError` on a machine that has FUDGE and no `xmllint` —
        # which is every WSL box here. `test_encode.py:322` has always guarded
        # both; this now does the same.
        print(f"[skip] {SCHEMA} or xmllint is absent; the trim was NOT "
              f"schema-validated. Validate it on a machine that has both "
              f"before committing it.")
        return
    result = subprocess.run(
        ["xmllint", "--noout", "--huge", "--schema", str(SCHEMA), str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"the trimmed file does not validate against GNDS 2.0:\n{result.stderr}"
        )
    print(f"[ok] validates against {SCHEMA.name}")


def main() -> None:
    for name, (sourceName, keepReactions) in TARGETS.items():
        tree = build(NEUTRONS / sourceName, keepReactions)
        ET.indent(tree, space="  ")
        with tempfile.NamedTemporaryFile("wb", suffix=".xml", delete=False) as handle:
            temporary = Path(handle.name)
        tree.write(temporary, encoding="UTF-8", xml_declaration=True)
        validate(temporary)
        output = HERE / name
        shutil.move(str(temporary), output)
        print(f"[ok] {output}  {output.stat().st_size / 1024:.0f} kB")


if __name__ == "__main__":
    sys.exit(main())
