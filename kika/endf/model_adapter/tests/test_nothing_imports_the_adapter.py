"""The adapters must stay off every critical path but their own.

``kika.endf.model_adapter`` and ``kika.ace.model_adapter`` each import the whole
GNDS model. If anything in ``kika.endf``/``kika.ace`` imports *them* at module
scope, then ``read_endf`` and ``read_ace`` — which the cluster pipeline, the
desktop app and every notebook call — start building the model on every parse,
and ``import kika`` gets slower for everyone to no purpose.

There is a second, sharper reason, and phase 3d ran into it. A lazy import from
an app-reachable code path is invisible to PyInstaller's modulegraph, so a
package missing from kika-app's ``hiddenimports`` raises ``ModuleNotFoundError``
in the frozen binary and nowhere else — dev, tests and CI all green. That is the
exact failure nine other modules were found in during P2. When the flat classes
became façades, ``kika.endf.model_adapter`` and ``kika.nuclear_data.model`` were
still absent from that list, and the desktop app would have broken the first
time it read a cross section. They are in the spec now; keep them there.

So this file is no longer "nothing imports the adapter" — it is "only these
modules do", which is a scoreboard rather than a prohibition. The allowlist has
**two halves with opposite futures**: the four flat-class façades, which empty
when the classes go in 1.0, and the front door `kika/_read.py`, which is the
library's public entry onto the model and stays forever. They are kept as
separate sets so that neither is mistaken for the other.

It lives under the ENDF adapter's tests and covers both adapters, rather than
being copied into each: one place to add the third adapter to.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
ADAPTERS = ("kika.endf.model_adapter", "kika.ace.model_adapter")

#: The only modules allowed to import an adapter, and why.
#:
#: **Phase 3d added these, and it is a real cost rather than a formality.** The
#: flat classes are façades over the model now, so their ``from_endf`` reaches
#: the adapter — inside the method, never at module scope, so ``read_endf`` is
#: still untouched and the dormancy test below still holds.
#:
#: What it does cost is the frozen build. ``kika.endf.model_adapter`` and
#: ``kika.nuclear_data.model`` were absent from kika-app's PyInstaller
#: ``hiddenimports``, and modulegraph cannot see a function-scope import — so
#: the first version of the façade would have raised ``ModuleNotFoundError``
#: the first time the desktop app read a cross section, with dev, tests and CI
#: all green. They are in the spec now. **Adding a module here means checking
#: that spec.**
FACADE_IMPORTERS = {
    "kika/nuclear_data/cross_section.py",
    "kika/nuclear_data/angular_distribution.py",
    "kika/nuclear_data/resonance_parameters.py",
    "kika/nuclear_data/nuclide_info.py",
}

#: Modules that import an adapter **and are meant to, permanently**.
#:
#: Kept apart from ``FACADE_IMPORTERS`` because the two have opposite futures.
#: That list is a deprecation scoreboard that empties at 1.0; this one is the
#: library's public entry point and empties never. Merging them would mean that
#: when the flat classes go, whoever deletes the last façade entry finds the list
#: still non-empty and has to work out why.
#:
#: **Phase 3e (2026-08-10) added the front door.** ``kika/_read.py`` is
#: ``kika.read`` — the one door onto the model — so importing the adapters is its
#: entire job. It does so *inside* ``_readEndf``/``_readAce`` rather than at
#: module scope, which is what keeps ``import kika`` from waking the model.
#:
#: **The frozen build was checked, and needs nothing.** A function-scope import
#: is invisible to PyInstaller's modulegraph, so the rule stands: adding an entry
#: here means checking kika-app's ``kika-api.spec``. Checked on 2026-08-10 —
#: ``kika.nuclear_data.model``, ``kika.endf.model_adapter`` and
#: ``kika.ace.model_adapter`` are already in ``hiddenimports`` (spec lines
#: 143-146) from phase 3d, and **kika-app calls no ``kika.read``** (zero grep
#: hits), so no spec change was owed. If the app ever does call it, ``kika._read``
#: has to go in that list — it is reached through ``kika.__getattr__``, which
#: modulegraph cannot see either.
#:
#: **Phase P2 of the PFNS roadmap (2026-08-10) added the second.**
#: ``kika/sampling/mf35_sampling.py`` reads MF35 into a ``CovarianceSuite``,
#: because the library's format-agnostic covariance object is the GNDS one and
#: a PFNS-specific container would have been the fourth. The import is inside
#: ``build_pfns_covariance``, so ``import kika.sampling`` — which the cluster
#: pipeline does on every run — still does not wake the model.
#:
#: **The frozen build was checked and needs nothing.** kika-app imports
#: ``kika.sampling.{mf31_sampling, utils, ace_perturbation,
#: ace_perturbation_separate, endf_perturbation, nubar_perturbation}`` and none
#: of the PFNS modules (grep, 2026-08-10), so no path in the app reaches this
#: import. If the app ever calls ``perturb_pfns_files``, ``kika-api.spec``'s
#: ``hiddenimports`` is where that has to be recorded.
#:
#: **P1 of the sampling migration (2026-08-11) added the third.**
#: ``kika/sampling/endf_perturbation.py`` reads MF34 into a ``CovarianceSuite``
#: in ``load_mf34_suite``, for the same reason the PFNS entry above does: the
#: library's format-agnostic covariance object is the GNDS one, and the
#: alternative was keeping ``LegendreCovariance`` on the sampling path forever.
#: The import is inside the function, so ``import kika.sampling`` still does not
#: wake the model.
#:
#: **The frozen build was checked, and this one is a live app path** — unlike the
#: PFNS entry. kika-api's ``routers/sampling.py`` imports ``perturb_ENDF_files``
#: at module scope and calls it, so the adapter really is reached in the frozen
#: binary. It needs no spec change *only* because ``kika.endf.model_adapter``
#: and ``kika.nuclear_data.model`` are already in ``kika-api.spec``'s
#: ``hiddenimports`` from phase 3d (spec lines 143-146). Verified 2026-08-11 —
#: if those entries are ever removed, this endpoint breaks in the frozen build
#: and nowhere else.
#:
#: **Phase 4 P1a (2026-08-12) added the fourth.**
#: ``kika/endf/processing/reconstruct.py`` is the ENDF adapter for resonance
#: reconstruction: MF2/151 in, ``MF3MT`` out. The physics moved onto the model,
#: so the middle of that sandwich is now ``decodeMF2MT151`` — which makes this
#: module an ENDF↔model boundary, and that is exactly what the adapter is for.
#: The alternative was to keep converting through the flat
#: ``ResonanceParameters``, whose ``c3..c6`` cannot hold an LRF=7 spin group,
#: and that is the defect the phase exists to fix.
#:
#: The imports are inside ``reconstruct()``, so ``import kika.endf.processing``
#: — which kika-app does, for ``detect_resonance_bounds`` — still does not wake
#: the model. **The frozen build was checked and needs nothing:**
#: ``kika.endf.model_adapter`` and ``kika.nuclear_data.model`` have been in
#: ``kika-api.spec``'s ``hiddenimports`` since phase 3d (spec lines 143-146),
#: and no path in kika-app reaches this function — the app reconstructs with
#: NJOY, through ``kika.processing.njoy_reconstruct``.
#: **§2.8 (2026-08-20) added the fifth.** ``kika/endf/writers/assemble.py`` is
#: the whole-file model → ENDF-6 writer: a ``ReactionSuite`` in, a tape out. It
#: is the mirror of ``kika/_read.py``'s ENDF half and importing the adapter is
#: its entire job, so it belongs in this half of the list rather than the
#: façade half. Every import is inside a function, so
#: ``import kika.endf.writers`` — which kika-app does, for ``section_ops`` and
#: ``update_directory`` — still does not wake the model.
#:
#: **The frozen build was checked and needs nothing, and the reason is not the
#: one the entries above give.** kika-app reaches no path in this module (grep
#: for ``kika.endf.writers.assemble``, ``writeEndfTape`` and ``kika.write``:
#: zero hits, 2026-08-20), which is what makes it safe — *not* the adapter
#: being in ``kika-api.spec``'s ``hiddenimports``, because as of 2026-08-20 it
#: is **not**: the four model/adapter entries are commented out and parked
#: there deliberately, since the commit the app pins predates them. If the app
#: ever calls this writer, restoring those four is a precondition, and the
#: comment block at ``kika-api.spec:128-153`` is where that is written down.
PERMANENT_IMPORTERS = {
    "kika/_read.py",
    "kika/sampling/mf35_sampling.py",
    "kika/sampling/endf_perturbation.py",
    "kika/endf/processing/reconstruct.py",
    "kika/endf/writers/assemble.py",
}

ALLOWED_IMPORTERS = FACADE_IMPORTERS | PERMANENT_IMPORTERS


def _importers(adapter: str, root: Path | None = None) -> list[str]:
    """Every **shipped** module under kika/ that imports ``adapter``.

    Adapter packages are skipped for the obvious reason. Test packages are
    skipped for a stated one: both harms this ratchet guards against are harms to
    *shipped* code — ``read_endf`` building the model on every parse, and a
    function-scope import PyInstaller cannot see. A test module is on neither the
    pipeline's path nor the frozen build's, and a test that checks the adapter has
    to import the adapter. Scanning them would make the ratchet fire on its own
    coverage, and the usual fix for that is to delete the coverage.
    """
    root = root if root is not None else REPO_ROOT
    found: list[str] = []
    for path in sorted((root / "kika").rglob("*.py")):
        relative = path.relative_to(root)
        if "model_adapter" in relative.parts or "tests" in relative.parts:
            continue
        tree = ast.parse(path.read_text(errors="replace"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if _resolves_to(node, relative, adapter):
                    found.append(f"{relative}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(adapter):
                        found.append(f"{relative}:{node.lineno}")
    return found


def _resolves_to(node: ast.ImportFrom, relative: Path, adapter: str) -> bool:
    """Does this ``from ... import`` name the adapter, absolute *or* relative?

    **Relative imports used to slip through**, and that is not hypothetical:
    ``from ..endf.model_adapter import decodeCovarianceSuite`` inside
    ``kika/sampling/`` has ``node.module == "endf.model_adapter"``, which does
    not start with ``kika.endf.model_adapter``, so the ratchet said nothing. A
    module could evade the allowlist by spelling its import differently, which
    is exactly the sort of accident this file exists to prevent. Found while
    adding the PFNS entry above; the import there is absolute now *and* on the
    list, but the hole is closed here so the next one cannot choose.
    """
    module = node.module or ""
    if node.level == 0:
        return module.startswith(adapter)
    # `relative` is like kika/sampling/mf35_sampling.py; level 1 is its own
    # package, level 2 the parent, and so on.
    package = relative.parts[:-1]
    base = package[: len(package) - (node.level - 1)]
    resolved = ".".join([*base, module]) if module else ".".join(base)
    return resolved.startswith(adapter)


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_only_the_facades_and_the_front_door_import_the_adapter(adapter):
    unexpected = [
        entry for entry in _importers(adapter)
        if entry.rsplit(":", 1)[0] not in ALLOWED_IMPORTERS
    ]
    assert not unexpected, (
        f"{adapter} is imported by a module that is neither one of the four flat "
        f"classes nor the front door:\n  " + "\n  ".join(unexpected)
        + "\n\nIt pulls in the whole GNDS model. Code that wants the model should "
          "ask the adapter for it directly, not reach it through kika.nuclear_data. "
          "There is one public entry point onto the model and it is kika.read; a "
          "second one is almost certainly a mistake. If it really is not, add it "
          "to PERMANENT_IMPORTERS *and* to kika-app/kika-api.spec's hiddenimports "
          "-- a function-scope import is invisible to PyInstaller."
    )


def test_the_allowlist_has_no_stale_entries():
    """A module that stopped importing the adapter must leave the list.

    Same shape as ``test_layering.py``'s ratchet: the list is a scoreboard for a
    deprecation, and it should only ever shrink. When the flat classes go in 1.0
    it empties and this file goes back to asserting *nothing* imports them.
    """
    importing = {
        entry.rsplit(":", 1)[0]
        for adapter in ADAPTERS for entry in _importers(adapter)
    }
    stale = sorted(ALLOWED_IMPORTERS - importing)
    assert not stale, (
        f"these no longer import an adapter and should be removed from "
        f"ALLOWED_IMPORTERS: {stale}"
    )


#: Built by concatenation rather than `textwrap.dedent` on an f-string: `body`
#: is multi-line, so its second line lands at column 0 and dedent then finds a
#: common prefix of "" and strips nothing, leaving the template's own lines
#: indented. The failure is an IndentationError in the subprocess, which reads
#: like the test being wrong about imports.
_LEAK_CHECK = """
leaked = sorted(
    m for m in sys.modules
    if m.startswith("kika.nuclear_data.model")
    or m.startswith("kika.endf.model_adapter")
    or m.startswith("kika.ace.model_adapter")
)
assert not leaked, "the model was built: " + ", ".join(leaked)
print("clean")
"""


def _leakCheck(body: str) -> str:
    return "import sys\n" + body + "\n" + _LEAK_CHECK


def _runClean(body: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _leakCheck(body)],
        capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == "clean"


def test_reading_an_endf_file_does_not_build_the_model(micro_tape):
    """The runtime half: parsing a tape must not wake the model."""
    _runClean(
        "from kika.endf.read_endf import read_endf\n"
        f"read_endf({str(micro_tape)!r})"
    )


@pytest.mark.tape
def test_reading_an_ace_file_does_not_build_the_model(fe56_ace):
    """The same, for ACE. ``tape``-marked: no ACE fixture is committed."""
    _runClean(
        "from kika.ace import read_ace\n"
        f"read_ace({str(fe56_ace)!r})"
    )


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_the_check_would_notice_an_importer(tmp_path, adapter):
    """A ratchet nobody has seen fail is not a ratchet."""
    fake = tmp_path / "kika" / "endf"
    fake.mkdir(parents=True)
    (fake / "innocent.py").write_text("import numpy\n")
    (fake / "guilty.py").write_text(f"from {adapter}.decode import somethingOrOther\n")

    found = _importers(adapter, root=tmp_path)
    assert found == ["kika/endf/guilty.py:1"]
    assert found[0].rsplit(":", 1)[0] not in ALLOWED_IMPORTERS
