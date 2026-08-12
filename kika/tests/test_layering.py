"""The layering ratchet: format code must not leak into the calculation layer.

`kika/tests/` is the one place in this repo that is not a package's own test
directory. What lives here are invariants of the repository as a whole, which
belong to no single package. This is the first of them.

**The invariant.** ``kika.processing``, ``kika.nuclear_data`` and ``kika.cov``
are the calculation layer. ``kika.endf`` and ``kika.ace`` are format layers. The
arrow points one way: formats may import the calculations, the calculations may
not import the formats.

**``kika/cov`` joined the guarded set before phase 4, not during it.** The
roadmap's phase 4 bullet is "extend the ratchet to ``kika/cov``: no
``kika.endf`` imports outside the encoders", and it belongs *before* the phase
that rewrites ``CrossSectionCovariance`` and ``LegendreCovariance`` to read and
write through ``CovarianceSuite`` — a net is worth nothing installed after the
fall. Nothing in ``kika/cov`` was changed to add it; the seven imports it had
then were written down as they were, which is what a ratchet is for. **Phase 4's
P4 has since taken it to six** — see the note on ``legacy_mg_plotting.py``
below.

The phase-4 wording needs one gloss. "Outside the encoders" is not a rule this
test can express, because ``kika/cov`` is not cleanly one layer: MF34 encoding
in ``legendre_covariance.py`` is format code that the roadmap explicitly keeps
("their COVERX/COVFIL/BOXER/GENDF I/O are kept as format encoders, on the same
footing as ENDF and ACE"), while ``parse_covmat.py`` reaching for ENDF's record
grammar is a genuine leak. A per-file frozen count says both at once: the
encoder's entries stay and are explained, the leak's entry stays and is
explained, and neither may grow.

**Why a ratchet and not a rule.** The arrow was inverted in places, worst of
all ~960 lines of pure numerics living inside ``kika/endf/processing/`` and
re-exported *upward* by shims in ``kika/processing/``. Phase 2 of the GNDS
roadmap moved four of them — ``linearization``, ``penetration``,
``resonance_formulas``, ``urr_formulas`` — and deleted the shims, which had no
other importers. ``resonance_bounds`` stayed put: it reads
``mf2.mt[151].isotopes`` and ``rng.lru/el/eh``, so it is an ENDF adapter and
moving it would have created a *new* violation rather than removing one.

Violations still exist, so a hard rule would just fail. This test freezes the
count instead: every violation that exists today is written down, and the count
per file may go **down** without touching this file but never up. Deleting a
violation needs no edit here; adding one fails immediately.

**What the count cannot see.** It counts import *statements*, not when they
run. The three surviving ``kika/processing`` entries are all deferred to call
time, which is a real improvement the number does not show — and the reason it
matters is concrete: while ``kika.processing`` imported ``kika.endf`` at import
time, nothing in ``kika.endf`` could import from ``kika.processing`` at module
scope, so ``interpolate_1d`` could not be moved down at all.

**Two kinds, counted separately.** A runtime import genuinely couples the
layers. An import under ``if TYPE_CHECKING:`` couples only the vocabulary — it
costs nothing at run time but still means the calculation layer describes
itself in ENDF's and ACE's types, which is precisely what the GNDS canonical
model is meant to end. Both are frozen; the runtime figure is the one that
matters.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Packages whose modules must not import a format package.
GUARDED_PACKAGES = ("kika/processing", "kika/nuclear_data", "kika/cov")

#: Packages they must not import.
FORBIDDEN_ROOTS = ("kika.endf", "kika.ace")

#: Runtime imports of a format package, per module. May only decrease.
#:
#: Phase 2 took kika/processing from 8 entries to 3. It does **not** reach
#: zero, and the original roadmap was wrong to say it would: the three that
#: remain each need ``kika.endf.read_endf`` to parse a PENDF tape, which is a
#: real dependency and not a layering accident. All three are deferred to call
#: time, so ``kika.processing`` no longer pulls ``kika.endf`` in at *import*
#: time — that mattered, because while it did, nothing in ``kika.endf`` could
#: import from ``kika.processing`` at module scope, and ``interpolate_1d``
#: could not move down at all.
#:
#: **Phase 3d makes this figure go up, on purpose.** The flat classes become
#: façades whose ``from_endf``/``to_endf`` route through
#: ``kika.endf.model_adapter``, so each gains an import of it. That is a worse
#: layering number and a better library: the alternative is two decoders for the
#: same file, which is how the ZA truncation and the LTT=3 losses came to differ
#: between them in the first place. These entries go to **zero** when the flat
#: classes are removed in 1.0 — they are the deprecation's cost, held visibly
#: rather than hidden. No *new* module may join the list.
#:
#: **D4 raises ``cross_section.py`` from 2 to 3, for the same reason and
#: knowingly.** ``from_ace`` now reads ACE's LQR block through
#: ``kika.ace.model_adapter``'s ``qValuesByMT`` instead of aligning MTR and LQR
#: a second time locally. The alignment is positional, undeclared in the file,
#: and has one subtlety (MT 2 is absent because elastic Q is zero *by
#: definition*) — precisely the kind of convention that drifts when it is
#: written twice. It already had: the library carried "ACE stores no reaction Q
#: values" in a docstring, a roadmap and an error message while the values sat
#: parsed in ``ace.q_values``. One decoder, one convention, one worse number.
#:
#: **M2 raises ``nuclide_info.py`` from 1 to 2, and it is the façade's own
#: definition rather than a new coupling.** ``NuclideInfo.to_endf`` has to
#: *construct* an ``MF1MT451``, so it imports the class it builds; the existing
#: entry is ``from_endf``'s ``decodeMF1MT451``. A canonical→format method that
#: did not name a format type would not be writing ENDF. Deferred to call time
#: like the rest, and it goes to zero with the flat classes at 1.0.
#:
#: **The ``kika/cov`` entries are seeded, not earned**, and they are not all the
#: same kind of thing — which is the point of listing them separately rather
#: than quoting one total for the package. There were three; phase 4's P4 cut
#: one, and the note on it is kept below because *why* it was cheap is the
#: transferable part.
#:
#: ``legendre_covariance.py`` (4) is the one the roadmap protects. Three of the
#: four are the MF34 *encoder* — ``to_mf34`` builds an ``MF34MT`` from
#: ``_make_lb5_record``/``_make_lb6_record`` (lines 315-316) and
#: ``write_mf34_to_file`` splices it (line 416). That is format code doing
#: format work, kept deliberately. The fourth (line 199) is ``read_endf`` in the
#: reading direction. All four are deferred to call time.
#:
#: ``parse_covmat.py`` (1) — **gone, phase 4 P4, and it was the genuine leak**:
#: the only import in ``kika/cov`` that ran at *import* time rather than at call
#: time. It took ENDF's record grammar — ``parse_number``, ``parse_line``,
#: ``parse_endf_id``, ``format_endf_number``, ``format_endf_data_line``,
#: ``ENDF_FORMAT_FLOAT``, ``ENDF_FORMAT_INT`` — to read and write
#: COVERX/COVFIL/BOXER, which are not ENDF files. They share a fixed-width
#: record convention, so the dependency was real rather than lazy, and deferring
#: it would have hidden the problem instead of fixing it. The grammar moved down
#: to :mod:`kika._records` — the same species as ``interpolate_1d_endf``, which
#: phase 2 moved to ``kika.processing`` precisely because the interpolation laws
#: are not ENDF's property either. ``kika.endf.utils`` re-exports it, so the
#: forty-odd importers inside ``kika/endf`` and the public-surface pin on
#: ``parse_endf_id`` are untouched. The prediction written here — "this entry
#: goes to zero and no other line here changes" — held exactly.
#:
#: ``multigroup/legacy_mg_plotting.py`` (1) — **cut in phase 4's P4**, and it was
#: the weakest of the three: a ``try``/``except`` around a **private** helper,
#: ``kika.endf.classes.mf4.plotting._plot_uncertainty_bands``. What made it cheap
#: was not the ``except`` but the helper itself — it never needed ENDF at all.
#: Its ``mf34_covmat`` argument *is* a ``LegendreCovariance``, and its body is one
#: call to ``get_uncertainties_for_legendre_coefficient``, a scan over that
#: container's own row labels, and an ``ax.fill_between``. Calculation-layer
#: drawing that happened to live in the format package. It moved to
#: ``kika/plotting/covariance.py`` as ``plot_legendre_uncertainty_bands``; the
#: old private name survives there as an alias for the two in-package callers.
#: Reaching through a leading underscore into another package is the part worth
#: recording, and it is now nobody's import.
RUNTIME_ALLOWLIST: dict[str, int] = {
    "kika/cov/legendre_covariance.py": 4,
    "kika/nuclear_data/angular_distribution.py": 13,
    "kika/nuclear_data/cross_section.py": 3,
    "kika/nuclear_data/nuclide_info.py": 2,
    "kika/nuclear_data/resonance_parameters.py": 2,
    "kika/processing/derived_covariance.py": 1,
    "kika/processing/njoy_pendf_cache.py": 1,
    "kika/processing/njoy_reconstruct.py": 1,
}

#: ``if TYPE_CHECKING:`` imports of a format package, per module.
#:
#: ``legendre_covariance.py``'s single entry is the ``MF34MT`` annotation on
#: ``to_mf34``'s return. It is the vocabulary half of the same encoder the
#: runtime list explains: a method that returns an ``MF34MT`` has to name the
#: type it returns.
TYPE_CHECKING_ALLOWLIST: dict[str, int] = {
    "kika/cov/legendre_covariance.py": 1,
    "kika/nuclear_data/angular_distribution.py": 2,
    "kika/nuclear_data/cross_section.py": 3,
    "kika/nuclear_data/nuclide_info.py": 2,
    "kika/nuclear_data/resonance_parameters.py": 1,
}


def _resolve(node: ast.ImportFrom | ast.Import, module_path: Path) -> list[str]:
    """Absolute dotted names a single import node brings in.

    Relative imports are resolved against the importing module's own package,
    so ``from ..endf.utils import x`` inside ``kika/processing/`` is recognised
    as ``kika.endf.utils`` — a relative spelling must not be a way around this
    test.
    """
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]

    if node.level == 0:
        return [node.module] if node.module else []

    parts = module_path.relative_to(REPO_ROOT).with_suffix("").parts
    package = list(parts[:-1])  # drop the module name itself
    if node.level > 1:
        package = package[: -(node.level - 1)]
    base = ".".join(package)
    return [f"{base}.{node.module}" if node.module else base]


def _is_forbidden(dotted: str) -> bool:
    return any(
        dotted == root or dotted.startswith(root + ".") for root in FORBIDDEN_ROOTS
    )


def scan_module(path: Path) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Return ``(runtime, type_checking)`` forbidden imports as (line, name)."""
    tree = ast.parse(path.read_text(), filename=str(path))

    type_checking_nodes: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        guard = (
            (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
            or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")
        )
        if guard:
            for child in ast.walk(node):
                type_checking_nodes.add(id(child))

    runtime: list[tuple[int, str]] = []
    typing_only: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for dotted in _resolve(node, path):
            if not _is_forbidden(dotted):
                continue
            target = typing_only if id(node) in type_checking_nodes else runtime
            target.append((node.lineno, dotted))
    return runtime, typing_only


def guarded_modules() -> list[Path]:
    """Every non-test module of the guarded packages."""
    found: list[Path] = []
    for package in GUARDED_PACKAGES:
        for path in sorted((REPO_ROOT / package).rglob("*.py")):
            if "tests" in path.parts:
                continue
            found.append(path)
    return found


def _counts(kind: int) -> dict[str, int]:
    result: dict[str, int] = {}
    for path in guarded_modules():
        hits = scan_module(path)[kind]
        if hits:
            result[str(path.relative_to(REPO_ROOT))] = len(hits)
    return result


# ---------------------------------------------------------------------------
# The ratchet
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kind,allowlist,label",
    [
        (0, RUNTIME_ALLOWLIST, "runtime"),
        (1, TYPE_CHECKING_ALLOWLIST, "TYPE_CHECKING"),
    ],
    ids=["runtime", "type-checking"],
)
def test_no_new_format_imports_in_the_calculation_layer(kind, allowlist, label):
    """No module may gain a format import, and none may appear in a clean one."""
    actual = _counts(kind)

    unlisted = sorted(set(actual) - set(allowlist))
    assert not unlisted, (
        f"new {label} import of kika.endf/kika.ace in a module that had none:\n  "
        + "\n  ".join(
            f"{m}: " + ", ".join(
                f"line {ln} -> {name}" for ln, name in scan_module(REPO_ROOT / m)[kind]
            )
            for m in unlisted
        )
    )

    grown = {m: (allowlist[m], n) for m, n in actual.items() if n > allowlist[m]}
    assert not grown, (
        f"{label} format imports increased:\n  "
        + "\n  ".join(f"{m}: {was} allowed, {now} found" for m, (was, now) in grown.items())
    )


@pytest.mark.parametrize(
    "kind,allowlist,label",
    [
        (0, RUNTIME_ALLOWLIST, "runtime"),
        (1, TYPE_CHECKING_ALLOWLIST, "TYPE_CHECKING"),
    ],
    ids=["runtime", "type-checking"],
)
def test_the_allowlist_has_no_stale_entries(kind, allowlist, label):
    """A violation that has been removed must be struck from the list.

    Without this the allowlist silently becomes a wish list, and the ratchet
    stops measuring anything.
    """
    actual = _counts(kind)
    stale = {
        m: (allowlist[m], actual.get(m, 0))
        for m in allowlist
        if actual.get(m, 0) < allowlist[m]
    }
    assert not stale, (
        f"{label} allowlist is out of date — lower or delete these entries:\n  "
        + "\n  ".join(f"{m}: {was} allowed, {now} found" for m, (was, now) in stale.items())
    )


# ---------------------------------------------------------------------------
# The ratchet itself
# ---------------------------------------------------------------------------

def test_the_ratchet_catches_a_violation(tmp_path, monkeypatch):
    """A ratchet nobody has seen fail is not a ratchet.

    Point the scanner at a throwaway tree containing one clean module and one
    that imports ``kika.endf``, and require it to find exactly the second.
    """
    fake_repo = tmp_path
    package = fake_repo / "kika" / "processing"
    package.mkdir(parents=True)
    (package / "clean.py").write_text("import numpy as np\n")
    (package / "dirty.py").write_text(
        "from kika.endf.utils import interpolate_1d_endf\n"
    )
    # A relative spelling must not slip past.
    (package / "sneaky.py").write_text("from ..endf.read_endf import read_endf\n")

    monkeypatch.setattr(f"{__name__}.REPO_ROOT", fake_repo)
    monkeypatch.setattr(f"{__name__}.GUARDED_PACKAGES", ("kika/processing",))

    found = _counts(0)
    assert found == {
        "kika/processing/dirty.py": 1,
        "kika/processing/sneaky.py": 1,
    }


# ---------------------------------------------------------------------------
# The half the count cannot see: *when* those imports run
# ---------------------------------------------------------------------------

#: Packages whose ``kika.endf`` imports must stay inside a function body. The
#: dependency is real in every case and only its *timing* is negotiable.
#:
#: ``kika/processing``'s three genuinely need ``read_endf`` to parse a PENDF
#: tape. **``kika/cov`` joined in phase 4's P4**, and it could not have before:
#: ``parse_covmat.py`` imported the record grammar at module scope, so this
#: check would have failed on the day it was written. Deferring that import
#: would have passed the check and left the leak; moving the grammar to
#: ``kika._records`` removed it, and the check is what stops it coming back as a
#: deferred import nobody counts.
_MUST_DEFER = ("kika/processing", "kika/cov")

#: The canonical model. Not a *format* package, so the ratchet above says
#: nothing about it — and it must not, because ``kika/processing`` computing on
#: the model is the whole point of phase 4. What is forbidden is only the
#: *timing*, for the same reason the ENDF imports are: see
#: ``test_processing_does_not_wake_the_model_at_import_time``.
_MODEL_ROOT = "kika.nuclear_data.model"


def _is_model(dotted: str) -> bool:
    return dotted == _MODEL_ROOT or dotted.startswith(_MODEL_ROOT + ".")


def _moduleScopeImports(path: Path, predicate=_is_forbidden) -> list[tuple[int, str]]:
    """Imports matching *predicate* that execute when the module is imported.

    "Module scope" means not lexically inside a ``def``/``async def``. A class
    body counts as module scope, because it runs at import time — that is the
    hole a nested-function-only check would leave.

    **An ``if TYPE_CHECKING:`` block does not count**, because it is False at
    run time and imports nothing. That exclusion was missing until 2026-08-10
    and had never fired: this helper only ever ran over ``kika/processing``,
    whose modules have no ``TYPE_CHECKING`` format import at all — the
    ``TYPE_CHECKING`` allowlist is entirely ``kika/nuclear_data`` and
    ``kika/cov``. So the defect was latent, and it would have surfaced as a
    false positive on ``kika/cov/legendre_covariance.py:13`` the moment anyone
    extended this check past ``kika/processing``. Found by trying to.
    """
    tree = ast.parse(path.read_text(), filename=str(path))

    deferred: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                deferred.add(id(child))
        elif isinstance(node, ast.If):
            test = node.test
            if (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
                isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
            ):
                for child in ast.walk(node):
                    deferred.add(id(child))

    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if id(node) in deferred:
            continue
        for dotted in _resolve(node, path):
            if predicate(dotted):
                hits.append((node.lineno, dotted))
    return hits


def test_processing_pulls_no_format_package_in_at_import_time():
    """The invariant the ratchet counts around rather than through.

    ``kika/processing/__init__.py`` used to import ``kika.endf`` at module
    scope, via ``njoy_reconstruct`` and ``derived_covariance``. The consequence
    was stronger than a bad number: **nothing in ``kika.endf`` could import from
    ``kika.processing`` at module scope at all**, because the cycle closed and
    ``import kika`` died with "partially initialized module". ``interpolate_1d``
    could not be moved down, and neither could anything else, ever, until it was
    fixed. Phase 2 fixed it by deferring three imports to call time.

    **Nothing checked that it stays fixed.** The ratchet counts import
    statements, not when they run, so re-hoisting one of the three to module
    scope leaves every existing test green and re-closes the cycle. The GNDS
    roadmap records this as "worth a dedicated test at some point" and notes
    that every phase 4-9 increment touching ``kika/processing`` has to keep the
    imports deferred — which makes this the net that belongs *before* phase 4,
    not after it.

    A subprocess check was the obvious shape and does not work: ``import
    kika.processing`` executes ``kika/__init__.py`` first, and that imports
    ``kika.endf.read_endf`` eagerly on line 9, so ``kika.endf`` is in
    ``sys.modules`` before ``kika.processing`` is reached whatever the latter
    does. The property is statically decidable, so it is decided statically.
    """
    offenders: dict[str, list[tuple[int, str]]] = {}
    for package in _MUST_DEFER:
        for path in sorted((REPO_ROOT / package).rglob("*.py")):
            if "tests" in path.parts:
                continue
            hits = _moduleScopeImports(path)
            if hits:
                offenders[str(path.relative_to(REPO_ROOT))] = hits

    assert not offenders, (
        "these modules import a format package at module scope, "
        "which re-closes the import cycle phase 2 opened:\n"
        + "\n".join(
            f"  {name}:{line} imports {dotted}"
            for name, hits in offenders.items()
            for line, dotted in hits
        )
        + "\nMove the import inside the function that needs it."
    )


def test_that_check_catches_a_hoisted_import(tmp_path, monkeypatch):
    """It bites. A module-scope import must be reported and a deferred one must not."""
    package = tmp_path / "kika" / "processing"
    package.mkdir(parents=True)

    deferred = package / "deferred.py"
    deferred.write_text(
        "def go():\n    from kika.endf.read_endf import read_endf\n    return read_endf\n"
    )
    assert _moduleScopeImports(deferred) == []

    hoisted = package / "hoisted.py"
    hoisted.write_text("from kika.endf.read_endf import read_endf\n")
    assert _moduleScopeImports(hoisted) == [(1, "kika.endf.read_endf")]

    # A class body is not a function body, and runs at import time.
    inClass = package / "in_class.py"
    inClass.write_text("class A:\n    from kika.endf import read_endf\n")
    assert _moduleScopeImports(inClass), "a class-body import is still import-time"

    # An ``if TYPE_CHECKING:`` block is False at run time and imports nothing,
    # so it is not an import-time import. Without this arm the check reports a
    # false positive on kika/cov/legendre_covariance.py:13 — which is how the
    # missing exclusion was found. Both spellings of the guard, because
    # ``scan_module`` accepts both and a helper that disagreed with it would be
    # worse than either.
    typingOnly = package / "typing_only.py"
    typingOnly.write_text(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from kika.endf.classes.mf34.mf34 import MF34MT\n"
    )
    assert _moduleScopeImports(typingOnly) == [], (
        "an `if TYPE_CHECKING:` import does not run at import time"
    )

    qualifiedGuard = package / "typing_only_qualified.py"
    qualifiedGuard.write_text(
        "import typing\n"
        "if typing.TYPE_CHECKING:\n"
        "    from kika.endf.classes.mf34.mf34 import MF34MT\n"
    )
    assert _moduleScopeImports(qualifiedGuard) == [], (
        "`if typing.TYPE_CHECKING:` is the same guard, spelled differently"
    )

    # ``__name__``, not a hardcoded dotted path: pytest may import this module
    # under a name that is not ``kika.tests.test_layering``, and patching the
    # wrong module object silently patches nothing — which looks exactly like a
    # check that does not bite.
    monkeypatch.setattr(f"{__name__}.REPO_ROOT", tmp_path)
    with pytest.raises(AssertionError, match="module scope"):
        test_processing_pulls_no_format_package_in_at_import_time()


# ---------------------------------------------------------------------------
# The same property, for the model
# ---------------------------------------------------------------------------

def test_processing_does_not_wake_the_model_at_import_time():
    """Phase 4's companion to the check above, and it guards a bigger number.

    Phase 4 moves the calculations onto ``kika.nuclear_data.model``, so
    ``kika/processing`` importing it is correct and expected — *at call time*.
    At module scope it is a regression nothing else would notice, because
    ``kika/__init__.py`` does ``from . import processing`` and would then build
    ~3 400 lines of model on every ``import kika``: for the cluster pipeline,
    the desktop app and every notebook, none of which asked to reconstruct
    anything. ``test_deprecation_surface.py::test_the_model_is_still_dormant_after_the_facade``
    would catch it, but only after the fact and from the far end of the
    library; this says which file and which line.

    Not folded into ``FORBIDDEN_ROOTS``: the model is not a format package and
    a *count* of these imports is the wrong instrument. Deferred imports of it
    are supposed to grow.

    **``kika/cov`` is covered too, since P4.** The number there is if anything
    larger: ``import kika`` loads ``kika.cov`` — measured, not assumed — so a
    module-scope model import in the multigroup collapse would wake the model
    for the cluster pipeline and kika-app alike, exactly as it would from
    ``kika/processing``.
    """
    offenders: dict[str, list[tuple[int, str]]] = {}
    for package in _MUST_DEFER:
        for path in sorted((REPO_ROOT / package).rglob("*.py")):
            if "tests" in path.parts:
                continue
            hits = _moduleScopeImports(path, predicate=_is_model)
            if hits:
                offenders[str(path.relative_to(REPO_ROOT))] = hits

    assert not offenders, (
        "these modules import the GNDS model at module scope, "
        "so `import kika` now builds the whole model:\n"
        + "\n".join(
            f"  {name}:{line} imports {dotted}"
            for name, hits in offenders.items()
            for line, dotted in hits
        )
        + "\nMove the import inside the function that needs it."
    )


def test_that_model_check_catches_a_hoisted_import(tmp_path, monkeypatch):
    """It bites, and it does not fire on the deferred imports P1a actually wrote."""
    package = tmp_path / "kika" / "processing"
    package.mkdir(parents=True)

    deferred = package / "deferred.py"
    deferred.write_text(
        "def go():\n"
        "    from kika.nuclear_data.model.functions import XYs1d\n"
        "    return XYs1d\n"
    )
    assert _moduleScopeImports(deferred, predicate=_is_model) == []

    hoisted = package / "hoisted.py"
    hoisted.write_text("from kika.nuclear_data.model.resonances import RMatrix\n")
    assert _moduleScopeImports(hoisted, predicate=_is_model) == [
        (1, "kika.nuclear_data.model.resonances")
    ]

    # The flat package is a different thing and is not what this checks: a
    # module-scope `from kika.nuclear_data.cross_section import ...` is a
    # deprecation question, not an import-cost one.
    flat = package / "flat.py"
    flat.write_text("from kika.nuclear_data.cross_section import CrossSection\n")
    assert _moduleScopeImports(flat, predicate=_is_model) == []

    monkeypatch.setattr(f"{__name__}.REPO_ROOT", tmp_path)
    with pytest.raises(AssertionError, match="module scope"):
        test_processing_does_not_wake_the_model_at_import_time()
