"""The layering ratchet: format code must not leak into the calculation layer.

`kika/tests/` is the one place in this repo that is not a package's own test
directory. What lives here are invariants of the repository as a whole, which
belong to no single package. This is the first of them.

**The invariant.** ``kika.processing`` and ``kika.nuclear_data`` are the
calculation layer. ``kika.endf`` and ``kika.ace`` are format layers. The arrow
points one way: formats may import the calculations, the calculations may not
import the formats.

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
GUARDED_PACKAGES = ("kika/processing", "kika/nuclear_data")

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
RUNTIME_ALLOWLIST: dict[str, int] = {
    "kika/nuclear_data/angular_distribution.py": 13,
    "kika/nuclear_data/cross_section.py": 2,
    "kika/nuclear_data/nuclide_info.py": 1,
    "kika/nuclear_data/resonance_parameters.py": 2,
    "kika/processing/derived_covariance.py": 1,
    "kika/processing/njoy_pendf_cache.py": 1,
    "kika/processing/njoy_reconstruct.py": 1,
}

#: ``if TYPE_CHECKING:`` imports of a format package, per module.
TYPE_CHECKING_ALLOWLIST: dict[str, int] = {
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
