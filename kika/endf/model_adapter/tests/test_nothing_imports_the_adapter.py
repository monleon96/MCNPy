"""The adapters must stay off every critical path but their own.

``kika.endf.model_adapter`` and ``kika.ace.model_adapter`` each import the whole
GNDS model. If anything in ``kika.endf``/``kika.ace`` imports *them* at module
scope, then ``read_endf`` and ``read_ace`` — which the cluster pipeline, the
desktop app and every notebook call — start building the model on every parse,
and ``import kika`` gets slower for everyone to no purpose.

There is a second, sharper reason. Both packages are deliberately **absent**
from kika-app's PyInstaller ``hiddenimports``. A lazy import of either from an
app-reachable code path would therefore raise ``ModuleNotFoundError`` in the
frozen binary and nowhere else — dev, tests and CI all green. That is the exact
failure nine other modules were found in during P2, and the spec's own comment
records it having happened before.

This file lives under the ENDF adapter's tests and covers both, rather than
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


def _importers(adapter: str, root: Path | None = None) -> list[str]:
    """Every file under kika/ that imports ``adapter``, excluding adapter packages."""
    root = root if root is not None else REPO_ROOT
    found: list[str] = []
    for path in sorted((root / "kika").rglob("*.py")):
        relative = path.relative_to(root)
        if "model_adapter" in relative.parts:
            continue
        tree = ast.parse(path.read_text(errors="replace"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(adapter):
                found.append(f"{relative}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(adapter):
                        found.append(f"{relative}:{node.lineno}")
    return found


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_nothing_in_the_library_imports_the_adapter(adapter):
    importers = _importers(adapter)
    assert not importers, (
        f"{adapter} is imported outside its own package:\n  "
        + "\n  ".join(importers)
        + "\n\nIt pulls in the whole GNDS model. Keep it reachable only from its "
          "own tests and from code that deliberately asks for the model."
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

    assert _importers(adapter, root=tmp_path) == ["kika/endf/guilty.py:1"]
