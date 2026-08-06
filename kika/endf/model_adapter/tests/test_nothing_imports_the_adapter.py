"""The adapter must stay off every critical path but its own.

``kika.endf.model_adapter`` imports the whole GNDS model. If anything in
``kika.endf`` imports *it* at module scope, then ``read_endf`` — which the
cluster pipeline, the desktop app and every notebook call — starts building the
model on every parse, and ``import kika`` gets slower for everyone to no
purpose.

There is a second, sharper reason. ``kika.endf.model_adapter`` is deliberately
**absent** from kika-app's PyInstaller ``hiddenimports``. A lazy import of it
from an app-reachable code path would therefore raise ``ModuleNotFoundError`` in
the frozen binary and nowhere else — dev, tests and CI all green. That is the
exact failure nine other modules were found in during P2, and the spec's own
comment records it having happened before.
"""
from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
ADAPTER = "kika.endf.model_adapter"


def _importers() -> list[str]:
    """Every file under kika/ that imports the adapter, excluding its own package."""
    found: list[str] = []
    for path in sorted((REPO_ROOT / "kika").rglob("*.py")):
        relative = path.relative_to(REPO_ROOT)
        if "model_adapter" in relative.parts:
            continue
        tree = ast.parse(path.read_text(errors="replace"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(ADAPTER):
                found.append(f"{relative}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(ADAPTER):
                        found.append(f"{relative}:{node.lineno}")
    return found


def test_nothing_in_the_library_imports_the_adapter():
    importers = _importers()
    assert not importers, (
        "kika.endf.model_adapter is imported outside its own package:\n  "
        + "\n  ".join(importers)
        + "\n\nIt pulls in the whole GNDS model. Keep it reachable only from its "
          "own tests and from code that deliberately asks for the model."
    )


def test_reading_an_endf_file_does_not_build_the_model(micro_tape):
    """The runtime half: parsing a tape must not wake the model."""
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(f"""
            import sys
            from kika.endf.read_endf import read_endf
            read_endf({str(micro_tape)!r})
            leaked = sorted(
                m for m in sys.modules
                if m.startswith("kika.nuclear_data.model") or m.startswith("{ADAPTER}")
            )
            assert not leaked, "read_endf built the model: " + ", ".join(leaked)
            print("clean")
        """)],
        capture_output=True, text=True, timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == "clean"


def test_the_check_would_notice_an_importer(tmp_path, monkeypatch):
    """A ratchet nobody has seen fail is not a ratchet."""
    fake = tmp_path / "kika" / "endf"
    fake.mkdir(parents=True)
    (fake / "innocent.py").write_text("import numpy\n")
    (fake / "guilty.py").write_text(f"from {ADAPTER}.decode import decodeMF3MT\n")

    monkeypatch.setattr(f"{__name__}.REPO_ROOT", tmp_path)
    assert _importers() == ["kika/endf/guilty.py:1"]
