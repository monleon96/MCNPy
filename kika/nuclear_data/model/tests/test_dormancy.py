"""The model must stay unreachable from a plain ``import kika``.

**Why this is the load-bearing test of phase 3.** Nothing outside the library
imports ``kika.nuclear_data`` by name — 0 scripts, 0 kika-app modules, 2
notebooks. But ``kika/__init__.py:13`` does ``from . import nuclear_data`` at
module scope, so *every* consumer imports the package transitively on
``import kika``. The blast radius is small for **names** and total for **import
health**: a syntax error, a circular import, a slow table build or a
module-level ``warnings.warn`` in the new model would break the cluster
pipeline, the desktop app and every notebook simultaneously.

The whole safety argument for phases 3a-3c is "the new model is built beside the
old one and nothing imports it". That is an argument about a fact, and this is
the test of the fact. It has to run in a **subprocess**: by the time this test
module is collected, the model is already in ``sys.modules`` — pytest imported
it to get here.

If this ever fails, do not relax it. Find what started importing the model and
stop it, until phase 3d deliberately wires it in.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap


def _in_subprocess(body: str) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, (
        f"subprocess failed:\n{completed.stdout}\n{completed.stderr}"
    )
    return completed.stdout.strip()


def test_import_kika_does_not_import_the_model():
    output = _in_subprocess("""
        import sys
        import kika
        leaked = sorted(m for m in sys.modules if m.startswith("kika.nuclear_data.model"))
        assert not leaked, (
            "`import kika` pulled in the GNDS model: " + ", ".join(leaked) +
            ". The model is meant to be dormant until phase 3d; something now "
            "imports it, and every consumer of the library pays for it."
        )
        print("dormant")
    """)
    assert output == "dormant"


def test_import_kika_nuclear_data_does_not_import_the_model():
    """Even reaching for the flat classes directly must not wake the model."""
    output = _in_subprocess("""
        import sys
        import kika.nuclear_data
        leaked = sorted(m for m in sys.modules if m.startswith("kika.nuclear_data.model"))
        assert not leaked, "kika.nuclear_data imported the model: " + ", ".join(leaked)
        print("dormant")
    """)
    assert output == "dormant"


def test_the_model_is_importable_on_purpose():
    """The other half: dormant is not the same as broken.

    Without this, deleting the package would also make the two tests above pass.
    """
    output = _in_subprocess("""
        import kika.nuclear_data.model as model
        assert model.XYs1d is not None
        assert model.parse_unit("MeV/c**2").factor == 1e6
        print("awake")
    """)
    assert output == "awake"


def test_the_model_adds_no_format_imports_of_its_own():
    """The layering rule at runtime, as a *delta*.

    ``kika/tests/test_layering.py`` scans the source; this catches the other
    route — a format package pulled in *indirectly*, through something the model
    imports, which a source scan of the model alone would miss.

    It has to be measured as a difference. ``kika/__init__.py`` eagerly imports
    ``read_endf`` and ``read_ace``, so by the time any ``kika.*`` submodule can
    be imported at all, both format packages are already loaded. Asking "is
    ``kika.endf`` in ``sys.modules``" after importing the model therefore always
    answers yes and measures nothing. The question worth asking is what the
    model *added*.
    """
    output = _in_subprocess("""
        import sys
        import kika  # the baseline every consumer already pays for
        before = {m for m in sys.modules if m.startswith(("kika.endf", "kika.ace"))}

        import kika.nuclear_data.model  # noqa: F401
        after = {m for m in sys.modules if m.startswith(("kika.endf", "kika.ace"))}

        added = sorted(after - before)
        assert not added, "the model pulled in format modules of its own: " + ", ".join(added)
        print("clean")
    """)
    assert output == "clean"
