"""Do the deployed scripts still fit the kika wheel staged on the cluster?

`scripts/` and the library are deployed **independently** — the scripts by the
`deploy-cluster` skill, the library as a wheel installed by hand into the
cluster venv — so they drift apart silently and the drift only surfaces when a
12-day sbatch job dies at 3 a.m. on an `ImportError`.

`test_deploy_smoke.py` proves the scripts *import*, but it imports them against
the working tree, so it says nothing about the kika actually installed on the
cluster. Phase 1 of the GNDS roadmap made the gap concrete: four deployed
scripts changed, and only a manual grep established that none of them had
picked up a new kika API, so all four still ran against the older wheel. That
check should not be manual.

**How it works.** ``ast``-walk every module in ``scripts/`` for the ``kika.*``
names it actually references, then resolve each one *in the staged venv's own
interpreter* — a subprocess, not this process, because the whole point is to
ask a different installation.

**Where the venv is.** ``KIKA_STAGED_VENV``, default
``/work/monleon-de-la-jan/myenv``. That path is on the cluster and is normally
**not mounted** on the workstation, so this skips by default and fails under
``--deep`` — the same contract every tape-backed test in this suite uses. It is
therefore a check you run deliberately, from somewhere that can see the venv,
before a deploy.

Deliberately *not* "pin the wheel version in the sbatch script": that hides the
drift instead of reporting it, and the cluster runs an older wheel on purpose
while a roadmap phase is in progress.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
DEFAULT_STAGED_VENV = "/work/monleon-de-la-jan/myenv"

#: Names a script may reference that the wheel is not expected to provide.
#: ``scripts`` is the sibling package, resolved by the deploy layout, not by kika.
IGNORED_ROOTS = ("scripts",)


def _staged_python() -> Path | None:
    """The staged venv's interpreter, or ``None`` if it is not reachable."""
    root = Path(os.environ.get("KIKA_STAGED_VENV", DEFAULT_STAGED_VENV))
    for candidate in (root / "bin" / "python", root / "bin" / "python3"):
        if candidate.is_file():
            return candidate
    return None


def _referenced_kika_names() -> dict[str, set[str]]:
    """``{dotted module: {attribute, ...}}`` for every ``kika`` import in scripts/.

    Both spellings are collected: ``from kika.x import y`` yields
    ``{"kika.x": {"y"}}``, and a bare ``import kika.x`` yields ``{"kika.x":
    set()}`` — the module must exist even when no attribute is named.
    """
    found: dict[str, set[str]] = {}
    for path in sorted(SCRIPTS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(errors="replace"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.split(".")[0] != "kika":
                    continue
                found.setdefault(module, set()).update(a.name for a in node.names)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "kika":
                        found.setdefault(alias.name, set())
    return {m: names for m, names in found.items() if m.split(".")[0] not in IGNORED_ROOTS}


REFERENCED = _referenced_kika_names()

_PROBE = r"""
import importlib, json, sys
missing = []
for module, names in json.load(sys.stdin).items():
    try:
        mod = importlib.import_module(module)
    except Exception as exc:
        missing.append(f"{module}: {type(exc).__name__}: {exc}")
        continue
    for name in names:
        if not hasattr(mod, name):
            missing.append(f"{module}.{name}: missing")
try:
    import kika
    version = getattr(kika, "__version__", "?")
except Exception as exc:
    version = f"<unimportable: {exc}>"
print(json.dumps({"missing": sorted(missing), "version": version, "python": sys.version.split()[0]}))
"""


def test_the_scan_found_something():
    """A glob or an ast walk that matched nothing would make this file a no-op."""
    assert len(REFERENCED) >= 5, f"only found kika imports in {sorted(REFERENCED)}"


@pytest.fixture(scope="module")
def staged_probe(request: pytest.FixtureRequest) -> dict:
    """Resolve every referenced name inside the staged venv's interpreter."""
    python = _staged_python()
    if python is None:
        root = os.environ.get("KIKA_STAGED_VENV", DEFAULT_STAGED_VENV)
        detail = (
            f"no interpreter under {root}; set KIKA_STAGED_VENV, or run this "
            f"from a machine that mounts the cluster work tree"
        )
        # Same contract as the tape fixtures: skip normally, fail under --deep.
        if request.config.getoption("--deep"):
            pytest.fail(f"--deep: staged venv not reachable: {detail}", pytrace=False)
        pytest.skip(f"staged venv not reachable: {detail}")

    payload = json.dumps({m: sorted(n) for m, n in REFERENCED.items()})
    completed = subprocess.run(
        [str(python), "-c", _PROBE],
        input=payload,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, (
        f"probe failed inside {python}:\n{completed.stderr.strip()}"
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_every_name_the_scripts_use_exists_in_the_staged_wheel(staged_probe):
    assert not staged_probe["missing"], (
        f"the deployed scripts reference kika names the staged wheel "
        f"(kika {staged_probe['version']}, python {staged_probe['python']}) "
        f"does not provide:\n  " + "\n  ".join(staged_probe["missing"]) + "\n\n"
        f"Either stage a newer wheel or hold the scripts back. Deploying the "
        f"scripts alone would break the next run."
    )


def test_the_staged_wheel_is_a_real_kika(staged_probe):
    """A venv with no kika at all fails loudly rather than reporting zero gaps."""
    assert not staged_probe["version"].startswith("<unimportable"), staged_probe["version"]
