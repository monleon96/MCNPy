"""Every deployed script must at least import.

The thesis pipeline does not run here. It runs on the ASNR cluster, launched
by sbatch from a copy of ``scripts/`` on the shared mount. When an import in
one of those files breaks — a function renamed in ``kika``, a module moved, a
dependency dropped — nothing says so until a job has queued, started, and died
minutes later, with the traceback in a slurm log nobody is watching.

This test is the cheapest possible guard against that: import every module in
``scripts/`` and require it to succeed. It proves nothing about correctness.
It only moves the discovery of a broken import from a dead cluster job to a
two-second local test run.

Scripts are written to be executed by editing their ALL_CAPS config constants
and running the file, so all real work sits behind ``if __name__ ==
"__main__"``. Importing one is therefore side-effect free apart from
module-level constants — which is itself worth enforcing, and this test does
enforce it: a script that starts doing work at import time will hang or fail
here.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPTS_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def script_modules() -> list[str]:
    """Every importable module in ``scripts/``, as dotted names."""
    return sorted(
        f"scripts.{path.stem}"
        for path in SCRIPTS_DIR.glob("*.py")
        if path.stem != "__init__"
    )


MODULES = script_modules()


def test_there_are_scripts_to_check():
    """A glob that silently matches nothing would make this file a no-op."""
    assert len(MODULES) >= 20, f"only found {len(MODULES)} scripts: {MODULES}"


@pytest.mark.parametrize("module_name", MODULES)
def test_script_imports(module_name):
    """Importing must not raise, and must not need cluster data to do it."""
    importlib.import_module(module_name)


_AUDIT_PROBE = '''
import sys, importlib

FORBIDDEN = ("/share_snc", "/soft_snc", "/SCRATCH")
opened = []

def hook(event, args):
    if event != "open":
        return
    target = args[0]
    if isinstance(target, (str, bytes)):
        text = target.decode() if isinstance(target, bytes) else target
        if text.startswith(FORBIDDEN):
            opened.append(text)

sys.path.insert(0, {repo!r})
sys.addaudithook(hook)

for name in {modules!r}:
    importlib.import_module(name)

print("\\n".join(sorted(set(opened))))
'''


def test_no_script_opens_the_shared_mount_at_import_time():
    """Config constants may name ``/share_snc`` paths; nothing may open one.

    A script that reads the mount while being imported cannot be imported on a
    machine without it, which would put it beyond the reach of this test and of
    CI. Checked in a subprocess with an audit hook, because by the time this
    process gets here every module is already imported and cached.
    """
    probe = _AUDIT_PROBE.format(repo=str(REPO_ROOT), modules=MODULES)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, timeout=300, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"importing the scripts failed in a subprocess:\n{result.stderr[-2000:]}"
    )
    touched = [line for line in result.stdout.splitlines() if line.strip()]
    assert not touched, (
        "these paths on the shared mount were opened at import time:\n  "
        + "\n  ".join(touched)
    )
