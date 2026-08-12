"""Every module under ``kika/cov/`` must import.

Written because three of them did not, and had not for eight months.

``kika/cov/plotting.py``, ``heatmap.py`` and ``mf34cov_heatmap.py`` were gutted
to shims by commit ``934ee05`` (December 2025), which moved the real plotting
into ``kika/plotting/``. The shims were pointed at ``kika.cov.legacy.*`` — a
package that was never committed, then or since. The multigroup sibling of that
same commit *was* created (``kika/cov/multigroup/legacy_mg_plotting.py``, 1550
lines), which is why the breakage looked like a deliberate pattern rather than
an omission.

Nothing caught it because nothing imported them: the modules had no test, and
their only in-library reader was one deferred import inside a method body
(``LegendreCovariance.plot_uncertainties``), so ``ModuleNotFoundError`` was
raised at *call* time on a public plotting method and at import time only for a
user who reached for the shim by name — which three workspace notebooks did.

This test is the cheap general form of that lesson. A deferred import cannot be
proven correct by the import graph, but a module that cannot be imported at all
can be, and that is the failure mode which actually occurred.
"""
from __future__ import annotations

import importlib
import pkgutil
import warnings

import pytest

import kika.cov


def _covModules() -> list[str]:
    """Every importable module under ``kika/cov/``, tests excluded."""
    found = []
    for info in pkgutil.walk_packages(kika.cov.__path__, prefix="kika.cov."):
        if ".tests" in info.name or info.name.endswith(".tests"):
            continue
        found.append(info.name)
    return sorted(found)


@pytest.mark.parametrize("moduleName", _covModules())
def test_the_module_imports(moduleName: str) -> None:
    """A shim to a package that does not exist is not a shim, it is a hole."""
    with warnings.catch_warnings():
        # Several of these modules warn DeprecationWarning at import on purpose.
        warnings.simplefilter("ignore", DeprecationWarning)
        importlib.import_module(moduleName)


def test_the_scan_actually_found_the_package() -> None:
    """Guards the parametrisation itself.

    ``walk_packages`` returning nothing would make every assertion above vacuous
    and the file would still be green, which is the way this kind of test dies.
    """
    modules = _covModules()
    assert len(modules) > 10, modules
    assert "kika.cov.legendre_covariance" in modules
    assert "kika.cov.multigroup.collapse" in modules
    assert not any(".tests" in m for m in modules), "test packages must be excluded"
