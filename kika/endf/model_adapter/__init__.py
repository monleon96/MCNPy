"""ENDF ↔ the GNDS-shaped canonical model.

**Import rule, enforced by a test.** Nothing outside this package and its tests
may import it. A module-level import from ``kika/endf/__init__.py`` or
``read_endf`` would put the model on the ``read_endf`` critical path and change
what ``import kika`` costs for the cluster pipeline and the desktop app. The
package is also deliberately absent from kika-app's PyInstaller
``hiddenimports``, so a lazy import from an app-reachable code path would break
the frozen build at run time and nowhere else.
"""
from .covariances import (
    decodeCovarianceSuite,
    decodeMF33MT,
    decodeMF34MT,
)
from .decode import decodeMF1MT451, decodeMF3MT, decodeReactionSuite
from .encode import encodeMF3MT

__all__ = [
    "decodeMF3MT", "decodeMF1MT451", "decodeReactionSuite", "encodeMF3MT",
    "decodeMF33MT", "decodeMF34MT", "decodeCovarianceSuite",
]
