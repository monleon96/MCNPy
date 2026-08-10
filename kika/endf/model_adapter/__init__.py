"""ENDF ↔ the GNDS-shaped canonical model.

**Import rule, enforced by a test.** Nothing outside this package and its tests
may import it. A module-level import from ``kika/endf/__init__.py`` or
``read_endf`` would put the model on the ``read_endf`` critical path and change
what ``import kika`` costs for the cluster pipeline and the desktop app. The
package is also deliberately absent from kika-app's PyInstaller
``hiddenimports``, so a lazy import from an app-reachable code path would break
the frozen build at run time and nowhere else.
"""
from .angular import decodeMF4MT, encodeMF4MT
from .covariances import (
    decodeCovarianceSuite,
    decodeMF31MT,
    decodeMF33MT,
    decodeMF34MT,
    decodeMF35MT,
    encodeMF31MT,
    encodeMF33MT,
    encodeMF34MT,
)
from .decode import decodeMF1MT451, decodeMF3MT, decodeReactionSuite
from .encode import encodeMF1MT451, encodeMF3MT
from .multiplicity import (
    attachNubar,
    decodeMF1Nubar,
    encodeMF1MT452,
    encodeMF1MT455,
    encodeMF1MT456,
    nubarHref,
)
from .resonances import decodeMF2MT151, encodeMF2MT151

__all__ = [
    "decodeMF3MT", "decodeMF1MT451", "decodeReactionSuite", "encodeMF3MT",
    "encodeMF1MT451",
    "decodeMF4MT", "encodeMF4MT", "decodeMF2MT151", "encodeMF2MT151",
    "decodeMF31MT", "decodeMF33MT", "decodeMF34MT", "decodeMF35MT",
    "decodeCovarianceSuite",
    "encodeMF31MT", "encodeMF33MT", "encodeMF34MT",
    "decodeMF1Nubar", "attachNubar", "nubarHref",
    "encodeMF1MT452", "encodeMF1MT455", "encodeMF1MT456",
]
