"""kika's canonical nuclear-data model, shaped and named after GNDS-2.1.

**Dormant on purpose.** ``kika/nuclear_data/__init__.py`` does not import this
package, and must not until phase 3d. Everything under ``kika/`` reaches
``kika.nuclear_data`` transitively on ``import kika`` — ``kika/__init__.py:13``
does ``from . import nuclear_data`` at module scope — so an import error, a
circular import or a slow table build in here would break the cluster pipeline,
the desktop app and every notebook at once, not merely the code that uses the
model. ``tests/test_dormancy.py`` asserts the package stays unreachable from a
plain ``import kika``.

**Naming.** Inside this package names are GNDS's, verbatim: classes take the
node name with its first letter capitalised (``XYs1d``, ``Regions1d``,
``PhysicalQuantity``), and attributes keep the spec's spelling exactly
(``outerDomainValue``, ``domainMin``, ``valueType``, ``interpolationQualifier``).
Do **not** "fix" these to snake_case — the phase 5 reader resolves GNDS nodes by
introspection on these names, and a rename silently breaks it.
``tests/test_gnds_naming.py`` enforces this, because a documented convention
does not survive contact with a linter or an agent. Local variables and private
helpers are still PEP 8. See ``NAMING.md``.

**No format imports.** Nothing here may import ``kika.endf`` or ``kika.ace``, at
runtime or under ``TYPE_CHECKING``. ``kika/tests/test_layering.py`` scans this
directory automatically via ``rglob``, and no allowlist entry may ever be added
for it.

What exists so far is §2-7 — units, quantities, enumerations, axes, values and
the one-dimensional functional containers. The ``reactionSuite`` hierarchy
(§9-25) is phase 3b.
"""
from __future__ import annotations

from .axes import Axes, Axis, Grid, crossSectionAxes
from .enums import (
    ENDF_INT_TO_INTERPOLATION,
    INTERPOLATION_TO_ENDF_INT,
    Frame,
    GridStyle,
    Interpolation,
    InterpolationQualifier,
    ValueType,
)
from .functions import (
    Constant1d,
    Function1d,
    Gridded1d,
    Legendre,
    Polynomial1d,
    Regions1d,
    Regions2d,
    Regions3d,
    XYs1d,
    XYs2d,
    XYs3d,
    Ys1d,
)
from .quantities import PhysicalQuantity
from .quantities import Uncertainty as ScalarUncertainty
from .uncertainties import Covariance, ListOfCovariances, Uncertainty
from .units import Unit, UnitError, check_mass_unit, conversion_factor, parse_unit
from .values import Values

__all__ = [
    # §3.5 units
    "Unit", "UnitError", "parse_unit", "conversion_factor", "check_mass_unit",
    # §2.3.3
    "PhysicalQuantity", "ScalarUncertainty",
    # §3.4
    "Frame", "Interpolation", "InterpolationQualifier", "GridStyle", "ValueType",
    "ENDF_INT_TO_INTERPOLATION", "INTERPOLATION_TO_ENDF_INT",
    # §5
    "Axes", "Axis", "Grid", "crossSectionAxes", "Values",
    # §6
    "Function1d", "XYs1d", "Regions1d", "Constant1d", "Polynomial1d",
    "Ys1d", "Legendre", "Gridded1d",
    "XYs2d", "XYs3d", "Regions2d", "Regions3d",
    # §7
    "Uncertainty", "Covariance", "ListOfCovariances",
]
