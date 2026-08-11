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

What exists is §2-7 (units, quantities, enumerations, axes, values, the
one-dimensional functional containers) and the §9-25 hierarchy — ``styles``,
``reactionSuite`` and its children, resonances split by formalism, and the
``covarianceSuite``. The containers are structurally complete and mostly empty:
nothing decodes an ENDF file into them yet, which is phase 3c.
"""
from __future__ import annotations

from .axes import Axes, Axis, Grid, crossSectionAxes, multiplicityAxes
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
    Function2d,
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
    fromEndfTab2,
    toEndfTab2,
)
from .conversion import ConversionReport
from .covariances import (
    CovarianceMatrix,
    CovarianceSection,
    CovarianceSuite,
    DataLink,
    Mixed,
    ParameterCovariance,
    ParameterCovarianceMatrix,
    ParameterLink,
    Slice,
    Slices,
    Sum,
)
from .cross_section_forms import (
    EVAL_LABEL,
    CoulombPlusNuclearElastic,
    CrossSection,
    Reference,
    ResonancesWithBackground,
    ThermalNeutronScatteringLaw1d,
    URR_probabilityTables1d,
)
from .distributions import (
    NOT_IMPLEMENTED_DISTRIBUTIONS,
    AngularEnergy,
    AngularTwoBody,
    Distribution,
    EnergyAngular,
    Isotropic2d,
    KalbachMann,
    NBodyPhaseSpace,
    Recoil,
    Uncorrelated,
    Unspecified,
)
from .output_channel import (DelayedNeutron, DelayedNeutrons,
                             FissionFragmentData, Multiplicity, OutputChannel,
                             Product, Products, Q)
from .pops import Nuclide, Particle, PoPs
from .provenance import AceProvenance, EndfProvenance, Provenance
from .quantities import PhysicalQuantity
from .quantities import Uncertainty as ScalarUncertainty
from .reaction_id import ReactionId
from .reactions import (
    FissionComponents,
    IncompleteReactions,
    OrphanProducts,
    Productions,
    Reaction,
    Reactions,
    Sums,
)
from .resonances import (
    BreitWigner,
    BreitWignerApproximation,
    Channel,
    Resonance,
    ResolvedRegion,
    Resonances,
    RMatrix,
    RMatrixSpinGroup,
    ScatteringRadius,
    SpinGroup,
    TabulatedWidths,
    UnresolvedRegion,
)
from .sums import Add, MultiplicitySum, MultiplicitySums, Summands
from .styles import (
    AngularDistributionReconstructed,
    CrossSectionReconstructed,
    Evaluated,
    GriddedCrossSection,
    Heated,
    HeatedMultiGroup,
    Realization,
    Style,
    StyleError,
    Styles,
    URR_probabilityTables,
)
from .suite import ApplicationData, ExternalFile, ExternalFiles, ReactionSuite
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
    "Axes", "Axis", "Grid", "crossSectionAxes", "multiplicityAxes", "Values",
    # §6
    "Function1d", "XYs1d", "Regions1d", "Constant1d", "Polynomial1d",
    "Ys1d", "Legendre", "Gridded1d",
    "Function2d", "XYs2d", "Regions2d", "fromEndfTab2", "toEndfTab2",
    "XYs3d", "Regions3d",
    # §7
    "Uncertainty", "Covariance", "ListOfCovariances",
    # §9-10 styles
    "Style", "Styles", "StyleError", "Evaluated", "Realization",
    "CrossSectionReconstructed", "AngularDistributionReconstructed",
    "Heated", "HeatedMultiGroup", "GriddedCrossSection", "URR_probabilityTables",
    # §12 PoPs
    "PoPs", "Particle", "Nuclide",
    # §14 the root
    "ReactionSuite", "ExternalFile", "ExternalFiles", "ApplicationData",
    # §15-16 reactions and their cross sections
    "Reaction", "Reactions", "Sums", "OrphanProducts", "FissionComponents",
    "Productions", "IncompleteReactions", "ReactionId",
    "CrossSection", "EVAL_LABEL", "ResonancesWithBackground", "Reference",
    "CoulombPlusNuclearElastic", "ThermalNeutronScatteringLaw1d",
    "URR_probabilityTables1d",
    # §17-18 output channels and distributions
    "OutputChannel", "Product", "Products", "Multiplicity", "Q",
    "FissionFragmentData", "DelayedNeutron", "DelayedNeutrons",
    "Add", "Summands", "MultiplicitySum", "MultiplicitySums",
    "Distribution", "AngularTwoBody", "Isotropic2d", "Unspecified", "Uncorrelated",
    "EnergyAngular", "AngularEnergy", "KalbachMann", "NBodyPhaseSpace",
    "Recoil", "NOT_IMPLEMENTED_DISTRIBUTIONS",
    # §19 resonances, by formalism
    "Resonances", "ResolvedRegion", "UnresolvedRegion", "ScatteringRadius",
    "BreitWigner", "BreitWignerApproximation", "Resonance", "SpinGroup",
    "RMatrix", "RMatrixSpinGroup", "Channel", "TabulatedWidths",
    # conversion bookkeeping (not GNDS nodes)
    "ConversionReport", "Provenance", "EndfProvenance", "AceProvenance",
    # §25 covariances
    "CovarianceSuite", "CovarianceSection", "DataLink", "CovarianceMatrix",
    "Mixed", "Sum", "Slice", "Slices",
    # §25.3 parameter covariances
    "ParameterCovariance", "ParameterCovarianceMatrix", "ParameterLink",
]
