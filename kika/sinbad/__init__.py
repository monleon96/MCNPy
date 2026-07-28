"""
SINBAD shielding benchmarks for KIKA.

Read SINBAD benchmark packages -- measured data, calculations, C/E, uncertainty
structure and sensitivity profiles -- as objects with ``to_dataframe()``.

This is a **different repository** from :mod:`kika.benchmarks`, which holds
ICSBEP/DICE criticality benchmarks. Nothing is shared between them.

kika ships no benchmark data. Point it at your own packages, then refer to them
by identifier::

    >>> import kika.sinbad as sinbad
    >>> sinbad.configure(path="~/sinbad-packages")
    >>> sinbad.list_benchmarks()
    ['SINBAD-ASPIS-IRON88']
    >>> b = sinbad.open("SINBAD-ASPIS-IRON88")
    >>> print(b.summary())
    >>> b.to_dataframe()          # the experimental data
    >>> b.to_dataframe("Al27")    # ... for one detector foil
    >>> b.ce(wide=True)           # C/E, positions x libraries
    >>> b.covariance("Al27")      # experimental covariance, labelled
    >>> b.uncertainty_budget()    # what the declared correlations are worth
    >>> b.unresolved()            # what the entry says it does NOT know
    >>> b.findings()              # what the entry says about its own data

An entry holds several measurement systems -- for ASPIS Iron-88, five
activation foils. Every table method takes an optional ``system`` argument that
accepts an identifier, a target nuclide, or any unambiguous fragment.

Or open a package directly, with no configuration at all::

    >>> from kika.sinbad import SinbadBenchmark
    >>> b = SinbadBenchmark.open("aspis-iron88.sinbad")

Both package forms are accepted -- a directory or a single ``.sinbad`` archive.
Everything except arrays is readable without optional dependencies; sensitivity
coefficients and covariance payloads need ``h5py``.
"""

from kika.sinbad.benchmark import (
    Calculation,
    CalculationInput,
    Measurement,
    MeasurementSystem,
    SensitivitySet,
    SinbadBenchmark,
)
from kika.sinbad.config import configure, get_config, get_library_path, reset_config
from kika.sinbad.exceptions import (
    ArrayBackendMissingError,
    LibraryNotConfiguredError,
    PackageNotFoundError,
    SinbadError,
)
from kika.sinbad.library import catalogue, find_package, list_benchmarks, scan
from kika.sinbad.package import SinbadPackage
from kika.sinbad.plotting import (
    plot_ce,
    plot_sensitivity,
    plot_sensitivity_depth,
    plot_uncertainty_budget,
)

#: Open a benchmark by identifier or path. Alias of
#: :meth:`kika.sinbad.SinbadBenchmark.open`.
open = SinbadBenchmark.open  # noqa: A001 - deliberate: sinbad.open(...) reads well

__all__ = [
    "SinbadBenchmark",
    "SinbadPackage",
    "Measurement",
    "MeasurementSystem",
    "Calculation",
    "CalculationInput",
    "SensitivitySet",
    "open",
    "configure",
    "get_config",
    "get_library_path",
    "reset_config",
    "scan",
    "list_benchmarks",
    "catalogue",
    "find_package",
    "plot_ce",
    "plot_sensitivity",
    "plot_sensitivity_depth",
    "plot_uncertainty_budget",
    "SinbadError",
    "PackageNotFoundError",
    "LibraryNotConfiguredError",
    "ArrayBackendMissingError",
]
