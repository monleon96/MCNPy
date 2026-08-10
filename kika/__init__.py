from .mcnp.parse_mctal import read_mctal
from .mcnp.parse_input import read_mcnp
from .mcnp.pert_generator import generate_pert_cards, generate_PERTcards, perturb_material, perturb_materials
from .sensitivities.sensitivity_processing import create_sdf_data, compute_sensitivity, compute_total_sensitivity, plot_sens_comparison
from .sensitivities.sdf import SDFData
from .sensitivities.sdf_parser import read_sdf
from .ace.parsers.parse_ace import read_ace
from .cov.parse_covmat import read_coverx, read_covfil, read_boxer, read_scale_covmat, read_njoy_covmat
from .endf.read_endf import read_endf
from .wwinp.read_wwinp import read_wwinp
from . import energy_grids
from . import materials
from . import nuclear_data
from . import processing
from .materials import Material, MaterialCollection, Nuclide, NuclideAccessor
from ._config import LIBRARY_VERSION, AUTHOR

__version__ = LIBRARY_VERSION
__author__ = AUTHOR

#: Lazily-resolved public names -> the private module holding each. See __getattr__.
_LAZY = {
    'read': '_read',
    'sniff_format': '_read',
}


def __getattr__(name):
    """PEP 562. ``kika.read`` resolves on first access, never on import.

    This is load-bearing, not tidiness. ``kika.read`` lives in ``kika/_read.py``,
    which reaches the model adapters and therefore ``kika.nuclear_data.model`` —
    and the model must stay dormant on a plain ``import kika``, because every
    consumer of this library (the cluster pipeline, kika-app, every notebook)
    pays for whatever import time costs. A module-scope ``from ._read import
    read`` here would wake the model for all of them to serve the few who call
    it.

    **The private module name matters.** ``importlib`` binds a submodule onto its
    package as a side effect of importing it, so a module named ``kika/read.py``
    would overwrite this function with itself on first access — first call
    returns the function, every later one returns the module. Hence ``_read``.

    ``kika/nuclear_data/__init__.py`` uses the same mechanism for the same
    reason. ``kika.nuclear_data.model.tests.test_dormancy`` is the test.
    """
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(f'.{_LAZY[name]}', __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    # Without this, the lazy names are invisible to tab-completion and dir() —
    # PEP 562 __getattr__ is consulted on access, never on enumeration.
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    'read', 'sniff_format',
    'read_mctal',
    'read_mcnp', 'generate_pert_cards', 'generate_PERTcards', 'perturb_material', 'perturb_materials',
    'compute_sensitivity', 'compute_total_sensitivity', 'plot_sens_comparison',
    'SDFData', 'create_sdf_data', 'read_sdf',
    'read_ace',
    'read_endf',
    'read_wwinp',
    'read_coverx', 'read_covfil', 'read_boxer', 'read_scale_covmat', 'read_njoy_covmat',
    'materials', 'Material', 'MaterialCollection', 'Nuclide', 'NuclideAccessor',
]

