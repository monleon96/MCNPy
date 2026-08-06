"""Format-agnostic canonical representations of nuclear data.

**These classes are deprecated.** kika's canonical model is
:mod:`kika.nuclear_data.model`, which is shaped and named after GNDS-2.1. The
four classes here — ``CrossSection``, ``AngularDistribution``,
``ResonanceParameters``, ``NuclideInfo`` — are now **façades** over it: their
fields, their order, their defaults and their signatures are exactly what they
were, and their bodies read the file through the model. They are scheduled for
removal in kika-nd 1.0.

Three of the four route their ``from_endf`` through
``kika.endf.model_adapter``. ``CrossSection`` does not, and the reason is
measured rather than preferred: a model round trip in its constructor costs 2.5x
and ``kika/processing/reconstruct.py`` builds one per MT per call, which the
cluster runs per sample per temperature. It exposes :meth:`CrossSection.to_model`
instead, so a caller who wants the model pays for it at the point of asking.

**Nothing is imported until it is used.** The names below resolve through a
module-level ``__getattr__`` (PEP 562), so ``import kika`` — which reaches this
package eagerly via ``kika/__init__.py`` — no longer loads four modules and
their numpy-heavy dependencies for callers that never touch them. The
deprecation warning fires **once per name, on first access**: not per instance,
which would put the ``warnings`` registry on the reconstruction hot path and
flood sbatch logs, and not at import, which would break every consumer at once
for a message none of them asked for.

The honest note, stated once: the model this points at cannot yet reconstruct
cross sections, group-average, or collapse covariances — that is phase 4. So
the warning says these classes are *going away*, not that there is a drop-in
replacement for everything they do today.
"""

import importlib
import warnings

__all__ = [
    "CrossSection",
    "AngularDistribution",
    "ResonanceParameters",
    "ResonanceRecord",
    "LGroup",
    "UnresolvedResonanceParameters",
    "URR_LGroup",
    "URR_JGroup",
    "NuclideInfo",
]

#: name -> the module it lives in. The whole public surface of this package.
_MODULES = {
    "CrossSection": "cross_section",
    "AngularDistribution": "angular_distribution",
    "ResonanceParameters": "resonance_parameters",
    "ResonanceRecord": "resonance_parameters",
    "LGroup": "resonance_parameters",
    "UnresolvedResonanceParameters": "resonance_parameters",
    "URR_LGroup": "resonance_parameters",
    "URR_JGroup": "resonance_parameters",
    "NuclideInfo": "nuclide_info",
}

#: Names already warned about, so the message appears once per process per name
#: rather than once per access. ``warnings``' own registry would mostly do this,
#: but it is keyed on the *call site*, so a name reached from twenty notebooks
#: cells would warn twenty times.
_WARNED: set = set()


def __getattr__(name: str):
    """Resolve a deprecated name, warning once, on first access (PEP 562)."""
    module = _MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name not in _WARNED:
        _WARNED.add(name)
        warnings.warn(
            f"kika.nuclear_data.{name} is deprecated and will be removed in "
            f"kika-nd 1.0. The canonical model is now kika.nuclear_data.model, "
            f"which is shaped after GNDS-2.1; {name} is a façade over it and "
            f"reads through kika.endf.model_adapter. Note that the model does "
            f"not yet cover reconstruction, group averaging or covariance "
            f"collapsing, so there is no drop-in replacement for every use yet.",
            DeprecationWarning,
            stacklevel=2,
        )

    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value  # subsequent accesses skip __getattr__ entirely
    return value


def __dir__():
    return sorted(__all__)
