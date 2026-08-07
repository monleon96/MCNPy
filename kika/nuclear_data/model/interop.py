"""The plain-data bridge between the model and the flat classes it replaces.

Phase 3d makes ``NuclideInfo``, ``AngularDistribution``, ``ResonanceParameters``
and ``CrossSection`` façades over the GNDS model. Every one of them needs the
same thing: model nodes in, the dict of constructor arguments the flat
dataclass expects out. That projection lives here.

**This module imports no flat class, and must not.** It returns dicts and
arrays, and the flat class does ``cls(**projection)``. The reason is the
dependency arrow the whole restructuring is about: ``kika/nuclear_data/model/``
is the bottom of the stack, and a model module importing
``kika.nuclear_data.cross_section`` would make the bottom depend on the layer
above it. Returning plain data keeps the arrow pointing one way and keeps this
module testable without constructing anything it does not own.

**One-way, and deliberately so.** There is no ``from_flat_*``. The flat classes
lose information the model keeps — a ``regions1d``'s per-region interpolation
becomes one "dominant" string, an ``RMatrix``'s per-channel widths do not fit
four positional columns — so a round trip through them is lossy by
construction. Anything that needs the model asks the adapter for it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "flatNuclideInfo", "flatCrossSection", "flatAngularDistribution",
    "flatResonanceParameters", "flatUnresolvedResonanceParameters",
    "dominantInterpolation", "INTERPOLATION_NAMES",
]

#: The flat classes' interpolation spellings, by ENDF INT code. Kept here rather
#: than derived from :class:`~kika.nuclear_data.model.enums.Interpolation`
#: because they are a *different* vocabulary — ``'linlin'``, not ``'lin-lin'`` —
#: and silently unifying them would change what every consumer reads.
INTERPOLATION_NAMES = {
    1: "histogram",
    2: "linlin",
    3: "linlog",
    4: "loglin",
    5: "loglog",
    6: "charged_particle",
}


def dominantInterpolation(regions: List[Tuple[int, int]]) -> str:
    """The flat classes' single interpolation string, from the ENDF regions.

    ``NBT`` is **cumulative**, so region *i* spans ``NBT[i] - NBT[i-1]`` points.
    Reading it as a per-region count is what made the original
    ``_dominant_interpolation`` always pick the last region rather than the
    widest — the phase 1 defect. The arithmetic is repeated here rather than
    imported because importing it would point this module upward.
    """
    if not regions:
        return "linlin"
    widest, previous, code = -1, 0, 2
    for nbt, intCode in regions:
        width = int(nbt) - previous
        if width > widest:
            widest, code = width, int(intCode)
        previous = int(nbt)
    return INTERPOLATION_NAMES.get(code, "linlin")


# ---------------------------------------------------------------------------
# MF1/451 -> NuclideInfo
# ---------------------------------------------------------------------------

def flatNuclideInfo(provenance) -> Dict[str, Any]:
    """``EndfProvenance`` from ``decodeMF1MT451`` → ``NuclideInfo(**this)``."""
    header = dict(provenance.headerFields)
    return {
        "nuclide_id": provenance.za or 0,
        "atomic_weight_ratio": provenance.awr or 0.0,
        "temperature": header.get("temp") or 0.0,
        "evaluation_info": dict(provenance.evaluationInfo),
        "metadata": {
            "mat": provenance.mat,
            **{key: header.get(key) for key in (
                "lrp", "lfi", "nlib", "nmod", "elis", "sta", "lis", "liso",
                "nfor", "awi", "emax", "lrel", "nsub", "nver", "ldrv",
            )},
            # What `to_endf` needs and no other field carries: the free-text
            # block and the directory. NWD and NXC are deliberately not stored
            # -- they are the lengths of these two lists, and a stored count
            # that can disagree with what it counts is a defect waiting to be
            # written. See `docs/library-gaps.md` M2.
            "text": list(provenance.descriptiveText),
            "directory": [tuple(entry) for entry in provenance.directory],
        },
    }


# ---------------------------------------------------------------------------
# MF3 -> CrossSection
# ---------------------------------------------------------------------------

def flatCrossSection(reaction, temperature: float = 0.0) -> Dict[str, Any]:
    """A model ``Reaction`` → ``CrossSection(**this)``.

    ``qi`` comes from the output channel and the rest from provenance, which is
    the split §17.1.1 makes: the reaction Q is physics and lives on the channel,
    ``QM`` and ``LR`` are ENDF bookkeeping.
    """
    from .cross_section_forms import EVAL_LABEL

    form = reaction.crossSection[EVAL_LABEL]
    energies, values, regions = form.toEndfRegions()
    provenance = reaction.provenance

    metadata: Dict[str, Any] = {
        "mat": provenance.mat,
        "awr": provenance.awr,
        "qm": provenance.qm,
        "lr": provenance.lr,
        "interpolation_regions": list(provenance.interpolationRegions or regions),
    }
    if reaction.outputChannel.Q.isKnown:
        metadata["qi"] = reaction.outputChannel.Q.value

    return {
        "energies": np.asarray(energies, dtype=float),
        "values": np.asarray(values, dtype=float),
        "reaction": reaction.id.ENDF_MT,
        "nuclide_id": provenance.za or 0,
        "temperature": temperature,
        "interpolation": dominantInterpolation(metadata["interpolation_regions"]),
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# MF4 -> AngularDistribution
# ---------------------------------------------------------------------------

#: ENDF LTT → the flat class's ``representation`` string.
_LTT_TO_REPRESENTATION = {0: "isotropic", 1: "legendre", 2: "tabulated", 3: "mixed"}


def flatAngularDistribution(distribution, provenance, mt: int) -> Dict[str, Any]:
    """An ``AngularTwoBody`` or ``Isotropic2d`` → ``AngularDistribution(**this)``.

    The flat class's ``coefficients`` is ``{order: array[n_energies]}`` — a dense
    rectangle padded to the highest order in the section — while the model keeps
    each energy's own coefficient list at its own length. The padding is
    reproduced exactly, including the trailing zeros the flat writer used to trim,
    so nothing that reads ``coefficients`` sees a different array than before.
    """
    from .distributions import Isotropic2d
    from .functions import Regions2d

    header = dict(provenance.headerFields)
    ltt = header.get("ltt")
    common = {
        "reaction": mt,
        "nuclide_id": provenance.za or 0,
        "frame": "LAB" if header.get("lct") == 1 else "CM",
        "metadata": {
            "mat": provenance.mat,
            "awr": provenance.awr,
            "ltt": ltt,
            "li": header.get("li"),
            "lct": header.get("lct"),
        },
    }

    if isinstance(distribution, Isotropic2d):
        return {
            **common,
            "energies": np.array([], dtype=float),
            "coefficients": {0: np.array([1.0])},
            "representation": "isotropic",
        }

    angular = distribution.angular
    if isinstance(angular, Regions2d) and ltt == 3:
        legendrePart, tabulatedPart = angular[0], angular[1]
        legendreFunctions = legendrePart.function1ds if hasattr(
            legendrePart, "function1ds") else list(legendrePart)
        tabulatedFunctions = tabulatedPart.function1ds if hasattr(
            tabulatedPart, "function1ds") else list(tabulatedPart)
    elif ltt == 2:
        legendreFunctions, tabulatedFunctions = [], _flatten(angular)
    else:
        legendreFunctions, tabulatedFunctions = _flatten(angular), []

    legendreEnergies = [f.outerDomainValue for f in legendreFunctions]
    tabulatedEnergies = [f.outerDomainValue for f in tabulatedFunctions]

    coefficients = _denseCoefficients(legendreFunctions)
    common["metadata"]["energy_interpolation"] = _outerRegions(
        angular[0] if (ltt == 3 and isinstance(angular, Regions2d)) else angular
    )
    # The two things a dense {order: array} cannot carry, and whose absence is
    # `docs/library-gaps.md` D2. NM is the evaluation's declared highest order;
    # `legendre_orders` is each energy's own NL, so a trailing zero coefficient
    # the evaluator wrote survives instead of being trimmed on the way out.
    common["metadata"]["nm"] = header.get("nm")
    common["metadata"]["legendre_orders"] = [
        int(np.asarray(f.coefficients).size) - 1 for f in legendreFunctions
    ]

    if ltt == 3:
        common["metadata"]["tab_interpolation"] = _outerRegions(angular[1])
        return {
            **common,
            "energies": np.asarray(legendreEnergies + tabulatedEnergies, dtype=float),
            "coefficients": coefficients,
            "representation": "mixed",
            "tabulated_data": {
                **_tabulatedData(tabulatedFunctions),
                "tabulated_energies": list(tabulatedEnergies),
                "legendre_energies": list(legendreEnergies),
            },
        }

    if ltt == 2:
        return {
            **common,
            "energies": np.asarray(tabulatedEnergies, dtype=float),
            "coefficients": {},
            "representation": "tabulated",
            "tabulated_data": _tabulatedData(tabulatedFunctions),
        }

    return {
        **common,
        "energies": np.asarray(legendreEnergies, dtype=float),
        "coefficients": coefficients,
        "representation": _LTT_TO_REPRESENTATION.get(ltt, "legendre"),
    }


def _flatten(form) -> List[Any]:
    from .functions import Regions2d

    if form is None:
        return []
    if isinstance(form, Regions2d):
        return form.function1ds
    return list(form.function1ds)


def _outerRegions(form) -> List[Tuple[int, int]]:
    from .functions import toEndfTab2

    if form is None:
        return []
    _, pairs = toEndfTab2(form)
    return pairs


def _denseCoefficients(functions) -> Dict[int, np.ndarray]:
    """``{order: array[n_energies]}``, padded to the section's highest order.

    ``a_0`` is 1 everywhere: the model stores it explicitly and ENDF leaves it
    implicit, and the flat class has always reported 1.0.
    """
    count = len(functions)
    if not count:
        return {}
    maxOrder = max(int(np.asarray(f.coefficients).size) - 1 for f in functions)
    dense: Dict[int, np.ndarray] = {0: np.ones(count, dtype=float)}
    for order in range(1, maxOrder + 1):
        column = np.zeros(count, dtype=float)
        for index, function in enumerate(functions):
            coefficients = np.asarray(function.coefficients, dtype=float)
            if order < coefficients.size:
                column[index] = coefficients[order]
        dense[order] = column
    return dense


def _tabulatedData(functions) -> Dict[str, Any]:
    cosines, probabilities, interpolation = [], [], []
    for function in functions:
        mu, p, pairs = function.toEndfRegions()
        cosines.append([float(v) for v in mu])
        probabilities.append([float(v) for v in p])
        interpolation.append([(int(a), int(b)) for a, b in pairs])
    return {
        "cosines": cosines,
        "probabilities": probabilities,
        "angular_interpolation": interpolation,
    }


# ---------------------------------------------------------------------------
# MF2/151 -> ResonanceParameters
# ---------------------------------------------------------------------------

#: The flat class's formalism names, by ENDF LRF.
_LRF_TO_NAME = {1: "SLBW", 2: "MLBW", 3: "RM", 7: "RML"}


def flatResonanceParameters(region, fields: Dict[str, Any], provenance
                            ) -> Optional[Dict[str, Any]]:
    """A ``ResolvedRegion`` + its ENDF range fields → ``ResonanceParameters(**this)``.

    ``None`` for a formalism the flat class cannot hold. That is not a failure
    of this projection: ``ResonanceRecord`` has four width columns and an
    R-Matrix-Limited spin group has one width *per channel* — five of them for
    Fe-57 in JEFF-4.0 — so there is nothing to project into. The caller warns;
    see ``docs/library-gaps.md`` D3.

    ``l_groups`` comes back as plain dicts rather than ``LGroup`` objects,
    because this module must not import the flat classes. The façade builds them.
    """
    from .resonances import BreitWigner, RMatrix

    formalism = region.formalism
    lrf = fields.get("lrf")
    name = _LRF_TO_NAME.get(lrf)
    if name is None:
        return None

    if isinstance(formalism, BreitWigner):
        groups = [
            {
                "awri": group.atomicWeightRatio,
                "l": group.L,
                "records": [r.toFlat() for r in group.resonances],
                "ap": group.scatteringRadius,
            }
            for group in formalism.resonanceParameters.spinGroups
        ]
    elif isinstance(formalism, RMatrix) and lrf == 3:
        groups = [
            {
                "awri": group.atomicWeightRatio,
                "l": group.channels[0].L if group.channels else None,
                "records": [
                    (energy, spin, *widths)
                    for energy, spin, widths in zip(
                        group.energies, group.spins, group.widths)
                ],
                "ap": group.channels[0].scatteringRadius if group.channels else None,
            }
            for group in formalism.spinGroups
        ]
    else:
        # LRF=7: channel-shaped, and c3..c6 cannot hold it.
        return None

    return {
        "nuclide_id": provenance.za or 0,
        "spin": fields.get("spi"),
        "scattering_radius": fields.get("ap"),
        "formalism": name,
        "energy_range": (region.domainMin, region.domainMax),
        "l_groups": groups,
        "metadata": {
            "mat": provenance.mat,
            "awr": provenance.awr,
            "lrf": lrf,
            "nlsc": fields.get("nlsc"),
            "abundance": fields.get("abn"),
        },
        "scattering_radius_table": fields.get("radius_table"),
    }


def flatUnresolvedResonanceParameters(unresolved, fields: Dict[str, Any],
                                      provenance) -> Dict[str, Any]:
    """An ``UnresolvedRegion`` → ``UnresolvedResonanceParameters(**this)``.

    ``l_groups`` is regrouped back by L, because the model keeps one spin group
    per ``(L, J)`` — which is what the data is — while the flat class nests J
    inside L. The nesting is a storage choice, so the regrouping is exact.
    """
    widths = unresolved.tabulatedWidths
    byL: Dict[int, Dict[str, Any]] = {}
    for group in widths.spinGroups:
        entry = byL.setdefault(group.L, {"awri": group.atomicWeightRatio,
                                         "l": group.L, "j_groups": []})
        channels = {c.label: c for c in group.channels}
        neutron = channels.get("neutron")
        entry["j_groups"].append({
            "j": group.J,
            "amun": neutron.degreesOfFreedom if neutron else 1.0,
            "d": _channelValue(group.levelSpacing),
            "gn0": _channelWidth(neutron),
            "gg": _channelWidth(channels.get("capture")),
            "gf": _channelWidth(channels.get("fission")),
            "gx": _channelWidth(channels.get("competitive")),
        })

    return {
        "nuclide_id": provenance.za or 0,
        "spin": fields.get("spi"),
        "scattering_radius": widths.scatteringRadius,
        "energy_range": (unresolved.domainMin, unresolved.domainMax),
        "lssf": int(widths.selfShieldingOnly),
        "l_groups": [byL[key] for key in sorted(byL)],
        "energy_grid": widths.energyGrid,
        "metadata": {
            "mat": provenance.mat,
            "awr": provenance.awr,
            "abundance": fields.get("abn"),
        },
    }


def _channelValue(values):
    """A one-element array is ENDF case A's scalar; anything longer is a table."""
    if values is None:
        return 0.0
    array = np.asarray(values, dtype=float)
    return float(array[0]) if array.size == 1 else array


def _channelWidth(channel):
    if channel is None:
        return 0.0
    if channel.widths is not None:
        return channel.widths
    return channel.constantWidth if channel.constantWidth is not None else 0.0
