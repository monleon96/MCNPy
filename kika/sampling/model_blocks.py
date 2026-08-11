"""The seam between a format-agnostic covariance object and the sampler.

:func:`kika.sampling.core.draw_samples` takes ``(key, matrix)`` pairs and knows
nothing about where they came from — which is the point. These functions are the
only thing that has to know that a ``CovarianceSuite`` exists, and they are
deliberately small: if adding a new source format means writing more than a
dozen lines here, the split between *processing* and *formatting* was not real.

**What is and is not format-specific.** The draw is generic and lives in
``core``. Reading a file into the model is format work by definition, and so is
writing perturbed values back onto a file's own representation — scaling MF3 on
its grid, renormalising an MF5 spectrum, rewriting MF2's resonance parameters.
The middle, which is this module and ``core``, is shared. That is why
``mf33_sampling.py`` and friends read as format-specific end to end: they bundle
load, draw and apply into one file, and only two thirds of that is really tied
to a format.

Nothing here imports :mod:`kika.nuclear_data.model`, and that is enforced
elsewhere by the layering ratchet: these functions duck-type the suite so that
``import kika.sampling`` never drags the model onto the critical path.
"""
from __future__ import annotations

from typing import Any, Dict, Hashable, List, Tuple

import numpy as np

__all__ = ["covariance_suite_blocks", "parameter_covariance_blocks",
           "parameter_covariance_index"]


def covariance_suite_blocks(suite, isotope: Any = None, mt: int = 18):
    """A ``CovarianceSuite``'s §25.2 sections → the ``(key, matrix)`` the core wants.

    One block per section. Deliberately *not* concatenated: the sections of one
    MF35 file have different orders (84, 641, 641, 641, 641 on ENDF/B-VIII.1
    U-235), so anything that assembled them at a common dimension would produce
    a malformed matrix without saying so.
    """
    blocks = []
    for index, section in enumerate(suite):
        blocks.append(((isotope, mt, index), np.asarray(section.form.matrix)))
    return blocks


def parameter_covariance_blocks(
    suite, isotope: Any = None, relative: Any = None,
) -> List[Tuple[Hashable, np.ndarray]]:
    """A ``CovarianceSuite``'s §25.3 parameter covariances → ``(key, matrix)``.

    The sibling of :func:`covariance_suite_blocks`, and the whole of what MF32
    needed on the sampling side — ``draw_samples`` is not touched, which is the
    test of whether the seam was in the right place.

    Blocks are kept separate for a stronger reason than in the §25.2 case: two
    parameter covariances describe *disjoint sets of parameters* (different
    energy ranges, or different short-range blocks of one LCOMP=1 body), so
    there is no matrix over their union that the file has anything to say
    about.

    ``relative`` filters by form: ``True`` for the relative covariances
    (§32.2.4's unresolved region), ``False`` for the absolute ones (every
    resolved body), ``None`` for both. **Draw them separately or not at all** —
    an absolute block wants ``returns="deltas"`` and a relative one
    ``returns="factors"``, and mixing the two in one call silently applies one
    convention to both.
    """
    blocks: List[Tuple[Hashable, np.ndarray]] = []
    for covariance in getattr(suite, "parameterCovariances", []):
        form = covariance.form
        if form is None:
            continue
        if relative is not None and bool(form.isRelative) != bool(relative):
            continue
        blocks.append(
            ((isotope, "MF32", covariance.label), np.asarray(form.matrix))
        )
    return blocks


def parameter_covariance_index(
    suite, isotope: Any = None,
) -> Dict[Hashable, Dict[str, Any]]:
    """What each block's rows *are*, keyed the same way as the blocks.

    A drawn sample of a cross-section covariance is interpretable from its
    grid; a drawn sample of a parameter covariance is not interpretable from
    anything the matrix carries. Row 47 is the neutron width of the twelfth
    resonance or it is nothing, so whatever eventually writes the perturbed
    parameters back into File 2 needs this alongside the draw, and it is
    returned separately rather than bolted onto the block so that
    ``draw_samples`` keeps taking plain matrices.
    """
    index: Dict[Hashable, Dict[str, Any]] = {}
    for covariance in getattr(suite, "parameterCovariances", []):
        form = covariance.form
        if form is None:
            continue
        rowData = covariance.rowData
        index[(isotope, "MF32", covariance.label)] = {
            "labels": form.rowLabels(),
            "values": (None if form.parameterValues is None
                       else np.asarray(form.parameterValues)),
            "isRelative": bool(form.isRelative),
            "href": None if rowData is None else rowData.href,
            "energyRange": None if rowData is None else rowData.incidentEnergyBand,
        }
    return index
