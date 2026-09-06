"""The legacy carrier, as blocks — the other half of the seam.

:mod:`kika.sampling.model_blocks` reads a ``CovarianceSuite`` and commits, in
its own docstring, to never importing the model side. This module is its
opposite number: it reads a :class:`~kika.cov.cross_section_covariance.CrossSectionCovariance`
and presents it in the same shape, so the draw can be migrated without the
source having to move at the same time.

**Why both exist, rather than one winning.** Two reasons, and only the second
is temporary.

* **ACE has nowhere else to get a covariance.** Its matrix arrives from a
  multigroup file — COVFIL, COVERX or BOXER — through
  :func:`kika.sampling.utils.load_covariance`, and there is no
  multigroup-to-``CovarianceSuite`` bridge. Building one would mean teaching
  the model about three NJOY formats to gain nothing the ACE path can use,
  since for ACE the covariance is a separate input file and what matters is
  that the carrier is shared, not where it came from.
* ~~**For MF33 the two assemblies are not the same covariance.**~~ **Superseded
  2026-08-13, and this paragraph outlived it by a fortnight.** What it measured
  is right — the carrier refines all seven components of the full Fe-56 tape
  onto one 730-bin global union (dimension 5110) where a *per-component* union
  keeps each on its native grid (630, 630, 124, 124, 124, 631, 124) and pads to
  the widest (4417) — and what it concluded from it is not. ``assemble_joint``
  took a per-component union because MF34 needs one; asked for a global union it
  reproduces the carrier **bit for bit**, all 26 112 100 cells, on a pooled grid
  identical to ``_build_union_grid``'s. So ``union='global'`` is the default and
  the MF33 source migration was never blocked on the choice: the global-vs-
  per-component question is the *improvement*, with its own before/after.
  ``gnds_endf_conflicts.md`` §10.1. The source migration landed as
  :func:`kika.sampling.mf33_sampling.loadCrossSectionBlocks`; what still reaches
  this module is the **call sites**, and why is the next paragraph.

* **``apply_legacy_autofix`` takes a carrier, and it is in the middle of both
  live call sites.** ``pendf_perturbation`` and ``nubar_perturbation`` call it
  between assembling the covariance and drawing, and it reaches
  ``CrossSectionCovariance.fix_covariance`` — so a call site cannot be moved off
  the carrier until autofix is retired or ported, which the sampling roadmap
  makes its own change, in kika-app first (``autofix='soft'`` is every kika-app
  default; ``None`` is every kika one, and then this function is a no-op). Found
  when the M1 source landed, and it is why the source can be equivalent and the
  call sites still be here.

**The NaN fill is left as the carrier states it.** An unstated cross block
comes back ``np.nan`` and stays that way until :func:`~kika.sampling.core.draw_samples`
zeroes it immediately before decomposing — which is exactly where
``generate_samples`` zeroes it today. Filling here instead would be defect D11's
fix, and that fix changes what ``autofix`` decides, so it is a separate change
with its own before/after rather than a side effect of an adapter.
"""
from __future__ import annotations

from typing import Any, Dict, Hashable, List, Optional, Tuple

import numpy as np

__all__ = [
    "cross_section_carrier_blocks",
    "cross_section_carrier_index",
]


def _key(cov, isotope: Optional[int]) -> Hashable:
    """The block key, shaped like ``cross_section_covariance_blocks``'.

    The third element is the ordered pair list, so a key identifies not just
    "the MF33 block of this isotope" but which components it was assembled
    from — two runs over different MT selections cannot collide.
    """
    return (isotope, "MF33", tuple(cov._get_param_pairs()))


def cross_section_carrier_blocks(
    cov, isotope: Optional[int] = None
) -> List[Tuple[Hashable, np.ndarray]]:
    """The carrier's super-matrix, as the one block it is.

    One block, not one per reaction: the whole point of the ``(N·G)×(N·G)``
    assembly is that MT2 and MT102 are drawn jointly, so splitting it would
    discard the cross-reaction correlation it exists to carry.
    """
    return [(_key(cov, isotope), np.asarray(cov.covariance_matrix, dtype=float))]


def cross_section_carrier_index(
    cov, isotope: Optional[int] = None
) -> Dict[Hashable, Dict[str, Any]]:
    """Where each ``(isotope, MT)`` component sits in the flat layout.

    The same dict shape :func:`kika.sampling.model_blocks.cross_section_covariance_index`
    returns, so a caller can be handed either. ``widths`` equals ``stride`` for
    every component here and cannot do otherwise — the carrier is in shared-grid
    mode, one grid for everything, which is precisely the difference the module
    docstring measures.
    """
    pairs = list(cov._get_param_pairs())
    stride = int(cov.num_groups)
    grid = list(cov.energy_grid)
    return {
        _key(cov, isotope): {
            "pairs": pairs,
            "stride": stride,
            "grids": {pair: grid for pair in pairs},
            "widths": {pair: stride for pair in pairs},
            "dimension": len(pairs) * stride,
        }
    }
