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
* **For MF33 the two assemblies are not the same covariance.** Measured on the
  full Fe-56 tape (JEFF-4.0, MT 1/2/4/5/16/102/103): the carrier refines all
  seven components onto one 730-bin global union grid, dimension 5110;
  ``cross_section_covariance_blocks`` keeps each on its native grid — 630, 630,
  124, 124, 124, 631, 124 — and pads to the widest, dimension 4417. Choosing
  between them is an evaluation decision about what is being sampled, so the
  MF33 source migration waits for it and the draw does not.

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
