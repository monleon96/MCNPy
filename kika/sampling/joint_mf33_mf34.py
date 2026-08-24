"""The joint (sigma, a_1..a_L) covariance of an `_a0cross` tape, off ONE parse.

WHY THIS MODULE EXISTS
----------------------
Every sampler in this library draws MF33 and MF34 as two unrelated objects.
That is right for every released evaluation — ENDF-6 has no standard home for a
magnitude/shape covariance, so nobody ships one — and it is **wrong for the
Fe-56 deliverable**, which does ship one, inside MF34's a₀ blocks:

    MT=2, MT1=2, LTT=3, NL=7, sub-subsections (0, l) for l = 1..6,
    LB=6 rectangular, rows = the MF33 magnitude grid, columns = order l's mesh.

`perturb_ENDF_files` cannot see them: `legendre_covariance_blocks` is called
with ``orders=range(1, L+1)`` and that filter is what keeps a₀'s 2317-bin,
150 MeV magnitude axis from meeting the shape axis. `perturb_PENDF_files` never
looks at MF34 at all. And `perturb_ENDF_PENDF_combined` draws the two
separately and pairs them by replica index — which is not merely "the cross term
is missing": pairing two balanced Sobol sets by index makes the mixture 18-21 %
narrower than independence, measured on the PFNS driver.

So the joint is assembled here, ONCE, and handed to `draw_samples` as a single
block. One block means one seed, one decomposition, one draw — the two halves
come out of the same multivariate normal and there is nothing left to pair.

THREE ENSEMBLES OUT OF ONE OBJECT
---------------------------------
``restrict_to`` cuts the assembled matrix down to a principal submatrix:

    "joint"  -> the whole thing, cross included
    "mf34"   -> the shape block alone (sigma frozen)
    "mf33"   -> the magnitude block alone (a_l frozen)

A principal submatrix of a PSD matrix is PSD, so restricting can introduce
nothing. And "mf34" + "mf33" is exactly the joint with the cross set to zero, so
comparing them against "joint" is a one-variable contrast: same code, same
space, same seed policy, same PSD treatment. That is the only arrangement in
which a difference downstream can be attributed to the cross term.

WHAT IT DOES NOT DO
-------------------
It does not rewrite MF3 — the deliverable ships the host's, deliberately — and
it does not touch MT1. The tape states MF33/MT1 and MF33/MT2 with no MT1xMT2
cross block, so drawing both would assert an independence the file does not
state, on top of MT1 being redundant with MT2 anyway. MT1 comes back out of
NJOY's redundancy reconstruction.

⚠ The two halves are relative to DIFFERENT nominals: MF33 to File 3 and
MF34/cross to MF4's own a_l. Both appliers already multiply by the right one,
but the index says so explicitly rather than leaving it to be re-derived.
"""
from __future__ import annotations

from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from ..endf import read_endf
from .mf34_cross import read_mf34_split
from .mf33_sampling import load_mf33_covariance
from .model_blocks import (
    assemble_mf33_mf34_joint,
    legendre_covariance_blocks,
    legendre_covariance_index,
)

__all__ = ["load_joint_mf33_mf34", "restrict_joint", "JOINT_SETS"]

#: The three ensembles, and what each freezes.
JOINT_SETS = ("joint", "mf34", "mf33")


def load_joint_mf33_mf34(
    endf_path: str,
    *,
    mt: int = 2,
    isotope: int = 26056,
    l_max: int = 6,
    require_cross: bool = True,
    energy_unit: str = "eV",
    logger=None,
) -> Tuple[List[Tuple[Hashable, np.ndarray]], Dict[str, Any]]:
    """Read *endf_path* once; return ``([(key, joint)], index)``.

    The return shape is what :func:`kika.sampling.core.draw_samples` takes, so
    the caller does no assembly of its own.

    ``require_cross`` defaults to **True**, unlike the reader's own default.
    A caller reaching this function has asked for the joint by name; handing it
    a block-diagonal matrix because the tape turned out to carry no a₀ sections
    would be a silent downgrade to the very factorisation this exists to
    replace. Pass ``False`` only to build the zero-cross control deliberately.

    Order of operations matters and is not free to change:

    1. MF33 first, because its grid is what every a₀ block's ROW grid is checked
       against. Checking is the whole reason the cross can be trusted to be
       Cauchy-Schwarz-compatible with the marginals it ships beside.
    2. ``read_mf34_split`` next, on the SAME parsed tape. It strips the a₀
       sub-subsections out of the section in place.
    3. ``decodeCovarianceSuite`` last, so the suite — and therefore the shape
       block — is built from the L≥1 remainder. That is why the shape half of
       the joint is bit-for-bit the block ``perturb_ENDF_files`` assembles today.
    """
    orders = list(range(1, int(l_max) + 1))

    # ⚑ TWO PARSES, AND IT IS NOT AN OVERSIGHT.
    #
    # `parse_mf34.py` wraps each MT in `except Exception` and logs the message.
    # `MemoryError` IS an Exception and its `str()` is EMPTY, so a parse that
    # runs out of RAM prints "Error parsing MT2 in MF34:" and hands back a tape
    # with NO MF34 — indistinguishable from a tape that legitimately has none,
    # which JEFF and JENDL do. Measured on this very deliverable at 570 MiB: MF3,
    # MF4 and MF33 perfect, MF34 gone, nothing raised.
    #
    # Reading the two families separately means the two peaks do not add, and
    # MF33 goes FIRST because the a₀ blocks' row axis has to be checked against
    # its grid. The `num_matrices` assertion below is the other half: a log with
    # no errors in it is not evidence that MF34 was read.
    endf33 = read_endf(str(endf_path), mf_numbers=[33])

    # 1. magnitude ---------------------------------------------------------
    cov33, _mf3, grid33, mts_present = load_mf33_covariance(
        str(endf_path), None, [int(mt)], energy_unit=energy_unit,
        logger=logger, endf_obj=endf33,
    )
    if int(mt) not in mts_present:
        raise ValueError(f"{endf_path}: MF33 has no MT{mt}")
    grid33 = np.asarray(grid33, dtype=float)
    idx33 = next(
        (i for i in range(len(cov33.matrices))
         if int(cov33.reaction_rows[i]) == int(mt)
         and int(cov33.reaction_cols[i]) == int(mt)),
        None,
    )
    if idx33 is None:
        raise ValueError(f"{endf_path}: MF33 has no MT{mt} self block")
    c33 = np.asarray(cov33.matrices[idx33], dtype=float)
    if not bool(cov33.is_relative[idx33]):
        raise ValueError(
            f"{endf_path}: MF33/MT{mt} came back absolute. The shape half and "
            f"the cross are relative, and a joint that mixes the two conventions "
            f"is not a covariance of anything. `load_mf33_covariance` converts "
            f"using PENDF File 3 — pass a PENDF if this file needs it."
        )

    del endf33

    # 2. the a_0 blocks, and the in-place strip ----------------------------
    endf = read_endf(str(endf_path), mf_numbers=[34])
    mf34_file = endf.get_file(34)
    if mf34_file is None or int(mt) not in getattr(mf34_file, "sections", {}):
        raise ValueError(
            f"{endf_path}: MF34/MT{mt} did not come back from the parser. If "
            f"the tape does carry it, the cause is almost certainly memory: "
            f"`parse_mf34` degrades a MemoryError to a warning with an empty "
            f"message and returns the tape without MF34. Check the log for "
            f"'Error parsing MT{mt} in MF34:' with nothing after the colon, and "
            f"budget ~4 GB."
        )
    split = read_mf34_split(
        str(endf_path), isotope=int(isotope), mt=int(mt), l_max=int(l_max),
        mf33_grid_ev=grid33, energy_unit=energy_unit,
        require_cross=bool(require_cross), endf=endf,
    )

    # 3. the shape half, from what is left ---------------------------------
    from ..endf.model_adapter import decodeCovarianceSuite
    suite, report = decodeCovarianceSuite(endf)
    blocks = legendre_covariance_blocks(
        suite, mt=[int(mt)], orders=orders, relative=True)
    a_index = legendre_covariance_index(
        suite, mt=[int(mt)], orders=orders, relative=True)
    if not blocks:
        raise ValueError(
            f"{endf_path}: no MF34 blocks with 1 <= L <= {l_max} survived the "
            f"a_0 strip; there is nothing for the cross term to correlate with"
        )
    (a_key, c34), = blocks
    n_mat = len(getattr(split.mf34, "matrices", ()) or ())
    if n_mat == 0:
        raise ValueError(
            f"{endf_path}: MF34 parsed to zero matrices. See the MemoryError "
            f"note above — never accept an empty MF34 implicitly."
        )

    cross_by_order = {
        int(b["l"]): (np.asarray(b["matrix"], dtype=float),
                      np.asarray(b["shape_grid_ev"], dtype=float))
        for b in split.cross
    }

    joint, index = assemble_mf33_mf34_joint(
        c33, grid33, c34, a_index[a_key], cross_by_order)

    index.update(
        source=str(endf_path),
        mt=int(mt),
        isotope=int(isotope),
        l_max=int(l_max),
        energy_unit=energy_unit,
        a_block_key=a_key,
        mf34_info=dict(split.info),
        unsupported=list(getattr(report, "unsupported", ()) or ()),
    )
    key = (int(isotope), "MF33xMF34", int(mt))
    return [(key, joint)], index


def restrict_joint(
    blocks: Sequence[Tuple[Hashable, np.ndarray]],
    index: Dict[str, Any],
    which: str,
) -> Tuple[List[Tuple[Hashable, np.ndarray]], Dict[str, Any]]:
    """Cut the joint down to one family. ``which`` is one of :data:`JOINT_SETS`.

    Returns a fresh ``([(key, matrix)], index)`` pair; the input is untouched.
    The index keeps every field, with ``n_sigma``/``n_a`` zeroed on the side
    that was dropped, so a downstream applier can tell from the index alone
    which half it is expected to write and does not have to infer it from a
    shape.

    ``which="joint"`` returns the input unchanged rather than copying it — the
    whole matrix is the object, and a copy of a 1.4 GB array to say "no change"
    is not a service.
    """
    if which not in JOINT_SETS:
        raise ValueError(f"which must be one of {JOINT_SETS}, got {which!r}")
    (key, joint), = list(blocks)
    m0, n_a = int(index["n_sigma"]), int(index["n_a"])
    if which == "joint":
        return [(key, joint)], dict(index)
    out = dict(index)
    if which == "mf33":
        sub = joint[:m0, :m0]
        out.update(n_a=0, dimension=m0, cross_orders=[], has_cross=False,
                   triplets=[], stride=0, widths={}, grids={})
    else:
        sub = joint[m0:, m0:]
        out.update(n_sigma=0, dimension=n_a, cross_orders=[], has_cross=False,
                   sigma_grid_ev=np.asarray(index["sigma_grid_ev"])[:0])
    out["restricted_to"] = which
    return [((key, which), np.ascontiguousarray(sub)), ], out
