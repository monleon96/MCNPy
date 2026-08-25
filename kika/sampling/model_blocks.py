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
:mod:`kika._covariance_forms`, the one thing they do import, duck-types it for
the same reason and imports nothing itself.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from kika._covariance_forms import require_single_matrix

__all__ = ["covariance_suite_blocks", "parameter_covariance_blocks",
           "parameter_covariance_index", "legendre_covariance_blocks",
           "legendre_covariance_index", "cross_section_covariance_blocks",
           "cross_section_covariance_index", "assemble_joint", "UNION_MODES",
           "CROSS_SECTION_MF"]


def covariance_suite_blocks(suite, isotope: Any = None, mt: int = 18):
    """A ``CovarianceSuite``'s §25.2 sections → the ``(key, matrix)`` the core wants.

    One block per section. Deliberately *not* concatenated: the sections of one
    MF35 file have different orders (84, 641, 641, 641, 641 on ENDF/B-VIII.1
    U-235), so anything that assembled them at a common dimension would produce
    a malformed matrix without saying so.

    **This is the right shape for MF35 and the wrong shape for MF34**, and the
    difference is not about the formats but about what the sections *are*. MF35's
    bands are separate quantities that happen to live in one file. MF34's
    Legendre orders are components of one distribution, and a file may state the
    covariance *between* two of them — a section whose ``rowData`` and
    ``columnData`` differ. Handing those to a sampler one at a time offers an
    off-diagonal corner as if it were a covariance (``docs/library/library-gaps.md`` D8)
    and drops the correlation besides. :func:`legendre_covariance_blocks` is the
    MF34 entry point for that reason; this function is not a general enumerator
    and callers should not reach for it by default.

    It also filters nothing: every section of the suite is emitted, MF33 and
    MF31 included, all keyed at ``mt``. That is tolerable only because the MF35
    caller hands it a suite it built itself, one MT wide.
    """
    blocks = []
    for index, section in enumerate(suite):
        form = require_single_matrix(
            getattr(section, "form", None),
            f"covariance section {getattr(section, 'label', index)!r}",
        )
        blocks.append(((isotope, mt, index), np.asarray(form.matrix)))
    return blocks


def _lift_matrix(src_grid: np.ndarray, dst_grid: np.ndarray) -> np.ndarray:
    """Map a covariance from its own energy bins onto a finer union of them.

    ``A[g, j] = 1`` where union bin *g* falls inside source bin *j*, so that
    ``A Σ Aᵀ`` restates Σ on the union grid: a source bin split in two by
    another section's boundaries becomes two union bins carrying that bin's
    variance and perfectly correlated with each other, which is what the file
    says about them.

    **A union bin outside the source's span gets an all-zero row.** The union is
    taken over every section mentioning a Legendre order, so it refines any one
    section's grid only when all of them share a range — and real files do not.
    Assuming otherwise was wrong at both ends (``docs/library/library-gaps.md`` D9 and
    D10): past the source's last boundary the cursor ran off the end and raised,
    which the shipped ``_a0cross`` tape does; below its first, the cursor sat at
    0 and handed out that section's first bin, which the shipped multigroup tape
    does, silently.

    Zero is right and not merely safe. A section states a covariance over its own
    bins and says *nothing* about a bin outside them, and nothing is zero
    covariance — not the value of the nearest bin it does cover, which is what
    clamping the cursor would assert.

    **This is byte-identical to
    :meth:`~kika.cov.legendre_covariance.LegendreCovariance._lift_matrix`**,
    which was fixed rather than routed around, and the duplication is knowing and
    temporary — the migration exists to stop this package depending on that class,
    so importing its private method would defeat the point.
    ``test_a_union_bin_below_a_section_gets_zero_not_its_first_bin`` asserts both
    against the same expected matrix and is the drift detector;
    ``refactor-backlog.md`` names them as the pair to collapse.
    """
    src_grid = np.asarray(src_grid, dtype=float)
    dst_grid = np.asarray(dst_grid, dtype=float)
    Gs, Gd = len(src_grid) - 1, len(dst_grid) - 1
    A = np.zeros((Gd, Gs), dtype=float)
    j = 0
    for g in range(Gd):
        eL, eH = dst_grid[g], dst_grid[g + 1]
        while j + 1 < len(src_grid) and src_grid[j + 1] <= eL + 1e-12:
            j += 1
        if j >= Gs or eH <= src_grid[0] + 1e-12 or eL >= src_grid[-1] - 1e-12:
            continue
        A[g, j] = 1.0
    return A


#: How :func:`assemble_joint` decides each component's bin structure. The two
#: modes are not two layouts of one covariance — **they are two different
#: covariances**, and which is wanted is a property of the quantity being
#: assembled rather than a preference.
#:
#: ``"per-component"``
#:     One union per key, over the grids of the blocks that mention it.
#:     Components keep their own widths and are zero-padded to the widest.
#:     Correct when the components genuinely live on different grids and the
#:     evaluation says nothing outside each one's span. This is
#:     :attr:`~kika.cov.legendre_covariance.LegendreCovariance.covariance_matrix`'s
#:     layout, and MF34's components — Legendre orders of one distribution —
#:     share a grid by construction, so it is also the *only* layout there.
#:
#: ``"global"``
#:     One union across every grid in the assembly, handed to every key. No
#:     padding; every row is stated. This is what
#:     ``mf33_sampling._build_union_grid`` builds and what
#:     :class:`~kika.cov.cross_section_covariance.CrossSectionCovariance`
#:     carries, i.e. **what the MF33 pipeline ships today**.
UNION_MODES = ("per-component", "global")


def _union_grids(entries, atol: float = 1e-12,
                 union: str = "per-component") -> Dict[Hashable, np.ndarray]:
    """The merged bin structure each key is stated on. See :data:`UNION_MODES`.

    **Measured on the full Fe-56 tape, 2026-08-13**, MF33 for MT 1/2/4/5/16/102/103
    with native grids of 630, 630, 124, 124, 124, 631 and 124 bins:

    | mode | dimension | zero-padded rows |
    |---|---|---|
    | ``per-component`` | 7 x 631 = **4417** | 2236 |
    | ``global`` | 7 x 730 = **5110** | 0 (2037 rows are inert, which is different) |

    The 5110 is bit-identical to the shipped carrier's matrix. That is the whole
    reason this parameter exists rather than a decision having been taken here.
    """
    if union not in UNION_MODES:
        raise ValueError(f"union must be one of {UNION_MODES}, got {union!r}")

    perKey: Dict[Hashable, list] = {}
    for rowKey, colKey, _matrix, rowGrid, colGrid in entries:
        perKey.setdefault(rowKey, []).extend(np.asarray(rowGrid, dtype=float))
        perKey.setdefault(colKey, []).extend(np.asarray(colGrid, dtype=float))

    if union == "global":
        # Every key gets the same pooled grid. Built by pooling the *per-key*
        # lists rather than re-walking `entries`, so the two modes cannot
        # disagree about which grids were in scope -- only about how they merge.
        pooled = [x for values in perKey.values() for x in values]
        perKey = {key: pooled for key in perKey}

    merged: Dict[Hashable, np.ndarray] = {}
    for key, values in perKey.items():
        grid = np.unique(np.asarray(values, dtype=float))
        kept = [grid[0]]
        for x in grid[1:]:
            if not np.isclose(x, kept[-1], rtol=0.0, atol=atol):
                kept.append(x)
        merged[key] = np.asarray(kept, dtype=float)
    return merged


def assemble_joint(entries, atol: float = 1e-12,
                   union: str = "per-component"):
    """Sections that share a quantity → the one matrix they are partitions of.

    *entries* is a sequence of ``(rowKey, colKey, matrix, rowGrid, colGrid)``.
    Keys are whatever identifies a component of the joint quantity — for MF34
    an ``(isotope, MT, L)`` triplet, and for the cross-*reaction* case that P3
    still owes, an ``(isotope, MT)`` pair. Returns ``(keys, joint, stride)``
    with *keys* in the order their rows appear.

    **Why this exists at all.** A file states a covariance in pieces: a square
    block per component, plus a rectangular one for each pair it correlates.
    Those pieces are not separately samplable — an off-diagonal block is not a
    covariance, its diagonal is not a variance, and its correlations are not
    bounded by 1. Only the assembled matrix is the thing the evaluation is
    talking about. Sampling the pieces one at a time draws from a distribution
    the file never described, and does it without failing.

    **The stride is uniform and it is deliberate.** Every component gets
    ``stride = max_G`` rows even where its own union grid is shorter, so short
    components carry trailing all-zero rows. Those are inert directions — a
    decomposition returns nothing for them — and the alternative, packing
    components at their own widths, makes a row's meaning depend on every
    component before it. This reproduces
    :attr:`~kika.cov.legendre_covariance.LegendreCovariance.covariance_matrix`,
    which is the layout the MF34 pipeline's existing samples were drawn in.

    *union* selects the bin structure — see :data:`UNION_MODES`. Under
    ``"global"`` every component is refined onto one pooled grid, so
    ``widths[key] == stride`` throughout and **there is no padding at all**;
    the uniform-stride paragraph above still holds, it simply has nothing to do.
    That is the mode that reproduces
    :attr:`~kika.cov.cross_section_covariance.CrossSectionCovariance.covariance_matrix`,
    and it is the MF33 pipeline's shipped layout. The default is
    ``"per-component"`` because MF34 — the migrated path, whose samples exist —
    was drawn that way.
    """
    entries = list(entries)
    if not entries:
        return [], np.zeros((0, 0), dtype=float), 0

    unions = _union_grids(entries, atol=atol, union=union)
    keys = sorted(unions)
    index = {key: i for i, key in enumerate(keys)}
    widths = {key: len(grid) - 1 for key, grid in unions.items()}
    stride = max(widths.values())

    # A repeated (rowKey, colKey) would be written twice into the same slot and
    # the second would win, silently. Neither this nor the carrier has a defined
    # answer for it -- summing and overwriting are both defensible and they
    # disagree -- so it is refused rather than guessed at. It does not arise on
    # any file to hand: both `to_ang_covmat` and `decodeMF34MT` aggregate a
    # pair's sub-subsections into one matrix before this sees them.
    seen = set()
    for rowKey, colKey, *_ in entries:
        if (rowKey, colKey) in seen:
            raise ValueError(
                f"two sections both state the block ({rowKey}, {colKey}); "
                f"assemble_joint has no defined way to combine them -- they "
                f"must be aggregated into one matrix before they get here"
            )
        seen.add((rowKey, colKey))

    joint = np.zeros((len(keys) * stride,) * 2, dtype=float)
    for rowKey, colKey, matrix, rowGrid, colGrid in entries:
        i, j = index[rowKey], index[colKey]
        Ar = _lift_matrix(rowGrid, unions[rowKey])
        Ac = _lift_matrix(colGrid, unions[colKey])
        lifted = Ar @ np.asarray(matrix, dtype=float) @ Ac.T

        r0, r1 = i * stride, i * stride + widths[rowKey]
        c0, c1 = j * stride, j * stride + widths[colKey]
        joint[r0:r1, c0:c1] = lifted
        if i != j:
            joint[c0:c1, r0:r1] = lifted.T
    return keys, joint, stride


def _endf_mfmt(link) -> Optional[Tuple[int, int]]:
    """``(MF, MT)`` from an ``ENDF_MFMT``, **whichever separator wrote it**.

    ``"34/2"`` and ``"34,2"`` both give ``(34, 2)``, and that is the whole point
    of the function. GNDS §25.2.3 (p. 363) defines the attribute as a *comma*
    separated pair and all 270 covariance files of ENDF/B-VIII.1-GNDS agree;
    kika's ENDF adapter writes a slash. A suite decoded from GNDS therefore
    matched none of this module's filters and — the part that made it worth
    fixing — **did not raise**: ``startswith("34/")`` returned ``False``,
    ``_mf34_entries`` returned ``[]`` and the caller assembled zero blocks.
    See ``docs/library/gnds_endf_conflicts.md`` §3.1 and §7.1.

    ``None`` for a link that has no ``ENDF_MFMT`` or whose value is not a pair
    of integers; the callers decide whether that is a skip or an error, because
    they mean different things by it.

    Duck-typed rather than imported from
    ``kika.endf.model_adapter.covariances``, which has the same two helpers.
    That import is what ``test_nothing_imports_the_adapter`` forbids and it is
    right to: reaching the adapter from ``kika.sampling`` pulls the whole GNDS
    model onto the sampler's import path, and this module's own docstring
    promises it duck-types the suite for exactly that reason. Two four-line
    readers of a documented spec field is the cheaper of the two costs.
    ``DataLink._mfmtParts`` is the model-side twin and does the same thing.
    """
    raw = getattr(link, "ENDF_MFMT", None) if link is not None else None
    if not raw:
        return None
    try:
        mf, mt = (int(part) for part in str(raw).replace("/", ",").split(",", 1))
    except ValueError:
        return None
    return mf, mt


def _is_endf_mf(link, mf: int) -> bool:
    """Does this link belong to *mf*? The separator-agnostic ``startswith``."""
    parts = _endf_mfmt(link)
    return parts is not None and parts[0] == mf


def _endf_mt(link) -> int:
    """The MT an ``ENDF_MFMT`` names — ``"34/2"`` and ``"34,2"`` → 2."""
    parts = _endf_mfmt(link)
    if parts is None:
        raise ValueError("a covariance link with no ENDF_MFMT cannot be placed")
    return parts[1]


def _legendre_order(link) -> int:
    """The Legendre order a link is sliced at (§25.2.5-6)."""
    slices = getattr(getattr(link, "slices", None), "slices", None) or ()
    for entry in slices:
        if entry.domainValue is not None:
            return int(entry.domainValue)
    raise ValueError(
        f"the link to {getattr(link, 'href', None)!r} carries no slice, so the "
        f"Legendre order it is about is unknown"
    )


def _selection(value) -> Optional[frozenset]:
    """``None`` / empty / ``[-1]`` mean "everything"; anything else is a set.

    ``[-1]`` is the pipeline's own spelling of "all of them"
    (``resolve_signed_request``), and it reaches here through
    ``perturb_ENDF_files``' ``legendre_coeffs``. Reading it as a literal order
    would select nothing and return an empty joint.
    """
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return frozenset({int(value)})
    values = [int(v) for v in value]
    if not values or values == [-1]:
        return None
    return frozenset(values)


def _mf34_entries(suite, mt=None, orders=None, relative=None):
    """The MF34 sections of *suite*, as ``assemble_joint`` entries.

    Filters on ``ENDF_MFMT``, which is what keeps an MF33 section in the same
    suite from being swept in — ``covariance_suite_blocks`` does not, and on the
    committed micro-tape it emits ``MF33-MT2`` among the MF34 blocks.

    *mt* and *orders* accept a single value, a sequence, or ``None`` for
    everything. **Both sides of a section must pass both tests**, which is the
    rule :func:`~kika.sampling.endf_perturbation._filter_mf34_covariance`
    applies and the only one that is coherent: a cross block whose row survives
    and whose column does not is an off-diagonal corner with nothing to be
    off-diagonal to, and placing it would state a correlation with a component
    that is not in the matrix.

    **The order filter is load-bearing, not a convenience.** It is what keeps
    L=0 out, and on an ``_a0cross`` tape the a₀ sections are the ones carrying
    the MF33↔MF34 cross term on the magnitude grid — 2346 bins to 150 MeV
    against the shape grid's 703 to 20 MeV. ``perturb_ENDF_files`` passes
    ``range(1, max_degree + 1)``, so excluding a₀ is what has always kept those
    two grids from meeting; ``_apply_factors_to_mf4_legendre`` documents that it
    relies on it.

    *relative* filters by form, as it does in
    :func:`parameter_covariance_blocks`: ``True`` for the relative sections,
    ``False`` for the absolute ones, ``None`` for both. **A sampling caller
    wants ``True``.** ``load_mf34_covariance`` has always dropped absolute MF34
    sections before handing anything to a draw, because the appliers *multiply*
    by what comes back and an absolute covariance does not describe a factor.
    The default is ``None`` so that assembling a file in order to look at it
    still shows everything the file states, and so that the byte-identity gate
    against ``to_ang_covmat`` — which does not drop them either — keeps
    comparing like with like.
    """
    mts = _selection(mt)
    ls = _selection(orders)

    entries = []
    for section in getattr(suite, "covarianceSections", suite):
        rowData, colData = section.rowData, section.columnData
        if not _is_endf_mf(rowData, 34):
            continue
        colData = rowData if colData is None else colData

        form = require_single_matrix(
            getattr(section, "form", None),
            f"MF34 covariance section {getattr(section, 'label', '?')!r}",
        )

        if relative is not None and bool(form.isRelative) != bool(relative):
            continue

        if mts is not None and not (
            _endf_mt(rowData) in mts and _endf_mt(colData) in mts
        ):
            continue
        if ls is not None and not (
            _legendre_order(rowData) in ls and _legendre_order(colData) in ls
        ):
            continue

        za = int(getattr(section.provenance, "za", None) or 0)
        rowGrid = np.asarray(form.rowGrid, dtype=float)
        colGrid = (rowGrid if form.columnGrid is None
                   else np.asarray(form.columnGrid, dtype=float))
        # LegendreCovariance keeps one grid per section and lifts both sides
        # with it. The model keeps two. They agree on every MF34 seen so far,
        # and where they would not, the model is the one telling the truth --
        # so use it, and say so, rather than reproduce the carrier's assumption
        # silently and call the difference a migration regression.
        if not np.array_equal(rowGrid, colGrid):
            warnings.warn(
                f"MF34 section {section.label!r} states different row and "
                f"column energy grids ({rowGrid.size} vs {colGrid.size} "
                f"boundaries). LegendreCovariance stores only the row grid and "
                f"lifts both sides with it, so this block is assembled "
                f"differently here -- deliberately, and the difference is real.",
                stacklevel=2,
            )
        entries.append((
            (za, _endf_mt(rowData), _legendre_order(rowData)),
            (za, _endf_mt(colData), _legendre_order(colData)),
            np.asarray(form.matrix, dtype=float),
            rowGrid,
            colGrid,
        ))
    return entries


def legendre_covariance_blocks(
    suite, isotope: Any = None, mt=None, orders=None, relative=None,
    atol: float = 1e-12,
) -> List[Tuple[Hashable, np.ndarray]]:
    """A ``CovarianceSuite``'s MF34 sections → the one joint block they describe.

    The MF34 counterpart of :func:`covariance_suite_blocks`, and deliberately
    not shaped like it: **one block, not one per section.** MF34's sections are
    partitions of a single covariance over ``(isotope, MT, Legendre order)``,
    and a file that correlates two orders states that as a section whose row and
    column links are sliced at different ``domainValue``s. Emitted separately,
    such a section is an off-diagonal corner offered as a covariance; assembled,
    it is the off-diagonal block it always was.

    On the committed Fe-56 micro-tape the difference is not marginal: the
    assembled matrix is 6×6 over ``[(26056,2,1), (26056,2,2)]`` with the
    cross-order entries reaching 19 % of the largest entry in the matrix.

    Returns the usual ``(key, matrix)`` pairs so ``draw_samples`` needs no
    changes — a list of one. :func:`legendre_covariance_index` says what the
    rows of that one block are, which is what a caller needs to write a drawn
    perturbation back onto MF34.

    *mt*, *orders* and *relative* select which components enter the joint; see
    :func:`_mf34_entries` for why both sides of a section have to pass, why the
    order filter is what keeps a₀ and its magnitude grid out, and why a caller
    that is about to *sample* wants ``relative=True``.
    """
    entries = _mf34_entries(suite, mt=mt, orders=orders, relative=relative)
    keys, joint, _stride = assemble_joint(entries, atol=atol)
    if not keys:
        return []
    return [((isotope, "MF34", tuple(keys)), joint)]


def legendre_covariance_index(
    suite, isotope: Any = None, mt=None, orders=None, relative=None,
    atol: float = 1e-12,
) -> Dict[Hashable, Dict[str, Any]]:
    """What the rows of :func:`legendre_covariance_blocks`' block are.

    The sibling of :func:`parameter_covariance_index`, and needed for the same
    reason: a drawn sample is a flat vector, and nothing in the matrix says that
    row 4 is the second energy bin of L=2. Returned separately so the block
    stays a plain matrix.

    ``triplets`` are in row order, each occupying ``stride`` rows; ``grids``
    gives the union bin boundaries per triplet, and ``widths`` how many of that
    triplet's ``stride`` rows are real rather than the zero padding the uniform
    stride implies.

    **The layout is derivable from the grids alone**, so this does not assemble
    anything. It looked natural to call :func:`assemble_joint` and read the
    shape off the result; on the Fe-56 ``_a0cross`` tape that is a second 2.16 GB
    allocation and 30 s to learn a stride, and calling both functions in the
    obvious order took the box to zero free memory.
    """
    entries = _mf34_entries(suite, mt=mt, orders=orders, relative=relative)
    if not entries:
        return {}
    unions = _union_grids(entries, atol=atol)
    keys = sorted(unions)
    widths = {key: len(unions[key]) - 1 for key in keys}
    stride = max(widths.values())
    return {
        (isotope, "MF34", tuple(keys)): {
            "triplets": list(keys),
            "stride": stride,
            "grids": {key: unions[key] for key in keys},
            "widths": widths,
            "dimension": len(keys) * stride,
        }
    }


#: The MFs :func:`_cross_section_entries` serves — the ones whose components are
#: **reactions on an energy grid**, keyed by ``(ZA, MT)``.
#:
#: MF31 is here because §31.1 makes the MT452/455/456 formats *"directly
#: analogous to those of File 33"*: same records, same grids, same
#: ``_build_union_grid`` (``mf31_sampling.py`` imports MF33's), and the same
#: ``CrossSectionCovariance`` carrier. What differs is what the covariance is
#: *about* — a multiplicity rather than a cross section — and that lives in the
#: ``href``, which this function does not read.
#:
#: MF34 is **not** here and cannot be: its key carries a third coordinate, the
#: Legendre order. MF35's is a band. Both have their own entry builders.
CROSS_SECTION_MF = (31, 33)


def _cross_section_entries(suite, mf: int = 33, mt=None, relative=None):
    """The MF31 or MF33 sections of *suite*, as :func:`assemble_joint` entries.

    The cross-*reaction* sibling of :func:`_mf34_entries`, and the same shape of
    problem one level out: where MF34 partitions one distribution over Legendre
    orders, MF33 partitions one covariance over **reactions**. A file that
    correlates MT2 with MT102 states it as a section whose row and column links
    name different reactions, and ``decodeMF33MT`` already emits those — one
    ``CovarianceSection`` per ``(row MT, column MT)`` block, cross blocks
    included.

    So the component key is the ``(ZA, MT)`` pair, and there is no third
    coordinate: the Legendre order that makes MF34's key a triplet has no MF33
    counterpart. That is the whole difference between the two functions.

    *mf* selects which file — see :data:`CROSS_SECTION_MF`. **One function
    rather than two because the two are the same records**, not because the two
    were similar enough to merge: `mf31_sampling.py` imports MF33's
    `_build_union_grid` and packs into the same carrier, so a separate entry
    builder would have been a copy with one integer changed.

    **The key is the pair ``_get_param_pairs`` uses, deliberately.**
    :attr:`~kika.cov.cross_section_covariance.CrossSectionCovariance.covariance_matrix`
    orders its blocks by ``sorted(pairs, key=lambda p: (p[0], p[1]))`` — isotope
    then reaction — and ``assemble_joint`` orders by ``sorted(keys)``, which on
    a tuple of two integers is the same order. The layouts therefore agree
    without either side being told about the other, and
    ``test_the_mf33_assembly_reproduces_the_carrier`` is what holds that.

    *mt* and *relative* filter as they do for MF34, and for the same reasons:
    both sides of a section must pass, because a cross block whose row survives
    and whose column does not is an off-diagonal corner with nothing to be
    off-diagonal to; and a caller that is about to *sample* wants
    ``relative=True``, because the appliers multiply by what comes back.
    """
    if mf not in CROSS_SECTION_MF:
        raise ValueError(
            f"mf must be one of {CROSS_SECTION_MF}, got {mf}. MF34's components "
            f"carry a Legendre order and MF35's a band, so their keys are not "
            f"(ZA, MT) pairs; use _mf34_entries or covariance_suite_blocks."
        )
    mts = _selection(mt)

    entries = []
    for section in getattr(suite, "covarianceSections", suite):
        rowData, colData = section.rowData, section.columnData
        if not _is_endf_mf(rowData, mf):
            continue
        colData = rowData if colData is None else colData

        form = require_single_matrix(
            getattr(section, "form", None),
            f"MF{mf} covariance section {getattr(section, 'label', '?')!r}",
        )

        if relative is not None and bool(form.isRelative) != bool(relative):
            continue

        if mts is not None and not (
            _endf_mt(rowData) in mts and _endf_mt(colData) in mts
        ):
            continue

        za = int(getattr(section.provenance, "za", None) or 0)
        rowGrid = np.asarray(form.rowGrid, dtype=float)
        colGrid = (rowGrid if form.columnGrid is None
                   else np.asarray(form.columnGrid, dtype=float))
        entries.append((
            (za, _endf_mt(rowData)),
            (za, _endf_mt(colData)),
            np.asarray(form.matrix, dtype=float),
            rowGrid,
            colGrid,
        ))
    return entries


def cross_section_covariance_blocks(
    suite, isotope: Any = None, mt=None, relative=None, atol: float = 1e-12,
    union: str = "global", mf: int = 33,
) -> List[Tuple[Hashable, np.ndarray]]:
    """A ``CovarianceSuite``'s MF33 (or MF31) sections → the one joint block.

    The model-side replacement for
    :attr:`~kika.cov.cross_section_covariance.CrossSectionCovariance.covariance_matrix`,
    and the assembly the MF33 migration needs before it can move a draw. Like
    :func:`legendre_covariance_blocks` it returns **one block, not one per
    section**, because a cross-MT block is not separately samplable.

    **Unstated blocks are zero here and NaN there, and that is equivalence
    rather than a change.** The carrier allocates ``np.full((N, N), np.nan)``
    and writes only the blocks the file carries, so a pair of MTs the
    evaluation does not correlate comes back NaN; ``generate_samples`` then
    replaces every non-finite entry with 0 immediately before decomposing. The
    matrix that is actually decomposed today is therefore the zero-filled one,
    which is what :func:`assemble_joint` builds directly.

    Measured, not reasoned about: on ``micro_fe56_mf33.endf`` — real JEFF-4.0
    MF33 for MT4 and MT16, whose cross block the evaluation does not state —
    the carrier's matrix is **50 % NaN**, and the drawn factors are finite
    regardless.

    What that fill *does* change is everything between the two points, and
    ``docs/library/library-gaps.md`` D11 is the entry: on a NaN matrix every eigenvalue
    solver fails, ``_safe_min_eigvalsh`` returns a fabricated ``0.0``, and
    ``autofix`` accepts unconditionally because ``0.0 >= accept_tol`` for any
    negative tolerance. Assembling zeros here makes that analysis start working,
    which is an improvement and therefore **not** this function's business to
    smuggle in — it moves numbers on any run that enables autofix, and it needs
    its own before/after.

    **The grid, and why the default is ``"global"`` — settled 2026-08-13 by
    measurement.** MF33's components are *different reactions*, and an evaluator
    has no reason to grid them alike: on the full Fe-56 tape MT1/2 are on 630
    bins, MT102 on 631 and MT4/5/16/103 on 124. So the two :data:`UNION_MODES`
    genuinely disagree — 7 x 730 = 5110 rows against 7 x 631 = 4417, of which
    2236 are padding. Nothing about MF34 prepared for this: its components are
    Legendre orders of one distribution and share a grid by construction.

    ``sampling_migration_roadmap.md`` deferred the MF33 source migration on that
    disagreement, reading it as an evaluation decision that "moves every drawn
    column either way". **It is a decision about the *improvement*, not about
    the *migration*.** Measured on that same tape: with ``union="global"`` this
    function's matrix is **bit-identical** to the carrier's zero-filled
    ``covariance_matrix``, all 5110 x 5110 of it, on a pooled grid identical to
    ``_build_union_grid``'s and with the key order ``_get_param_pairs()`` gives.

    So the default is the shipped layout, and moving the source onto the model
    changes no number — *equivalent first, then improve*. Choosing
    ``"per-component"`` instead is the improvement, and it keeps its own
    before/after.

    *mf*, *mt* and *relative* select which components enter the joint; see
    :func:`_cross_section_entries`. **The MF is in the block key**, so an MF31
    and an MF33 assembly of the same isotope cannot collide in a caller's dict —
    they are covariances of different quantities that happen to share a key
    shape.
    """
    entries = _cross_section_entries(suite, mf=mf, mt=mt, relative=relative)
    keys, joint, _stride = assemble_joint(entries, atol=atol, union=union)
    if not keys:
        return []
    return [((isotope, f"MF{mf}", tuple(keys)), joint)]


def cross_section_covariance_index(
    suite, isotope: Any = None, mt=None, relative=None, atol: float = 1e-12,
    union: str = "global", mf: int = 33,
) -> Dict[Hashable, Dict[str, Any]]:
    """What the rows of :func:`cross_section_covariance_blocks`' block are.

    The MF33 sibling of :func:`legendre_covariance_index`, and what replaces
    ``extract_mt_param_blocks`` (``kika/sampling/mf33_sampling.py``) once the
    draw moves: that function returns ``MT -> slice`` by multiplying an index by
    ``num_groups``, which is only right while every component is the same width.
    ``widths`` is what says when it is not.

    ``pairs`` are in row order, each occupying ``stride`` rows; ``grids`` gives
    the union bin boundaries per pair, and ``widths`` how many of that pair's
    ``stride`` rows are real rather than the zero padding a uniform stride
    implies.

    Derived from the grids alone, so nothing is assembled — the same reason
    :func:`legendre_covariance_index` does not call :func:`assemble_joint`.

    *union* **must match what** :func:`cross_section_covariance_blocks` **was
    called with**, and shares its default for that reason: the index says which
    rows of the block mean what, and an index built on one bin structure
    describing a block built on the other is wrong everywhere without being
    wrong anywhere visible. Under the ``"global"`` default every ``width``
    equals ``stride`` and ``grids`` holds the one pooled grid seven times, which
    is the shape ``extract_mt_param_blocks`` assumes today.
    """
    entries = _cross_section_entries(suite, mf=mf, mt=mt, relative=relative)
    if not entries:
        return {}
    unions = _union_grids(entries, atol=atol, union=union)
    keys = sorted(unions)
    widths = {key: len(unions[key]) - 1 for key in keys}
    stride = max(widths.values())
    return {
        (isotope, f"MF{mf}", tuple(keys)): {
            "pairs": list(keys),
            "stride": stride,
            "grids": {key: unions[key] for key in keys},
            "widths": widths,
            "dimension": len(keys) * stride,
        }
    }


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
