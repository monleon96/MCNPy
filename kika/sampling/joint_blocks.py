"""A perturbation request over several quantities, and what of it must be drawn together.

The pipelines this is written to replace each perturb one thing.
``mf33_sampling`` does cross sections, ``endf_perturbation`` does angular
distributions, ``mf35_sampling`` does the fission spectrum, ``nubar_perturbation``
does the multiplicity, and ``combined_perturbation`` runs two of them side by
side and pairs their samples by index. That pairing is the part that does not
survive contact with the question this module exists to answer: **is sample *i*
of the cross section and sample *i* of the angular distribution one realisation
of the evaluation, or two?**

It is one realisation only if the evaluation says the two quantities are
uncorrelated. When it says otherwise -- and MF34 says otherwise the moment a
file carries the L=0 sections that state the magnitude alongside the shape --
drawing them apart produces an ensemble whose correlation structure is not the
file's, and nothing in either run reports it. ``combined_perturbation``'s own
docstring records the assumption ("MF33 <-> MF34 cross-correlations are
structurally zero, so independent draws are correct"), which is true of the
tapes it was written for and is a property of those tapes rather than of ENDF.

So this module does not assume. It reads which blocks the file **states**, and
groups the requested components accordingly: components connected by a stated
covariance go into one matrix and are drawn once, components with nothing
between them stay in separate matrices and are drawn independently. Both are
correct; the file decides which.

**The grouping is coarse on purpose** -- see :data:`GROUPINGS`. The default
keeps every requested component of one MF in one block, which is what each of
today's pipelines already does, and merges two MFs only when a section actually
crosses them. So a request for MF33 alone assembles the matrix the MF33 pipeline
ships, bit for bit, and a request for MF33 *and* MF34 assembles two independent
blocks whose matrices are those same two -- asking for both at once costs
nothing and changes nothing, which is what makes it safe to ask for both at
once.

**One key shape for every quantity.** :class:`ComponentKey` is a 4-tuple,
``(ZA, MF, MT, index)``, where the index is the Legendre order for MF34, the
band for MF35, and 0 where the quantity has no third coordinate. The existing
builders key MF33 by ``(ZA, MT)`` and MF34 by ``(ZA, MT, L)``, which cannot be
mixed in one matrix -- not merely because a 2-tuple will not sort against a
3-tuple, but because a row of a mixed joint has to say which quantity it is a
row of. The re-keying is *order-preserving*: within one MF the sort is by the
same fields in the same order, so an assembled matrix is unchanged and the
single-MF gates keep comparing like with like.

Nothing here re-reads a covariance file. The entry builders in
:mod:`kika.sampling.model_blocks` do that, and this module re-keys and groups
what they return, so a fix to how MF34 is read reaches this path too.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import (Any, Dict, Hashable, List, NamedTuple, Optional, Sequence,
                    Tuple)

import numpy as np

from kika.sampling.model_blocks import (CROSS_SECTION_MF,
                                        _cross_section_entries, _mf34_entries,
                                        _mf35_entries, _union_grids,
                                        assemble_joint, mf35_band_domains)

__all__ = ["ComponentKey", "Selection", "QUANTITY_OF_MF", "MF_OF_QUANTITY",
           "SUPPORTED_MF", "PER_SECTION_MF", "GROUPINGS", "collectEntries",
           "samplingGroups", "requestIndex", "assembleRequest",
           "describeRequest", "componentDomains", "normaliseRequest",
           "rowFamilies"]


#: What each covariance file is a covariance *of*, in the model's vocabulary
#: rather than ENDF's. The name is the model node a perturbation of that MF
#: lands on, which is what the applier dispatches on.
QUANTITY_OF_MF = {
    31: "multiplicity",
    33: "crossSection",
    34: "angularDistribution",
    35: "energyDistribution",
}

#: The MFs this module will assemble. MF35 joined the day its model applier was
#: written -- :func:`~kika.nuclear_data.model.perturbation.applySpectrumFactors`
#: -- and not before, on the rule that assembling a covariance nothing can apply
#: only produces a draw with nowhere to go.
#:
#: **MF35 is not grouped like the other three**, and that is a property of the
#: file rather than a preference: its sections are the covariance of one energy
#: distribution over *disjoint* bands of incident energy, no cross-band block
#: exists in ENDF-6 §35, and their orders differ (84, 641, 641, 641, 641 on
#: ENDF/B-VIII.1's U-235). Merging them into one matrix under the ``"mf"``
#: grouping would state a dimension no section has. So a request for MF35
#: assembles **one block per band** whatever the grouping says -- see
#: :func:`samplingGroups`.
SUPPORTED_MF = (31, 33, 34, 35)

#: The MFs whose sections stay one matrix per section, never merged with a
#: sibling. Only MF35, and the reason is in :data:`SUPPORTED_MF`.
PER_SECTION_MF = (35,)

#: How components are partitioned into matrices that are drawn independently.
#:
#: ``"mf"``
#:     One group per MF, merged across MFs wherever a section states a block
#:     whose two sides live in different files. This is the default and it is
#:     the *conservative* choice: it never splits two components the evaluation
#:     correlates, and it reproduces what every shipped pipeline does within its
#:     own MF -- zero blocks included, since a pair of MTs a file does not
#:     correlate is stated as zero in the joint and drawn as part of it.
#:
#: ``"stated"``
#:     Connected components over the blocks the file actually states. Finer:
#:     MT4 and MT16 with no cross block between them become two matrices and two
#:     independent draws. The distribution is the same; the realisations for a
#:     given seed are not, because two draws of *n* dimensions are not one draw
#:     of *2n*. It is an improvement -- smaller matrices, and a decomposition
#:     that cannot mix two uncorrelated reactions through a shared null space --
#:     and it moves every drawn column, so it is opt-in and owes its own
#:     before/after.
GROUPINGS = ("mf", "stated")


class ComponentKey(NamedTuple):
    """One component of a joint covariance: a quantity, a reaction, a slice.

    ``index`` is the third coordinate where the quantity has one -- the Legendre
    order for MF34, the incident-energy band for MF35 -- and ``0`` where it does
    not. Zero rather than ``None`` so keys of different quantities sort against
    each other, which is what lets one matrix hold both.
    """

    za: int
    mf: int
    mt: int
    index: int = 0

    @property
    def quantity(self) -> str:
        """The model node this component perturbs, e.g. ``'crossSection'``."""
        return QUANTITY_OF_MF.get(self.mf, f"MF{self.mf}")

    def describe(self) -> str:
        tail = ""
        if self.mf == 34:
            tail = f", L={self.index}"
        elif self.mf == 35:
            tail = f", band={self.index}"
        return f"MF{self.mf}/MT{self.mt} (ZA {self.za}{tail})"


@dataclass(frozen=True)
class Selection:
    """One line of a request: which components of one MF to perturb.

    ``mt`` and ``index`` accept a single value, a sequence, or ``None`` for
    everything the file states; ``index`` is the Legendre order for MF34 and is
    *refused* for the MFs that have no third coordinate rather than ignored.

    ``relative`` defaults to ``True`` and that default is load-bearing: the
    appliers multiply by what comes back, and an absolute covariance does not
    describe a factor. ``load_mf34_covariance`` has always dropped absolute MF34
    sections before a draw for that reason. MF33's absolute sections are not
    dropped but *converted*, and the conversion divides by a central value this
    module does not have -- so a caller assembling MF33 from a tape with
    absolute blocks converts first, through
    :func:`~kika.sampling.mf33_sampling.relativiseAbsoluteSections`, and hands
    the converted sections here.
    """

    mf: int
    mt: Optional[Any] = None
    index: Optional[Any] = None
    relative: Optional[bool] = True

    def __post_init__(self) -> None:
        if self.mf not in SUPPORTED_MF:
            raise ValueError(
                f"MF{self.mf} is not one of {SUPPORTED_MF}: "
                f"{QUANTITY_OF_MF.get(self.mf, 'that quantity')} has no model "
                f"applier yet, and a covariance nothing can apply would produce "
                f"a draw with nowhere to go"
            )
        if self.index is not None and self.mf not in (34, 35):
            raise ValueError(
                f"MF{self.mf} components carry no third coordinate, so "
                f"index={self.index!r} selects nothing. Only MF34 and MF35 take "
                f"one -- the Legendre order there, the incident-energy band here"
            )


#: The model's name for each covariance file, inverted: what a request written
#: in the model's vocabulary is keyed by. See :func:`normaliseRequest`.
MF_OF_QUANTITY = {quantity: mf for mf, quantity in QUANTITY_OF_MF.items()}

#: What the third coordinate is called, per quantity, in the model spelling.
INDEX_NAME_OF_MF = {34: "order", 35: "band"}


def _mtOfReaction(reaction, suite) -> int:
    """An MT from either spelling of a reaction: its number, or its label."""
    if isinstance(reaction, (int, np.integer)):
        return int(reaction)
    if not isinstance(reaction, str):
        raise TypeError(
            f"a reaction is named by MT number or by label, got {reaction!r}")
    if suite is None:
        raise ValueError(
            f"reaction {reaction!r} is named by label, and resolving a label "
            f"needs the ReactionSuite it lives in. Pass the suite to "
            f"normaliseRequest, or name the reaction by MT"
        )
    lookups = [getattr(suite, "reactionByLabel", None)]
    sums = getattr(getattr(suite, "sums", None), "reactions", None) or ()
    for lookup in lookups:
        if lookup is None:
            continue
        try:
            found = lookup(reaction)
        except KeyError:
            found = None
        if found is not None and getattr(found, "ENDF_MT", None) is not None:
            return int(found.ENDF_MT)
    for summed in sums:
        if getattr(summed, "label", None) == reaction and \
                getattr(summed, "ENDF_MT", None) is not None:
            return int(summed.ENDF_MT)
    raise KeyError(
        f"no reaction labelled {reaction!r} in the suite; it holds "
        f"{sorted(getattr(getattr(suite, 'reactions', None), 'labels', None) or [r.label for r in suite.reactions])}"
    )


def _selectionOfQuantity(quantity: str, value, suite) -> Optional[Selection]:
    """One line of a model-vocabulary request, as the MF/MT selection it means.

    Accepted shapes of *value*, for ``"crossSection"`` and the rest alike:
    ``None``/``True`` for every reaction the file states; a reaction (MT
    number or label) or a list of them; or a dict with ``reaction`` (same),
    ``order`` (MF34) or ``band`` (MF35) -- ``index`` is accepted as the neutral
    name -- and ``relative``.
    """
    mf = MF_OF_QUANTITY[quantity]
    if value is None or value is True:
        return None
    if isinstance(value, Selection):
        return value
    if not isinstance(value, dict):
        reactions = value if isinstance(value, (list, tuple)) else [value]
        return Selection(mf=mf, mt=[_mtOfReaction(r, suite) for r in reactions])

    fields = dict(value)
    reaction = fields.pop("reaction", fields.pop("mt", None))
    indexName = INDEX_NAME_OF_MF.get(mf)
    index = fields.pop("index", None)
    for name in ("order", "band"):
        if name in fields:
            if name != indexName:
                raise ValueError(
                    f"{quantity} has no {name!r}; "
                    + (f"its third coordinate is {indexName!r}" if indexName
                       else "it has no third coordinate")
                )
            index = fields.pop(name)
    relative = fields.pop("relative", True)
    if fields:
        raise ValueError(
            f"unknown field(s) {sorted(fields)} in the {quantity} selection; "
            f"known: reaction, {indexName or 'index'}, relative"
        )
    mt = None
    if reaction is not None:
        reactions = reaction if isinstance(reaction, (list, tuple)) else [reaction]
        mt = [_mtOfReaction(r, suite) for r in reactions]
    return Selection(mf=mf, mt=mt, index=index, relative=relative)


def normaliseRequest(request, suite=None):
    """A request in the model's vocabulary, as the MF/MT spelling it means.

    Two spellings are accepted everywhere a request is taken. The ENDF one
    keys by covariance file and names reactions by MT::

        {33: None, 34: {"mt": [2], "index": [1, 2, 3]}}

    The model one keys by **quantity** and names reactions by MT *or by
    label*, with the third coordinate called what it is::

        {"crossSection": None,
         "angularDistribution": {"reaction": "MT2", "order": [1, 2, 3]}}

    The second is the one that makes sense of a GNDS source, where a reaction
    is ``"n + Fe56"`` and nothing is called MF34. It resolves to the first
    here, and the first is what the entry builders read -- a covariance link
    still carries ``ENDF_MFMT`` in GNDS, so the mapping is the file's own.
    Labels need *suite* (the :class:`ReactionSuite`) to resolve; MT numbers do
    not. The two spellings may be mixed in one request, and a request that
    is already in the first spelling comes back unchanged.
    """
    if isinstance(request, Selection) or not isinstance(request, dict):
        return request
    out: Dict[int, Any] = {}
    for key, value in request.items():
        if isinstance(key, str) and key in MF_OF_QUANTITY:
            mf = MF_OF_QUANTITY[key]
            selection = _selectionOfQuantity(key, value, suite)
        else:
            try:
                mf = int(key)
            except (TypeError, ValueError):
                raise ValueError(
                    f"request key {key!r} is neither an MF number nor one of "
                    f"the quantities {sorted(MF_OF_QUANTITY)}"
                ) from None
            selection = value
        if mf in out:
            raise ValueError(
                f"the request names MF{mf} twice (once as "
                f"{QUANTITY_OF_MF.get(mf, mf)!r}); say it once"
            )
        out[mf] = selection
    return out


def rowFamilies(index) -> Dict[Hashable, List[str]]:
    """One label per row of each block: which component the row belongs to.

    What :func:`kika.cov.conditioning.inspect_blocks` takes as *families*, so
    that "this variance is an outlier" is judged against the bins of the same
    reaction or the same Legendre order rather than against the whole joint,
    and so that a definiteness finding can say which pair of components
    carries the negative mass. Padding rows under a ``per-component`` union
    get their component's label too; they are inert and the checks skip them.
    """
    families: Dict[Hashable, List[str]] = {}
    for key, meta in index.items():
        labels: List[str] = []
        for component in meta["components"]:
            labels += [component.describe()] * int(meta["stride"])
        families[key] = labels
    return families


def _asSelections(request) -> List[Selection]:
    """A request in any of its accepted spellings, as a list of selections."""
    request = normaliseRequest(request)
    if isinstance(request, Selection):
        return [request]
    if isinstance(request, dict):
        out: List[Selection] = []
        for mf, value in request.items():
            if value is None or value is True:
                out.append(Selection(mf=int(mf)))
            elif isinstance(value, Selection):
                out.append(Selection(mf=int(mf), mt=value.mt, index=value.index,
                                     relative=value.relative))
            elif isinstance(value, dict):
                out.append(Selection(mf=int(mf), **value))
            else:
                out.append(Selection(mf=int(mf), mt=value))
        return out
    return [item if isinstance(item, Selection) else Selection(**item)
            for item in request]


def _crossSectionKey(pair, mf: int) -> ComponentKey:
    return ComponentKey(int(pair[0]), mf, int(pair[1]), 0)


def _legendreKey(triplet) -> ComponentKey:
    return ComponentKey(int(triplet[0]), 34, int(triplet[1]), int(triplet[2]))


def _bandKey(triplet) -> ComponentKey:
    return ComponentKey(int(triplet[0]), 35, int(triplet[1]), int(triplet[2]))


def collectEntries(suite, request) -> List[Tuple[ComponentKey, ComponentKey,
                                                 np.ndarray, np.ndarray,
                                                 np.ndarray]]:
    """The requested sections of *suite*, as :func:`assemble_joint` entries.

    *request* may be a :class:`Selection`, a sequence of them, or a mapping
    ``{mf: mt-list, or a dict of Selection fields}``.

    The reading is delegated:
    :func:`~kika.sampling.model_blocks._cross_section_entries` for MF31 and
    MF33, :func:`~kika.sampling.model_blocks._mf34_entries` for MF34. Their
    filters apply here unchanged -- both sides of a section have to pass, and
    the order filter is what keeps L=0 and its magnitude grid out of a request
    that did not ask for it. All this adds is the fourth coordinate.

    Raises if a selection matches nothing. A request for MT16 against a file
    that does not state MT16 is a mistake in the request, and quietly returning
    fewer components than were asked for is how an ensemble comes to be
    perturbed in fewer places than its own metadata claims.
    """
    entries: List[Tuple[ComponentKey, ComponentKey, np.ndarray, np.ndarray,
                        np.ndarray]] = []
    for selection in _asSelections(request):
        if selection.mf in CROSS_SECTION_MF:
            found = _cross_section_entries(suite, mf=selection.mf,
                                           mt=selection.mt,
                                           relative=selection.relative)
            keys = [(_crossSectionKey(row, selection.mf),
                     _crossSectionKey(col, selection.mf))
                    for row, col, *_ in found]
        elif selection.mf == 35:
            # `relative` is not forwarded, and that is the one place a
            # selection's default is deliberately ignored. It defaults to True
            # because the other three appliers *multiply* by what comes back;
            # MF35's bands are absolute by construction -- LB=7 is the
            # covariance of probabilities that already sum to one -- so
            # forwarding the default would filter every section out and the
            # request would raise "MF35 states no section" against a file that
            # states four.
            found = _mf35_entries(suite, mt=selection.mt, bands=selection.index)
            keys = [(_bandKey(row), _bandKey(col)) for row, col, *_ in found]
        else:
            found = _mf34_entries(suite, mt=selection.mt, orders=selection.index,
                                  relative=selection.relative)
            keys = [(_legendreKey(row), _legendreKey(col))
                    for row, col, *_ in found]
        if not found:
            raise ValueError(
                f"MF{selection.mf} states no section matching mt={selection.mt!r}, "
                f"index={selection.index!r}, relative={selection.relative!r}. A "
                f"selection that matches nothing is a mistake in the request, "
                f"not an empty perturbation"
            )
        for (rowKey, colKey), (_r, _c, matrix, rowGrid, colGrid) in zip(keys, found):
            entries.append((rowKey, colKey, matrix, rowGrid, colGrid))
    return entries


def componentDomains(suite, request) -> Dict[ComponentKey, Tuple[float, float]]:
    """``{component: (lo, hi)}`` for the components that have an outer domain.

    Today that is MF35 and only MF35: a band is the covariance of an energy
    distribution over a *range of incident energies*, and the factors are stated
    on the outgoing grid, so the range is a coordinate the block itself does not
    carry. Without it a realisation knows how the spectrum moves and not where.

    Kept out of :func:`collectEntries` because an ``assemble_joint`` entry is
    five elements and three of the four builders have no such coordinate to put
    in a sixth. Both read the same sections through the same filters -- see
    :func:`~kika.sampling.model_blocks._mf35_sections` -- so they cannot
    disagree about which section is band 3.
    """
    domains: Dict[ComponentKey, Tuple[float, float]] = {}
    for selection in _asSelections(request):
        if selection.mf != 35:
            continue
        found = mf35_band_domains(suite, mt=selection.mt, bands=selection.index)
        for triplet, domain in found.items():
            domains[_bandKey(triplet)] = (float(domain[0]), float(domain[1]))
    return domains


def _componentsOf(entries) -> List[ComponentKey]:
    """Every component any entry mentions, sorted."""
    keys = set()
    for rowKey, colKey, *_ in entries:
        keys.add(rowKey)
        keys.add(colKey)
    return sorted(keys)


def samplingGroups(entries, grouping: str = "mf") -> List[List[ComponentKey]]:
    """Partition the components into the sets that have to be drawn together.

    Returns one sorted list of :class:`ComponentKey` per group, the groups
    ordered by their first component, so the partition is deterministic and a
    block key built from it is stable across runs.

    See :data:`GROUPINGS` for what the two modes mean. Under either, a stated
    block always joins its two sides: the modes differ only in what they do with
    components the file leaves unconnected.
    """
    if grouping not in GROUPINGS:
        raise ValueError(f"grouping must be one of {GROUPINGS}, got {grouping!r}")

    parent: Dict[ComponentKey, ComponentKey] = {k: k for k in _componentsOf(entries)}

    def find(key: ComponentKey) -> ComponentKey:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: ComponentKey, b: ComponentKey) -> None:
        rootA, rootB = find(a), find(b)
        if rootA != rootB:
            parent[max(rootA, rootB)] = min(rootA, rootB)

    for rowKey, colKey, *_ in entries:
        union(rowKey, colKey)

    if grouping == "mf":
        # Everything of one MF becomes one node set, so a cross-file section
        # merges two whole files rather than two components of them. Half a
        # merge -- MT2's L=0 joined to MF33 while its L=1 stays outside -- would
        # state a correlation with a component that is not in the matrix, which
        # is the defect the entry builders' "both sides must pass" rule exists
        # to prevent, one level out.
        firstOfMf: Dict[int, ComponentKey] = {}
        for key in list(parent):
            if key.mf in PER_SECTION_MF:
                # One matrix per section, under either grouping. See
                # `SUPPORTED_MF`: MF35's bands are disjoint in incident energy,
                # ENDF states no block between two of them, and their orders
                # differ -- so "everything of one MF in one matrix" would
                # assemble a dimension no section has.
                continue
            if key.mf in firstOfMf:
                union(firstOfMf[key.mf], key)
            else:
                firstOfMf[key.mf] = key

    groups: Dict[ComponentKey, List[ComponentKey]] = {}
    for key in parent:
        groups.setdefault(find(key), []).append(key)
    return [sorted(members) for _root, members in sorted(groups.items())]


def _unionModeFor(keys: Sequence[ComponentKey]) -> str:
    """The bin structure a group is assembled on. See ``model_blocks.UNION_MODES``.

    ``"global"`` for a group made only of cross-section-like components
    (MF31/MF33): that is the layout those pipelines ship and the one measured
    bit-identical to the carrier. ``"per-component"`` for anything else, and for
    a **mixed** group it is not a preference but the only workable answer --
    pooling MF33's grid with MF34's restates every component on the union of
    both, and on a real Fe-56 file that is some 1400 bins times thirteen
    components where per-component gives a seventh of it. MF34's own components
    share a grid by construction, so within MF34 the two modes agree anyway.
    """
    return "global" if all(k.mf in CROSS_SECTION_MF for k in keys) else "per-component"


def _groupLabel(keys: Sequence[ComponentKey]) -> str:
    """``'MF33'`` for a single-file group, ``'MF33+MF34'`` for a merged one."""
    return "+".join(f"MF{mf}" for mf in sorted({k.mf for k in keys}))


def _entriesOfGroup(entries, members):
    """The entries whose two sides are both in *members*."""
    inside = set(members)
    return [entry for entry in entries if entry[0] in inside and entry[1] in inside]


def requestIndex(entries, *, isotope: Any = None, grouping: str = "mf",
                 atol: float = 1e-12, domains=None
                 ) -> Dict[Hashable, Dict[str, Any]]:
    """What the rows of each assembled block are, without assembling anything.

    The sibling of :func:`~kika.sampling.model_blocks.legendre_covariance_index`
    and for the same reason: on a real MF34 tape the joint is gigabytes, and
    calling the two functions in the obvious order allocates it twice. The
    layout follows from the grids alone.

    Each value carries ``components`` (in row order, one ``stride`` of rows
    each), ``grids`` and ``widths`` per component, the ``dimension``, the
    ``union`` mode the group was laid out under, and the ``quantities`` it
    perturbs.

    The field is ``components``, where the two single-MF indices say ``pairs``
    and ``triplets``. One name, because the point of the key being a 4-tuple is
    that a consumer no longer has to know which quantity it is looking at in
    order to know what a row is.

    *domains* is what :func:`componentDomains` returned, when the request names
    a quantity that has an outer coordinate the block is not stated on -- MF35's
    incident band. It is carried through into ``domains`` on each block's entry,
    per component, so a :class:`~kika.sampling.perturbation_set.PerturbationSet`
    built from this index knows *where* each block applies as well as *what* it
    says.
    """
    index: Dict[Hashable, Dict[str, Any]] = {}
    for members in samplingGroups(entries, grouping=grouping):
        groupEntries = _entriesOfGroup(entries, members)
        union = _unionModeFor(members)
        unions = _union_grids(groupEntries, atol=atol, union=union)
        keys = sorted(unions)
        widths = {key: len(unions[key]) - 1 for key in keys}
        stride = max(widths.values())
        index[(isotope, _groupLabel(keys), tuple(keys))] = {
            "components": list(keys),
            "stride": stride,
            "grids": {key: unions[key] for key in keys},
            "widths": widths,
            "dimension": len(keys) * stride,
            "union": union,
            "quantities": sorted({key.quantity for key in keys}),
            "domains": {key: tuple(domains[key])
                        for key in keys if key in (domains or {})},
        }
    return index


def assembleRequest(entries, *, isotope: Any = None, grouping: str = "mf",
                    atol: float = 1e-12, domains=None
                    ) -> Tuple[List[Tuple[Hashable, np.ndarray]],
                               Dict[Hashable, Dict[str, Any]]]:
    """The blocks a request assembles to, and the index saying what their rows are.

    Returns ``(blocks, index)``, which is what
    :func:`~kika.sampling.core.draw_samples` and the appliers take. One block
    per group: several blocks means several independent draws, which is exactly
    what the file said when it stated no covariance between them.

    Built through :func:`~kika.sampling.model_blocks.assemble_joint` rather than
    beside it, so a single-MF request produces the same matrix as that MF's own
    entry point, bit for bit.
    """
    index = requestIndex(entries, isotope=isotope, grouping=grouping, atol=atol,
                         domains=domains)
    blocks: List[Tuple[Hashable, np.ndarray]] = []
    for key, meta in index.items():
        groupEntries = _entriesOfGroup(entries, meta["components"])
        keys, joint, stride = assemble_joint(groupEntries, atol=atol,
                                             union=meta["union"])
        if stride != meta["stride"] or list(keys) != list(meta["components"]):
            raise RuntimeError(
                f"the index and the assembly disagree about group {key}: stride "
                f"{meta['stride']} vs {stride}, {len(meta['components'])} vs "
                f"{len(keys)} components. They are built from the same grids, so "
                f"this means one of them stopped being derived from them"
            )
        blocks.append((key, joint))
    return blocks, index


def describeRequest(entries, *, grouping: str = "mf") -> str:
    """One line per group: what it holds, how big it is, and why it is separate.

    Meant to be logged before a run and read by a human. "These two quantities
    were drawn independently" is a claim about the evaluation, and a run that
    does not state it cannot be checked afterwards.
    """
    groups = samplingGroups(entries, grouping=grouping)
    stated = {tuple(sorted((row.mf, col.mf)))
              for row, col, *_ in entries if row.mf != col.mf}
    head = f"{len(groups)} independent draw(s), grouping={grouping!r}"
    head += (f"; stated cross-file blocks: {sorted(stated)}" if stated
             else "; no section states a block across two files")
    lines = [head]
    for number, members in enumerate(groups):
        union = _unionModeFor(members)
        unions = _union_grids(_entriesOfGroup(entries, members), union=union)
        stride = max(len(grid) - 1 for grid in unions.values())
        lines.append(
            f"  [{number}] {_groupLabel(members)}: {len(members)} component(s), "
            f"stride {stride}, dimension {len(members) * stride}, union={union}"
        )
    return "\n".join(lines)
