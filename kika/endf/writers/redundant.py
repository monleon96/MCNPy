"""
Rebuild the redundant (summation) cross sections of MF3 from their partials.

ENDF-6 stores several cross sections twice: MT1 is the total, but MT2 and the
non-elastic partials that make it up are given as well, and the format requires
the two to agree. Editing a partial -- transferring MT52 in from another
evaluation, say -- leaves MT4, MT3 and MT1 stating the old sum, and the file is
then internally inconsistent even though every individual section parses.

This module recomputes the affected redundant sections. It is deliberately
conservative about *which* ones it touches: see :func:`recompute_redundant_mf3`.

The sum rules themselves live in :data:`kika._constants.MF3_SUM_RULES`, next to
the ACE table they intentionally differ from.
"""
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..classes.mf3.mf3mt import MF3MT
from ..parsers.parse_mf3 import parse_mf3_mt
from ..utils import (
    PaddingProbe,
    format_tab1,
    parse_endf_id,
    record_width,
)
from ..._constants import MF3_SUM_ORDER, MF3_SUM_RULES
from ...utils import get_endf_logger

logger = get_endf_logger(__name__)

__all__ = [
    "RedundantUpdate",
    "recompute_redundant_mf3",
    "resolve_sum_components",
]

#: How far a redundant MT may sit from the sum of its partials and still count
#: as "this file's partials do represent it" -- the baseline gate of
#: :func:`recompute_redundant_mf3`.
#:
#: Set from measurement, because the two populations it has to separate are
#: further apart than the format's own round-off would suggest. Distributed
#: evaluations are mostly exact to ~1e-6, the round-off of the
#: six-significant-digit ENDF float, but not always: ENDF/B-VIII.1 B-10 states
#: MT3 0.19% off its partials, JEFF-4.0 U-238 MT4 0.71%, JENDL-5 Fe-56 MT1
#: 0.88%, and JEFF-4.0 U-235 MT1 2.3%. Those are real files a user will open,
#: and refusing to maintain their totals would make the feature look broken.
#:
#: A tape cut down to a few sections is the case that must be caught, and it
#: lands an order of magnitude away: ``micro_fe56_structural.endf`` keeps MT1,
#: MT2 and MT102 out of a full Fe-56, and its MT1 sits 63% above MT2+MT102.
#: 10% is the gap between the two.
DEFAULT_TOLERANCE = 0.1

#: Below this, a difference is the six-digit ENDF float talking and not the
#: cross section. A resummation that lands here on an unchanged grid is
#: reported as leaving the section alone, so re-transferring an identical
#: section does not churn the tape.
ROUNDOFF = 1e-5

#: HEAD is line 1 of an MF3 section, so its TAB1 record starts at line 2.
_TAB1_START_LINE = 2

#: Columns 1-44 of the TAB1 header hold QM, QI, L1 and LR -- untouched by a
#: resummation, and preserved byte for byte so the rewrite shows up as a change
#: to the table and nothing else. Columns 45-66 (NR, NP) are regenerated.
_TAB1_PRESERVED_COLUMNS = 44

_FIELDS_PER_LINE = 3  # x/y pairs, or (NBT, INT) pairs, per 66-column record


@dataclass
class RedundantUpdate:
    """What happened to one redundant MT."""

    mt: int
    #: ``'updated'``, ``'unchanged'``, or ``'skipped'``.
    status: str
    #: The partials it was (or would have been) summed from.
    components: Tuple[int, ...] = ()
    #: Why it was skipped, or left unchanged. Empty when it was updated.
    reason: str = ""
    points_before: int = 0
    points_after: int = 0
    #: Largest relative move of this MT against its previous values.
    max_rel_change: float = 0.0
    #: How far the *original* file was from satisfying this sum rule, when a
    #: baseline was supplied. ``None`` when it was not, or could not be tested.
    baseline_deviation: Optional[float] = None

    def describe(self) -> str:
        if self.status == "updated":
            return (f"MT{self.mt} rebuilt from {len(self.components)} partials "
                    f"({self.points_before} -> {self.points_after} points, "
                    f"max change {self.max_rel_change:.3%})")
        return f"MT{self.mt} {self.status}: {self.reason}"


# ---------------------------------------------------------------------------
# Sum rules
# ---------------------------------------------------------------------------

def resolve_sum_components(mt: int, present: Iterable[int]) -> Tuple[int, ...]:
    """The partials of redundant *mt* that *present* actually contains.

    A ``'@X'`` entry in the rule means "X, or what X is made of". Using X when
    the file gives it is what keeps the sum from double-counting; falling back
    to its own partials when it does not is what keeps a file that gives
    MT600-649 but no MT103 from silently losing its (n,p) contribution.

    Returns an empty tuple when nothing resolves -- the caller must not treat
    that as "the sum is zero".
    """
    available = set(present)

    def _walk(target: int, seen: frozenset) -> List[int]:
        if target in seen:  # the rules are acyclic; this is belt and braces
            return []
        seen = seen | {target}
        out: List[int] = []
        rule = MF3_SUM_RULES.get(target)
        if rule is None:
            return []
        for entry in rule[0]:
            if isinstance(entry, str):
                ref = int(entry.lstrip("@"))
                if ref in available:
                    out.append(ref)
                else:
                    out.extend(_walk(ref, seen))
            elif entry in available:
                out.append(entry)
        return out

    ordered: List[int] = []
    for comp in _walk(mt, frozenset()):
        if comp != mt and comp not in ordered:
            ordered.append(comp)
    return tuple(ordered)


# ---------------------------------------------------------------------------
# Summation on a union grid
# ---------------------------------------------------------------------------

def _union_grid(grids: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Union of ENDF energy grids, keeping repeated energies.

    A repeated energy is a step discontinuity in sigma, not a duplicate to be
    tidied away: the two entries carry the left and right limits. ``np.unique``
    would collapse the pair and turn the step into a ramp across the whole
    neighbouring interval, which is a real change to the cross section.

    Returns the grid and, for each of its points, the 0-based index of that
    point among the repeats of its own energy.
    """
    values = np.unique(np.concatenate(grids))
    multiplicity = np.ones(values.size, dtype=int)
    for grid in grids:
        seen, counts = np.unique(grid, return_counts=True)
        np.maximum.at(multiplicity, np.searchsorted(values, seen), counts)

    expanded = np.repeat(values, multiplicity)
    starts = np.repeat(np.cumsum(multiplicity) - multiplicity, multiplicity)
    return expanded, np.arange(expanded.size) - starts


def _evaluate(section: MF3MT, grid: np.ndarray, occurrence: np.ndarray) -> np.ndarray:
    """Sigma of *section* over *grid*, zero outside its own energy range.

    Points of *grid* that the section carries verbatim are taken verbatim
    rather than interpolated, which is what makes a repeated energy come out
    right: where the section repeats it too, its k-th entry answers the k-th
    repeat, so its discontinuity survives into the sum. Where the section does
    not repeat it, that energy is a continuity point of this partial and every
    repeat gets the same value.
    """
    energies = np.asarray(section.energies, dtype=float)
    values = np.asarray(section.cross_sections, dtype=float)
    out = np.asarray(
        section.get_cross_section(grid, out_of_range="zero"), dtype=float
    ).copy()
    if energies.size == 0:
        return out

    lo = np.searchsorted(energies, grid, side="left")
    hi = np.searchsorted(energies, grid, side="right")
    repeats = hi - lo
    exact = repeats > 0
    if exact.any():
        idx = lo + np.minimum(occurrence, np.maximum(repeats - 1, 0))
        np.clip(idx, 0, energies.size - 1, out=idx)
        out[exact] = values[idx[exact]]
    return out


def _sum_partials(
    sections: Dict[int, MF3MT],
    components: Sequence[int],
    extra_grids: Sequence[np.ndarray] = (),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum *components* onto the union of their grids (plus *extra_grids*).

    Returns the grid, its per-energy repeat indices, and the summed values.
    """
    grids = [np.asarray(sections[mt].energies, dtype=float) for mt in components]
    grids.extend(g for g in extra_grids if len(g))
    grid, occurrence = _union_grid(grids)

    total = np.zeros(grid.size, dtype=float)
    for mt in components:
        total += _evaluate(sections[mt], grid, occurrence)
    return grid, occurrence, total


def _relative_deviation(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Largest relative gap between two curves sampled on the same grid.

    Divided by the local value, with a floor six decades under the section's
    own peak. Without the floor the deep minima of a resonance cross section
    -- where both curves are numerically zero -- would dominate every
    comparison with meaningless ratios.
    """
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.size == 0:
        return 0.0
    scale = float(np.max(np.abs(reference))) or 1.0
    denominator = np.maximum(np.abs(reference), 1e-6 * scale)
    return float(np.max(np.abs(candidate - reference) / denominator))


# ---------------------------------------------------------------------------
# Reading and rewriting MF3 sections in raw tape text
# ---------------------------------------------------------------------------

def _index_mf3(lines: Sequence[str]) -> "Dict[int, List[int]]":
    """``{MT: [line indices]}`` for the MF3 data lines of a tape."""
    index: Dict[int, List[int]] = {}
    for i, line in enumerate(lines):
        text = line.rstrip("\r\n")
        # 75 columns is a whole record: the MT field ends there, and the
        # sequence number after it is optional. Only the line ending is
        # stripped -- rstripping spaces as well would shorten a record whose
        # last data field is blank-padded and drop it from the index.
        if len(text) < 75:
            continue
        _, mf, mt = parse_endf_id(text)
        if mf == 3 and mt is not None and mt > 0:
            index.setdefault(mt, []).append(i)
    return index


def _probe_padding(section_lines: Sequence[str], nr: int, npoints: int):
    """How this section's writer filled the unused fields of its short lines."""
    probe = PaddingProbe()
    interp_lines = -(-nr // _FIELDS_PER_LINE)
    interp_end = _TAB1_START_LINE + interp_lines
    probe.observe_interp(section_lines, interp_end, nr)
    data_lines = -(-npoints // _FIELDS_PER_LINE)
    probe.observe_pairs(section_lines, interp_end + data_lines, npoints)
    return probe.resolve()


def _render(section: MF3MT, section_lines: Sequence[str], pad_style, width: int) -> List[str]:
    """Serialize *section* back to tape lines, reusing its original header.

    The HEAD record is copied rather than regenerated. It carries ZA and AWR
    and, in some evaluations, a non-zero field the ``MF3MT`` dataclass does not
    model -- ENDF/B-VIII.1 B-10 writes L2=2 on MF3/MT2 -- so regenerating it
    would quietly drop data this function has no business touching. The same
    reasoning keeps columns 1-44 of the TAB1 header (QM, QI, L1, LR).
    """
    if len(section_lines) < 2:
        raise ValueError(
            f"MF3/MT{section.number}: {len(section_lines)} line(s), too short to "
            f"hold a HEAD and a TAB1 record")
    mat = section._mat if section._mat is not None else 0
    tab1_lines, _ = format_tab1(
        section._qm, section._qi, 0, section._lr,
        section.energy_interpolation,
        section.energies, section.cross_sections,
        mat, 3, section.number, _TAB1_START_LINE,
        pad=pad_style.pairs, interp_pad=pad_style.interp,
    )
    original_tab1_header = section_lines[1].rstrip("\r\n")
    tab1_lines[0] = (
        original_tab1_header[:_TAB1_PRESERVED_COLUMNS]
        + tab1_lines[0][_TAB1_PRESERVED_COLUMNS:]
    )
    rendered = [section_lines[0].rstrip("\r\n")] + tab1_lines
    return [line[:width] for line in rendered]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def recompute_redundant_mf3(
    content: str,
    *,
    changed_mts: Optional[Iterable[int]] = None,
    protected_mts: Iterable[int] = (),
    baseline_content: Optional[str] = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Tuple[str, List[RedundantUpdate]]:
    """Rebuild the MF3 summation cross sections that an edit has invalidated.

    Each redundant MT is resummed from the partials the file actually carries,
    onto the union of their energy grids *and its own*. Keeping its own grid
    costs a few points and means the rewrite can never drop energies the
    evaluator put there -- which matters when a partial interpolates
    logarithmically, since the sum of a log-log and a lin-lin partial is not
    exactly representable in either and the evaluator's extra points are what
    carries the accuracy.

    Parameters
    ----------
    content : str
        The tape to rewrite.
    changed_mts : iterable of int, optional
        The MF3 MTs that were edited. Only redundants reached from these are
        rebuilt, and only through redundants that themselves changed: editing
        MT52 rebuilds MT4, then MT3 and MT1 because MT4 moved. ``None`` means
        "rebuild every summation MT in the file".
    protected_mts : iterable of int
        MTs never to overwrite. Pass the sections the user asked for by name:
        transferring MT1 explicitly and then replacing it with the local sum
        would discard the very section they moved.
    baseline_content : str, optional
        The tape as it was before the edit. When given, a redundant MT is only
        rebuilt if it agreed with its partials *beforehand* -- the invariant is
        restored where it held, and a file that never satisfied it (a tape cut
        down to a few MTs, most obviously) is reported and left alone rather
        than having its total quietly replaced by a sum over whatever survived.
    tolerance : float
        Relative agreement required of that baseline check.

    Returns
    -------
    (content, updates)
        The rewritten tape, and one :class:`RedundantUpdate` per redundant MT
        considered. MF1/451 is *not* touched -- run
        :func:`~kika.endf.writers.update_directory.update_mf1_directory`
        afterwards, since line counts change.
    """
    lines = content.splitlines(keepends=True)
    index = _index_mf3(lines)
    if not index:
        return content, []

    width = record_width(lines)
    protected = set(protected_mts)

    sections: Dict[int, MF3MT] = {}
    pad_styles = {}
    for mt, positions in index.items():
        section_lines = [lines[i].rstrip("\r\n") for i in positions]
        try:
            section = parse_mf3_mt(section_lines, mt)
        except Exception as exc:  # a section we cannot read is one we cannot sum
            logger.warning(f"MF3/MT{mt}: not parsed, excluded from resummation ({exc})")
            continue
        sections[mt] = section
        pad_styles[mt] = _probe_padding(
            section_lines, section.num_interpolation_regions, len(section.energies)
        )

    baseline = _parse_baseline(baseline_content) if baseline_content else None

    dirty = set(sections) if changed_mts is None else set(changed_mts)
    updates: List[RedundantUpdate] = []
    rewritten: Dict[int, MF3MT] = {}

    for mt in MF3_SUM_ORDER:
        if mt not in sections:
            continue
        components = resolve_sum_components(mt, sections)
        if changed_mts is not None and not (dirty & set(components)):
            continue

        if mt in protected:
            updates.append(RedundantUpdate(
                mt=mt, status="skipped", components=components,
                reason="transferred explicitly; left as the user placed it",
            ))
            continue
        if not components:
            updates.append(RedundantUpdate(
                mt=mt, status="skipped",
                reason="the file carries none of its partials",
            ))
            continue

        deviation = _baseline_deviation(baseline, mt) if baseline else None
        if deviation is not None and deviation > tolerance:
            updates.append(RedundantUpdate(
                mt=mt, status="skipped", components=components,
                baseline_deviation=deviation,
                reason=(f"was already {deviation:.2%} from the sum of its "
                        f"partials before the edit"),
            ))
            continue

        section = sections[mt]
        old_energies = np.asarray(section.energies, dtype=float)
        grid, occurrence, total = _sum_partials(
            sections, components, extra_grids=[old_energies])

        previous = (_evaluate(section, grid, occurrence) if old_energies.size
                    else np.zeros_like(total))
        change = _relative_deviation(previous, total)

        if change <= ROUNDOFF and np.array_equal(grid, old_energies):
            updates.append(RedundantUpdate(
                mt=mt, status="unchanged", components=components,
                points_before=old_energies.size, points_after=grid.size,
                baseline_deviation=deviation,
                reason="already equal to the sum of its partials",
            ))
            continue

        section._energies = [float(e) for e in grid]
        section._cross_sections = [float(v) for v in total]
        section._interpolation = [(int(grid.size), 2)]
        section._nr = 1
        section._np = int(grid.size)
        rewritten[mt] = section
        dirty.add(mt)

        updates.append(RedundantUpdate(
            mt=mt, status="updated", components=components,
            points_before=int(old_energies.size), points_after=int(grid.size),
            max_rel_change=change, baseline_deviation=deviation,
        ))
        logger.debug(updates[-1].describe())

    if not rewritten:
        return content, updates

    # Splice from the bottom up so the earlier line indices stay valid.
    for mt in sorted(rewritten, key=lambda m: index[m][0], reverse=True):
        positions = index[mt]
        original = [lines[i].rstrip("\r\n") for i in positions]
        terminator = lines[positions[0]][len(original[0]):] or "\n"
        block = [line + terminator
                 for line in _render(rewritten[mt], original, pad_styles[mt], width)]
        lines[positions[0]:positions[-1] + 1] = block

    return "".join(lines), updates


def _parse_baseline(content: str) -> Dict[int, MF3MT]:
    """The MF3 sections of the pre-edit tape, for the consistency gate."""
    lines = content.splitlines(keepends=True)
    sections: Dict[int, MF3MT] = {}
    for mt, positions in _index_mf3(lines).items():
        try:
            sections[mt] = parse_mf3_mt(
                [lines[i].rstrip("\r\n") for i in positions], mt
            )
        except Exception:
            continue
    return sections


def _baseline_deviation(baseline: Dict[int, MF3MT], mt: int) -> Optional[float]:
    """How far the pre-edit tape was from satisfying this sum rule.

    ``None`` when the question does not arise -- the MT was not there before,
    or none of its partials were -- and the caller should go ahead.
    """
    if mt not in baseline:
        return None
    components = resolve_sum_components(mt, baseline)
    if not components:
        return None
    own = np.asarray(baseline[mt].energies, dtype=float)
    grid, occurrence, total = _sum_partials(baseline, components, extra_grids=[own])
    reference = _evaluate(baseline[mt], grid, occurrence)
    return _relative_deviation(reference, total)
