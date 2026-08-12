"""Pre-flight for a covariance matrix: can it be sampled, and what would a repair cost?

Every perturbation pipeline in this library repairs its covariance on the way
past. ``generate_samples`` clamps variances, rescales threshold bins, rescales
statistical outliers, caps what is left, masks inert rows and then projects onto
the PSD cone — six repairs, all of them inside the call that draws the samples,
several of them not reachable any other way. So the first time anyone learns
that an evaluation's covariance needed surgery is *after* a run has produced
tapes, and the only record of what was done is a log line.

This module is the other half of that arrangement, and it does no surgery at
all. :func:`inspect_blocks` reads matrices and answers three questions:

1. **Can this be sampled as it stands?** Non-finite entries, negative stated
   variances and correlations outside [-1, 1] make the answer no, and no repair
   makes it yes without changing what the evaluation says.
2. **If it can, will the draw reproduce what the file states?** A matrix with a
   negative eigenvalue is sampled by projecting it somewhere else first; a
   matrix with one bin at 10⁹× the median of its neighbours is sampled with
   that bin capped. Both succeed. Neither reproduces the input.
3. **What would each available repair actually do to it?** Not "clip is
   eigenvector-preserving" — the measured diagonal movement, Frobenius
   distance, resulting spectrum and surviving sum rule, on *this* matrix.

Question 3 is the point. The choice between ``clip``, ``higham`` and
``clip_rescale`` is not decidable in the abstract: for PFNS the deciding fact is
that ``C·1 ≈ 0`` and a congruence transform destroys it, which was established
last session by running the candidates by hand on four Cf-252 bands and
measuring. :func:`predict_psd_repairs` is that experiment as a function.

**Nothing here modifies a matrix that a caller keeps.** The remedy predictions
build repaired copies to measure them and throw them away. The appliers live in
:mod:`kika.cov.decomposition`, are unchanged, and are still what the pipelines
call. Wiring inspection *into* the pipelines — so that a run declares its
repairs up front instead of discovering them — is the migration, and this module
is deliberately the half of it that changes no result.

**Format-agnostic by construction.** A block is a ``(key, matrix)`` pair, as in
:func:`kika.sampling.core.draw_samples`; the key can be anything hashable. The
one optional piece of structure is *families* — a label per row saying which
rows are comparable to each other, so "this variance is an outlier" can mean
"among its own reaction" rather than "among everything in the block". MF33
passes reaction labels, MF35 passes band labels, MF32 passes parameter names,
and a caller with nothing to say passes nothing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from .decomposition import (
    _PSD_AUTO_THRESHOLD,
    cap_variance_congruence,
    clip_and_rescale,
    clip_projection,
    nearest_psd_higham,
    rescale_threshold_bins_congruence,
)

__all__ = [
    "BLOCKS",
    "DISTORTS",
    "NOTE",
    "Remedy",
    "Finding",
    "BlockReport",
    "ConditioningReport",
    "ConditioningPlan",
    "PlanStep",
    "apply_plan",
    "as_blocks",
    "block_key_text",
    "inert_row_mask",
    "inspect_matrix",
    "inspect_blocks",
    "predict_psd_repairs",
    "APPLIABLE_REMEDIES",
    "INERT_VARIANCE_FLOOR",
    "DEFAULT_NULL_TOL",
    "DEFAULT_PSD_CANDIDATES",
    "OUTLIER_FACTOR",
    "PREDICT_HIGHAM_MAX_ORDER",
    "PREDICT_MAX_ORDER",
    "ROUNDOFF_RATIO",
    "SUM_RULE_TOL",
]

#: A draw either fails or returns something that is not a sample of anything.
BLOCKS = "blocks"
#: A draw succeeds, but not of the covariance that was handed in.
DISTORTS = "distorts"
#: True, worth knowing, implies no action.
NOTE = "note"

_SEVERITY_ORDER = {BLOCKS: 0, DISTORTS: 1, NOTE: 2}

#: Variance at or below this counts as "the evaluation states nothing here".
#: Matches ``_DIAG_FLOOR`` in :func:`kika.sampling.generators.generate_samples`.
INERT_VARIANCE_FLOOR = 1e-12

#: Eigenvalues at or below ``tol × λ_max`` are null directions. Matches
#: ``DEFAULT_NULL_TOL`` in :mod:`kika.sampling.core`.
DEFAULT_NULL_TOL = 1e-10

#: Variance above this multiple of its family median is an evaluator artefact
#: rather than heterogeneity. Matches ``flag_outlier_variance_bins``.
OUTLIER_FACTOR = 1000.0

#: A negativity ratio at or below ``order × this`` is the eigensolver, not the
#: evaluation, and ``definiteness`` does not fire on it.
#:
#: **Found by closing the loop, and it is not cosmetic.** Without a tolerance the
#: check fires on any ``λ_min < 0`` at all, so a matrix that has just been
#: projected onto the PSD cone still reports as not PSD — clip leaves its
#: clipped eigenvalues at zero and reconstruction puts round-off back on the
#: wrong side. Measured on the module's own fixtures: -2.3e-17 and +3.9e-18
#: after a clip that started from -1.4e-04. A pre-flight that cannot certify the
#: matrix its own recommendation produced is a pre-flight nobody can act on, and
#: ``apply_plan`` → ``inspect_blocks`` is exactly the loop a run has to close.
#:
#: Scaled by the order because the error in a symmetric eigendecomposition grows
#: with it. ``order × eps`` sits two orders above what the fixtures measure and
#: eleven below the smallest real indefiniteness seen on a file (-1.4e-04 on a
#: Cf-252 MF35 band, -1.3e-06 on the Fe-56 MF34 joint), so nothing an evaluation
#: actually states is anywhere near it.
ROUNDOFF_RATIO = float(np.finfo(np.float64).eps)

#: ``max|Σ_j C_ij| / max|C|`` below this means the block carries a sum rule.
#:
#: Calibrated, not guessed. The four Cf-252 MF35 bands come in at 2e-6 to 7e-6
#: — *not* at float64 epsilon, because ENDF writes six significant figures and
#: the identity can only hold to what was written. Everything with no sum rule
#: measures O(1) on the same statistic, since an n-term row sum of unrelated
#: entries does not cancel: 1.7 for Th-232 MF32, 1.9 for Fe-56 MF33, 1.6 for a
#: random low-rank matrix. Five orders of separation, so the threshold sits in
#: the middle and its exact value does not matter.
SUM_RULE_TOL = 1e-3

#: PSD repairs measured by default. All three, because the answer is not
#: predictable from the algorithms and the two cheap ones do not bracket the
#: third: on a Cf-252 MF35 band ``clip`` is the best of them on the sum rule
#: and the worst on the diagonal, ``clip_rescale`` is the reverse, and Higham
#: lands in the middle of both. A pre-flight that omitted it would recommend
#: from two points on a three-point trade-off.
DEFAULT_PSD_CANDIDATES = ("clip", "clip_rescale", "higham")

#: Iteration cap used when *predicting* what Higham would do; the real applier
#: has no cap. 200 rather than the applier's default because it is within 5% of
#: the converged answer on a real band (stated-diagonal movement 5.78e-3 against
#: 5.52e-3) for 1/20th of the time — convergence takes ~1900 iterations and
#: ~13 s where 200 costs ~0.6 s. Predictions made under the cap report
#: ``converged=False`` rather than passing the number off as the converged one.
_PREDICT_HIGHAM_MAX_ITER = 200

#: Blocks larger than this are not put through remedy prediction — three
#: eigendecompositions of a 4000x4000 is not a pre-flight cost. The finding is
#: still raised; only its remedies come back unmeasured, with a reason.
PREDICT_MAX_ORDER = 1500

#: Higham gets a far lower ceiling of its own. It is O(iterations x n^3) where
#: the others are one n^3, so the ~0.6 s it costs at n=122 becomes minutes by
#: n=1500 — measured 0.03 s at 50 iterations and 0.57 s at 200 on a 122x122.
#: Past this order it is reported as an option and not measured.
PREDICT_HIGHAM_MAX_ORDER = 250


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Remedy:
    """One available repair, and what it was measured to do to this matrix.

    ``effect`` is not a description of the algorithm. It is the outcome of
    running it here: how far the diagonal moved, how far the whole matrix
    moved, what the spectrum became, whether a sum rule survived. Two matrices
    can rank the same three remedies in opposite orders, which is why the
    ranking is not hardcoded anywhere in this module.
    """

    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    effect: Dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def describe(self) -> str:
        if self.effect.get("predicted") is False:
            return f"{self.name}(not measured: {self.effect.get('reason', 'no reason given')})"
        bits = []
        diagonal = self.effect.get("stated_diagonal_max_relative_change")
        if diagonal is not None:
            bits.append(f"stated σ² {'exact' if diagonal < 1e-12 else f'{diagonal:.1%}'}")
        added = self.effect.get("max_variance_added_to_an_inert_row")
        if added is not None and self.effect.get("n_inert_rows"):
            bits.append(
                f"{self.effect['n_inert_rows']} inert rows "
                + ("untouched" if added == 0.0 else f"gain ≤{added:.1e} σ²")
            )
        frobenius = self.effect.get("frobenius_relative_change")
        if frobenius is not None:
            bits.append(f"‖ΔC‖/‖C‖ {frobenius:.2e}")
        before = self.effect.get("sum_rule_residual_before")
        after = self.effect.get("sum_rule_residual_after")
        if after is not None:
            bits.append(f"C·1 {before:.1e}→{after:.1e}" if before is not None else f"C·1 {after:.1e}")
        if self.effect.get("converged") is False:
            bits.append(f"did NOT converge in {self.effect.get('iterations')} iter")
        seconds = self.effect.get("seconds")
        if seconds is not None:
            bits.append(f"{seconds * 1e3:.0f} ms")
        detail = ", ".join(bits)
        return f"{self.name}({detail})" if detail else self.name


@dataclass(frozen=True)
class Finding:
    """One thing that is true about a matrix, with its evidence and options."""

    check: str
    severity: str
    summary: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    remedies: Tuple[Remedy, ...] = ()

    def __str__(self) -> str:
        line = f"[{self.severity}] {self.check}: {self.summary}"
        if self.remedies:
            line += "\n    options: " + "; ".join(r.describe() for r in self.remedies)
        return line


@dataclass(frozen=True)
class BlockReport:
    """Everything :func:`inspect_matrix` found in one block."""

    key: Hashable
    order: int
    findings: Tuple[Finding, ...]

    @property
    def blocking(self) -> Tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == BLOCKS)

    @property
    def distorting(self) -> Tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == DISTORTS)

    @property
    def samplable(self) -> bool:
        """True when a draw would run and mean something without any repair."""
        return not self.blocking

    @property
    def faithful(self) -> bool:
        """True when a draw would additionally reproduce the stated covariance."""
        return not self.blocking and not self.distorting

    def __str__(self) -> str:
        head = f"{self.key} ({self.order}x{self.order})"
        if not self.findings:
            return f"{head}: clean"
        body = "\n".join(f"  {line}" for f in self.findings for line in str(f).split("\n"))
        return f"{head}:\n{body}"


@dataclass(frozen=True)
class ConditioningReport:
    """The pre-flight for a whole set of blocks.

    Truthiness is *not* defined on this object on purpose. "Is the report ok"
    has two different answers — :attr:`samplable` and :attr:`faithful` — and a
    caller that writes ``if report:`` has certainly not decided which one it
    meant.
    """

    blocks: Tuple[BlockReport, ...]

    @property
    def samplable(self) -> bool:
        return all(b.samplable for b in self.blocks)

    @property
    def faithful(self) -> bool:
        return all(b.faithful for b in self.blocks)

    def failing(self) -> Tuple[BlockReport, ...]:
        return tuple(b for b in self.blocks if not b.samplable)

    def distorting(self) -> Tuple[BlockReport, ...]:
        return tuple(b for b in self.blocks if b.samplable and not b.faithful)

    def findings(self, check: Optional[str] = None) -> Tuple[Finding, ...]:
        """Every finding across every block, optionally filtered by check name."""
        return tuple(
            f for b in self.blocks for f in b.findings
            if check is None or f.check == check
        )

    def summary(self) -> str:
        """One line. What a script prints before deciding whether to go on."""
        n = len(self.blocks)
        bad = len(self.failing())
        odd = len(self.distorting())
        if bad:
            return f"{bad}/{n} block(s) cannot be sampled as they stand; {odd} more would be distorted"
        if odd:
            return f"{n} block(s) samplable; {odd} would not reproduce the stated covariance without a repair"
        return f"{n} block(s) clean — samplable as stated, no repair needed"

    def __str__(self) -> str:
        parts = [self.summary(), ""]
        parts += [str(b) for b in self.blocks if b.findings]
        return "\n".join(parts).rstrip()

    def recommended_plan(self) -> "ConditioningPlan":
        """A default a human can accept or edit — never one applied unseen.

        **It repairs definiteness and nothing else.** That is a position, not an
        omission, and it is the one the three-stage design implies: a negative
        eigenvalue is the only finding a draw cannot decline to act on, because
        the decomposition projects the matrix somewhere PSD whether or not
        anyone chose to. Every other repair — capping a variance, rescaling an
        outlier bin, masking an inert row — changes what the evaluation states
        in order to make it more pleasant to sample, and which of those is
        admissible is a judgement about the evaluation. The pipelines make all
        six today, silently; the point of a plan is that they stop.

        So blocks with no definiteness finding get an explicit ``none`` step
        rather than no step. A plan that names every block says *this one was
        looked at and left alone*, which is what makes the applied log a record
        of the whole run rather than of its exceptions. What was deliberately
        left out is carried on :attr:`ConditioningPlan.notes`.

        The PSD choice follows ``psd_method="auto"`` — clip below the module
        threshold, Higham above it — because the plan's job here is to make the
        current behaviour visible, not to change it. The one override is a
        block carrying a sum rule: ``C·1 ≈ 0`` is destroyed by any congruence
        and degraded ~500x by Higham, so those get ``clip`` with the reason
        recorded.
        """
        steps: List[PlanStep] = []
        notes: List[str] = []

        for block in self.blocks:
            definiteness = next(
                (f for f in block.findings
                 if f.check == "definiteness" and f.severity == DISTORTS),
                None,
            )
            has_sum_rule = any(f.check == "sum_rule" for f in block.findings)

            if definiteness is None:
                # Including the case where definiteness came back at severity
                # `blocks` — a matrix with no positive eigenvalue at all states
                # no variance anywhere, and there is nothing for a projection to
                # project. It falls through to the unrepaired notes below.
                steps.append(PlanStep(
                    key=block.key, remedy="none",
                    reason="no PSD repair needed or none applies",
                ))
            else:
                choice = definiteness.evidence["auto_would_choose"]
                reason = (
                    f"{definiteness.summary}; psd_method='auto' resolves to "
                    f"{choice!r}"
                )
                if has_sum_rule and choice != "clip":
                    reason = (
                        f"{definiteness.summary}; auto would pick {choice!r}, "
                        "overridden because this block carries a sum rule and "
                        "clip is the only candidate that preserves C·1"
                    )
                    choice = "clip"
                steps.append(PlanStep(key=block.key, remedy=choice, reason=reason))

            for finding in block.findings:
                if finding.check in ("definiteness", "sum_rule") or not finding.remedies:
                    continue
                notes.append(
                    f"{block_key_text(block.key)}: {finding.check} "
                    f"({finding.severity}) left unrepaired — "
                    f"{finding.summary}. Available: "
                    + ", ".join(r.name for r in finding.remedies)
                )
            for finding in block.blocking:
                if not finding.remedies:
                    notes.append(
                        f"{block_key_text(block.key)}: {finding.check} blocks the "
                        f"draw and no repair is offered — {finding.summary}"
                    )

        return ConditioningPlan(steps=tuple(steps), notes=tuple(notes))

    def to_markdown(self) -> str:
        """A table, for pasting into a run log or a notebook cell.

        Summaries carry ``|ρ|``-style notation, so cell text is escaped — an
        unescaped pipe silently splits the row into extra columns and the table
        renders wrong rather than failing.
        """
        rows = [
            "| block | n | severity | check | summary |",
            "|---|---|---|---|---|",
        ]
        for block in self.blocks:
            if not block.findings:
                rows.append(f"| `{block.key}` | {block.order} | — | — | clean |")
                continue
            for finding in block.findings:
                rows.append(
                    f"| `{block.key}` | {block.order} | {finding.severity} "
                    f"| {finding.check} | {finding.summary.replace('|', r'\|')} |"
                )
        return "\n".join([self.summary(), "", *rows])


# ---------------------------------------------------------------------------
# The plan — stage 2 of inspect / decide / apply
# ---------------------------------------------------------------------------

def block_key_text(key: Hashable) -> str:
    """A block key as the text a plan is written and matched by.

    A plan is serialised to JSON and read back next to ``run_metadata.json``,
    and block keys are tuples of ints — ``(26056, 2, 1)`` — which JSON has no
    type for. Rather than inventing an encoding that round-trips tuples, a step
    identifies its block by this text and :func:`apply_plan` matches on it. The
    consequence is worth stating: **two blocks whose keys render alike are the
    same block as far as a plan is concerned**, which is why the function
    recurses into tuples instead of calling ``str`` on them. ``str`` of a tuple
    uses ``repr`` on the elements, so a numpy integer renders as
    ``np.int64(26056)`` under NumPy 2 and ``26056`` under NumPy 1, and a plan
    written on one machine would not match on the other.
    """
    if isinstance(key, tuple):
        return "(" + ", ".join(block_key_text(part) for part in key) + ")"
    if isinstance(key, np.generic):
        return str(key.item())
    return str(key)


def _jsonable(value):
    """NumPy scalars and arrays → the Python types ``json`` can write."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass(frozen=True)
class PlanStep:
    """One repair, on one block, with the parameters it runs with.

    *reason* is not decoration. A run directory that says ``clip`` records what
    was done; one that says ``clip`` *because* λ_min/λ_max was -1.4e-04 and the
    block carries a sum rule records why, and that is the difference between a
    reproducible run and a repeatable one.
    """

    key: Hashable
    remedy: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def __str__(self) -> str:
        head = f"{block_key_text(self.key)}: {self.remedy}"
        if self.parameters:
            head += "(" + ", ".join(f"{k}={v!r}" for k, v in self.parameters.items()) + ")"
        return f"{head}  — {self.reason}" if self.reason else head

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block": block_key_text(self.key),
            "remedy": self.remedy,
            "parameters": _jsonable(self.parameters),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanStep":
        return cls(
            key=data["block"],
            remedy=data["remedy"],
            parameters=dict(data.get("parameters") or {}),
            reason=data.get("reason", ""),
        )


@dataclass(frozen=True)
class ConditioningPlan:
    """The repairs a run is authorised to make, frozen before it starts.

    Stage 2 of the design: a human reads a :class:`ConditioningReport`, decides,
    and writes the decision down. The plan then goes in the run directory beside
    ``run_metadata.json`` and, per ``run-metadata-is-the-config-authority``, **it
    is the authority** — a script default that disagrees with a shipped plan is
    the script's bug, not the plan's.

    Two things it deliberately is not:

    * **Not a policy.** There is no "recommended" flag on a step and no severity
      here. By the time a plan exists the trade-offs have been decided; what is
      left is a list of operations. :meth:`ConditioningReport.recommended_plan`
      produces a starting point, and a human is expected to edit it.
    * **Not ordered by block.** Steps are applied in the order they are listed,
      per block, which matters: capping a variance and then projecting onto the
      PSD cone is not the same matrix as projecting and then capping.

    A round trip through :meth:`to_dict` / :meth:`from_dict` keeps the steps and
    their order. It does **not** keep the block key's Python type — see
    :func:`block_key_text` — because a plan identifies blocks by text.
    """

    steps: Tuple[PlanStep, ...] = ()
    notes: Tuple[str, ...] = ()

    def __str__(self) -> str:
        if not self.steps:
            return "empty plan — no block is authorised for any repair"
        lines = [str(step) for step in self.steps]
        if self.notes:
            lines += ["", "left unrepaired:"] + [f"  {note}" for note in self.notes]
        return "\n".join(lines)

    def steps_for(self, key: Hashable) -> Tuple[PlanStep, ...]:
        """The steps naming *key*, in plan order."""
        text = block_key_text(key)
        return tuple(s for s in self.steps if block_key_text(s.key) == text)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "steps": [step.to_dict() for step in self.steps],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConditioningPlan":
        version = data.get("version", 1)
        if version != 1:
            raise ValueError(
                f"conditioning plan version {version} is not readable by this "
                "version of kika; the plan is the authority, so this is a "
                "refusal rather than a best-effort read"
            )
        return cls(
            steps=tuple(PlanStep.from_dict(s) for s in data.get("steps", ())),
            notes=tuple(data.get("notes", ())),
        )


# ---------------------------------------------------------------------------
# Block normalisation
# ---------------------------------------------------------------------------

def as_blocks(blocks) -> List[Tuple[Hashable, np.ndarray]]:
    """Accept a mapping, a sequence of ``(key, matrix)`` pairs, or bare matrices.

    A pair is recognised by its second element being 2-D, which a key never is.
    Ordering is taken as given and never sorted: downstream, the block index
    sets the seed offset, so reordering the input must reorder the draws with
    it rather than silently repairing itself.
    """
    if hasattr(blocks, "items"):
        return [(key, np.asarray(m, dtype=float)) for key, m in blocks.items()]

    out: List[Tuple[Hashable, np.ndarray]] = []
    for index, entry in enumerate(blocks):
        if isinstance(entry, tuple) and len(entry) == 2 and np.ndim(entry[1]) == 2:
            key, matrix = entry
        else:
            key, matrix = index, entry
        out.append((key, np.asarray(matrix, dtype=float)))
    return out


# ---------------------------------------------------------------------------
# Remedy prediction
# ---------------------------------------------------------------------------

def _clipped(matrix: np.ndarray) -> np.ndarray:
    """``V·clip(Λ, 0)·Vᵀ`` — what ``psd_method="clip"`` does to a matrix.

    Was a fourth copy of the projection; now
    :func:`kika.cov.decomposition.clip_projection`, which is where the three
    ways the copies differed are written down.

    **This one is for measurement, and the sampling one is not.** It symmetrises
    and it rebuilds unconditionally, where ``svd_decomposition`` does neither —
    so a prediction made here and a draw made there can disagree in the last
    bits. That is the right way round: a prediction reporting ‖ΔC‖/‖C‖ to two
    significant figures does not care, and a draw that must reproduce a shipped
    ensemble does. :func:`apply_plan` uses the sampling form.
    """
    return clip_projection(matrix, floor=0.0, verbose=False)[0]


def _sum_rule_residual(matrix: np.ndarray) -> float:
    """``max|Σ_j C_ij| / max|C|`` — zero iff every row sums to zero.

    The quantity a normalised distribution's covariance has to keep. MF35
    covaries group-integrated probabilities that sum to one, so its covariance
    annihilates the all-ones vector; any repair that does not preserve that is
    disqualified for PFNS however good its other numbers look.
    """
    scale = float(np.max(np.abs(matrix))) if matrix.size else 0.0
    if scale <= 0.0:
        return 0.0
    return float(np.max(np.abs(matrix.sum(axis=1))) / scale)


def _effect(
    original: np.ndarray,
    repaired: np.ndarray,
    seconds: float,
    *,
    inert_floor: float = INERT_VARIANCE_FLOOR,
) -> Dict[str, Any]:
    """Measure one repair. Two diagonal numbers, and the reason there are two.

    A single "max relative change on the diagonal" is unreadable on real data:
    a PFNS band's lowest groups hold ~1e-17 of the total, so ``clip`` moving
    one of them by 1e-19 in absolute terms reports as 6.8e+15 %. True, and it
    tells you nothing about the variances the evaluation actually stated.

    So the diagonal is split at the same floor the pipelines use to decide a
    row is inert. ``stated_diagonal_max_relative_change`` covers rows the file
    gives real variance and answers "did my uncertainties move". ``inert_rows_
    gaining_variance`` counts rows the file gives none that came out with some,
    and answers the question that number was hiding — which is the whole reason
    ``clip_rescale`` was written.
    """
    diag_before = np.diag(original)
    diag_after = np.diag(repaired)
    largest_variance = float(np.max(np.abs(diag_before))) if diag_before.size else 0.0
    floor = inert_floor * largest_variance
    stated = np.abs(diag_before) >= floor if largest_variance > 0 else np.ones_like(diag_before, bool)

    if np.any(stated):
        moved = float(np.max(
            np.abs(diag_after[stated] - diag_before[stated]) / np.abs(diag_before[stated])
        ))
    else:
        moved = 0.0

    inert = ~stated if largest_variance > 0 else np.zeros_like(diag_before, bool)
    gained = int(np.count_nonzero(inert & (np.abs(diag_after) >= floor)))
    # The *change*, not the resulting value — those differ, and reporting the
    # value instead credits a repair that restores a tiny stated variance with
    # having invented it. Reported absolutely as well as by count, because the
    # count almost never fires and the absolute number almost always matters:
    # on the Cf-252 bands ``clip`` adds ~1e-18 of variance to rows whose stated
    # variance is ~7e-17, which is nothing against a largest stated variance of
    # 7e-5 — and a perturbation of order one once PFNS divides a drawn delta by
    # a group probability of the same size. Whether that is negligible is a
    # question about the caller's downstream, so the caller gets the number.
    added = (
        float(np.max(np.abs(diag_after[inert] - diag_before[inert])))
        if np.any(inert) else 0.0
    )

    frobenius = float(np.linalg.norm(original))
    spectrum = np.linalg.eigvalsh(repaired)
    largest = float(spectrum.max()) if spectrum.size else 0.0
    return {
        "stated_diagonal_max_relative_change": moved,
        "inert_rows_gaining_variance": gained,
        "max_variance_added_to_an_inert_row": added,
        "n_inert_rows": int(np.count_nonzero(inert)),
        "frobenius_relative_change": (
            float(np.linalg.norm(repaired - original) / frobenius) if frobenius > 0 else 0.0
        ),
        "min_eigenvalue_ratio": float(spectrum.min() / largest) if largest > 0 else 0.0,
        "sum_rule_residual_after": _sum_rule_residual(repaired),
        "seconds": seconds,
    }


def predict_psd_repairs(
    matrix: np.ndarray,
    *,
    candidates: Sequence[str] = DEFAULT_PSD_CANDIDATES,
    higham_max_iter: int = _PREDICT_HIGHAM_MAX_ITER,
    inert_floor: float = INERT_VARIANCE_FLOOR,
) -> Tuple[Remedy, ...]:
    """Run each candidate PSD repair on *matrix* and measure what it did.

    This is the function the module exists for. It answers "which projection
    should this covariance use" with numbers from this covariance rather than
    from the literature, because the properties that decide it — does the
    diagonal have to be exact, does a sum rule have to survive, is the matrix
    small enough for Higham to terminate — are properties of the data.

    The repaired matrices are measured and discarded; nothing is returned but
    the measurements. ``higham_max_iter`` caps the *prediction* only, and a
    prediction that hit the cap reports ``converged=False`` rather than
    pretending the number is the converged one.
    """
    original = np.asarray(matrix, dtype=float)
    remedies: List[Remedy] = []

    for name in candidates:
        start = time.perf_counter()
        info: Dict[str, Any] = {}
        if name == "clip":
            repaired = _clipped(original)
            note = "preserves eigenvectors, not the diagonal"
        elif name == "clip_rescale":
            repaired, info = clip_and_rescale(original, verbose=False)
            note = (
                "preserves the diagonal exactly; a congruence, so it moves C·1. "
                "Defined only on a non-negative diagonal — its scale factor is "
                "sqrt(diag(C₀)/diag(C_clip)), so a row whose stated variance is "
                "negative comes back identically zero rather than repaired. Read "
                "this remedy together with the negative_variance check"
            )
        elif name == "higham":
            repaired, info = nearest_psd_higham(
                original, preserve_diagonal=True, max_iter=higham_max_iter,
                verbose=False,
            )
            note = "nearest PSD holding the diagonal fixed; iterative, and the slow one"
        elif name == "none":
            repaired = original
            note = "no repair — only admissible if the matrix is already PSD"
        else:
            raise ValueError(
                f"unknown PSD repair {name!r}; expected one of "
                "'clip', 'clip_rescale', 'higham', 'none'"
            )
        effect = _effect(original, repaired, time.perf_counter() - start, inert_floor=inert_floor)
        effect["sum_rule_residual_before"] = _sum_rule_residual(original)
        effect["predicted"] = True
        if name == "higham":
            effect["converged"] = bool(info.get("converged", False))
            effect["iterations"] = info.get("iterations")
        remedies.append(Remedy(name=name, effect=effect, note=note))

    return tuple(remedies)


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------

def _check_finite(matrix: np.ndarray) -> Optional[Finding]:
    bad = ~np.isfinite(matrix)
    count = int(bad.sum())
    if not count:
        return None
    rows = np.flatnonzero(bad.any(axis=1))
    return Finding(
        check="finiteness",
        severity=BLOCKS,
        summary=f"{count} non-finite entr{'y' if count == 1 else 'ies'} across {rows.size} row(s)",
        evidence={"n_non_finite": count, "rows": rows.tolist()[:64]},
        remedies=(
            Remedy(
                name="zero_non_finite",
                note=(
                    "what every sampling path already does silently. Correct when the "
                    "entries are couplings the processing code never computed; wrong "
                    "when they are a stated covariance that failed to parse — those two "
                    "look identical here and the distinction is the caller's"
                ),
            ),
        ),
    )


def _check_symmetry(matrix: np.ndarray) -> Optional[Finding]:
    scale = float(np.max(np.abs(matrix))) if matrix.size else 0.0
    if scale <= 0.0:
        return None
    asymmetry = float(np.max(np.abs(matrix - matrix.T)) / scale)
    if asymmetry <= 1e-12:
        return None
    severity = DISTORTS if asymmetry < 1e-6 else BLOCKS
    return Finding(
        check="symmetry",
        severity=severity,
        summary=f"max|C - Cᵀ|/max|C| = {asymmetry:.2e}",
        evidence={"relative_asymmetry": asymmetry},
        remedies=(
            Remedy(
                name="symmetrise",
                note=(
                    "(C + Cᵀ)/2. Below ~1e-12 this is float64 round-off from the "
                    "reconstruction and symmetrising is free; well above it, half of a "
                    "genuinely asymmetric matrix has been discarded and the question is "
                    "which half was right"
                ),
            ),
        ),
    )


def _check_negative_variance(matrix: np.ndarray) -> Optional[Finding]:
    diagonal = np.diag(matrix)
    bad = np.isfinite(diagonal) & (diagonal < 0.0)
    count = int(bad.sum())
    if not count:
        return None
    return Finding(
        check="negative_variance",
        severity=BLOCKS,
        summary=f"{count} stated variance(s) below zero, smallest {float(diagonal[bad].min()):.3e}",
        evidence={"n_negative": count, "indices": np.flatnonzero(bad).tolist()[:64]},
        remedies=(),  # deliberately none: σ² < 0 is a data error, not a conditioning one
    )


def _check_correlation_bound(matrix: np.ndarray) -> Optional[Finding]:
    diagonal = np.diag(matrix)
    usable = np.isfinite(diagonal) & (diagonal > 0.0)
    if usable.sum() < 2:
        return None
    sub = matrix[np.ix_(usable, usable)]
    sigma = np.sqrt(np.diag(sub))
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = sub / np.outer(sigma, sigma)
    finite = np.isfinite(correlation)
    excess = np.abs(correlation) - 1.0
    bad = finite & (excess > 1e-10)
    count = int(bad.sum())
    if not count:
        return None
    worst = float(np.max(np.abs(correlation[bad])))
    return Finding(
        check="correlation_bound",
        severity=BLOCKS,
        summary=f"{count} off-diagonal correlation(s) outside [-1, 1], worst |ρ| = {worst:.4f}",
        evidence={"n_out_of_range": count, "max_abs_correlation": worst},
        remedies=(
            Remedy(
                name="renormalise",
                note=(
                    "clip ρ to [-1, 1] and rebuild. Almost always the symptom of an "
                    "INTG-packed correlation matrix reconstructed at the stated NDIGIT "
                    "and then read back at a different one, in which case the covariance "
                    "was never the one the file meant"
                ),
            ),
        ),
    )


def _unmeasured(name: str, reason: str, note: str) -> Remedy:
    return Remedy(name=name, effect={"predicted": False, "reason": reason}, note=note)


def _spectrum(matrix: np.ndarray) -> Optional[np.ndarray]:
    """The eigenvalues, computed once for every check that wants them.

    Three checks here need the spectrum — definiteness, rank, and the auto
    resolution inside the first. Computing it per check costs three O(n³)
    decompositions, which is invisible at n = 122 and is the entire runtime of
    an inspection at n = 2781 (Th-232's resolved range). A pre-flight expensive
    enough that people skip it does not get run.

    Returns None when the matrix has non-finite entries: ``eigvalsh`` would
    raise, and ``_check_finite`` has already reported the real problem.
    """
    if not np.isfinite(matrix).all():
        return None
    spectrum = np.linalg.eigvalsh(matrix)
    return spectrum if spectrum.size else None


def _check_definiteness(
    matrix: np.ndarray,
    *,
    spectrum: Optional[np.ndarray],
    predict: bool,
    candidates: Sequence[str],
    inert_floor: float,
) -> Optional[Finding]:
    if spectrum is None:
        return None
    largest = float(spectrum.max())
    smallest = float(spectrum.min())
    if largest <= 0.0:
        return Finding(
            check="definiteness",
            severity=BLOCKS,
            summary="no positive eigenvalue — the block states no variance anywhere",
            evidence={"max_eigenvalue": largest},
        )
    order = matrix.shape[0]
    ratio = -smallest / largest
    if ratio <= order * ROUNDOFF_RATIO:
        return None

    if not predict:
        remedies = [
            _unmeasured(
                name,
                f"order {order} exceeds PREDICT_MAX_ORDER ({PREDICT_MAX_ORDER})",
                "",
            )
            for name in candidates
        ]
        skipped = "; remedies not measured at this order"
    else:
        skipped = ""
        # Higham alone is O(iterations x n^3), so it gets its own ceiling
        # rather than being dropped from the whole prediction at n = 251.
        affordable = [
            name for name in candidates
            if name != "higham" or order <= PREDICT_HIGHAM_MAX_ORDER
        ]
        remedies = list(predict_psd_repairs(
            matrix, candidates=affordable, inert_floor=inert_floor,
        ))
        remedies += [
            _unmeasured(
                name,
                f"order {order} exceeds PREDICT_HIGHAM_MAX_ORDER "
                f"({PREDICT_HIGHAM_MAX_ORDER}); measure it deliberately with "
                "predict_psd_repairs(matrix, candidates=('higham',))",
                "",
            )
            for name in candidates if name not in affordable
        ]

    would_auto = "clip" if ratio < _PSD_AUTO_THRESHOLD else "higham"
    return Finding(
        check="definiteness",
        severity=DISTORTS,
        summary=(
            f"not PSD: λ_min/λ_max = -{ratio:.3e} "
            f"(psd_method='auto' would pick '{would_auto}')" + skipped
        ),
        evidence={
            "min_eigenvalue": smallest,
            "max_eigenvalue": largest,
            "negativity_ratio": ratio,
            "n_negative": int((spectrum < 0.0).sum()),
            "auto_would_choose": would_auto,
        },
        remedies=tuple(remedies),
    )


def _check_rank(
    matrix: np.ndarray, *, spectrum: Optional[np.ndarray], null_tol: float
) -> Optional[Finding]:
    if spectrum is None:
        return None
    largest = float(spectrum.max())
    if largest <= 0.0:
        return None
    n = matrix.shape[0]
    rank = int((spectrum > null_tol * largest).sum())
    if rank == n:
        return None
    return Finding(
        check="rank",
        severity=NOTE,
        summary=f"rank {rank} of {n} — {n - rank} null direction(s) at tol {null_tol:.0e}",
        evidence={"rank": rank, "order": n, "null_fraction": (n - rank) / n},
        remedies=(
            Remedy(
                name="truncate",
                parameters={"null_tol": null_tol},
                note=(
                    "draw in the retained subspace only. draw_samples already does this "
                    "and generate_samples does not; it is what stops null-direction "
                    "debris from becoming a perturbation where the stated σ is zero"
                ),
            ),
        ),
    )


def inert_row_mask(
    matrix: np.ndarray, *, floor: float = INERT_VARIANCE_FLOOR
) -> np.ndarray:
    """Boolean mask of the rows whose variance the evaluation does not state.

    ``True`` where the diagonal is non-finite or below *floor* in magnitude —
    a reaction below threshold, or a bin the processing code never computed.

    One predicate, three callers, and that is the point: the ``inert_rows``
    finding, the ``mask_inert`` remedy and the draw's own drop all have to
    agree about which rows are inert, and until this existed they agreed by
    each carrying the same expression.
    """
    diagonal = np.diag(matrix)
    return ~(np.isfinite(diagonal) & (np.abs(diagonal) >= floor))


def _check_inert_rows(matrix: np.ndarray, *, floor: float) -> Optional[Finding]:
    inert = inert_row_mask(matrix, floor=floor)
    count = int(inert.sum())
    if not count:
        return None
    severity = BLOCKS if count == matrix.shape[0] else NOTE
    summary = f"{count}/{matrix.shape[0]} row(s) state no variance (σ² < {floor:.0e})"
    if severity is BLOCKS:
        summary += " — the whole block is inert"
    return Finding(
        check="inert_rows",
        severity=severity,
        summary=summary,
        evidence={"n_inert": count, "indices": np.flatnonzero(inert).tolist()[:64], "floor": floor},
        remedies=(
            Remedy(
                name="mask_inert",
                parameters={"floor": floor},
                note=(
                    "drop the rows before decomposing and pin them to no perturbation. "
                    "generate_samples does this unconditionally; it is wrong wherever a "
                    "sum rule couples the dropped rows to the kept ones, because the "
                    "identity holds on the full vector and not on the retained subspace"
                ),
            ),
        ),
    )


def _check_variance_outliers(
    matrix: np.ndarray,
    *,
    families: Optional[Sequence[Hashable]],
    factor: float,
    min_family_size: int,
) -> Optional[Finding]:
    diagonal = np.diag(matrix)
    usable = np.isfinite(diagonal) & (diagonal > 0.0)
    if usable.sum() < min_family_size:
        return None

    labels = (
        np.asarray(["*"] * matrix.shape[0], dtype=object)
        if families is None
        else np.asarray(list(families), dtype=object)
    )
    if labels.shape[0] != matrix.shape[0]:
        raise ValueError(
            f"families has {labels.shape[0]} labels for a {matrix.shape[0]}-row block"
        )

    flagged: List[int] = []
    detail: List[Dict[str, Any]] = []
    for label in dict.fromkeys(labels.tolist()):
        member = (labels == label) & usable
        if int(member.sum()) < min_family_size:
            continue
        values = diagonal[member]
        median = float(np.median(values))
        if median <= 0.0:
            continue
        over = np.flatnonzero(member & (diagonal > factor * median))
        for index in over.tolist():
            flagged.append(index)
            detail.append({
                "index": index,
                "family": label,
                "variance": float(diagonal[index]),
                "family_median": median,
                "ratio": float(diagonal[index] / median),
            })

    if not flagged:
        return None
    worst = max(d["ratio"] for d in detail)
    return Finding(
        check="variance_outliers",
        severity=DISTORTS,
        summary=(
            f"{len(flagged)} variance(s) above {factor:g}x their family median, "
            f"worst {worst:.2e}x"
        ),
        evidence={"n_flagged": len(flagged), "detail": detail[:64], "factor": factor},
        remedies=(
            Remedy(
                name="rescale_to_family_median",
                parameters={"indices": flagged, "targets": [d["family_median"] for d in detail]},
                note=(
                    "one-sided congruence to the median — preserves correlations and PSD. "
                    "rescale_threshold_bins_congruence takes exactly these two lists"
                ),
            ),
            Remedy(
                name="cap",
                note=(
                    "one global ceiling on every variance, via the same congruence. "
                    "Coarser, and it is what runs today as max_relative_std"
                ),
            ),
        ),
    )


def _check_sum_rule(matrix: np.ndarray, *, tol: float) -> Optional[Finding]:
    if not np.isfinite(matrix).all():
        return None
    residual = _sum_rule_residual(matrix)
    if residual > tol:
        return None
    return Finding(
        check="sum_rule",
        severity=NOTE,
        summary=f"C·1 ≈ 0 to {residual:.1e} — this block covaries a normalised quantity",
        evidence={"residual": residual, "tol": tol},
        remedies=(
            Remedy(
                name="preserve",
                note=(
                    "constrains the PSD choice rather than offering one: the all-ones "
                    "vector is a null direction, so clip (which fixes eigenvectors) keeps "
                    "the identity and clip_rescale (a congruence, which maps 1 to D·1) "
                    "does not. Measured at ~500x degradation on the Cf-252 MF35 bands"
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def inspect_matrix(
    matrix,
    *,
    key: Hashable = 0,
    families: Optional[Sequence[Hashable]] = None,
    square_form: bool = True,
    null_tol: float = DEFAULT_NULL_TOL,
    inert_floor: float = INERT_VARIANCE_FLOOR,
    outlier_factor: float = OUTLIER_FACTOR,
    min_family_size: int = 4,
    sum_rule_tol: float = SUM_RULE_TOL,
    psd_candidates: Sequence[str] = DEFAULT_PSD_CANDIDATES,
    predict: bool = True,
    predict_max_order: int = PREDICT_MAX_ORDER,
) -> BlockReport:
    """Run every check against one covariance matrix.

    Parameters
    ----------
    matrix :
        Square, in whatever space the caller intends to sample in. Checks that
        depend on the space — the outlier factor most of all — are calibrated
        for a *relative* covariance, which is what every pipeline here hands to
        a sampler.
    families :
        One label per row saying which rows are comparable. Outlier detection
        takes medians within a family, so passing ``(zaid, mt)`` per row asks
        "large for its own reaction" and passing nothing asks "large for this
        block", which on a multi-reaction block is the wrong question.
    square_form :
        Whether the rows and the columns index the *same* quantities. False for
        a cross block — MF34's L=1 against L=2, or an MF33 MT2-against-MT102
        block — where the matrix is an off-diagonal partition of a larger
        covariance. Almost every check assumes square form, because on a cross
        block the diagonal is not a variance, asymmetry is expected and the
        spectrum means nothing; passing False runs only the checks that still
        apply and says why the rest were skipped.
    psd_candidates :
        Which PSD repairs to measure. See :data:`DEFAULT_PSD_CANDIDATES`.
    predict :
        Whether to run those repairs at all. Turning it off makes the
        inspection cheap and leaves the definiteness finding with unmeasured
        options; blocks over ``predict_max_order`` turn it off themselves.

    Returns
    -------
    BlockReport
        Findings ordered by severity, then by the order the checks run in.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {matrix.shape}")
    if square_form and matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"expected a square matrix, got shape {matrix.shape} — a rectangular "
            "block is a cross covariance, so pass square_form=False"
        )

    if not square_form:
        candidates = [
            _check_finite(matrix),
            Finding(
                check="cross_block",
                severity=NOTE,
                summary=(
                    "rows and columns index different quantities — not a covariance "
                    "in its own right and never samplable alone"
                ),
                evidence={"shape": tuple(int(s) for s in matrix.shape)},
                remedies=(
                    Remedy(
                        name="assemble",
                        note=(
                            "it conditions only inside the joint matrix its row and "
                            "column blocks belong to. Every check but finiteness is "
                            "skipped here because there is nothing for them to mean: "
                            "the diagonal is not a variance and the spectrum is not "
                            "a spectrum of anything"
                        ),
                    ),
                ),
            ),
        ]
        findings = [f for f in candidates if f is not None]
        findings.sort(key=lambda f: _SEVERITY_ORDER[f.severity])
        return BlockReport(key=key, order=int(matrix.shape[0]), findings=tuple(findings))

    do_predict = predict and matrix.shape[0] <= predict_max_order
    spectrum = _spectrum(matrix)
    candidates = [
        _check_finite(matrix),
        _check_negative_variance(matrix),
        _check_symmetry(matrix),
        _check_correlation_bound(matrix),
        _check_definiteness(
            matrix, spectrum=spectrum, predict=do_predict,
            candidates=psd_candidates, inert_floor=inert_floor,
        ),
        _check_variance_outliers(
            matrix, families=families, factor=outlier_factor,
            min_family_size=min_family_size,
        ),
        _check_inert_rows(matrix, floor=inert_floor),
        _check_rank(matrix, spectrum=spectrum, null_tol=null_tol),
        _check_sum_rule(matrix, tol=sum_rule_tol),
    ]
    findings = [f for f in candidates if f is not None]
    findings.sort(key=lambda f: _SEVERITY_ORDER[f.severity])
    return BlockReport(key=key, order=int(matrix.shape[0]), findings=tuple(findings))


def _per_block(value, key, default=None):
    """Resolve a per-block argument that may be one value or a key->value map."""
    if value is None:
        return default
    if hasattr(value, "get") and not isinstance(value, (str, bytes)):
        resolved = value.get(key)
        return default if resolved is None else resolved
    return value


def inspect_blocks(blocks, *, families=None, square_form=None, **kwargs) -> ConditioningReport:
    """Run :func:`inspect_matrix` over a set of blocks.

    *blocks* takes the same three shapes :func:`kika.sampling.core.draw_samples`
    accepts — a mapping, ``(key, matrix)`` pairs, or bare matrices — so a
    caller can inspect exactly what it is about to sample without rearranging
    it. *families* and *square_form* may each be a single value applied to
    every block, or a mapping from block key to that block's value.

    Usage is the two-step the pipelines do not have today::

        report = inspect_blocks(blocks)
        print(report)                       # decide, in front of a human
        if not report.samplable:
            raise SystemExit(report.summary())
    """
    reports = [
        inspect_matrix(
            matrix,
            key=key,
            families=_per_block(families, key),
            square_form=_per_block(square_form, key, default=True),
            **kwargs,
        )
        for key, matrix in as_blocks(blocks)
    ]
    return ConditioningReport(blocks=tuple(reports))


# ---------------------------------------------------------------------------
# Stage 3 — apply
# ---------------------------------------------------------------------------

#: The remedies :func:`apply_plan` can carry out, and what each delegates to.
#:
#: ``preserve`` is absent on purpose. It is what the ``sum_rule`` finding
#: offers, and it is a *constraint on the other choices* — "whatever you pick,
#: it must not move C·1" — not an operation. A plan naming it is a plan whose
#: author read a constraint as an instruction, so it is refused by name rather
#: than silently doing nothing.
APPLIABLE_REMEDIES = (
    "none", "clip", "clip_rescale", "higham", "cap",
    "rescale_to_family_median", "mask_inert",
)


def _apply_step(matrix: np.ndarray, step: PlanStep, *, verbose: bool, logger):
    """One step. Returns ``(matrix, info)``; the arithmetic is not new here."""
    parameters = dict(step.parameters)
    remedy = step.remedy

    if remedy == "none":
        return matrix, {}

    if remedy == "clip":
        # Exactly `svd_decomposition`'s clip: floor 0.0, no fold-up to
        # symmetry, and an already-PSD matrix passes through untouched. That
        # last one is what makes conditioning-then-`psd_method="none"`
        # bit-identical to `psd_method="clip"` rather than merely equal.
        return clip_projection(
            matrix, floor=0.0, symmetrise=False, passthrough_when_clean=True,
            verbose=verbose, logger=logger, label="plan clip",
        )

    if remedy == "clip_rescale":
        return clip_and_rescale(matrix, verbose=verbose, logger=logger, **parameters)

    if remedy == "higham":
        parameters.setdefault("preserve_diagonal", True)
        return nearest_psd_higham(matrix, verbose=verbose, logger=logger, **parameters)

    if remedy == "cap":
        if "max_variance" not in parameters:
            raise ValueError(
                f"step {step} needs a max_variance parameter — a cap with no "
                "ceiling is not a repair"
            )
        return cap_variance_congruence(
            matrix, parameters.pop("max_variance"),
            verbose=verbose, logger=logger, **parameters,
        )

    if remedy == "rescale_to_family_median":
        missing = {"indices", "targets"} - set(parameters)
        if missing:
            raise ValueError(
                f"step {step} needs {sorted(missing)}; the `variance_outliers` "
                "finding carries both on its remedy"
            )
        indices = parameters.pop("indices")
        targets = parameters.pop("targets")
        return rescale_threshold_bins_congruence(
            matrix, indices, targets, verbose=verbose, logger=logger, **parameters,
        )

    if remedy == "mask_inert":
        # **Equivalent in effect to what `generate_samples` does, and not
        # bit-identical to it.** That path drops the inert rows and decomposes
        # the (n-k) submatrix; this one zeroes them and decomposes the full n,
        # which leaves the same retained subspace but hands LAPACK a different
        # matrix, so the last bits of the eigenvectors move. Zeroing is the
        # form that belongs in a plan: a block's key indexes its rows, and a
        # step that changed the block's order would invalidate every other
        # step's indices and the parameter mapping the appliers read.
        floor = float(parameters.pop("floor", INERT_VARIANCE_FLOOR))
        if parameters:
            raise ValueError(f"mask_inert takes only `floor`; got {sorted(parameters)}")
        # The same absolute test `_check_inert_rows` makes, so the finding and
        # the repair cannot disagree about which rows are inert.
        inert = inert_row_mask(matrix, floor=floor)
        masked = matrix.copy()
        masked[inert, :] = 0.0
        masked[:, inert] = 0.0
        return masked, {"n_masked": int(inert.sum()), "floor": floor}

    if remedy == "preserve":
        raise ValueError(
            "`preserve` is a constraint the sum_rule finding places on the PSD "
            "choice, not an operation — it says clip is admissible and a "
            "congruence is not. Put that choice in the plan instead"
        )

    raise ValueError(
        f"unknown remedy {step.remedy!r}; {sorted(APPLIABLE_REMEDIES)} are appliable"
    )


def apply_plan(
    blocks,
    plan: ConditioningPlan,
    *,
    verbose: bool = False,
    logger=None,
) -> Tuple[List[Tuple[Hashable, np.ndarray]], Tuple[Dict[str, Any], ...]]:
    """Stage 3: carry out *plan* on *blocks*, and say what was carried out.

    The new code here is the dispatch. Every repair is the applier that runs
    today in :mod:`kika.cov.decomposition`, called with the parameters a human
    approved instead of with whatever the sampler's defaults were — which is the
    whole of the change, and the reason a conditioned draw can be compared
    against an unconditioned one bit for bit.

    Blocks come back **in the order they went in**, repaired or not. That is not
    cosmetic: downstream the block index sets the seed offset, so a function
    that dropped an untouched block or sorted the output would silently reseed
    every draw after it.

    A plan step naming a block that is not here **raises**. A plan is written
    against a specific set of matrices, and one that half-fits is far more
    likely to be a plan from another run than a plan with a typo — applying the
    steps that happen to match would then condition some blocks and quietly
    leave others raw.

    Returns
    -------
    (blocks, applied)
        *applied* has one record per step actually run, with the applier's own
        info dict and two cheap measurements: how far the stated diagonal moved
        and how far the whole matrix moved. **The spectrum is not measured
        here** — that is another O(n³) per block, and the honest check that a
        plan did what it was chosen for is to run :func:`inspect_blocks` on the
        result, which reports it against every other criterion at the same time.
    """
    normalised = as_blocks(blocks)
    present = {block_key_text(key) for key, _ in normalised}
    unmatched = [
        step for step in plan.steps if block_key_text(step.key) not in present
    ]
    if unmatched:
        raise ValueError(
            "the plan names block(s) that are not in this set: "
            + ", ".join(sorted({block_key_text(s.key) for s in unmatched}))
            + f"; blocks present: {sorted(present)}"
        )

    conditioned: List[Tuple[Hashable, np.ndarray]] = []
    applied: List[Dict[str, Any]] = []

    for key, matrix in normalised:
        current = matrix
        for step in plan.steps_for(key):
            start = time.perf_counter()
            repaired, info = _apply_step(current, step, verbose=verbose, logger=logger)
            seconds = time.perf_counter() - start

            diagonal_before, diagonal_after = np.diag(current), np.diag(repaired)
            largest = float(np.max(np.abs(diagonal_before))) if diagonal_before.size else 0.0
            stated = (
                np.abs(diagonal_before) >= INERT_VARIANCE_FLOOR * largest
                if largest > 0 else np.zeros_like(diagonal_before, bool)
            )
            moved = (
                float(np.max(np.abs(
                    (diagonal_after[stated] - diagonal_before[stated])
                    / diagonal_before[stated]
                ))) if np.any(stated) else 0.0
            )
            norm = float(np.linalg.norm(current))
            applied.append({
                "block": block_key_text(key),
                "remedy": step.remedy,
                "parameters": _jsonable(step.parameters),
                "reason": step.reason,
                "n": int(current.shape[0]),
                "changed": bool(repaired is not current),
                "stated_diagonal_max_relative_change": moved,
                "frobenius_relative_change": (
                    float(np.linalg.norm(repaired - current) / norm) if norm > 0 else 0.0
                ),
                "seconds": seconds,
                "applier_info": _jsonable(info),
            })
            current = repaired
        conditioned.append((key, current))

    return conditioned, tuple(applied)
