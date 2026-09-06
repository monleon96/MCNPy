"""A run's own account of itself: events, not lines.

The perturbation pipelines this library grew up with logged the way most
pipelines do -- a ``logging.FileHandler`` and a few hundred f-strings -- and the
result was a file a person could grep and nothing could read. Nothing in it
said which sample a line was about, how long a stage took, or which block a
warning concerned; that had to be recovered from the shape of the text.

This module records **events** instead. An :class:`Event` says what happened
(*kind*), to what (*subject*, *sample*), when, for how long, and carries a
JSON payload with the numbers -- dimensions, rank, λ_min/λ_max, realised
covariance error, the files written. The same list of events is written twice
from one source: ``run.log.jsonl`` for a program (the app's timeline, a
dashboard, a test) and ``run.log`` rendered for a person. Neither is derived
from the other, so they cannot disagree.

The Python ``logging`` interface is kept: a :class:`RunLog` given a *logger*
forwards every event to it as one line, so a caller that already has a log
file keeps getting one.

Kinds are a closed set (:data:`EVENT_KINDS`). A consumer that filters on
``kind == "drawn"`` has to be able to rely on the spelling.
"""
from __future__ import annotations

import datetime as _dt
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

__all__ = ["Event", "RunLog", "EVENT_KINDS", "LEVELS"]

#: What an event can be. In the order a run produces them, roughly.
EVENT_KINDS = (
    "started",      # the call, with its arguments
    "read",         # the source file decoded to the model
    "request",      # what the request became once normalised
    "assembled",    # the covariance blocks, with their dimensions
    "inspected",    # the pre-flight's findings, per block
    "conditioned",  # one repair step actually applied
    "drawn",        # one block drawn, with its diagnostics
    "applied",      # one realisation put on the model
    "checked",      # a check on a realisation (positivity, sum rule, normalisation)
    "emitted",      # one file written for one sample
    "written",      # a run-level file written (factors table, plan, metadata)
    "note",         # something true of the run that no output file states
    "warning",
    "error",
    "finished",
)

LEVELS = ("info", "warning", "error")


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def _secs(seconds: Optional[float]) -> str:
    return "" if seconds is None else f"{seconds:.3f} s" if seconds < 10 else f"{seconds:.1f} s"


def _size(nbytes) -> str:
    if nbytes is None:
        return ""
    n = float(nbytes)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{nbytes} B"


def _short(subject: str) -> str:
    """``MF34/MT2 (ZA 26056, L=1)`` -> ``MF34/MT2 L=1``: the ZA is in the header."""
    text = subject
    if "(ZA " in text:
        head, _, tail = text.partition("(ZA ")
        inner = tail.rstrip(")")
        extra = inner.split(",", 1)[1].strip() if "," in inner else ""
        text = head.strip() + (f" {extra}" if extra else "")
    return text


def _jsonable(value):
    """The payload as JSON can hold it. Numpy scalars and arrays included."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class Event:
    """One thing that happened during a run."""

    kind: str
    message: str
    at: str
    level: str = "info"
    subject: Optional[str] = None
    sample: Optional[int] = None
    seconds: Optional[float] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind, "level": self.level, "at": self.at,
            "sample": self.sample, "subject": self.subject,
            "seconds": self.seconds, "message": self.message,
            "payload": _jsonable(self.payload),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        return cls(kind=data["kind"], message=data.get("message", ""),
                   at=data.get("at", ""), level=data.get("level", "info"),
                   subject=data.get("subject"), sample=data.get("sample"),
                   seconds=data.get("seconds"),
                   payload=dict(data.get("payload") or {}))

    def render(self) -> str:
        """One line for a person. The clock, the kind, who, what, how long."""
        clock = self.at[11:23] if len(self.at) >= 23 else self.at
        mark = {"info": " ", "warning": "!", "error": "E"}[self.level]
        where = ""
        if self.sample is not None:
            where += f"#{self.sample:04d} "
        if self.subject:
            where += f"{self.subject} "
        tail = f"  ({self.seconds:.3f} s)" if self.seconds is not None else ""
        scalars = {
            key: value for key, value in self.payload.items()
            if isinstance(value, (int, float, str, bool)) and not isinstance(value, bool)
            or isinstance(value, bool)
        }
        detail = ""
        if scalars:
            bits = []
            for key, value in scalars.items():
                if isinstance(value, float):
                    bits.append(f"{key}={value:.4g}")
                else:
                    bits.append(f"{key}={value}")
            detail = "  | " + ", ".join(bits)
        return f"{clock} {mark} {self.kind:<11} {where}{self.message}{tail}{detail}"


class RunLog:
    """The events of one run, in order, and the two files they become.

    Parameters
    ----------
    logger
        An optional ``logging.Logger`` (or anything with ``info``/``warning``/
        ``error``). Every event is forwarded to it as its rendered line, so
        the old interface keeps working for a caller that has one.
    """

    def __init__(self, logger=None, *, label: str = "") -> None:
        self.label = label
        self.events: List[Event] = []
        self._logger = logger
        self._started = time.perf_counter()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def event(self, kind: str, message: str, *, subject: Optional[str] = None,
              sample: Optional[int] = None, seconds: Optional[float] = None,
              level: str = "info", **payload) -> Event:
        if kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {kind!r}; known: {EVENT_KINDS}")
        if level not in LEVELS:
            raise ValueError(f"unknown level {level!r}; known: {LEVELS}")
        if kind in ("warning", "error") and level == "info":
            level = kind
        event = Event(kind=kind, message=message, at=_now(), level=level,
                      subject=None if subject is None else str(subject),
                      sample=sample, seconds=seconds, payload=dict(payload))
        self.events.append(event)
        self._forward(event)
        return event

    def note(self, message: str, **payload) -> Event:
        return self.event("note", message, **payload)

    def warning(self, message: str, *, subject=None, sample=None, **payload) -> Event:
        return self.event("warning", message, subject=subject, sample=sample,
                          level="warning", **payload)

    def error(self, message: str, *, subject=None, sample=None, **payload) -> Event:
        return self.event("error", message, subject=subject, sample=sample,
                          level="error", **payload)

    @contextmanager
    def timed(self, kind: str, message: str, *, subject=None, sample=None,
              **payload) -> Iterator[Dict[str, Any]]:
        """Record *kind* when the block finishes, with how long it took.

        Yields a dict the caller may fill with payload as it learns things;
        whatever is in it at exit is recorded. An exception inside becomes an
        ``error`` event naming the stage and is re-raised, so a run that died
        says where.
        """
        extra: Dict[str, Any] = {}
        start = time.perf_counter()
        try:
            yield extra
        except Exception as failure:
            self.error(f"{message}: {type(failure).__name__}: {failure}",
                       subject=subject, sample=sample,
                       seconds=time.perf_counter() - start, stage=kind)
            raise
        self.event(kind, message, subject=subject, sample=sample,
                   seconds=time.perf_counter() - start, **payload, **extra)

    def absorb(self, events) -> None:
        """Take in events recorded elsewhere, as they were.

        A worker process keeps its own :class:`RunLog` and sends the events
        back; they are appended here with their own timestamps and forwarded
        to the logger like anything recorded directly. Accepts
        :class:`Event` objects or their ``to_dict`` form.
        """
        for item in events:
            event = item if isinstance(item, Event) else Event.from_dict(item)
            self.events.append(event)
            self._forward(event)

    def _forward(self, event: Event) -> None:
        if self._logger is None:
            return
        line = event.render()
        if event.level == "error":
            self._logger.error(line)
        elif event.level == "warning":
            self._logger.warning(line)
        else:
            self._logger.info(line)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Event]:
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    def of(self, kind: str) -> List[Event]:
        return [event for event in self.events if event.kind == kind]

    def problems(self) -> List[Event]:
        """Every event that is not plain information."""
        return [event for event in self.events if event.level != "info"]

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._started

    def summary(self) -> str:
        counts: Dict[str, int] = {}
        for event in self.events:
            counts[event.kind] = counts.get(event.kind, 0) + 1
        parts = [f"{count} {kind}" for kind, count in counts.items()]
        problems = self.problems()
        head = f"{len(self.events)} event(s): " + ", ".join(parts)
        if problems:
            head += f"; {len(problems)} warning(s)/error(s)"
        return head

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [event.to_dict() for event in self.events]

    def render(self) -> str:
        """The whole log for a person, header and footer included."""
        lines = [f"run log{' ' + self.label if self.label else ''}: "
                 f"{len(self.events)} event(s)", ""]
        lines += [event.render() for event in self.events]
        problems = self.problems()
        if problems:
            lines += ["", f"{len(problems)} warning(s)/error(s):"]
            lines += [f"  {event.render()}" for event in problems]
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # The report: what a person reads
    # ------------------------------------------------------------------

    def verdict(self) -> str:
        """One line: did it work. The first line of the report, and the last."""
        errors = self.of("error")
        finished = self.of("finished")
        warnings = [e for e in self.problems() if e.level == "warning"]
        started = self.of("started")
        dry = bool(started and started[0].payload.get("dryRun"))
        what = "DRY RUN" if dry else "RUN"
        if errors:
            first = errors[0]
            where = f" at stage '{first.payload['stage']}'" if "stage" in first.payload else ""
            return f"{what} FAILED{where}: {first.message}"
        if not finished:
            return f"{what} DID NOT FINISH: the log ends after '{self.events[-1].kind}'" \
                if self.events else f"{what} NEVER STARTED"
        n = finished[0].payload.get("nSamples")
        head = f"{what} OK" + (f": {n} sample(s)" if n is not None else "")
        if warnings:
            head += f", {len(warnings)} warning(s) -- read them below"
        return head

    def report(self, *, sampleLines: int = 50) -> str:
        """The run as a person would want to read it.

        Built from the events alone -- the same ones ``run.log.jsonl`` holds --
        so re-rendering a log read back from disk gives the same text. Sections
        in the order the run happened: the verdict, what was run, what was read,
        what was assembled, what the pre-flight found and did, the draw, the
        samples, the files, and every warning again at the end where a reader
        who skipped to the bottom finds them.

        *sampleLines* caps the per-sample section: past it the samples are
        summarised and only the ones with a warning are listed, because a
        thousand identical lines are not something anyone reads.
        """
        L: List[str] = []
        add = L.append
        started = self.of("started")
        s0 = started[0].payload if started else {}
        finished = self.of("finished")

        title = "kika perturbation run" + (f"  [{self.label}]" if self.label else "")
        add(title)
        add("=" * max(len(title), 60))
        add(self.verdict())
        add("")

        # -- What was run ------------------------------------------------
        reads = self.of("read")
        if reads:
            r = reads[0]
            add(f"Source     {r.subject or s0.get('source', '?')} ({r.payload.get('format', '?')})")
        requests = self.of("request")
        if requests:
            add(f"Request    {requests[0].message}")
        if started:
            add(f"Draw       {s0.get('nSamples', '?')} sample(s), seed {s0.get('seed', '?')}, "
                f"{s0.get('space', '?')} space, {s0.get('samplingMethod', '?')}/"
                f"{s0.get('decompositionMethod', '?')}, psd_method={s0.get('psdMethod', '?')}")
            formats = s0.get("formats") or []
            out = s0.get("outputDir")
            if s0.get("dryRun"):
                add("Output     dry run: no tapes" + (f"; run-level files in {out}" if out else ""))
            elif out:
                add(f"Output     {out}  ({', '.join(formats)})")
            else:
                add("Output     none (no outputDir)")
        if finished and finished[0].seconds is not None:
            add(f"Elapsed    {finished[0].seconds:.1f} s")
        add("")

        # -- Read ----------------------------------------------------------
        step = 0
        if reads:
            r = reads[0]
            step += 1
            add(f"{step}  Read           {_secs(r.seconds)}")
            for key in ("decode", "covarianceDecode", "suiteDecode"):
                if key in r.payload:
                    add(f"     {key}: {r.payload[key]}")
            add("")

        # -- Blocks --------------------------------------------------------
        assembled = self.of("assembled")
        if assembled:
            step += 1
            head = next((e for e in assembled if e.subject is None), None)
            add(f"{step}  Blocks         "
                + (head.payload.get("description", head.message).split("\n")[0]
                   if head else f"{len(assembled)} block(s)"))
            for e in assembled:
                if e.subject is None:
                    continue
                d = e.payload.get("dimension", "?")
                add(f"     {e.subject:<44s} {d}x{d}  union={e.payload.get('union', '?')}")
            add("")

        # -- Pre-flight and conditioning -------------------------------------
        inspected = self.of("inspected")
        conditioned = self.of("conditioned")
        if inspected or conditioned:
            step += 1
            head = next((e for e in inspected if e.subject is None), None)
            mode = next((e.payload.get("mode") for e in conditioned + inspected
                         if e.payload.get("mode")), "?")
            add(f"{step}  Pre-flight     ({mode}) "
                + (head.payload.get("summary", head.message) if head else ""))
            remedies = {e.subject: e for e in conditioned if e.subject}
            findings = [e for e in inspected if e.subject]
            seen = set()
            for e in findings:
                mark = "!" if e.level != "info" else " "
                line = f"   {mark} {e.subject}: {e.message}"
                if e.payload.get("check") == "definiteness" and e.subject in remedies:
                    rem = remedies[e.subject]
                    moved = rem.payload.get("stated_diagonal_max_relative_change")
                    line += f"  -> {rem.payload.get('remedy')}"
                    if moved:
                        line += f" (stated variances moved up to {moved:.1%})"
                    seen.add(e.subject)
                add(line)
            for subject, rem in remedies.items():
                if subject not in seen:
                    add(f"     {subject}: {rem.message}")
            add("")

        # -- Draw ----------------------------------------------------------
        drawn = self.of("drawn")
        if drawn:
            step += 1
            head = next((e for e in drawn if e.subject is None), None)
            add(f"{step}  Draw           {_secs(head.seconds) if head else ''}")
            for e in drawn:
                if e.subject:
                    add(f"     {e.subject:<44s} {e.message}")
            add("")

        # -- Samples ---------------------------------------------------------
        applied = self.of("applied")
        if applied:
            step += 1
            samples = sorted({e.sample for e in applied if e.sample is not None})
            checks = self.of("checked")
            emitted = self.of("emitted")
            flagged = {e.sample for e in self.problems() if e.sample is not None}
            add(f"{step}  Samples        {len(samples)} applied"
                + (f", {len(flagged)} with warnings" if flagged else ""))
            listed = samples if len(samples) <= sampleLines else sorted(flagged)
            if len(samples) > sampleLines:
                add(f"     (listing only the {len(listed)} sample(s) with warnings; "
                    f"see run.log.jsonl for all {len(samples)})")
                # A range over the whole run per component, so the reader sees
                # the ensemble even when the lines are not listed.
                byComponent: Dict[str, List[float]] = {}
                for e in checks:
                    if e.subject and "min" in e.payload:
                        lo, hi = byComponent.setdefault(e.subject, [float("inf"), float("-inf")])
                        byComponent[e.subject] = [min(lo, e.payload["min"]), max(hi, e.payload["max"])]
                for subject, (lo, hi) in byComponent.items():
                    add(f"     {subject:<44s} over all samples: {lo:.3f} .. {hi:.3f}")
            for n in listed:
                parts = []
                for e in checks:
                    if e.sample == n and e.subject and "min" in e.payload:
                        parts.append(f"{_short(e.subject)} {e.payload['min']:.3f}..{e.payload['max']:.3f}")
                files = []
                for e in emitted:
                    if e.sample == n:
                        size = e.payload.get("bytes")
                        files.append(f"{e.subject} {_size(size)}" if size else str(e.subject))
                mark = "!" if n in flagged else " "
                line = f"   {mark} #{n:04d}  " + " | ".join(parts[:8]) + (" | ..." if len(parts) > 8 else "")
                if files:
                    line += "   -> " + ", ".join(files)
                add(line)
                for e in self.problems():
                    if e.sample == n:
                        add(f"           ! {e.message}")
            add("")

        # -- Files -----------------------------------------------------------
        written = self.of("written")
        if written or (s0.get("outputDir") and finished):
            step += 1
            names = [f"{e.subject} {_size(e.payload['bytes'])}" if e.payload.get("bytes")
                     else str(e.subject) for e in written]
            if s0.get("outputDir"):
                # Written after the log itself, so they are not events.
                names += ["run_metadata.json", "run.log", "run.log.jsonl"]
            add(f"{step}  Files          " + ", ".join(names))
            add("")

        # -- Warnings, again ---------------------------------------------------
        problems = self.problems()
        if problems:
            add(f"WARNINGS AND ERRORS ({len(problems)})")
            for e in problems:
                where = f"[sample {e.sample:04d}] " if e.sample is not None else ""
                where += f"{e.subject}: " if e.subject else ""
                add(f"  {'E' if e.level == 'error' else '!'} {where}{e.message}")
        else:
            add("No warnings.")
        add("")
        add(self.verdict().replace(" -- read them below", " (listed above)"))
        return "\n".join(L) + "\n"

    def write(self, directory, *, stem: str = "run") -> Tuple[Path, Path]:
        """``<stem>.log.jsonl`` for a program and ``<stem>.log`` for a person.

        The text file is the report followed by the full timeline, so a reader
        gets the verdict and the story first and the raw events only if they
        keep scrolling.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        jsonl = directory / f"{stem}.log.jsonl"
        text = directory / f"{stem}.log"
        with jsonl.open("w", encoding="utf-8") as handle:
            for record in self.to_dicts():
                handle.write(json.dumps(record) + "\n")
        text.write_text(
            self.report() + "\n" + "-" * 72 + "\ntimeline (every event; the "
            "same data as run.log.jsonl)\n" + "-" * 72 + "\n" + self.render(),
            encoding="utf-8")
        return jsonl, text

    @classmethod
    def read(cls, path) -> "RunLog":
        """A log back from its ``.jsonl``."""
        log = cls()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                log.events.append(Event.from_dict(json.loads(line)))
        return log

    def __repr__(self) -> str:
        return f"RunLog({self.summary()})"
