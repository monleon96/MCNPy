"""
The record substrate MF32 is built from.

MF32 is unusual among the covariance files kika reads: it re-declares File 2's
resonance parameters, interleaves them with their uncertainties, and then packs
a correlation matrix into integer INTG records. The result is a section whose
*bulk* is numbers nobody navigates by — 240 131 lines of them on Ta-181 — sitting
behind a handful of small control records that everything does navigate by.

So the substrate splits the two. :class:`Record` decodes its six fields eagerly,
because there are few of them and the parser branches on them. :class:`PackedList`
keeps its body as the text it was read from and decodes only when asked, because
``parse_line`` costs 1.56 s per 20 000 lines and Ta-181 would otherwise add ~19 s
to every ``read_endf`` that touches it.

Keeping the text also buys byte-identity outright, which matters more here than
elsewhere: MF32 in the wild does not always match ENDF-102 §32. Th-232 in
ENDF/B-VIII.1 writes NM into *both* trailing fields of its INTG control record
where the manual says the second is zero, and puts a 3 in a field the manual
draws as 0. Re-deriving those fields from a decoded model would silently correct
the file — which is not kika's job, and would break ``str(parse(s)) == s``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from ...utils import format_endf_id_columns, parse_line

#: Width of the data part of an ENDF record. Columns 67-80 carry MAT/MF/MT and
#: the sequence number, and are re-stamped on every write rather than stored.
DATA_WIDTH = 66


def _stamp(body: str, mat: int, mf: int, mt: int, line_num: int) -> str:
    """Re-attach the identification columns to a stored 66-column body."""
    return body.ljust(DATA_WIDTH) + format_endf_id_columns(mat, mf, mt, line_num)


@dataclass
class PackedList:
    """The body of a LIST record, kept as the lines it was read from.

    ``values`` decodes on first use and caches. Assigning through
    :meth:`set_values` replaces the body, so a caller that edits a covariance
    matrix gets its edit written out; a caller that only reads gets the original
    bytes back.
    """

    lines: List[str] = field(default_factory=list)
    count: int = 0
    _values: Optional[List[float]] = field(default=None, repr=False)
    _edited: bool = field(default=False, repr=False)

    @property
    def values(self) -> List[float]:
        """The ``count`` numbers of the body, decoded on demand."""
        if self._values is None:
            decoded: List[float] = []
            for line in self.lines:
                parsed = parse_line(line.ljust(DATA_WIDTH))
                for i in range(1, 7):
                    if len(decoded) >= self.count:
                        break
                    value = parsed.get(f"C{i}")
                    decoded.append(0.0 if value is None else value)
            self._values = decoded
        return self._values

    def set_values(self, values: Sequence[float]) -> None:
        """Replace the body. The stored text stops being authoritative."""
        self._values = list(values)
        self.count = len(self._values)
        self._edited = True

    def emit(self, mat: int, mf: int, mt: int,
             line_num: int) -> Tuple[List[str], int]:
        if self._edited:
            from ...utils import format_data_values
            return format_data_values(list(self.values), mat, mf, mt, line_num)
        out = []
        for line in self.lines:
            out.append(_stamp(line, mat, mf, mt, line_num))
            line_num += 1
        return out, line_num

    def __len__(self) -> int:
        return self.count


@dataclass
class Record:
    """One CONT or LIST record: its six decoded fields, its text, its body.

    ``body`` is ``None`` for a CONT record and a :class:`PackedList` for a LIST
    record. The six fields are named after the ENDF-6 record template positions
    rather than after any particular meaning, because MF32 reuses the same
    positions for LCOMP, NSRS, NDIGIT, NRSA and half a dozen other quantities
    depending on where in the tree the record sits.
    """

    raw: str = ""
    c1: object = None
    c2: object = None
    l1: int = 0
    l2: int = 0
    n1: int = 0
    n2: int = 0
    body: Optional[PackedList] = None

    @classmethod
    def read(cls, lines: List[str], index: int,
             with_body: bool = False) -> Tuple["Record", int]:
        """Read one record starting at *index*; return it and the next index."""
        head = lines[index]
        parsed = parse_line(head.ljust(80))
        record = cls(
            raw=head[:DATA_WIDTH],
            c1=parsed.get("C1"), c2=parsed.get("C2"),
            l1=int(parsed.get("C3") or 0), l2=int(parsed.get("C4") or 0),
            n1=int(parsed.get("C5") or 0), n2=int(parsed.get("C6") or 0),
        )
        index += 1
        if with_body:
            span = (record.n1 + 5) // 6
            record.body = PackedList(
                lines=[line[:DATA_WIDTH] for line in lines[index:index + span]],
                count=record.n1,
            )
            index += span
        return record, index

    def emit(self, mat: int, mf: int, mt: int,
             line_num: int) -> Tuple[List[str], int]:
        out = [_stamp(self.raw, mat, mf, mt, line_num)]
        line_num += 1
        if self.body is not None:
            body_lines, line_num = self.body.emit(mat, mf, mt, line_num)
            out.extend(body_lines)
        return out, line_num

    @property
    def values(self) -> List[float]:
        """The LIST body's numbers, or an empty list for a CONT record."""
        return self.body.values if self.body is not None else []


def emit_all(records: Sequence[Optional[Record]], mat: int, mf: int, mt: int,
             line_num: int) -> Tuple[List[str], int]:
    """Emit a sequence of records in order, skipping the absent ones."""
    out: List[str] = []
    for record in records:
        if record is None:
            continue
        lines, line_num = record.emit(mat, mf, mt, line_num)
        out.extend(lines)
    return out, line_num
