"""MF5: energy distributions of secondary neutrons (ENDF-6 §5).

A section is a HEAD record and NK subsections, each an independent
distribution law weighted by ``p_k(E)``. The laws live in
:mod:`~kika.endf.classes.mf5.partials`; this module is the section around them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..mt import MT
from ...utils import (
    ENDF_FORMAT_FLOAT,
    ENDF_FORMAT_INT,
    format_endf_data_line,
    format_endf_send_record,
)
from .partials import MF5Partial, MF5PartialTabulated


@dataclass
class MF5MT(MT):
    """One MT section of MF5."""

    _za: Optional[float] = None
    _awr: Optional[float] = None
    _nk: Optional[int] = None
    _mat: Optional[int] = None
    _mf: int = 5
    partials: List[MF5Partial] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def num_partials(self) -> int:
        return self._nk if self._nk is not None else len(self.partials)

    def tabulated_partials(self) -> List[Tuple[int, MF5PartialTabulated]]:
        """``(index, partial)`` for every LF=1 subsection — the perturbable ones."""
        return [
            (index, partial)
            for index, partial in enumerate(self.partials)
            if isinstance(partial, MF5PartialTabulated)
        ]

    def report_gaps(self) -> List[str]:
        """One line per subsection this class kept verbatim rather than decoded.

        **Not optional.** Declaring MF5 supported without saying which laws are
        only passed through turns a silent skip into a silent false claim of
        coverage, which is the worse of the two. Everything that reports on
        kika's ENDF coverage calls this.

        The test is ``is_decoded`` and not ``isinstance(.., MF5PartialRaw)``:
        the analytic laws in :mod:`~kika.endf.classes.mf5.analytic` *are* raw
        partials -- they keep their bytes so they can emit them -- and are read
        all the same, so an isinstance check here would report every LF=5
        subsection on the reference tape as a gap it is not.
        """
        return [
            f"MF5/MT{self.number} partial {index} is stored verbatim: "
            f"{partial.describe()}"
            for index, partial in enumerate(self.partials)
            if not partial.is_decoded
        ]

    # ------------------------------------------------------------------
    def __str__(self) -> str:
        """Serialise back to ENDF: HEAD, the NK subsections, SEND.

        The sequence number runs across the whole section, so a subsection that
        grew by thousands of lines carries the ones after it along with it.
        """
        mat = self._mat if self._mat is not None else 0
        mf = self._mf
        mt = self.number
        nk = self.num_partials

        line_num = 1
        lines = [format_endf_data_line(
            [self._za, self._awr, 0, 0, nk, 0],
            mat, mf, mt, line_num,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT],
        )]
        line_num += 1

        for partial in self.partials:
            partial_lines, line_num = partial.emit(mat, mf, mt, line_num)
            lines.extend(partial_lines)

        lines.append(format_endf_send_record(mat, mf))
        return "\n".join(lines)

    def __repr__(self) -> str:
        laws = ",".join(str(p.lf) for p in self.partials)
        return f"MF5MT({self.number}, NK={self.num_partials}, LF=[{laws}])"
