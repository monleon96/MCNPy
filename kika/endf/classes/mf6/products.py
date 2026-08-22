"""The product subsections of an MF6 section (ENDF-6 §6.1).

A section is a HEAD and ``NK`` products. Each product is one TAB1 — carrying
``ZAP``, ``AWP``, ``LIP`` and ``LAW`` in its header fields and the yield
``y(E)`` in its table — followed by whatever its LAW says (:mod:`.laws`).

**The yield is not the multiplicity of a reaction, it is of a product.** A
single MT can list the neutron, the photons and the residual separately, and
ENDF/B-VIII.1's Pu-239 MT18 lists sixty-one of them. ``products`` is therefore
a list and not a dict keyed by ``ZAP``: the same ZAP recurs, once per discrete
photon line.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from ...utils import PadStyle, format_tab1
from .laws import MF6Law


@dataclass
class MF6Product:
    """One product of an MF6 section: its yield, and its law body."""

    zap: float = 0.0
    awp: float = 0.0
    lip: int = 0
    y_interp: List[Tuple[int, int]] = field(default_factory=list)
    y_energies: List[float] = field(default_factory=list)
    y_values: List[float] = field(default_factory=list)
    law_data: MF6Law = field(default_factory=MF6Law)

    @property
    def law(self) -> int:
        """``LAW``, held by the body so there is one source of truth for it."""
        return self.law_data.law

    @property
    def za(self) -> int:
        """``ZAP`` as an integer. 0 is a photon, 1 a neutron, 1001 a proton.

        Rounded rather than truncated, for the reason ``_za`` is elsewhere:
        ENDF's fixed-format floats do not round-trip exactly, and ``int()`` on
        ``26055.999999`` names the wrong nuclide.
        """
        return int(round(self.zap))

    @property
    def is_photon(self) -> bool:
        return self.za == 0

    def emit_header(self, mat: int, mf: int, mt: int,
                    line_num: int, pad: PadStyle) -> Tuple[List[str], int]:
        """The product TAB1: ``ZAP, AWP, LIP, LAW, NR, NP`` then ``y(E)``."""
        return format_tab1(
            self.zap, self.awp, self.lip, self.law,
            self.y_interp, self.y_energies, self.y_values,
            mat, mf, mt, line_num, pad=pad.pairs, interp_pad=pad.interp,
        )

    def emit(self, mat: int, mf: int, mt: int, line_num: int,
             pad: PadStyle) -> Tuple[List[str], int]:
        lines, line_num = self.emit_header(mat, mf, mt, line_num, pad)
        body, line_num = self.law_data.emit(mat, mf, mt, line_num, pad)
        lines.extend(body)
        return lines, line_num

    def describe(self) -> str:
        return f"ZAP={self.za} AWP={self.awp} LIP={self.lip}, {self.law_data.describe()}"

    def report_gaps(self, mt: int, index: int) -> List[str]:
        return self.law_data.report_gaps(mt, index)
