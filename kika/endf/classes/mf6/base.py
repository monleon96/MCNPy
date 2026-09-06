"""MF6: energy-angle distributions of reaction products (ENDF-6 §6).

A section is a HEAD record and ``NK`` product subsections, each an independent
distribution law. The products live in :mod:`~kika.endf.classes.mf6.products`
and the law bodies in :mod:`~kika.endf.classes.mf6.laws`; this module is the
section around them.

**This is where modern evaluations put inelastic scattering.** MF4 and MF5 split
a secondary distribution into an angular part and an energy part that know
nothing about each other; MF6 keeps the correlation, which is the whole reason
it exists and the reason a tape read without it is missing more than a file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..mt import MT
from ...utils import (
    ENDF_FORMAT_FLOAT,
    ENDF_FORMAT_INT,
    PadStyle,
    format_endf_data_line,
    format_endf_send_record,
)
from .products import MF6Product

#: ``LCT`` → what frame the distributions are in. LCT=3 is the mixed case added
#: for evaluations that give light recoils in the centre of mass and everything
#: else in the laboratory; C-12 in ENDF/B-VIII.1 is the readiest witness.
FRAMES = {1: "lab", 2: "centre-of-mass", 3: "centre-of-mass for two-body, lab otherwise"}


@dataclass
class MF6MT(MT):
    """One MT section of MF6."""

    _za: Optional[float] = None
    _awr: Optional[float] = None
    #: ``JP``, the HEAD's third field. Zero on almost every section; U-235's
    #: MT18 writes 11. ENDF-6 gives it to fission, to say whether the products
    #: listed are the prompt ones or the total. kika stores and re-emits it and
    #: does **not** interpret it — but it has to be stored, because writing the
    #: 0 that "the third field of a HEAD is unused" would suggest loses it, and
    #: a lost flag on MT18 changes what the section claims its products are.
    _jp: int = 0
    _lct: int = 2
    _nk: Optional[int] = None
    _mat: Optional[int] = None
    _mf: int = 6
    products: List[MF6Product] = field(default_factory=list)
    #: How this section's writer padded its short lines. Probed at parse time
    #: rather than assumed; see :class:`~kika.endf.utils.PadStyle`.
    pad: PadStyle = field(default_factory=PadStyle)

    # ------------------------------------------------------------------
    @property
    def zaid(self) -> Optional[int]:
        return int(round(self._za)) if self._za is not None else None

    @property
    def atomic_weight_ratio(self) -> Optional[float]:
        return self._awr

    @property
    def jp(self) -> int:
        """``JP``: whether MT18's products are the prompt ones or the total."""
        return self._jp

    @property
    def num_products(self) -> int:
        return self._nk if self._nk is not None else len(self.products)

    @property
    def frame(self) -> str:
        """What ``LCT`` says the secondary distributions are referred to."""
        return FRAMES.get(self._lct, f"LCT={self._lct}")

    def products_for(self, za: int) -> List[MF6Product]:
        """Every product with this ``ZAP`` — there is often more than one."""
        return [p for p in self.products if p.za == za]

    # ------------------------------------------------------------------
    def __str__(self) -> str:
        """Serialise back to ENDF: HEAD, the NK products, SEND.

        The sequence number runs across the whole section, so a product that
        grew by thousands of lines carries the ones after it along with it.
        """
        mat = self._mat if self._mat is not None else 0
        mf = self._mf
        mt = self.number

        line_num = 1
        lines = [format_endf_data_line(
            [self._za, self._awr, self._jp, self._lct, self.num_products, 0],
            mat, mf, mt, line_num,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT,
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT],
        )]
        line_num += 1

        for product in self.products:
            product_lines, line_num = product.emit(mat, mf, mt, line_num, self.pad)
            lines.extend(product_lines)

        lines.append(format_endf_send_record(mat, mf))
        return "\n".join(lines)

    def report_gaps(self) -> List[str]:
        """One line per product this class reads less of than the file carries.

        **Not optional**, for the reason
        :meth:`kika.endf.classes.mf5.base.MF5MT.report_gaps` gives. MF6's list
        is normally not empty even though every law is decoded, because a
        product may defer its distribution to MF14 or MF15 — files kika does not
        read — and a section that says nothing about that reads as complete.
        """
        gaps: List[str] = []
        for index, product in enumerate(self.products):
            gaps.extend(product.report_gaps(self.number, index))
        return gaps

    def __repr__(self) -> str:
        laws = ",".join(str(p.law) for p in self.products)
        return (f"MF6MT({self.number}, LCT={self._lct}, "
                f"NK={self.num_products}, LAW=[{laws}])")
