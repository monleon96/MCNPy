"""
Resonance cross-section formulas: SLBW, MLBW, Reich-Moore.

Re-exports from ``kika.endf.processing.resonance_formulas`` — the
implementation already accepts any object with ``.energy``, ``.spin``,
``.c3``–``.c6`` attributes (duck typing), so it works with both
``kika.endf.classes.mf2.mf2mt151.Resonance`` and
``kika.nuclear_data.ResonanceRecord``.
"""

from kika.endf.processing.resonance_formulas import (
    statistical_spin_factor,
    slbw_cross_sections,
    mlbw_cross_sections,
    reich_moore_cross_sections,
)

__all__ = [
    "statistical_spin_factor",
    "slbw_cross_sections",
    "mlbw_cross_sections",
    "reich_moore_cross_sections",
]
