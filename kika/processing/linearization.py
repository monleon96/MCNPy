"""
Adaptive energy grid generation (broken-stick linearization).

Re-exports ``kika.endf.processing.linearization.linearize`` — the
implementation is already format-agnostic (pure numpy).
"""

from kika.endf.processing.linearization import linearize

__all__ = ["linearize"]
