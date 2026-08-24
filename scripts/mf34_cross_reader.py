"""Re-export of the MF33↔MF34 cross-block reader, which now lives in the library.

MOVED 2026-08-24 to ``kika/sampling/mf34_cross.py``. The implementation is
unchanged; what changed is who needs it. The reader used to serve the chi^2
alone, so it lived beside the chi^2 scripts. The deliverable's sampler needs the
same blocks — an ensemble drawn from an ``_a0cross`` tape that ignores the a₀
sections is drawn from a distribution the file does not describe — and this
track's whole failure history (§L, §L3, §L9) is "a joint was certified that was
not the joint being shipped". Two readers of the same bytes is how that happens.

This module stays so that every existing import keeps working:

    from scripts.mf34_cross_reader import read_mf34_split   # still fine

⚠ It now depends on the INSTALLED kika, not on this directory. `/work` is not
mounted from WSL, so the cluster venv's version cannot be inspected from here
([[cluster-venv-is-not-inspectable]]): deploying a chi^2 script that imports
this one also requires the staged wheel to be new enough. The import below
raises rather than falling back to a local copy — a fallback would give the two
readers this move exists to prevent, and it would do it silently.
"""
from __future__ import annotations

from kika.sampling.mf34_cross import MF34WithCross, read_mf34_split

__all__ = ["read_mf34_split", "MF34WithCross"]
