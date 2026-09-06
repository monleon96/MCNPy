"""ACE ↔ the GNDS-shaped canonical model.

Same rule as ``kika/endf/model_adapter``: **nothing outside this package and its
tests may import it.** It lives under ``kika/ace/`` because it imports both the
ACE classes and the model, and the dependency arrow runs format → calculation,
never the reverse. A module-level import from ``kika/ace/__init__.py`` would put
the model on the ``read_ace`` critical path and change what ``import kika``
costs for the cluster pipeline and the desktop app.
"""
from .decode import (EVALUATED_LABEL, GRIDDED_LABEL, HEATED_LABEL, URR_LABEL,
                     aceStyles, decodeAce, decodeAceReaction, qValuesByMT)

__all__ = [
    "decodeAce", "decodeAceReaction", "aceStyles", "qValuesByMT",
    "EVALUATED_LABEL", "HEATED_LABEL", "GRIDDED_LABEL", "URR_LABEL",
]
