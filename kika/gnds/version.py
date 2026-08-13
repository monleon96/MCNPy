"""Which GNDS format versions this reader accepts, and why the list is short.

**One code path serves 2.0 and 2.1.** That is a finding, not a shortcut, and it
rests on two independent sources:

* The 2.1 specification's own foreword (NEA/WKP(2025)6, p. 3) describes 2.1 as
  *"a modest update with several formally approved change requests compared to
  version 2.0"* whose changes *"focus on improving handling for thermal neutron
  scattering law data, along with many fixes for inconsistencies and errors in
  the previous specification document"* — against **149** approved change
  requests for 1.9 → 2.0, which it calls *"a major update"*. kika reads no
  thermal scattering law data at all: its ENDF parser registry has no MF7, and
  ``ThermalNeutronScatteringLaw1d`` in the model is a pointer with no payload.
* FUDGE 6.10.0 — the reference implementation, and the translator that produced
  every GNDS file in circulation — lists ``2.1`` among its allowed format
  versions and contains **no branch on it anywhere**. Its only version-dependent
  node renames (``LUPY/ancestry.py``'s ``monikerByFormat``) are for 1.10:
  ``section`` → ``covarianceSection``, ``crossSections`` → ``crossSectionSums``.

So a version-dispatch layer between 2.0 and 2.1 would have zero branches in it,
and building one would be paying for a difference that does not exist. If a
future version does change a node kika reads, the branch belongs **at that
node**, with a fixture of each version beside it — not in an abstraction laid
down in advance.

**1.9 and 1.10 are refused rather than attempted.** That is where the real break
is, and a reader that half-works on them would produce a ``reactionSuite`` with
silently missing content, which is worse than a refusal.

kika's model is built to the 2.1 specification; every published library is
written in 2.0. Both are read here. The *writer* preserves whatever version it
was handed — see :mod:`kika.gnds.encode`.
"""
from __future__ import annotations

from typing import Optional, Tuple

__all__ = [
    "ACCEPTED", "REFUSED", "MODEL_FORMAT", "DEFAULT_WRITE_FORMAT",
    "UnsupportedGndsVersion", "checkFormat",
]

#: Read through the same code path. See the module docstring.
ACCEPTED: Tuple[str, ...] = ("2.0", "2.1")

#: Refused, with the reason the message carries. The keys are the ``format``
#: strings those versions actually write.
REFUSED = {
    "1.9": (
        "GNDS 1.9 predates the 149 approved change requests that produced 2.0, "
        "so its node names and structure differ from what this reader knows"
    ),
    "1.10": (
        "GNDS 1.10 predates the 149 approved change requests that produced 2.0. "
        "Among other differences it calls a covarianceSection a 'section' and "
        "crossSectionSums 'crossSections'"
    ),
}

#: The specification kika's model is shaped to. Not the same as what it reads,
#: and not the same as what it writes by default -- the three are deliberately
#: distinct and each has its own reason.
MODEL_FORMAT = "2.1"

#: What the writer emits when the suite has no GNDS origin to preserve -- an
#: ENDF-sourced ``reactionSuite``, say. 2.0 rather than 2.1 because 2.0 is what
#: every published library and every tool in circulation is written in, and a
#: file nothing else reads is not interoperability. Flip this when that changes.
DEFAULT_WRITE_FORMAT = "2.0"


class UnsupportedGndsVersion(ValueError):
    """The file declares a GNDS version this reader will not attempt."""


def checkFormat(declared: Optional[str], source: str = "this file") -> str:
    """Validate a root node's ``format`` attribute, returning it unchanged.

    Parameters
    ----------
    declared
        The ``format`` attribute of a ``reactionSuite`` or ``covarianceSuite``.
        ``None`` means the attribute was absent, which is itself a refusal:
        §14.1.1 makes it required, and a file that omits it is not a file whose
        version can be assumed.
    source
        Named in the message, so a failure says *which* file was refused.

    Raises
    ------
    UnsupportedGndsVersion
        For a version in :data:`REFUSED`, for an absent attribute, and for
        anything unrecognised. An unrecognised version is refused rather than
        attempted **because it is probably newer**: reading a 2.2 file with 2.1
        rules would not fail, it would quietly return whatever the unchanged
        nodes happened to yield, and the caller would have no way to tell.
    """
    if declared is None:
        raise UnsupportedGndsVersion(
            f"{source} has no 'format' attribute on its root node. §14.1.1 "
            f"requires one, and without it there is no version to read this "
            f"against."
        )
    if declared in ACCEPTED:
        return declared
    if declared in REFUSED:
        raise UnsupportedGndsVersion(
            f"{source} declares GNDS {declared}, which kika does not read. "
            f"{REFUSED[declared]}. Accepted: {', '.join(ACCEPTED)}."
        )
    raise UnsupportedGndsVersion(
        f"{source} declares GNDS {declared!r}, which kika does not recognise. "
        f"Accepted: {', '.join(ACCEPTED)}. If {declared!r} is newer than 2.1, "
        f"it is refused rather than attempted on purpose — reading it with 2.1 "
        f"rules would not raise, it would return whatever the unchanged nodes "
        f"happen to yield, and nothing would say so."
    )
