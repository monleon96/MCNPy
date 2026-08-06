"""Declared-but-unimplemented higher-dimensional §6 forms.

The roadmap's rule for the whole model: **empty slots exist, they are not
absent**. A node kika cannot yet represent must raise ``NotImplementedError``
naming the GNDS node, so a reader meeting one gets told what is missing rather
than a silent wrong answer or an ``AttributeError`` three frames away.

These are the 2-d and 3-d forms. Phase 7b fills them, together with the §18
distribution laws. Until then, constructing one is the error.
"""
from __future__ import annotations

from typing import Any

__all__ = ["XYs2d", "XYs3d", "Regions2d", "Regions3d", "NOT_IMPLEMENTED_NODES"]


class _UnimplementedNode:
    """Base for a declared GNDS node with no implementation behind it yet."""

    gndsNodeName: str = "?"
    plannedFor: str = "phase 7b"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            f"GNDS node {self.gndsNodeName!r} is declared in kika's model but not "
            f"implemented ({self.plannedFor}). It is present rather than absent so "
            f"that a reader meeting one is told what is missing instead of failing "
            f"somewhere else."
        )


class XYs2d(_UnimplementedNode):
    gndsNodeName = "XYs2d"


class XYs3d(_UnimplementedNode):
    gndsNodeName = "XYs3d"


class Regions2d(_UnimplementedNode):
    gndsNodeName = "regions2d"


class Regions3d(_UnimplementedNode):
    gndsNodeName = "regions3d"


#: Every node declared here, for the reader in phase 5 to consult when it needs
#: to say "I know this node exists and I cannot read it yet".
NOT_IMPLEMENTED_NODES = {
    cls.gndsNodeName: cls for cls in (XYs2d, XYs3d, Regions2d, Regions3d)
}
