"""What a drawn perturbation *is*, written down once.

A draw produces a flat vector of factors. On its own that vector says nothing:
which reaction each stretch of it belongs to, what energies it is stated on,
whether it multiplies or adds, and what happens at a bin edge are all carried
somewhere else -- in the shape of the code that made it and the shape of the
code that consumes it, which is how a perturbation comes to mean two different
things at its two ends.

:class:`PerturbationSet` is that meaning as data. It sits in a run directory
beside the :class:`~kika.cov.conditioning.ConditioningPlan`, and between them
they say everything that was done to a covariance and everything that was drawn
from it.

**Its index already exists.** :func:`kika.sampling.mf33_sampling.loadCrossSectionBlocks`
returns ``{key: {pairs, stride, grids, widths, dimension}}`` -- the ``*_index``
this repository has always passed around, only as data rather than as a
convention. What this class adds on top is the part that was never written
down anywhere: the semantics.

**Why the semantics is not a free-form string.** ``"relative"`` versus
``"absolute"`` is the difference between multiplying a cross section and adding
to it, and a file that does not say which cannot be read back safely a year
later. The values here are a closed set, checked on construction, and the edge
convention travels with them because a factor block is piecewise-constant and
therefore says nothing at all about its own discontinuities -- that rule lives
in :func:`kika.nuclear_data.model.perturbation.applyFactors` and is named here
so a reader of the file knows which applier produced it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = ["PerturbationSet", "SEMANTICS"]

#: The ways a factor block can act on the quantity it perturbs. A closed set,
#: because "the file says ``relative`` and the code assumed ``absolute``" is a
#: failure that produces plausible numbers.
SEMANTICS = ("multiplicative-relative",)

#: Which discontinuity convention the factors were drawn to be applied under.
#: Named rather than implied: a piecewise-constant block is silent about its
#: own steps, so the rule lives in the applier and the set records which one.
EDGE_RULE = "endf-step-duplicate"

_FORMAT_VERSION = 1


@dataclass(frozen=True)
class PerturbationSet:
    """One drawn realisation, for one covariance, over one or more reactions.

    Parameters
    ----------
    label
        The §9.3 style label this realisation will be written under --
        ``'realization-0007'``. It is the name the perturbed form takes inside
        a :class:`~kika.nuclear_data.model.suite.ReactionSuite`, so it belongs
        to the perturbation and not to whoever applies it.
    key
        Which covariance was drawn, ``'MF33'`` or ``'MF31'``.
    factors
        ``MT -> one factor per bin``.
    binEdges
        ``MT -> the bin boundaries those factors are stated on``, one longer
        than the factors.
    """

    label: str
    key: str
    factors: Dict[int, np.ndarray]
    binEdges: Dict[int, np.ndarray]
    semantics: str = SEMANTICS[0]
    edgeRule: str = EDGE_RULE
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.semantics not in SEMANTICS:
            raise ValueError(
                f"semantics {self.semantics!r} is not one of {list(SEMANTICS)}. "
                f"A perturbation that does not say how it acts cannot be applied"
            )
        if set(self.factors) != set(self.binEdges):
            missing = set(self.factors) ^ set(self.binEdges)
            raise ValueError(
                f"MT {sorted(missing)} has factors without a grid or a grid "
                f"without factors; a block and its bins are one object"
            )
        for mt, values in self.factors.items():
            edges = self.binEdges[mt]
            if len(values) != len(edges) - 1:
                raise ValueError(
                    f"MT{mt}: {len(values)} factor(s) on {len(edges) - 1} bin(s)"
                )

    # ------------------------------------------------------------------
    # From a draw
    # ------------------------------------------------------------------

    @classmethod
    def fromDraw(cls, factors: Sequence[float],
                 index: Mapping[Hashable, Mapping[str, Any]], *,
                 label: str, provenance: Optional[Dict[str, Any]] = None
                 ) -> "PerturbationSet":
        """Cut one realisation out of a flat factor vector.

        *index* is what :func:`~kika.sampling.mf33_sampling.loadCrossSectionBlocks`
        returned, and *factors* one row of what
        :func:`~kika.sampling.multigroup_draw.draw_relative_factors` returned
        against it.

        ``widths`` is read rather than assumed. Under the shipped ``global``
        union every component is ``stride`` wide and the two are the same
        number; under ``per-component`` they are not, and the tail of each
        component's stride is the zero padding a uniform stride implies. Slicing
        by ``stride`` there would hand a reaction its neighbour's padding as if
        it were its own factors.
        """
        if len(index) != 1:
            raise ValueError(
                f"expected one covariance key, got {sorted(index)}. A "
                f"PerturbationSet is one draw of one covariance"
            )
        (indexKey, meta), = index.items()
        pairs = list(meta["pairs"])
        stride = int(meta["stride"])
        widths = meta["widths"]
        grids = meta["grids"]

        values = np.asarray(factors, dtype=float).ravel()
        expected = len(pairs) * stride
        if values.size != expected:
            raise ValueError(
                f"{values.size} factor(s) for an index of {len(pairs)} "
                f"component(s) x stride {stride} = {expected}"
            )

        blockFactors: Dict[int, np.ndarray] = {}
        blockEdges: Dict[int, np.ndarray] = {}
        for position, pair in enumerate(pairs):
            mt = int(pair[-1])
            width = int(widths[pair] if isinstance(widths, Mapping) else widths[position])
            start = position * stride
            blockFactors[mt] = values[start:start + width].copy()
            grid = grids[pair] if isinstance(grids, Mapping) else grids[position]
            blockEdges[mt] = np.asarray(grid, dtype=float)

        return cls(
            label=label,
            key=str(indexKey[1]) if isinstance(indexKey, tuple) else str(indexKey),
            factors=blockFactors,
            binEdges=blockEdges,
            provenance=dict(provenance or {}),
        )

    # ------------------------------------------------------------------
    # Applying
    # ------------------------------------------------------------------

    def reactions(self) -> Tuple[int, ...]:
        """The MTs this set perturbs, ascending."""
        return tuple(sorted(self.factors))

    def apply(self, function1d, mt: int):
        """Perturb *function1d* with this set's block for *mt*.

        Returns ``(perturbed, diagnostics)``, the pair
        :func:`~kika.nuclear_data.model.perturbation.applyFactors` returns.
        Raises :class:`KeyError` for an MT this set does not carry, rather than
        returning the function unchanged: "no perturbation for this reaction"
        and "a perturbation of one" are different answers, and silently
        conflating them is how an ensemble comes to be narrower than it claims.
        """
        from kika.nuclear_data.model.perturbation import applyFactors

        if mt not in self.factors:
            raise KeyError(
                f"this PerturbationSet perturbs MT {list(self.reactions())}, "
                f"not MT{mt}"
            )
        return applyFactors(function1d, self.factors[mt], self.binEdges[mt])

    def applyToSuite(self, suite) -> Dict[int, Dict[str, float]]:
        """Put a perturbed form under :attr:`label` on every reaction it covers.

        The evaluated form is left where it is and the realisation goes beside
        it, which is what §9.1's multi-form container and §9.3's ``realization``
        style are for -- and what ``encodeMF3MT(..., label=)`` then writes out.
        Reactions this set does not cover are not touched and not reported as
        perturbed.
        """
        from kika.nuclear_data.model import EVAL_LABEL

        diagnostics: Dict[int, Dict[str, float]] = {}
        for mt in self.reactions():
            reaction = suite.reactionByENDF_MT(mt)
            if reaction is None:
                continue
            perturbed, info = self.apply(reaction.crossSection[EVAL_LABEL], mt)
            reaction.crossSection[self.label] = perturbed
            diagnostics[mt] = info
        return diagnostics

    # ------------------------------------------------------------------
    # On disk
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-ready mapping. Grids and factors become lists of floats."""
        return {
            "format": _FORMAT_VERSION,
            "label": self.label,
            "key": self.key,
            "semantics": self.semantics,
            "edgeRule": self.edgeRule,
            "provenance": dict(self.provenance),
            "blocks": {
                str(mt): {
                    "factors": [float(v) for v in self.factors[mt]],
                    "binEdges": [float(e) for e in self.binEdges[mt]],
                }
                for mt in self.reactions()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PerturbationSet":
        version = int(data.get("format", 0))
        if version != _FORMAT_VERSION:
            raise ValueError(
                f"perturbation set format {version}, this kika writes and reads "
                f"{_FORMAT_VERSION}"
            )
        blocks = data["blocks"]
        return cls(
            label=data["label"],
            key=data["key"],
            factors={int(mt): np.asarray(b["factors"], dtype=float)
                     for mt, b in blocks.items()},
            binEdges={int(mt): np.asarray(b["binEdges"], dtype=float)
                      for mt, b in blocks.items()},
            semantics=data.get("semantics", SEMANTICS[0]),
            edgeRule=data.get("edgeRule", EDGE_RULE),
            provenance=dict(data.get("provenance", {})),
        )

    def write(self, path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path) -> "PerturbationSet":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def __repr__(self) -> str:
        return (f"PerturbationSet({self.label!r}, {self.key}, "
                f"MT={list(self.reactions())})")
