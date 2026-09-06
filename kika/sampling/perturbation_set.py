"""What a drawn perturbation *is*, written down once -- for every quantity.

A draw produces a flat vector of factors. On its own that vector says nothing:
which reaction each stretch of it belongs to, which *quantity* of that reaction,
what energies it is stated on, whether it multiplies or adds, and what happens
at a bin edge are all carried somewhere else -- in the shape of the code that
made it and the shape of the code that consumes it, which is how a perturbation
comes to mean two different things at its two ends.

:class:`PerturbationSet` is that meaning as data. It sits in a run directory
beside the :class:`~kika.cov.conditioning.ConditioningPlan`, and between them
they say everything that was done to a covariance and everything that was drawn
from it.

**One realisation, not one covariance.** A request may cover cross sections,
angular distributions and multiplicities at once, and
:mod:`kika.sampling.joint_blocks` decides which of them are drawn together and
which apart. What comes out of that is several blocks and therefore several
draws -- but *one* realisation of the evaluation, and it has to be applied and
written as one. So this object holds every block of one realisation, keyed by
:class:`~kika.sampling.joint_blocks.ComponentKey`, and records which components
were drawn together in :attr:`groups`, because "these two were independent" is a
claim about the evaluation that a file has to carry if anyone is to check it
later.

**Its index already exists.** :func:`kika.sampling.joint_blocks.requestIndex`
returns ``{blockKey: {components, stride, grids, widths, dimension, ...}}`` --
the ``*_index`` this repository has always passed around, only as data rather
than as a convention. What this class adds on top is the part that was never
written down anywhere: the semantics.

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
from typing import (Any, Dict, Hashable, List, Mapping, Optional, Sequence,
                    Tuple)

import numpy as np

from kika.sampling.joint_blocks import ComponentKey

__all__ = ["PerturbationSet", "SEMANTICS", "EDGE_RULE"]

#: The ways a factor block can act on the quantity it perturbs. A closed set,
#: because "the file says ``relative`` and the code assumed ``absolute``" is a
#: failure that produces plausible numbers.
SEMANTICS = ("multiplicative-relative",)

#: Which discontinuity convention the factors were drawn to be applied under.
#: Named rather than implied: a piecewise-constant block is silent about its own
#: steps, so the rule lives in the applier and the set records which one.
EDGE_RULE = "endf-step-duplicate"

_FORMAT_VERSION = 2


@dataclass(frozen=True)
class PerturbationSet:
    """One drawn realisation of an evaluation, over one or more quantities.

    Parameters
    ----------
    label
        The §9.3 style label this realisation will be written under --
        ``'realization-0007'``. It is the name the perturbed forms take inside a
        :class:`~kika.nuclear_data.model.suite.ReactionSuite`, so it belongs to
        the perturbation and not to whoever applies it.
    factors
        ``ComponentKey -> one factor per bin``.
    binEdges
        ``ComponentKey -> the bin boundaries those factors are stated on``, one
        longer than the factors.
    groups
        The components that were drawn together, one tuple per block. Two
        components in different tuples were drawn independently, which is a
        statement about what the evaluation correlates and is worth keeping.
    """

    label: str
    factors: Dict[ComponentKey, np.ndarray]
    binEdges: Dict[ComponentKey, np.ndarray]
    groups: Tuple[Tuple[ComponentKey, ...], ...] = ()
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
                f"{sorted(missing)} has factors without a grid or a grid "
                f"without factors; a block and its bins are one object"
            )
        for component, values in self.factors.items():
            edges = self.binEdges[component]
            if len(values) != len(edges) - 1:
                raise ValueError(
                    f"{component.describe()}: {len(values)} factor(s) on "
                    f"{len(edges) - 1} bin(s)"
                )
        grouped = {component for group in self.groups for component in group}
        if self.groups and grouped != set(self.factors):
            raise ValueError(
                f"the groups cover {len(grouped)} component(s) and the factors "
                f"{len(self.factors)}; a realisation cannot say two different "
                f"things about what it perturbs"
            )

    # ------------------------------------------------------------------
    # From a draw
    # ------------------------------------------------------------------

    @classmethod
    def fromDraw(cls, drawn, index: Mapping[Hashable, Mapping[str, Any]], *,
                 label: str, provenance: Optional[Dict[str, Any]] = None
                 ) -> "PerturbationSet":
        """Cut one realisation out of what a draw returned.

        *index* is what :func:`~kika.sampling.joint_blocks.requestIndex`
        returned. *drawn* is either ``{blockKey: one row of factors}`` -- sample
        *i* of what :func:`~kika.sampling.core.draw_samples` produced, which is
        the normal case -- or, when the index has exactly one block, that row on
        its own.

        ``widths`` is read rather than assumed. Under the ``global`` union every
        component is ``stride`` wide and the two are the same number; under
        ``per-component`` they are not, and the tail of each component's stride
        is the zero padding a uniform stride implies. Slicing by ``stride``
        there would hand a component its neighbour's padding as if it were its
        own factors -- and a factor of zero is not a small perturbation, it is a
        deleted cross section.
        """
        if not isinstance(drawn, Mapping):
            if len(index) != 1:
                raise ValueError(
                    f"a bare factor vector needs an index of one block, got "
                    f"{len(index)}. With several blocks the rows have to say "
                    f"which block they came from"
                )
            (onlyKey,) = index
            drawn = {onlyKey: drawn}

        missing = set(index) - set(drawn)
        if missing:
            raise ValueError(
                f"{len(missing)} block(s) of the index were not drawn: a "
                f"realisation that silently omits a block is perturbed in fewer "
                f"places than its metadata claims"
            )

        factors: Dict[ComponentKey, np.ndarray] = {}
        binEdges: Dict[ComponentKey, np.ndarray] = {}
        groups: List[Tuple[ComponentKey, ...]] = []
        for blockKey, meta in index.items():
            components = list(meta.get("components") or meta.get("pairs")
                              or meta.get("triplets"))
            stride = int(meta["stride"])
            widths, grids = meta["widths"], meta["grids"]
            values = np.asarray(drawn[blockKey], dtype=float).ravel()
            expected = len(components) * stride
            if values.size != expected:
                raise ValueError(
                    f"block {blockKey}: {values.size} factor(s) for "
                    f"{len(components)} component(s) x stride {stride} = "
                    f"{expected}"
                )
            group: List[ComponentKey] = []
            for position, component in enumerate(components):
                component = cls._asComponentKey(component, blockKey)
                lookup = components[position]
                width = int(widths[lookup] if isinstance(widths, Mapping)
                            else widths[position])
                start = position * stride
                factors[component] = values[start:start + width].copy()
                grid = grids[lookup] if isinstance(grids, Mapping) else grids[position]
                binEdges[component] = np.asarray(grid, dtype=float)
                group.append(component)
            groups.append(tuple(group))

        return cls(label=label, factors=factors, binEdges=binEdges,
                   groups=tuple(groups), provenance=dict(provenance or {}))

    @staticmethod
    def _asComponentKey(component, blockKey) -> ComponentKey:
        """A component of any of the three index shapes, as one key.

        ``requestIndex`` already gives :class:`ComponentKey`. The two
        single-quantity indices predate it and give ``(ZA, MT)`` for MF33/MF31
        and ``(ZA, MT, L)`` for MF34, so the MF has to come from the block key,
        which is where those two put it.
        """
        if isinstance(component, ComponentKey):
            return component
        label = blockKey[1] if isinstance(blockKey, tuple) and len(blockKey) > 1 else ""
        mf = int(str(label).replace("MF", "")) if str(label).startswith("MF") else 0
        if len(component) == 3:
            return ComponentKey(int(component[0]), 34, int(component[1]),
                                int(component[2]))
        if not mf:
            raise ValueError(
                f"block key {blockKey!r} does not name an MF, so the component "
                f"{component!r} cannot be placed. Use requestIndex, whose "
                f"components carry it"
            )
        return ComponentKey(int(component[0]), mf, int(component[1]), 0)

    # ------------------------------------------------------------------
    # What is in it
    # ------------------------------------------------------------------

    def components(self) -> Tuple[ComponentKey, ...]:
        """Every component this set perturbs, sorted."""
        return tuple(sorted(self.factors))

    def reactions(self) -> Tuple[int, ...]:
        """The MTs this set perturbs, ascending, whatever the quantity."""
        return tuple(sorted({component.mt for component in self.factors}))

    def quantities(self) -> Tuple[str, ...]:
        """The model nodes this set touches, e.g. ``('crossSection',)``."""
        return tuple(sorted({component.quantity for component in self.factors}))

    def block(self, mt: int, *, mf: int = 33, order: int = 0, za: Optional[int] = None):
        """``(factors, binEdges)`` for one component, addressed the readable way.

        A convenience for callers and tests that know which reaction they mean
        but not which ZA the tape stated it under.
        """
        matches = [component for component in self.factors
                   if component.mt == mt and component.mf == mf
                   and component.index == order
                   and (za is None or component.za == za)]
        if not matches:
            raise KeyError(
                f"this PerturbationSet perturbs "
                f"{[c.describe() for c in self.components()]}, not "
                f"MF{mf}/MT{mt} index {order}"
            )
        if len(matches) > 1:
            raise KeyError(
                f"{len(matches)} components match MF{mf}/MT{mt} index {order} "
                f"(different ZA); name the ZA"
            )
        return self.factors[matches[0]], self.binEdges[matches[0]]

    def describe(self) -> str:
        """One line per group, saying what was drawn with what."""
        lines = [f"{self.label}: {len(self.factors)} component(s) in "
                 f"{len(self.groups) or 1} draw(s), {', '.join(self.quantities())}"]
        for number, group in enumerate(self.groups):
            lines.append(f"  [{number}] " + ", ".join(c.describe() for c in group))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Applying
    # ------------------------------------------------------------------

    def apply(self, function1d, component: ComponentKey):
        """Perturb *function1d* with this set's block for *component*.

        Returns ``(perturbed, diagnostics)``, the pair
        :func:`~kika.nuclear_data.model.perturbation.applyFactors` returns.
        Raises :class:`KeyError` for a component this set does not carry, rather
        than returning the function unchanged: "no perturbation for this one"
        and "a perturbation of one" are different answers, and silently
        conflating them is how an ensemble comes to be narrower than it claims.
        """
        from kika.nuclear_data.model.perturbation import applyFactors

        if component not in self.factors:
            raise KeyError(
                f"this PerturbationSet perturbs "
                f"{[c.describe() for c in self.components()]}, not "
                f"{component.describe()}"
            )
        return applyFactors(function1d, self.factors[component],
                            self.binEdges[component])

    def _crossSectionBlocks(self) -> Dict[Tuple[int, int], ComponentKey]:
        """Which component perturbs each reaction's ``sigma(E)``, and the refusal.

        Two components can claim the same cross section: MF33's own block, and
        MF34's Legendre order 0, which is the *magnitude* -- in MF4 ``a_0`` is
        identically 1 and the size lives in MF3, so an L=0 covariance is a
        statement about ``sigma(E)`` on MF34's grid. Applying both would
        multiply the cross section twice and neither factor would be what the
        file said.

        Refused rather than reconciled. Which of the two a run wants is a
        modelling decision -- they are two estimates of the same uncertainty on
        two grids -- and it belongs in the request, not in an applier picking
        one silently.
        """
        claims: Dict[Tuple[int, int], List[ComponentKey]] = {}
        for component in self.components():
            isMagnitude = component.mf == 34 and component.index == 0
            if component.mf == 33 or isMagnitude:
                claims.setdefault((component.za, component.mt), []).append(component)

        resolved = {}
        for reaction, components in claims.items():
            if len(components) > 1:
                raise ValueError(
                    f"ZA {reaction[0]} MT{reaction[1]}: "
                    f"{[c.describe() for c in components]} all perturb the same "
                    f"cross section -- MF33 states its uncertainty and MF34's "
                    f"L=0 states the magnitude's, and applying both multiplies "
                    f"sigma twice. Ask for one of them"
                )
            resolved[reaction] = components[0]
        return resolved

    def applyToSuite(self, suite, *, multiplicityResolver=None,
                     displaced: Optional[Dict[ComponentKey, Any]] = None
                     ) -> Dict[ComponentKey, Dict[str, Any]]:
        """Put a perturbed form under :attr:`label` on every node this set covers.

        The evaluated form is left where it is and the realisation goes beside
        it, which is what §9.1's multi-form container and §9.3's ``realization``
        style are for -- and what ``encodeMF3MT(..., label=)`` then writes out.
        Nodes this set does not cover are not touched and not reported as
        perturbed.

        Three dispatches, by what the component *is* rather than by which file
        it came from:

        * ``crossSection`` -- MF33, and MF34's L=0 magnitude, which lands on the
          same node. See :meth:`_crossSectionBlocks` for why both at once is
          refused.
        * ``angularDistribution`` -- MF34's L>=1, all orders of one reaction in
          one call, because a Legendre vector is perturbed once and not once per
          order.
        * ``multiplicity`` -- MF31. It **replaces** the form rather than sitting
          beside it, and that asymmetry is an open model decision rather than an
          oversight; see :meth:`_applyMultiplicity`. It needs
          *multiplicityResolver*, and without one it raises rather than skipping:
          a nu-bar silently left unperturbed inside a realisation that claims to
          carry it is exactly the failure this class exists to prevent.

        Parameters
        ----------
        multiplicityResolver
            ``callable(suite, mt) -> Multiplicity``, for MF31 only. The model
            does not know which node an ENDF MT names -- that is the adapter's
            question, and ``kika.endf.model_adapter.multiplicity.nubarNode`` is
            its answer, the same one the MF1 encoders use. Passing it in rather
            than importing it is what keeps the sampling layer off the adapter.
        displaced
            A dict the applier fills with ``component -> the form it replaced``,
            so a caller can put the evaluation back. Required for MF31, because
            without it the replacement cannot be undone.
        """
        from kika.nuclear_data.model import EVAL_LABEL

        diagnostics: Dict[ComponentKey, Dict[str, Any]] = {}

        multiplicities = [c for c in self.components() if c.mf == 31]
        if multiplicities:
            diagnostics.update(self._applyMultiplicity(
                suite, multiplicities, multiplicityResolver, displaced))

        # `reactionByENDF_MT` raises for an MT the suite does not have, and
        # that is what should happen: a request named the reaction, so its
        # absence is a mistake in the request or a tape that does not carry it,
        # not something to skip past. It searches `sums` too, which is where the
        # ENDF adapter now puts MT1 and MT4.
        for (_za, mt), component in self._crossSectionBlocks().items():
            reaction = suite.reactionByENDF_MT(mt)
            perturbed, info = self.apply(reaction.crossSection[EVAL_LABEL],
                                         component)
            reaction.crossSection[self.label] = self._labelled(perturbed)
            diagnostics[component] = info

        byReaction: Dict[Tuple[int, int], Dict[int, ComponentKey]] = {}
        for component in self.components():
            if component.mf == 34 and component.index != 0:
                byReaction.setdefault((component.za, component.mt), {})[
                    component.index] = component
        for (_za, mt), orders in byReaction.items():
            reaction = suite.reactionByENDF_MT(mt)
            product, angular = self._angularOf(reaction, mt)
            perturbed, info = self._applyAngular(angular, orders)
            self._putRealisation(product, perturbed)
            for order, component in orders.items():
                diagnostics[component] = {
                    "n_inserted": info["n_inserted"],
                    **info["per_order"].get(order, {}),
                }
        return diagnostics

    def _applyMultiplicity(self, suite, components, resolver, displaced):
        """Put a nu-bar realisation on the node an MT names, and say what it cost.

        **The three MTs are three different nodes, measured rather than assumed**
        (``micro_u235_nubar.endf``, 2026-09-06): MT456, the prompt nu-bar, is the
        fission product's own ``multiplicity`` -- a ``Regions1d`` of 95 points --
        while MT452 and MT455 are ``multiplicitySum`` nodes under ``suite.sums``,
        "total fission neutron multiplicity" and "delayed fission neutron
        multiplicity", because §18.4 puts delayed neutrons on precursor families
        and the aggregate is a sum over them. All three carry a real
        ``Regions1d``, so the *arithmetic* is :func:`applyFactors` unchanged.

        **What is not unchanged is where the realisation goes.** A
        ``Multiplicity`` is not a ``Component`` -- §17.3's census found one form
        in 230 562 nodes -- so there is no labelled slot beside the evaluation
        and the realisation has to take the form's place. Two consequences, and
        neither is hidden:

        * the evaluated form is handed back through *displaced* and has to be put
          back, which is what
          :func:`kika.sampling.model_perturbation._forget` does once a sample is
          written;
        * a GNDS file written from a suite in this state carries the nu-bar
          realisation *instead of* the evaluation, where the cross section and the
          distribution carry both. That is the asymmetry making ``Multiplicity`` a
          ``Component`` would remove, and it is what M5 exists to decide -- taken
          by nobody here.

        **The sum rule is enforced, not checked.** ENDF-6 requires
        nu_452 = nu_455 + nu_456, so the family is not perturbed member by
        member: it goes through
        :func:`~kika.sampling.mf31_sampling.perturbNubarFamilyOnModel`, which is
        the model-side twin of the rule this project already runs -- perturb the
        components, **derive** the redundant member from them, and discard the
        derived member's own factor block. A member with no block of its own
        rides the total's when there is one, so "perturb everything with the
        total" still holds the rule; a family that is not derivable, because the
        tape does not carry every contributor, is perturbed member by member, as
        it is there.

        A consequence worth stating: rebuilding the derived member also repairs
        whatever residual the input evaluation carried, which moves the central
        value by something the perturbation did not ask for.
        :func:`~kika.sampling.mf31_sampling.sum_rule_residual` is what puts a
        number on that.
        """
        if resolver is None:
            raise NotImplementedError(
                f"{[c.describe() for c in components]}: a multiplicity has no "
                f"labelled form to put a realisation under -- `Multiplicity` is "
                f"not a `Component` (one form in the whole library, §17.3), so a "
                f"realisation has to replace it. Pass multiplicityResolver "
                f"(kika.endf.model_adapter.multiplicity.nubarNode) and a "
                f"displaced dict to say that is what you want. Whether the model "
                f"should grow a labelled slot instead is a model decision, M5 in "
                f"docs/library/perturbation_model_roadmap.md and D29 in "
                f"library-gaps.md, and it is not an applier's to take"
            )
        if displaced is None:
            raise ValueError(
                "perturbing a multiplicity replaces the evaluated form, so a "
                "`displaced` dict is required: without it the evaluation cannot "
                "be put back and the suite silently stops carrying it"
            )

        from kika.sampling.mf31_sampling import perturbNubarFamilyOnModel

        nodes = {}
        for mt in (452, 455, 456):
            try:
                node = resolver(suite, mt)
            except (KeyError, ValueError):
                node = None
            if node is not None and getattr(node, "form", None) is not None:
                nodes[mt] = node
        for component in components:
            if component.mt not in nodes:
                raise ValueError(
                    f"{component.describe()}: this suite carries no such nu-bar, "
                    f"so there is nothing to perturb"
                )

        grid = self._familyGrid(components, self.binEdges)
        blocks = {component.mt: self.factors[component] for component in components}
        forms = {mt: node.form for mt, node in nodes.items()}
        perturbed, info = perturbNubarFamilyOnModel(forms, blocks, grid)

        byMT = {component.mt: component for component in components}
        za = components[0].za
        diagnostics = {}
        for mt, form in perturbed.items():
            if form is forms[mt]:
                continue                      # untouched, and not reported as
            # The derived member has no component in the request -- its own
            # factor block is discarded by the rule -- and it is still part of
            # this realisation: it was rewritten, it has to be emitted, and its
            # diagnostics carry the residual the rebuild repaired. So it gets a
            # key of its own rather than being dropped for not having been asked
            # for.
            component = byMT.get(mt) or ComponentKey(za, 31, mt)
            displaced[component] = forms[mt]
            nodes[mt].form = self._labelled(form)
            diagnostics[component] = info.get(mt, {})
        return diagnostics


    @staticmethod
    def _familyGrid(components, binEdges):
        """The one grid the family is perturbed on, or a refusal.

        :func:`~kika.sampling.mf31_sampling.perturbNubarFamilyOnModel` takes a
        single ``bins`` array for the whole family, and it has to: the derived
        member is rebuilt from the others, so "which bin is this factor for" must
        mean the same thing for all of them. MF31 assembles under the ``global``
        union, where every component is stated on the pooled grid, so this holds
        by construction -- and it is checked rather than assumed, because a
        per-component assembly would break it silently and the result would be a
        total derived from parts perturbed on grids that do not line up.
        """
        grids = [np.asarray(binEdges[component], dtype=float)
                 for component in components]
        for grid in grids[1:]:
            if grid.shape != grids[0].shape or not np.array_equal(grid, grids[0]):
                raise ValueError(
                    "the nu-bar family is perturbed on one grid and these blocks "
                    "are stated on different ones; MF31 assembles under the "
                    "'global' union for exactly this reason"
                )
        return grids[0]


    def _applyAngular(self, angular, orders: Mapping[int, ComponentKey]):
        from kika.nuclear_data.model.perturbation import applyLegendreFactors

        return applyLegendreFactors(
            angular,
            {order: self.factors[component] for order, component in orders.items()},
            {order: self.binEdges[component] for order, component in orders.items()},
        )

    @staticmethod
    def _angularOf(reaction, mt: int):
        """The product whose evaluated distribution carries Legendre coefficients.

        MF34 is a covariance of *the* angular distribution of the reaction, and
        ENDF has one per MT; §17.2.1 hangs it on a product, so the product has
        to be found. Raises rather than guessing when a channel has two
        candidates -- a recoil and its partner both carry an angular
        distribution, and perturbing the wrong one is invisible.
        """
        from kika.nuclear_data.model import EVAL_LABEL

        channel = getattr(reaction, "outputChannel", None)
        candidates = []
        for product in (getattr(channel, "products", None) or ()):
            distribution = getattr(product, "distribution", None)
            if distribution is None:
                continue
            form = (distribution.get(EVAL_LABEL)
                    if hasattr(distribution, "get") else distribution)
            if form is not None and getattr(form, "angular", None) is not None:
                candidates.append((product, form.angular))
        if not candidates:
            raise ValueError(
                f"MT{mt} has no product carrying an evaluated angular "
                f"distribution, so an MF34 perturbation has nothing to act on"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"MT{mt} has {len(candidates)} products carrying an angular "
                f"distribution and MF34 states one covariance; which one it is "
                f"about is not in the file, so it is not for this to guess"
            )
        return candidates[0]

    def _labelled(self, form):
        """*form* carrying this realisation's label as its own.

        **Not a formality, and the GNDS writer is what proves it.** A §9.1
        container keys its forms by label, and a form also carries one; the
        applier returns a node that kept the evaluated form's, so a realisation
        stored under ``realization-0007`` still said ``eval`` about itself.
        ``gnds/encode.py``'s ``crossSection`` writes ``_function(element, form)``
        and that takes the label off the *form*, so both cross sections came out
        as ``label="eval"`` -- two forms with one label, which no reader can
        tell apart and which the schema does not allow. The ENDF path never saw
        it: ``encodeMF3MT(..., label=)`` looks the form up by key and writes one
        section. So the container and the form have to agree, and this is where
        they are made to.
        """
        import dataclasses

        if getattr(form, "label", None) == self.label:
            return form
        try:
            return dataclasses.replace(form, label=self.label)
        except TypeError:
            return form

    def _putRealisation(self, product, angular) -> None:
        """The perturbed distribution, beside the evaluated one, under the label."""
        import dataclasses

        from kika.nuclear_data.model import EVAL_LABEL

        distribution = product.distribution
        form = distribution[EVAL_LABEL]
        distribution[self.label] = self._labelled(
            dataclasses.replace(form, angular=angular))

    # ------------------------------------------------------------------
    # On disk
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-ready mapping. Grids and factors become lists of floats."""
        return {
            "format": _FORMAT_VERSION,
            "label": self.label,
            "semantics": self.semantics,
            "edgeRule": self.edgeRule,
            "provenance": dict(self.provenance),
            "groups": [[list(component) for component in group]
                       for group in self.groups],
            "blocks": [
                {
                    "component": list(component),
                    "quantity": component.quantity,
                    "factors": [float(v) for v in self.factors[component]],
                    "binEdges": [float(e) for e in self.binEdges[component]],
                }
                for component in self.components()
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PerturbationSet":
        version = int(data.get("format", 0))
        if version != _FORMAT_VERSION:
            raise ValueError(
                f"perturbation set format {version}, this kika writes and reads "
                f"{_FORMAT_VERSION}"
            )
        factors, binEdges = {}, {}
        for block in data["blocks"]:
            component = ComponentKey(*(int(v) for v in block["component"]))
            factors[component] = np.asarray(block["factors"], dtype=float)
            binEdges[component] = np.asarray(block["binEdges"], dtype=float)
        groups = tuple(tuple(ComponentKey(*(int(v) for v in component))
                             for component in group)
                       for group in data.get("groups", ()))
        return cls(label=data["label"], factors=factors, binEdges=binEdges,
                   groups=groups, semantics=data.get("semantics", SEMANTICS[0]),
                   edgeRule=data.get("edgeRule", EDGE_RULE),
                   provenance=dict(data.get("provenance", {})))

    def write(self, path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path) -> "PerturbationSet":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def __repr__(self) -> str:
        return (f"PerturbationSet({self.label!r}, "
                f"{len(self.factors)} component(s), "
                f"{'+'.join(self.quantities())})")
