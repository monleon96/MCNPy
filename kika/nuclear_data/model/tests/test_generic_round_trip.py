"""Every node survives a generic walk. The cheapest proxy for phases 5 and 7c.

The GNDS reader (phase 5) and writer (phase 7c) are months away, and both work
the same way: walk the tree generically, turn each node into its fields, and
rebuild it from them. Bugs in a *model* that only appear under a generic walk —
a field that cannot be reconstructed from what the node exposes, a container
that loses its element type, a cycle — are cheap to fix now and expensive to
find in three months, with a half-written XML reader in the way.

So the walk is written here, in the tests, deliberately naive and deliberately
not part of the model. Putting it in the model would be designing the serialiser
now, on guesses about what the reader needs; running it as a test costs nothing
and still catches the modelling mistakes.

What this does **not** test is XML, xPath ``href`` resolution, or the GNDS
element ordering. Those are phase 5's, and they need the specification's
serialisation rules rather than a dataclass walk.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from kika.nuclear_data import model as m


def to_dict(node):
    """A dataclass tree to nested plain data. Arrays become lists."""
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        return {
            "__class__": type(node).__name__,
            **{f.name: to_dict(getattr(node, f.name)) for f in dataclasses.fields(node)},
        }
    if isinstance(node, np.ndarray):
        return {"__array__": node.tolist()}
    if isinstance(node, (list, tuple)):
        return [to_dict(v) for v in node]
    if isinstance(node, dict):
        return {k: to_dict(v) for k, v in node.items()}
    if isinstance(node, m.Frame.__mro__[0]) and hasattr(node, "value"):
        return {"__enum__": str(node)}
    return node


def from_dict(data, registry):
    """The inverse, resolving class names against a registry."""
    if isinstance(data, dict) and "__class__" in data:
        cls = registry[data["__class__"]]
        kwargs = {
            k: from_dict(v, registry)
            for k, v in data.items()
            if k != "__class__" and _isInitField(cls, k)
        }
        return cls(**kwargs)
    if isinstance(data, dict) and "__array__" in data:
        return np.asarray(data["__array__"], dtype=float)
    if isinstance(data, dict) and "__enum__" in data:
        return data["__enum__"]
    if isinstance(data, dict):
        return {k: from_dict(v, registry) for k, v in data.items()}
    if isinstance(data, list):
        return [from_dict(v, registry) for v in data]
    return data


def _isInitField(cls, name: str) -> bool:
    for f in dataclasses.fields(cls):
        if f.name == name:
            return f.init
    return False


REGISTRY = {
    name: getattr(m, name)
    for name in m.__all__
    if isinstance(getattr(m, name), type)
}
REGISTRY.update({
    "Resonance": m.Resonance,
    "SpinGroup": m.SpinGroup,
    "ResonanceParameters": __import__(
        "kika.nuclear_data.model.resonances.breit_wigner", fromlist=["x"]
    ).ResonanceParameters,
})


def _nodes():
    """One populated instance of each dataclass node worth walking."""
    return {
        "PhysicalQuantity": m.PhysicalQuantity(value=55.36, unit="amu", label="mass"),
        "Axis": m.Axis(index=1, label="energy_in", unit="eV"),
        "Axes": m.crossSectionAxes(),
        "Values": m.Values(np.array([1.0, 2.0, 3.0])),
        "XYs1d": m.XYs1d(xs=[1.0, 10.0], ys=[2.0, 3.0]),
        "Regions1d": m.Regions1d.fromEndfRegions(
            np.array([1.0, 5.0, 10.0]), np.array([1.0, 2.0, 3.0]), [(2, 2), (3, 5)]
        ),
        "Evaluated": m.Evaluated(
            label="eval", library="JEFF", version="4.0",
            temperature=m.PhysicalQuantity(value=0.0, unit="K"),
            projectileEnergyDomain=m.RangeQuantity(min=1e-5, max=1.5e8, unit="eV"),
        ),
        "RangeQuantity": m.RangeQuantity(min=1e-5, max=1.5e8, unit="eV"),
        "Background": m.Background(
            resolvedRegion=m.XYs1d(xs=[1e-5, 8.5e5], ys=[0.0, 1.0]),
            fastRegion=m.XYs1d(xs=[8.5e5, 1.5e8], ys=[1.0, 2.0]),
        ),
        "ResonancesWithBackground": m.ResonancesWithBackground(
            background=m.Background(
                resolvedRegion=m.XYs1d(xs=[1e-5, 8.5e5], ys=[0.0, 1.0])
            ),
            resonanceRegionHref="/reactionSuite/resonances",
            label="eval",
        ),
        "CrossSectionSum": m.CrossSectionSum(
            id=m.ReactionId(label="total", ENDF_MT=1),
            summands=m.Summands([m.Add(href="/reactionSuite/reactions")]),
        ),
        "AngularTwoBody": m.AngularTwoBody(
            label="eval", recoilHref="../../product[@label='n']"
        ),
        "Realization": m.Realization(label="s7", derivedFrom="eval"),
        "ReactionId": m.ReactionId(label="n + Fe56", ENDF_MT=2),
        "Q": m.Q(value=0.0, unit="eV"),
        "Product": m.Product(pid="n"),
        "OutputChannel": m.OutputChannel(genre="twoBody"),
        "Slice": m.Slice(dimension=1, domainValue=1.0),
        "DataLink": m.DataLink.forLegendreOrder("/x", 1),
        "CovarianceMatrix": m.CovarianceMatrix(matrix=np.eye(3)),
        "CovarianceSection": m.CovarianceSection(label="MT2"),
        "Resonance": m.Resonance(energy=1.15e3, spin=0.5, totalWidth=1.4,
                                 neutronWidth=1.2, captureWidth=0.2),
        "SpinGroup": m.SpinGroup(L=0, resonances=[m.Resonance(energy=1.0, spin=0.5)]),
        "ResolvedRegion": m.ResolvedRegion(domainMin=1e-5, domainMax=8.5e5),
        "Particle": m.Particle(id="n"),
        "Nuclide": m.Nuclide(id="Fe56", Z=26, A=56),
    }


@pytest.mark.parametrize("name", sorted(_nodes()))
def test_every_node_round_trips_through_a_generic_walk(name):
    original = _nodes()[name]

    rebuilt = from_dict(to_dict(original), REGISTRY)

    assert type(rebuilt) is type(original)
    assert to_dict(rebuilt) == to_dict(original), (
        f"{name} does not survive a generic to_dict/from_dict walk. Phase 5's "
        f"reader and phase 7c's writer both walk the tree this way."
    )


def test_the_walk_would_notice_a_difference():
    """Otherwise the comparison above could be vacuously true."""
    a = m.XYs1d(xs=[1.0, 10.0], ys=[2.0, 3.0])
    b = m.XYs1d(xs=[1.0, 10.0], ys=[2.0, 4.0])
    assert to_dict(a) != to_dict(b)


def test_a_full_suite_walks_without_recursing_forever():
    """``Product.outputChannel`` makes the structure recursive (§17.1.1).

    Breakup and decay channels are output channels in their own right, so a
    walker that does not terminate on the leaves runs until the stack does.
    """
    suite = m.ReactionSuite(evaluation="JEFF-4.0", projectile="n", target="Fe56")
    suite.styles.add(m.Evaluated(label="eval"))
    reaction = m.Reaction(id=m.ReactionId(label="n + Fe56", ENDF_MT=2))
    reaction.crossSection["eval"] = m.XYs1d(xs=[1.0, 10.0], ys=[2.0, 3.0])
    reaction.outputChannel.products.products.append(
        m.Product(pid="n", outputChannel=m.OutputChannel(genre="twoBody"))
    )
    suite.reactions.append(reaction)

    walked = to_dict(suite.reactions.reactions[0])
    assert walked["__class__"] == "Reaction"
    assert walked["outputChannel"]["products"]["products"][0]["pid"] == "n"
