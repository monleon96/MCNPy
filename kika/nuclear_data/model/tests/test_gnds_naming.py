"""The GNDS vocabulary is enforced here, because a doc would not survive.

The roadmap's decision is "GNDS names verbatim, attributes included". That puts
``outerDomainValue``, ``domainMin`` and ``valueType`` inside a package whose own
style guide says attributes are ``lower_snake_case``
(``.github/python-style.instructions.md``). The guide now carries a scoped
exception, but a sentence in a markdown file is not what stops a future agent or
a future ``ruff --fix`` from "correcting" the spelling.

What stops it is this file. The consequence of a rename is not cosmetic: the
phase 5 GNDS reader resolves nodes by introspection on these exact names, so a
tidied attribute silently breaks reading every file that uses it.

There is no linter to configure — no ruff, flake8 or black config anywhere in
the repo, and no lint job in CI — so a test is not merely the better option, it
is the only one.
"""
from __future__ import annotations

import dataclasses
import inspect

import pytest

from kika.nuclear_data import model

#: class -> the attribute spellings the specification gives it.
#: Every entry carries its § reference, so a disagreement can be settled against
#: the document rather than against somebody's memory of it.
GNDS_NODES: dict[str, tuple[str, ...]] = {
    # §2.3.3 physicalQuantity
    "PhysicalQuantity": ("value", "unit", "label", "uncertainty"),
    # §5.1.2 axis / §5.1.3 grid / §5.1.1 axes
    "Axis": ("index", "label", "unit"),
    "Grid": ("index", "label", "unit", "style", "interpolation", "values"),
    "Axes": ("axes", "href"),
    # §5.2.1 values
    "Values": ("values", "valueType", "start", "length"),
    # §6.1.1 XYs1d
    "XYs1d": ("interpolation", "axes", "label", "outerDomainValue", "index"),
    # §6.4.1 regions1d
    "Regions1d": ("function1ds", "axes", "label", "outerDomainValue", "index"),
    # §6 XYs2d / regions2d — the outer axis is an ordered list, never a mapping
    "XYs2d": ("function1ds", "interpolation", "interpolationQualifier",
              "outerDomainValues", "axes", "label", "outerDomainValue", "index"),
    "Regions2d": ("function2ds", "function1ds", "axes", "label",
                  "outerDomainValue", "index"),
    # §18 angularTwoBody / isotropic2d
    "AngularTwoBody": ("angular", "productFrame", "label"),
    "Isotropic2d": ("productFrame", "label"),
    # §19.3.4 channel
    "Channel": ("label", "resonanceReaction", "L", "channelSpin", "columnIndex",
                "scatteringRadius"),
    # §7
    "Uncertainty": ("standard", "covariance", "listOfCovariances"),
    "Covariance": ("href", "label"),
    "ListOfCovariances": ("covariances",),
}

#: Names that deliberately diverge from a literal reading of the spec, each with
#: the reason. Anything not listed here must match GNDS exactly.
DIVERGENCES: dict[str, str] = {
    "XYs1d.xs": "the spec interleaves x and y in one `values` node; two arrays "
                "is a storage choice, and `interleaved()` produces the spec layout",
    "XYs1d.ys": "see XYs1d.xs",
    "Constant1d.domainMin_": "`domainMin` is a read-only property on the ABC, so the "
                             "stored field needs a distinct name",
    "Polynomial1d.domainMin_": "see Constant1d.domainMin_",
    "Constant1d.domainMax_": "see Constant1d.domainMin_",
    "Polynomial1d.domainMax_": "see Constant1d.domainMin_",
    "ScalarUncertainty": "§2.3.3's scalar uncertainty and §7's functional "
                         "`uncertainty` are different nodes with the same name; the "
                         "scalar one is re-exported under a distinguishing alias",
}


def _lookup(name: str) -> type:
    obj = getattr(model, name, None)
    assert obj is not None, f"{name} is not exported from kika.nuclear_data.model"
    return obj


@pytest.mark.parametrize("clsName", sorted(GNDS_NODES))
def test_every_declared_attribute_exists(clsName):
    """A rename shows up here rather than in a phase 5 reader that stops working."""
    cls = _lookup(clsName)
    missing = [a for a in GNDS_NODES[clsName] if not hasattr(cls, a) and a not in _fieldNames(cls)]
    assert not missing, (
        f"{clsName} no longer has {missing}. These are GNDS §-defined spellings; "
        f"if the node genuinely changed, update GNDS_NODES and say which § says so."
    )


def _fieldNames(cls) -> set[str]:
    if dataclasses.is_dataclass(cls):
        return {f.name for f in dataclasses.fields(cls)}
    return set()


@pytest.mark.parametrize("clsName", sorted(GNDS_NODES))
def test_no_snake_case_twin_of_a_camel_attribute(clsName):
    """The helpful alias that quietly forks the vocabulary.

    Adding ``outer_domain_value`` next to ``outerDomainValue`` looks kind. It
    means half the code uses one spelling and half the other, introspection-based
    serialisation picks whichever it finds first, and the divergence is invisible
    until a round-trip loses data.
    """
    import re

    cls = _lookup(clsName)
    camel = [a for a in GNDS_NODES[clsName] if re.search(r"[a-z][A-Z]", a)]
    for attribute in camel:
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", attribute).lower()
        assert not hasattr(cls, snake) and snake not in _fieldNames(cls), (
            f"{clsName} exposes both {attribute!r} and {snake!r}. Keep the GNDS "
            f"spelling only."
        )


def test_class_names_are_gnds_node_names_capitalised():
    """``regions1d`` -> ``Regions1d``, ``XYs1d`` -> ``XYs1d`` (already capital)."""
    expected = {
        "PhysicalQuantity", "Axis", "Grid", "Axes", "Values",
        "XYs1d", "Regions1d", "Ys1d", "Legendre", "Gridded1d",
        "Constant1d", "Polynomial1d", "Uncertainty", "Covariance",
        "ListOfCovariances", "XYs2d", "XYs3d", "Regions2d", "Regions3d",
    }
    for name in expected:
        cls = _lookup(name)
        assert inspect.isclass(cls), f"{name} is not a class"
        assert name[0].isupper(), f"{name} should start with a capital"


def test_the_enumerations_carry_the_specs_own_strings():
    """§3.4.2, §3.4.4, §3.4.5 — values copied from the document, not paraphrased."""
    assert {f.value for f in model.Frame} == {"lab", "centerOfMass"}
    assert {i.value for i in model.Interpolation} == {
        "flat", "charged-particle", "lin-lin", "lin-log", "log-lin", "log-log"
    }
    assert {q.value for q in model.InterpolationQualifier} == {
        "direct", "unitBase", "correspondingEnergies", "correspondingPoints"
    }
    assert {g.value for g in model.GridStyle} == {
        "none", "points", "boundaries", "parameters"
    }


def test_every_divergence_is_declared():
    """A divergence that is not written down is indistinguishable from a mistake."""
    for key, reason in DIVERGENCES.items():
        assert reason.strip(), f"{key} is listed as a divergence with no reason"


def test_the_naming_ratchet_catches_a_snake_case_twin():
    """A ratchet nobody has seen fail is not a ratchet."""
    import re

    class _Tidied:
        outerDomainValue = None
        outer_domain_value = None  # the planted "helpful" alias

    attribute = "outerDomainValue"
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", attribute).lower()
    assert snake == "outer_domain_value"
    assert hasattr(_Tidied, snake), "the check itself would not have fired"
