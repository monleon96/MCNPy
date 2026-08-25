"""§9.1's form container, and the two ratchets that come with having one.

The abstraction is not here to save lines — there are only two containers in the
model and ``Multiplicity`` is measured not to be a third. It is here because the
two copies had drifted apart, and drift is what these tests catch.
"""
from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

from kika.nuclear_data import model
from kika.nuclear_data.model.component import EVAL_LABEL, Component

MODEL_DIR = pathlib.Path(inspect.getfile(model)).parent


@pytest.mark.parametrize("cls", [model.CrossSection, model.Distribution])
def test_the_form_containers_share_one_implementation(cls):
    """§16.1.1 and §18.1.1 are the same shape, so they are the same code."""
    assert issubclass(cls, Component)


def test_nothing_hand_rolls_a_second_form_mapping():
    """The ratchet: a third ``forms`` dict written by hand instead of inherited.

    This is how the divergence happened the first time. ``Distribution`` was
    written as its own mapping and quietly ended up without ``evaluated``, so its
    callers reached into ``.forms`` and re-implemented it — twice, differently.
    """
    offenders = []
    for path in MODEL_DIR.rglob("*.py"):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            declaresForms = any(
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and stmt.target.id == "forms"
                for stmt in node.body
            )
            if declaresForms and node.name != "Component":
                offenders.append(f"{path.name}:{node.name}")
    assert not offenders, (
        f"{offenders} declare their own `forms` mapping. Inherit from "
        f"`Component` instead; §9.1 is one shape and it has one implementation."
    )


def test_no_class_in_the_model_defines_a_method_twice():
    """``CrossSection`` had **two** ``__repr__``, and the second won.

    The first — the one written to print ``(no forms decoded)`` and to sort the
    labels — could never run, and nothing said so: both definitions are valid
    Python and the class works either way. Python has no warning for it and no
    linter is configured in this repo, so this is the only thing that would.
    """
    offenders = []
    for path in MODEL_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            seen = set()
            for stmt in node.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if stmt.name in seen:
                        offenders.append(f"{path.name}:{node.name}.{stmt.name}")
                    seen.add(stmt.name)
    assert not offenders, (
        f"{offenders} are defined twice in one class body; the later definition "
        f"silently shadows the earlier one."
    )


@pytest.mark.parametrize("cls", [model.CrossSection, model.Distribution])
def test_an_empty_container_is_present_and_says_so(cls):
    """Declared-and-empty is not absent — the model's rule, and now in one place."""
    container = cls()
    assert bool(container) is True
    assert len(container) == 0
    assert repr(container) == f"{cls.__name__}(no forms decoded)"


@pytest.mark.parametrize("cls", [model.CrossSection, model.Distribution])
def test_the_missing_label_error_names_the_node_and_what_it_has(cls):
    container = cls()
    container["recon"] = object()
    with pytest.raises(KeyError) as raised:
        container[EVAL_LABEL]
    message = str(raised.value)
    assert cls.gndsNodeName in message and "'recon'" in message


@pytest.mark.parametrize("cls", [model.CrossSection, model.Distribution])
def test_evaluated_is_reachable_without_touching_the_dict(cls):
    """What ``Distribution`` lacked, and what its callers were working round."""
    form = object()
    container = cls()
    assert container.hasEvaluated is False
    assert container.get(EVAL_LABEL) is None
    container[EVAL_LABEL] = form
    assert container.hasEvaluated is True
    assert container.evaluated is form
    assert list(container.items()) == [(EVAL_LABEL, form)]
    assert list(container) == [EVAL_LABEL]
    assert EVAL_LABEL in container


def test_only_the_cross_section_can_be_evaluated_at_a_point():
    """σ(E) is 1-d; §18.1.1's forms are not, so ``evaluate`` does not generalise."""
    assert hasattr(model.CrossSection, "evaluate")
    assert not hasattr(model.Distribution, "evaluate")


def test_multiplicity_is_deliberately_not_a_component():
    """§17.3 *could* be a mapping. The census says it is not one in practice.

    All 230 562 ``<multiplicity>`` nodes in ENDF/B-VIII.1-GNDS carry exactly one
    form, all labelled ``eval``. Recorded as a test so that "finish the
    refactor" is a decision someone has to overturn rather than one they can
    make by tidying.
    """
    assert not issubclass(model.Multiplicity, Component)
    assert "form" in {f.name for f in __import__("dataclasses").fields(model.Multiplicity)}
