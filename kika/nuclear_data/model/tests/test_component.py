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


def _extendsAProperty(stmt) -> bool:
    """Is this second definition a ``@name.setter`` / ``.getter`` / ``.deleter``?

    A property accessor redefines the name **on purpose and out loud** -- the
    decorator names the property it is extending -- so it is not the failure
    this test is about, which is a second definition that shadows the first
    silently. ``Multiplicity.form`` is the first one in the model: a read
    property over ``forms`` plus a setter that keeps ``multiplicity.form = x``
    working for the code that built nodes that way.
    """
    for decorator in stmt.decorator_list:
        if (isinstance(decorator, ast.Attribute)
                and decorator.attr in ("setter", "getter", "deleter")
                and isinstance(decorator.value, ast.Name)
                and decorator.value.id == stmt.name):
            return True
    return False


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
                    if stmt.name in seen and not _extendsAProperty(stmt):
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


def test_multiplicity_is_a_component_and_form_still_means_the_evaluated_one():
    """§17.3 is a mapping since 2026-09-06, and the reason it was not is intact.

    The census that kept it a single form still says what it said -- all 230 562
    ``<multiplicity>`` nodes in ENDF/B-VIII.1-GNDS carry one form, labelled
    ``eval`` -- and it stopped being the right question. It describes what
    distributed libraries contain; a perturbation run writes a realisation of
    the nu-bar beside its evaluation, so kika is now the library that ships the
    second label. The old docstring named that trigger in advance.

    What has to hold for the change to be free is that ``.form`` still answers
    what it always answered, so no reader had to move. That is what this pins,
    together with the case that has no honest answer.
    """
    assert issubclass(model.Multiplicity, Component)

    empty = model.Multiplicity()
    assert empty.form is None and len(empty) == 0

    evaluated = object()
    one = model.Multiplicity(form=evaluated)
    assert one.form is evaluated
    assert list(one.keys()) == [EVAL_LABEL], (
        "a form with no label of its own is filed under 'eval', which is what "
        "every decoder produced before this was a mapping")

    one["realization-0007"] = object()
    assert one.form is evaluated, (
        "a realisation must not displace the evaluation; that is the whole "
        "point of the container")

    onlyOther = model.Multiplicity(forms={"recon": evaluated})
    assert onlyOther.form is evaluated, (
        "a single form under another label is still 'the' form -- a GNDS "
        "document may label its only multiplicity anything")

    ambiguous = model.Multiplicity(forms={"recon": object(), "other": object()})
    with pytest.raises(KeyError, match="not a question with one answer"):
        ambiguous.form
