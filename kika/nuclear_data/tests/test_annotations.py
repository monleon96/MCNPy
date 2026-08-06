"""Every annotation in the canonical layer must name something the module has.

``kika/nuclear_data`` uses ``from __future__ import annotations``, so a name
used in a signature but never imported is not a runtime error — the annotation
is just a string nobody evaluates. It only surfaces when something *does*
evaluate it: ``typing.get_type_hints``, Sphinx's ``autodoc_typehints``, or a
type checker.

That is how ``CrossSection.get_cross_section`` shipped with ``Union`` and
``ArrayLike`` in its signature and neither one imported anywhere. Nothing in
the suite looked, so nothing complained.

The invariant is *not* "every annotation resolves at runtime". It deliberately
cannot be: this layer must not import ``kika.endf`` or ``kika.ace`` at runtime
(see ``kika/tests/test_layering.py``), so ``MF3MT``, ``Ace`` and friends live
under ``if TYPE_CHECKING`` and are absent when the module is loaded. That is
correct and must stay that way.

The invariant is: **every name in an annotation is either imported normally or
declared under ``TYPE_CHECKING``.** Neither is what the defect looked like. So
this test rebuilds the namespace a type checker would see — module globals plus
the ``TYPE_CHECKING`` imports, executed here in the test where reaching into
the format packages is allowed — and resolves against that.
"""
from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import pkgutil
import typing

import pytest

import kika.nuclear_data


_PREFIX = "kika.nuclear_data."


def _public_modules():
    """Every module in ``kika.nuclear_data``, **recursively**, tests excluded.

    ``walk_packages`` rather than ``iter_modules``: phase 3 of the GNDS roadmap
    adds ``kika/nuclear_data/model/`` with subpackages of its own, and
    ``iter_modules`` would have reached only their ``__init__``. Walking also
    means an import error anywhere in the new model fails this suite the first
    time it runs, which is the cheapest place to find one.
    """
    for info in pkgutil.walk_packages(kika.nuclear_data.__path__, prefix=_PREFIX):
        parts = info.name[len(_PREFIX):].split(".")
        if any(part.startswith("_") or part == "tests" for part in parts):
            continue
        yield importlib.import_module(info.name)


def _checker_namespace(module) -> dict:
    """Module globals plus the names it imports under ``if TYPE_CHECKING``.

    This is the namespace a type checker or a documentation build resolves
    annotations in. Executing the guarded imports is fine *here* — the ban on
    importing format packages applies to the library, not to its tests.
    """
    namespace = dict(vars(module))
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        guarded = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
            isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
        )
        if not guarded:
            continue
        for stmt in node.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                exec(compile(ast.Module([stmt], []), "<type-checking>", "exec"), namespace)
    return namespace


def _annotated_callables():
    """(label, module, object) for every public class, function and method."""
    for module in _public_modules():
        for cls_name, cls in vars(module).items():
            if not inspect.isclass(cls) or cls.__module__ != module.__name__:
                continue
            yield f"{module.__name__}.{cls_name}", module, cls
            for fn_name, fn in vars(cls).items():
                if fn_name.startswith("_"):
                    continue
                fn = getattr(fn, "__func__", fn)  # unwrap class/staticmethod
                if inspect.isfunction(fn):
                    yield f"{module.__name__}.{cls_name}.{fn_name}", module, fn


_CASES = list(_annotated_callables())

# A package with no classes would make this test silently vacuous.
assert _CASES, "no annotated callables found in kika.nuclear_data"


@pytest.mark.parametrize(
    "label,module,obj", _CASES, ids=[case[0] for case in _CASES]
)
def test_annotations_name_something_the_module_has(label, module, obj):
    try:
        typing.get_type_hints(
            obj, globalns=_checker_namespace(module), include_extras=True
        )
    except NameError as exc:
        pytest.fail(
            f"{label} is annotated with a name that is neither imported nor "
            f"declared under TYPE_CHECKING: {exc}"
        )


def test_dataclass_fields_resolve():
    """Field annotations too — ``dataclasses.fields()`` alone never checks them."""
    for module in _public_modules():
        namespace = _checker_namespace(module)
        for cls_name, cls in vars(module).items():
            if not (inspect.isclass(cls) and dataclasses.is_dataclass(cls)):
                continue
            if cls.__module__ != module.__name__:
                continue
            hints = typing.get_type_hints(cls, globalns=namespace)
            missing = {f.name for f in dataclasses.fields(cls)} - set(hints)
            assert not missing, (
                f"{module.__name__}.{cls_name} has fields with unresolvable "
                f"annotations: {sorted(missing)}"
            )
