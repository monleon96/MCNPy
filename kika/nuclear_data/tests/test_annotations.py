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

**Whose namespace, though.** An inherited annotation belongs to the class that
*declared* it, and is written in the vocabulary of that class's module.
``Component`` declares ``forms: Dict[str, Any]`` and imports ``Any``;
``CrossSection`` and ``Distribution`` inherit the field and their own modules
import neither. ``typing.get_type_hints`` normally handles that — it walks the
MRO and resolves each base in its own module's globals — but the moment it is
handed an explicit ``globalns`` it uses that one namespace for *every* base, so
the base's own imports stop counting and the subclass is asked for a name it
never needed. That is not a defect in the code: nothing evaluates these
annotations at run time, and ``from __future__ import annotations`` keeps them
strings.

So the resolution here walks the MRO itself and gives each class the namespace
of the module that declared it. The alternative — importing ``Any`` into two
modules that do not use it — would fix the symptom by making the subclass
namespace accidentally contain what the base needed, and would leave the next
inherited annotation to fail the same way.
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


_NAMESPACES: dict = {}


def _namespace_for(module) -> dict:
    """:func:`_checker_namespace`, computed once per module.

    The MRO walk below asks for the same handful of modules repeatedly, and
    building one namespace re-parses the module's source.
    """
    key = module.__name__
    if key not in _NAMESPACES:
        _NAMESPACES[key] = _checker_namespace(module)
    return _NAMESPACES[key]


def _class_hints(cls) -> dict:
    """Resolve *cls*'s annotations, each in the namespace that declared it.

    ``typing.get_type_hints(cls, globalns=ns)`` applies *ns* to every class in
    the MRO, which is wrong for an inherited annotation: it is written in its
    own module's vocabulary. This walks the MRO in reverse — base first, so a
    subclass that re-declares a field wins, which is what ``get_type_hints``
    does too — and resolves each class against the module that declared it.

    Raises ``NameError`` naming the class that actually owns the bad annotation,
    rather than the subclass that merely inherited it.
    """
    hints: dict = {}
    for base in reversed(cls.__mro__):
        own = base.__dict__.get("__annotations__") or {}
        if not own:
            continue
        module = importlib.import_module(base.__module__)
        namespace = (
            _namespace_for(module) if base.__module__.startswith(_PREFIX)
            else vars(module)
        )
        for name, annotation in own.items():
            # `base.__dict__["__annotations__"]` and not `get_type_hints(base)`:
            # that walks the MRO again and would apply this one namespace to
            # every base of *this* base, which is the very thing being fixed.
            if not isinstance(annotation, str):
                hints[name] = annotation
                continue
            try:
                hints[name] = eval(annotation, dict(namespace))  # noqa: S307
            except NameError as exc:
                raise NameError(
                    f"{base.__module__}.{base.__qualname__}.{name}: {exc}"
                ) from exc
    return hints


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
        if inspect.isclass(obj):
            _class_hints(obj)
        else:
            typing.get_type_hints(
                obj, globalns=_namespace_for(module), include_extras=True
            )
    except NameError as exc:
        pytest.fail(
            f"{label} is annotated with a name that is neither imported nor "
            f"declared under TYPE_CHECKING: {exc}"
        )


def test_dataclass_fields_resolve():
    """Field annotations too — ``dataclasses.fields()`` alone never checks them.

    ``dataclasses.fields()`` reports the *merged* set, inherited fields
    included, which is exactly where the merged ``__annotations__`` loses track
    of who declared what. Hence :func:`_class_hints`.
    """
    for module in _public_modules():
        for cls_name, cls in vars(module).items():
            if not (inspect.isclass(cls) and dataclasses.is_dataclass(cls)):
                continue
            if cls.__module__ != module.__name__:
                continue
            hints = _class_hints(cls)
            missing = {f.name for f in dataclasses.fields(cls)} - set(hints)
            assert not missing, (
                f"{module.__name__}.{cls_name} has fields with unresolvable "
                f"annotations: {sorted(missing)}"
            )
