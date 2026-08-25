"""GNDS-2.1 §9.1: the container that holds one *form* per style label.

**§9.1 describes the shape and never names it.** A data container holds several
forms, each tagged with the ``label`` of a style, and the spec's own example is a
``crossSection`` carrying a ``resonancesWithBackground`` labelled ``eval`` beside
an ``XYs1d`` labelled ``recon``. What the document does not do is give that
container a node name: every instance of it is spelled as the quantity it holds
-- ``crossSection``, ``distribution`` -- and never as the pattern. FUDGE names
the pattern ``component``, and that is the name taken here.

**This is a base class and not a node**, which is why it does not breach the
package's class-name-is-the-GNDS-node-name rule: nothing is ever written as
``<component>``. The nodes keep their own names --
:class:`~kika.nuclear_data.model.cross_section_forms.CrossSection`,
:class:`~kika.nuclear_data.model.distributions.Distribution` -- and inherit the
mapping from here, so a FUDGE reader meets a concept they know without kika
inventing a noun GNDS does not have. Declared in ``tests/test_gnds_naming.py``'s
``DIVERGENCES`` so it is not mistaken for a spelling error.

**Why a base at all, when the model has exactly two of them.** ``multiplicity``
(§17.3) is an ``xs:choice`` with ``maxOccurs="unbounded"`` and so *could* be one,
but the census in :class:`~kika.nuclear_data.model.output_channel.Multiplicity`
settles it -- all 230 562 nodes in ENDF/B-VIII.1-GNDS carry one form and all are
labelled ``eval`` -- and ``Q`` (§17.1.1) is a value node, not a container. Two
instances is not enough duplication to pay for an abstraction on its own.

What pays for it is that the two copies had **already diverged**, and the
divergence was leaking:

* ``Distribution`` grew neither :attr:`~Component.evaluated` nor a ``KeyError``
  naming the labels it does hold, so its callers reached past the container into
  the raw dict and re-implemented both -- ``product.distribution.forms.get(
  EVAL_LABEL)`` in ``kika/endf/model_adapter/decode.py`` and
  ``{} if distribution is None else distribution.forms`` in ``kika/gnds/
  encode.py``. Every one of those is a local copy of a rule §18.1.1 states once.
* ``CrossSection`` had ended up with **two** ``__repr__`` definitions, the second
  silently shadowing the first, so the ``(no forms decoded)`` message the first
  was written to print could never appear and the labels came out unsorted.

One implementation removes that class of drift by construction, which is a
different and better reason than saving twenty lines.

**What is deliberately *not* here.**

``evaluate()`` stays on :class:`CrossSection`. sigma(E) is a one-dimensional
question; a distribution form is two- or three-dimensional and ``evaluate(x)``
on one either means something else or means nothing. On the base it would be
surface that looks callable everywhere and answers nowhere.

There is no ``getForm(style, inherit=True)``. FUDGE's ``inherit`` walks
``derivedFrom``, and kika already holds that chain -- in
:class:`~kika.nuclear_data.model.styles.Styles`, where §9 puts it. Resolving it
from inside a container would mean every ``crossSection`` carrying a reference
back to the suite's ``styles`` node just to answer a lookup. The chain is
resolved on ``styles`` and the resolved label is handed to the mapping, which
keeps this class a plain mapping and keeps the provenance in one place.

The class is not ``Generic[FormT]`` either. §16.1.1's and §18.1.1's choices are
open sets of unrelated dataclasses with no common base -- ``ResonancesWithBack\
ground``, ``Reference``, ``XYs1d``, ``Uncorrelated`` -- so the parameter could
only ever be bound to ``object`` or to a union that no checker gains anything
from. Inventing a ``Form`` protocol to give it something to bind to would be
patching a type onto the model from outside instead of letting the model say
what it is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, ItemsView, Iterator, KeysView, ValuesView

__all__ = ["Component", "EVAL_LABEL"]

#: §16.1.1: *"For evaluated files, one element must contain the label='eval'
#: attribute."* §18.1.1 says the same of a distribution, so the constant belongs
#: to the container rather than to either node.
EVAL_LABEL = "eval"


@dataclass
class Component:
    """A mapping from style label to form, per §9.1.

    ``component['eval']`` is the evaluated form and ``component['recon']`` the
    reconstructed one, side by side and each labelled, instead of one silently
    replacing the other.
    """

    forms: Dict[str, Any] = field(default_factory=dict)

    #: The node the subclass actually is, used in the messages. ``Component``
    #: itself is never serialised, so the base value is only ever a fallback.
    #: A ``ClassVar`` and not a field: it is a property of the class, and a
    #: dataclass field would put it in ``__init__`` where a caller could set it.
    gndsNodeName: ClassVar[str] = "component"

    # -- the mapping -------------------------------------------------------

    def __len__(self) -> int:
        return len(self.forms)

    def __iter__(self) -> Iterator[str]:
        return iter(self.forms)

    def __bool__(self) -> bool:
        # A declared slot is *present* even when empty; only `len()` speaks to
        # content. Without this, `if reaction.crossSection:` would read a
        # reaction whose forms have not been decoded yet as one that has no
        # cross section at all.
        return True

    def __contains__(self, label: object) -> bool:
        return label in self.forms

    def __getitem__(self, label: str) -> Any:
        try:
            return self.forms[label]
        except KeyError:
            raise KeyError(
                f"this {self.gndsNodeName} has no form labelled {label!r}; "
                f"it has {sorted(self.forms)}"
            ) from None

    def __setitem__(self, label: str, form: Any) -> None:
        self.forms[label] = form

    def get(self, label: str, default: Any = None) -> Any:
        """The form under ``label``, or ``default``.

        Here so that "this form may not have been decoded" can be said through
        the container. Every caller that reached into ``.forms`` to say it was
        one that had no other way to.
        """
        return self.forms.get(label, default)

    def keys(self) -> KeysView[str]:
        return self.forms.keys()

    def values(self) -> ValuesView[Any]:
        return self.forms.values()

    def items(self) -> ItemsView[str, Any]:
        return self.forms.items()

    def __repr__(self) -> str:
        if not self.forms:
            return f"{type(self).__name__}(no forms decoded)"
        described = ", ".join(
            f"{label}={type(form).__name__}"
            for label, form in sorted(self.forms.items())
        )
        return f"{type(self).__name__}({described})"

    # -- §9.1's evaluated form ---------------------------------------------

    @property
    def evaluated(self) -> Any:
        """The ``label='eval'`` form the spec requires of an evaluated file."""
        return self[EVAL_LABEL]

    @property
    def hasEvaluated(self) -> bool:
        return EVAL_LABEL in self.forms
