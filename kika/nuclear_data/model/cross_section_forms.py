"""GNDS-2.1 §16.1.1 ``crossSection``: several forms, one per style label.

§16.1.1 lists the allowed children — ``XYs1d``, ``regions1d``,
``resonancesWithBackground``, ``CoulombPlusNuclearElastic``,
``thermalNeutronScatteringLaw1d``, ``reference``, ``gridded1d``, ``Ys1d``,
``URR_probabilityTables1d`` — and constrains them: *"only one of children marked
with * for each unique style label is allowed"*, and *"For evaluated files, one
element must contain the label='eval' attribute."*

So a ``crossSection`` is a **mapping from style label to form**, and that is how
:class:`CrossSection` is built. The evaluated file's resonance-plus-background
representation and the reconstructed pointwise one live side by side, each
labelled, instead of one silently replacing the other.

**Naming divergence.** GNDS gives no uniform node name for "a container holding
several forms"; FUDGE calls it a *component*. The class here takes the name of
the node it actually is — ``crossSection`` → ``CrossSection`` — and the
form mapping hangs off it as ``forms``. Recorded in ``NAMING.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional

from .functions import Function1d, Gridded1d, Regions1d, XYs1d, Ys1d

__all__ = [
    "Form",
    "Background",
    "ResonancesWithBackground",
    "Reference",
    "CoulombPlusNuclearElastic",
    "ThermalNeutronScatteringLaw1d",
    "URR_probabilityTables1d",
    "CrossSection",
    "EVAL_LABEL",
]

#: §16.1.1: *"For evaluated files, one element must contain the label='eval'."*
EVAL_LABEL = "eval"

#: Anything that can be a form of a crossSection.
Form = object


@dataclass
class Background:
    """§16.1.1's ``background``: **three** functions, one per resonance region.

    Not one curve. The evaluator states a separate background over the resolved
    region, over the unresolved region and above them both, because the three
    are added to different things — reconstructed resonances, an average
    cross section, and nothing at all. FUDGE's schema
    (``CrossSectionResonanceBackgroundType``) makes ``resolvedRegion`` or
    ``unresolvedRegion`` mandatory and the rest optional, and every
    ``resonancesWithBackground`` in ENDF/B-VIII.1-GNDS carries at least two of
    the three.

    **This was a single ``Function1d`` and that was wrong**, in the way that
    does not announce itself: a reader keeping only one region produces a
    background that is right over part of the domain and silently absent over
    the rest, which looks like a cross section with a step in it rather than
    like a missing field.

    Each region holds an ``XYs1d`` or a ``regions1d`` — the schema allows no
    other functional there.
    """

    resolvedRegion: Optional[Function1d] = None
    unresolvedRegion: Optional[Function1d] = None
    fastRegion: Optional[Function1d] = None

    def __bool__(self) -> bool:
        # Present-and-empty is not absent; the reactionSuite's rule.
        return True

    def __len__(self) -> int:
        return sum(region is not None for region in self.regions.values())

    @property
    def regions(self) -> Dict[str, Optional[Function1d]]:
        """``{node name: function}``, in the order §16.1.1 declares them."""
        return {
            "resolvedRegion": self.resolvedRegion,
            "unresolvedRegion": self.unresolvedRegion,
            "fastRegion": self.fastRegion,
        }

    def __repr__(self) -> str:
        present = ", ".join(
            f"{name}={type(region).__name__}"
            for name, region in self.regions.items() if region is not None
        )
        return f"Background({present or 'no region'})"


@dataclass
class ResonancesWithBackground:
    """§16.1.1. "Reconstruct the resonances and add them to this background."

    Not a function: it is an instruction plus the background. Evaluating it
    means running a reconstructor, which is why it carries no ``evaluate``. The
    honest consequence is that a consumer meeting this form has to *choose* a
    reconstructor and say so — the thing `endf.pendf` was made explicit for.
    """

    background: Optional[Background] = None
    resonanceRegionHref: Optional[str] = None
    label: Optional[str] = None


@dataclass
class Reference:
    """§16.1.1. A link to another cross section form, possibly in another reaction."""

    href: str
    label: Optional[str] = None


@dataclass
class CoulombPlusNuclearElastic:
    """§16.1.1. Charged-particle elastic, or a reference to it."""

    href: Optional[str] = None
    label: Optional[str] = None


@dataclass
class ThermalNeutronScatteringLaw1d:
    """§16.1.1. A reference to the double-differential TNSL cross section."""

    href: Optional[str] = None
    label: Optional[str] = None


@dataclass
class URR_probabilityTables1d:  # noqa: N801 - GNDS node name, see NAMING.md
    """§16.1.1. Unresolved-region probability tables, a processed representation."""

    href: Optional[str] = None
    label: Optional[str] = None


@dataclass
class CrossSection:
    """σ(E) for one reaction, in as many representations as the file carries.

    Keyed by style label, so ``crossSection['eval']`` is the evaluated form and
    ``crossSection['recon']`` the reconstructed one, exactly as §9.1's example
    describes.
    """

    forms: Dict[str, object] = field(default_factory=dict)

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

    def __contains__(self, label: str) -> bool:
        return label in self.forms

    def __getitem__(self, label: str):
        try:
            return self.forms[label]
        except KeyError:
            raise KeyError(
                f"this crossSection has no form labelled {label!r}; "
                f"it has {sorted(self.forms)}"
            ) from None

    def __setitem__(self, label: str, form: object) -> None:
        self.forms[label] = form

    def __repr__(self) -> str:
        if not self.forms:
            return "CrossSection(no forms decoded)"
        described = ", ".join(
            f"{label}={type(form).__name__}" for label, form in sorted(self.forms.items())
        )
        return f"CrossSection({described})"

    @property
    def evaluated(self):
        """The ``label='eval'`` form §16.1.1 requires of an evaluated file."""
        return self[EVAL_LABEL]

    @property
    def hasEvaluated(self) -> bool:
        return EVAL_LABEL in self.forms

    def evaluate(self, x, label: str = EVAL_LABEL, outOfRange: str = "zero"):
        """Evaluate one named form.

        The label is explicit and defaults to ``eval`` rather than to "whatever
        is there". A container with a reconstructed form and an evaluated one
        that disagree is normal, not exceptional, so picking silently would be
        picking wrongly about half the time.
        """
        form = self[label]
        if not hasattr(form, "evaluate"):
            raise TypeError(
                f"the {label!r} form is a {type(form).__name__}, which is a "
                f"representation rather than a function; it cannot be evaluated "
                f"without choosing how to turn it into one"
            )
        return form.evaluate(x, outOfRange)

    def __repr__(self) -> str:
        pairs = ", ".join(f"{k}={type(v).__name__}" for k, v in self.forms.items())
        return f"CrossSection({pairs})"
