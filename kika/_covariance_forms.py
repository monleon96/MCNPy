"""What a §25.2 covariance form has to be before a consumer can compute on it.

A GNDS covariance section does **not** have to hold one matrix. §25.2 lets it
state its covariance as a ``mixed`` — several forms, on different grids, that
together make the covariance — or as a ``sum``, which points at other sections
instead of carrying numbers of its own. ENDF's MF33 says the same thing with NI
sub-subsections, and kika's ENDF adapter merges them onto the union of their
grids before the model ever sees them, so the whole library was written against
the merged shape and only meets the un-merged one on the GNDS path.

**What that costs, measured rather than feared.** Fe-56's MT1, MT2 and MT102
are each a ``mixed`` of a 3×3 and a 628×628 on the GNDS path and one 630×630 on
the ENDF path (``docs/library/gnds_endf_conflicts.md`` §2.2, §7.4). Code that reaches
for ``form.matrix`` on the first of those gets an ``AttributeError``, which is
loud and fine. Code that reached for ``form.components[0]`` would get a 3×3
where it wanted a 628×628 — a plausible-looking wrong answer — and code that
*skips* what it cannot read drops the section without a word. Both of those
have existed here.

**Why this is a raise and not a merge.** Adding the components up is a real
operation kika will need and a decision nobody has taken: one component may be
a ``shortRangeSelfScalingVariance``, whose magnitude depends on the processing
group width and which therefore cannot be added onto a fixed grid at all
(``docs/library/gnds_endf_conflicts.md`` §2.2, still open). Until that is decided, the
honest answer to "give me this section's matrix" is that the section does not
state one.

**Why a private top-level module.** The raise belongs on the consumer side, not
in the reader — the reader is lenient and reports (§11.5) — and the consumers
are in two packages that do not import each other: ``kika/cov`` and
``kika/sampling``. Same reasoning, and the same precedent, as
``kika/_records.py``: put the shared thing where both callers reach it without
one of them importing the other. Nothing here imports the model, so
``kika/sampling/model_blocks.py`` keeps the property its docstring claims.
"""
from __future__ import annotations

__all__ = ["require_single_matrix", "explain_missing_matrix"]

#: Where the deferred decision is written down, quoted in every message below
#: so that a caller who meets one has somewhere to go.
_REFERENCE = "docs/library/gnds_endf_conflicts.md §2.2"


def _node_name(form: object) -> str:
    """The GNDS tag a model class is written as. ``Mixed`` → ``mixed``."""
    name = type(form).__name__
    return name[:1].lower() + name[1:]


def explain_missing_matrix(form: object, where: str) -> str:
    """The sentence to raise with, naming what was found instead of a matrix.

    Duck-typed on purpose: ``components`` and ``summands`` are asked for by
    ``hasattr``, not by ``isinstance``, so this module stays free of the model
    and ``kika.sampling`` stays free of it too.
    """
    if form is None:
        return (
            f"{where} carries no covariance form at all, so there is no matrix "
            f"to read. A section with no form is what a decode that could not "
            f"read the section leaves behind — check the suite's report before "
            f"treating this as a covariance of zero."
        )

    tag = _node_name(form)
    components = getattr(form, "components", None)
    summands = getattr(form, "summands", None)

    if components is not None:
        return (
            f"{where} states its covariance as a <{tag}> of {len(components)} "
            f"components, not as one matrix. GNDS §25.2 lets a section give "
            f"several forms on different grids, and kika does not add them up: "
            f"one of them may be a shortRangeSelfScalingVariance, whose "
            f"magnitude depends on the processing group width, so summing them "
            f"onto a fixed grid would state a number the file does not "
            f"({_REFERENCE}). The ENDF path merges them before the model sees "
            f"them and the GNDS path does not, which is why the same evaluation "
            f"reaches here in two shapes. Reach for `form.components` and "
            f"decide which of them your calculation means — taking the first "
            f"is not it, it is the 3×3 short-range block on Fe-56."
        )

    if summands is not None:
        return (
            f"{where} states its covariance as a <{tag}> of {len(summands)} "
            f"summands, which name other sections rather than carrying a "
            f"matrix. Resolving them is the caller's: follow "
            f"`form.summands[i].href` into the suite and combine with "
            f"`.coefficient` ({_REFERENCE})."
        )

    return (
        f"{where} holds a <{tag}>, which carries no matrix. kika reads it and "
        f"the model keeps it, but this calculation needs one gridded matrix per "
        f"section and cannot make one out of it ({_REFERENCE})."
    )


def require_single_matrix(form: object, where: str) -> object:
    """*form* back if it carries a matrix; a ``ValueError`` saying why not.

    ``ValueError`` and not ``NotImplementedError``: the section is well formed
    and kika read it correctly. What is missing is a decision about what its
    components mean together, and that is not a gap in the code the caller can
    wait to be filled.
    """
    if form is None or getattr(form, "matrix", None) is None:
        raise ValueError(explain_missing_matrix(form, where))
    return form
