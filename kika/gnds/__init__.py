"""GNDS — reading and writing the Generalized Nuclear Database Structure.

**The model must stay asleep when this package is imported, and everything
here is arranged around that.** Nothing under ``kika/gnds`` imports
:mod:`kika.nuclear_data.model` at module scope, for the same reason
:mod:`kika.endf.model_adapter` does not: everything under ``kika/`` reaches
``kika.nuclear_data`` transitively on ``import kika``, so a module-level import
of the model would wake it for the cluster pipeline, the desktop app and every
notebook at once. ``kika/nuclear_data/model/tests/test_dormancy.py`` asserts the
model stays unreachable from a plain ``import kika``, and the GNDS reader is
reached through :func:`kika.read`, which imports it inside the call.

This module used to say *"the package is empty at import time, and must stay
that way"*, and that was the invariant stated one size too large. What must
stay true is the **dormancy**, not the emptiness: :func:`capabilities` is
exported below and imported eagerly, and it costs nothing because
:mod:`kika.gnds._capabilities` is a table of strings that imports nothing but
the standard library. ``import kika.gnds`` still loads zero model modules;
``import kika.gnds.nodes`` loads thirty, which is why the capability table is
written by hand rather than derived from :data:`kika.gnds.nodes.NODES`, and why
the join between the two lives in ``tests/test_capabilities.py``.

**What kika supports, as something a user can ask.** :func:`capabilities`
answers it for all 300 nodes ``gnds.xsd`` and ``covariances.xsd`` declare, plus
the twelve names kika uses that they do not::

    >>> import kika.gnds
    >>> print(kika.gnds.capabilities().summary())
    >>> print(kika.gnds.capabilities(coverage="unsupported").text())

It says what the *library* can lose, without opening a file;
``suite.report`` says what *your* file lost. Neither stands in for the other.

**Which versions this reads, and why there is no compatibility layer.**
:mod:`kika.gnds.version` accepts ``2.0`` and ``2.1`` through one code path and
refuses ``1.9``/``1.10`` by name. That is not a shortcut. The 2.1 specification
(NEA/WKP(2025)6, foreword) describes itself as *"a modest update"* over 2.0
whose changes *"focus on improving handling for thermal neutron scattering law
data"* — which kika does not read at all — against **149** approved change
requests for 1.9 → 2.0. And FUDGE 6.10.0 — the reference implementation, and
the copy installed here at ``/soft_snc/FUDGE/6.10.0/`` — lists 2.1 among its
allowed format versions while containing not one branch on it. **Measured
2026-08-24** over that tree: ``fudge/GNDS_formatVersion.py`` declares
``allowed = ('1.10', '2.0', '2.1')`` with ``default = '2.0'``, and outside that
module ``2.1`` appears **0** times against **61** for ``1.10`` — every
per-version branch it has is a 1.10 rename. For everything kika models, 2.0 and
2.1 are the same format, so a version-dispatch layer would have zero branches
in it. If a later version does change a node kika reads, the branch goes in at
that node, with a fixture of each version beside it.

That measurement is of **6.10.0, the version installed here**, and it is the
whole point of naming it: FUDGE's ``master`` has since moved on — an external
architecture review of 2026-08-24 reports it declaring ``1.10``, ``2.0``,
``2.1`` and ``2.2.rc1``, and defaulting to 2.1 rather than 2.0 when writing
(not verified here; the installed copy is 6.10.0). That does not reopen the
decision, because the specification's own foreword is the other half of the
argument and does not move. It does mean this paragraph must be re-measured
rather than trusted the day the installed FUDGE changes.

kika's model is built to 2.1; the published libraries are written in 2.0. Both
are read here, and the writer emits whichever version the suite came in as —
2.0 when it came from ENDF and has no GNDS origin to preserve.

XML only. §2.4 admits JSON and HDF5 as alternative meta-languages; nothing
publishes in them and neither is implemented.

**What is in here**, none of it exported from this module — import the module
you want, or go through :func:`kika.read` / :func:`kika.write`:

======================  ===================================================
:mod:`~kika.gnds.version`             the format gate, ``ACCEPTED = ("2.0", "2.1")``
:mod:`~kika.gnds.xpath`               ``href`` resolution, both dialects, and
                                      ``externalFile`` checksums
:mod:`~kika.gnds.primitives`          §5-6 ``values``/``axes``/the functionals
:mod:`~kika.gnds.styles`              §9, shared by both roots
:mod:`~kika.gnds.decode`              §14 ``reactionSuite`` → the model
:mod:`~kika.gnds.distributions`       §18, split out when phase 7b started
                                      filling it — the reader spanned five
                                      specification chapters and this is the
                                      one that grows
:mod:`~kika.gnds.resonances`          §19, split out because it is not shaped
                                      like the rest of a ``reactionSuite``
:mod:`~kika.gnds.covariances`         §25 ``covarianceSuite`` → the model
:mod:`~kika.gnds.encode`              the model → XML, both roots
:mod:`~kika.gnds.encode_resonances`   §19, the other direction
======================  ===================================================

**No intermediate format-classes layer**, unlike :mod:`kika.endf`'s
parsers → classes → adapter. GNDS's tree *is* the model's shape, so this goes
XML → model directly, and the thing that would have been the classes layer —
"what the file said, in the file's own terms" — is the ``ConversionReport`` every
reader and writer returns.
"""
from __future__ import annotations

from ._capabilities import (CAPABILITIES, Capabilities, Capability, Coverage,
                            NOT_IN_SCHEMA, capabilities)

#: The private module name is load-bearing, not a style choice. ``importlib``
#: binds a submodule onto its package as a side effect of importing it, so a
#: module named ``capabilities.py`` would overwrite the function above with
#: itself on first access — the first call returns the function and every later
#: one returns the module. ``kika/__init__.py:40-43`` documents the same trap.
__all__ = ["capabilities", "Capabilities", "Capability", "Coverage",
           "CAPABILITIES", "NOT_IN_SCHEMA"]
