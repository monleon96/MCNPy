"""GNDS — reading and writing the Generalized Nuclear Database Structure.

**The package is empty at import time, and must stay that way.** Nothing here
imports :mod:`kika.nuclear_data.model` at module scope, for the same reason
:mod:`kika.endf.model_adapter` does not: everything under ``kika/`` reaches
``kika.nuclear_data`` transitively on ``import kika``, so a module-level import
of the model would wake it for the cluster pipeline, the desktop app and every
notebook at once. ``kika/nuclear_data/model/tests/test_dormancy.py`` asserts the
model stays unreachable from a plain ``import kika``, and the GNDS reader is
reached through :func:`kika.read`, which imports it inside the call.

**Which versions this reads, and why there is no compatibility layer.**
:mod:`kika.gnds.version` accepts ``2.0`` and ``2.1`` through one code path and
refuses ``1.9``/``1.10`` by name. That is not a shortcut. The 2.1 specification
(NEA/WKP(2025)6, foreword) describes itself as *"a modest update"* over 2.0
whose changes *"focus on improving handling for thermal neutron scattering law
data"* — which kika does not read at all — against **149** approved change
requests for 1.9 → 2.0. And FUDGE 6.10.0, the reference implementation, lists
2.1 among its allowed format versions while containing not one branch on it:
its only per-version renames are for 1.10. For everything kika models, 2.0 and
2.1 are the same format, so a version-dispatch layer would have zero branches
in it. If a later version does change a node kika reads, the branch goes in at
that node, with a fixture of each version beside it.

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

__all__ = []
