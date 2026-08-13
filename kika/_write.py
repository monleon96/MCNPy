"""``kika.write()`` — one door out, symmetric with :mod:`kika._read`.

    >>> report = kika.write(evaluation, "Fe56.gnds.xml")
    >>> report.summary()

The same shape as the door in: one function, the format chosen at the threshold,
one object type going through. What comes back is not the file — it is the
:class:`~kika.nuclear_data.model.conversion.ConversionReport`, because the file
cannot tell you what is missing from it and the report can.

**Only GNDS.** ``format="endf"`` raises, and the message says why rather than
"not implemented": kika's ENDF writer is a *patch-in-place* editor
(``kika/endf/write_endf.py`` rewrites the sections it was given inside the tape
it read) and there is no whole-file model → ENDF-6 writer anywhere in the
library. Producing one is not a missing function, it is MF1/451 headers,
sequence numbering, TAB1 pagination and the union grids ENDF requires — a
project, and one nobody has asked for while the ENDF tapes are the *input*.

**Two files, not one.** §25.1.1 makes ``covarianceSuite`` a root node in its own
right. A suite with covariances is therefore written as a pair — the evaluation,
and a sibling under ``Covariances/`` — with the ``externalFile`` entry and its
SHA-1 written to match what was actually put on disk. A ``reactionSuite`` naming
a file that does not exist is worse than one naming none, because the first is a
broken link and the second is an honest absence.

**Where this is not for.** The thesis pipeline perturbs ENDF tapes and scores
ENDF tapes; nothing here is on that path. This exists so that a model built or
modified in kika can leave in a format somebody else's code reads — phase 8's
``realization`` ensemble is the case it was built for.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

__all__ = ["write"]

#: Formats the door can write. ENDF is refused by name; see :func:`write`.
WRITE_FORMATS = ("gnds",)

#: Where a covariance sibling goes, relative to the evaluation. The layout the
#: ENDF/B-VIII.1 GNDS distribution uses, so a pair kika writes sits where a
#: reader of that distribution already looks.
COVARIANCE_SUBDIRECTORY = "Covariances"


def write(suite, path, format: str = "gnds", gnds: Optional[str] = None):
    """Write a :class:`ReactionSuite` out, and say what did not go with it.

    Parameters
    ----------
    suite
        The evaluation. Its ``covarianceSuite``, if it has one, is written
        beside it as the separate document §25.1.1 requires.
    path
        Where the ``reactionSuite`` goes. The covariance sibling is derived from
        it — same stem, under ``Covariances/``.
    format
        ``'gnds'``. ``'endf'`` raises with the reason.
    gnds
        ``'2.0'`` or ``'2.1'``, forcing the declared version. The default
        mirrors what the suite was read from, and is ``'2.0'`` for a suite with
        no GNDS origin — see :func:`kika.gnds.encode.chooseFormat`.

    Returns
    -------
    ConversionReport
        **Read it.** A partial model writes a partial file, and the file itself
        cannot say so: a product whose §18 law kika does not read comes out with
        an empty ``<distribution/>``, which looks like nothing at all. The report
        names each one, and says that such a file does not validate — which is
        deliberate, because the alternative is asserting on the evaluator's
        behalf that no distribution was given.
    """
    format = format.lower()
    if format == "endf":
        raise NotImplementedError(
            "kika cannot write a model out as an ENDF-6 tape. Its ENDF writer "
            "(kika/endf/write_endf.py) patches sections into a tape it already "
            "read; there is no whole-file model -> ENDF writer, and building one "
            "means MF1/451 headers, sequence numbering, TAB1 pagination and "
            "ENDF's union-grid rules, not a missing function. To modify a tape, "
            "read it with read_endf and write it back through that path."
        )
    if format not in WRITE_FORMATS:
        raise ValueError(
            f"format must be one of {WRITE_FORMATS}, got {format!r}"
        )
    return _writeGnds(suite, Path(os.fspath(path)), gnds)


def _writeGnds(suite, path: Path, gnds: Optional[str]):
    # Imported here, not at module scope: importing the encoder imports the
    # model, and `import kika` must not wake it. Same reason as `_read`.
    from kika.gnds.encode import (chooseFormat, serialise, sha1,
                                  writeCovarianceSuite, writeReactionSuite)
    from kika.nuclear_data.model import ConversionReport, ExternalFile

    report = ConversionReport()
    format = chooseFormat(suite, gnds)

    # Both links below are rewritten on the caller's own objects, so they are
    # put back before returning. **Writing a file must not edit the model you
    # were handed**: without this, `kika.write(suite, a)` followed by
    # `kika.write(suite, b)` leaves the suite naming b, and a user who wrote a
    # copy out for inspection finds their evaluation quietly changed.
    originalExternalFiles = list(suite.externalFiles.files)
    originalBackLinks = (list(suite.covarianceSuite.externalFiles)
                         if suite.covarianceSuite is not None else None)
    try:
        return _writeLinkedPair(suite, path, format, report,
                                writeReactionSuite, writeCovarianceSuite,
                                serialise, sha1, ExternalFile)
    finally:
        suite.externalFiles.files = originalExternalFiles
        if originalBackLinks is not None:
            suite.covarianceSuite.externalFiles = originalBackLinks


def _writeLinkedPair(suite, path, format, report, writeReactionSuite,
                     writeCovarianceSuite, serialise, sha1, ExternalFile):
    covariancePath = None
    if suite.covarianceSuite is not None:
        covariancePath = (
            path.parent / COVARIANCE_SUBDIRECTORY
            / f"{path.name.replace('.xml', '')}-covar.xml"
        )
        # **The back-link is rewritten too.** Every `rowData` href in a
        # covariance file is `$reactions#/…`, and `$reactions` resolves through
        # that file's *own* `externalFiles`. Carrying over the entry the suite
        # was read with names the file kika read, not the one it is about to
        # write, and the pair comes out with all its cross-section links dead —
        # found by reading the written pair back, not by inspection.
        #
        # No checksum: the reactionSuite names the covariance file's digest, so
        # a digest in the other direction cannot exist. The distribution's own
        # covariance files carry none for the same reason.
        suite.covarianceSuite.externalFiles = [
            entry for entry in suite.covarianceSuite.externalFiles
            if entry.label != "reactions"
        ] + [ExternalFile(label="reactions", path=f"../{path.name}")]
        tree, report = writeCovarianceSuite(suite.covarianceSuite, format, report)
        covariancePath.parent.mkdir(parents=True, exist_ok=True)
        payload = serialise(tree)
        covariancePath.write_bytes(payload)

        # The externalFile is rewritten to match what was *actually written*,
        # digest included. Carrying the digest the suite was read with would
        # name a file whose contents kika has just changed, and the next reader
        # would report a checksum failure kika caused.
        relative = f"{COVARIANCE_SUBDIRECTORY}/{covariancePath.name}"
        suite.externalFiles.files = [
            entry for entry in suite.externalFiles.files
            if entry.label != "covariances"
        ] + [ExternalFile(label="covariances", path=relative,
                          checksum=sha1(payload), algorithm="sha1")]

    tree, report = writeReactionSuite(suite, format, report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialise(tree))

    if covariancePath is not None:
        report.warn(
            f"the covariances were written as a separate document, "
            f"{covariancePath.name}, which §25.1.1 requires; the evaluation's "
            f"externalFile entry and its SHA-1 were rewritten to match it"
        )
    return report
