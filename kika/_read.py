"""``kika.read()`` — one door into the GNDS model, whatever the file is.

**The decision this file implements (2026-08-10).** A user should not have to
pick a format module, learn a format-shaped object and then learn a second one
for the next file. The format is a *door*, not a namespace to live in:

    >>> ev = kika.read("Fe56.endf")      # or an ACE file, or a GNDS XML file
    >>> ev.reactions[102].crossSection
    >>> E, xs = ev.cross_section(102)

The format is sniffed at the threshold and one object type comes out — a
:class:`~kika.nuclear_data.model.suite.ReactionSuite`, GNDS-shaped and
GNDS-named. The GNDS reader **registers a door here** rather than becoming a
third top-level ``read_*`` name; :func:`kika.gnds.decode.readReactionSuite` is
still the low road, as ``read_endf`` and ``read_ace`` are.

**The low road stays first-class.** ``read_endf`` and ``read_ace`` are not
deprecated and are not going away. Two reasons, both concrete. The model does not
yet cover MF6 or MF12-15 (phase 7) — MF6 is *parsed* but not modelled, MF12-15
not read at all, and of MF5 only LF=1 is modelled — so a user who needs those
needs the file itself; and evaluators legitimately work in ENDF's own terms — the Fe-56 chi2
work reads MF33/MF34 structure directly and should keep doing so. What this door
adds is a *default*, not a monopoly.

**Not for the pipeline.** Decoding into the model costs roughly 2.5x a bare parse
(see ``kika/nuclear_data/__init__.py``), and the full Fe-56 tape already takes
minutes to read. The cluster pipeline and the reconstruction hot path keep using
``read_endf``. Nobody should "modernise" them onto this.

**Why this module is at the top level, and why its name starts with an
underscore.** It imports both format packages and the model, so it is a
format-layer *consumer* and sits above all of them; ``kika/tests/test_layering.py``
guards ``kika/processing`` and ``kika/nuclear_data`` against importing formats,
this file is in neither, and the arrow it follows (formats -> model) is the
permitted one.

The underscore is not a style preference. Naming it ``kika/read.py`` makes the
submodule ``kika.read`` collide with the function ``kika.read``, and the import
system wins: ``from .read import read`` binds the *module* onto the package as a
side effect, so the first attribute access returns the function and every one
after it returns the module. That was observed, not theorised. The module is
private and the only public name is the function.

**It is imported lazily.** ``kika/__init__.py`` resolves ``read`` through a PEP
562 ``__getattr__``, because importing this module imports the model, and
``kika.nuclear_data.model`` must stay dormant on a plain ``import kika`` — the
cluster, the app and every notebook pay for anything that is not.
``kika/nuclear_data/model/tests/test_dormancy.py`` is the test of that fact.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple, Union

__all__ = ["read", "sniff_format", "UnknownFormatError"]

#: Formats the door can open. All three are served.
#:
#: Measured on this machine, warm cache, nothing else running: ENDF/B-VIII.1's
#: Fe-56 through this door costs **2.85 s as GNDS** (18.8 MB, and that includes
#: following the externalFile and reading its 7 covariance sections) against
#: **5.97 s as ENDF-6** (25.6 MB, MF1/2/3/4/33 — the tape's MF12 and MF14 have
#: no parser and are not read at all; its MF6 was in that list until MF6 gained
#: one, so this figure predates six MF6 sections now being read). Neither is
#: the six minutes the JEFF host tape costs; that tape is larger and carries MF34.
FORMATS = ("endf", "ace", "gnds")

#: MF numbers whose content belongs to the covarianceSuite rather than the
#: reactionSuite (GNDS §25.1.1). Kept here because the door has to know which
#: redirect notices it has already acted on. See :func:`_dropRedirectsWeActedOn`.
#:
#: MF35 joined the list when the PFNS work taught `decodeCovarianceSuite` to
#: read it, and MF31 was already here before the nu-bar work taught it the same.
#: Without the entry the door would leave the redirect notice standing with
#: nothing replacing it -- telling the user to call `decodeCovarianceSuite` for
#: a file it had, in fact, just decoded.
COVARIANCE_MF = (31, 32, 33, 34, 35)


class UnknownFormatError(ValueError):
    """The file matched none of the formats kika can open."""


# ----------------------------------------------------------------------
# Sniffing
# ----------------------------------------------------------------------

def _headLines(path, count: int = 6):
    """The first few lines, decoded leniently.

    ``errors="replace"`` on purpose: a binary file must reach the sniffer and be
    *rejected*, not raise ``UnicodeDecodeError`` from inside it, because the
    error a user needs to see is "this is not a format kika reads".
    """
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return [handle.readline().rstrip("\n") for _ in range(count)]


def _looksLikeEndf(lines) -> bool:
    """ENDF-6 records carry MAT/MF/MT in columns 67-75 of every line.

    Two matching lines are required rather than one. A single line of some other
    format can have digits in those columns by coincidence; two consecutive ones
    with MF and MT both in range is not a coincidence.
    """
    matched = 0
    for line in lines:
        if len(line) < 75:
            continue
        try:
            int(line[66:70])            # MAT
            mf = int(line[70:72])
            mt = int(line[72:75])
        except ValueError:
            continue
        if 0 <= mf <= 99 and 0 <= mt <= 999:
            matched += 1
    return matched >= 2


def _looksLikeAce(lines) -> bool:
    """A type-1 ACE header is ``ZAID AWR TEMP DATE``, e.g. ``26056.02c 55.45 ...``.

    **Known limitation, stated rather than guessed at:** the ACE 2.0 header
    begins with a version token instead, and kika has no 2.0 file to test
    against. Such a file falls through to :class:`UnknownFormatError`, whose
    message says what was tried — which is the honest outcome. Pass
    ``format="ace"`` to force it.
    """
    if not lines or not lines[0].strip():
        return False
    fields = lines[0].split()
    if len(fields) < 3:
        return False
    zaid = fields[0]
    if "." not in zaid:
        return False
    za, _, suffix = zaid.partition(".")
    if not za.isdigit() or not suffix[:-1].isdigit() or not suffix[-1:].isalpha():
        return False
    try:
        float(fields[1])                # atomic weight ratio
        float(fields[2])                # temperature, MeV
    except ValueError:
        return False
    return True


def _looksLikeGnds(lines) -> bool:
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        return stripped.startswith("<?xml") or stripped.startswith("<reactionSuite")
    return False


def sniff_format(path) -> str:
    """Which of :data:`FORMATS` this file is, decided on **content**.

    Extensions are not consulted. In this field they are unreliable to the point
    of being misleading — ``.dat``, ``.txt``, ``.endf``, no extension at all, and
    ACE files named after their ZAID are all routine.

    Raises :class:`UnknownFormatError` naming what was tried, so the message is
    actionable rather than "could not read file".
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"no such file: {path}")
    lines = _headLines(path)
    if _looksLikeGnds(lines):
        return "gnds"
    if _looksLikeAce(lines):
        return "ace"
    if _looksLikeEndf(lines):
        return "endf"
    raise UnknownFormatError(
        f"{path} is not an ENDF tape (no MAT/MF/MT in columns 67-75), not a "
        f"type-1 ACE file (first line is not 'ZAID AWR TEMP DATE') and not GNDS "
        f"XML. If you know what it is, pass format= explicitly."
    )


# ----------------------------------------------------------------------
# The door
# ----------------------------------------------------------------------

def read(path, format: Optional[str] = None, covariances: bool = True):
    """Read any supported file into a :class:`ReactionSuite`.

    Parameters
    ----------
    path
        The file. Its format is detected from content unless ``format`` is given.
    format
        ``'endf'``, ``'ace'`` or ``'gnds'``, forcing the choice. For the file
        whose header is malformed, or an ACE 2.0 file the sniffer cannot place.
    covariances
        When true, the covariances are decoded onto ``suite.covarianceSuite``.
        For ENDF that means MF31/33/34 off the same tape; for GNDS it means
        **following the ``externalFile`` link to the sibling file and reading
        it**, which is a second file on disk and may not be there — its absence
        is a report entry, never an error. Ignored for ACE, which carries none.

        GNDS §25.1.1 makes the covariance suite a root node in its own right;
        kika hangs it off the evaluation for convenience and the writer emits it
        separately.

    Returns
    -------
    ReactionSuite
        With ``suite.report`` set — what the decode lost, approximated or could
        not represent. Check it. A partial decode is normal today (MF6 is
        parsed but not modelled, MF12-15 not read at all, and of MF5 only its
        tabulated LF=1 reaches the model) and the report is the only thing that
        says so.
    """
    if format is None:
        format = sniff_format(path)
    format = format.lower()
    if format not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}, got {format!r}")

    if format == "gnds":
        return _readGnds(path, covariances=covariances)
    if format == "ace":
        return _readAce(path)
    return _readEndf(path, covariances=covariances)


def _readEndf(path, covariances: bool):
    # Imported here, not at module scope: this is what wakes the model, and it
    # must not happen until somebody actually calls read().
    from kika.endf.model_adapter.covariances import decodeCovarianceSuite
    from kika.endf.model_adapter.decode import decodeReactionSuite
    from kika.endf.read_endf import read_endf

    # No mf_numbers filter. Restricting the parse would be faster, but
    # decodeReactionSuite declares "unsupported" by looking at which MFs are
    # *present* in the parsed object — so filtering the parse would quietly
    # shrink the report, and the report is the reason this door is trustworthy.
    endf = read_endf(str(path))
    suite, report = decodeReactionSuite(endf)

    decoded = ()
    if covariances:
        present = tuple(mf for mf in COVARIANCE_MF if mf in getattr(endf, "mf", {}))
        if present:
            suite.covarianceSuite, report = decodeCovarianceSuite(
                endf, report, evaluation=suite.evaluation,
                target=suite.target,
            )
            decoded = present

    _noteUnparsedMFs(path, endf, report)
    _dropRedirectsWeActedOn(report, decoded)
    suite.report = report
    return suite


def _readGnds(path, covariances: bool):
    """The GNDS door. One file in, and optionally the sibling it names.

    **The covariance suite is a second file**, which is the one way this door
    differs from the ENDF one. §14.1.1's ``externalFiles`` gives its relative
    path and its SHA-1; the reader resolves the path against *this* file's
    directory and checks the digest. Every failure along the way — the sibling
    missing, the digest not matching, the file turning out not to be a
    ``covarianceSuite`` — is an entry in the report and not an exception,
    because a user handed one half of a pair is in a normal situation and their
    cross sections are still perfectly readable.

    The two documents are handed to each other: the covariance reader gets this
    file back under the label *its own* ``externalFiles`` uses for it, so the
    ``$reactions#…`` hrefs on every ``rowData`` resolve. Without that, reading
    the pair together would produce exactly the report a covariance file read
    alone produces, and the "which cross section is this about" link would be
    lost with both files in hand.
    """
    from kika.gnds.covariances import readCovarianceSuite
    from kika.gnds.decode import readReactionSuite
    from kika.gnds.xpath import Document, readExternalFiles

    document = Document.parse(path)
    suite, report = readReactionSuite(document)
    if covariances:
        _attachGndsCovariances(document, suite, report,
                               readCovarianceSuite, Document, readExternalFiles)
    suite.report = report
    return suite


def _attachGndsCovariances(document, suite, report,
                           readCovarianceSuite, Document, readExternalFiles):
    for entry in suite.externalFiles:
        resolved = None
        if document.path is not None:
            resolved = (document.path.parent / entry.path).resolve()
        if resolved is None or not resolved.is_file():
            report.lost(
                f"externalFile {entry.label!r} names {entry.path}, which is not "
                f"beside {getattr(document.path, 'name', 'this file')}; its "
                f"covariances were not read"
            )
            continue

        sibling = Document.parse(resolved)
        if sibling.root.tag != "covarianceSuite":
            report.warn(
                f"externalFile {entry.label!r} is a <{sibling.root.tag}>, not a "
                f"<covarianceSuite>; it was not read"
            )
            continue

        # The label the *sibling* uses for this file, so its own hrefs resolve.
        back = {
            other.label: document
            for other in readExternalFiles(sibling.root)
            if document.path is not None
            and (resolved.parent / other.path).resolve() == document.path.resolve()
        }
        suite.covarianceSuite, _ = readCovarianceSuite(sibling, back, report)
        return


def _readAce(path):
    from kika.ace.model_adapter.decode import decodeAce
    from kika.ace.parsers.parse_ace import read_ace

    ace = read_ace(str(path))
    suite, report = decodeAce(ace)
    suite.report = report
    return suite


def _noteUnparsedMFs(path, endf, report) -> None:
    """Record MFs the *tape* carries that no parser touched.

    Without this the report is silent about them, and silence reads as "nothing
    was lost". ``endf.mf`` holds only the MFs that had a registered parser —
    ``parse_endf.py`` computes ``skipped_mfs`` and drops it into a log line
    nobody reads — so the question "did this tape have an MF12?" cannot be
    answered from the parsed object at all. The tape is rescanned here with the
    same helper the parser uses, so the two can never disagree.
    """
    try:
        from kika.endf.parsers.parse_endf import _scan_available_mf
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            onTape = set(_scan_available_mf(handle.readlines()))
    except Exception as exc:                                  # noqa: BLE001
        report.warn(f"could not rescan {path} for unparsed MF sections: {exc}")
        return
    for mf in sorted(onTape - set(getattr(endf, "mf", {}))):
        report.unsupportedNode(
            f"MF{mf} is on the tape and kika has no parser for it; it was never "
            f"read, so nothing in this suite reflects it"
        )


def _dropRedirectsWeActedOn(report, decodedCovarianceMFs) -> None:
    """Remove the "call decodeCovarianceSuite for it" notices we then obeyed.

    ``decodeReactionSuite`` files each covariance MF under ``unsupported`` with a
    note saying it belongs to the covarianceSuite instead (see
    ``kika/endf/model_adapter/decode.py``, the ``present & SUPPORTED_MF`` loop).
    That is correct for a bare ``decodeReactionSuite`` call. It is *false* by the
    time this door returns, because the door has decoded them and hung the result
    on ``suite.covarianceSuite``.

    Leaving them would make ``report.isClean`` false — and the repr shout
    "3 unsupported" — for every ordinary covariance tape, which is precisely how
    a report trains its reader to ignore it. Only the notices for MFs actually
    decoded are dropped; a tape read with ``covariances=False`` keeps all of them.
    """
    if not decodedCovarianceMFs:
        return
    prefixes = tuple(f"MF{mf} is present and parsed;" for mf in decodedCovarianceMFs)
    report.unsupported[:] = [
        entry for entry in report.unsupported if not entry.startswith(prefixes)
    ]
