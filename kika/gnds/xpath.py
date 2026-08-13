"""GNDS ``href`` resolution — §4.2's xPath links, in the two dialects real files use.

A GNDS evaluation is not one tree. A ``covarianceSection`` says what it is about
by pointing at a cross section in a *different file*; a ``resonancesWithBackground``
points at the ``resonances`` node in its own; a covariance's column grid points
at its row grid to avoid writing the same 124 numbers twice. All three are
``href`` attributes, and they come in two shapes — measured across the 829 files
of the ENDF/B-VIII.1 GNDS distribution, not taken from the specification:

``$reactions#/reactionSuite/reactions/reaction[@label='n + Fe56']/crossSection``
    The ``$label`` names an entry in the referring file's ``externalFiles``, and
    what follows ``#`` is an absolute path in *that* document.

``/reactionSuite/resonances`` and ``../../grid[@index='2']/values``
    Absolute and relative paths inside the current document.

**Why this is not ``ElementTree.find``.** Two reasons, both load-bearing.
``find`` has no parent axis worth the name — an ``Element`` holds no pointer to
its parent — so ``../..`` cannot be walked without a parent map built for the
document. And ``find``'s predicate parser is not quote-aware, while GNDS labels
routinely are not identifiers: this evaluation's capture channel is literally
``Fe57 + photon [inclusive]``, so the href ends
``[@label='Fe57 + photon [inclusive]']`` and any bracket matching that does not
respect the quotes truncates the label at the wrong ``]``. That is not a corner
case invented here; it is the first href in Fe-56.

**Failures are reported, not raised.** A ``covarianceSuite`` read on its own —
which is how four of the committed fixtures exist, and how a user who was handed
one file will read it — has ``$reactions#`` hrefs pointing at a document that is
simply not present. That is a normal, expected state, so :meth:`Resolver.resolve`
returns a :class:`Resolution` whose ``problem`` says what could not be reached,
and the decoders file it in the ``ConversionReport``. Raising would make the
common case an error.
"""
from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

__all__ = [
    "ExternalFile", "Document", "Resolution", "Resolver",
    "readExternalFiles", "verifyChecksum", "splitSteps", "parseStep",
]

#: §3.4.3. The two digest algorithms GNDS names for an ``externalFile``.
CHECKSUM_ALGORITHMS = {"md5": hashlib.md5, "sha1": hashlib.sha1}


# ---------------------------------------------------------------------------
# Path syntax
# ---------------------------------------------------------------------------

def splitSteps(path: str) -> List[str]:
    """Split an xPath on ``/``, ignoring separators inside predicates and quotes.

    A plain ``path.split("/")`` is wrong for the same reason a regex is: a
    quoted predicate value may contain any character, ``/`` and ``]`` included.
    """
    steps: List[str] = []
    current: List[str] = []
    depth = 0
    quote: Optional[str] = None

    for character in path:
        if quote is not None:
            current.append(character)
            if character == quote:
                quote = None
            continue
        if character in "'\"":
            quote = character
            current.append(character)
            continue
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
        elif character == "/" and depth == 0:
            steps.append("".join(current))
            current = []
            continue
        current.append(character)

    steps.append("".join(current))
    if quote is not None:
        raise ValueError(f"unterminated quote in xPath {path!r}")
    if depth:
        raise ValueError(f"unbalanced predicate brackets in xPath {path!r}")
    return steps


def parseStep(step: str) -> Tuple[str, Dict[str, str]]:
    """One step → ``(tag, {attribute: value})``.

    Only attribute predicates are supported, because only attribute predicates
    occur: across the whole distribution every predicate is ``[@label='...']``,
    ``[@index='...']`` or the like. A positional predicate raises rather than
    being silently ignored, which is the difference between "kika cannot follow
    this link" and "kika followed the wrong element".
    """
    tag = step
    predicates: Dict[str, str] = {}

    opening = _firstUnquoted(step, "[")
    if opening is not None:
        tag = step[:opening]
        for raw in _predicates(step[opening:]):
            if not raw.startswith("@"):
                raise ValueError(
                    f"xPath step {step!r} uses a predicate kika does not "
                    f"implement ({raw!r}); only attribute predicates such as "
                    f"[@label='...'] occur in GNDS files and only those are read"
                )
            name, _, value = raw[1:].partition("=")
            predicates[name.strip()] = value.strip().strip("'\"")
    return tag, predicates


def _firstUnquoted(text: str, target: str) -> Optional[int]:
    quote: Optional[str] = None
    for index, character in enumerate(text):
        if quote is not None:
            if character == quote:
                quote = None
        elif character in "'\"":
            quote = character
        elif character == target:
            return index
    return None


def _predicates(text: str) -> List[str]:
    """``"[@a='1'][@b='2']"`` → ``["@a='1'", "@b='2'"]``, quote-aware."""
    out: List[str] = []
    current: List[str] = []
    depth = 0
    quote: Optional[str] = None
    for character in text:
        if quote is not None:
            current.append(character)
            if character == quote:
                quote = None
            continue
        if character in "'\"":
            quote = character
            current.append(character)
            continue
        if character == "[":
            depth += 1
            if depth == 1:
                current = []
                continue
        elif character == "]":
            depth -= 1
            if depth == 0:
                out.append("".join(current))
                continue
        current.append(character)
    return out


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """One parsed GNDS file, plus the parent map its relative hrefs need.

    The parent map is built once, lazily, and only for a document something
    actually walks upward in. Fe-56 is 1.1 million elements; building a parent
    map for a file nobody dereferences into would be pure cost.
    """

    root: ET.Element
    path: Optional[Path] = None
    _parents: Optional[Dict[ET.Element, ET.Element]] = field(
        default=None, repr=False, compare=False
    )

    @classmethod
    def parse(cls, path) -> "Document":
        path = Path(path)
        return cls(root=ET.parse(path).getroot(), path=path)

    @property
    def parents(self) -> Dict[ET.Element, ET.Element]:
        if self._parents is None:
            self._parents = {
                child: parent for parent in self.root.iter() for child in parent
            }
        return self._parents

    @property
    def name(self) -> str:
        return self.path.name if self.path is not None else f"<{self.root.tag}>"


@dataclass
class ExternalFile:
    """§14.1.1 ``externalFile``: a label, a relative path and usually a digest."""

    label: str
    path: str
    checksum: Optional[str] = None
    algorithm: Optional[str] = None

    def resolvedAgainst(self, referrer: Optional[Path]) -> Optional[Path]:
        """The absolute path, taken relative to the file that named it."""
        if referrer is None:
            return None
        return Path(os.path.normpath(referrer.parent / self.path))


def readExternalFiles(root: ET.Element) -> List[ExternalFile]:
    """The ``externalFiles`` entries of a ``reactionSuite`` or ``covarianceSuite``."""
    container = root.find("externalFiles")
    if container is None:
        return []
    return [
        ExternalFile(
            label=entry.attrib.get("label", ""),
            path=entry.attrib.get("path", ""),
            checksum=entry.attrib.get("checksum"),
            algorithm=entry.attrib.get("algorithm"),
        )
        for entry in container.findall("externalFile")
    ]


def verifyChecksum(entry: ExternalFile, path: Path) -> Optional[str]:
    """``None`` when the file matches its declared digest, else why it does not.

    §3.4.3 allows ``md5`` and ``sha1``. An entry with no checksum is not a
    failure — the distribution's covariance files point back at their
    ``reactionSuite`` with no digest at all — so it returns ``None`` too, and the
    caller that wants to distinguish "verified" from "nothing to verify" asks
    ``entry.checksum``.
    """
    if not entry.checksum:
        return None
    algorithm = (entry.algorithm or "sha1").lower()
    if algorithm not in CHECKSUM_ALGORITHMS:
        return (
            f"externalFile {entry.label!r} declares algorithm {entry.algorithm!r}; "
            f"§3.4.3 allows {' and '.join(sorted(CHECKSUM_ALGORITHMS))}, so the "
            f"digest was not checked"
        )
    try:
        digest = CHECKSUM_ALGORITHMS[algorithm](path.read_bytes()).hexdigest()
    except OSError as exc:
        return f"externalFile {entry.label!r} could not be read for checksumming: {exc}"
    if digest != entry.checksum:
        return (
            f"externalFile {entry.label!r} ({path.name}) has {algorithm} "
            f"{digest}, but the referring file declares {entry.checksum}. The "
            f"two were not written together; one of them has been edited."
        )
    return None


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

@dataclass
class Resolution:
    """What an ``href`` reached, or why it reached nothing."""

    element: Optional[ET.Element] = None
    document: Optional[Document] = None
    problem: Optional[str] = None

    def __bool__(self) -> bool:
        return self.element is not None


class Resolver:
    """Resolves hrefs within one document and into the documents it names.

    ``documents`` maps an ``externalFile`` label onto a parsed
    :class:`Document`. A label with no entry is not an error at construction —
    it becomes a ``problem`` on the resolution that needs it, so reading a
    covariance file whose ``reactionSuite`` sibling was not supplied reports the
    links it could not follow instead of refusing to read at all.
    """

    def __init__(self, primary: Document,
                 documents: Optional[Dict[str, Document]] = None) -> None:
        self.primary = primary
        self.documents: Dict[str, Document] = dict(documents or {})

    # -- the door ---------------------------------------------------------

    def resolve(self, href: str,
                context: Optional[ET.Element] = None) -> Resolution:
        """Follow one ``href``. Never raises for a link it cannot follow."""
        if not href:
            return Resolution(problem="empty href")

        document = self.primary
        path = href
        if href.startswith("$"):
            label, separator, path = href[1:].partition("#")
            if not separator:
                return Resolution(problem=(
                    f"href {href!r} names an external file but has no '#' to "
                    f"separate the label from the path within it"
                ))
            if label not in self.documents:
                known = ", ".join(sorted(self.documents)) or "none"
                return Resolution(problem=(
                    f"href {href!r} points into the external file labelled "
                    f"{label!r}, which was not supplied (available: {known})"
                ))
            document = self.documents[label]
            context = None            # an absolute path in the other document

        try:
            steps = splitSteps(path)
        except ValueError as exc:
            return Resolution(problem=f"href {href!r} is malformed: {exc}")

        if steps and steps[0] == "":
            # A leading '/' makes the path absolute; the first named step is
            # then the root's own tag, which is matched rather than descended.
            steps = steps[1:]
            if not steps:
                return Resolution(element=document.root, document=document)
            if steps[0] != document.root.tag:
                return Resolution(problem=(
                    f"href {href!r} starts at /{steps[0]}, but "
                    f"{document.name} is a <{document.root.tag}>"
                ))
            current = document.root
            steps = steps[1:]
        elif context is None:
            return Resolution(problem=(
                f"href {href!r} is relative and no context element was given"
            ))
        else:
            current = context

        return self._walk(current, steps, document, href)

    # -- walking ----------------------------------------------------------

    def _walk(self, current: ET.Element, steps: List[str],
              document: Document, href: str) -> Resolution:
        for position, step in enumerate(steps):
            if step in ("", "."):
                continue
            if step == "..":
                parent = document.parents.get(current)
                if parent is None:
                    return Resolution(problem=(
                        f"href {href!r} walks above the root of {document.name}"
                    ))
                current = parent
                continue
            try:
                tag, predicates = parseStep(step)
            except ValueError as exc:
                return Resolution(problem=f"href {href!r}: {exc}")

            match = self._child(current, tag, predicates)
            if match is None:
                reached = "/".join(steps[:position]) or current.tag
                return Resolution(problem=(
                    f"href {href!r} has no {step!r} under <{current.tag}> in "
                    f"{document.name} (reached {reached})"
                ))
            current = match
        return Resolution(element=current, document=document)

    @staticmethod
    def _child(parent: ET.Element, tag: str,
               predicates: Dict[str, str]) -> Optional[ET.Element]:
        """The first child matching ``tag`` and every predicate.

        First rather than only: GNDS intends these links to be unique, and
        across the distribution they are. Nothing here enforces uniqueness,
        because enforcing it would mean scanning every sibling on every step of
        every link for a defect no file in circulation has.
        """
        for child in parent:
            if child.tag != tag:
                continue
            if all(child.attrib.get(name) == value
                   for name, value in predicates.items()):
                return child
        return None
