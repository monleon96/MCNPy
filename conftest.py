"""Repo-wide pytest configuration.

Three jobs, and nothing else:

1. **One tape resolver.** Before this file there were four hardcoded absolute
   paths and three private ``_resolve_*`` helpers that disagreed with each
   other. Everything that needs a real ENDF/ACE tape, a Serpent input or an
   NJOY binary now goes through the fixtures defined here, which look under a
   single root: ``$KIKA_TAPES`` (default ``/share_snc/snc/JuanMonleon``).

2. **Markers.** ``tape``, ``njoy``, ``gnds`` and ``slow`` are applied
   *automatically* from the fixtures a test requests, so the marker can never
   drift from what the test actually needs. Declared in ``pyproject.toml``.

3. **``--deep``.** Without it, a machine with no tapes and no NJOY runs the
   suite green with a third of it silently skipped, and reports nothing. With
   it, every ``tape``/``njoy`` skip becomes a **failure**, and the session ends
   with a list of exactly which tapes could not be resolved.

Typical use::

    pytest                                  # fast lane, skips what it lacks
    pytest -m "not tape and not njoy"       # what CI runs
    pytest --deep                           # workstation: prove nothing skipped
    KIKA_TAPES=/other/root pytest --deep
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Where tapes are looked for
# --------------------------------------------------------------------------

#: Root of the shared data tree. Everything in ``_TAPES`` is relative to one of
#: the roots returned by :func:`_search_roots`.
_DEFAULT_TAPE_ROOT = "/share_snc/snc/JuanMonleon"


def _search_roots() -> Tuple[Path, ...]:
    """Directories searched for tapes, most specific first.

    ``KIKA_ENDF_FILES`` and ``<repo>/files/endf`` are kept because several test
    modules already documented them; they are now additional roots instead of
    competing resolvers.
    """
    roots = [Path(os.environ.get("KIKA_TAPES", _DEFAULT_TAPE_ROOT))]
    env_endf = os.environ.get("KIKA_ENDF_FILES")
    if env_endf:
        roots.append(Path(env_endf))
    roots.append(REPO_ROOT / "files" / "endf")
    roots.append(REPO_ROOT / "files")
    return tuple(roots)


#: Logical tape name -> candidate paths relative to each search root, in order
#: of preference. The first candidate that exists wins.
_TAPES: Dict[str, Sequence[str]] = {
    # JEFF-4.0 Fe-56 with MF4 grafted from JEFF-3.3 — the host tape the whole
    # thesis pipeline is built on. NJOY reconstruction only reads MF1/2/3, so
    # the MF4 graft is irrelevant to the reconr tests that use it.
    "fe56_host": (
        "jeff40_with_MF4_from_jeff33/26-Fe-56g.txt",
        "26-Fe-56g.txt",
        "Fe56_jeff4.0_n.endf",
    ),
    "fe57_host": (
        "jeff40_with_MF4_from_jeff33/26-Fe-57g.txt",
        "jeff40_with_MF4_from_jeff33/Fe57_jeff4.0_n.endf",
        "Fe57_jeff4.0_n.endf",
    ),
    "fe56_jendl": (
        "JENDL-5/Fe56_jendl5_n.endf",
        "Fe56_jendl5_n.endf",
        "JENDL-5/260560.jendl5",
    ),
    "u235": ("jeff40-endf/92-U-235g.txt", "92-U-235g.txt"),
    "th232": ("jeff40-endf/90-Th-232g.txt", "90-Th-232g.txt"),
    "pu241": ("jeff40-endf/94-Pu-241g.txt", "94-Pu-241g.txt"),
    "u238": ("jeff40-endf/92-U-238g.txt", "U238_jeff4.0_n.endf"),
    "serpent_input": ("serpent/PWRSphere.sss2", "PWRSphere.sss2"),
    "fe56_ace": ("ACE_samples/26056.06c", "26056.06c"),
    # Sample covariance files for the reaction-transfer test. Not on the shared
    # tree at present; the repo-local `files/cov/` root is where they belong.
    "u5_nubar_covfil": ("cov/tape33_ENDF_U5_nubar_56", "COV/tape33_ENDF_U5_nubar_56"),
    "u5_boxer": ("cov/tape33_U5_ENDF_Scale56.boxer", "COV/tape33_U5_ENDF_Scale56.boxer"),
}

#: NJOY candidates tried when ``NJOY_EXECUTABLE`` is unset.
_NJOY_CANDIDATES = (
    Path.home() / "NJOY2016" / "build" / "njoy",
    Path("/usr/local/bin/njoy"),
    Path("/opt/njoy2016/njoy"),
)

#: Filled in as fixtures ask for things that turn out to be missing, and
#: reported once at the end of the session.
_UNRESOLVED: Dict[str, str] = {}


def resolve_tape(name: str) -> Optional[Path]:
    """Return the path of logical tape *name*, or ``None`` if not reachable."""
    try:
        candidates = _TAPES[name]
    except KeyError:  # pragma: no cover - programming error, not a data gap
        raise KeyError(
            f"Unknown tape {name!r}. Known tapes: {sorted(_TAPES)}"
        ) from None
    for root in _search_roots():
        for rel in candidates:
            path = root / rel
            if path.is_file():
                return path
    return None


def resolve_njoy() -> Optional[Path]:
    """Return a usable NJOY executable, or ``None``."""
    env = os.environ.get("NJOY_EXECUTABLE")
    if env:
        path = Path(env)
        return path if path.is_file() else None
    for candidate in _NJOY_CANDIDATES:
        if candidate.is_file():
            return candidate
    found = shutil.which("njoy")
    return Path(found) if found else None


# --------------------------------------------------------------------------
# --deep
# --------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--deep",
        action="store_true",
        default=False,
        help=(
            "Turn every 'tape' and 'njoy' skip into a failure. Use on a machine "
            "that has the shared data tree, to prove the suite really ran."
        ),
    )


def _deep(config: pytest.Config) -> bool:
    return bool(config.getoption("--deep"))


def _missing(request: pytest.FixtureRequest, what: str, detail: str):
    """Skip, or fail under ``--deep``, and record the gap for the summary."""
    _UNRESOLVED[what] = detail
    message = f"{what} not reachable: {detail}"
    if _deep(request.config):
        pytest.fail(f"--deep: {message}", pytrace=False)
    pytest.skip(message)


# --------------------------------------------------------------------------
# Automatic markers
# --------------------------------------------------------------------------

#: Fixtures whose presence means the test needs the shared data tree.
_TAPE_FIXTURES = frozenset(
    {f"{name}_tape" for name in _TAPES}
    | {"serpent_input", "fe56_ace", "endf_dir", "tape_root"}
)
#: Fixtures whose presence means the test spawns NJOY.
_NJOY_FIXTURES = frozenset({"njoy_exe"})


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Mark tests from the fixtures they request.

    Hand-applied markers drift from reality the moment a test grows a new
    dependency. Deriving them from ``fixturenames`` cannot drift.
    """
    for item in items:
        names = set(getattr(item, "fixturenames", ()))
        if names & _TAPE_FIXTURES:
            item.add_marker(pytest.mark.tape)
        if names & _NJOY_FIXTURES:
            item.add_marker(pytest.mark.njoy)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    """Catch-all for ``--deep``.

    The fixtures above fail loudly on their own, but a module-level
    ``@pytest.mark.skipif`` never reaches a fixture. This turns any skip on a
    ``tape``/``njoy``-marked test into a failure when ``--deep`` is on, so the
    guarantee holds regardless of how the skip was expressed.
    """
    outcome = yield
    report = outcome.get_result()
    if not _deep(item.config) or report.outcome != "skipped":
        return
    if not (item.get_closest_marker("tape") or item.get_closest_marker("njoy")):
        return
    reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else report.longrepr
    report.outcome = "failed"
    report.longrepr = f"--deep: test skipped but external data was required ({reason})"


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    """List what could not be resolved, so the gap is actionable."""
    if not _UNRESOLVED:
        return
    terminalreporter.section("kika: unresolved external data")
    terminalreporter.write_line(
        f"KIKA_TAPES={os.environ.get('KIKA_TAPES', _DEFAULT_TAPE_ROOT)}"
    )
    for what, detail in sorted(_UNRESOLVED.items()):
        terminalreporter.write_line(f"  {what}: {detail}")


# --------------------------------------------------------------------------
# Fixtures — external data
# --------------------------------------------------------------------------

def _tape_fixture(name: str):
    """Build a session-scoped fixture returning logical tape *name*."""

    def _fixture(request: pytest.FixtureRequest) -> Path:
        path = resolve_tape(name)
        if path is None:
            _missing(
                request,
                name,
                "tried " + ", ".join(_TAPES[name]) + f" under {_search_roots()}",
            )
        return path

    _fixture.__name__ = f"{name}_tape"
    _fixture.__doc__ = f"Path to the {name} tape (skips, or fails under --deep)."
    return pytest.fixture(scope="session", name=f"{name}_tape")(_fixture)


fe56_host_tape = _tape_fixture("fe56_host")
fe57_host_tape = _tape_fixture("fe57_host")
fe56_jendl_tape = _tape_fixture("fe56_jendl")
u235_tape = _tape_fixture("u235")
th232_tape = _tape_fixture("th232")
pu241_tape = _tape_fixture("pu241")
u238_tape = _tape_fixture("u238")
u5_nubar_covfil_tape = _tape_fixture("u5_nubar_covfil")
u5_boxer_tape = _tape_fixture("u5_boxer")


@pytest.fixture(scope="session")
def serpent_input(request: pytest.FixtureRequest) -> Path:
    """The PWRSphere Serpent input used by the material-parsing tests."""
    path = resolve_tape("serpent_input")
    if path is None:
        _missing(request, "serpent_input", "PWRSphere.sss2 not found")
    return path


@pytest.fixture(scope="session")
def fe56_ace(request: pytest.FixtureRequest) -> Path:
    """A real Fe-56 ACE file, for the ACE round-trip gate."""
    path = resolve_tape("fe56_ace")
    if path is None:
        _missing(request, "fe56_ace", "no Fe-56 ACE under ACE_samples/")
    return path


@pytest.fixture(scope="session")
def njoy_exe(request: pytest.FixtureRequest) -> Path:
    """A working NJOY executable."""
    exe = resolve_njoy()
    if exe is None:
        _missing(
            request,
            "njoy",
            "set NJOY_EXECUTABLE, or install NJOY at one of "
            + ", ".join(str(p) for p in _NJOY_CANDIDATES),
        )
    return exe


@pytest.fixture(scope="session")
def tape_root() -> Path:
    """The configured tape root itself, for tests that build their own paths."""
    return Path(os.environ.get("KIKA_TAPES", _DEFAULT_TAPE_ROOT))


# --------------------------------------------------------------------------
# Fixtures — committed micro-tapes (always available, never marked)
# --------------------------------------------------------------------------

MICRO_TAPE_DIR = REPO_ROOT / "kika" / "endf" / "tests" / "data"


@pytest.fixture(scope="session")
def micro_tape() -> Path:
    """Committed Fe-56 slice: MF1/451, MF2/151, MF3/MT2, MF4/MT2, verbatim.

    Real ENDF text cut section-by-section from the JEFF-4.0 host tape, with the
    MF1/451 directory rebuilt. No record was reformatted, so anything that
    parses the real tape parses this one. Regenerate with
    ``REGEN_MICRO_TAPES=1 pytest kika/endf/tests/test_micro_tape_regen.py``.
    """
    path = MICRO_TAPE_DIR / "micro_fe56_structural.endf"
    if not path.is_file():  # pragma: no cover - fixture is committed
        pytest.fail(f"committed micro-tape is missing: {path}")
    return path


@pytest.fixture(scope="session")
def micro_cov_tape() -> Path:
    """Committed synthetic tape carrying MF33/MT2 and MF34/MT2 on a small grid."""
    path = MICRO_TAPE_DIR / "micro_fe56_cov.endf"
    if not path.is_file():  # pragma: no cover - fixture is committed
        pytest.fail(f"committed micro-tape is missing: {path}")
    return path
