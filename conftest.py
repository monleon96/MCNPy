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

A bare ``pytest`` also holds back the ``tape``/``njoy``/``slow``/``perf``
tests — the ones that read the shared data tree, shell out, or measure wall
clock, and nearly all of the wall clock between them. Say ``-m``, ``-k`` or
``--deep`` and you get exactly what you asked for instead. **A bare run is
therefore not a proof that nothing broke on a real tape** — that is what
``--deep`` is for, and it is the one to run before pushing anything that
touches a parser or a writer.

Typical use::

    pytest                                  # fast lane: no tapes, no NJOY, no slow
    pytest -m tape                          # the 113 that need the shared tree
    pytest -m njoy                          # the ones that spawn NJOY
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

#: The shared *library* tree, one level up from the personal one: whole
#: evaluated libraries as distributed, in per-library directories
#: (``endfb81/``, ``jeff40/``, ``jendl5/`` …). The personal root holds working
#: copies and grafts; this one holds the pristine originals, and the PFNS work
#: is the first to need a tape that exists only here (ENDF/B-VIII.1 has no
#: counterpart under ``JuanMonleon/``). Kept as a second root rather than
#: reached by ``../../lib/endf`` from the first, which would break the moment
#: ``KIKA_TAPES`` points somewhere else.
_DEFAULT_LIB_ROOT = "/share_snc/lib/endf"


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
    roots.append(Path(os.environ.get("KIKA_LIB_TAPES", _DEFAULT_LIB_ROOT)))
    roots.append(_downloadCacheRoot())
    roots.append(REPO_ROOT / "files" / "endf")
    roots.append(REPO_ROOT / "files")
    return tuple(roots)


def _downloadCacheRoot() -> Path:
    """``kika.endf.remote``'s own IAEA download cache, as a search root.

    The MF32 work is the first to need tapes that exist on no shared tree here
    — every nuclide carrying a resonance-parameter covariance came down through
    ``kika.endf.remote``. Reading the cache root from the module that owns it
    keeps ``KIKA_ENDF_CACHE_DIR`` working for the tests as well as for the
    downloader; an import failure just drops the root, since it is one of
    several and a missing tape already skips.
    """
    try:
        from kika.endf.remote.constants import get_cache_dir
        return Path(get_cache_dir())
    except Exception:  # pragma: no cover - the other roots still apply
        return Path.home() / ".kika" / "endf_cache"


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
    # PFNS: the two U-235 evaluations the MF5/MF35 work is gated against, plus
    # Cf-252 as the cheap one. ``u235`` above is the same JEFF-4.0 material
    # reached through the personal root; ``u235_b81`` is a different evaluation
    # and both are needed, because the padding divergence and the outgoing-grid
    # mismatch only show up on ENDF/B-VIII.1.
    "u235_b81": ("endfb81/n-092_U_235.endf", "n-092_U_235.endf"),
    "cf252_b81": ("endfb81/n-098_Cf_252.endf", "n-098_Cf_252.endf"),
    "pu239_b81": ("endfb81/n-094_Pu_239.endf", "n-094_Pu_239.endf"),
    "th232": ("jeff40-endf/90-Th-232g.txt", "90-Th-232g.txt"),
    "pu241": ("jeff40-endf/94-Pu-241g.txt", "94-Pu-241g.txt"),
    "u238": ("jeff40-endf/92-U-238g.txt", "U238_jeff4.0_n.endf"),
    # MF32, the resonance-parameter covariances. No Fe-56 evaluation in any
    # library carries one, so this work is gated on other nuclides entirely,
    # chosen to cover one sub-format each — see ``docs/mf32-notes.md`` in
    # kika-workspace for the survey these came from. The second candidate of
    # each pair is the ``kika.endf.remote`` cache layout, which is where they
    # actually are on this machine.
    #
    # LCOMP=0, the ENDF/B-V-compatible per-L format (LRF=2):
    "cm244_b81": ("endfb81/n-096_Cm_244.endf", "endfb8.1/n/96244.endf"),
    "am241_b81": ("endfb81/n-095_Am_241.endf", "endfb8.1/n/95241.endf"),
    # LCOMP=2 compact, without and with an INTG correlation block:
    "na23_b81": ("endfb81/n-011_Na_023.endf", "endfb8.1/n/11023.endf"),
    # Th-232 is the only tape here with two ranges: LCOMP=2 (LRF=3) and then an
    # unresolved LRU=2 covariance, §32.2.4. It is the whole URR coverage.
    "th232_b81": ("endfb81/n-090_Th_232.endf", "endfb8.1/n/90232.endf"),
    # LCOMP=2 for R-Matrix Limited (LRF=7). W-186 is the smallest of the three;
    # Cl-35 is the one whose NJS (8) differs from its NJSX (7).
    "w186_b81": ("endfb81/n-074_W_186.endf", "endfb8.1/n/74186.endf"),
    "cl35_b81": ("endfb81/n-017_Cl_035.endf", "endfb8.1/n/17035.endf"),
    "cu63_b81": ("endfb81/n-029_Cu_063.endf", "endfb8.1/n/29063.endf"),
    # LCOMP=1, the general format with a full covariance triangle. Ta-181 is
    # here for one reason beyond its format: its MF32 is 240 131 lines, the
    # only section on this machine long enough to wrap the sequence number.
    "mn55_b81": ("endfb81/n-025_Mn_055.endf", "endfb8.1/n/25055.endf"),
    "ta181_b81": ("endfb81/n-073_Ta_181.endf", "endfb8.1/n/73181.endf"),
    "mn55_jendl": ("jendl5/n/25055.endf", "JENDL-5/Mn55_jendl5_n.endf"),
    "pu239_jendl": ("jendl5/n/94239.endf", "JENDL-5/Pu239_jendl5_n.endf"),
    # The ENDF/B-VIII.1 GNDS distribution, as published by the NNDC. Only the
    # oracle tests need it: they compare what the GNDS reader gets out of Fe-56
    # against what the ENDF adapter gets out of the same evaluation, and that
    # needs the whole 18.8 MB file rather than the committed trim. Everything
    # else under ``kika/gnds`` runs off committed fixtures and needs no share.
    "fe56_gnds": (
        "ENDF-B-VIII.1-GNDS/ENDF-B-VIII.1-GNDS/neutrons/n-026_Fe_056.endf.gnds.xml",
        "ENDF-B-VIII.1-GNDS/neutrons/n-026_Fe_056.endf.gnds.xml",
    ),
    # The same evaluation as ``fe56_gnds`` below, in ENDF-6. This pair is the
    # phase 5 oracle: one evaluation, two encodings, and any disagreement
    # between them is a reader defect rather than a physics difference. Not the
    # same tape as ``fe56_host`` — that is JEFF-4.0 with a JEFF-3.3 MF4 graft.
    "fe56_b81": ("endfb81/n-026_Fe_056.endf", "n-026_Fe_056.endf"),
    # MF6, the energy-angle distributions. These three are not a sample: they
    # are the carriers. Swept over all 557 ENDF/B-VIII.1 neutron tapes, LAW=7
    # occurs twice in the whole library and both are in Be-9's MT16 (which is
    # also one of only 29 LCT=1 sections); LAW=6 occurs five times and three of
    # them are Li-6's MT41; and C-12's MT5 is the readiest LCT=3. See
    # ``docs/mf6_notes.md`` in kika-workspace for the census.
    "be9_b81": ("endfb81/n-004_Be_009.endf", "n-004_Be_009.endf"),
    "li6_b81": ("endfb81/n-003_Li_006.endf", "n-003_Li_006.endf"),
    "c12_b81": ("endfb81/n-006_C_012.endf", "n-006_C_012.endf"),
    "fe56_gnds_cov": (
        "ENDF-B-VIII.1-GNDS/ENDF-B-VIII.1-GNDS/neutrons/Covariances/"
        "n-026_Fe_056.endf.gnds-covar.xml",
        "ENDF-B-VIII.1-GNDS/neutrons/Covariances/n-026_Fe_056.endf.gnds-covar.xml",
    ),
    # MF7, the thermal scattering law. A sublibrary of its own (NSUB=12): no
    # neutron evaluation in ENDF/B-VIII.1 carries an MF7, so every one of these
    # is a ``tsl-`` tape. They are large — ``tsl-HinH2O`` is 87 MB and its
    # MF7/MT4 alone is 1.14 million records — which is why the committed
    # fixtures are small and the full-size witnesses are reached from here.
    #
    # ENDF/B-VIII.1, one per format branch that needs a full-size witness:
    "tsl_h_h2o": ("endfb81/tsl/tsl-HinH2O.endf",),        # 94 temperatures, NS=1
    "tsl_ortho_h": ("endfb81/tsl/tsl-ortho-H.endf",),     # the only LASYM=1
    "tsl_be_metal": ("endfb81/tsl/tsl-Be-metal.endf",),   # LTHR=1 + MF7/451
    "tsl_un": ("endfb81/tsl/tsl-NinUN.endf",),            # LTHR=3
    "tsl_s_ch4": ("endfb81/tsl/tsl-s-CH4.endf",),         # LTHR=2, LAT=0, NS=1
    # JEFF-4.0's own ``tsl/``, a different dialect and not merely different
    # content: 80-column records with sequence numbers, CRLF line endings,
    # elemental ZA (4000, not the pseudo-ZA 126 ENDF/B writes for the same
    # material), and LIST bodies padded with explicit zeros. ``tsl_Be_BeO.txt``
    # is the control — the *same* evaluation as ENDF/B's, adopted, 75-column.
    "tsl_jeff_be": ("jeff40/tsl/tsl_4-Be.txt",),
    "tsl_jeff_be_beo": ("jeff40/tsl/tsl_Be_BeO.txt",),
    "tsl_jeff_si": ("jeff40/tsl/tsl_Si.txt",),
    "serpent_input": ("serpent/PWRSphere.sss2", "PWRSphere.sss2"),
    # Smallest real Fe-56 ACE on the share (16 MB) is preferred over the
    # 112 MB JEFF one: the round-trip gate reads and rewrites the whole XSS,
    # and the format exercised is the same.
    "fe56_ace": (
        "ACE_samples/n-Fe056-ace.tendl.02c",
        "ACE_samples/26056.06c",
        "26056.06c",
    ),
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
    | {"serpent_input", "fe56_ace", "tape_root"}
)
#: Fixtures whose presence means the test spawns NJOY.
_NJOY_FIXTURES = frozenset({"njoy_exe"})

#: Fixtures whose presence means the test reads GNDS. Both kinds are here on
#: purpose: the committed fixtures need no shared tree and are *not* also
#: ``tape``-marked, while ``fe56_gnds_tape``/``fe56_gnds_cov_tape`` are in
#: ``_TAPES`` and so pick up ``tape`` from the set above as well. That makes
#: ``-m gnds`` mean "everything that reads a GNDS file", which is the question
#: worth asking, and ``-m "gnds and not tape"`` the fast lane of it.
#:
#: This marker was declared in ``pyproject.toml`` and applied by nothing until
#: the GNDS reader arrived, so ``-m gnds`` selected zero tests while the module
#: docstring above claimed it was applied automatically. It is applied now.
_GNDS_FIXTURES = frozenset({
    "fe56_gnds_tape", "fe56_gnds_cov_tape",
    "gnds_data_dir", "micro_fe56_gnds", "micro_ta182_gnds", "micro_be9_gnds",
    "h2_gnds", "h2_gnds_cov", "h3_gnds", "s36_gnds",
    "gnds_covariance_fixture",
})

#: Marks held back from a bare ``pytest``. Measured 2026-08-07: 29 of 1511
#: tests carry one of these and they dominate the wall clock — the seven
#: ``njoy`` tests alone run longer than the other 1504 together, because each
#: spawns the NJOY executable. A bare ``pytest`` used to run them, so verifying
#: a one-function change cost half an hour on a box that several sessions
#: share. Everything else still runs: this is not a fast *subset*, it is the
#: whole suite minus the parts that shell out or measure wall clock.
#:
#: ``tape`` joined them 2026-08-17, for the same reason one measurement later.
#: The three marks above left 98 non-``slow`` ``tape`` tests reading the share
#: on every bare run — 27 MB for the Fe-56 host tape alone, most of it network
#: IO. Measured after the change: a bare ``pytest`` is **8 min 00 s** (2 485
#: passed, 157 skipped) on a box shared with two other sessions, and one file
#: alone, ``kika/gnds/tests/test_covariance_oracle.py``, drops from 13.6 s to
#: 1.6 s. The *before* was never timed, so the "forty minutes" this change was
#: argued from is the previous session's figure, not one taken here — and what
#: remains of the eight minutes is mostly ``scripts/``, which no tape marker
#: touches. The same rule as the other three applies and is what makes this
#: safe: ``-m tape``, ``-k`` and ``--deep`` all still run them, and CI passes an
#: explicit ``-m``, so nothing about what *gets* verified changes — only what a
#: no-argument invocation defaults to.
#:
#: The alternative was the ``KIKA_TAPES=`` trick, and it is worse: ``Path("")``
#: is ``PosixPath('.')``, so an empty value makes the search root the *current
#: directory* rather than nowhere, and it leaves ``KIKA_LIB_TAPES`` and the
#: download cache resolving ~23 tapes anyway. A mark is deterministic.
_EXPENSIVE_MARKS = frozenset({"njoy", "slow", "perf", "tape"})


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Mark tests from the fixtures they request, then hold back the expensive ones.

    Hand-applied markers drift from reality the moment a test grows a new
    dependency. Deriving them from ``fixturenames`` cannot drift — so the
    auto-marking has to happen before anything reads those marks, including the
    hold-back below.

    The hold-back applies only when the invocation expressed no opinion. Any of
    ``-m``, ``-k`` or ``--deep`` means the caller aimed at something specific
    and gets exactly what they aimed at; in particular ``--deep`` must keep
    running everything, since its whole purpose is to prove nothing was skipped.
    """
    for item in items:
        names = set(getattr(item, "fixturenames", ()))
        if names & _TAPE_FIXTURES:
            item.add_marker(pytest.mark.tape)
        if names & _NJOY_FIXTURES:
            item.add_marker(pytest.mark.njoy)
        if names & _GNDS_FIXTURES:
            item.add_marker(pytest.mark.gnds)

    if (config.getoption("markexpr") or config.getoption("keyword")
            or _deep(config)):
        return

    held = pytest.mark.skip(
        reason="expensive by default; run it with --deep, -m or -k"
    )
    for item in items:
        if set(item.keywords) & _EXPENSIVE_MARKS:
            item.add_marker(held)


# ``--deep`` is enforced in :func:`_missing`, which every fixture below goes
# through, and nowhere else. An earlier version added a
# ``pytest_runtest_makereport`` wrapper that promoted *any* skip on a
# ``tape``/``njoy`` test to a failure, as a catch-all for module-level
# ``skipif``. It was wrong twice over: pytest reports an ``xfail`` as a skip, so
# it turned all five pinned defects into failures, and it also caught opt-in
# skips like ``REGEN_MICRO_TAPES``. The catch-all is not needed either — every
# test that reads external data now asks a fixture for it.


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
u235_b81_tape = _tape_fixture("u235_b81")
cf252_b81_tape = _tape_fixture("cf252_b81")
pu239_b81_tape = _tape_fixture("pu239_b81")
u5_nubar_covfil_tape = _tape_fixture("u5_nubar_covfil")
u5_boxer_tape = _tape_fixture("u5_boxer")
fe56_gnds_tape = _tape_fixture("fe56_gnds")
fe56_gnds_cov_tape = _tape_fixture("fe56_gnds_cov")
fe56_b81_tape = _tape_fixture("fe56_b81")

#: The MF6 law carriers. Named one at a time rather than looped, to match the
#: block above; the census that picked them is in ``docs/mf6_notes.md``.
be9_b81_tape = _tape_fixture("be9_b81")
li6_b81_tape = _tape_fixture("li6_b81")
c12_b81_tape = _tape_fixture("c12_b81")

tsl_h_h2o_tape = _tape_fixture("tsl_h_h2o")
tsl_ortho_h_tape = _tape_fixture("tsl_ortho_h")
tsl_be_metal_tape = _tape_fixture("tsl_be_metal")
tsl_un_tape = _tape_fixture("tsl_un")
tsl_s_ch4_tape = _tape_fixture("tsl_s_ch4")
tsl_jeff_be_tape = _tape_fixture("tsl_jeff_be")
tsl_jeff_be_beo_tape = _tape_fixture("tsl_jeff_be_beo")
tsl_jeff_si_tape = _tape_fixture("tsl_jeff_si")

#: The eleven evaluations on this machine that carry an MF32, in the order the
#: MF32 survey lists them. ``MF32_TAPES`` is exported so a test can parametrise
#: over the whole set rather than naming fixtures one at a time.
MF32_TAPES = (
    "cm244_b81", "am241_b81", "na23_b81", "th232_b81",
    "w186_b81", "cl35_b81", "cu63_b81",
    "mn55_b81", "ta181_b81", "mn55_jendl", "pu239_jendl",
)
for _name in MF32_TAPES:
    globals()[f"{_name}_tape"] = _tape_fixture(_name)
del _name


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


@pytest.fixture(scope="session")
def micro_pfns_tape() -> Path:
    """Committed Cf-252 slice: MF1/451, MF3/MT18, MF5/MT18+455, MF35/MT18.

    Cut from ENDF/B-VIII.1 the same way as ``micro_fe56_structural.endf`` —
    whole sections dropped, not one record reformatted. Cf-252 rather than
    U-235 because it is the smallest real evaluation carrying the full PFNS
    set, and it brings a real LF=1 MT18, a real LF=5 MT455 and four real LB=7
    covariance bands for ~770 kB.
    """
    path = MICRO_TAPE_DIR / "micro_cf252_pfns.endf"
    if not path.is_file():  # pragma: no cover - fixture is committed
        pytest.fail(f"committed micro-tape is missing: {path}")
    return path


#: Committed MF7 micro-tape name -> what it is the fixture for. The cut recipes
#: live in ``kika/endf/tests/test_micro_tape_regen.py``.
#:
#: MF7 could not be cut the way the other fixtures were. Everywhere else the
#: bulk of a tape is sections the test does not need, so dropping them leaves a
#: small file carrying the real thing. In a TSL evaluation the bulk **is** the
#: thing: MF7/MT4 is 98-99 % of every one of them, and it cannot be trimmed
#: without rewriting records. So the set is built two ways — one whole tape
#: small enough to commit outright, and three cut down to their *elastic*
#: sections, which keeps a real MT2 of each LTHR branch for a few hundred kB.
MICRO_TSL = {
    "sch4": "whole tsl-s-CH4.endf: MT2 LTHR=2, MT4 with NS=1 and LAT=0",
    "bemetal_elastic": "MT2 LTHR=1 (2306 Bragg edges, 11 T) + MF7/451",
    "un_elastic": "MT2 LTHR=3 (coherent then incoherent) + MF7/451",
    "jeff_be_elastic": "JEFF-4.0 dialect: 80-column, zero-padded LIST bodies",
}


@pytest.fixture(scope="session", params=sorted(MICRO_TSL))
def micro_tsl_tape(request) -> Path:
    """Each committed MF7 micro-tape in turn. See :data:`MICRO_TSL`."""
    path = MICRO_TAPE_DIR / f"micro_tsl_{request.param}.endf"
    if not path.is_file():  # pragma: no cover - fixture is committed
        pytest.fail(f"committed micro-tape is missing: {path}")
    return path


def _micro_tsl_fixture(key: str):
    def _fixture() -> Path:
        path = MICRO_TAPE_DIR / f"micro_tsl_{key}.endf"
        if not path.is_file():  # pragma: no cover - fixture is committed
            pytest.fail(f"committed micro-tape is missing: {path}")
        return path

    _fixture.__name__ = f"micro_tsl_{key}_tape"
    _fixture.__doc__ = f"Committed MF7 micro-tape: {MICRO_TSL[key]}."
    return pytest.fixture(scope="session", name=f"micro_tsl_{key}_tape")(_fixture)


micro_tsl_sch4_tape = _micro_tsl_fixture("sch4")
micro_tsl_bemetal_elastic_tape = _micro_tsl_fixture("bemetal_elastic")
micro_tsl_un_elastic_tape = _micro_tsl_fixture("un_elastic")
micro_tsl_jeff_be_elastic_tape = _micro_tsl_fixture("jeff_be_elastic")


#: MF32 micro-tape key -> the sub-format it is the fixture for. Kept here so a
#: test can parametrise over the set; the cut recipe lives in
#: ``kika/endf/tests/test_micro_tape_regen.py``.
MICRO_MF32 = {
    "na23": "LCOMP=2, no INTG block",
    "cm244": "LCOMP=0, the ENDF/B-V-compatible per-L format",
    "th232": "LCOMP=2 with INTG, plus an unresolved LRU=2 range",
    "cl35": "LCOMP=2 for R-Matrix Limited, NJS=8 against NJSX=7",
}


@pytest.fixture(scope="session", params=sorted(MICRO_MF32))
def micro_mf32_tape(request: pytest.FixtureRequest) -> Path:
    """Each committed MF32 slice in turn: MF1/451, MF2/151, MF32/151, verbatim.

    Cut section-by-section from ENDF/B-VIII.1 like the other real-slice
    fixtures, so every surviving byte is the evaluator's. Between them the four
    cover every MF32 sub-format reachable on this machine except LCOMP=1, whose
    smallest evaluation is too large to commit — see ``MF32_TAPES``.
    """
    path = MICRO_TAPE_DIR / f"micro_{request.param}_mf32.endf"
    if not path.is_file():  # pragma: no cover - fixtures are committed
        pytest.fail(f"committed micro-tape is missing: {path}")
    return path


@pytest.fixture(scope="session")
def micro_nubar_tape() -> Path:
    """Committed U-235 slice: MF1/451+452+455+456, MF3/MT18, MF31/452+455+456.

    Cut from ENDF/B-VIII.1 the verbatim way, and it is the only fixture that
    covers nu-bar at all. U-235 rather than a smaller fissile nuclide because
    its MF31 carries all three MTs *and* the NC-type (LTY=0) total — the
    covariance the file declares as a sum of MT455 and MT456 rather than
    storing — which is the case a fixture built from stored matrices alone
    would never exercise. 914 lines.
    """
    path = MICRO_TAPE_DIR / "micro_u235_nubar.endf"
    if not path.is_file():  # pragma: no cover - fixture is committed
        pytest.fail(f"committed micro-tape is missing: {path}")
    return path


@pytest.fixture(scope="session")
def micro_pfns_cov_tape() -> Path:
    """Committed synthetic tape carrying a tiny MF5/MT18 and its MF35/MT18.

    The sampler's fast lane: eight outgoing groups over two incident bands,
    small enough that a test can assert on the whole covariance by hand.
    """
    path = MICRO_TAPE_DIR / "micro_pfns_cov.endf"
    if not path.is_file():  # pragma: no cover - fixture is committed
        pytest.fail(f"committed micro-tape is missing: {path}")
    return path


# --------------------------------------------------------------------------
# Fixtures — committed GNDS files
# --------------------------------------------------------------------------

#: The committed fixtures mirror the distribution's own layout — evaluations at
#: the top, covariances under ``Covariances/`` — rather than being flattened
#: into one directory. That is not tidiness: an ``externalFile`` entry names its
#: sibling by *relative path* (``Covariances/n-001_H_002.endf.gnds-covar.xml``
#: one way, ``../n-009_F_019.endf.gnds.xml`` the other), so flattening them
#: would make every committed href resolve to somewhere that does not exist and
#: the reader would be tested against a layout no real library uses.
GNDS_DATA_DIR = REPO_ROOT / "kika" / "gnds" / "tests" / "data"


def _gnds_file(name: str) -> Path:
    path = GNDS_DATA_DIR / name
    if not path.is_file():  # pragma: no cover - fixtures are committed
        pytest.fail(f"committed GNDS fixture is missing: {path}")
    return path


@pytest.fixture(scope="session")
def gnds_data_dir() -> Path:
    """The committed GNDS fixture directory itself.

    For the tests that walk the whole set — "every committed fixture parses",
    "every one declares a format this reader accepts" — rather than naming one.
    Walk it with ``rglob``: the covariances are one level down, in
    ``Covariances/``, exactly as the distribution ships them.
    """
    if not GNDS_DATA_DIR.is_dir():  # pragma: no cover - committed
        pytest.fail(f"committed GNDS fixture directory is missing: {GNDS_DATA_DIR}")
    return GNDS_DATA_DIR


@pytest.fixture(scope="session")
def h2_gnds() -> Path:
    """n + H2 from ENDF/B-VIII.1, **unmodified**, with its covariance sibling.

    The paired fixture, and the only one where a cross-file ``href`` and the
    SHA-1 in ``externalFiles`` are both real. 113 kB for 1013 nodes, which is
    what makes it the pair worth committing: ``regions1d`` and ``regions2d``,
    a ``reference`` form, ``crossSectionSum`` with resolvable ``summands``,
    ``angularTwoBody``/``isotropic2d``/``unspecified`` and
    ``uncorrelated``/``NBodyPhaseSpace`` — all of them laws kika reads, since
    ``uncorrelated`` landed in phase 7b.
    """
    return _gnds_file("n-001_H_002.endf.gnds.xml")


@pytest.fixture(scope="session")
def h2_gnds_cov() -> Path:
    """The covariance sibling of :func:`h2_gnds`, unmodified. 11 kB, 4 sections."""
    return _gnds_file("Covariances/n-001_H_002.endf.gnds-covar.xml")


@pytest.fixture(scope="session")
def h3_gnds() -> Path:
    """n + H3 from ENDF/B-VIII.1, **unmodified**, and here for one node.

    ``uncorrelated/angular`` is an ``xs:choice`` of three (``gnds.xsd:1686``),
    and the census measured its members across the 558 distributed neutron
    evaluations: ``isotropic2d`` 126 095, ``XYs2d`` **406 in 144 files**,
    ``forward`` 0. The three fixtures beside it carry **25 ``isotropic2d`` and
    not one ``XYs2d``**, so that branch of the reader shipped with
    ``uncorrelated`` without ever having seen a real one. This is the smallest
    of the 144 files that fixes it — 59 kB — and it carries exactly one, with
    fifteen sub-functions and axes of its own.

    It also reads with an **empty** ``unsupported`` list, which no other
    committed evaluation does: every law in it is one kika implements.
    """
    return _gnds_file("n-001_H_003.endf.gnds.xml")


@pytest.fixture(scope="session")
def micro_fe56_gnds() -> Path:
    """Fe-56 cut down to its resonances. One of the two trimmed GNDS fixtures.

    No distributed file under 1.6 MB carries a resolved resonance region, and
    Fe-56 — the material this project is about — is 18.8 MB, so this one is
    trimmed: ``resonances`` verbatim (``RMatrix``, five spin groups, their
    ``resonanceParameters`` tables), the two reactions its
    ``resonanceReactions`` link to, everything else dropped or truncated.

    **Its cross sections are abridged and are not physics.** Assert on structure
    here; anything comparing numbers against the ENDF path wants
    ``fe56_gnds_tape``, which is the whole file. Rebuild with
    ``python kika/gnds/tests/data/build_micro_fe56_gnds.py``, which validates
    the result against FUDGE's GNDS 2.0 schema before writing it.
    """
    return _gnds_file("micro_fe56.gnds.xml")


@pytest.fixture(scope="session")
def micro_ta182_gnds() -> Path:
    """The other §19 resolved formalism and the unresolved region, in one file.

    ``micro_fe56_gnds`` covers ``RMatrix``; between them the two cover §19.
    Ta-182 carries a ``BreitWigner`` (MultiLevel, ``calculateChannelRadius``)
    **and** a ``tabulatedWidths`` with six (L, J) groups, and its whole
    ``resonances`` subtree is 11.7 kB — the 373 kB of the distributed file is
    ``reactions`` and ``sums``, which is what the trim removes.

    Same warning as Fe-56: the resonance block is verbatim, everything outside
    it is abridged and is not physics. Built by the same script.
    """
    return _gnds_file("micro_ta182.gnds.xml")


@pytest.fixture(scope="session")
def s36_gnds() -> Path:
    """n + S36 from ENDF/B-VIII.1, **unmodified**, and the witness of three nodes.

    Committed for ``KalbachMann`` — it is the smallest of the 272 evaluations
    carrying one and holds **six** — and it turned out to be the smallest file
    carrying ``branching1d`` and ``branching3d`` as well, five of each. 226 kB
    for three of the four nodes phase 7b had left, which is why it is copied
    whole rather than trimmed: the trim would have had to keep most of it.

    All six ``KalbachMann`` are ``f`` + ``r`` with no ``a``, which is not this
    file being unusual — the census counted **0 ``a`` in 3 730** across the
    distribution.
    """
    return _gnds_file("n-016_S_036.endf.gnds.xml")


@pytest.fixture(scope="session")
def micro_be9_gnds() -> Path:
    """Be-9's ``2n + 2He4`` reaction, trimmed. The **only** §18.5 witness there is.

    ``angularEnergy`` occurs **twice in the whole ENDF/B-VIII.1 GNDS
    distribution**, and both are in ``n-004_Be_009`` — in this one reaction, one
    per product. There is no second file to choose instead, so the alternative
    to trimming was committing 0.92 MB for two nodes.

    It is trimmed rather than copied for the reason the other two are: the
    construct exists only in a file too big to commit. Its ``resonances`` is a
    lone ``scatteringRadius`` with no ``resonanceReactions``, so unlike Fe-56
    and Ta-182 the kept reaction is chosen by what it *carries*, not by what the
    resonance block links to.

    Same warning as the other trims: **the numbers are abridged and are not
    physics.** What is faithful is the shape — including the axis order, which
    is the one thing that distinguishes §18.5 from §18.4 and which no schema
    checks.
    """
    return _gnds_file("micro_be9.gnds.xml")


#: Covariance fixture -> the construct it is here for. Each is the *smallest*
#: file in the 270-file ENDF/B-VIII.1 covariance distribution carrying it, and
#: each is committed unmodified. Between them they cover every covariance
#: construct the library uses; the survey that picked them is in the phase 5
#: section of ``docs/gnds_roadmap.md``.
GNDS_COVARIANCE_FIXTURES = {
    "n-014_Si_032": "parameterCovariances alone — a suite with no covarianceSections at all",
    "n-069_Tm_171": "averageParameterCovariance (URR), and array compression='flattened'",
    "n-009_F_019": "sum/summand, columnData, crossTerm, mixed, shortRangeSelfScalingVariance",
    "n-057_La_139": "slices/slice — the MF34 Legendre-order link — plus MF40",
}


@pytest.fixture(scope="session", params=sorted(GNDS_COVARIANCE_FIXTURES))
def gnds_covariance_fixture(request: pytest.FixtureRequest) -> Path:
    """Each committed covariance fixture in turn. See :data:`GNDS_COVARIANCE_FIXTURES`.

    These are covariance files **without** their ``reactionSuite`` sibling: the
    distribution's ``externalFile`` entry points at a file that is not committed
    beside them. That is deliberate, not an oversight — a ``covarianceSuite``
    read on its own is a case the reader has to handle, and handling it means
    reporting the unresolvable ``href`` rather than raising.
    """
    return _gnds_file(f"Covariances/{request.param}.endf.gnds-covar.xml")
