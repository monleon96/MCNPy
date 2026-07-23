"""
Build the benchmarks SQLite database from a DICE ``sensitivity/`` folder.

This is a one-time step run by a user who has their own NEA-licensed DICE data.
It walks the gzipped SCALE/TSUNAMI SDF files, parses each with
:func:`kika.sensitivities.read_sdf`, keeps the region-integrated (system-total)
profiles (and, optionally, the per-mixture spatial sub-profiles), precomputes
coarse-energy-region integral sensitivities for fast screening, and stores full
per-group sensitivity + error vectors as compressed BLOBs.

Alongside the sensitivities it ingests the cheap auxiliary DICE data that lives in
sibling folders of ``sensitivity/`` (auto-detected): experimental k-eff
(``keff/``), 299-group flux spectra (``spectra/``), 1-group reaction-rate balances
(``balance/``), and the raw KENO/MCNP input decks (``input/``, stored verbatim).

Parsing is CPU/IO-bound and runs across a process pool; the SQLite writes happen
serially in the main process. The raw DICE folder is only needed here; afterwards
the self-contained database is all kika reads.
"""

import datetime
import gzip
import json
import os
import sqlite3
import tempfile
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from kika.benchmarks import config
from kika.benchmarks._aux import (
    parse_balance_summary,
    parse_keff_workbook,
    parse_spectrum,
)
from kika.benchmarks._blob import pack_f32
from kika.benchmarks._constants import (
    CODE_PRIORITY,
    EPITHERMAL_MAX_MEV,
    GROUP_PRIORITY,
    LIBRARY_PRIORITY,
    SCHEMA_VERSION,
    THERMAL_MAX_MEV,
)
from kika.benchmarks._naming import benchmark_id_to_keff_abbrevs, parse_filename
from kika.benchmarks.exceptions import BenchmarksError
from kika.sensitivities.sdf_parser import read_sdf

_SCHEMA_SQL = """
CREATE TABLE benchmarks (
    benchmark_id TEXT PRIMARY KEY,
    category     TEXT,
    btype        TEXT,
    spectrum     TEXT,
    series       INTEGER,
    case_number  INTEGER,
    keff_exp     REAL,
    keff_exp_unc REAL
);
CREATE INDEX ix_bench_category ON benchmarks(category);
CREATE INDEX ix_bench_spectrum ON benchmarks(spectrum);

CREATE TABLE profiles (
    profile_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    benchmark_id    TEXT NOT NULL REFERENCES benchmarks(benchmark_id),
    code            TEXT,
    library         TEXT,
    group_structure TEXT,
    ngroups         INTEGER,
    variant_index   INTEGER DEFAULT 0,
    keff            REAL,
    keff_rel_err    REAL,
    pert_energies   BLOB,
    is_preferred    INTEGER DEFAULT 0,
    source_filename TEXT,
    UNIQUE(benchmark_id, code, library, group_structure, variant_index)
);
CREATE INDEX ix_prof_benchmark ON profiles(benchmark_id);

CREATE TABLE profile_sensitivities (
    profile_id   INTEGER NOT NULL REFERENCES profiles(profile_id),
    zaid         INTEGER NOT NULL,
    mt           INTEGER NOT NULL,
    unit         INTEGER NOT NULL DEFAULT 0,
    region       INTEGER NOT NULL DEFAULT 0,
    nuclide      TEXT,
    reaction     TEXT,
    s_thermal    REAL,
    s_epithermal REAL,
    s_fast       REAL,
    s_total      REAL,
    s_abs_total  REAL,
    sensitivity  BLOB,
    error        BLOB,
    PRIMARY KEY (profile_id, zaid, mt, unit, region)
);
CREATE INDEX ix_sens_screen ON profile_sensitivities(zaid, mt, unit, region, s_abs_total);
CREATE INDEX ix_sens_nuclide ON profile_sensitivities(nuclide, reaction);

CREATE TABLE benchmark_spectra (
    benchmark_id TEXT PRIMARY KEY REFERENCES benchmarks(benchmark_id),
    ngroups      INTEGER,
    energies     BLOB,
    flux         BLOB
);

CREATE TABLE benchmark_balance (
    benchmark_id TEXT PRIMARY KEY REFERENCES benchmarks(benchmark_id),
    summary      TEXT,
    raw          BLOB
);

CREATE TABLE benchmark_inputs (
    benchmark_id TEXT NOT NULL REFERENCES benchmarks(benchmark_id),
    code         TEXT,
    deck         BLOB,
    PRIMARY KEY (benchmark_id, code)
);

CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
"""


def _region_integrals(pert_energies: np.ndarray, sensitivity: np.ndarray):
    """Return (thermal, epithermal, fast, total, abs_total) integral sensitivities.

    Groups are assigned to a region by their upper energy edge. On the standard
    DICE grids 0.625 eV and 100 keV are exact group boundaries, so the assignment
    is exact.
    """
    upper = pert_energies[1:]  # upper edge of each group (ascending, MeV)
    thermal = float(sensitivity[upper <= THERMAL_MAX_MEV].sum())
    fast = float(sensitivity[upper > EPITHERMAL_MAX_MEV].sum())
    epithermal = float(
        sensitivity[(upper > THERMAL_MAX_MEV) & (upper <= EPITHERMAL_MAX_MEV)].sum()
    )
    total = float(sensitivity.sum())
    abs_total = float(np.abs(sensitivity).sum())
    return thermal, epithermal, fast, total, abs_total


def _parse_file(task: tuple):
    """
    Worker: parse one gzipped SDF into a serializable payload.

    Runs in a pool process. Returns ``("ok", payload)`` where payload holds the
    benchmark/profile metadata and already-compressed vectors, or
    ``("skip", filename, reason)``. Compression happens here so it parallelizes.
    """
    gz_path, folder_category, store_errors, store_region_profiles = task
    name = os.path.basename(gz_path)
    meta = parse_filename(name, folder_category=folder_category)
    tmp_path = None
    try:
        with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sdf", delete=False, encoding="utf-8"
        ) as out:
            tmp_path = out.name
            out.write(text)
        sdf = read_sdf(tmp_path)
    except Exception as exc:  # noqa: BLE001 - report and continue
        return ("skip", name, f"{type(exc).__name__}: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    def _is_system_total(d):
        return (d.unit or 0, d.region or 0) == (0, 0)

    if store_region_profiles:
        kept = list(sdf.data)
    else:
        kept = [d for d in sdf.data if _is_system_total(d)]
    if not kept:
        return ("skip", name, "no region-integrated profiles")

    pert = np.asarray(sdf.pert_energies, dtype=float)
    ngroups = len(pert) - 1
    reactions = []
    for d in kept:
        sens = np.asarray(d.sensitivity, dtype=float)
        s_t, s_e, s_f, s_tot, s_abs = _region_integrals(pert, sens)
        reactions.append(
            (
                d.zaid,
                d.mt,
                int(d.unit or 0),
                int(d.region or 0),
                d.nuclide,
                d.reaction_name,
                s_t,
                s_e,
                s_f,
                s_tot,
                s_abs,
                pack_f32(sens),
                pack_f32(d.error) if store_errors else None,
            )
        )
    payload = {
        "benchmark_id": meta.benchmark_id,
        "category": meta.category,
        "btype": meta.btype,
        "spectrum": meta.spectrum,
        "series": meta.series,
        "case_number": meta.case_number,
        "code": meta.code,
        "library": meta.library,
        "group_structure": meta.group_structure or f"{ngroups}-Group",
        "ngroups": ngroups,
        "variant_index": meta.variant_index,
        "keff": sdf.r0,
        "keff_rel_err": sdf.e0,
        "pert_energies": pack_f32(pert),
        "source_filename": name,
        "reactions": reactions,
    }
    return ("ok", payload)


def _write_payload(conn: sqlite3.Connection, payload: dict) -> int:
    """Insert one parsed benchmark payload. Returns number of reaction rows added."""
    conn.execute(
        "INSERT OR IGNORE INTO benchmarks "
        "(benchmark_id, category, btype, spectrum, series, case_number) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            payload["benchmark_id"],
            payload["category"],
            payload["btype"],
            payload["spectrum"],
            payload["series"],
            payload["case_number"],
        ),
    )
    cur = conn.execute(
        "INSERT OR IGNORE INTO profiles "
        "(benchmark_id, code, library, group_structure, ngroups, variant_index, "
        " keff, keff_rel_err, pert_energies, source_filename) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            payload["benchmark_id"],
            payload["code"],
            payload["library"],
            payload["group_structure"],
            payload["ngroups"],
            payload["variant_index"],
            payload["keff"],
            payload["keff_rel_err"],
            payload["pert_energies"],
            payload["source_filename"],
        ),
    )
    if cur.rowcount == 0:
        return 0  # duplicate profile key
    profile_id = cur.lastrowid
    rows = [(profile_id, *r) for r in payload["reactions"]]
    conn.executemany(
        "INSERT OR IGNORE INTO profile_sensitivities "
        "(profile_id, zaid, mt, unit, region, nuclide, reaction, s_thermal, "
        " s_epithermal, s_fast, s_total, s_abs_total, sensitivity, error) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    return len(rows)


# ---------------------------------------------------------------------------
# Auxiliary data workers (spectra / balance / input decks)
# ---------------------------------------------------------------------------
def _parse_aux_file(task: tuple):
    """
    Worker: parse one auxiliary DICE file (spectrum, balance or input deck).

    ``task = (kind, path)`` with ``kind`` in ``{'spec', 'bal', 'inp'}``. Returns a
    ``(kind, benchmark_id, ...)`` payload tuple or ``("skip", name, reason)``.
    """
    kind, path = task
    name = os.path.basename(path)
    benchmark_id = name.split("_")[0]
    try:
        if kind == "inp":
            with open(path, "rb") as fh:
                raw = fh.read()
            code = "KENO" if "_KENO_" in name else "MCNP" if "_MCNP_" in name else None
            return ("inp", benchmark_id, code, raw)

        if kind == "bal":
            with open(path, "rb") as fh:
                raw = fh.read()
            text = gzip.decompress(raw).decode("utf-8", errors="replace")
            summary = json.dumps(parse_balance_summary(text))
            return ("bal", benchmark_id, summary, raw)

        # kind == "spec"
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        parsed = parse_spectrum(text)
        if parsed is None:
            return ("skip", name, "no spectrum rows")
        energies, flux = parsed
        # The SPEC E-up column is in eV; store in MeV so the spectrum shares the
        # sensitivity profile's energy axis.
        energies = [e / 1.0e6 for e in energies]
        return (
            "spec",
            benchmark_id,
            len(energies),
            pack_f32(energies),
            pack_f32(flux),
        )
    except Exception as exc:  # noqa: BLE001 - report and continue
        return ("skip", name, f"{type(exc).__name__}: {exc}")


def _write_aux(conn: sqlite3.Connection, payload: tuple, known_ids: set) -> bool:
    """Insert one aux payload if its benchmark is known. Returns True on insert."""
    kind = payload[0]
    benchmark_id = payload[1]
    if benchmark_id not in known_ids:
        return False
    if kind == "spec":
        _, _, ngroups, energies, flux = payload
        conn.execute(
            "INSERT OR IGNORE INTO benchmark_spectra "
            "(benchmark_id, ngroups, energies, flux) VALUES (?, ?, ?, ?)",
            (benchmark_id, ngroups, energies, flux),
        )
    elif kind == "bal":
        _, _, summary, raw = payload
        conn.execute(
            "INSERT OR IGNORE INTO benchmark_balance (benchmark_id, summary, raw) "
            "VALUES (?, ?, ?)",
            (benchmark_id, summary, raw),
        )
    elif kind == "inp":
        _, _, code, raw = payload
        conn.execute(
            "INSERT OR IGNORE INTO benchmark_inputs (benchmark_id, code, deck) "
            "VALUES (?, ?, ?)",
            (benchmark_id, code, raw),
        )
    else:
        return False
    return True


def _ingest_keff(conn: sqlite3.Connection, keff_path: Path) -> int:
    """Join the NRG experimental-keff workbook onto the benchmarks. Returns match count."""
    table = parse_keff_workbook(str(keff_path))
    matched = 0
    for (benchmark_id,) in conn.execute("SELECT benchmark_id FROM benchmarks").fetchall():
        for abbrev in benchmark_id_to_keff_abbrevs(benchmark_id):
            if abbrev in table:
                keff, unc = table[abbrev]
                conn.execute(
                    "UPDATE benchmarks SET keff_exp = ?, keff_exp_unc = ? "
                    "WHERE benchmark_id = ?",
                    (keff, unc, benchmark_id),
                )
                matched += 1
                break
    return matched


def build_benchmarks_db(
    source_dir: Optional[str] = None,
    db_path: Optional[str] = None,
    categories: Optional[List[str]] = None,
    store_errors: bool = True,
    store_region_profiles: bool = False,
    source_root: Optional[str] = None,
    overwrite: bool = True,
    n_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """
    Build the benchmarks database from DICE data.

    Parameters
    ----------
    source_dir : str, optional
        Path to the DICE ``sensitivity/`` folder (containing HEU/IEU/... subfolders).
        Resolved via :func:`kika.benchmarks.config.get_source_dir` if not given.
    db_path : str, optional
        Output database path. Resolved via config/env/default if not given.
    categories : list of str, optional
        Restrict ingest to these categories (e.g. ``["HEU"]``) — useful for testing.
    store_errors : bool, optional
        Store per-group relative-error vectors alongside the sensitivities.
        Default True (roughly doubles the per-group vector storage).
    store_region_profiles : bool, optional
        Also keep the non-system-total per-mixture spatial sub-profiles (the
        non-``(0, 0)`` ``(unit, region)`` profiles). Default False. Enabling this
        multiplies the sensitivity payload by ~2-3x; screening and preferred-variant
        selection still use only the ``(0, 0)`` system totals.
    source_root : str, optional
        DiceData root used to locate the auxiliary sibling folders
        (``input/``, ``spectra/``, ``balance/``, ``keff/``). Defaults to the parent
        of ``source_dir``. Whichever of those folders exist are ingested; missing
        ones are skipped silently.
    overwrite : bool, optional
        Remove any existing database at ``db_path`` first. Default True.
    n_workers : int, optional
        Number of worker processes. Default ``os.cpu_count() - 1`` (minimum 1).
    progress_callback : callable, optional
        Called as ``progress_callback(index, total, benchmark_id)`` per file.

    Returns
    -------
    dict
        Summary counts: ``benchmarks``, ``profiles``, ``reactions``, ``spectra``,
        ``balance``, ``inputs``, ``keff_matched``, ``db_bytes``, ``files_total``,
        ``files_skipped``, ``skipped``.
    """
    source_dir = config.get_source_dir(source_dir)
    if not source_dir or not os.path.isdir(source_dir):
        raise BenchmarksError(
            f"DICE source directory not found: {source_dir!r}. Pass source_dir=, set "
            "KIKA_DICE_SOURCE_DIR, or call benchmarks.configure(source_dir=...)."
        )
    source_dir = Path(source_dir)
    root = Path(source_root) if source_root else source_dir.parent
    db_path = config.get_db_path(db_path)

    if os.path.exists(db_path):
        if not overwrite:
            raise BenchmarksError(f"Database already exists: {db_path} (overwrite=False)")
        os.remove(db_path)
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

    files = list(_iter_sdf_files(source_dir, categories))
    total = len(files)
    tasks = [
        (str(gz), gz.parent.name, store_errors, store_region_profiles) for gz in files
    ]
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    skipped: List[tuple] = []
    n_reactions = 0

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    conn.executescript(_SCHEMA_SQL)

    def _consume(i, result):
        nonlocal n_reactions
        if progress_callback is not None:
            label = result[1] if result[0] == "skip" else result[1]["benchmark_id"]
            progress_callback(i, total, label)
        if result[0] == "skip":
            skipped.append((result[1], result[2]))
        else:
            added = _write_payload(conn, result[1])
            if added == 0:
                skipped.append((result[1]["source_filename"], "duplicate profile key"))
            else:
                n_reactions += added

    aux_counts = {"spectra": 0, "balance": 0, "inputs": 0}
    keff_matched = 0

    try:
        if n_workers == 1:
            for i, task in enumerate(tasks):
                _consume(i, _parse_file(task))
        else:
            with Pool(n_workers) as pool:
                for i, result in enumerate(
                    pool.imap_unordered(_parse_file, tasks, chunksize=1)
                ):
                    _consume(i, result)

        _assign_preferred_variants(conn)
        conn.commit()

        # -- auxiliary data ------------------------------------------------
        known_ids = {
            row[0] for row in conn.execute("SELECT benchmark_id FROM benchmarks")
        }
        aux_counts = _ingest_aux_data(
            conn, root, categories, known_ids, n_workers
        )

        keff_path = root / "keff" / "NRG_VanDerMark_Tables.xlsx"
        if keff_path.is_file():
            try:
                keff_matched = _ingest_keff(conn, keff_path)
            except Exception as exc:  # noqa: BLE001 - keff is optional
                skipped.append((keff_path.name, f"keff ingest failed: {exc}"))

        n_benchmarks = conn.execute("SELECT COUNT(*) FROM benchmarks").fetchone()[0]
        n_profiles = conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]

        _write_meta(
            conn,
            {
                "schema_version": str(SCHEMA_VERSION),
                "ingest_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "source_dir": str(source_dir),
                "store_errors": str(store_errors),
                "store_region_profiles": str(store_region_profiles),
                "n_benchmarks": str(n_benchmarks),
                "n_profiles": str(n_profiles),
                "n_reactions": str(n_reactions),
                "n_spectra": str(aux_counts["spectra"]),
                "n_balance": str(aux_counts["balance"]),
                "n_inputs": str(aux_counts["inputs"]),
                "n_keff_matched": str(keff_matched),
                "files_total": str(total),
                "files_skipped": str(len(skipped)),
            },
        )
        conn.commit()
        conn.execute("VACUUM")
        conn.commit()
    finally:
        conn.close()

    if progress_callback is not None:
        progress_callback(total, total, "done")

    return {
        "benchmarks": n_benchmarks,
        "profiles": n_profiles,
        "reactions": n_reactions,
        "spectra": aux_counts["spectra"],
        "balance": aux_counts["balance"],
        "inputs": aux_counts["inputs"],
        "keff_matched": keff_matched,
        "db_bytes": os.path.getsize(db_path),
        "files_total": total,
        "files_skipped": len(skipped),
        "skipped": skipped,
    }


def _iter_sdf_files(source_dir: Path, categories: Optional[List[str]]):
    """Yield DICE ``*_SENS*.gz`` paths, optionally restricted to some categories."""
    if categories:
        for cat in categories:
            yield from sorted((source_dir / cat).glob("*_SENS*.gz"))
    else:
        yield from sorted(source_dir.rglob("*_SENS*.gz"))


def _iter_aux_files(folder: Path, categories: Optional[List[str]], pattern: str):
    """Yield auxiliary ``*_<TYPE>*.gz`` paths under a DiceData sibling folder."""
    if not folder.is_dir():
        return
    if categories:
        for cat in categories:
            yield from sorted((folder / cat).glob(pattern))
    else:
        yield from sorted(folder.rglob(pattern))


def _ingest_aux_data(
    conn: sqlite3.Connection,
    root: Path,
    categories: Optional[List[str]],
    known_ids: set,
    n_workers: int,
) -> dict:
    """Ingest spectra / balance / input decks from DiceData sibling folders."""
    aux_tasks: List[tuple] = []
    for kind, folder_name, pattern in (
        ("spec", "spectra", "*_SPEC*.gz"),
        ("bal", "balance", "*_BAL*.gz"),
        ("inp", "input", "*_INP*.gz"),
    ):
        for path in _iter_aux_files(root / folder_name, categories, pattern):
            aux_tasks.append((kind, str(path)))

    counts = {"spectra": 0, "balance": 0, "inputs": 0}
    if not aux_tasks:
        return counts

    key = {"spec": "spectra", "bal": "balance", "inp": "inputs"}

    def _consume(result):
        if result[0] == "skip":
            return
        if _write_aux(conn, result, known_ids):
            counts[key[result[0]]] += 1

    if n_workers == 1:
        for task in aux_tasks:
            _consume(_parse_aux_file(task))
    else:
        with Pool(n_workers) as pool:
            for result in pool.imap_unordered(_parse_aux_file, aux_tasks, chunksize=4):
                _consume(result)
    conn.commit()
    return counts


def _assign_preferred_variants(conn: sqlite3.Connection) -> None:
    """Flag one preferred profile per benchmark by the priority ordering."""
    rows = conn.execute(
        "SELECT profile_id, benchmark_id, code, library, group_structure, variant_index "
        "FROM profiles"
    ).fetchall()
    best: dict = {}  # benchmark_id -> (sort_key, profile_id)
    for profile_id, benchmark_id, code, library, gs, vidx in rows:
        key = _variant_sort_key(code, library, gs, vidx)
        if benchmark_id not in best or key < best[benchmark_id][0]:
            best[benchmark_id] = (key, profile_id)
    conn.executemany(
        "UPDATE profiles SET is_preferred = 1 WHERE profile_id = ?",
        [(pid,) for _, pid in best.values()],
    )


def _variant_sort_key(code, library, group_structure, variant_index) -> tuple:
    """Priority tuple for choosing the preferred profile of a benchmark (lower wins)."""
    ci = CODE_PRIORITY.index(code) if code in CODE_PRIORITY else len(CODE_PRIORITY)
    li = (
        LIBRARY_PRIORITY.index(library)
        if library in LIBRARY_PRIORITY
        else len(LIBRARY_PRIORITY)
    )
    gi = (
        GROUP_PRIORITY.index(group_structure)
        if group_structure in GROUP_PRIORITY
        else len(GROUP_PRIORITY)
    )
    return (ci, li, gi, variant_index)


def _write_meta(conn: sqlite3.Connection, meta: dict) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        list(meta.items()),
    )
