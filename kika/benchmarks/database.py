"""
Read access to the built benchmarks SQLite database.

The database is produced by :func:`kika.benchmarks.build_benchmarks_db` from a
user's own DICE data. This module only reads it. Connection handling mirrors
``kika.exfor.database.X4ProDatabase`` (lazy connection, ``sqlite3.Row`` factory,
parameterized queries).
"""

import json
import os
import sqlite3
from typing import List, Optional

from kika.benchmarks import config
from kika.benchmarks._blob import unpack_f32
from kika.benchmarks.exceptions import (
    BenchmarkNotFoundError,
    DatabaseNotConfiguredError,
)

# Whitelist mapping energy region -> indexed integral-sensitivity column. Used to
# build screening SQL safely (never interpolate user input into column names).
_REGION_COLUMN = {
    "thermal": "s_thermal",
    "epithermal": "s_epithermal",
    "fast": "s_fast",
    "total": "s_total",
}


class BenchmarksDatabase:
    """Interface to a built ``kika_benchmarks.db`` SQLite database."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = config.get_db_path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    # -- connection lifecycle ------------------------------------------------
    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            if not self.db_path or not os.path.isfile(self.db_path):
                raise DatabaseNotConfiguredError(
                    f"Benchmarks database not found at {self.db_path!r}. Build it with "
                    "kika.benchmarks.build_benchmarks_db(...) or set the path via "
                    "benchmarks.configure(db_path=...) / KIKA_BENCHMARKS_DB_PATH."
                )
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "BenchmarksDatabase":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- metadata ------------------------------------------------------------
    def get_statistics(self) -> dict:
        """Return counts and ingest metadata for the database."""
        conn = self._get_connection()
        stats = {
            "benchmarks": conn.execute("SELECT COUNT(*) FROM benchmarks").fetchone()[0],
            "profiles": conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0],
            "reactions": conn.execute(
                "SELECT COUNT(*) FROM profile_sensitivities"
            ).fetchone()[0],
        }
        meta = {
            row["key"]: row["value"]
            for row in conn.execute("SELECT key, value FROM meta")
        }
        stats["meta"] = meta
        return stats

    # -- catalog -------------------------------------------------------------
    def list_benchmarks(
        self,
        category: Optional[str] = None,
        spectrum: Optional[str] = None,
        library: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> List[dict]:
        """List benchmarks with optional filters (one row per benchmark)."""
        conn = self._get_connection()
        conditions, params = [], []
        if category:
            conditions.append("b.category = ?")
            params.append(category)
        if spectrum:
            conditions.append("b.spectrum = ?")
            params.append(spectrum)
        if search:
            conditions.append("b.benchmark_id LIKE ?")
            params.append(f"%{search}%")
        if library:
            conditions.append(
                "EXISTS (SELECT 1 FROM profiles p WHERE p.benchmark_id = b.benchmark_id "
                "AND p.library = ?)"
            )
            params.append(library)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = (
            "SELECT b.benchmark_id, b.category, b.btype, b.spectrum, b.series, "
            "b.case_number, "
            "(SELECT COUNT(*) FROM profiles p WHERE p.benchmark_id = b.benchmark_id) "
            "AS n_variants "
            f"FROM benchmarks b {where} ORDER BY b.benchmark_id LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])
        return [dict(r) for r in conn.execute(query, params)]

    def get_benchmark(self, benchmark_id: str) -> dict:
        """Return a benchmark's metadata plus the list of its profile variants."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM benchmarks WHERE benchmark_id = ?", (benchmark_id,)
        ).fetchone()
        if row is None:
            raise BenchmarkNotFoundError(f"Unknown benchmark id: {benchmark_id!r}")
        result = dict(row)
        result["profiles"] = [
            dict(p)
            for p in conn.execute(
                "SELECT profile_id, code, library, group_structure, ngroups, keff, "
                "keff_rel_err, is_preferred, source_filename "
                "FROM profiles WHERE benchmark_id = ? "
                "ORDER BY is_preferred DESC, profile_id",
                (benchmark_id,),
            )
        ]
        return result

    def get_preferred_profile(self, benchmark_id: str) -> dict:
        """Return the preferred profile row for a benchmark."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM profiles WHERE benchmark_id = ? "
            "ORDER BY is_preferred DESC, profile_id LIMIT 1",
            (benchmark_id,),
        ).fetchone()
        if row is None:
            raise BenchmarkNotFoundError(
                f"No profiles for benchmark id: {benchmark_id!r}"
            )
        return dict(row)

    # -- vectors (for plotting / UQ) ----------------------------------------
    def get_profile_vector(
        self, profile_id: int, zaid: Optional[int] = None, mt: Optional[int] = None
    ) -> dict:
        """
        Return a profile's full per-group sensitivity vectors.

        The returned dict matches the app's ``SDFSensitivityData`` shape so the
        existing SDF plotting components can render it directly::

            {"pert_energies": [...], "reactions": [
                {"zaid", "mt", "nuclide", "reaction_name", "sensitivity": [...],
                 "error": [...] | None}, ...]}
        """
        conn = self._get_connection()
        prof = conn.execute(
            "SELECT ngroups, pert_energies FROM profiles WHERE profile_id = ?",
            (profile_id,),
        ).fetchone()
        if prof is None:
            raise BenchmarkNotFoundError(f"Unknown profile_id: {profile_id}")
        ngroups = prof["ngroups"]
        pert_energies = unpack_f32(prof["pert_energies"], ngroups + 1).tolist()

        # Only the system-total profiles (unit, region) == (0, 0); per-mixture
        # sub-profiles (when ingested) are excluded from the plotting vector.
        conditions = ["profile_id = ?", "unit = 0", "region = 0"]
        params = [profile_id]
        if zaid is not None:
            conditions.append("zaid = ?")
            params.append(zaid)
        if mt is not None:
            conditions.append("mt = ?")
            params.append(mt)
        query = (
            "SELECT zaid, mt, nuclide, reaction, sensitivity, error "
            f"FROM profile_sensitivities WHERE {' AND '.join(conditions)} "
            "ORDER BY zaid, mt"
        )
        reactions = []
        for r in conn.execute(query, params):
            reactions.append(
                {
                    "zaid": r["zaid"],
                    "mt": r["mt"],
                    "nuclide": r["nuclide"],
                    "reaction_name": r["reaction"],
                    "sensitivity": unpack_f32(r["sensitivity"], ngroups).tolist(),
                    "error": (
                        unpack_f32(r["error"], ngroups).tolist()
                        if r["error"] is not None
                        else None
                    ),
                }
            )
        return {"pert_energies": pert_energies, "reactions": reactions}

    # -- screening -----------------------------------------------------------
    def screen(
        self,
        zaid: int,
        mt: Optional[int] = None,
        region: str = "total",
        sensitivity_threshold: float = 0.0,
        fraction_threshold: Optional[float] = None,
        preferred_only: bool = True,
        limit: int = 100,
    ) -> List[dict]:
        """
        Return benchmark reactions sensitive to ``(zaid, mt)`` in an energy region,
        ranked by absolute integral sensitivity (descending).
        """
        if region not in _REGION_COLUMN:
            raise ValueError(
                f"Invalid region {region!r}; expected one of {tuple(_REGION_COLUMN)}"
            )
        col = _REGION_COLUMN[region]
        conn = self._get_connection()

        # Screen only against system-total profiles; per-mixture sub-profiles
        # (when ingested) never participate in screening.
        conditions = ["ps.zaid = ?", "ps.unit = 0", "ps.region = 0"]
        params: list = [zaid]
        if mt is not None:
            conditions.append("ps.mt = ?")
            params.append(mt)
        if preferred_only:
            conditions.append("p.is_preferred = 1")
        conditions.append(f"ABS(ps.{col}) >= ?")
        params.append(sensitivity_threshold)
        if fraction_threshold is not None:
            conditions.append(
                f"ps.s_abs_total > 0 AND ABS(ps.{col}) / ps.s_abs_total >= ?"
            )
            params.append(fraction_threshold)

        query = (
            "SELECT b.benchmark_id, b.category, b.spectrum, b.btype, "
            "p.profile_id, p.code, p.library, p.group_structure, p.keff, "
            "ps.zaid, ps.mt, ps.nuclide, ps.reaction, "
            f"ps.{col} AS s_region, ps.s_total, ps.s_abs_total "
            "FROM profile_sensitivities ps "
            "JOIN profiles p ON p.profile_id = ps.profile_id "
            "JOIN benchmarks b ON b.benchmark_id = p.benchmark_id "
            f"WHERE {' AND '.join(conditions)} "
            f"ORDER BY ABS(ps.{col}) DESC LIMIT ?"
        )
        params.append(limit)

        results = []
        for r in conn.execute(query, params):
            d = dict(r)
            d["fraction"] = (
                abs(d["s_region"]) / d["s_abs_total"] if d["s_abs_total"] else 0.0
            )
            results.append(d)
        return results

    # -- auxiliary per-benchmark data ----------------------------------------
    def get_experimental_keff(self, benchmark_id: str) -> dict:
        """Return ``{'keff', 'keff_unc'}`` (the ICSBEP benchmark value), or empty dict."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT keff_exp, keff_exp_unc FROM benchmarks WHERE benchmark_id = ?",
            (benchmark_id,),
        ).fetchone()
        if row is None:
            raise BenchmarkNotFoundError(f"Unknown benchmark id: {benchmark_id!r}")
        if row["keff_exp"] is None:
            return {}
        return {"keff": row["keff_exp"], "keff_unc": row["keff_exp_unc"]}

    def get_spectrum(self, benchmark_id: str) -> Optional[dict]:
        """Return the 299-group flux spectrum ``{'energies', 'flux'}`` or None."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT ngroups, energies, flux FROM benchmark_spectra WHERE benchmark_id = ?",
            (benchmark_id,),
        ).fetchone()
        if row is None:
            return None
        n = row["ngroups"]
        return {
            "energies": unpack_f32(row["energies"], n).tolist(),
            "flux": unpack_f32(row["flux"], n).tolist(),
        }

    def get_balance(self, benchmark_id: str) -> Optional[dict]:
        """Return the reaction-rate balance ``{'summary', 'text'}`` or None.

        ``summary`` holds the extracted scalars (keff, leakage, n_zones);
        ``text`` is the full balance file rendered verbatim.
        """
        conn = self._get_connection()
        row = conn.execute(
            "SELECT summary, raw FROM benchmark_balance WHERE benchmark_id = ?",
            (benchmark_id,),
        ).fetchone()
        if row is None:
            return None
        import gzip

        text = gzip.decompress(row["raw"]).decode("utf-8", errors="replace")
        return {"summary": json.loads(row["summary"]), "text": text}

    def get_input_deck(
        self, benchmark_id: str, code: Optional[str] = None
    ) -> List[dict]:
        """Return the raw input deck(s) as ``[{'code', 'text'}]`` (verbatim).

        Pass ``code`` (``'KENO'`` / ``'MCNP'``) to fetch a single deck; otherwise all
        decks stored for the benchmark are returned.
        """
        import gzip

        conn = self._get_connection()
        if code is not None:
            rows = conn.execute(
                "SELECT code, deck FROM benchmark_inputs "
                "WHERE benchmark_id = ? AND code = ?",
                (benchmark_id, code),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT code, deck FROM benchmark_inputs WHERE benchmark_id = ? "
                "ORDER BY code",
                (benchmark_id,),
            ).fetchall()
        return [
            {
                "code": r["code"],
                "text": gzip.decompress(r["deck"]).decode("utf-8", errors="replace"),
            }
            for r in rows
        ]

    def get_benchmark_dashboard(self, benchmark_id: str) -> dict:
        """Return the whole per-benchmark record in one bundle (for the app dashboard).

        Combines metadata, experimental keff, the profile-variant list, the preferred
        profile's sensitivity vectors, the flux spectrum, the balance, and the list of
        available input-deck codes.
        """
        result = self.get_benchmark(benchmark_id)  # metadata + keff_exp + profiles
        preferred = next(
            (p for p in result["profiles"] if p.get("is_preferred")),
            result["profiles"][0] if result["profiles"] else None,
        )
        result["experimental_keff"] = self.get_experimental_keff(benchmark_id)
        result["preferred_profile_id"] = preferred["profile_id"] if preferred else None
        result["profile_vector"] = (
            self.get_profile_vector(preferred["profile_id"]) if preferred else None
        )
        result["spectrum"] = self.get_spectrum(benchmark_id)
        result["balance"] = self.get_balance(benchmark_id)
        result["input_codes"] = [
            r["code"]
            for r in self._get_connection().execute(
                "SELECT code FROM benchmark_inputs WHERE benchmark_id = ? ORDER BY code",
                (benchmark_id,),
            )
        ]
        return result
