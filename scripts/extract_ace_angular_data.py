"""
Extract elastic angular distribution PDFs from ACE sample files to parquet.

Usage:
    Edit SETS_TO_PROCESS to select which set(s) to extract, then run:
        python3 scripts/extract_ace_angular_data.py

One set at a time is recommended to keep memory low (~40 MB peak per ACE file).
"""

import gc
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── Add project root to path ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kika.ace.parsers import read_ace

# ── Configuration ─────────────────────────────────────────────────────────

N_MU_POINTS = 200
MT_NUMBER = 2
N_SAMPLES = 100  # first 100 samples per set (1-based: 1..100)
N_PROCS = 1  # number of parallel workers (set >1 for multiprocessing)

SETS = {
    "jeff40": (
        "/SCRATCH/users/monleon-de-la-jan/MCNPy_LIB/JEFF40_LEG_ALL"
        "/ace/293.6/26056/{s}/260560_40_{s}.02c"
    ),
    "jendl5": (
        "/SCRATCH/users/monleon-de-la-jan/MCNPy_LIB/JENDL5_LEG"
        "/ace/293.6/26056/{s}/260560_50_{s}.02c"
    ),
    "this_work_orig": (
        "/SCRATCH/users/monleon-de-la-jan/MCNPy_LIB/EXFOR_FIT_500_SAMPLES"
        "/ace/293.6/26056/{s}/260560_40_{s}.02c"
    ),
    "this_work_new": (
        "/SCRATCH/users/monleon-de-la-jan/MCNPy_LIB/NEW_EXFOR_FIT"
        "/ace/293.6/26056/{s}/260560_40_{s}.02c"
    ),
}

# Uncomment the set(s) you want to process
SETS_TO_PROCESS = ["jeff40"]
# SETS_TO_PROCESS = ["jendl5"]
# SETS_TO_PROCESS = ["this_work_orig"]
# SETS_TO_PROCESS = ["this_work_new"]
# SETS_TO_PROCESS = list(SETS.keys())  # all at once

OUTPUT_DIR = Path("/share_snc/snc/JuanMonleon/EXFOR/ace_angular_parquet")


# ── Core extraction ──────────────────────────────────────────────────────

def extract_one_ace(ace_path: str, mu_grid: np.ndarray) -> tuple:
    """
    Extract elastic angular PDFs from a single ACE file.

    Parameters
    ----------
    ace_path : str
        Path to the ACE file.
    mu_grid : np.ndarray
        Cosine grid (unused directly; num_points controls the grid inside to_dataframe).

    Returns
    -------
    energies : list[float]
        Energy grid in MeV.
    pdf_matrix : np.ndarray
        Shape (n_energies, n_mu). PDF values at each (energy, mu) point.
    """
    ace = read_ace(ace_path)

    # Get elastic energy grid
    energies = ace.angular_distributions.elastic.energies  # list[float]
    n_e = len(energies)
    n_mu = len(mu_grid)

    pdf_matrix = np.empty((n_e, n_mu), dtype=np.float32)

    for i, E in enumerate(energies):
        df = ace.angular_distributions.to_dataframe(
            mt=MT_NUMBER,
            energy=E,
            interpolate=True,
            num_points=n_mu,
        )
        if df is not None and "pdf" in df.columns:
            pdf_matrix[i, :] = df["pdf"].values.astype(np.float32)
        else:
            pdf_matrix[i, :] = 0.5  # isotropic fallback

    # Free memory
    del ace
    gc.collect()

    return energies, pdf_matrix


def _process_one_sample(args: tuple) -> tuple:
    """
    Worker function for parallel extraction (top-level for pickling).

    Parameters
    ----------
    args : tuple
        (sample_i, template, mu_grid, set_name)

    Returns
    -------
    (sample_i, energies, pdf_matrix) on success
    (sample_i, None, error_msg) on failure
    """
    sample_i, template, mu_grid, set_name = args
    s = f"{sample_i:04d}"
    ace_path = template.format(s=s)

    if not Path(ace_path).exists():
        return (sample_i, None, f"missing {ace_path}")

    try:
        energies, pdf_matrix = extract_one_ace(ace_path, mu_grid)
        return (sample_i, energies, pdf_matrix)
    except Exception as exc:
        return (sample_i, None, str(exc))


def build_arrow_schema(n_mu: int) -> pa.Schema:
    """Build the pyarrow schema for the parquet file."""
    fields = [
        pa.field("sample_idx", pa.int32()),
        pa.field("energy_idx", pa.int32()),
        pa.field("energy_mev", pa.float64()),
    ]
    for j in range(n_mu):
        fields.append(pa.field(f"pdf_{j:03d}", pa.float32()))
    return pa.schema(fields)


def _write_sample_to_parquet(writer, sample_i, energies, pdf_matrix, pdf_cols,
                             schema):
    """Write one sample's data to the parquet writer."""
    n_e = len(energies)
    data = {
        "sample_idx": np.full(n_e, sample_i, dtype=np.int32),
        "energy_idx": np.arange(n_e, dtype=np.int32),
        "energy_mev": np.array(energies, dtype=np.float64),
    }
    for j in range(len(pdf_cols)):
        data[pdf_cols[j]] = pdf_matrix[:, j]

    table = pa.Table.from_pydict(data, schema=schema)
    writer.write_table(table)


def process_set(set_name: str):
    """Extract all samples for one set and write to parquet."""
    template = SETS[set_name]
    mu_grid = np.linspace(-1, 1, N_MU_POINTS)
    schema = build_arrow_schema(N_MU_POINTS)

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{set_name}_ace_angular.parquet"
    mu_path = output_dir / f"{set_name}_mu_grid.npy"

    # Save mu grid
    np.save(mu_path, mu_grid)

    pdf_cols = [f"pdf_{j:03d}" for j in range(N_MU_POINTS)]
    n_done = 0
    n_failed = 0
    t0 = time.time()

    writer = pq.ParquetWriter(str(parquet_path), schema, compression="snappy")
    try:
        if N_PROCS > 1:
            # ── Parallel mode ────────────────────────────────────────
            print(f"  [{set_name}] Using {N_PROCS} parallel workers")
            worker_args = [
                (sample_i, template, mu_grid, set_name)
                for sample_i in range(1, N_SAMPLES + 1)
            ]
            with multiprocessing.Pool(N_PROCS) as pool:
                for result in pool.imap_unordered(_process_one_sample,
                                                  worker_args):
                    sample_i = result[0]
                    if result[1] is None:
                        error_msg = result[2]
                        print(f"  [{set_name}] WARNING: sample {sample_i} "
                              f"failed: {error_msg}")
                        n_failed += 1
                        continue

                    energies, pdf_matrix = result[1], result[2]
                    _write_sample_to_parquet(writer, sample_i, energies,
                                            pdf_matrix, pdf_cols, schema)
                    n_done += 1
                    elapsed = time.time() - t0
                    print(
                        f"  [{set_name}] Sample {sample_i}/{N_SAMPLES} done "
                        f"({len(energies)} energies, {elapsed:.1f}s elapsed)"
                    )
        else:
            # ── Sequential mode ──────────────────────────────────────
            for sample_i in range(1, N_SAMPLES + 1):
                result = _process_one_sample(
                    (sample_i, template, mu_grid, set_name)
                )
                if result[1] is None:
                    error_msg = result[2]
                    print(f"  [{set_name}] WARNING: sample {sample_i} "
                          f"failed: {error_msg}")
                    n_failed += 1
                    continue

                energies, pdf_matrix = result[1], result[2]
                _write_sample_to_parquet(writer, sample_i, energies,
                                        pdf_matrix, pdf_cols, schema)
                n_done += 1
                elapsed = time.time() - t0
                print(
                    f"  [{set_name}] Sample {sample_i}/{N_SAMPLES} done "
                    f"({len(energies)} energies, {elapsed:.1f}s elapsed)"
                )

    finally:
        writer.close()

    print(
        f"\n[{set_name}] Finished: {n_done} samples written, "
        f"{n_failed} failed/missing"
    )
    print(f"  Parquet: {parquet_path}")
    print(f"  Mu grid: {mu_path}")


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for name in SETS_TO_PROCESS:
        if name not in SETS:
            print(f"Unknown set: {name}")
            continue
        print(f"\n{'='*60}")
        print(f"Processing set: {name}")
        print(f"{'='*60}")
        process_set(name)
