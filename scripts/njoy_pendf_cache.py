"""On-disk cache for NJOY-reconstructed PENDF / MT→σ(E) maps.

NJOY RECONR runs are slow (10–60 s for typical evaluations). The v3 pipeline
(``exfor_to_endf_sampling_v3.py``) and the v3 notebooks under
``myworkspace/JEFF/`` all need the same xs_map (Dict[int, MF3MT|CrossSection])
to drive the MF33 propagation channel, so we cache the result of the first
reconstruction on disk and reuse it on subsequent runs.

Cache key is ``{endf_stem}__tol{tolerance}.pkl`` — simple and human-readable.
**If you modify the input ENDF in place, delete the matching .pkl file by
hand**; the cache does not check mtime/checksum to keep the key transparent.

Format: pickle of the xs_map dict only. The full parsed ``Endf`` object (with
MF33 etc.) is reloaded fresh on every call via ``read_endf`` — that's fast
enough that caching it isn't worth the fragility cost.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

# Make sure kika is importable when callers don't add the repo root to
# sys.path themselves.
_kika_path = Path(__file__).resolve().parent.parent
if str(_kika_path) not in sys.path:
    sys.path.insert(0, str(_kika_path))

from kika.endf.read_endf import read_endf
from kika.processing.derived_covariance import _build_xs_map


DEFAULT_CACHE_DIR = Path("/share_snc/snc/JuanMonleon/cache/njoy_pendf")


def get_or_reconstruct_xs_map(
    endf_path: str | Path,
    njoy_executable: Optional[str] = "/soft_snc/NJOY/2016.78/bin/njoy",
    tolerance: float = 0.001,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    *,
    force_refresh: bool = False,
    verbose: bool = True,
) -> Tuple[object, Dict[int, object], str]:
    """Return ``(endf, xs_map, source)``.

    On a cache hit, ``source == 'cache'`` and the pickled xs_map is returned
    directly. On a miss, NJOY (or kika's in-Python reconstructor when
    ``njoy_executable=None``) is run and the result is pickled before
    returning. ``force_refresh=True`` bypasses the cache and re-runs NJOY.

    Parameters
    ----------
    endf_path
        Input ENDF file (typically with MF2 resonance parameters).
    njoy_executable
        Path to the NJOY binary. Defaults to the cluster install. Pass
        ``None`` to use kika's pure-Python reconstructor (slower but no
        external dependency).
    tolerance
        Reconstruction tolerance passed to NJOY RECONR.
    cache_dir
        Directory holding the .pkl files. Created if missing.
    force_refresh
        If True, ignore any cached file and re-run reconstruction.
    verbose
        Print where the result came from (cache hit / miss + path + timing).

    Returns
    -------
    endf : Endf
        Parsed ENDF object (always re-read from ``endf_path`` — used for
        MF33 etc. by the caller).
    xs_map : Dict[int, MF3MT | CrossSection]
        MT → cross-section source. Pass to ``MF33MT.to_xs_covmat(mf3_sections=...)``
        and ``project_to_grid(xs_source=xs_map[mt], ...)``.
    source : {'cache', 'fresh'}
        Where the xs_map came from this call.
    """
    endf_path = Path(endf_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{endf_path.stem}__tol{tolerance}.pkl"

    endf = read_endf(str(endf_path))

    if cache_path.exists() and not force_refresh:
        t0 = time.time()
        with open(cache_path, "rb") as f:
            xs_map = pickle.load(f)
        if verbose:
            n_mts = len(xs_map)
            print(f"[NJOY cache] source=cache  ({time.time()-t0:.2f}s)  "
                  f"{cache_path}  ({n_mts} MTs)")
        return endf, xs_map, "cache"

    if verbose:
        what = ("NJOY RECONR" if njoy_executable
                else "kika in-Python reconstructor")
        print(f"[NJOY cache] miss — running {what} (this may take 10–60 s)…")
    t0 = time.time()
    xs_map = _build_xs_map(
        endf,
        njoy_executable=njoy_executable,
        tolerance=tolerance,
        endf_path=endf_path,
    )
    elapsed = time.time() - t0

    with open(cache_path, "wb") as f:
        pickle.dump(xs_map, f)
    if verbose:
        n_mts = len(xs_map)
        print(f"[NJOY cache] source=fresh  ({elapsed:.1f}s)  "
              f"saved to {cache_path}  ({n_mts} MTs)")

    return endf, xs_map, "fresh"
