"""Constants for the benchmarks (ICSBEP/DICE) subpackage."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Database location / configuration
# ---------------------------------------------------------------------------
# Environment variable users can set to point at their built benchmarks DB.
DB_PATH_ENV_VAR = "KIKA_BENCHMARKS_DB_PATH"

# Environment variable pointing at the raw DICE ``sensitivity/`` folder used by
# the ingest step (only needed when building the database).
SOURCE_DIR_ENV_VAR = "KIKA_DICE_SOURCE_DIR"

# Default self-contained database location (mirrors the ~/.kika convention used
# by kika/endf/remote/constants.py). The database is built once from the user's
# own NEA-licensed DICE data; kika ships no benchmark data itself.
DEFAULT_DB_DIR = Path.home() / ".kika" / "benchmarks"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "kika_benchmarks.db"

# Bump when the on-disk schema changes in a backward-incompatible way.
# v3: SDF errors and response uncertainty are absolute standard deviations;
#     profiles.keff_unc replaces the misleading keff_rel_err name; repeated
#     occurrences are resolved by an explicit ingest policy.
SCHEMA_VERSION = 3

# ---------------------------------------------------------------------------
# Coarse energy regions (MeV, ascending). Standard criticality split points:
#   thermal    : E <= 0.625 eV
#   epithermal : 0.625 eV < E <= 100 keV
#   fast       : E > 100 keV
# A group is assigned to a region by its UPPER energy edge (documented choice;
# boundary-straddling groups are rare on the 238-group DICE grid).
# ---------------------------------------------------------------------------
THERMAL_MAX_MEV = 0.625e-6
EPITHERMAL_MAX_MEV = 0.1

REGIONS = ("thermal", "epithermal", "fast", "total")

# DICE benchmark categories (top-level folders under sensitivity/).
DICE_CATEGORIES = ("HEU", "IEU", "LEU", "MIX", "PU", "SPEC", "U233")

# ---------------------------------------------------------------------------
# Preferred-variant selection. A single physical benchmark case may have several
# sensitivity profiles (different transport code, nuclear-data library, group
# structure). One variant is flagged ``is_preferred`` per benchmark so that
# screening returns each benchmark once by default. Lower index = preferred.
# ---------------------------------------------------------------------------
CODE_PRIORITY = ["KENO", "MCNP"]
LIBRARY_PRIORITY = [
    "ENDF-B-VIII.1",
    "ENDF-B-VIII.0",
    "ENDF-B-VII.1",
    "ENDF-B-VII.0",
    "ABBN-93",
]
GROUP_PRIORITY = ["238-Group", "299-Group", "continuous"]

# ---------------------------------------------------------------------------
# Reaction-name aliases -> ENDF MT, so the screening API accepts human names in
# addition to MT integers and the canonical kika MT_TO_REACTION labels.
# ---------------------------------------------------------------------------
REACTION_ALIASES = {
    "total": 1,
    "elastic": 2,
    "el": 2,
    "inelastic": 4,
    "nonelastic": 3,
    "n2n": 16,
    "fission": 18,
    "fiss": 18,
    "capture": 102,
    "ngamma": 102,
    "n,gamma": 102,
    "absorption": 101,
    "disappearance": 101,
    "nubar": 452,
    "chi": 1018,
}
