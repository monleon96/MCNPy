"""Constants for the SINBAD subpackage."""

from pathlib import Path

#: Environment variable pointing at a directory of SINBAD packages.
LIBRARY_PATH_ENV_VAR = "KIKA_SINBAD_PATH"

#: Default library location when nothing else is configured.
DEFAULT_LIBRARY_PATH = Path.home() / ".kika" / "sinbad"

#: Extension of the single-file distribution form of a package.
PACKAGE_SUFFIX = ".sinbad"

#: Filenames inside a package. The description is authoritative; the JSON is a
#: generated view of the same model, which is what makes it safe to read either.
CANONICAL_DESCRIPTION = "benchmark.xml"
GENERATED_DESCRIPTION = "benchmark.json"
MANIFEST = "manifest.xml"
