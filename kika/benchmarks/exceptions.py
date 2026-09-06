"""Exceptions for the benchmarks subpackage."""


class BenchmarksError(Exception):
    """Base class for all benchmarks-subpackage errors."""


class DatabaseNotConfiguredError(BenchmarksError):
    """Raised when no benchmarks database path is configured or the file is missing."""


class BenchmarkNotFoundError(BenchmarksError):
    """Raised when a requested benchmark id or profile is not present in the database."""
