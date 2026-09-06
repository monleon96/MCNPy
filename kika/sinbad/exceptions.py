"""Exceptions raised by :mod:`kika.sinbad`."""


class SinbadError(Exception):
    """Base class for every error raised by the SINBAD subpackage."""


class PackageNotFoundError(SinbadError):
    """No SINBAD package could be found at the given path or identifier."""


class LibraryNotConfiguredError(SinbadError):
    """No package library has been configured and none was found by default."""


class ArrayBackendMissingError(SinbadError):
    """The package stores arrays in HDF5 but ``h5py`` is not installed.

    Scalar and descriptive content is always readable without it; only spectra,
    sensitivity coefficients and covariance matrices need an array backend.
    """
