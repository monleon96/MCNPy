"""Custom exceptions for WWINP file processing."""


class WWINPFormatError(Exception):
    """Raised when the WWINP file structure is invalid."""

    def __init__(self, message: str, line: int | None = None):
        if line is not None:
            message = f"Line {line}: {message}"
        super().__init__(message)
        self.line = line


class WWINPParsingError(Exception):
    """Raised when an error occurs during WWINP file parsing."""

    def __init__(self, message: str, line: int | None = None):
        if line is not None:
            message = f"Line {line}: {message}"
        super().__init__(message)
        self.line = line
