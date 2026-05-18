from importlib.metadata import PackageNotFoundError, version

try:
    LIBRARY_VERSION = version("kika-nd")
except PackageNotFoundError:
    LIBRARY_VERSION = "0.0.0+local"

AUTHOR = "Juan Antonio Monleon de la Lluvia <juanjuanmonleon@gmail.com>"
