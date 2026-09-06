"""Command-line helpers that are part of the package but not part of its API.

One module per job, each runnable as ``python -m kika.scripts.<name>``. They may
import from :mod:`kika`; nothing in :mod:`kika` may import from here.

This file exists so the directory is a real package rather than an implicit
namespace one: every other subpackage of kika is regular, and a namespace
package here would behave differently under a zip import or a partial install
for no gain.
"""
