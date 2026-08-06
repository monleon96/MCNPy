---
name: "Python_style_guidelines"
description: "Python style, naming, docstrings and path handling"
applyTo: "**/*.py,**/*.pyi"
---

## Python version
- Target **Python 3.12** only.

## General style
- Follow **PEP 8** for formatting, naming, and imports.
- Prefer explicit, readable code over clever or compact code.
- Use type hints consistently in new or modified code.

## Naming
- Modules, packages, and directories: `lower_snake_case`.
- Classes: `CamelCase`.
- Functions and methods: `lower_snake_case`.
- Variables, parameters, and attributes: `lower_snake_case`.
- Module-level constants: `UPPER_SNAKE_CASE`.

### Exception — `kika/nuclear_data/model/` and the format adapters that name its nodes

Inside the canonical model, names are **GNDS's, verbatim**. Classes take the
GNDS node name with its first letter capitalised (`ReactionSuite`, `XYs1d`,
`Regions1d`, `BreitWigner`, `CovarianceSection`); attributes and child accessors
keep the specification's spelling exactly (`outerDomainValue`, `derivedFrom`,
`ENDF_MT`, `crossSection`, `outputChannel`, `rowData`, `domainValue`,
`domainMin`); enumerated values keep their hyphens (`'lin-lin'`,
`'charged-particle'`).

**Do not "fix" these to snake_case, and do not add a snake_case alias beside
one.** The GNDS reader and writer resolve nodes by introspection on these exact
names, so a rename produces a reader that silently stops finding a node, and an
alias forks the vocabulary so that introspection picks whichever it meets first.

Local variables and private helpers inside the model are still
`lower_snake_case`; only the node and attribute vocabulary is GNDS's. Everything
outside that directory is PEP 8 as above.

Enforced by `kika/nuclear_data/model/tests/test_gnds_naming.py`. Rationale and
the list of declared divergences: `kika/nuclear_data/model/NAMING.md`.

## Docstrings
- Use **numpydoc-style docstrings** for all public classes, functions, and methods.
- First line: short summary sentence.
- Optionally follow with a blank line and a longer description.
- When relevant, include sections in this order: `Parameters`, `Returns`, `Yields`, `Raises`, `Attributes`, `Examples`.
- Non-trivial private helpers must have at least a short docstring.

## Imports and module layout
- Order imports in groups, separated by a blank line:
  1. Standard library imports
  2. Third-party imports
  3. Local imports
- Avoid wildcard imports like `from module import *`.
- Use explicit, clear import names.

## Path handling
- Prefer `pathlib.Path` over `os.path`.
- Public APIs that accept paths should accept both `str` and `Path`.
- Convert incoming path arguments to `Path` objects at the start of the function.
- Use `Path` methods for filesystem operations (for example: `open`, `read_text`, `write_text`, `joinpath` or `/`).

