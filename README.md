# MCNPy

[![Version](https://img.shields.io/badge/version-0.1.1-blue.svg)](https://github.com/monleon96/MCNPy)

MCNPy is a Python package for working with MCNP input and output files. It serves as a lightweight alternative to mcnptools, offering only the features I found essential for my work. More functionalities will be added over time as needed.

## Documentation

Documentation is available at [MCNPy Documentation](https://mcnpy.readthedocs.io/en/latest/#).

*Note:* While this documentation covers the essentials, it may not be the most refined yet. I hope to enhance and expand it in the near future.

## Installation

```bash
pip install mcnpy
```

## Quick Start

```python
import mcnpy

# Read a mctal (.m) file
mctal = mcnpy.read_mctal("path/to/your/mctal")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details <https://www.gnu.org/licenses/>.
