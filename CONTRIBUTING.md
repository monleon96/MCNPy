# Contributing to KIKA

Thank you for your interest in contributing to KIKA! This guide covers everything you need to get started.

## Reporting Bugs & Suggesting Features

Open a [GitHub issue](https://github.com/monleon96/kika/issues) with a clear description. For bugs, include:

- Steps to reproduce
- Expected vs. actual behavior
- Python version and OS

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/<your-username>/kika.git
   cd kika
   ```

2. Install dependencies with Poetry:
   ```bash
   poetry install --with dev
   ```

3. Copy the environment sample and fill in any required values:
   ```bash
   cp .env.sample .env
   ```

## Running Tests

```bash
poetry run pytest
```

## Code Style

Follow the project's Python style guide at [`.github/python-style.instructions.md`](.github/python-style.instructions.md).

## Submitting Changes

1. Create a branch from `develop`:
   ```bash
   git checkout develop
   git checkout -b your-feature-branch
   ```

2. Make your changes and ensure tests pass.

3. Open a Pull Request against the `develop` branch.
