# Install and build

The validated platform is **Linux with Python 3.9–3.13**. Windows and other
operating systems are currently unsupported.

Run commands from the project root. Activate your Python or Anaconda environment
first, or create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install

Install the validated scientific and notebook dependencies, then the project:

```bash
python -m pip install -r requirements.txt
python -m pip install .
```

For a minimal runtime installation, use only `python -m pip install .`.
Optional extras are available with `python -m pip install ".[gmpy,notebook]"`.

Try an example from the checkout:

```bash
python -m examples.bsk --solver both --backend both
```

## Development

Install the project in editable mode with its dependencies and development tools:

```bash
python -m pip install -r requirements-dev.txt
MPLBACKEND=Agg python -m pytest
```

## Build distributions

```bash
python -m pip install "build>=1.2,<2"
python -m build
```

This creates a source archive (`.tar.gz`) and wheel (`.whl`) in `dist/`.
With dependencies already installed, install the built wheel with:

```bash
python -m pip install --force-reinstall --no-deps dist/overscreening_breakdown-0.1.0-py3-none-any.whl
```

Force reinstalling replaces earlier local builds with the same version.
Examples and notebooks are included in the checkout and source archive; the
wheel contains the runtime package.

See [detailed installation instructions](docs/installation.md) and the
[public API migration guide](docs/public_api.md) for further information.
