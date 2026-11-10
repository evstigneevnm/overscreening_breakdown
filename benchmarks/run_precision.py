"""Opt-in arithmetic study at a fixed case, N, and quadrature geometry."""

from benchmarks.run_manufactured import main as run


def main(argv=None):
    return run(argv, precision_study=True)


if __name__ == "__main__":
    raise SystemExit(main())
