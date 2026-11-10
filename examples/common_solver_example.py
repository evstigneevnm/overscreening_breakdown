"""Compatibility entry point for explicit projection and typed restarts.

Use ``python -m examples.restart_and_projection`` for the current example.
The existing ``run(directory)`` return pair remains available.
"""

if __package__:
    from .restart_and_projection import main, run
else:
    from restart_and_projection import main, run


if __name__ == "__main__":
    main()
