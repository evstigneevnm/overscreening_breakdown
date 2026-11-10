"""Use a headless backend before pytest imports any numerical test modules."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
