"""Internal single-attempt subprocess protocol for the benchmark runner."""

import argparse
import json
from pathlib import Path
import sys

from benchmarks.run_manufactured import _execute_attempt, _export_attempt


def main(argv=None):
    request_path, response_path = list(sys.argv[1:] if argv is None else argv)
    request = json.loads(Path(request_path).read_text())
    options = argparse.Namespace(**request["options"])
    row = request["record"]
    row.update(_execute_attempt(options, request["selector"], request["N"],
                                request["precision"], request["order"],
                                expected_backend=row["mpmath_backend"]))
    row["export_seconds"] = _export_attempt(Path(request["attempt_path"]), row)
    # IPC serialization is intentionally outside the timed scientific export.
    Path(response_path).write_text(json.dumps(row, allow_nan=False)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
