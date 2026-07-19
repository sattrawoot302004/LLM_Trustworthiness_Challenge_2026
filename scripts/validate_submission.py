from __future__ import annotations

import csv
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python scripts/validate_submission.py INPUT_CSV SUBMISSION_CSV")
        return 2

    input_path = Path(sys.argv[1])
    submission_path = Path(sys.argv[2])

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        input_rows = list(csv.DictReader(handle))
    with submission_path.open("r", encoding="utf-8-sig", newline="") as handle:
        output_rows = list(csv.DictReader(handle))

    if len(input_rows) != len(output_rows):
        raise ValueError(f"row count mismatch: {len(input_rows)} != {len(output_rows)}")

    for index, (input_row, output_row) in enumerate(
        zip(input_rows, output_rows, strict=True),
        start=2,
    ):
        if input_row.get("id") != output_row.get("id"):
            raise ValueError(f"id mismatch at row {index}")
        if not (output_row.get("response") or "").strip():
            raise ValueError(f"empty response at row {index}")

    print(f"valid submission: {len(output_rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
