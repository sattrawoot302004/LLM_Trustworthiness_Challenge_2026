from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any


Record = dict[str, Any]


def find_input_csv(input_dir: str | os.PathLike[str]) -> Path:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")

    candidates = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".csv"
    )
    for path in candidates:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = set(reader.fieldnames or [])
            if {"id", "query"}.issubset(fieldnames):
                return path
        except UnicodeDecodeError:
            continue

    raise FileNotFoundError(
        f"No CSV with required columns id,query found under {root}"
    )


def read_queries(path: str | os.PathLike[str]) -> list[Record]:
    records: list[Record] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"id", "query"}.issubset(reader.fieldnames):
            raise ValueError("Input CSV must contain id and query columns")

        for index, row in enumerate(reader):
            record_id = str(row.get("id", "")).strip()
            query = str(row.get("query", "")).strip()
            if not record_id:
                raise ValueError(f"Missing id at row {index + 2}")
            records.append(
                {
                    "id": record_id,
                    "query": query,
                    "original_index": index,
                }
            )

    return records


def write_submission(
    records: list[Record],
    responses: list[str],
    output_path: str | os.PathLike[str],
) -> None:
    if len(records) != len(responses):
        raise ValueError(
            f"Record/response count mismatch: {len(records)} != {len(responses)}"
        )

    final_path = Path(output_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "response"],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for record, response in zip(records, responses, strict=True):
            if not str(response).strip():
                raise ValueError(f"Empty response for id={record['id']}")
            writer.writerow({"id": record["id"], "response": response})

        handle.flush()
        os.fsync(handle.fileno())

    tmp_path.replace(final_path)
