from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any


Record = dict[str, Any]
INPUT_TEXT_COLUMNS = ("question", "query")


def _input_columns(fieldnames: list[str] | None) -> tuple[str, str] | None:
    """Return the physical id/text column names from a supported input CSV."""
    normalized = {
        str(name).strip().lower(): str(name)
        for name in (fieldnames or [])
        if name is not None
    }
    id_column = normalized.get("id")
    if id_column is None:
        return None
    for candidate in INPUT_TEXT_COLUMNS:
        text_column = normalized.get(candidate)
        if text_column is not None:
            return id_column, text_column
    return None


def find_input_csv(input_dir: str | os.PathLike[str]) -> Path:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")

    candidates = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".csv"
    )
    expected_path = root / "dataset.csv"
    if expected_path in candidates:
        candidates.remove(expected_path)
        candidates.insert(0, expected_path)
    for path in candidates:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = _input_columns(reader.fieldnames)
            if columns is not None:
                return path
        except UnicodeDecodeError:
            continue

    raise FileNotFoundError(
        f"No CSV with required columns id,question (or id,query) found under {root}"
    )


def read_queries(path: str | os.PathLike[str]) -> list[Record]:
    records: list[Record] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = _input_columns(reader.fieldnames)
        if columns is None:
            raise ValueError(
                "Input CSV must contain id and question columns "
                "(legacy query is also supported)"
            )
        id_column, text_column = columns

        for index, row in enumerate(reader):
            record_id = str(row.get(id_column, "")).strip()
            query = str(row.get(text_column, "")).strip()
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


def validate_submission(
    records: list[Record],
    output_path: str | os.PathLike[str],
) -> int:
    """Validate the exact evaluator contract after the atomic rename."""
    output_records: list[dict[str, str]] = []
    with Path(output_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "response"]:
            raise ValueError(
                "Submission columns must be exactly id,response in that order"
            )
        for row_number, row in enumerate(reader, start=2):
            record_id = str(row.get("id", ""))
            response = str(row.get("response", ""))
            if not response.strip():
                raise ValueError(f"Empty response in submission row {row_number}")
            output_records.append({"id": record_id, "response": response})

    expected_ids = [str(record["id"]) for record in records]
    actual_ids = [record["id"] for record in output_records]
    if actual_ids != expected_ids:
        raise ValueError("Submission ids/order do not match the input dataset")
    if len(output_records) != len(records):
        raise ValueError(
            f"Submission row count mismatch: {len(output_records)} != {len(records)}"
        )
    return len(output_records)
