from __future__ import annotations

import json
import os
import time
from pathlib import Path

from app.config import load_config
from app.io_csv import (
    find_input_csv,
    read_queries,
    validate_submission,
    write_submission,
)
from app.progress import report_progress


def log(message: str) -> None:
    print(f"[llm-trust] {message}", flush=True)


def write_run_status(config: dict, status: dict) -> None:
    status_path = Path(config["paths"]["status_file"])
    status_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = status_path.with_suffix(status_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(status_path)


def try_write_run_status(config: dict, status: dict) -> bool:
    """Keep optional diagnostics from blocking the evaluator handshake."""
    try:
        write_run_status(config, status)
    except Exception as exc:
        log(f"warning: could not write run status: {exc}")
        return False
    log(f"run status written: {config['paths']['status_file']}")
    return True


def remove_stale_artifacts(config: dict) -> None:
    """A failed model run must not leave evaluator-facing artifacts behind."""
    for key in ("output_file", "status_file"):
        path = Path(config["paths"][key])
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)


def main() -> None:
    started_at = time.monotonic()
    sleep_seconds = max(0.0, float(os.environ.get("STARTUP_SLEEP_SECONDS", "10")))
    if sleep_seconds:
        log(f"startup sleep {sleep_seconds:g}s")
        time.sleep(sleep_seconds)

    config_path = os.environ.get("CONFIG_PATH", "/workspace/configs/production.yaml")
    log(f"loading config: {config_path}")
    config = load_config(config_path)
    remove_stale_artifacts(config)

    log(f"finding input csv under: {config['paths']['input_dir']}")
    input_path = find_input_csv(config["paths"]["input_dir"])
    log(f"reading input csv: {input_path}")
    records = read_queries(input_path)
    log(f"loaded records: {len(records)}")

    total_records = len(records)

    log("starting model pipeline")
    from app.pipeline import TrustworthinessPipeline

    pipeline = TrustworthinessPipeline(config)
    responses = pipeline.process(records)
    run_status = {
        "status": "model_pipeline_completed",
        "diagnostics": pipeline.diagnostics,
    }
    log("model pipeline completed")

    log(f"writing submission: {config['paths']['output_file']}")
    write_submission(
        records=records,
        responses=responses,
        output_path=config["paths"]["output_file"],
    )
    written_records = validate_submission(
        records=records,
        output_path=config["paths"]["output_file"],
    )
    log(f"submission written and validated: {written_records} rows")

    run_status["records"] = total_records
    run_status["elapsed_seconds"] = round(time.monotonic() - started_at, 3)

    try_write_run_status(config, run_status)

    try:
        report_progress(
            executable=config["paths"]["progress_program"],
            completed=total_records,
            timeout_seconds=30.0,
        )
    except Exception as exc:
        run_status["progress_error"] = str(exc)
        run_status["elapsed_seconds"] = round(time.monotonic() - started_at, 3)
        try_write_run_status(config, run_status)
        raise

    run_status["progress_completed"] = total_records
    run_status["elapsed_seconds"] = round(time.monotonic() - started_at, 3)
    try_write_run_status(config, run_status)
    log(f"final progress reported: {total_records}/{total_records}")

if __name__ == "__main__":
    main()
