from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

from app.config import load_config
from app.io_csv import find_input_csv, read_queries, write_submission
from app.normalization import normalize_text
from app.policies.fallback import fallback_for_route
from app.policies.rule_guard import inspect_query
from app.progress import report_progress
from app.risk_router import RouteName


def log(message: str) -> None:
    print(f"[llm-trust] {message}", flush=True)


def build_emergency_responses(records: list[dict]) -> list[str]:
    responses: list[str] = []
    for record in records:
        rule_result = inspect_query(normalize_text(record["query"]))
        if rule_result.severity == "unsafe":
            route_name = RouteName.UNSAFE
        elif rule_result.severity == "controversial":
            route_name = RouteName.SAFE_SENSITIVE
        else:
            route_name = RouteName.SAFE_DIRECT
        responses.append(
            fallback_for_route(route_name, original_query=record["query"])
        )
    return responses


def write_run_status(config: dict, status: dict) -> None:
    status_path = Path(config["paths"]["status_file"])
    status_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = status_path.with_suffix(status_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(status_path)


def main() -> None:
    sleep_seconds = float(os.environ.get("STARTUP_SLEEP_SECONDS", "10"))
    log(f"startup sleep {sleep_seconds:g}s")
    time.sleep(sleep_seconds)

    config_path = os.environ.get("CONFIG_PATH", "/workspace/configs/production.yaml")
    log(f"loading config: {config_path}")
    config = load_config(config_path)

    log(f"finding input csv under: {config['paths']['input_dir']}")
    input_path = find_input_csv(config["paths"]["input_dir"])
    log(f"reading input csv: {input_path}")
    records = read_queries(input_path)
    log(f"loaded records: {len(records)}")

    try:
        log("starting model pipeline")
        from app.pipeline import TrustworthinessPipeline

        pipeline = TrustworthinessPipeline(config)
        responses = pipeline.process(records)
        run_status = {
            "status": "model_pipeline_completed",
            "diagnostics": pipeline.diagnostics,
        }
        log("model pipeline completed")
    except Exception as exc:
        log("model pipeline failed; using emergency fallback responses")
        traceback.print_exc()
        responses = build_emergency_responses(records)
        run_status = {
            "status": "emergency_fallback",
            "error": str(exc),
            "records": len(records),
        }

    log(f"writing submission: {config['paths']['output_file']}")
    write_submission(
        records=records,
        responses=responses,
        output_path=config["paths"]["output_file"],
    )
    log("submission written")

    write_run_status(config, run_status)
    log(f"run status written: {config['paths']['status_file']}")

    report_progress(
        executable=config["paths"]["progress_program"],
        completed=len(records),
    )
    log("progress reported")


if __name__ == "__main__":
    main()
