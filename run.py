from __future__ import annotations

import os
import time
import traceback

from app.config import load_config
from app.io_csv import find_input_csv, read_queries, write_submission
from app.normalization import normalize_text
from app.policies.fallback import (
    GENERAL_ERROR_FALLBACK,
    SAFE_SENSITIVE_FALLBACK,
    UNSAFE_FALLBACK,
)
from app.policies.rule_guard import inspect_query
from app.progress import report_progress


def log(message: str) -> None:
    print(f"[llm-trust] {message}", flush=True)


def build_emergency_responses(records: list[dict]) -> list[str]:
    responses: list[str] = []
    for record in records:
        rule_result = inspect_query(normalize_text(record["query"]))
        if rule_result.severity == "unsafe":
            responses.append(UNSAFE_FALLBACK)
        elif rule_result.severity == "controversial":
            responses.append(SAFE_SENSITIVE_FALLBACK)
        else:
            responses.append(GENERAL_ERROR_FALLBACK)
    return responses


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
        log("model pipeline completed")
    except Exception:
        log("model pipeline failed; using emergency fallback responses")
        traceback.print_exc()
        responses = build_emergency_responses(records)

    log(f"writing submission: {config['paths']['output_file']}")
    write_submission(
        records=records,
        responses=responses,
        output_path=config["paths"]["output_file"],
    )
    log("submission written")

    report_progress(
        executable=config["paths"]["progress_program"],
        completed=len(records),
    )
    log("progress reported")


if __name__ == "__main__":
    main()
