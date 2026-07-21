from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "input_dir": "/model/test",
        "output_file": "/result/submission.csv",
        "status_file": "/result/run_status.json",
        "progress_program": "/benchmark_lib/progress",
        "prompt_dir": "/workspace/prompts",
    },
    "models": {
        "root": "/opt/models",
        "generator": "/opt/models/generator",
        "thai_guard": "/opt/models/thai_guard",
    },
    "generation": {
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.76,
        "max_num_seqs": 8,
        "temperature": 0.2,
        "top_p": 0.9,
        "seed": 42,
    },
    "guards": {
        "thai_device": "cuda",
        "thai_batch_size": 8,
        "thai_harmful_threshold": 0.50,
        "thai_max_length": 128,
    },
    "limits": {
        "safe_direct_tokens": 384,
        "safe_sensitive_tokens": 224,
        "unsafe_tokens": 128,
        "rewrite_tokens": 224,
        "max_response_chars": 5000,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    import yaml

    config = DEFAULT_CONFIG
    config_path = Path(path)

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        config = _deep_merge(config, loaded)

    model_root = os.environ.get("MODEL_ROOT")
    if model_root:
        config["models"]["root"] = model_root
        config["models"]["generator"] = str(Path(model_root) / "generator")
        config["models"]["thai_guard"] = str(Path(model_root) / "thai_guard")

    config["paths"]["input_dir"] = os.environ.get(
        "INPUT_DIR", config["paths"]["input_dir"]
    )
    config["paths"]["output_file"] = os.environ.get(
        "OUTPUT_FILE", config["paths"]["output_file"]
    )
    config["paths"]["status_file"] = os.environ.get(
        "STATUS_FILE", config["paths"]["status_file"]
    )
    config["paths"]["progress_program"] = os.environ.get(
        "PROGRESS_PROGRAM", config["paths"]["progress_program"]
    )

    return config


def read_prompt(config: dict[str, Any], name: str) -> str:
    prompt_path = Path(config["paths"]["prompt_dir"]) / name
    return prompt_path.read_text(encoding="utf-8").strip()
