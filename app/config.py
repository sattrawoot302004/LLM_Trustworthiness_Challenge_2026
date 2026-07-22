from __future__ import annotations

import copy
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
        # Tuned for one 40 GB GPU.  The remaining memory is intentionally left
        # to the CUDA context and the Thai safety classifier.
        "gpu_memory_utilization": 0.84,
        "max_num_seqs": 24,
        "max_num_batched_tokens": 16384,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "language_model_only": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "speculative_config": {
            "method": "qwen3_next_mtp",
            "num_speculative_tokens": 2,
        },
        "temperature": 0.2,
        "top_p": 0.9,
        "seed": 42,
    },
    "guards": {
        "thai_device": "cuda",
        "thai_batch_size": 64,
        "thai_harmful_threshold": 0.50,
        "thai_max_length": 128,
        "thai_response_token_reserve": 64,
        "thai_min_visible_response_tokens": 0,
        "thai_context_preserve_max_visible_tokens": 0,
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

    # Always return a fresh tree; environment overrides from one invocation
    # must not leak into later tests or repeated in-process runs.
    config = copy.deepcopy(DEFAULT_CONFIG)
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

    # Runtime overrides make the same image usable on a smaller/larger GPU
    # without rebuilding it.  Invalid values fail early with a useful error.
    generation_overrides = {
        "VLLM_GPU_MEMORY_UTILIZATION": ("gpu_memory_utilization", float),
        "VLLM_MAX_NUM_SEQS": ("max_num_seqs", int),
        "VLLM_MAX_NUM_BATCHED_TOKENS": ("max_num_batched_tokens", int),
    }
    for variable, (key, converter) in generation_overrides.items():
        value = os.environ.get(variable)
        if value is not None:
            config["generation"][key] = converter(value)

    thai_batch_size = os.environ.get("THAI_GUARD_BATCH_SIZE")
    if thai_batch_size is not None:
        config["guards"]["thai_batch_size"] = int(thai_batch_size)

    utilization = float(config["generation"]["gpu_memory_utilization"])
    if not 0.0 < utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must be in the range (0, 1]")
    for section, key in (
        ("generation", "max_model_len"),
        ("generation", "max_num_seqs"),
        ("generation", "max_num_batched_tokens"),
        ("guards", "thai_batch_size"),
    ):
        if int(config[section][key]) < 1:
            raise ValueError(f"{key} must be at least 1")

    return config


def read_prompt(config: dict[str, Any], name: str) -> str:
    prompt_path = Path(config["paths"]["prompt_dir"]) / name
    return prompt_path.read_text(encoding="utf-8").strip()
