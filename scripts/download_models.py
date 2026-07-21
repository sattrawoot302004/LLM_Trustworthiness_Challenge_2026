from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("MODEL_DOWNLOAD_DIR", ROOT / "models"))


def download(
    repo_id: str,
    destination: Path,
    ignore_patterns: list[str] | None = None,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(destination),
        local_dir_use_symlinks=False,
        ignore_patterns=ignore_patterns,
    )


def main() -> None:
    download(
        "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        MODELS_DIR / "generator",
    )
    download(
        "typhoon-ai/ThaiSafetyClassifier",
        MODELS_DIR / "thai_guard",
        ignore_patterns=[
            "checkpoint-*",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "trainer_state.json",
            "training_args.bin",
        ],
    )


if __name__ == "__main__":
    main()
