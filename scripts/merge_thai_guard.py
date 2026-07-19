from __future__ import annotations

import shutil
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "models" / "thai_guard_base"
ADAPTER_DIR = ROOT / "models" / "thai_guard_adapter"
OUTPUT_DIR = ROOT / "models" / "thai_guard"


def copy_full_model(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, output_dir / path.name)


def find_adapter_dir(source_dir: Path) -> Path | None:
    if (source_dir / "adapter_config.json").exists():
        return source_dir

    checkpoints = sorted(source_dir.glob("checkpoint-*"))
    for checkpoint in reversed(checkpoints):
        if (checkpoint / "adapter_config.json").exists():
            return checkpoint

    return None


def main() -> None:
    if (OUTPUT_DIR / "config.json").exists() and (
        (OUTPUT_DIR / "model.safetensors").exists()
        or (OUTPUT_DIR / "pytorch_model.bin").exists()
    ):
        print(f"Thai guard is already available at {OUTPUT_DIR}")
        return

    if (ADAPTER_DIR / "config.json").exists() and (
        (ADAPTER_DIR / "model.safetensors").exists()
        or (ADAPTER_DIR / "pytorch_model.bin").exists()
    ):
        copy_full_model(ADAPTER_DIR, OUTPUT_DIR)
        print(f"Copied ThaiSafetyClassifier full model to {OUTPUT_DIR}")
        return

    adapter_dir = find_adapter_dir(ADAPTER_DIR)
    if adapter_dir is None:
        raise FileNotFoundError(
            f"No full model or PEFT adapter found under {ADAPTER_DIR}"
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), local_files_only=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR), local_files_only=True)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        str(BASE_DIR),
        num_labels=2,
        local_files_only=True,
    )
    from peft import PeftModel

    adapter_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
        local_files_only=True,
    )
    merged_model = adapter_model.merge_and_unload()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
