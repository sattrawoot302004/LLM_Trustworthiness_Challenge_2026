from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("MODEL_DOWNLOAD_DIR", ROOT / "models"))

REQUIRED_FILES = {
    "generator": MODELS_DIR / "generator" / "config.json",
    "qwen_guard": MODELS_DIR / "qwen_guard" / "config.json",
    "thai_guard": MODELS_DIR / "thai_guard" / "config.json",
}


def main() -> int:
    missing = [
        f"{name}: {path}"
        for name, path in REQUIRED_FILES.items()
        if not path.exists()
    ]

    if missing:
        print("Missing required model files:")
        for item in missing:
            print(f"- {item}")
        print()
        print("For local model preparation, run:")
        print("python -m pip install -r requirements.in")
        print("python scripts/download_models.py")
        print()
        print("Docker build downloads models into /opt/models automatically.")
        return 1

    print("All required model files are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
