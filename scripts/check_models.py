from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = {
    "generator": ROOT / "models" / "generator" / "config.json",
    "qwen_guard": ROOT / "models" / "qwen_guard" / "config.json",
    "thai_guard": ROOT / "models" / "thai_guard" / "config.json",
}


def main() -> int:
    missing = [
        f"{name}: {path.relative_to(ROOT)}"
        for name, path in REQUIRED_FILES.items()
        if not path.exists()
    ]

    if missing:
        print("Missing required model files:")
        for item in missing:
            print(f"- {item}")
        print()
        print("Run these before docker compose build:")
        print("python -m pip install -r requirements.in")
        print("python scripts/download_models.py")
        print("python scripts/merge_thai_guard.py")
        return 1

    print("All required model files are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
