from __future__ import annotations

import subprocess
from pathlib import Path


def report_progress(executable: str, completed: int) -> None:
    progress_path = Path(executable)

    if not progress_path.exists():
        raise FileNotFoundError(f"Progress program not found: {progress_path}")

    subprocess.run(
        [str(progress_path), str(completed)],
        check=True,
        timeout=30,
    )
