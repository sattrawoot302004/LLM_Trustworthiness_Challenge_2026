from __future__ import annotations

import subprocess
import time
from pathlib import Path


def report_progress(
    executable: str,
    completed: int,
    *,
    attempts: int = 3,
    timeout_seconds: float = 15.0,
) -> None:
    if completed < 0:
        raise ValueError("completed must be non-negative")
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    progress_path = Path(executable)

    if not progress_path.is_file():
        raise FileNotFoundError(f"Progress program not found: {progress_path}")

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            subprocess.run(
                [str(progress_path), str(completed)],
                check=True,
                timeout=timeout_seconds,
            )
            return
        except (OSError, subprocess.SubprocessError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(0.5 * attempt, 1.5))

    raise RuntimeError(
        f"Progress program failed after {attempts} attempt(s): {progress_path}"
    ) from last_error
