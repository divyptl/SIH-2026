"""Download the fine-tuned checkpoints from the Hub into ./checkpoints.

Runs in the container before the API starts. WEIGHTS_REPO names the model repo
upload_weights.py created; HF_TOKEN (a Space secret) is needed when it is private.
Without WEIGHTS_REPO the API still starts, with every specialist disabled.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

TARGET = Path(__file__).resolve().parent / "checkpoints"


def main() -> None:
    repo = os.environ.get("WEIGHTS_REPO", "").strip()
    if not repo:
        print("WEIGHTS_REPO is not set; starting without fine-tuned checkpoints.", file=sys.stderr)
        return

    started = time.perf_counter()
    snapshot_download(
        repo_id=repo,
        repo_type="model",
        local_dir=TARGET,
        token=os.environ.get("HF_TOKEN") or None,
    )
    print(f"Checkpoints from {repo} ready in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
