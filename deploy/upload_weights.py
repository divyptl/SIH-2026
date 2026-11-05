"""Upload the checkpoints the API serves to a (private) model repo on the Hub.

One-off, from a machine that has checkpoints/ (about 2.3 GB goes up). The API
container downloads this repo at startup (fetch_weights.py), keeping the same
layout, so the paths in the Dockerfile resolve unchanged.

    hf auth login     # or set HF_TOKEN to a token with write access
    uv run --no-project --with huggingface-hub python deploy/upload_weights.py <user>/satquery-weights
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

CHECKPOINTS = Path(__file__).resolve().parents[2] / "checkpoints"

# Keep in step with the *_CHECKPOINT / GROUNDING_RERANKERS paths in the Dockerfile.
FILES = [
    "grounding/v3/last.pt",
    "grounding/combined/reranker.pt",
    "grounding/combined/reranker_seed1.pt",
    "grounding/combined/reranker_seed2.pt",
    "grounding/combined/reranker_seed3.pt",
    "grounding/combined/reranker_seed4.pt",
    "change_detection_v2/best.pt",
    "fusion_best.pt",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo", help="Model repo to upload to, e.g. my-user/satquery-weights")
    parser.add_argument("--public", action="store_true", help="Create the repo public (default: private)")
    args = parser.parse_args()

    missing = [name for name in FILES if not (CHECKPOINTS / name).is_file()]
    if missing:
        sys.exit(f"Missing under {CHECKPOINTS}: {', '.join(missing)}")

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=CHECKPOINTS,
        allow_patterns=FILES,
        commit_message="Upload SatQuery AI checkpoints",
    )
    print(f"Uploaded {len(FILES)} checkpoints to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
