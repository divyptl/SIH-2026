"""
STUB — do not use until Phase 4 (small RS domain adaptation) begins.

Will run LoRA fine-tuning of the MiniGPT-v2 + SkyEyeGPT backbone on a
small subset produced by dataset.load_training_subset(), using
lora_config.DEFAULT_LORA_CONFIG. Not implemented yet — see README.md
"Roadmap" for why this comes after baseline evaluation, not before.
"""
from __future__ import annotations


def main() -> int:
    raise NotImplementedError(
        "LoRA training is a Phase 4 stub. Implement only after Phases 1-3 "
        "(baseline SkyEyeGPT inference, SatQuery schema integration, and "
        "RSVQA/VRSBench evaluation) are complete — see ml/vqa/README.md."
    )


if __name__ == "__main__":
    raise SystemExit(main())
