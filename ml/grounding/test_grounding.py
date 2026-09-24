"""
Smoke tests for the grounding module.

Tests:
    1. Model loads from pre-trained checkpoint
    2. Forward pass produces valid output shapes
    3. Inference wrapper returns expected dict structure
    4. Box coordinate sanity checks
    5. Dataset box parsing logic

Usage:
    python -m ml.grounding.test_grounding
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
# Windows consoles default to cp1252, which cannot encode the box-drawing and
# arrow characters in this module's progress output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_box_parsing():
    """Test VRSBench box coordinate parsing."""
    from ml.grounding.dataset import VRSBenchGroundingDataset

    ds = VRSBenchGroundingDataset.__new__(VRSBenchGroundingDataset)
    ds.split = "test"
    ds.samples = []

    # Single box [x1, y1, x2, y2] in 0–100 → [cx, cy, w, h] in 0–1
    result = ds._parse_boxes([10, 20, 50, 60])
    assert result is not None and len(result) == 1
    cx, cy, w, h = result[0]
    assert abs(cx - 0.30) < 0.01, f"Expected cx≈0.30, got {cx}"
    assert abs(cy - 0.40) < 0.01, f"Expected cy≈0.40, got {cy}"
    assert abs(w - 0.40) < 0.01, f"Expected w≈0.40, got {w}"
    assert abs(h - 0.40) < 0.01, f"Expected h≈0.40, got {h}"

    # Multiple boxes
    result = ds._parse_boxes([[0, 0, 50, 50], [50, 50, 100, 100]])
    assert result is not None and len(result) == 2

    # JSON string
    result = ds._parse_boxes('[10, 20, 50, 60]')
    assert result is not None and len(result) == 1

    # Edge cases
    assert ds._parse_boxes(None) is None
    assert ds._parse_boxes([]) is None
    assert ds._parse_boxes("invalid") is None

    # Full-image box
    result = ds._parse_boxes([0, 0, 100, 100])
    assert result is not None
    cx, cy, w, h = result[0]
    assert abs(cx - 0.50) < 0.01
    assert abs(cy - 0.50) < 0.01
    assert abs(w - 1.00) < 0.01
    assert abs(h - 1.00) < 0.01

    print("  ✓ Box parsing tests passed")


def test_transforms():
    """Test augmentation with box coordinate consistency."""
    import torch
    from PIL import Image

    from ml.grounding.transforms import GroundingAugmentation

    # Create a dummy image and boxes
    image = Image.new("RGB", (256, 256), color=(100, 150, 200))
    boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4]])  # Center box

    # No augmentation
    aug_off = GroundingAugmentation(augment=False)
    out_img, out_boxes, out_text = aug_off(image, boxes, "the car on the left")
    assert torch.allclose(out_boxes, boxes), "No-augment should preserve boxes"
    assert out_text == "the car on the left", "No-augment should preserve text"

    # With augmentation (run multiple times to exercise random paths)
    aug_on = GroundingAugmentation(augment=True)
    off_center = torch.tensor([[0.2, 0.5, 0.1, 0.1]])  # box on the left
    for _ in range(20):
        out_img, out_boxes, out_text = aug_on(image, off_center, "the car on the left")
        # Boxes should still be valid (cx, cy in [0,1], w, h > 0)
        assert (out_boxes[:, :2] >= 0).all() and (out_boxes[:, :2] <= 1).all(), \
            f"Box centers out of range: {out_boxes}"
        assert (out_boxes[:, 2:] > 0).all(), f"Box dimensions non-positive: {out_boxes}"
        # A flipped box must come with a flipped description
        flipped = out_boxes[0, 0] > 0.5
        assert out_text == ("the car on the right" if flipped else "the car on the left"), \
            f"Text {out_text!r} disagrees with box {out_boxes}"

    from ml.grounding.transforms import swap_left_right
    assert swap_left_right("Right-most tank, left of the top-left roof") == \
        "Left-most tank, right of the top-right roof"
    assert swap_left_right("the leftmost bright building") == "the rightmost bright building"

    print("  ✓ Transform tests passed")


def test_config():
    """Test config dataclass defaults."""
    from ml.grounding.config import ModelConfig, TrainConfig

    model_cfg = ModelConfig()
    assert model_cfg.model_id == "IDEA-Research/grounding-dino-tiny"
    assert model_cfg.freeze_backbone is True
    assert model_cfg.freeze_text_encoder is True
    assert model_cfg.box_threshold == 0.25

    train_cfg = TrainConfig()
    assert train_cfg.epochs == 20
    assert train_cfg.batch_size == 8
    assert train_cfg.grad_accum_steps == 2
    assert train_cfg.num_workers == 6
    assert train_cfg.eval_every == 5
    assert train_cfg.lr == 1e-5
    assert train_cfg.data_name == "xiang709/VRSBench"

    # Test device resolution
    device = train_cfg.resolve_device()
    assert device in ("cuda", "mps", "cpu")

    print("  ✓ Config tests passed")


def test_reranker():
    """Test the candidate re-ranker on synthetic candidates (no download)."""
    import torch

    from ml.grounding.config import RerankConfig
    from ml.grounding.rerank import CandidateReranker, _pairwise_rank, rerank_loss

    # Rank features: 0 for the smallest valid value, 1 for the largest,
    # padding ignored.
    values = torch.tensor([[0.9, 0.1, 0.5, 0.0]])
    mask = torch.tensor([[True, True, True, False]])
    ranks = _pairwise_rank(values, mask)[0, :3].tolist()
    assert ranks == [1.0, 0.0, 0.5], ranks

    b, k, t = 3, 10, 12
    torch.manual_seed(0)
    inputs = {
        "hidden": torch.randn(b, k, 256),
        "roi": torch.randn(b, k, 256),
        "boxes": torch.rand(b, k, 4) * 0.4 + 0.1,
        "scores": torch.rand(b, k),
        "cand_mask": torch.ones(b, k, dtype=torch.bool),
        "tok_logits": torch.randn(b, k, t),
        "text": torch.randn(b, t, 256),
        "text_mask": torch.ones(b, t, dtype=torch.bool),
        "input_ids": torch.randint(1, 30522, (b, t)),
    }
    inputs["cand_mask"][0, 6:] = False
    inputs["text_mask"][1, 8:] = False

    model = CandidateReranker(RerankConfig(num_layers=2))
    logits, refined = model(inputs)
    assert logits.shape == (b, k) and refined.shape == (b, k, 4)
    assert torch.isinf(logits[0, 6:]).all(), "Padding candidates must never be picked"
    assert torch.isfinite(logits[0, :6]).all()

    # Target = candidate 2 exactly, so every sample has a positive.
    gt = inputs["boxes"][:, 2].clone()
    loss, stats = rerank_loss(logits, refined, inputs, gt, min_iou=0.5, refine=True)
    loss.backward()
    assert torch.isfinite(loss) and {"rank", "l1", "giou"} <= stats.keys()

    picked = model.select(inputs)
    assert picked["order"][0, 0] < 6
    print("  ✓ Re-ranker tests passed")


def test_model_loading():
    """Test that GroundingDINO loads and produces valid outputs."""
    import torch
    from PIL import Image

    from ml.grounding.config import ModelConfig
    from ml.grounding.model import GroundingModel

    print("  Loading GroundingDINO-Tiny (this may take a moment)...")
    config = ModelConfig(freeze_backbone=True)
    model = GroundingModel(config)

    # Check param counts
    total = model.get_total_params()
    trainable = model.get_trainable_params()
    assert total > 0, "Model should have parameters"
    assert trainable < total, "Some params should be frozen"
    print(f"  Total: {total/1e6:.1f}M, Trainable: {trainable/1e6:.1f}M")

    # Forward pass with dummy image
    dummy_image = Image.new("RGB", (800, 800), color=(100, 150, 200))
    text = "buildings ."

    inputs = model.preprocess(dummy_image, text)
    assert "pixel_values" in inputs
    assert "input_ids" in inputs

    model.model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    assert outputs.pred_boxes is not None
    assert outputs.logits is not None
    assert outputs.pred_boxes.shape[-1] == 4  # (B, num_queries, 4)
    print(f"  Output shapes: boxes={outputs.pred_boxes.shape}, logits={outputs.logits.shape}")

    from ml.grounding.rerank import extract_candidates
    cands = extract_candidates(
        outputs, inputs["pixel_values"], inputs["input_ids"], inputs["attention_mask"],
        k=10, nms_iou=0.5, max_text_tokens=48,
    )
    assert cands["hidden"].shape == (1, 10, 256)
    assert cands["roi"].abs().sum() > 0, "RoI features should be pooled from the encoder"
    assert cands["cand_mask"].all()
    assert cands["text_mask"][0].sum() == inputs["attention_mask"][0].sum()
    print("  Candidate extraction OK")

    print("  ✓ Model loading tests passed")


def test_inference_wrapper():
    """Test the high-level inference API."""
    from PIL import Image

    from ml.grounding.inference import GroundingInference

    print("  Loading inference wrapper...")
    model = GroundingInference.from_pretrained(device="cpu")

    # Create a dummy satellite image
    dummy_image = Image.new("RGB", (512, 512), color=(80, 120, 60))

    # Test ground()
    detections = model.ground(dummy_image, "buildings")
    assert isinstance(detections, list)
    for det in detections:
        assert "box" in det and "score" in det and "label" in det
        assert len(det["box"]) == 4
        assert 0 <= det["score"] <= 1
        # Box coordinates should be within image bounds (with small tolerance)
        for coord in det["box"]:
            assert -10 <= coord <= 522, f"Box coord out of range: {coord}"

    # Test get_analysis()
    analysis = model.get_analysis(dummy_image, "roads and water")
    assert "query" in analysis
    assert "detections" in analysis
    assert "num_detections" in analysis
    assert "image_size" in analysis
    assert analysis["image_size"] == [512, 512]
    assert isinstance(analysis["detections"], list)

    print(f"  Detected {analysis['num_detections']} objects (zero-shot on dummy image)")
    print("  ✓ Inference wrapper tests passed")


def main():
    print("=" * 50)
    print("  Grounding Module — Smoke Tests")
    print("=" * 50)

    test_config()
    test_box_parsing()
    test_transforms()
    test_reranker()

    # These tests download the model (~700MB first time)
    print("\n  [Model tests require downloading GroundingDINO-Tiny]")
    try:
        test_model_loading()
        test_inference_wrapper()
    except OSError as e:
        # Only a failed download is tolerable here. Catching everything meant a
        # real API break (a renamed kwarg) was reported as a pass.
        print(f"  ⚠ Model tests skipped — download unavailable: {e}")

    print("\n" + "=" * 50)
    print("  All smoke tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
