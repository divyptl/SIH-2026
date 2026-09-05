"""
Unit and integration tests for Siamese Change-VQA specialist model.

Tests:
    1. Configuration dataclasses
    2. Tokenizer and text encoding
    3. Paired bi-temporal transforms
    4. Siamese backbone and multi-scale feature extraction
    5. Bi-temporal difference and interaction module
    6. Change mask grounding head and bounding box extraction
    7. Full end-to-end SiameseChangeVQA forward pass
    8. Multi-task loss computation and gradient backpropagation
    9. CDVQA dataset creation and batch collating
    10. Controller schema compliance (predict -> ModelResponse)
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap – ensures both the SIH-2026 root and C_VQA dir are findable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # …/sih/SIH-2026
CVQA_DIR = Path(__file__).resolve().parent                    # …/ml/C_VQA
ML_DIR = CVQA_DIR.parent                                      # …/ml

for _p in (str(PROJECT_ROOT), str(ML_DIR), str(CVQA_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if TYPE_CHECKING:
    import torch  # Pyrefly sees real torch types
else:
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Project imports – try the fully-qualified package path first (when running
# from the repo root), then fall back to bare names (when running from C_VQA)
# ---------------------------------------------------------------------------
try:
    from ml.C_VQA.config import ModelConfig, TrainConfig
    from ml.C_VQA.dataset import CDVQADataset, cdvqa_collate_fn
    from ml.C_VQA.inference import ChangeVQAModel
    from ml.C_VQA.model import (
        CANONICAL_ANSWERS,
        ChangeMaskHead,
        ChangeVQALoss,
        SiameseBackbone,
        SiameseChangeVQA,
        SimpleTokenizer,
    )
    from ml.C_VQA.transforms import PairedBitemporalTransform
    from ml.controller.schema import ModelRequest, ModelResponse
except (ImportError, ModuleNotFoundError):
    from config import ModelConfig, TrainConfig  # type: ignore[no-redef]
    from dataset import CDVQADataset, cdvqa_collate_fn  # type: ignore[no-redef]
    from inference import ChangeVQAModel  # type: ignore[no-redef]
    from model import (  # type: ignore[no-redef]
        CANONICAL_ANSWERS,
        ChangeMaskHead,
        ChangeVQALoss,
        SiameseBackbone,
        SiameseChangeVQA,
        SimpleTokenizer,
    )
    from transforms import PairedBitemporalTransform  # type: ignore[no-redef]
    from controller.schema import ModelRequest, ModelResponse  # type: ignore[no-redef]


class TestSiameseChangeVQA(unittest.TestCase):
    """Test suite for the Change Detection / Change-VQA module."""

    def setUp(self):
        if torch is None:
            self.skipTest("PyTorch is not installed in the test environment.")

    def test_config(self):
        """Verify config dataclass defaults."""
        m_cfg = ModelConfig()
        self.assertEqual(m_cfg.backbone, "resnet18")
        self.assertEqual(m_cfg.visual_feature_dim, 256)
        self.assertEqual(m_cfg.num_classes, 64)

        t_cfg = TrainConfig()
        self.assertEqual(t_cfg.batch_size, 16)
        self.assertIn(t_cfg.resolve_device(), ["cpu", "cuda", "mps"])

    def test_tokenizer(self):
        """Verify query tokenization and batch encoding."""
        tok = SimpleTokenizer()
        q = "Has the built-up area increased or decreased?"
        ids, mask = tok.encode(q, max_length=16)

        self.assertEqual(len(ids), 16)
        self.assertEqual(len(mask), 16)
        self.assertEqual(ids[0], tok.w2i[tok.SOS_TOKEN])
        self.assertIn(tok.w2i[tok.EOS_TOKEN], ids)

        # Batch encode
        t_ids, t_masks = tok.batch_encode([q, "What changed between these two images?"], max_length=16)
        self.assertEqual(t_ids.shape, (2, 16))
        self.assertEqual(t_masks.shape, (2, 16))

    def test_paired_transform(self):
        """Verify synchronized transforms on paired images and mask."""
        transform = PairedBitemporalTransform(image_size=128, augment=False)
        img1 = torch.rand(3, 128, 128)
        img2 = torch.rand(3, 128, 128)
        mask = (torch.rand(1, 128, 128) > 0.7).float()

        t1, t2, m = transform(img1, img2, mask)
        self.assertEqual(t1.shape, (3, 128, 128))
        self.assertEqual(t2.shape, (3, 128, 128))
        self.assertEqual(m.shape, (1, 128, 128))  # pyrefly: ignore[missing-attribute]

    def test_siamese_backbone(self):
        """Verify weight-shared Siamese backbone extracts multi-scale features."""
        backbone = SiameseBackbone(backbone_name="resnet18", pretrained=False)
        t1 = torch.randn(2, 3, 128, 128)
        t2 = torch.randn(2, 3, 128, 128)

        feats_t1, feats_t2 = backbone(t1, t2)
        self.assertEqual(len(feats_t1), 4)
        self.assertEqual(len(feats_t2), 4)

        # Stage 1: H/4 (32x32), Stage 4: H/32 (4x4)
        self.assertEqual(feats_t1[0].shape, (2, 64, 32, 32))
        self.assertEqual(feats_t1[3].shape, (2, 512, 4, 4))

    def test_end_to_end_forward(self):
        """Verify full forward pass of SiameseChangeVQA model."""
        cfg = ModelConfig(backbone="resnet18", pretrained=False, visual_feature_dim=64, text_embed_dim=64, num_classes=30)
        model = SiameseChangeVQA(cfg)
        model.eval()

        t1 = torch.randn(2, 3, 128, 128)
        t2 = torch.randn(2, 3, 128, 128)
        queries = ["Has the built-up area increased?", "Did the forest area decrease?"]

        with torch.no_grad():
            outputs = model(t1=t1, t2=t2, question_text=queries)

        self.assertIn("answer_logits", outputs)
        self.assertIn("change_mask_logits", outputs)
        self.assertIn("change_mask_prob", outputs)
        self.assertIn("predicted_answer_text", outputs)

        self.assertEqual(outputs["answer_logits"].shape, (2, 30))
        self.assertEqual(outputs["change_mask_logits"].shape, (2, 1, 128, 128))
        self.assertEqual(outputs["change_mask_prob"].shape, (2, 1, 128, 128))
        self.assertEqual(len(outputs["predicted_answer_text"]), 2)

    def test_multi_task_loss_and_gradients(self):
        """Verify multi-task loss computation and gradient backpropagation."""
        cfg = ModelConfig(backbone="resnet18", pretrained=False, visual_feature_dim=64, text_embed_dim=64, num_classes=20)
        model = SiameseChangeVQA(cfg)
        model.train()

        criterion = ChangeVQALoss(vqa_weight=1.0, mask_bce_weight=0.5, mask_dice_weight=0.5)

        t1 = torch.randn(2, 3, 64, 64)
        t2 = torch.randn(2, 3, 64, 64)
        target_answers = torch.tensor([1, 4], dtype=torch.long)
        target_masks = (torch.rand(2, 1, 64, 64) > 0.8).float()

        outputs = model(t1=t1, t2=t2)
        loss, metrics = criterion(outputs, target_answers, target_masks)

        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_vqa", metrics)
        self.assertIn("loss_mask_bce", metrics)
        self.assertIn("mask_iou", metrics)

        # Backward pass
        loss.backward()  # pyrefly: ignore[not-callable]
        # Verify gradient flow to Siamese backbone
        self.assertIsNotNone(model.backbone.stage1[0].weight.grad)
        self.assertGreater(model.backbone.stage1[0].weight.grad.abs().sum().item(), 0.0)  # pyrefly: ignore[not-callable]

    def test_bounding_box_extraction(self):
        """Verify extraction of change bounding boxes from probability mask."""
        mask = np.zeros((128, 128), dtype=np.float32)
        # Add two synthetic change clusters
        mask[20:50, 30:60] = 0.95
        mask[80:110, 70:100] = 0.88

        bboxes = SiameseChangeVQA.extract_bounding_boxes(mask, threshold=0.5, min_area=30)
        self.assertGreaterEqual(len(bboxes), 2)
        self.assertIn("bbox", bboxes[0])
        self.assertIn("normalized_bbox", bboxes[0])
        self.assertIn("confidence", bboxes[0])

    def test_dataset_and_collate(self):
        """Verify CDVQADataset and custom collate_fn on the real CDVQA dataset."""
        real_cdvqa_root = PROJECT_ROOT / "data" / "cdvqa"
        dataset = CDVQADataset(root=real_cdvqa_root, split="train")
        self.assertGreater(len(dataset), 0)

        item = dataset[0]
        self.assertIn("t1", item)
        self.assertIn("t2", item)
        self.assertIn("mask", item)
        self.assertIn("question_ids", item)
        self.assertIn("answer_idx", item)
        self.assertEqual(item["t1"].shape, (3, 256, 256))
        self.assertEqual(item["t2"].shape, (3, 256, 256))

        batch = cdvqa_collate_fn([dataset[0], dataset[1]])
        self.assertEqual(batch["t1"].shape[0], 2)
        self.assertEqual(batch["t2"].shape[0], 2)
        self.assertEqual(batch["answer_targets"].shape[0], 2)

    def test_controller_schema_integration(self):
        """Verify ChangeVQAModel implements SpecialistModel protocol using real satellite image pairs."""
        cfg = ModelConfig(backbone="resnet18", pretrained=False, visual_feature_dim=64, text_embed_dim=64, num_classes=20)
        model = SiameseChangeVQA(cfg)
        inference_engine = ChangeVQAModel(model=model, config=cfg, device="cpu", image_size=64)

        real_cdvqa_root = PROJECT_ROOT / "data" / "cdvqa"
        t1_path = real_cdvqa_root / "train" / "images_t1" / "train_0001_t1.png"
        t2_path = real_cdvqa_root / "train" / "images_t2" / "train_0001_t2.png"

        request = ModelRequest(
            query="What changed between these two dates?",
            images=[str(t1_path), str(t2_path)],
            modalities=["optical", "optical"],
            task_hint="change_detection",
        )

        response: ModelResponse = inference_engine.predict(request)

        self.assertIsInstance(response, ModelResponse)
        self.assertEqual(response.model_name, "Siamese-VLM-CDVQA")
        self.assertIsInstance(response.answer, str)
        self.assertGreater(len(response.answer), 0)
        self.assertGreaterEqual(response.confidence, 0.0)
        self.assertLessEqual(response.confidence, 1.0)
        self.assertGreater(len(response.evidence), 0)

        # Check evidence types
        ev_types = [e.type for e in response.evidence]
        self.assertIn("mask", ev_types)
        self.assertIn("metrics", ev_types)

    def test_real_cdvqa_dataset_loader(self):
        """Verify ml.datasets.cdvqa.CDVQADataset operates on real formats and enforces real data."""
        from ml.datasets.cdvqa import CDVQADataset as BenchmarkCDVQADataset

        # 1. Verify it loads the real CDVQA dataset correctly
        real_cdvqa_root = PROJECT_ROOT / "data" / "cdvqa"
        real_ds = BenchmarkCDVQADataset(root=real_cdvqa_root, split="train")
        self.assertGreater(len(real_ds), 0)

        # 2. Verify it raises FileNotFoundError when no real dataset exists and does not fake data
        with tempfile.TemporaryDirectory() as empty_dir:
            with self.assertRaises(FileNotFoundError):
                BenchmarkCDVQADataset(root=empty_dir, split="train")

        # 2. Verify it correctly parses real-world multi-question CDVQA format
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            t1_file = tmp_path / "real_t1.png"
            t2_file = tmp_path / "real_t2.png"
            mask_file = tmp_path / "real_mask.png"

            # Create dummy real image files
            from PIL import Image
            img = Image.new("RGB", (64, 64), color=(100, 150, 200))
            img.save(t1_file)
            img.save(t2_file)
            mask = Image.new("L", (64, 64), color=255)
            mask.save(mask_file)

            ann_data = {
                "samples": [
                    {
                        "id": "real_pair_01",
                        "image_t1": str(t1_file),
                        "image_t2": str(t2_file),
                        "mask": str(mask_file),
                        "questions": [
                            {"question": "Has the built-up area increased?", "answer": "increased", "change_type": "urban"},
                            {"question": "Did vegetation decrease?", "answer": "no", "change_type": "vegetation"}
                        ]
                    }
                ]
            }
            with open(tmp_path / "train_annotations.json", "w", encoding="utf-8") as f:
                import json
                json.dump(ann_data, f)

            real_ds = BenchmarkCDVQADataset(root=tmp_path, split="train", auto_generate=False)
            self.assertEqual(len(real_ds), 2)  # Unpacked 2 questions from real pair
            item = real_ds[0]
            self.assertEqual(item["sample_id"], "real_pair_01_0")
            self.assertEqual(item["raw_answer"], "increased")
            self.assertEqual(item["change_type"], "urban")

    def test_sen12_dataset_schema(self):
        """Verify ml.datasets.sen12.SEN12Dataset adheres to the Readme.md schema."""
        from ml.datasets.sen12 import SEN12Dataset

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            urban_s1 = tmp_path / "urban" / "s1"
            urban_s2 = tmp_path / "urban" / "s2"
            urban_s1.mkdir(parents=True)
            urban_s2.mkdir(parents=True)

            from PIL import Image
            sar_im = Image.new("L", (64, 64), color=128)
            opt_im = Image.new("RGB", (64, 64), color=(30, 100, 50))
            sar_im.save(urban_s1 / "patch_001.png")
            opt_im.save(urban_s2 / "patch_001.png")

            ds = SEN12Dataset(root=tmp_path, terrains=["urban"], split="train")
            self.assertEqual(len(ds), 1)
            sar_t, opt_t = ds[0]
            # Verify shapes: sar (1, H, W) and opt (3, H, W) float32
            self.assertEqual(sar_t.shape, (1, 64, 64))
            self.assertEqual(opt_t.shape, (3, 64, 64))


if __name__ == "__main__":
    unittest.main()
