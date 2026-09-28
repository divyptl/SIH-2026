"""
Unit tests for ml.vqa (single-image VQA).

They use a tiny, untrained model (ResNet-18, no pretrained weights), so they
run on CPU without a trained checkpoint.

Run with:
    python -m unittest ml.vqa.test_vqa -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from .config import ModelConfig
from .inference import ModelRequest, ModelResponse, VQAModel
from .model import VQAModel as CoreVQAModel

ANSWERS = ["yes", "no", "2"]
WORDS = ["is", "there", "a", "water", "body", "in", "the", "image", "?", "how", "many", "ships"]


def _tiny_core() -> CoreVQAModel:
    torch.manual_seed(0)
    config = ModelConfig(backbone="resnet18", pretrained=False, num_classes=len(ANSWERS),
                         vocab_size=len(WORDS) + 4)
    return CoreVQAModel(config, answers=ANSWERS, question_vocab=WORDS)


def _make_rgb_image(path: Path, size=(64, 64)) -> Path:
    Image.new("RGB", size, color=(120, 130, 80)).save(path)
    return path


class VQAModelTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        self.rgb_path = _make_rgb_image(self.tmp_path / "sample.png")
        self.jpg_path = _make_rgb_image(self.tmp_path / "sample.jpg")
        self.model = VQAModel(backend=_tiny_core(), device="cpu")

    def tearDown(self):
        self.tmpdir.cleanup()

    def _request(self, **overrides):
        base = dict(
            query="Is there a water body in the image?",
            images=[str(self.rgb_path)],
            modalities=["optical"],
            task_hint="vqa",
        )
        base.update(overrides)
        return ModelRequest(**base)

    # 1. RGB image + VQA: the answer comes from the model's vocabulary
    def test_rgb_vqa(self):
        resp = self.model.predict(self._request())
        self.assertIsInstance(resp, ModelResponse)
        self.assertIn(resp.answer, ANSWERS)
        self.assertEqual(resp.model_name, "SatQuery-VQA")

    # 2. Captioning is refused: a classifier cannot write captions
    def test_captioning_not_supported(self):
        resp = self.model.predict(self._request(task_hint="captioning", query=""))
        self.assertTrue(resp.answer.startswith("ERROR"))
        self.assertIn("Captioning", resp.answer)

    # 3. GeoTIFF + VQA (needs a real fixture + rasterio)
    def test_geotiff_vqa_skipped_without_fixture(self):
        self.skipTest(
            "Requires a real GeoTIFF fixture + rasterio. Add one under "
            "tests/fixtures/ once available and remove this skip."
        )

    # 4. Invalid image path
    def test_invalid_image_path(self):
        resp = self.model.predict(self._request(images=["/no/such/file.jpg"]))
        self.assertTrue(resp.answer.startswith("ERROR"))
        self.assertEqual(resp.confidence, 0.0)

    # 5. No image
    def test_no_image(self):
        resp = self.model.predict(self._request(images=[]))
        self.assertTrue(resp.answer.startswith("ERROR"))

    # 6. Two images passed to single-image VQA
    def test_two_images_rejected(self):
        resp = self.model.predict(
            self._request(images=[str(self.rgb_path), str(self.jpg_path)])
        )
        self.assertTrue(resp.answer.startswith("ERROR"))
        self.assertIn("Change-VQA", resp.answer)

    # 7. Unsupported modality
    def test_unsupported_modality(self):
        resp = self.model.predict(self._request(modalities=["thermal"]))
        self.assertTrue(resp.answer.startswith("ERROR"))

    # 8. Empty query (VQA mode)
    def test_empty_query_vqa(self):
        resp = self.model.predict(self._request(query="", task_hint="vqa"))
        self.assertTrue(resp.answer.startswith("ERROR"))

    # 9. Malformed ModelRequest (missing a required field)
    def test_malformed_request_raises_at_construction(self):
        with self.assertRaises(TypeError):
            ModelRequest(images=[str(self.rgb_path)], modalities=["optical"])  # missing `query`

    # 10. ModelResponse validity / shape
    def test_model_response_shape(self):
        resp = self.model.predict(self._request())
        self.assertIsInstance(resp.answer, str)
        self.assertIsInstance(resp.confidence, float)
        self.assertIsInstance(resp.evidence, list)
        self.assertIsInstance(resp.model_name, str)
        self.assertIsInstance(resp.execution_time_ms, float)
        self.assertGreaterEqual(resp.confidence, 0.0)
        self.assertLessEqual(resp.confidence, 1.0)

    # 11. answer(): best-first top-k whose first entry is the answer
    def test_answer_top_k(self):
        result = self.model.answer(Image.open(self.rgb_path), "How many ships?", top_k=3)
        probs = [p for _, p in result["top"]]
        self.assertEqual(probs, sorted(probs, reverse=True))
        self.assertEqual(result["top"][0], (result["answer"], result["confidence"]))
        self.assertAlmostEqual(sum(probs), 1.0, places=4)

    # 12. Checkpoint round trip keeps vocabularies and weights
    def test_checkpoint_round_trip(self):
        core = _tiny_core().eval()
        path = self.tmp_path / "vqa.pt"
        torch.save(core.checkpoint_dict(train_config={"image_size": 64}), path)
        loaded = CoreVQAModel.from_checkpoint(str(path))
        self.assertEqual(loaded.answers_vocab, ANSWERS)
        self.assertEqual(loaded.tokenizer.vocab, core.tokenizer.vocab)
        self.assertEqual(loaded.image_size, 64)
        image = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            a = core(image=image, question_text="how many ships ?")["answer_logits"]
            b = loaded(image=image, question_text="how many ships ?")["answer_logits"]
        self.assertTrue(torch.allclose(a, b))

    # 13. A Change-VQA checkpoint is refused, not silently loaded
    def test_change_vqa_checkpoint_refused(self):
        path = self.tmp_path / "c_vqa.pt"
        torch.save({"model": {"difference_module.proj.weight": torch.zeros(1)}}, path)
        with self.assertRaises(ValueError) as ctx:
            CoreVQAModel.from_checkpoint(str(path))
        self.assertIn("Change-VQA", str(ctx.exception))


class VQADatasetTests(unittest.TestCase):
    # 14. Answers outside the vocabulary are dropped for training, kept (as
    #     unknown) for evaluation — never relabelled as another answer
    def test_unknown_answers(self):
        from .training.dataset import UNKNOWN_ANSWER, VQADataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_rgb_image(root / "a.png")
            rows = [{"image": "a.png", "question": "is there a ship ?", "answer": "yes"},
                    {"image": "a.png", "question": "what colour ?", "answer": "turquoise"}]
            (root / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            core = _tiny_core()
            train = VQADataset(root, "train", ANSWERS, core.tokenizer, drop_unknown_answers=True)
            full = VQADataset(root, "train", ANSWERS, core.tokenizer)
            self.assertEqual(len(train), 1)
            self.assertEqual(len(full), 2)
            self.assertEqual(int(full[1]["answer_label"]), UNKNOWN_ANSWER)
            self.assertEqual(int(full[0]["answer_label"]), ANSWERS.index("yes"))


if __name__ == "__main__":
    unittest.main()
