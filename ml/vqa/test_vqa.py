"""
Unit tests for ml.vqa.

These tests use a lightweight fake backend so they run without a GPU,
MiniGPT-v2, or the SkyEyeGPT checkpoint. Once Phase 1 (real SkyEyeGPT
inference) is confirmed working, add separate integration tests that
exercise SkyEyeGPTModel directly against real images — do not weaken
these unit tests to require a GPU.

Run with:
    python -m unittest ml.vqa.test_vqa -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from .inference import VQAModel, ModelRequest, ModelResponse
from .model import BaseVQAModel


class FakeBackend(BaseVQAModel):
    """Deterministic stand-in for SkyEyeGPTModel."""

    def answer(self, image, question):
        return f"FAKE_ANSWER: {question}"

    def caption(self, image):
        return "FAKE_CAPTION: a remote sensing scene"

    def generate(self, image, prompt):
        return f"FAKE_GEN: {prompt}"

    def generation_score(self):
        return None


def _make_rgb_image(path: Path, size=(64, 64)) -> Path:
    Image.new("RGB", size, color=(120, 130, 80)).save(path)
    return path


class VQAModelTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        self.rgb_path = _make_rgb_image(self.tmp_path / "sample.png")
        self.jpg_path = _make_rgb_image(self.tmp_path / "sample.jpg")
        self.model = VQAModel(backend=FakeBackend())

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

    # 1. RGB image + VQA
    def test_rgb_vqa(self):
        resp = self.model.predict(self._request())
        self.assertIsInstance(resp, ModelResponse)
        self.assertIn("FAKE_ANSWER", resp.answer)
        self.assertEqual(resp.model_name, "SkyEyeGPT-VQA")

    # 2. RGB image + captioning
    def test_rgb_captioning(self):
        resp = self.model.predict(self._request(task_hint="captioning", query=""))
        self.assertIn("FAKE_CAPTION", resp.answer)
        self.assertEqual(resp.model_name, "SkyEyeGPT-Captioning")

    # 3. GeoTIFF + VQA (needs a real fixture + rasterio; kept as an
    #    explicit skip so the suite documents what's still missing
    #    instead of silently omitting the case).
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


if __name__ == "__main__":
    unittest.main()
