"""
Tests for the multi-source Change-VQA data pipeline and tiled inference.

The fixtures are tiny synthetic datasets written to a temporary directory; they
exercise the code paths (reading each layout, tiling, question generation,
loading, one training step), not model quality.

Run:  python -m unittest ml.C_VQA.test_pipeline
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ml.C_VQA.config import ModelConfig
from ml.C_VQA.dataset import CDVQADataset, cdvqa_collate_fn
from ml.C_VQA.inference import ChangeVQAModel
from ml.C_VQA.model import ChangeVQALoss, SiameseChangeVQA, SimpleTokenizer
from ml.C_VQA.prepare import prepare, tiles_for
from ml.C_VQA.qa import (
    ANSWERS,
    FOCUS_ANY,
    FOCUS_BUILDING,
    FOCUS_SEMANTIC,
    LAND_COVER,
    ChangeLabels,
    generate_questions,
)
from ml.C_VQA.sources import SECOND_PALETTE, RawPair, read_source


def _answers(qa: list[dict]) -> dict[str, str]:
    return {q["type"]: q["answer"] for q in qa}


def _write_folder_dataset(root: Path, size: int = 512) -> None:
    """LEVIR-style layout: <split>/A, B, label with one changed square per pair."""
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        for sub in ("A", "B", "label"):
            (root / split / sub).mkdir(parents=True, exist_ok=True)
        for i in range(2):
            before = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
            after = before.copy()
            mask = np.zeros((size, size), dtype=np.uint8)
            after[100:220, 300:420] = 255
            mask[100:220, 300:420] = 255
            Image.fromarray(before).save(root / split / "A" / f"p{i}.png")
            Image.fromarray(after).save(root / split / "B" / f"p{i}.png")
            Image.fromarray(mask).save(root / split / "label" / f"p{i}.png")


class TestQuestionGeneration(unittest.TestCase):
    def test_every_answer_is_in_the_vocabulary(self):
        change = np.zeros((256, 256), dtype=bool)
        change[:64, :64] = True
        before = np.full((256, 256), LAND_COVER.index("vegetation"))
        after = before.copy()
        after[change] = LAND_COVER.index("water")
        labels = ChangeLabels(change, FOCUS_SEMANTIC, before, after,
                              labelled_classes=("water", "vegetation", "built-up"))
        qa = generate_questions(labels, "tile")
        self.assertTrue(all(q["answer"] in ANSWERS for q in qa))
        answers = _answers(qa)
        self.assertEqual(answers["what_changed"], "vegetation to water")
        self.assertEqual(answers["trend_water"], "increased")
        self.assertEqual(answers["trend_vegetation"], "decreased")
        self.assertEqual(answers["flooding"], "yes")
        self.assertEqual(answers["vegetation_loss"], "yes")
        self.assertEqual(answers["where"], "north-west")

    def test_binary_building_source_asks_nothing_it_cannot_answer(self):
        change = np.zeros((256, 256), dtype=bool)
        change[100:150, 100:150] = True
        labels = ChangeLabels(change, FOCUS_BUILDING, gained={"built-up": change},
                              labelled_classes=("built-up",))
        answers = _answers(generate_questions(labels, "tile"))
        self.assertEqual(answers["what_changed"], "new buildings")
        self.assertEqual(answers["new_buildings"], "yes")
        # LEVIR-style labels carry no vegetation, water or demolition information.
        for kind in ("trend_vegetation", "vegetation_loss", "flooding", "demolished"):
            self.assertNotIn(kind, answers)

    def test_unlabelled_change_type_stays_generic(self):
        change = np.ones((256, 256), dtype=bool)
        answers = _answers(generate_questions(ChangeLabels(change, FOCUS_ANY), "tile"))
        self.assertEqual(answers["what_changed"], "land-cover change")
        self.assertEqual(answers["scale"], "large scale change")

    def test_no_change(self):
        answers = _answers(generate_questions(ChangeLabels(np.zeros((256, 256), bool), FOCUS_ANY), "t"))
        self.assertEqual(answers["any_change"], "no")
        self.assertEqual(answers["count"], "0")
        self.assertNotIn("where", answers)


class TestSources(unittest.TestCase):
    def test_folder_reader_and_multiscale_tiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_folder_dataset(Path(tmp))
            pairs = list(read_source({"type": "levir", "root": tmp}))
            self.assertEqual(len(pairs), 4)
            self.assertEqual({p.split for p in pairs}, {"train", "val"})
            tiles = list(tiles_for(pairs[0], scales=[1, 2, 4]))
            # 512 px: four 256 px tiles at scale 1, one at scale 2; scale 4 is too small.
            self.assertEqual(sorted({t[1] for t in tiles}), [1, 2])
            self.assertEqual(len(tiles), 5)
            self.assertEqual({t[2] for t in tiles}, {0.5, 1.0})

    def test_second_palette_is_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sub in ("im1", "im2", "label1", "label2"):
                (root / sub).mkdir()
            image = np.zeros((512, 512, 3), dtype=np.uint8)
            label1 = np.full((512, 512, 3), 255, dtype=np.uint8)
            label2 = label1.copy()
            label1[:100, :100] = (0, 255, 0)     # tree
            label2[:100, :100] = (128, 0, 0)     # building
            for sub, array in (("im1", image), ("im2", image), ("label1", label1), ("label2", label2)):
                Image.fromarray(array).save(root / sub / "a.png")
            (pair,) = list(read_source({"type": "second", "root": tmp}))
            self.assertEqual(int(pair.change.sum()), 100 * 100)
            self.assertEqual(int(pair.after[0, 0]), LAND_COVER.index("built-up"))

            label1[0, 0] = (1, 2, 3)  # not a SECOND colour
            Image.fromarray(label1).save(root / "label1" / "a.png")
            with self.assertRaises(ValueError):
                list(read_source({"type": "second", "root": tmp}))
        self.assertIn((255, 255, 255), SECOND_PALETTE)


class TestPrepareAndTrain(unittest.TestCase):
    def test_prepare_load_and_one_training_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = Path(tmp) / "raw", Path(tmp) / "out"
            _write_folder_dataset(raw)
            info = prepare({"sources": [{"type": "levir", "root": str(raw)}]}, out,
                           scales=[1, 2], max_tiles=100, keep_unchanged=1.0)
            self.assertEqual(info["gsd_range_m"], [0.5, 1.0])
            self.assertEqual(info["answer_vocab"], ANSWERS)

            train = CDVQADataset(root=out, split="train")
            self.assertEqual(train.answer_vocab, ANSWERS)
            batch = cdvqa_collate_fn([train[i] for i in range(4)])
            self.assertEqual(set(batch["sources"]), {"levir"})
            self.assertIn(batch["gsds"][0], (0.5, 1.0))

            config = ModelConfig(backbone="resnet18", pretrained=False,
                                 vocab_size=len(train.question_vocab), num_classes=len(ANSWERS))
            model = SiameseChangeVQA(config)
            model.tokenizer = SimpleTokenizer(train.question_vocab)
            outputs = model(t1=batch["t1"], t2=batch["t2"], question_ids=batch["question_ids"])
            loss, _ = ChangeVQALoss()(outputs, batch["answer_targets"], batch["mask_targets"])
            loss.backward()
            self.assertTrue(torch.isfinite(loss))


class TestTiledInference(unittest.TestCase):
    def test_large_pairs_are_tiled_and_stitched_to_full_size(self):
        model = SiameseChangeVQA(ModelConfig(backbone="resnet18", pretrained=False))
        specialist = ChangeVQAModel(model, device="cpu", tile_batch_size=4)
        image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (600, 700, 3), dtype=np.uint8))
        result = specialist.analyze_pair(image, image, "What changed?")
        self.assertGreater(result["tiles"], 1)
        self.assertEqual(result["mask_prob"].shape, (600, 700))

        small = specialist.analyze_pair(image.resize((256, 256)), image.resize((256, 256)), "What changed?")
        self.assertEqual(small["tiles"], 1)


class TestRegions(unittest.TestCase):
    def setUp(self):
        model = SiameseChangeVQA(ModelConfig(backbone="resnet18", pretrained=False))
        self.specialist = ChangeVQAModel(model, device="cpu")

    def test_thin_strands_do_not_merge_regions_and_specks_are_ignored(self):
        mask = np.zeros((512, 512), dtype=np.float32)
        mask[50:150, 50:150] = 1.0      # blob A
        mask[50:150, 300:400] = 1.0     # blob B
        mask[99:101, 150:300] = 1.0     # 2 px strand joining them
        mask[450:452, 450:452] = 1.0    # speck
        regions, total = self.specialist._regions(mask)
        self.assertEqual(total, 2)
        self.assertEqual(len(regions), 2)
        self.assertAlmostEqual(regions[0]["share"], 100 * 100 / 512**2, places=3)

    def test_count_reports_all_regions_but_marks_the_largest(self):
        mask = np.zeros((512, 512), dtype=np.float32)
        for i in range(12):
            y, x = divmod(i, 4)
            mask[20 + y * 120 : 60 + y * 120, 20 + x * 120 : 60 + x * 120] = 1.0
        regions, total = self.specialist._regions(mask)
        self.assertEqual(total, 12)
        self.assertEqual(len(regions), 8)
        for region in regions:
            region["label"] = "bare ground to water"
        text = ChangeVQAModel._describe("yes", 7.3, regions, total)
        self.assertIn("in 12 regions; the 8 largest are marked", text)
        self.assertIn("bare ground to water", text)

    def test_regions_are_labelled_by_the_model(self):
        image = Image.fromarray(np.random.default_rng(1).integers(0, 255, (512, 512, 3), dtype=np.uint8))
        mask = np.zeros((512, 512), dtype=np.float32)
        mask[100:300, 100:300] = 1.0
        regions, _ = self.specialist._regions(mask)
        self.specialist._describe_regions(image, image, regions)
        self.assertIn(regions[0]["label"], self.specialist.model.answers_vocab)
        self.assertTrue(0.0 <= regions[0]["label_confidence"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
