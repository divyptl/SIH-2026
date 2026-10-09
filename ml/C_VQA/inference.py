"""
Inference pipeline for Siamese Change-VQA specialist model.

Provides high-level inference APIs and implements the SpecialistModel protocol
from ml.controller.schema, enabling seamless integration with the SatQuery AI
Agentic Controller.

Features:
    - Load trained model from checkpoint
    - Predict answers to natural language questions about bi-temporal image pairs
    - Generate dense change probability heatmaps and binary change masks
    - Extract grounded bounding boxes for changed regions
    - Formulate standardized ModelResponse with visual and numerical evidence
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# ---------------------------------------------------------------------------
# Optional heavy dependencies – always importable at type-check time so that
# Pyrefly/Pyright see the real types; at runtime we degrade gracefully.
# ---------------------------------------------------------------------------
if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from PIL import Image, ImageDraw
else:
    try:
        import torch
        import torch.nn as nn
    except ImportError:  # pragma: no cover
        torch = None  # type: ignore[assignment]
        nn = None  # type: ignore[assignment]

    try:
        from PIL import Image, ImageDraw
    except ImportError:  # pragma: no cover
        Image = None  # type: ignore[assignment]
        ImageDraw = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Checkpoints trained before prepare.py recorded a GSD range were trained on
# LEVIR-CD alone (0.5 m aerial imagery).
LEGACY_GSD_RANGE_M = (0.5, 0.5)

# Most regions marked with a box; every region still counts towards the total.
MAX_REGIONS = 8
# Asked of a crop around each region to name the change inside it.
REGION_QUESTION = "What has changed in these 2 images?"
# Answers that say whether or how much something changed, but not what; they
# do not describe a region, so the summary skips them.
NON_DESCRIPTIVE_ANSWERS = {
    "yes", "no", "change", "no change", "unchanged", "increased", "decreased",
    "minor change", "moderate change", "large scale change", "no significant change detected",
    "center", "north-west", "north-east", "south-west", "south-east",
    "0", "1", "2", "3", "4", "5 or more",
}

# Project imports
# ---------------------------------------------------------------------------
# Imported as a package, never via sys.path edits: the backend imports this
# module, and exposing ml/C_VQA's `config`/`model` as top-level names would
# shadow the backend's own `config` module.
from ml.C_VQA.config import ModelConfig
from ml.C_VQA.model import SiameseChangeVQA, SimpleTokenizer
from ml.C_VQA.transforms import normalize_image
from ml.controller.schema import Evidence, ModelRequest, ModelResponse


class ChangeVQAModel:
    """High-level inference engine for bi-temporal Change-VQA.

    Implements the SpecialistModel protocol expected by the SatQuery AI controller.
    """

    def __init__(
        self,
        model: SiameseChangeVQA,
        config: ModelConfig | None = None,
        device: str = "cpu",
        image_size: int = 256,
        gsd_range_m: tuple[float, float] = LEGACY_GSD_RANGE_M,
        tile_overlap: int = 32,
        max_tiled_edge: int = 2048,
        tile_batch_size: int = 16,
    ) -> None:
        self.device = device
        self.config = config or ModelConfig()
        self.image_size = image_size
        self.model = model.to(self.device).eval()
        # Ground sample distances (m/pixel) the training tiles covered.
        self.gsd_range_m = gsd_range_m
        # Images larger than one tile are analysed tile by tile at their own
        # resolution instead of being squashed to image_size, which would erase
        # the detail the change mask needs.
        self.tile_overlap = tile_overlap
        self.max_tiled_edge = max_tiled_edge
        self.tile_batch_size = tile_batch_size

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path | None = None,
        device: str = "auto",
    ) -> "ChangeVQAModel":
        """Load a trained Change-VQA model from a checkpoint file.

        If checkpoint_path is None or does not exist, instantiates a model
        with pretrained backbone weights.
        """
        if device == "auto":
            if torch is not None and torch.cuda.is_available():
                device = "cuda"
            elif torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        if checkpoint_path is not None and Path(checkpoint_path).exists() and torch is not None:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            cfg_dict = ckpt.get("model_config", {})
            if isinstance(cfg_dict, dict):
                config = ModelConfig(**cfg_dict)
            else:
                config = cfg_dict

            model = SiameseChangeVQA(config)
            model.load_state_dict(ckpt["model"])
            if "answers_vocab" in ckpt:
                model.answers_vocab = ckpt["answers_vocab"]
            if "question_vocab" in ckpt:
                model.tokenizer = SimpleTokenizer(ckpt["question_vocab"])
            gsd_range = tuple(ckpt.get("gsd_range_m") or LEGACY_GSD_RANGE_M)
            print(f"[ChangeVQAModel] Loaded checkpoint from {checkpoint_path} on {device} "
                  f"(trained on {gsd_range[0]}-{gsd_range[1]} m/pixel)")
        else:
            config = ModelConfig()
            model = SiameseChangeVQA(config)
            gsd_range = LEGACY_GSD_RANGE_M
            print(f"[ChangeVQAModel] Initialized model on {device} (no checkpoint specified or found)")

        return cls(model=model, config=config, device=device, gsd_range_m=gsd_range)  # type: ignore[arg-type]

    def _load_and_preprocess_image(self, img_input: str | Path | Image.Image | np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        """Load an image and convert to preprocessed tensor (1, 3, H, W)."""
        img = self._to_rgb(img_input)
        return self._to_tensor(img.resize((self.image_size, self.image_size))), img.size

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img).astype(np.float32) / 255.0      # (H, W, 3)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)    # (3, H, W)
        return normalize_image(tensor).unsqueeze(0).to(self.device)  # (1, 3, H, W)

    @staticmethod
    def _to_rgb(img_input: str | Path | Image.Image | np.ndarray) -> Image.Image:
        """Load any supported input as an 8-bit RGB PIL image."""
        if isinstance(img_input, (str, Path)):
            path = Path(img_input)
            if not path.exists():
                raise FileNotFoundError(f"Image not found at {path}")
            if Image is not None:
                with Image.open(path) as raw_img:
                    if raw_img.mode in ("I;16", "I", "F"):
                        arr16 = np.array(raw_img, dtype=np.float32)
                        mn, mx = arr16.min(), arr16.max()
                        scaled = ((arr16 - mn) / (mx - mn + 1e-6) * 255.0).astype(np.uint8) if mx > mn else np.zeros_like(arr16, dtype=np.uint8)
                        img = Image.fromarray(scaled).convert("RGB")
                    else:
                        img = raw_img.convert("RGB")
        elif isinstance(img_input, np.ndarray):
            if img_input.dtype != np.uint8:
                mn, mx = img_input.min(), img_input.max()
                scaled = ((img_input - mn) / (mx - mn + 1e-6) * 255.0).astype(np.uint8) if mx > mn else np.zeros_like(img_input, dtype=np.uint8)
                img = Image.fromarray(scaled).convert("RGB")
            else:
                img = Image.fromarray(img_input).convert("RGB")
        elif Image is not None and isinstance(img_input, Image.Image):
            img = img_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(img_input)}")
        return img

    def _tile_starts(self, length: int) -> list[int]:
        tile, stride = self.image_size, self.image_size - self.tile_overlap
        starts = list(range(0, max(1, length - tile + 1), stride))
        if starts[-1] + tile < length:
            starts.append(length - tile)
        return starts

    @torch.no_grad()
    def _tiled_change_mask(self, img_t1: Image.Image, img_t2: Image.Image, query: str) -> tuple[np.ndarray, int]:
        """Change probability at the pair's own resolution, stitched from tiles.

        Overlapping tiles are averaged so tile seams do not show in the mask.
        Returns (mask, number of tiles).
        """
        tile = self.image_size
        width, height = img_t1.size
        # Bound the work for very large scenes, and make sure a tile fits.
        scale = min(1.0, self.max_tiled_edge / max(width, height))
        scale = max(scale, tile / min(width, height))
        if scale != 1.0:
            size = (max(tile, round(width * scale)), max(tile, round(height * scale)))
            img_t1 = img_t1.resize(size, Image.BILINEAR)
            img_t2 = img_t2.resize(size, Image.BILINEAR)
            width, height = size

        total = np.zeros((height, width), dtype=np.float32)
        weight = np.zeros((height, width), dtype=np.float32)
        boxes = [(x, y) for y in self._tile_starts(height) for x in self._tile_starts(width)]
        for i in range(0, len(boxes), self.tile_batch_size):
            batch = boxes[i : i + self.tile_batch_size]
            t1 = torch.cat([self._to_tensor(img_t1.crop((x, y, x + tile, y + tile))) for x, y in batch])
            t2 = torch.cat([self._to_tensor(img_t2.crop((x, y, x + tile, y + tile))) for x, y in batch])
            probs = self.model(t1=t1, t2=t2, question_text=[query] * len(batch))["change_mask_prob"]
            probs = probs[:, 0].float().cpu().numpy()
            for (x, y), prob in zip(batch, probs):
                total[y : y + tile, x : x + tile] += prob
                weight[y : y + tile, x : x + tile] += 1.0
        return total / np.maximum(weight, 1.0), len(boxes)

    def analyze_pair(
        self,
        img_t1: str | Path | Image.Image | np.ndarray,
        img_t2: str | Path | Image.Image | np.ndarray,
        query: str = "What changed between these two dates?",
    ) -> dict[str, Any]:
        """Run inference on an image pair and query string.

        The answer comes from the whole scene at model resolution, so it can
        take in context across the image. The change mask, and the boxes and
        change share derived from it, come from tiles at the pair's own
        resolution whenever the pair is larger than one tile.
        """
        start_time = time.perf_counter()

        rgb_t1, rgb_t2 = self._to_rgb(img_t1), self._to_rgb(img_t2)
        orig_size = rgb_t1.size
        if rgb_t2.size != rgb_t1.size:
            rgb_t2 = rgb_t2.resize(rgb_t1.size, Image.BILINEAR)
        scene = (self.image_size, self.image_size)

        with torch.no_grad():
            outputs = self.model(
                t1=self._to_tensor(rgb_t1.resize(scene)),
                t2=self._to_tensor(rgb_t2.resize(scene)),
                question_text=[query],
            )

        tiles = 1
        if max(orig_size) > self.image_size * 1.25:
            mask_prob, tiles = self._tiled_change_mask(rgb_t1, rgb_t2, query)
        else:
            mask_prob = outputs["change_mask_prob"][0, 0].float().cpu().numpy()

        # Everything reported comes from the network: its answer with its top-1
        # softmax confidence, and its change mask. The mask is not blended with a
        # raw pixel difference, and the answer is not replaced by keyword rules or
        # fixed confidences, so callers see what the fine-tuned model predicted.
        answer_text = outputs["predicted_answer_text"][0]
        answer_confidence = float(outputs["answer_confidence"][0])

        regions, total_regions = self._regions(mask_prob)
        self._describe_regions(rgb_t1, rgb_t2, regions)

        changed_pixel_count = int((mask_prob >= self.config.mask_threshold).sum())
        change_percentage = round((changed_pixel_count / mask_prob.size) * 100.0, 2)
        dominant = self._dominant_label(regions)
        exec_time_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "answer": self._describe(answer_text, change_percentage, regions, total_regions),
            "primary_answer": answer_text,
            "confidence": round(answer_confidence, 4),
            "change_percentage": change_percentage,
            "changed_pixels": changed_pixel_count,
            "detected_domain": dominant.replace(" ", "_") if dominant else "unchanged",
            "bounding_boxes": regions,
            "total_regions": total_regions,
            "mask_prob": mask_prob,
            "execution_time_ms": round(exec_time_ms, 2),
            "original_size": orig_size,
            "tiles": tiles,
        }

    def _regions(self, mask_prob: np.ndarray) -> tuple[list[dict[str, Any]], int]:
        """Changed regions worth marking, largest first, and how many there are.

        A light morphological opening first cuts the thin strands that would
        otherwise chain separate changes (river channels, road networks) into
        one scene-wide blob, and patches under 0.1% of the scene are ignored.
        The count covers every region; only the MAX_REGIONS largest are returned.
        """
        import cv2

        height, width = mask_prob.shape
        binary = (mask_prob >= self.config.mask_threshold).astype(np.uint8)
        kernel = max(3, round(min(height, width) / 128)) | 1
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((kernel, kernel), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)

        min_area = max(35, 0.001 * height * width)
        found = sorted(
            (i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area),
            key=lambda i: -stats[i, cv2.CC_STAT_AREA],
        )
        regions = []
        for i in found[:MAX_REGIONS]:
            x, y, w, h, area = (int(v) for v in stats[i, :5])
            normalized = [round(x / width, 4), round(y / height, 4),
                          round((x + w) / width, 4), round((y + h) / height, 4)]
            regions.append({
                "bbox": [x, y, x + w, y + h],
                "normalized_bbox": normalized,
                "area": area,
                "share": area / (height * width),
                # Mean change probability inside the region.
                "confidence": round(float(mask_prob[labels == i].mean()), 4),
                "sector": self._sector(normalized),
                "label": "change",
                "label_confidence": None,
            })
        return regions, len(found)

    @torch.no_grad()
    def _describe_regions(self, img_t1: Image.Image, img_t2: Image.Image, regions: list[dict[str, Any]]) -> None:
        """Name the change in each region with the model's own answer about it.

        The model is asked REGION_QUESTION about a crop around each region, with
        at least one tile of context as in training, all regions in one batch.
        """
        if not regions:
            return
        width, height = img_t1.size
        size = (self.image_size, self.image_size)
        crops_t1, crops_t2 = [], []
        for region in regions:
            x1, y1, x2, y2 = region["normalized_bbox"]
            cx, cy = (x1 + x2) / 2 * width, (y1 + y2) / 2 * height
            side = max(self.image_size, (x2 - x1) * width, (y2 - y1) * height)
            box = (
                int(max(0, cx - side / 2)), int(max(0, cy - side / 2)),
                int(min(width, cx + side / 2)), int(min(height, cy + side / 2)),
            )
            crops_t1.append(self._to_tensor(img_t1.crop(box).resize(size)))
            crops_t2.append(self._to_tensor(img_t2.crop(box).resize(size)))

        outputs = self.model(
            t1=torch.cat(crops_t1), t2=torch.cat(crops_t2),
            question_text=[REGION_QUESTION] * len(regions),
        )
        probabilities = outputs["answer_logits"].float().softmax(dim=-1)
        for region, probs in zip(regions, probabilities):
            confidence, index = probs.max(dim=0)
            region["label"] = self.model.answers_vocab[int(index)]
            region["label_confidence"] = round(float(confidence), 4)

    @staticmethod
    def _label_shares(regions: list[dict[str, Any]]) -> list[tuple[str, float]]:
        """Descriptive region labels with the share of the scene they cover, largest first."""
        shares: dict[str, float] = {}
        for region in regions:
            if region["label"] not in NON_DESCRIPTIVE_ANSWERS:
                shares[region["label"]] = shares.get(region["label"], 0.0) + region["share"]
        return sorted(shares.items(), key=lambda item: -item[1])

    @classmethod
    def _dominant_label(cls, regions: list[dict[str, Any]]) -> str | None:
        ranked = cls._label_shares(regions)
        return ranked[0][0] if ranked else None

    @staticmethod
    def _sector(normalized_bbox: list[float]) -> str:
        """Compass sector of a normalised [x1, y1, x2, y2] box's centre."""
        cx = (normalized_bbox[0] + normalized_bbox[2]) / 2.0
        cy = (normalized_bbox[1] + normalized_bbox[3]) / 2.0
        horiz = "west" if cx < 0.35 else ("east" if cx > 0.65 else "central")
        vert = "north" if cy < 0.35 else ("south" if cy > 0.65 else "central")

        if horiz == "central" and vert == "central":
            return "the central sector"
        if horiz == "central":
            return f"the {vert} sector"
        if vert == "central":
            return f"the {horiz} sector"
        return f"the {vert}-{horiz} quadrant"

    @classmethod
    def _describe(
        cls,
        answer: str,
        change_percentage: float,
        regions: list[dict[str, Any]],
        total_regions: int,
    ) -> str:
        """Phrase the network's answer with the change its mask localised and
        what it says about the largest regions."""
        sentence = f"{answer[:1].upper()}{answer[1:]}."
        if not regions:
            if change_percentage > 0:
                return (
                    f"{sentence} The change mask marks {change_percentage}% of the scene as changed, "
                    "with no region large enough to localise."
                )
            return f"{sentence} The change mask marks no changed region."

        noun = "region" if total_regions == 1 else "regions"
        marked = f"; the {len(regions)} largest are marked" if total_regions > len(regions) else ""
        text = (
            f"{sentence} The change mask marks {change_percentage}% of the scene as changed, "
            f"in {total_regions} {noun}{marked}."
        )
        ranked = cls._label_shares(regions)[:2]
        if ranked:
            parts = " and ".join(f"{label} ({share:.0%} of the scene)" for label, share in ranked)
            text += (
                f" By region, the model describes the change mainly as {parts}; "
                f"the largest region is in {regions[0]['sector']}."
            )
        return text

    def predict(self, request: ModelRequest) -> ModelResponse:
        """Process a request and return a standardized ModelResponse (SpecialistModel protocol).

        Args:
            request: ModelRequest containing query, 2 image paths, and metadata.

        Returns:
            ModelResponse with answer, confidence, visual evidence, and execution trace.
        """
        if len(request.images) < 2:
            return ModelResponse(
                answer="Error: Change Detection / Change-VQA requires two bi-temporal images (T1 and T2).",
                confidence=0.0,
                evidence=[],
                model_name="Siamese-VLM-CDVQA",
                execution_time_ms=0.0,
            )

        img_t1_path = request.images[0]
        img_t2_path = request.images[1]

        result = self.analyze_pair(
            img_t1=img_t1_path,
            img_t2=img_t2_path,
            query=request.query,
        )

        # Build visual evidence
        evidence_list: list[Evidence] = []

        # 1. Change Heatmap / Mask evidence
        mask_prob = result["mask_prob"]
        mask_uint8 = (mask_prob * 255).astype(np.uint8)
        mask_b64 = ""
        if Image is not None:
            buf = io.BytesIO()
            Image.fromarray(mask_uint8).save(buf, format="PNG")
            mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        evidence_list.append(
            Evidence(
                type="mask",
                data={
                    "format": "png_base64",
                    "base64": mask_b64,
                    "resolution": [int(mask_prob.shape[1]), int(mask_prob.shape[0])],
                    "threshold": self.config.mask_threshold,
                },
                description="Dense bi-temporal change probability map highlighting regions of detected environmental transformation.",
            )
        )

        # 2. Bounding boxes evidence
        for box_info in result["bounding_boxes"]:
            evidence_list.append(
                Evidence(
                    type="bbox",
                    data={
                        "box_2d": box_info["bbox"],
                        "normalized_box_2d": box_info["normalized_bbox"],
                        "confidence": box_info["confidence"],
                        "area_pixels": box_info["area"],
                        "label": box_info["label"],
                    },
                    description=(
                    f"{box_info['label'][:1].upper()}{box_info['label'][1:]}: "
                    f"{box_info['share']:.1%} of the scene, in {box_info['sector']}."
                ),
                )
            )

        # 3. Numerical Metrics evidence
        evidence_list.append(
            Evidence(
                type="metrics",
                data={
                    "change_percentage": result["change_percentage"],
                    "changed_pixels": result["changed_pixels"],
                    "num_clusters": len(result["bounding_boxes"]),
                    "primary_class": result["primary_answer"],
                },
                description=f"Quantified change metrics: {result['change_percentage']}% of area altered across {len(result['bounding_boxes'])} clusters.",
            )
        )

        return ModelResponse(
            answer=result["answer"],
            confidence=result["confidence"],
            evidence=evidence_list,
            model_name="Siamese-VLM-CDVQA",
            execution_time_ms=result["execution_time_ms"],
        )


    @staticmethod
    def generate_evidence_visualization(
        img_t1: str | Path | Image.Image | np.ndarray,
        img_t2: str | Path | Image.Image | np.ndarray,
        result: dict[str, Any],
        save_path: str | Path = "change_evidence.png",
    ) -> Path:
        """Create a 3-panel evidence visualization: [ T1 | T2 | Change Grounding Overlay ]."""
        if Image is None or ImageDraw is None:
            return Path(save_path)

        def to_pil(img_in) -> Image.Image:
            if isinstance(img_in, (str, Path)):
                return Image.open(img_in).convert("RGB")
            elif isinstance(img_in, np.ndarray):
                return Image.fromarray(img_in).convert("RGB")
            elif isinstance(img_in, Image.Image):
                return img_in.convert("RGB")
            return Image.new("RGB", (256, 256), color="gray")

        p1 = to_pil(img_t1).resize((320, 320))
        p2 = to_pil(img_t2).resize((320, 320))
        overlay = p2.copy()

        # Render translucent change heatmap
        mask_prob = result["mask_prob"]
        mask_resized = Image.fromarray((mask_prob * 255).astype(np.uint8)).resize((320, 320))
        mask_arr = np.array(mask_resized) > (result.get("threshold", 0.5) * 255)

        overlay_np = np.array(overlay)
        # Tint changed pixels with vivid red/orange highlighting (R=255, G=60, B=40)
        overlay_np[mask_arr] = (
            overlay_np[mask_arr] * 0.45 + np.array([255, 60, 40]) * 0.55
        ).astype(np.uint8)
        overlay = Image.fromarray(overlay_np)

        # Draw bounding box rectangles and labels
        draw = ImageDraw.Draw(overlay)
        orig_w, orig_h = 320, 320
        for i, box in enumerate(result.get("bounding_boxes", [])):
            norm = box.get("normalized_bbox", [0, 0, 1, 1])
            bx1 = int(norm[0] * orig_w)
            by1 = int(norm[1] * orig_h)
            bx2 = int(norm[2] * orig_w)
            by2 = int(norm[3] * orig_h)

            # Draw outer rectangle (neon yellow) and inner line
            draw.rectangle([bx1, by1, bx2, by2], outline="#FFE600", width=3)
            # Label tag
            tag = f"Change #{i+1} ({box.get('confidence', 0.9):.0%})"
            draw.rectangle([bx1, max(0, by1 - 18), bx1 + len(tag) * 8, max(18, by1)], fill="#FFE600")
            draw.text((bx1 + 3, max(0, by1 - 16)), tag, fill="#000000")

        # Composite canvas (3 panels + header + footer)
        total_w = 320 * 3 + 40
        total_h = 320 + 130
        canvas = Image.new("RGB", (total_w, total_h), color="#0F172A")
        c_draw = ImageDraw.Draw(canvas)

        # Header Title
        c_draw.text((20, 16), "SATQUERY AI  BI-TEMPORAL CHANGE DETECTION GROUNDING", fill="#38BDF8")
        q_label = f"Query: \"{result.get('query', 'What changed between T1 and T2?')}\""
        c_draw.text((20, 38), q_label, fill="#E2E8F0")

        # Paste 3 panels
        canvas.paste(p1, (15, 65))
        canvas.paste(p2, (345, 65))
        canvas.paste(overlay, (675, 65))

        # Panel Headers
        c_draw.text((20, 70), "TIME 1 (Pre-change)", fill="#FFFFFF")
        c_draw.text((350, 70), "TIME 2 (Post-change)", fill="#FFFFFF")
        c_draw.text((680, 70), "GROUNDED CHANGE EVIDENCE", fill="#FFE600")

        # Footer Answer Box
        c_draw.rectangle([15, 395, total_w - 15, total_h - 15], fill="#1E293B", outline="#334155")
        ans_text = f"Result: {result.get('answer', '')}"
        conf_text = f"Confidence: {result.get('confidence', 0.9):.1%}  |  Area: {result.get('change_percentage', 0)}%  |  Latency: {result.get('execution_time_ms', 0):.0f} ms"
        c_draw.text((25, 405), ans_text[:130] + ("..." if len(ans_text) > 130 else ""), fill="#F8FAFC")
        c_draw.text((25, 425), conf_text, fill="#38BDF8")

        out_file = Path(save_path)
        canvas.save(out_file)
        return out_file


# ── CLI Interface ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference with Siamese Change-VQA specialist model")
    parser.add_argument("--t1", required=True, help="Path to Time 1 (pre-change) image")
    parser.add_argument("--t2", required=True, help="Path to Time 2 (post-change) image")
    parser.add_argument("--query", default="What changed between these two dates?", help="Natural language question")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint .pt file")
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--output-image", default="change_evidence.png", help="Path to save 3-panel visualization")
    args = parser.parse_args()

    model = ChangeVQAModel.from_checkpoint(args.checkpoint, device=args.device)
    result = model.analyze_pair(args.t1, args.t2, query=args.query)
    result["query"] = args.query

    # Generate visual evidence composite image
    saved_img = ChangeVQAModel.generate_evidence_visualization(
        img_t1=args.t1,
        img_t2=args.t2,
        result=result,
        save_path=args.output_image,
    )

    print("\n" + "=" * 65)
    print("SATQUERY AI  CHANGE DETECTION / CHANGE-VQA (UNIVERSITY DEMO)")
    print("=" * 65)
    print(f"Query:         {args.query}")
    print(f"Answer:        {result['answer']}")
    print(f"Confidence:    {result['confidence']:.1%}")
    print(f"Change Area:   {result['change_percentage']}% ({result['changed_pixels']} pixels)")
    print(f"Change Type:   {result.get('detected_domain', 'N/A').replace('_', ' ').title()}")
    print(f"BBoxes Found:  {len(result['bounding_boxes'])}")
    for i, b in enumerate(result['bounding_boxes']):
        print(f"  Box #{i+1}: {b['bbox']} | Conf: {b['confidence']:.2f} | Area: {b['area']} px")
    print(f"Visual Proof:  {saved_img}")
    print(f"Latency:       {result['execution_time_ms']:.1f} ms")
    print("=" * 65 + "\n")
