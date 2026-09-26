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
    ) -> None:
        self.device = device
        self.config = config or ModelConfig()
        self.image_size = image_size
        self.model = model.to(self.device).eval()

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
            print(f"[ChangeVQAModel] Loaded checkpoint from {checkpoint_path} on {device}")
        else:
            config = ModelConfig()
            model = SiameseChangeVQA(config)
            print(f"[ChangeVQAModel] Initialized model on {device} (no checkpoint specified or found)")

        return cls(model=model, config=config, device=device)

    def _load_and_preprocess_image(self, img_input: str | Path | Image.Image | np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        """Load an image and convert to preprocessed tensor (1, 3, H, W)."""
        orig_size = (256, 256)
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
            orig_size = img.size
        elif isinstance(img_input, np.ndarray):
            if img_input.dtype != np.uint8:
                mn, mx = img_input.min(), img_input.max()
                scaled = ((img_input - mn) / (mx - mn + 1e-6) * 255.0).astype(np.uint8) if mx > mn else np.zeros_like(img_input, dtype=np.uint8)
                img = Image.fromarray(scaled).convert("RGB")
            else:
                img = Image.fromarray(img_input).convert("RGB")
            orig_size = img.size
        elif Image is not None and isinstance(img_input, Image.Image):
            img = img_input.convert("RGB")
            orig_size = img.size
        else:
            raise ValueError(f"Unsupported image input type: {type(img_input)}")

        # Resize to model resolution
        resized = img.resize((self.image_size, self.image_size))
        arr = np.array(resized).astype(np.float32) / 255.0  # (H, W, 3)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)    # (3, H, W)
        tensor = normalize_image(tensor).unsqueeze(0)       # (1, 3, H, W)
        return tensor.to(self.device), orig_size

    def analyze_pair(
        self,
        img_t1: str | Path | Image.Image | np.ndarray,
        img_t2: str | Path | Image.Image | np.ndarray,
        query: str = "What changed between these two dates?",
    ) -> dict[str, Any]:
        """Run inference on an image pair and query string."""
        start_time = time.perf_counter()

        t1_tensor, orig_size = self._load_and_preprocess_image(img_t1)
        t2_tensor, _ = self._load_and_preprocess_image(img_t2)

        with torch.no_grad():
            outputs = self.model(
                t1=t1_tensor,
                t2=t2_tensor,
                question_text=[query],
            )

        exec_time_ms = (time.perf_counter() - start_time) * 1000.0

        # Everything reported comes from the network: its answer with its top-1
        # softmax confidence, and its change mask. The mask is not blended with a
        # raw pixel difference, and the answer is not replaced by keyword rules or
        # fixed confidences, so callers see what the fine-tuned model predicted.
        answer_text = outputs["predicted_answer_text"][0]
        answer_confidence = float(outputs["answer_confidence"][0])
        mask_prob = outputs["change_mask_prob"][0, 0].float().cpu().numpy()

        bboxes = self.model.extract_bounding_boxes(
            mask_prob,
            threshold=self.config.mask_threshold,
            min_area=35,
            max_boxes=8,
        )

        changed_pixel_count = int((mask_prob >= self.config.mask_threshold).sum())
        change_percentage = round((changed_pixel_count / mask_prob.size) * 100.0, 2)

        return {
            "answer": self._describe(answer_text, change_percentage, bboxes),
            "primary_answer": answer_text,
            "confidence": round(answer_confidence, 4),
            "change_percentage": change_percentage,
            "changed_pixels": changed_pixel_count,
            # Trained on LEVIR-CD, whose masks label building change only.
            "detected_domain": "building_change" if bboxes else "unchanged",
            "bounding_boxes": bboxes,
            "mask_prob": mask_prob,
            "execution_time_ms": round(exec_time_ms, 2),
            "original_size": orig_size,
        }

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
        bboxes: list[dict[str, Any]],
    ) -> str:
        """Phrase the network's answer with the change its mask localised."""
        sentence = f"{answer[:1].upper()}{answer[1:]}."
        if bboxes:
            sectors = ", ".join(dict.fromkeys(cls._sector(b["normalized_bbox"]) for b in bboxes[:3]))
            return (
                f"{sentence} The change mask marks {change_percentage}% of the scene as changed, "
                f"in {len(bboxes)} region(s), mainly {sectors}."
            )
        if change_percentage > 0:
            return (
                f"{sentence} The change mask marks {change_percentage}% of the scene as changed, "
                "with no region large enough to localise."
            )
        return f"{sentence} The change mask marks no changed region."

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
                    description=f"Detected change cluster with confidence {box_info['confidence']:.2f} ({box_info['area']} px).",
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
