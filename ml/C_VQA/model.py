"""
Siamese Vision Encoder + VLM Head for Remote Sensing Change-VQA.

Architecture:
    Time 1 Image (T1) ──────┐
    (3, H, W)               ▼
                   ┌─────────────────┐
                   │ Siamese Vision  │ (Shared ResNet backbone)
                   │ Encoder         │ Multi-scale feature extraction
                   └─────────────────┘
                            ▲
    Time 2 Image (T2) ──────┘
    (3, H, W)
             │
             ▼
     ┌─────────────────────────────────────────────────────────┐
     │ Bi-temporal Difference & Interaction Module             │
     │ - Multi-scale difference: |F1_i - F2_i|                 │
     │ - Channel concatenation: [F1, F2, |F1 - F2|]            │
     │ - Spatial-temporal cross attention correlation          │
     └─────────────────────────────────────────────────────────┘
             │
             ├─────────────────────────────────────────┐
             ▼                                         ▼
    ┌──────────────────────────────────┐     ┌─────────────────────────────────┐
    │ Change Grounding / Mask Head     │     │ Vision-Language (VLM) Head      │
    │ Multi-scale decoder upsampling   │     │ - Question Text Tokenizer/Embed │
    │ Output: (1, H, W) change heatmap │     │ - Visual-Text Cross-Attention   │
    │ Bounding box extraction          │     │ - Answer Classification / Gen   │
    └──────────────────────────────────┘     └─────────────────────────────────┘
             │                                         │
             ▼                                         ▼
      Change Mask & BBoxes                      Predicted Answer + Confidence
    (Evidence: Visual Grounding)              ("Built-up area increased by 15%")
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ml.C_VQA.config import ModelConfig


# ── Siamese Vision Backbone ──────────────────────────────────────────────

class SiameseBackbone(nn.Module):
    """Twin weight-sharing backbone for bi-temporal image feature extraction.

    Extracts features at multiple spatial resolutions (stages 1, 2, 3, 4)
    to support both dense spatial mask prediction and high-level VLM reasoning.
    """

    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True, in_channels: int = 3) -> None:
        super().__init__()
        weights = "DEFAULT" if pretrained else None

        if backbone_name == "resnet18":
            base = models.resnet18(weights=weights)
            self.dims = [64, 128, 256, 512]
        elif backbone_name == "resnet34":
            base = models.resnet34(weights=weights)
            self.dims = [64, 128, 256, 512]
        elif backbone_name == "resnet50":
            base = models.resnet50(weights=weights)
            self.dims = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Choose resnet18/34/50.")

        if in_channels != 3:
            old_conv = base.conv1
            base.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    base.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        # Stage 1: stem + layer1 (stride 4, H/4, W/4)
        self.stage1 = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
        )
        # Stage 2: layer2 (stride 8, H/8, W/8)
        self.stage2 = base.layer2
        # Stage 3: layer3 (stride 16, H/16, W/16)
        self.stage3 = base.layer3
        # Stage 4: layer4 (stride 32, H/32, W/32)
        self.stage4 = base.layer4

    def forward_single(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale features for a single image."""
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        return [s1, s2, s3, s4]

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Shared forward pass for both time steps (Siamese)."""
        feats_t1 = self.forward_single(t1)
        feats_t2 = self.forward_single(t2)
        return feats_t1, feats_t2


# ── Bi-temporal Difference & Interaction Module ──────────────────────────

class BitemporalDifferenceModule(nn.Module):
    """Computes multi-scale difference and temporal attention interaction.

    Combines subtraction |F1 - F2| and concatenation [F1, F2] with a channel
    attention gate to emphasize significant environmental transformations.
    """

    def __init__(self, in_dims: list[int], out_dim: int = 256) -> None:
        super().__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim

        # 1x1 convs to unify intermediate difference channels
        self.proj_s1 = nn.Conv2d(in_dims[0], 64, kernel_size=1)
        self.proj_s2 = nn.Conv2d(in_dims[1], 128, kernel_size=1)
        self.proj_s3 = nn.Conv2d(in_dims[2], 256, kernel_size=1)

        # Deep stage 4 fusion: [F1, F2, |F1 - F2|] -> out_dim
        self.deep_proj = nn.Sequential(
            nn.Conv2d(in_dims[3] * 3, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

        # Channel Attention Gate for deep difference features
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_dim, out_dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim // 4, out_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        feats_t1: list[torch.Tensor],
        feats_t2: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Compute difference features.

        Returns:
            deep_fused: (B, out_dim, H/32, W/32) visual change features.
            diff_stages: Multi-scale difference feature maps [s1, s2, s3].
        """
        # Absolute differences at each scale
        diff_s1 = self.proj_s1(torch.abs(feats_t1[0] - feats_t2[0]))
        diff_s2 = self.proj_s2(torch.abs(feats_t1[1] - feats_t2[1]))
        diff_s3 = self.proj_s3(torch.abs(feats_t1[2] - feats_t2[2]))

        # Stage 4 concatenation + difference
        f1_s4 = feats_t1[3]
        f2_s4 = feats_t2[3]
        abs_diff_s4 = torch.abs(f1_s4 - f2_s4)
        cat_s4 = torch.cat([f1_s4, f2_s4, abs_diff_s4], dim=1)

        deep_fused = self.deep_proj(cat_s4)
        gate = self.channel_gate(deep_fused)
        deep_fused = deep_fused * gate + deep_fused

        return deep_fused, [diff_s1, diff_s2, diff_s3]


# ── Change Mask Grounding Head ──────────────────────────────────────────

class ChangeMaskHead(nn.Module):
    """Multi-scale decoder predicting pixel-wise change probability map (H, W).

    Upsamples deep change features while fusing skip connections from earlier
    stages, producing a high-resolution change segmentation mask.
    """

    def __init__(self, visual_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        # Level 4 to Level 3 (H/32 -> H/16)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(visual_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(hidden_dim + 256, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Level 3 to Level 2 (H/16 -> H/8)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(hidden_dim // 2 + 128, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
        )

        # Level 2 to Level 1 (H/8 -> H/4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(hidden_dim // 4 + 64, hidden_dim // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
        )

        # Final 4x upsampling to original image resolution (H/4 -> H)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_dim // 4, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, deep_feats: torch.Tensor, diff_stages: list[torch.Tensor]) -> torch.Tensor:
        """Decode change mask logits.

        Args:
            deep_feats: (B, visual_dim, H/32, W/32)
            diff_stages: [s1 (H/4), s2 (H/8), s3 (H/16)]

        Returns:
            logits: (B, 1, H, W)
        """
        x = self.up4(deep_feats)
        x = torch.cat([x, diff_stages[2]], dim=1)
        x = self.fuse3(x)

        x = self.up3(x)
        x = torch.cat([x, diff_stages[1]], dim=1)
        x = self.fuse2(x)

        x = self.up2(x)
        x = torch.cat([x, diff_stages[0]], dim=1)
        x = self.fuse1(x)

        logits = self.final_up(x)
        return logits


# ── Text Tokenizer & Encoder ─────────────────────────────────────────────

class SimpleTokenizer:
    """Lightweight rule-based tokenizer for remote sensing Change-VQA queries.

    Builds an in-memory vocabulary of words from question-answer pairs and maps
    text strings to fixed-length integer token sequences.
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"

    DEFAULT_VOCAB = [
        "<pad>", "<unk>", "<sos>", "<eos>",
        "what", "where", "how", "has", "did", "is", "are", "there", "any", "the",
        "changed", "change", "between", "these", "two", "dates", "images", "t1", "t2",
        "built-up", "building", "buildings", "area", "urban", "expansion", "road", "roads",
        "vegetation", "forest", "trees", "water", "waterbody", "river", "lake", "flood",
        "increased", "decreased", "unchanged", "no", "yes", "construction", "new",
        "cleared", "destroyed", "submerged", "disappeared", "appeared", "loss", "gain",
        "north", "south", "east", "west", "center", "quadrant", "percentage", "count",
        "residential", "industrial", "agricultural", "barren", "land", "cover", "ground",
    ]

    def __init__(self, vocab: list[str] | None = None) -> None:
        raw_vocab = self.DEFAULT_VOCAB if vocab is None else vocab
        # Ensure unique preserving order
        seen = set()
        self.vocab = []
        for w in raw_vocab:
            if w not in seen:
                seen.add(w)
                self.vocab.append(w)

        self.w2i = {w: i for i, w in enumerate(self.vocab)}
        self.i2w = {i: w for i, w in enumerate(self.vocab)}

    @property
    def pad_id(self) -> int:
        return self.w2i[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.w2i[self.UNK_TOKEN]

    def tokenize(self, text: str) -> list[str]:
        """Normalize and split text into tokens."""
        clean = text.lower().replace("?", " ? ").replace(".", " . ").replace(",", " , ").replace("-", " - ")
        return [w.strip() for w in clean.split() if w.strip()]

    def encode(self, text: str, max_length: int = 32) -> tuple[list[int], list[int]]:
        """Encode text to token ids and attention mask."""
        tokens = self.tokenize(text)
        tokens = [self.SOS_TOKEN] + tokens[: max_length - 2] + [self.EOS_TOKEN]
        ids = [self.w2i.get(t, self.unk_id) for t in tokens]
        mask = [1] * len(ids)

        # Pad to max_length
        if len(ids) < max_length:
            pad_len = max_length - len(ids)
            ids.extend([self.pad_id] * pad_len)
            mask.extend([0] * pad_len)

        return ids, mask

    def batch_encode(self, texts: list[str], max_length: int = 32, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of texts into tensors."""
        all_ids = []
        all_masks = []
        for t in texts:
            ids, mask = self.encode(t, max_length=max_length)
            all_ids.append(ids)
            all_masks.append(mask)
        return torch.tensor(all_ids, dtype=torch.long, device=device), torch.tensor(all_masks, dtype=torch.bool, device=device)


class TextEncoder(nn.Module):
    """Encodes tokenized natural language question into continuous representations."""

    def __init__(self, vocab_size: int = 2000, embed_dim: int = 256, max_len: int = 32) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token IDs (B, L) -> Text embeddings (B, L, embed_dim)."""
        seq_len = input_ids.shape[1]
        tokens = self.token_embed(input_ids)
        x = tokens + self.pos_embed[:, :seq_len, :]
        return self.norm(x)


# ── Vision-Language Cross-Modal Fusion ───────────────────────────────────

class CrossModalAttentionLayer(nn.Module):
    """Bidirectional cross-attention between question tokens and visual change tokens."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, ff_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Query attends to Key-Value with residual connection."""
        attn_out, _ = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
        )
        x = self.norm1(query + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


class CrossModalFusion(nn.Module):
    """Stacks cross-attention layers fusing visual change tokens with question tokens."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, num_layers: int = 2, ff_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            CrossModalAttentionLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        text_tokens: torch.Tensor | None = None,
        visual_tokens: torch.Tensor | None = None,
        query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Question queries the visual change tokens."""
        q = text_tokens if text_tokens is not None else query
        if q is None or visual_tokens is None:
            raise ValueError("Must provide text_tokens/query and visual_tokens to CrossModalFusion.")
        x = q
        for layer in self.layers:
            x = layer(query=x, key_value=visual_tokens)
        return x


# ── Canonical Answers & Classification Head ──────────────────────────────

CANONICAL_ANSWERS = [
    "unchanged",
    "increased",
    "decreased",
    "new built-up area constructed",
    "building expansion",
    "vegetation cleared / deforestation",
    "vegetation regrowth / afforestation",
    "water body expanded / flooded",
    "water body receded / drought",
    "road newly constructed",
    "agricultural land converted to urban",
    "bare soil to built-up",
    "demolition of existing structures",
    "yes",
    "no",
    "north-east",
    "north-west",
    "south-east",
    "south-west",
    "center",
    "large scale change",
    "moderate change",
    "minor change",
    "no significant change detected",
    "1",
    "2",
    "3",
    "4",
    "5 or more",
]


class ChangeVQAHead(nn.Module):
    """Predicts answer logits and confidence scores from fused cross-modal features."""

    def __init__(self, in_dim: int = 256, hidden_dim: int = 256, num_classes: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, fused_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute answer logits and softmax confidence.

        Args:
            fused_features: (B, in_dim) pooled cross-modal representation.

        Returns:
            logits: (B, num_classes)
            confidence: (B,) top-1 probability
        """
        logits = self.mlp(fused_features)
        probs = F.softmax(logits, dim=-1)
        confidence, _ = probs.max(dim=-1)
        return logits, confidence


# ── Full End-to-End Siamese Change-VQA Model ────────────────────────────

class SiameseChangeVQA(nn.Module):
    """End-to-end Siamese Vision Encoder + VLM Head for bi-temporal Change-VQA.

    Combines:
        1. SiameseBackbone (shared weights for T1 and T2 images)
        2. BitemporalDifferenceModule (multi-scale differences and correlation)
        3. ChangeMaskHead (pixel-level change probability map and bounding box grounding)
        4. TextEncoder (question tokenization and projection)
        5. CrossModalFusion (transformer cross-attention between questions and change visual tokens)
        6. ChangeVQAHead (answer prediction and confidence estimation)
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()

        # 1. Siamese Backbone
        self.backbone = SiameseBackbone(
            backbone_name=self.config.backbone,
            pretrained=self.config.pretrained,
            in_channels=self.config.in_channels,
        )

        # 2. Bi-temporal Difference
        self.difference_module = BitemporalDifferenceModule(
            in_dims=self.backbone.dims,
            out_dim=self.config.visual_feature_dim,
        )

        # 3. Dense Change Mask Head
        self.mask_head = ChangeMaskHead(
            visual_dim=self.config.visual_feature_dim,
            hidden_dim=self.config.mask_hidden_dim,
        )

        # 4. Text Tokenizer & Encoder
        self.tokenizer = SimpleTokenizer()
        self.text_encoder = TextEncoder(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.text_embed_dim,
            max_len=self.config.max_question_length,
        )

        # Visual patch projector to align visual dim with text embed dim
        self.visual_projector = nn.Sequential(
            nn.Conv2d(self.config.visual_feature_dim, self.config.text_embed_dim, kernel_size=1),
            nn.BatchNorm2d(self.config.text_embed_dim),
            nn.ReLU(inplace=True),
        )

        # 5. Cross-Modal Fusion
        self.cross_modal_fusion = CrossModalFusion(
            embed_dim=self.config.text_embed_dim,
            num_heads=self.config.num_cross_attention_heads,
            num_layers=self.config.cross_attention_layers,
            ff_dim=self.config.feedforward_dim,
            dropout=self.config.dropout,
        )

        # 6. VQA Answer Head
        self.vqa_head = ChangeVQAHead(
            in_dim=self.config.text_embed_dim,
            hidden_dim=self.config.answer_hidden_dim,
            num_classes=self.config.num_classes,
            dropout=self.config.dropout,
        )

        # Canonical vocabulary of answer labels
        self.answers_vocab = CANONICAL_ANSWERS.copy()
        while len(self.answers_vocab) < self.config.num_classes:
            self.answers_vocab.append(f"answer_category_{len(self.answers_vocab)}")

    def forward(
        self,
        t1: torch.Tensor,
        t2: torch.Tensor,
        question_ids: torch.Tensor | None = None,
        question_text: list[str] | str | None = None,
    ) -> dict[str, Any]:
        """Forward pass for bi-temporal Change-VQA.

        Args:
            t1: (B, 3, H, W) Time 1 image tensor.
            t2: (B, 3, H, W) Time 2 image tensor.
            question_ids: Optional (B, L) encoded question token IDs.
            question_text: Optional question string or list of strings.

        Returns:
            Dict containing:
                - 'answer_logits': (B, num_classes)
                - 'answer_confidence': (B,)
                - 'change_mask_logits': (B, 1, H, W)
                - 'change_mask_prob': (B, 1, H, W)
                - 'predicted_answer_idx': (B,)
                - 'predicted_answer_text': list[str]
        """
        device = t1.device
        batch_size = t1.shape[0]

        # 1. Siamese feature extraction
        feats_t1, feats_t2 = self.backbone(t1, t2)

        # 2. Bi-temporal difference & interaction
        deep_fused, diff_stages = self.difference_module(feats_t1, feats_t2)

        # 3. Dense Change Grounding Mask
        mask_logits = self.mask_head(deep_fused, diff_stages)
        mask_prob = torch.sigmoid(mask_logits)

        # 4. Text Processing
        if question_ids is None:
            if question_text is None:
                question_text = ["What changed between these two dates?"] * batch_size
            elif isinstance(question_text, str):
                question_text = [question_text] * batch_size

            question_ids, _ = self.tokenizer.batch_encode(
                question_text,
                max_length=self.config.max_question_length,
                device=str(device),
            )

        text_tokens = self.text_encoder(question_ids)  # (B, L_q, text_dim)

        # 5. Visual Tokens Formulation
        proj_visual = self.visual_projector(deep_fused)  # (B, text_dim, H/32, W/32)
        b, c, h, w = proj_visual.shape
        visual_tokens = proj_visual.flatten(2).permute(0, 2, 1)  # (B, N_v, text_dim)

        # 6. Cross-Modal Fusion
        fused_text = self.cross_modal_fusion(query=text_tokens, visual_tokens=visual_tokens)

        # Pool fused representation (e.g. SOS token or mean pool)
        pooled = fused_text.mean(dim=1)  # (B, text_dim)

        # 7. Answer Prediction
        logits, confidence = self.vqa_head(pooled)
        pred_idx = logits.argmax(dim=-1)
        pred_answers = [self.answers_vocab[int(idx.item())] for idx in pred_idx]

        return {
            "answer_logits": logits,
            "answer_confidence": confidence,
            "change_mask_logits": mask_logits,
            "change_mask_prob": mask_prob,
            "predicted_answer_idx": pred_idx,
            "predicted_answer_text": pred_answers,
        }

    @staticmethod
    def extract_bounding_boxes(
        mask_prob: torch.Tensor | np.ndarray,
        threshold: float = 0.5,
        min_area: int = 50,
        max_boxes: int = 10,
    ) -> list[dict[str, Any]]:
        """Extract bounding boxes and statistics from change probability map.

        Args:
            mask_prob: (H, W) or (1, H, W) numpy array or torch tensor in [0, 1].
            threshold: Probability threshold for binary change detection.
            min_area: Minimum pixel area to consider a valid change cluster.
            max_boxes: Maximum number of bounding boxes to return.

        Returns:
            List of dicts:
                - 'bbox': [x1, y1, x2, y2] in normalized [0, 1] or pixel coords
                - 'area': pixel count
                - 'confidence': mean probability within the component
                - 'label': description of changed region
        """
        if isinstance(mask_prob, torch.Tensor):
            arr: np.ndarray = mask_prob.squeeze().detach().cpu().numpy()
        else:
            arr = np.squeeze(np.asarray(mask_prob))

        h, w = arr.shape
        binary = (arr >= threshold).astype(np.uint8)

        boxes = []
        try:
            import cv2  # type: ignore[import-untyped]
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < min_area:
                    continue
                x = int(stats[i, cv2.CC_STAT_LEFT])
                y = int(stats[i, cv2.CC_STAT_TOP])
                width = int(stats[i, cv2.CC_STAT_WIDTH])
                height = int(stats[i, cv2.CC_STAT_HEIGHT])
                x1, y1, x2, y2 = x, y, x + width, y + height

                # Compute mean confidence inside component
                comp_mask = (labels == i)
                conf = float(arr[comp_mask].mean())

                boxes.append({
                    "bbox": [x1, y1, x2, y2],
                    "normalized_bbox": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)],
                    "area": int(area),
                    "confidence": round(conf, 4),
                    "label": "detected_change_region",
                })
        except ImportError:
            # Pure Python/NumPy 8-connectivity connected components fallback
            visited = np.zeros((h, w), dtype=bool)
            ys, xs = np.where(binary > 0)
            coords = list(zip(ys, xs))
            for start_y, start_x in coords:
                if visited[start_y, start_x]:
                    continue
                queue = [(start_y, start_x)]
                visited[start_y, start_x] = True
                comp_pixels = []
                while queue:
                    cy, cx = queue.pop(0)
                    comp_pixels.append((cy, cx))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] > 0 and not visited[ny, nx]:
                                visited[ny, nx] = True
                                queue.append((ny, nx))

                if len(comp_pixels) >= min_area:
                    py = [p[0] for p in comp_pixels]
                    px = [p[1] for p in comp_pixels]
                    x1, y1, x2, y2 = int(min(px)), int(min(py)), int(max(px) + 1), int(max(py) + 1)
                    conf = float(np.mean([arr[p[0], p[1]] for p in comp_pixels]))
                    boxes.append({
                        "bbox": [x1, y1, x2, y2],
                        "normalized_bbox": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)],
                        "area": len(comp_pixels),
                        "confidence": round(conf, 4),
                        "label": "detected_change_region",
                    })

        # Sort by area descending and cap
        boxes = sorted(boxes, key=lambda b: b["area"], reverse=True)[:max_boxes]
        return boxes


# ── Multi-Task Loss Function ─────────────────────────────────────────────

class SoftDiceLoss(nn.Module):
    """Soft Dice loss for binary change segmentation."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        intersection = (pred_prob * target).sum(dim=(1, 2, 3))
        union = pred_prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class ChangeVQALoss(nn.Module):
    """Joint Multi-Task Loss for Change-VQA.

    L_total = lambda_vqa * L_CE(vqa) + lambda_bce * L_BCE(mask) + lambda_dice * L_Dice(mask)
    """

    def __init__(
        self,
        vqa_weight: float = 1.0,
        mask_bce_weight: float = 0.5,
        mask_dice_weight: float = 0.5,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.vqa_weight = vqa_weight
        self.mask_bce_weight = mask_bce_weight
        self.mask_dice_weight = mask_dice_weight

        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = SoftDiceLoss()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        answer_targets: torch.Tensor,
        mask_targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss and return scalar loss plus metric breakdown."""
        # 1. VQA Classification Loss
        vqa_loss = self.ce_loss(outputs["answer_logits"], answer_targets)

        # Accuracy
        preds = outputs["answer_logits"].argmax(dim=-1)
        vqa_acc = (preds == answer_targets).float().mean().item()

        total_loss = self.vqa_weight * vqa_loss
        metrics = {
            "loss_vqa": vqa_loss.item(),
            "vqa_accuracy": vqa_acc,
        }

        # 2. Change Mask Segmentation Loss (if mask ground truth available)
        if mask_targets is not None:
            mask_logits = outputs["change_mask_logits"]
            mask_prob = outputs["change_mask_prob"]

            if mask_targets.dim() == 3:
                mask_targets = mask_targets.unsqueeze(1)

            bce = self.bce_loss(mask_logits, mask_targets)
            dice = self.dice_loss(mask_prob, mask_targets)
            mask_loss = self.mask_bce_weight * bce + self.mask_dice_weight * dice

            total_loss = total_loss + mask_loss

            # IoU metric
            with torch.no_grad():
                pred_bin = (mask_prob > 0.5).float()
                intersection = (pred_bin * mask_targets).sum().item()
                union = ((pred_bin + mask_targets) > 0).float().sum().item()
                iou = (intersection + 1e-6) / (union + 1e-6)

            metrics["loss_mask_bce"] = bce.item()
            metrics["loss_mask_dice"] = dice.item()
            metrics["mask_iou"] = iou

        metrics["loss_total"] = total_loss.item()
        return total_loss, metrics
