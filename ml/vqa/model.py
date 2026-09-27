"""
Vision Encoder + VLM Head for Remote Sensing VQA.

Architecture:
    Input Image (3, H, W) ──────┐
                                ▼
                       ┌─────────────────┐
                       │ Vision Encoder  │ (ResNet/ConvNeXt backbone)
                       │                 │ Multi-scale feature extraction
                       └─────────────────┘
                                │
                                ▼
                      (Project to visual tokens)
                                │
                                ▼
                       ┌─────────────────────────────────┐
                       │ Vision-Language (VLM) Head      │
                       │ - Question Text Tokenizer/Embed │
                       │ - Visual-Text Cross-Attention   │
                       │ - Answer Classification / Gen   │
                       └─────────────────────────────────┘
                                │
                                ▼
                  Predicted Answer + Confidence
                  ("Residential Area")
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ml.vqa.config import ModelConfig


# ── Vision Backbone ────────────────────────────────────────────────────────

class VisionBackbone(nn.Module):
    """Backbone for single image feature extraction.

    Extracts features at multiple spatial resolutions, though typically
    only the final stage (stage4) is used for standard VQA visual tokens.
    """

    def __init__(self, backbone_name: str = "convnext_tiny", pretrained: bool = True, in_channels: int = 3) -> None:
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        self.backbone_type = "convnext" if "convnext" in backbone_name else "resnet"

        if backbone_name == "convnext_tiny":
            base = models.convnext_tiny(weights=weights)
            self.dims = [96, 192, 384, 768]
            self.stage1 = nn.Sequential(base.features[0], base.features[1])
            self.stage2 = nn.Sequential(base.features[2], base.features[3])
            self.stage3 = nn.Sequential(base.features[4], base.features[5])
            self.stage4 = nn.Sequential(base.features[6], base.features[7])

            if in_channels != 3:
                old_conv = base.features[0][0]
                new_conv = nn.Conv2d(
                    in_channels,
                    old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None,
                )
                if pretrained:
                    with torch.no_grad():
                        new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
                        if old_conv.bias is not None:
                            new_conv.bias[:] = old_conv.bias
                self.stage1[0][0] = new_conv

        elif backbone_name in ("resnet18", "resnet34", "resnet50"):
            if backbone_name == "resnet18":
                base = models.resnet18(weights=weights)
                self.dims = [64, 128, 256, 512]
            elif backbone_name == "resnet34":
                base = models.resnet34(weights=weights)
                self.dims = [64, 128, 256, 512]
            elif backbone_name == "resnet50":
                base = models.resnet50(weights=weights)
                self.dims = [256, 512, 1024, 2048]

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

            self.stage1 = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool, base.layer1,
            )
            self.stage2 = base.layer2
            self.stage3 = base.layer3
            self.stage4 = base.layer4
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Choose convnext_tiny/resnet18/34/50.")

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale features for a single image."""
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        return [s1, s2, s3, s4]


# ── Text Tokenizer & Encoder ─────────────────────────────────────────────

class SimpleTokenizer:
    """Lightweight rule-based tokenizer for remote sensing VQA queries."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"

    DEFAULT_VOCAB = [
        "<pad>", "<unk>", "<sos>", "<eos>",
        "what", "where", "how", "is", "are", "there", "any", "the",
        "built-up", "building", "buildings", "area", "urban", "road", "roads",
        "vegetation", "forest", "trees", "water", "waterbody", "river", "lake",
        "north", "south", "east", "west", "center", "quadrant", "percentage", "count",
        "residential", "industrial", "agricultural", "barren", "land", "cover", "ground",
        "yes", "no", "type", "many", "visible", "scene", "describe"
    ]

    def __init__(self, vocab: list[str] | None = None) -> None:
        raw_vocab = self.DEFAULT_VOCAB if vocab is None else vocab
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
        seq_len = input_ids.shape[1]
        tokens = self.token_embed(input_ids)
        x = tokens + self.pos_embed[:, :seq_len, :]
        return self.norm(x)


# ── Vision-Language Cross-Modal Fusion ───────────────────────────────────

class CrossModalAttentionLayer(nn.Module):
    """Cross-attention between question tokens and visual tokens."""

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

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
        )
        x = self.norm1(query + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


class CrossModalFusion(nn.Module):
    """Stacks cross-attention layers fusing visual tokens with question tokens."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, num_layers: int = 2, ff_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            CrossModalAttentionLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, text_tokens: torch.Tensor, visual_tokens: torch.Tensor) -> torch.Tensor:
        x = text_tokens
        for layer in self.layers:
            x = layer(query=x, key_value=visual_tokens)
        return x


# ── Canonical Answers & Classification Head ──────────────────────────────

CANONICAL_ANSWERS = [
    "yes",
    "no",
    "residential",
    "industrial",
    "agricultural",
    "commercial",
    "forest",
    "water",
    "bare land",
    "road",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "more than 10",
    "sparse",
    "dense",
]


class VQAHead(nn.Module):
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
        logits = self.mlp(fused_features)
        probs = F.softmax(logits, dim=-1)
        confidence, _ = probs.max(dim=-1)
        return logits, confidence


# ── Full End-to-End VQA Model ────────────────────────────

class VQAModel(nn.Module):
    """End-to-end Vision Encoder + VLM Head for single-image VQA."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()

        # 1. Vision Backbone
        self.backbone = VisionBackbone(
            backbone_name=self.config.backbone,
            pretrained=self.config.pretrained,
            in_channels=self.config.in_channels,
        )

        # 2. Text Tokenizer & Encoder
        self.tokenizer = SimpleTokenizer()
        self.text_encoder = TextEncoder(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.text_embed_dim,
            max_len=self.config.max_question_length,
        )

        # 3. Visual patch projector (map deep features to text embedding dimension)
        self.visual_projector = nn.Sequential(
            nn.Conv2d(self.backbone.dims[-1], self.config.text_embed_dim, kernel_size=1),
            nn.BatchNorm2d(self.config.text_embed_dim),
            nn.ReLU(inplace=True),
        )

        # 4. Cross-Modal Fusion
        self.cross_modal_fusion = CrossModalFusion(
            embed_dim=self.config.text_embed_dim,
            num_heads=self.config.num_cross_attention_heads,
            num_layers=self.config.cross_attention_layers,
            ff_dim=self.config.feedforward_dim,
            dropout=self.config.dropout,
        )

        # 5. VQA Answer Head
        self.vqa_head = VQAHead(
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
        image: torch.Tensor,
        question_ids: torch.Tensor | None = None,
        question_text: list[str] | str | None = None,
    ) -> dict[str, Any]:
        """Forward pass for single-image VQA.

        Args:
            image: (B, 3, H, W) Image tensor.
            question_ids: Optional (B, L) encoded question token IDs.
            question_text: Optional question string or list of strings.

        Returns:
            Dict containing:
                - 'answer_logits': (B, num_classes)
                - 'answer_confidence': (B,)
                - 'predicted_answer_idx': (B,)
                - 'predicted_answer_text': list[str]
        """
        device = image.device
        batch_size = image.shape[0]

        # 1. Vision feature extraction
        feats = self.backbone(image)
        deep_feats = feats[-1]  # Use Stage 4 features

        # 2. Visual Tokens Formulation
        proj_visual = self.visual_projector(deep_feats)  # (B, text_dim, H/32, W/32)
        b, c, h, w = proj_visual.shape
        visual_tokens = proj_visual.flatten(2).permute(0, 2, 1)  # (B, N_v, text_dim)

        # 3. Text Processing
        if question_ids is None:
            if question_text is None:
                question_text = ["What is visible in this image?"] * batch_size
            elif isinstance(question_text, str):
                question_text = [question_text] * batch_size

            question_ids, _ = self.tokenizer.batch_encode(
                question_text,
                max_length=self.config.max_question_length,
                device=str(device),
            )

        text_tokens = self.text_encoder(question_ids)  # (B, L_q, text_dim)

        # 4. Cross-Modal Fusion
        fused_text = self.cross_modal_fusion(text_tokens=text_tokens, visual_tokens=visual_tokens)

        # Mean-pool over real tokens only.
        token_mask = (question_ids != self.tokenizer.pad_id).unsqueeze(-1).to(fused_text.dtype)
        pooled = (fused_text * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp(min=1.0)  # (B, text_dim)

        # 5. Answer Prediction
        logits, confidence = self.vqa_head(pooled)
        pred_idx = logits.argmax(dim=-1)
        pred_answers = [self.answers_vocab[int(idx.item())] for idx in pred_idx]

        return {
            "answer_logits": logits,
            "answer_confidence": confidence,
            "predicted_answer_idx": pred_idx,
            "predicted_answer_text": pred_answers,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> "VQAModel":
        """Load model from a saved checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = ModelConfig(**checkpoint.get("config", {}))
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model

# ── Multi-Task Loss Function ─────────────────────────────────────────────

class VQALoss(nn.Module):
    """Loss for VQA."""

    def __init__(
        self,
        vqa_weight: float = 1.0,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.vqa_weight = vqa_weight
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, preds: dict[str, Any], targets: dict[str, Any]) -> dict[str, torch.Tensor]:
        loss_vqa = self.ce_loss(preds["answer_logits"], targets["answer_label"])
        total_loss = self.vqa_weight * loss_vqa

        return {
            "loss": total_loss,
            "loss_vqa": loss_vqa,
        }
