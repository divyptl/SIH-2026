"""
Dual-Encoder model for Optical-SAR contrastive alignment.

Architecture:
    ┌─────────────┐     ┌────────────────────┐      ┌───────────┐
    │ SAR image   │-->  │  SAR Encoder       │ -->  │           │
    │ (1, H, W)   │     │  (ResNet backbone) │      │ Shared    │
    └─────────────┘     │  + Projection MLP  │      │ Embedding │ --> Contrastive Loss
    ┌─────────────┐     ├────────────────────┤      │ Space     │
    │ Optical img │-->  │  Optical Encoder   │ -->  │ (256-dim) │
    │ (3, H, W)   │     │  (ResNet backbone) │      │           │
    └─────────────┘     │  + Projection MLP  │      └───────────┘
                        └────────────────────┘

The SAR and optical encoders share the same architecture but have separate
weights. Each encoder consists of a ResNet backbone followed by a 2-layer
projection MLP that maps features into a shared embedding space.

The NT-Xent contrastive loss pulls matching SAR-optical pairs together and
pushes non-matching pairs apart in the embedding space.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ── Backbone factory ────────────────────────────────────────────────────

def _make_backbone(name: str, pretrained: bool, in_channels: int) -> tuple[nn.Module, int]:
    """Create a ResNet backbone and return (backbone, feature_dim).

    Args:
        name: One of 'resnet18', 'resnet34', 'resnet50'.
        pretrained: Whether to load ImageNet pretrained weights.
        in_channels: Number of input channels (1 for SAR, 3 for optical).

    Returns:
        Tuple of (backbone_without_fc, output_feature_dim).
    """
    weights = "DEFAULT" if pretrained else None

    if name == "resnet18":
        base = models.resnet18(weights=weights)
        feat_dim = 512
    elif name == "resnet34":
        base = models.resnet34(weights=weights)
        feat_dim = 512
    elif name == "resnet50":
        base = models.resnet50(weights=weights)
        feat_dim = 2048
    else:
        raise ValueError(f"Unsupported backbone: {name}. Use resnet18/34/50.")

    # Modify first conv layer if input channels != 3
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
        # Initialize by averaging the pretrained weights across the RGB channels
        if pretrained:
            with torch.no_grad():
                base.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

    # Remove the final FC layer — we want feature vectors, not class logits
    backbone = nn.Sequential(
        base.conv1, base.bn1, base.relu, base.maxpool,
        base.layer1, base.layer2, base.layer3, base.layer4,
        base.avgpool,
        nn.Flatten(),
    )

    return backbone, feat_dim


# ── Projection Head ─────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """2-layer MLP projection head (following SimCLR/CLIP design).

    Maps backbone features to a lower-dimensional, L2-normalized embedding.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return F.normalize(x, dim=-1)  # L2 normalize for cosine similarity


# ── Dual Encoder ─────────────────────────────────────────────────────────

class DualEncoder(nn.Module):
    """CLIP-style dual encoder for SAR-optical cross-modal alignment.

    Encodes SAR images (1-channel) and optical images (3-channel) into a
    shared embedding space where matching pairs are close and non-matching
    pairs are far apart.

    Args:
        backbone: ResNet variant name ('resnet18', 'resnet34', 'resnet50').
        pretrained: Use ImageNet pretrained weights for initialization.
        embed_dim: Dimensionality of the shared embedding space.
        projection_hidden: Hidden layer size in the projection MLP.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        embed_dim: int = 256,
        projection_hidden: int = 512,
    ) -> None:
        super().__init__()

        # SAR encoder (1-channel input)
        self.sar_backbone, sar_feat_dim = _make_backbone(backbone, pretrained, in_channels=1)
        self.sar_projector = ProjectionHead(sar_feat_dim, projection_hidden, embed_dim)

        # Optical encoder (3-channel input)
        self.opt_backbone, opt_feat_dim = _make_backbone(backbone, pretrained, in_channels=3)
        self.opt_projector = ProjectionHead(opt_feat_dim, projection_hidden, embed_dim)

    def encode_sar(self, x: torch.Tensor) -> torch.Tensor:
        """Encode SAR images to embeddings. Input: (B, 1, H, W) -> (B, embed_dim)."""
        features = self.sar_backbone(x)
        return self.sar_projector(features)

    def encode_optical(self, x: torch.Tensor) -> torch.Tensor:
        """Encode optical images to embeddings. Input: (B, 3, H, W) -> (B, embed_dim)."""
        features = self.opt_backbone(x)
        return self.opt_projector(features)

    def forward(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode both modalities. Returns (sar_embeds, optical_embeds)."""
        return self.encode_sar(sar), self.encode_optical(optical)

    def get_backbone_features(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get raw backbone features before projection (for downstream tasks)."""
        return self.sar_backbone(sar), self.opt_backbone(optical)


# ── Contrastive Loss ─────────────────────────────────────────────────────

class ContrastiveLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss.

    Also known as InfoNCE loss. Used in SimCLR and CLIP.

    For a batch of N SAR-optical pairs, this loss treats the matching pair
    as the positive and the other 2(N-1) cross-modal pairings as negatives.

    Args:
        temperature: Initial temperature scaling factor.
        learn_temperature: Whether to make temperature a learnable parameter.
    """

    def __init__(self, temperature: float = 0.07, learn_temperature: bool = True) -> None:
        super().__init__()
        if learn_temperature:
            # Log-parameterize for numerical stability (CLIP-style)
            self.log_temperature = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))
        else:
            self.register_buffer("log_temperature", torch.tensor(math.log(1.0 / temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(-self.log_temperature)

    def forward(
        self,
        sar_embeds: torch.Tensor,
        opt_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the symmetric NT-Xent loss.

        Args:
            sar_embeds: (B, D) L2-normalized SAR embeddings.
            opt_embeds: (B, D) L2-normalized optical embeddings.

        Returns:
            loss: Scalar loss value.
            metrics: Dict with 'loss', 'sar2opt_acc', 'opt2sar_acc', 'temperature'.
        """
        # Cosine similarity matrix scaled by temperature
        # logits[i, j] = similarity(sar_i, opt_j) / temperature
        logit_scale = self.log_temperature.exp()
        logits = logit_scale * (sar_embeds @ opt_embeds.T)

        # Labels: the diagonal entries are the matching pairs
        batch_size = sar_embeds.shape[0]
        labels = torch.arange(batch_size, device=sar_embeds.device)

        # Symmetric loss: SAR->Optical and Optical->SAR
        loss_s2o = F.cross_entropy(logits, labels)
        loss_o2s = F.cross_entropy(logits.T, labels)
        loss = (loss_s2o + loss_o2s) / 2.0

        # Matching accuracy (for monitoring)
        with torch.no_grad():
            s2o_acc = (logits.argmax(dim=1) == labels).float().mean().item()
            o2s_acc = (logits.T.argmax(dim=1) == labels).float().mean().item()

        metrics = {
            "loss": loss.item(),
            "sar2opt_acc": s2o_acc,
            "opt2sar_acc": o2s_acc,
            "temperature": self.temperature.item(),
        }

        return loss, metrics


# ── Terrain Classifier ───────────────────────────────────────────────────

class TerrainClassifier(nn.Module):
    """Lightweight terrain classification head.

    Operates on the concatenated SAR + optical backbone features (before
    projection) to classify terrain type. This acts as a multi-task auxiliary
    loss that encourages the encoders to learn semantically meaningful features.

    Args:
        feature_dim: Backbone feature dimensionality (per modality).
        num_classes: Number of terrain classes (default 4: agri/barren/grass/urban).
        hidden_dim: Hidden layer size.
    """

    TERRAIN_CLASSES = ["agri", "barrenland", "grassland", "urban"]

    def __init__(
        self,
        feature_dim: int = 512,
        num_classes: int = 4,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        # Takes concatenated SAR + optical features
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, sar_features: torch.Tensor, opt_features: torch.Tensor) -> torch.Tensor:
        """Classify terrain from concatenated features. Returns logits (B, num_classes)."""
        combined = torch.cat([sar_features, opt_features], dim=-1)
        return self.classifier(combined)


# ── Quick test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing DualEncoder...")

    model = DualEncoder(backbone="resnet18", pretrained=False, embed_dim=256)
    loss_fn = ContrastiveLoss(temperature=0.07)
    terrain_head = TerrainClassifier(feature_dim=512, num_classes=4)

    # Fake batch
    batch_size = 8
    sar = torch.randn(batch_size, 1, 224, 224)
    optical = torch.randn(batch_size, 3, 224, 224)

    # Forward pass
    sar_emb, opt_emb = model(sar, optical)
    print(f"SAR embeddings:     {sar_emb.shape}")   # (8, 256)
    print(f"Optical embeddings: {opt_emb.shape}")    # (8, 256)

    # Contrastive loss
    loss, metrics = loss_fn(sar_emb, opt_emb)
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"SAR->Opt acc: {metrics['sar2opt_acc']:.2%}")
    print(f"Opt->SAR acc: {metrics['opt2sar_acc']:.2%}")
    print(f"Temperature:  {metrics['temperature']:.4f}")

    # Terrain classification
    sar_feat, opt_feat = model.get_backbone_features(sar, optical)
    terrain_logits = terrain_head(sar_feat, opt_feat)
    print(f"Terrain logits: {terrain_logits.shape}")  # (8, 4)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nTotal model params: {total_params:.1f}M")
    print("All tests passed.")
