"""
Dual-Encoder model for Optical-SAR contrastive alignment.

Architecture (v3 — Hybrid ResNet + Self-Attention Transformer):

    ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐   ┌──────────────┐   ┌───────────┐
    │ SAR image   │─> │ ResNet-50    │─> │ Self-Attention   │─> │ GAP + Proj   │─> │           │
    │ (1, H, W)   │   │ Backbone     │   │ Transformer      │   │ MLP          │   │ Shared    │
    └─────────────┘   │ (spatial)    │   │ (2 layers)       │   └──────────────┘   │ Embedding │─> NT-Xent
    ┌─────────────┐   ├──────────────┤   ├──────────────────┤   ┌──────────────┐   │ Space     │   Loss
    │ Optical img │─> │ ResNet-50    │─> │ Self-Attention   │─> │ GAP + Proj   │─> │ (256-dim) │
    │ (3, H, W)   │   │ Backbone     │   │ Transformer      │   │ MLP          │   │           │
    └─────────────┘   │ (spatial)    │   │ (2 layers)       │   └──────────────┘   └───────────┘
                      └──────────────┘   └──────────────────┘

The SAR and optical branches are completely independent (NO cross-attention)
to preserve the strict isolation required by contrastive learning.

Each branch: ResNet backbone extracts a 7×7 spatial feature grid, which is
flattened into 49 "tokens". A Self-Attention Transformer allows each token
to attend to all other tokens, building a global understanding of the image's
spatial geometry. The attended tokens are then pooled and projected into the
shared embedding space via a 2-layer MLP.

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
    """Create a backbone that outputs SPATIAL feature maps (B, C, H, W).

    Unlike v2, this does NOT include avgpool or flatten — we need the
    spatial grid for the Self-Attention Transformer.

    Args:
        name: One of 'convnext_tiny', 'resnet18', 'resnet34', 'resnet50'.
        pretrained: Whether to load ImageNet pretrained weights.
        in_channels: Number of input channels (1 for SAR, 3 for optical).

    Returns:
        Tuple of (backbone, output_feature_dim).
        backbone outputs (B, feature_dim, H, W) spatial feature maps.
    """
    weights = "DEFAULT" if pretrained else None

    if name == "convnext_tiny":
        base = models.convnext_tiny(weights=weights)
        feat_dim = 768

        if in_channels != 3:
            old_conv = base.features[0][0]  # Stem conv
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
            base.features[0][0] = new_conv

        # Return feature extractor only (no avgpool/flatten)
        # Output: (B, 768, 7, 7) for 224×224 input
        backbone = base.features
        return backbone, feat_dim

    elif name == "resnet18":
        base = models.resnet18(weights=weights)
        feat_dim = 512
    elif name == "resnet34":
        base = models.resnet34(weights=weights)
        feat_dim = 512
    elif name == "resnet50":
        base = models.resnet50(weights=weights)
        feat_dim = 2048
    else:
        raise ValueError(f"Unsupported backbone: {name}. Use convnext_tiny/resnet18/34/50.")

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

    # Return SPATIAL feature maps (no avgpool, no flatten)
    # Output: (B, feat_dim, 7, 7) for 224×224 input
    backbone = nn.Sequential(
        base.conv1, base.bn1, base.relu, base.maxpool,
        base.layer1, base.layer2, base.layer3, base.layer4,
    )

    return backbone, feat_dim


# ── Self-Attention Transformer Block ────────────────────────────────────

class SelfAttentionBlock(nn.Module):
    """Lightweight Self-Attention Transformer for spatial feature refinement.

    Takes a spatial feature map (B, C, H, W), flattens it into a sequence
    of H*W tokens, runs multi-head self-attention with learned positional
    embeddings, and returns the refined spatial feature map.

    Uses a linear projection to reduce the channel dimension before attention
    for VRAM efficiency (e.g., 2048 → 512), then projects back. A residual
    connection ensures the backbone features are preserved.

    Args:
        feat_dim: Input feature channel dimension from the backbone.
        attn_dim: Internal attention dimension (projected from feat_dim).
        num_heads: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dropout: Dropout rate inside the Transformer.
        spatial_size: Expected spatial size of input (7 for 224×224 input).
    """

    def __init__(
        self,
        feat_dim: int,
        attn_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        spatial_size: int = 7,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.attn_dim = attn_dim

        # Project from backbone channels to attention dimension
        self.proj_in = nn.Linear(feat_dim, attn_dim)

        # Learnable positional embeddings for the spatial grid
        num_tokens = spatial_size * spatial_size  # 49 for 7×7
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_tokens, attn_dim) * 0.02
        )

        # Standard Transformer Encoder with Pre-Norm (more stable training)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=attn_dim,
            nhead=num_heads,
            dim_feedforward=attn_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm architecture (GPT-style, more stable)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.norm = nn.LayerNorm(attn_dim)

        # Project back to original backbone dimension. Zero-init so the block
        # starts as an identity: a randomly initialised projection adds noise on
        # the same scale as the pretrained ResNet features, wiping out the
        # ImageNet initialisation before contrastive training can use it.
        self.proj_out = nn.Linear(attn_dim, feat_dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine spatial features via self-attention with a residual connection.

        Args:
            x: Spatial feature map (B, C, H, W) from the backbone.

        Returns:
            Refined feature map (B, C, H, W) = input + attention output.
        """
        B, C, H, W = x.shape

        # Flatten spatial dims to token sequence: (B, H*W, C)
        tokens = x.flatten(2).transpose(1, 2)

        # Project down for VRAM-efficient attention: (B, H*W, attn_dim)
        tokens_proj = self.proj_in(tokens)

        # Add positional embeddings
        tokens_proj = tokens_proj + self.pos_embed[:, : H * W, :]

        # Self-Attention (each SAR/optical token attends to its own tokens)
        tokens_proj = self.transformer(tokens_proj)
        tokens_proj = self.norm(tokens_proj)

        # Project back to original backbone dimension: (B, H*W, C)
        tokens_out = self.proj_out(tokens_proj)

        # Reshape back to spatial: (B, C, H, W)
        out = tokens_out.transpose(1, 2).view(B, C, H, W)

        # Residual connection preserves backbone features
        return x + out


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
    """Hybrid ResNet + Self-Attention dual encoder for SAR-optical alignment.

    Encodes SAR images (1-channel) and optical images (3-channel) into a
    shared embedding space where matching pairs are close and non-matching
    pairs are far apart.

    The architecture passes spatial feature maps from the ResNet backbone
    through independent Self-Attention Transformer blocks before pooling
    and projecting into the shared embedding space.

    Args:
        backbone: ResNet variant name ('resnet18', 'resnet34', 'resnet50').
        pretrained: Use ImageNet pretrained weights for initialization.
        embed_dim: Dimensionality of the shared embedding space.
        projection_hidden: Hidden layer size in the projection MLP.
        use_attention: Whether to use Self-Attention Transformer blocks.
        attn_dim: Internal attention dimension (projected from backbone channels).
        attn_heads: Number of attention heads.
        attn_layers: Number of Transformer encoder layers.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        embed_dim: int = 256,
        projection_hidden: int = 512,
        use_attention: bool = True,
        attn_dim: int = 512,
        attn_heads: int = 8,
        attn_layers: int = 2,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention

        # SAR encoder (1-channel input)
        self.sar_backbone, sar_feat_dim = _make_backbone(backbone, pretrained, in_channels=1)

        # Optical encoder (3-channel input)
        self.opt_backbone, opt_feat_dim = _make_backbone(backbone, pretrained, in_channels=3)

        # Self-Attention blocks (independent per modality — NO cross-attention)
        if use_attention:
            self.sar_attention = SelfAttentionBlock(
                sar_feat_dim, attn_dim, attn_heads, attn_layers,
            )
            self.opt_attention = SelfAttentionBlock(
                opt_feat_dim, attn_dim, attn_heads, attn_layers,
            )

        # Global Average Pooling (replaces the avgpool removed from backbone)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Projection heads
        self.sar_projector = ProjectionHead(sar_feat_dim, projection_hidden, embed_dim)
        self.opt_projector = ProjectionHead(opt_feat_dim, projection_hidden, embed_dim)

        # Cached backbone features (populated during forward(), used by get_backbone_features())
        self._cached_sar_feat: torch.Tensor | None = None
        self._cached_opt_feat: torch.Tensor | None = None

    def encode_sar(self, x: torch.Tensor) -> torch.Tensor:
        """Encode SAR images to embeddings. Input: (B, 1, H, W) -> (B, embed_dim)."""
        feat_map = self.sar_backbone(x)              # (B, C, 7, 7)
        if self.use_attention:
            feat_map = self.sar_attention(feat_map)   # (B, C, 7, 7) — spatially refined
        features = self.pool(feat_map).flatten(1)     # (B, C)
        self._cached_sar_feat = features
        return self.sar_projector(features)

    def encode_optical(self, x: torch.Tensor) -> torch.Tensor:
        """Encode optical images to embeddings. Input: (B, 3, H, W) -> (B, embed_dim)."""
        feat_map = self.opt_backbone(x)               # (B, C, 7, 7)
        if self.use_attention:
            feat_map = self.opt_attention(feat_map)    # (B, C, 7, 7) — spatially refined
        features = self.pool(feat_map).flatten(1)      # (B, C)
        self._cached_opt_feat = features
        return self.opt_projector(features)

    def forward(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode both modalities. Returns (sar_embeds, optical_embeds).

        Also caches backbone features for use by get_backbone_features().
        """
        return self.encode_sar(sar), self.encode_optical(optical)

    def get_backbone_features(
        self,
        sar: torch.Tensor | None = None,
        optical: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get pooled backbone features after attention (for downstream tasks).

        If forward() was called first, returns the cached features (no extra
        compute). Otherwise falls back to running the full pipeline directly.
        """
        sar_feat = self._cached_sar_feat
        opt_feat = self._cached_opt_feat

        if sar_feat is None and sar is not None:
            feat_map = self.sar_backbone(sar)
            if self.use_attention:
                feat_map = self.sar_attention(feat_map)
            sar_feat = self.pool(feat_map).flatten(1)
        if opt_feat is None and optical is not None:
            feat_map = self.opt_backbone(optical)
            if self.use_attention:
                feat_map = self.opt_attention(feat_map)
            opt_feat = self.pool(feat_map).flatten(1)

        if sar_feat is None or opt_feat is None:
            raise RuntimeError(
                "Backbone features unavailable. Call forward() first or pass input tensors."
            )

        return sar_feat, opt_feat


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
    print("Testing Hybrid DualEncoder (ResNet-50 + Self-Attention)...\n")

    model = DualEncoder(
        backbone="resnet50",
        pretrained=False,
        embed_dim=256,
        use_attention=True,
        attn_dim=512,
        attn_heads=8,
        attn_layers=2,
    )
    loss_fn = ContrastiveLoss(temperature=0.07, learn_temperature=False)
    terrain_head = TerrainClassifier(feature_dim=2048, num_classes=4)

    # Fake batch
    batch_size = 4
    sar = torch.randn(batch_size, 1, 224, 224)
    optical = torch.randn(batch_size, 3, 224, 224)

    # Forward pass
    sar_emb, opt_emb = model(sar, optical)
    print(f"SAR embeddings:     {sar_emb.shape}")   # (4, 256)
    print(f"Optical embeddings: {opt_emb.shape}")    # (4, 256)

    # Contrastive loss
    loss, metrics = loss_fn(sar_emb, opt_emb)
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"SAR->Opt acc: {metrics['sar2opt_acc']:.2%}")
    print(f"Opt->SAR acc: {metrics['opt2sar_acc']:.2%}")
    print(f"Temperature:  {metrics['temperature']:.4f}")

    # Terrain classification
    sar_feat, opt_feat = model.get_backbone_features(sar, optical)
    print(f"Backbone SAR features:     {sar_feat.shape}")   # (4, 2048)
    print(f"Backbone Optical features: {opt_feat.shape}")   # (4, 2048)
    terrain_logits = terrain_head(sar_feat, opt_feat)
    print(f"Terrain logits: {terrain_logits.shape}")  # (4, 4)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    attn_params = 0
    if model.use_attention:
        attn_params += sum(p.numel() for p in model.sar_attention.parameters()) / 1e6
        attn_params += sum(p.numel() for p in model.opt_attention.parameters()) / 1e6
    backbone_params = total_params - attn_params

    print(f"\nBackbone params:   {backbone_params:.1f}M")
    print(f"Attention params:  {attn_params:.1f}M")
    print(f"Total model params: {total_params:.1f}M")
    print("\nAll tests passed!")
