"""
Dual-Encoder model for Optical-SAR contrastive alignment, Dense Spatial Alignment Maps,
2D Fourier Phase-Consistency Loss, Hard-Negative Terrain Mining, and Cross-Gated Classification.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _make_backbone(name: str, pretrained: bool, in_channels: int) -> tuple[nn.Module, int]:
    weights = "DEFAULT" if pretrained else None

    if name == "convnext_tiny":
        base = models.convnext_tiny(weights=weights)
        feat_dim = 768
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
                    new_conv.weight[:] = (
                        old_conv.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1) / in_channels
                    )
                    if old_conv.bias is not None:
                        new_conv.bias[:] = old_conv.bias
            base.features[0][0] = new_conv
        return base.features, feat_dim

    if name == "resnet18":
        base, feat_dim = models.resnet18(weights=weights), 512
    elif name == "resnet34":
        base, feat_dim = models.resnet34(weights=weights), 512
    elif name == "resnet50":
        base, feat_dim = models.resnet50(weights=weights), 2048
    else:
        raise ValueError(f"Unsupported backbone: {name}. Use convnext_tiny/resnet18/34/50.")

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
                base.conv1.weight[:] = (
                    old_conv.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1) / in_channels
                )

    backbone = nn.Sequential(
        base.conv1, base.bn1, base.relu, base.maxpool,
        base.layer1, base.layer2, base.layer3, base.layer4,
    )
    return backbone, feat_dim


class SelfAttentionBlock(nn.Module):
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
        self.proj_in = nn.Linear(feat_dim, attn_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, spatial_size * spatial_size, attn_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=attn_dim,
            nhead=num_heads,
            dim_feedforward=attn_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(attn_dim)
        self.proj_out = nn.Linear(attn_dim, feat_dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens = self.proj_in(x.flatten(2).transpose(1, 2))
        tokens = tokens + self.pos_embed[:, : H * W, :]
        tokens = self.proj_out(self.norm(self.transformer(tokens)))
        return x + tokens.transpose(1, 2).view(B, C, H, W)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class DualEncoder(nn.Module):
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
        sar_in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.sar_in_channels = sar_in_channels

        self.sar_backbone, sar_feat_dim = _make_backbone(backbone, pretrained, in_channels=sar_in_channels)
        self.opt_backbone, opt_feat_dim = _make_backbone(backbone, pretrained, in_channels=3)

        if use_attention:
            self.sar_attention = SelfAttentionBlock(sar_feat_dim, attn_dim, attn_heads, attn_layers)
            self.opt_attention = SelfAttentionBlock(opt_feat_dim, attn_dim, attn_heads, attn_layers)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sar_projector = ProjectionHead(sar_feat_dim, projection_hidden, embed_dim)
        self.opt_projector = ProjectionHead(opt_feat_dim, projection_hidden, embed_dim)

        self._cached_sar_feat: torch.Tensor | None = None
        self._cached_opt_feat: torch.Tensor | None = None
        self._cached_sar_map: torch.Tensor | None = None
        self._cached_opt_map: torch.Tensor | None = None

    def encode_sar(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.sar_backbone(x)
        if self.use_attention:
            feat_map = self.sar_attention(feat_map)
        self._cached_sar_map = feat_map
        features = self.pool(feat_map).flatten(1)
        self._cached_sar_feat = features
        return self.sar_projector(features)

    def encode_optical(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.opt_backbone(x)
        if self.use_attention:
            feat_map = self.opt_attention(feat_map)
        self._cached_opt_map = feat_map
        features = self.pool(feat_map).flatten(1)
        self._cached_opt_feat = features
        return self.opt_projector(features)

    def forward(self, sar: torch.Tensor, optical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_sar(sar), self.encode_optical(optical)

    def get_backbone_features(
        self,
        sar: torch.Tensor | None = None,
        optical: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sar_feat, opt_feat = self._cached_sar_feat, self._cached_opt_feat

        if sar_feat is None and sar is not None:
            feat_map = self.sar_backbone(sar)
            if self.use_attention:
                feat_map = self.sar_attention(feat_map)
            self._cached_sar_map = feat_map
            sar_feat = self.pool(feat_map).flatten(1)
        if opt_feat is None and optical is not None:
            feat_map = self.opt_backbone(optical)
            if self.use_attention:
                feat_map = self.opt_attention(feat_map)
            self._cached_opt_map = feat_map
            opt_feat = self.pool(feat_map).flatten(1)

        if sar_feat is None or opt_feat is None:
            raise RuntimeError("Backbone features unavailable. Call forward() first or pass input tensors.")
        return sar_feat, opt_feat

    def get_spatial_feature_maps(
        self,
        sar: torch.Tensor | None = None,
        optical: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unpooled (B, C, H, W) spatial feature maps for Fourier phase alignment & dense maps."""
        sar_map, opt_map = self._cached_sar_map, self._cached_opt_map
        if sar_map is None and sar is not None:
            sar_map = self.sar_backbone(sar)
            if self.use_attention:
                sar_map = self.sar_attention(sar_map)
        if opt_map is None and optical is not None:
            opt_map = self.opt_backbone(optical)
            if self.use_attention:
                opt_map = self.opt_attention(opt_map)
        if sar_map is None or opt_map is None:
            raise RuntimeError("Spatial feature maps unavailable. Call forward() first or pass input tensors.")
        return sar_map, opt_map

    @torch.no_grad()
    def compute_dense_similarity_map(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor,
        out_size: tuple[int, int] = (224, 224),
    ) -> torch.Tensor:
        """
        Projects each spatial grid cell (e.g., 7x7 tokens) through the modality projectors
        and computes a dense pixel-level cosine similarity map upsampled to `out_size`.
        Low similarity regions localize clouds, terrain shadows, or structural changes.
        """
        sar_map, opt_map = self.get_spatial_feature_maps(sar, optical)
        B, C, H, W = sar_map.shape

        sar_tokens = sar_map.permute(0, 2, 3, 1).reshape(B * H * W, C)
        opt_tokens = opt_map.permute(0, 2, 3, 1).reshape(B * H * W, C)

        sar_proj = self.sar_projector(sar_tokens).view(B, H, W, -1)
        opt_proj = self.opt_projector(opt_tokens).view(B, H, W, -1)

        # Token-wise cosine similarity: (B, 1, H, W)
        sim_grid = (sar_proj * opt_proj).sum(dim=-1, keepdim=True).permute(0, 3, 1, 2)
        return F.interpolate(sim_grid, size=out_size, mode="bilinear", align_corners=False)


class FourierPhaseLoss(nn.Module):
    """
    2D Fast Fourier Transform (FFT) Phase-Consistency Loss.
    Aligns structural edges and geometric boundaries across SAR and Optical feature maps
    in the frequency phase domain while remaining invariant to amplitude/intensity inversions.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, sar_map: torch.Tensor, opt_map: torch.Tensor) -> torch.Tensor:
        # Compute L2 spatial energy map across channels: (B, H, W) in float32 for FFT stability
        sar_energy = torch.linalg.vector_norm(sar_map.float(), ord=2, dim=1)
        opt_energy = torch.linalg.vector_norm(opt_map.float(), ord=2, dim=1)

        # Zero-mean spatial maps before 2D real FFT
        sar_energy = sar_energy - sar_energy.mean(dim=(-2, -1), keepdim=True)
        opt_energy = opt_energy - opt_energy.mean(dim=(-2, -1), keepdim=True)

        fft_sar = torch.fft.rfft2(sar_energy, norm="ortho")
        fft_opt = torch.fft.rfft2(opt_energy, norm="ortho")

        # Isolate phase by normalizing complex coefficients to unit magnitude
        unit_sar = fft_sar / (fft_sar.abs() + self.eps)
        unit_opt = fft_opt / (fft_opt.abs() + self.eps)

        # Cosine of phase difference Re(e^{i(\phi_sar - \phi_opt)})
        phase_cos = (unit_sar * torch.conj(unit_opt)).real
        return (1.0 - phase_cos).mean()


class ContrastiveLoss(nn.Module):
    """
    NT-Xent (InfoNCE) loss with Hard-Negative Terrain Mining.
    Off-diagonal pairs sharing the same terrain class receive a hard-negative margin boost.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learn_temperature: bool = True,
        hard_negative_weight: float = 1.25,
    ) -> None:
        super().__init__()
        self.hard_negative_weight = hard_negative_weight
        init_val = torch.tensor(math.log(1.0 / temperature))
        if learn_temperature:
            self.log_temperature = nn.Parameter(init_val)
        else:
            self.register_buffer("log_temperature", init_val)

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(-self.log_temperature)

    def forward(
        self,
        sar_embeds: torch.Tensor,
        opt_embeds: torch.Tensor,
        terrain_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        batch_size = sar_embeds.shape[0]
        raw_logits = self.log_temperature.exp() * (sar_embeds @ opt_embeds.T)
        labels = torch.arange(batch_size, device=sar_embeds.device)

        if terrain_labels is not None and self.hard_negative_weight > 1.0:
            valid = terrain_labels >= 0
            same_terrain = (
                (terrain_labels.unsqueeze(1) == terrain_labels.unsqueeze(0))
                & valid.unsqueeze(1)
                & valid.unsqueeze(0)
            )
            eye = torch.eye(batch_size, dtype=torch.bool, device=raw_logits.device)
            hard_neg_mask = same_terrain & (~eye)
            margin = math.log(self.hard_negative_weight)
            logits_s2o = raw_logits + hard_neg_mask.to(raw_logits.dtype) * margin
            logits_o2s = raw_logits.T + hard_neg_mask.T.to(raw_logits.dtype) * margin
        else:
            logits_s2o = raw_logits
            logits_o2s = raw_logits.T

        loss = (F.cross_entropy(logits_s2o, labels) + F.cross_entropy(logits_o2s, labels)) / 2.0

        with torch.no_grad():
            s2o_acc = (raw_logits.argmax(dim=1) == labels).float().mean().item()
            o2s_acc = (raw_logits.T.argmax(dim=1) == labels).float().mean().item()

        return loss, {
            "loss": loss.item(),
            "sar2opt_acc": s2o_acc,
            "opt2sar_acc": o2s_acc,
            "temperature": self.temperature.item(),
        }


class TerrainClassifier(nn.Module):
    TERRAIN_CLASSES = ["agri", "barrenland", "grassland", "urban", "water"]

    def __init__(
        self,
        feature_dim: int = 512,
        num_classes: int = 5,
        hidden_dim: int = 256,
        use_cross_gate: bool = False,
    ) -> None:
        super().__init__()
        self.use_cross_gate = use_cross_gate
        if use_cross_gate:
            self.sar_gate = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.Sigmoid())
            self.opt_gate = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.Sigmoid())

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, sar_features: torch.Tensor, opt_features: torch.Tensor) -> torch.Tensor:
        if self.use_cross_gate:
            sar_refined = sar_features * self.opt_gate(opt_features)
            opt_refined = opt_features * self.sar_gate(sar_features)
            combined = torch.cat([sar_refined, opt_refined], dim=-1)
        else:
            combined = torch.cat([sar_features, opt_features], dim=-1)
        return self.classifier(combined)