"""
Second-stage candidate re-ranker for referring-expression grounding.

GroundingDINO is a detector: every one of its 900 queries is scored against
each text token on its own, and the referred object is taken to be the most
confident query. That works when the expression names a unique object, but
cannot resolve expressions that pick one object out of several ("the leftmost
ship", "the larger of the two tanks"), because no query ever sees the others.

Measured on 1,000 VRSBench validation expressions, the fine-tuned model ranks a
correct box (IoU >= 0.5) first for 50.6% of expressions, but has one among its
top 10 distinct boxes for 85.1%. This module closes that gap:

    image + expression ──► GroundingDINO (frozen) ──► top-K candidates
                                                          │
         per candidate: decoder embedding, RoI-pooled     │
         image feature, box geometry, token-match scores  ▼
                                   ┌─────────────────────────────────┐
                                   │ Transformer decoder              │
      fused text-token features ──►│  candidates self-attend (compare │
                                   │  with each other) and cross-     │
                                   │  attend to the expression        │
                                   └───────────────┬─────────────────┘
                                                   ▼
                                  score per candidate (+ box correction)

Because GroundingDINO is frozen, its outputs are computed once per expression
and cached (`build_rerank_cache.py`), so training the re-ranker takes minutes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert, generalized_box_iou, nms, roi_align

from ml.grounding.config import RerankConfig

# Stored in place of -inf for masked token logits; finite so fp16 can hold it.
MASKED_LOGIT = -30.0

# Downsampling factor of each GroundingDINO feature level: three Swin stages
# plus one extra strided conv on top.
LEVEL_STRIDES = (8, 16, 32, 64)


# ── Candidate extraction ─────────────────────────────────────────────────

def _level_shapes(height: int, width: int, total: int) -> list[tuple[int, int]]:
    """Spatial size of each flattened encoder level, checked against its length."""
    shapes = [(math.ceil(height / s), math.ceil(width / s)) for s in LEVEL_STRIDES]
    if sum(h * w for h, w in shapes) != total:
        raise ValueError(
            f"Encoder has {total} vision tokens, which does not match levels "
            f"{shapes} for a {height}x{width} input"
        )
    return shapes


@torch.no_grad()
def extract_candidates(
    outputs,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    k: int,
    nms_iou: float,
    max_text_tokens: int,
) -> dict[str, torch.Tensor]:
    """Turn a GroundingDINO forward pass into fixed-size re-ranker inputs.

    Candidates are the model's queries in order of confidence, with near
    duplicates removed by NMS and the list padded to `k`.

    Returns a dict of tensors (batch first):
        hidden      (B, K, 256)  decoder embedding of each candidate query
        roi         (B, K, 256)  image features pooled inside each box
        boxes       (B, K, 4)    cx, cy, w, h normalized 0-1
        scores      (B, K)       GroundingDINO confidence
        cand_mask   (B, K)       True for real (non-padding) candidates
        tok_logits  (B, K, T)    each candidate's match logit per text token
        text        (B, T, 256)  fused text-token features
        text_mask   (B, T)       True for real text tokens
        input_ids   (B, T)       the expression's BERT token ids, 0-padded
    """
    logits = outputs.logits.float()                               # (B, Q, T0)
    logits = torch.nan_to_num(logits, neginf=MASKED_LOGIT).clamp(min=MASKED_LOGIT)
    scores = logits.sigmoid().max(dim=-1).values                  # (B, Q)
    boxes = outputs.pred_boxes.float()                            # (B, Q, 4)
    hidden = outputs.last_hidden_state.float()                    # (B, Q, 256)
    text = outputs.encoder_last_hidden_state_text.float()         # (B, T0, 256)
    vision = outputs.encoder_last_hidden_state_vision.float()     # (B, L, 256)

    batch, _, height, width = pixel_values.shape
    dim = hidden.shape[-1]
    t0 = min(text.shape[1], max_text_tokens)
    device = hidden.device

    # Finest feature level (stride 8) resolves the small vehicles and ships
    # that make up much of VRSBench.
    (fh, fw), *_ = _level_shapes(height, width, vision.shape[1])
    fine = vision[:, : fh * fw].transpose(1, 2).reshape(batch, dim, fh, fw)

    out = {
        "hidden": torch.zeros(batch, k, dim, device=device),
        "roi": torch.zeros(batch, k, dim, device=device),
        "boxes": torch.zeros(batch, k, 4, device=device),
        "scores": torch.zeros(batch, k, device=device),
        "cand_mask": torch.zeros(batch, k, dtype=torch.bool, device=device),
        "tok_logits": torch.full((batch, k, max_text_tokens), MASKED_LOGIT, device=device),
        "text": torch.zeros(batch, max_text_tokens, dim, device=device),
        "text_mask": torch.zeros(batch, max_text_tokens, dtype=torch.bool, device=device),
        "input_ids": torch.zeros(batch, max_text_tokens, dtype=torch.long, device=device),
    }
    out["text"][:, :t0] = text[:, :t0]
    out["text_mask"][:, :t0] = attention_mask[:, :t0].bool()
    out["input_ids"][:, :t0] = input_ids[:, :t0] * attention_mask[:, :t0]

    scale = torch.tensor([width, height, width, height], device=device, dtype=torch.float32)
    for b in range(batch):
        xyxy = box_convert(boxes[b], "cxcywh", "xyxy")
        keep = nms(xyxy, scores[b], nms_iou)[:k]
        n = len(keep)
        out["hidden"][b, :n] = hidden[b, keep]
        out["boxes"][b, :n] = boxes[b, keep]
        out["scores"][b, :n] = scores[b, keep]
        out["cand_mask"][b, :n] = True
        out["tok_logits"][b, :n, :t0] = logits[b, keep, :t0]

        rois = torch.cat([torch.zeros(n, 1, device=device), xyxy[keep] * scale], dim=1)
        pooled = roi_align(
            fine[b:b + 1], rois, output_size=4, spatial_scale=1 / LEVEL_STRIDES[0],
            sampling_ratio=2, aligned=True,
        )
        out["roi"][b, :n] = pooled.mean(dim=(2, 3))

    return out


# ── Model ────────────────────────────────────────────────────────────────

def _pairwise_rank(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Each candidate's rank (0 = smallest, 1 = largest) among the valid ones.

    Spatial and size superlatives ("leftmost", "largest") are about order
    within the image, which raw coordinates only express implicitly.
    """
    less = (values.unsqueeze(1) < values.unsqueeze(2)) & mask.unsqueeze(1)   # (B, K, K)
    count = mask.sum(dim=1, keepdim=True).clamp(min=2) - 1
    return less.sum(dim=2).float() / count


def geometry_features(boxes: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Hand-built per-candidate geometry: absolute position, size, order, confidence."""
    cx, cy, w, h = boxes.unbind(-1)
    w = w.clamp(min=1e-4)
    h = h.clamp(min=1e-4)
    area = w * h
    k = boxes.shape[1]
    position = torch.arange(k, device=boxes.device).float().div(k).expand_as(cx)
    return torch.stack([
        cx, cy, w, h,
        cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2,
        area.sqrt(), torch.log(w / h),
        ((cx - 0.5) ** 2 + (cy - 0.5) ** 2).sqrt(),
        _pairwise_rank(cx, mask), _pairwise_rank(cy, mask), _pairwise_rank(area, mask),
        scores, torch.logit(scores.clamp(1e-4, 1 - 1e-4)) / 10, position,
    ], dim=-1)


GEOMETRY_DIM = 17


class CandidateReranker(nn.Module):
    """Scores GroundingDINO's candidate boxes against a referring expression."""

    def __init__(self, config: RerankConfig | None = None, feature_dim: int = 256) -> None:
        super().__init__()
        self.config = config or RerankConfig()
        d = self.config.d_model

        self.text_proj = nn.Sequential(nn.Linear(feature_dim, d), nn.LayerNorm(d))
        # GroundingDINO's fused text features come from a frozen BERT tuned to
        # match object nouns to regions; position words ("left", "top-most")
        # barely register in them. Trainable word embeddings give the
        # re-ranker a direct handle on every word.
        self.word_embed = nn.Embedding(self.config.vocab_size, d, padding_idx=0)
        self.pos_embed = nn.Embedding(self.config.max_text_tokens, d)
        self.cand_proj = nn.Sequential(
            nn.Linear(2 * feature_dim + d, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d),
        )
        self.geom_proj = nn.Sequential(nn.Linear(GEOMETRY_DIM, d), nn.GELU(), nn.Linear(d, d))

        layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=self.config.num_heads, dim_feedforward=4 * d,
            dropout=self.config.dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, self.config.num_layers)
        self.norm = nn.LayerNorm(d)
        self.score_head = nn.Linear(d, 1)
        self.box_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4))
        nn.init.zeros_(self.box_head[-1].weight)
        nn.init.zeros_(self.box_head[-1].bias)

    def forward(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits (B, K) with padding at -inf, refined boxes (B, K, 4) cxcywh)."""
        mask = inputs["cand_mask"]
        text_mask = inputs["text_mask"]
        boxes = inputs["boxes"]
        ids = inputs["input_ids"]
        positions = torch.arange(ids.shape[1], device=ids.device)
        text = self.text_proj(inputs["text"]) + self.word_embed(ids) + self.pos_embed(positions)

        # What each candidate matched in the expression, as a text summary.
        weights = inputs["tok_logits"].masked_fill(~text_mask.unsqueeze(1), -1e4).softmax(-1)
        matched_text = weights @ text                                      # (B, K, d)

        cand = self.cand_proj(torch.cat([inputs["hidden"], inputs["roi"], matched_text], dim=-1))
        cand = cand + self.geom_proj(geometry_features(boxes, inputs["scores"], mask))

        x = self.decoder(
            cand, text, tgt_key_padding_mask=~mask, memory_key_padding_mask=~text_mask,
        )
        x = self.norm(x)

        logits = self.score_head(x).squeeze(-1).masked_fill(~mask, float("-inf"))

        delta = self.box_head(x).tanh()
        cx, cy, w, h = boxes.unbind(-1)
        refined = torch.stack([
            cx + 0.25 * delta[..., 0] * w,
            cy + 0.25 * delta[..., 1] * h,
            w * torch.exp(0.5 * delta[..., 2]),
            h * torch.exp(0.5 * delta[..., 3]),
        ], dim=-1).clamp(0, 1)
        return logits, refined

    @torch.no_grad()
    def select(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Candidates in re-ranked order, with probabilities and final boxes."""
        logits, refined = self(inputs)
        final = refined if self.config.refine_boxes else inputs["boxes"]
        probs = logits.softmax(-1)
        order = probs.argsort(dim=-1, descending=True)
        return {
            "order": order,
            "probs": probs.gather(1, order),
            "boxes": final.gather(1, order.unsqueeze(-1).expand(-1, -1, 4)),
        }


# ── Loss ─────────────────────────────────────────────────────────────────

def candidate_ious(boxes: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """IoU of every candidate with its expression's ground-truth box. (B, K)"""
    a = box_convert(boxes, "cxcywh", "xyxy")
    g = box_convert(gt, "cxcywh", "xyxy").unsqueeze(1)
    lt = torch.max(a[..., :2], g[..., :2])
    rb = torch.min(a[..., 2:], g[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = (a[..., 2:] - a[..., :2]).prod(-1) + (g[..., 2:] - g[..., :2]).prod(-1) - inter
    return (inter / union.clamp(min=1e-6)).masked_fill(~mask, 0.0)


def rerank_loss(
    logits: torch.Tensor,
    refined: torch.Tensor,
    inputs: dict[str, torch.Tensor],
    gt: torch.Tensor,
    min_iou: float,
    refine: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Cross-entropy toward the best-overlapping candidate, plus box refinement.

    Expressions with no candidate reaching `min_iou` have nothing correct to
    pick and are left out of the ranking loss.
    """
    ious = candidate_ious(inputs["boxes"], gt, inputs["cand_mask"])
    best_iou, target = ious.max(dim=1)
    has_target = best_iou >= min_iou

    if has_target.any():
        # Label smoothing spread over the real candidates only: padding sits at
        # -inf, and F.cross_entropy's smoothing would put mass there.
        mask = inputs["cand_mask"][has_target]
        logp = logits[has_target].log_softmax(-1).masked_fill(~mask, 0.0)
        nll = -logp.gather(1, target[has_target].unsqueeze(1)).squeeze(1)
        uniform = -logp.sum(-1) / mask.sum(-1)
        loss_rank = (0.95 * nll + 0.05 * uniform).mean()
    else:
        loss_rank = logits.sum() * 0.0

    loss = loss_rank
    stats = {"rank": loss_rank.item()}
    if refine and has_target.any():
        rows = torch.arange(len(target), device=target.device)[has_target]
        pred = refined[rows, target[has_target]]
        true = gt[has_target]
        loss_l1 = F.l1_loss(pred, true)
        loss_giou = (1 - torch.diag(generalized_box_iou(
            box_convert(pred, "cxcywh", "xyxy"), box_convert(true, "cxcywh", "xyxy"),
        ))).mean()
        loss = loss + 2.0 * loss_l1 + loss_giou
        stats.update({"l1": loss_l1.item(), "giou": loss_giou.item()})
    return loss, stats


# ── Loading and inference ────────────────────────────────────────────────

def load_reranker(path: str, device: str) -> CandidateReranker:
    """Load a checkpoint written by train_rerank.py."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    known = {f for f in RerankConfig.__dataclass_fields__}
    config = RerankConfig(**{k: v for k, v in ckpt["config"].items() if k in known})
    model = CandidateReranker(config)
    model.load_state_dict(ckpt["model"])
    model.detector_checkpoint = ckpt.get("detector_checkpoint")
    return model.to(device).eval()


@torch.no_grad()
def rerank_outputs(
    reranker: CandidateReranker,
    outputs,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Re-rank a GroundingDINO forward pass.

    Returns, per image and in re-ranked order: 'boxes' (B, K, 4) cxcywh 0-1,
    'probs' (B, K) re-ranker probabilities, and 'valid' (B, K).
    """
    cfg = reranker.config
    cands = extract_candidates(
        outputs, pixel_values, input_ids, attention_mask,
        k=cfg.num_candidates, nms_iou=cfg.nms_iou, max_text_tokens=cfg.max_text_tokens,
    )
    picked = reranker.select(cands)
    picked["valid"] = cands["cand_mask"].gather(1, picked["order"])
    return picked
