from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure both repo root and ml/ are on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ML_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_REPO_ROOT), str(_ML_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from datasets.sen12 import SEN12Dataset
from datasets.qxs import QXSDataset
from fusion.config import TrainConfig
from fusion.inference import FusionModel
from fusion.train import collate_with_terrain
from fusion.transforms import PairedTransform, normalize_optical, normalize_sar


def compute_retrieval_metrics(sim_matrix: np.ndarray) -> dict[str, float]:
    N = sim_matrix.shape[0]
    ranks = np.zeros(N, dtype=np.int64)
    batch_size = 500

    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        batch_sim = sim_matrix[i:end]
        sorted_indices = np.argsort(batch_sim, axis=1)[:, ::-1]
        for j in range(end - i):
            correct_idx = i + j
            ranks[i + j] = np.where(sorted_indices[j] == correct_idx)[0][0]

    return {
        "R@1": float(100.0 * np.mean(ranks < 1)),
        "R@5": float(100.0 * np.mean(ranks < 5)),
        "R@10": float(100.0 * np.mean(ranks < 10)),
        "MedR": float(np.median(ranks) + 1),
        "MeanR": float(np.mean(ranks) + 1),
    }


@torch.no_grad()
def evaluate_retrieval(
    checkpoint_path: str,
    device: str = "cuda",
    apply_lee: bool = False,
) -> dict[str, float]:
    print(f"\n{'=' * 65}")
    print(f"Evaluating Checkpoint : {checkpoint_path}")
    print(f"SAR Physics Lee Filter: {'ENABLED (Refined Lee 7x7)' if apply_lee else 'DISABLED (Baseline)'}")
    print(f"{'=' * 65}")

    fm = FusionModel.from_checkpoint(checkpoint_path, device=device)
    train_cfg = TrainConfig()

    pair_transform_val = PairedTransform(
        size=train_cfg.image_size,
        augment=False,
        apply_lee=apply_lee,
    )

    val_datasets = []
    if train_cfg.dataset in ("sen12", "both"):
        val_datasets.append(
            SEN12Dataset(
                root=train_cfg.data_root,
                terrains=train_cfg.terrains,
                split="val",
                pair_transform=pair_transform_val,
                return_metadata=True,
            )
        )
    if train_cfg.dataset in ("qxs", "both"):
        val_datasets.append(
            QXSDataset(
                root=train_cfg.qxs_root,
                split="val",
                pair_transform=pair_transform_val,
                return_metadata=True,
            )
        )

    val_loaders = [
        DataLoader(
            ds,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            pin_memory=train_cfg.pin_memory,
            collate_fn=collate_with_terrain,
        )
        for ds in val_datasets
    ]

    total_val_pairs = sum(len(ds) for ds in val_datasets)
    print(f"Total Validation Pairs: {total_val_pairs}")

    sar_embeddings, opt_embeddings = [], []
    terrain_preds, terrain_targets = [], []

    t0 = time.time()
    for domain_idx, loader in enumerate(val_loaders):
        print(f"  Processing Domain {domain_idx + 1}/{len(val_loaders)}...")
        for sar, optical, terrain_idx in loader:
            sar = sar.to(fm.device, non_blocking=True)
            optical = optical.to(fm.device, non_blocking=True)

            # Ensure normalization if dataset returns [0, 1] tensors
            if sar.min() >= 0.0 and sar.max() <= 1.0:
                sar = normalize_sar(sar)
            if optical.min() >= 0.0 and optical.max() <= 1.0:
                optical = normalize_optical(optical)

            sar_emb, opt_emb = fm.model(sar, optical)
            sar_embeddings.append(sar_emb.cpu().numpy())
            opt_embeddings.append(opt_emb.cpu().numpy())

            if fm.terrain_head is not None and terrain_idx is not None:
                sar_feat, opt_feat = fm.model.get_backbone_features()
                logits = fm.terrain_head(sar_feat, opt_feat)
                preds = logits.argmax(dim=-1).cpu().numpy()
                terrain_preds.append(preds)
                terrain_targets.append(terrain_idx.numpy())

    sar_embeddings = np.concatenate(sar_embeddings, axis=0)
    opt_embeddings = np.concatenate(opt_embeddings, axis=0)
    elapsed = time.time() - t0
    print(f"Extracted {sar_embeddings.shape[0]} pairs in {elapsed:.1f}s")

    sar_embeddings /= np.linalg.norm(sar_embeddings, axis=1, keepdims=True) + 1e-8
    opt_embeddings /= np.linalg.norm(opt_embeddings, axis=1, keepdims=True) + 1e-8

    sim_matrix = sar_embeddings @ opt_embeddings.T
    s2o = compute_retrieval_metrics(sim_matrix)
    o2s = compute_retrieval_metrics(sim_matrix.T)

    print("\n[SAR -> Optical Retrieval]")
    print(f"  R@1: {s2o['R@1']:6.2f}% | R@5: {s2o['R@5']:6.2f}% | R@10: {s2o['R@10']:6.2f}% | MedR: {s2o['MedR']:.1f}")

    print("[Optical -> SAR Retrieval]")
    print(f"  R@1: {o2s['R@1']:6.2f}% | R@5: {o2s['R@5']:6.2f}% | R@10: {o2s['R@10']:6.2f}% | MedR: {o2s['MedR']:.1f}")

    terrain_acc = 0.0
    if terrain_preds:
        all_preds = np.concatenate(terrain_preds)
        all_targets = np.concatenate(terrain_targets)
        valid = all_targets >= 0
        if np.any(valid):
            terrain_acc = float(100.0 * np.mean(all_preds[valid] == all_targets[valid]))
            print(f"\n[Cross-Modal Terrain Classification Accuracy]: {terrain_acc:.2f}%")

    return {
        "s2o_R1": s2o["R@1"],
        "s2o_R5": s2o["R@5"],
        "s2o_R10": s2o["R@10"],
        "o2s_R1": o2s["R@1"],
        "o2s_R5": o2s["R@5"],
        "o2s_R10": o2s["R@10"],
        "terrain_acc": terrain_acc,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Optical-SAR Retrieval & Terrain Classification")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/fusion/v3/best.pt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers")
    parser.add_argument("--dataset", type=str, default=None, choices=["sen12", "qxs", "both"])
    parser.add_argument("--data-root", type=str, default=None, help="Override SEN1-2 dataset path")
    parser.add_argument("--qxs-root", type=str, default=None, help="Override QXS dataset path")
    parser.add_argument("--ablate-lee", action="store_true", help="Run both Baseline and Lee-filtered evaluation")
    args = parser.parse_args()

    common_kwargs = dict(
        checkpoint_path=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_root,
        qxs_root=args.qxs_root,
        dataset=args.dataset,
    )

    base_metrics = evaluate_retrieval(**common_kwargs, apply_lee=False)
    if args.ablate_lee:
        lee_metrics = evaluate_retrieval(**common_kwargs, apply_lee=True)
        print(f"\n{'=' * 65}")
        print("PAPER TABLE 1: ZERO-RETRAINING SAR LEE PHYSICS ABLATION")
        print(f"{'=' * 65}")
        print(f"Metric        | Raw SAR (Baseline) | Refined Lee Physics | Delta")
        print(f"--------------+--------------------+---------------------+-------")
        for k in ("s2o_R1", "s2o_R5", "s2o_R10", "o2s_R1", "o2s_R5", "o2s_R10", "terrain_acc"):
            b, l = base_metrics[k], lee_metrics[k]
            print(f"{k:13s} | {b:17.2f}% | {l:18.2f}% | {l - b:+6.2f}%")

@torch.no_grad()
def evaluate_retrieval(
    checkpoint_path: str,
    device: str = "cuda",
    apply_lee: bool = False,
    batch_size: int | None = None,
    num_workers: int | None = None,
    data_root: str | None = None,
    qxs_root: str | None = None,
    dataset: str | None = None,
) -> dict[str, float]:
    print(f"\n{'=' * 65}")
    print(f"Evaluating Checkpoint : {checkpoint_path}")
    print(f"SAR Physics Lee Filter: {'ENABLED (Refined Lee 7x7)' if apply_lee else 'DISABLED (Baseline)'}")
    print(f"{'=' * 65}")

    fm = FusionModel.from_checkpoint(checkpoint_path, device=device)
    train_cfg = TrainConfig()
    train_cfg.auto_scale_hardware()

    # Apply explicit CLI overrides if provided
    if batch_size is not None:
        train_cfg.batch_size = batch_size
    if num_workers is not None:
        train_cfg.num_workers = num_workers
    if data_root is not None:
        train_cfg.data_root = data_root
    if qxs_root is not None:
        train_cfg.qxs_root = qxs_root
    if dataset is not None:
        train_cfg.dataset = dataset

    print(f"Runtime Config: batch_size={train_cfg.batch_size}, num_workers={train_cfg.num_workers}, device={fm.device}")