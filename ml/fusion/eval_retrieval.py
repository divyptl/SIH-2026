import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.datasets.sen12 import SEN12Dataset
from ml.datasets.qxs import QXSDataset
from ml.fusion.model import DualEncoder
from ml.fusion.config import TrainConfig
from ml.fusion.transforms import PairedTransform
from ml.fusion.train import collate_with_terrain
from ml.fusion.inference import FusionModel

def evaluate_retrieval(checkpoint_path: str, device: str = "cuda"):
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = FusionModel.from_checkpoint(checkpoint_path, device=device)
    train_cfg = TrainConfig()

    print("\nLoading datasets...")
    pair_transform_val = PairedTransform(size=train_cfg.image_size, augment=False)
    
    val_datasets = []
    if train_cfg.dataset in ("sen12", "both"):
        val_datasets.append(SEN12Dataset(
            root=train_cfg.data_root,
            terrains=train_cfg.terrains,
            split="val",
            pair_transform=pair_transform_val,
            return_metadata=True,
        ))
    if train_cfg.dataset in ("qxs", "both"):
        val_datasets.append(QXSDataset(
            root=train_cfg.qxs_root,
            split="val",
            pair_transform=pair_transform_val,
            return_metadata=True,
        ))
        
    val_loaders = [
        DataLoader(
            ds,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            pin_memory=train_cfg.pin_memory,
            collate_fn=collate_with_terrain,
        ) for ds in val_datasets
    ]

    total_val_pairs = sum(len(ds) for ds in val_datasets)
    print(f"Total Validation Pairs: {total_val_pairs}")
    
    sar_embeddings = []
    opt_embeddings = []
    
    print("\nExtracting embeddings...")
    t0 = time.time()
    
    # We will extract embeddings for each domain and concatenate them
    for domain_idx, loader in enumerate(val_loaders):
        print(f"  Processing Domain {domain_idx + 1}/{len(val_loaders)}...")
        for sar, optical, _ in loader:
            sar_emb = model.embed_sar(sar)
            opt_emb = model.embed_optical(optical)
            
            sar_embeddings.append(sar_emb)
            opt_embeddings.append(opt_emb)

    sar_embeddings = np.concatenate(sar_embeddings, axis=0)
    opt_embeddings = np.concatenate(opt_embeddings, axis=0)
    
    print(f"Extracted {sar_embeddings.shape[0]} embeddings in {time.time() - t0:.1f}s")
    
    # Normalize for cosine similarity
    print("Normalizing embeddings...")
    sar_embeddings = sar_embeddings / (np.linalg.norm(sar_embeddings, axis=1, keepdims=True) + 1e-8)
    opt_embeddings = opt_embeddings / (np.linalg.norm(opt_embeddings, axis=1, keepdims=True) + 1e-8)
    
    print("Computing SAR -> Optical similarity matrix...")
    # S2O similarity matrix
    # sar_embeddings: (N, D), opt_embeddings: (N, D)
    # sim_matrix: (N, N) where sim_matrix[i, j] is sim(sar_i, opt_j)
    
    def compute_recall(sim_matrix):
        N = sim_matrix.shape[0]
        # Sort each row in descending order
        # We want to find the rank of the correct match (the diagonal element)
        ranks = np.zeros(N)
        
        # Process in batches to save memory
        batch_size = 500
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            batch_sim = sim_matrix[i:end]
            # argsort sorts ascending, so we take [:, ::-1] to get descending
            sorted_indices = np.argsort(batch_sim, axis=1)[:, ::-1]
            
            for j in range(end - i):
                correct_idx = i + j
                rank = np.where(sorted_indices[j] == correct_idx)[0][0]
                ranks[i+j] = rank
                
        r1 = 100.0 * np.mean(ranks < 1)
        r5 = 100.0 * np.mean(ranks < 5)
        r10 = 100.0 * np.mean(ranks < 10)
        return r1, r5, r10

    # For 3600 items, doing a full matrix multiplication is fast (3600 x 3600 = ~13M floats)
    sim_matrix = sar_embeddings @ opt_embeddings.T
    
    print("Calculating Recall for SAR -> Optical (Retrieving Optical given SAR query)...")
    s2o_r1, s2o_r5, s2o_r10 = compute_recall(sim_matrix)
    print(f"  R@1:  {s2o_r1:.2f}%")
    print(f"  R@5:  {s2o_r5:.2f}%")
    print(f"  R@10: {s2o_r10:.2f}%")
    
    print("\nCalculating Recall for Optical -> SAR (Retrieving SAR given Optical query)...")
    # For O2S, we just transpose the similarity matrix
    o2s_r1, o2s_r5, o2s_r10 = compute_recall(sim_matrix.T)
    print(f"  R@1:  {o2s_r1:.2f}%")
    print(f"  R@5:  {o2s_r5:.2f}%")
    print(f"  R@10: {o2s_r10:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/fusion/v2/best.pt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    evaluate_retrieval(args.checkpoint, args.device)
