"""
PyTorch Dataset for the QXS-SAROPT dataset.

Loads co-registered SAR (GaoFen-3) and optical (Google Earth) image pairs
for use in optical-SAR fusion contrastive pretraining.

Dataset structure:
    raw/
        opt_256_oc_0.2/
            1.png
            2.png
            ...
        sar_256_oc_0.2/
            1.png
            2.png
            ...

Usage:
    from ml.datasets.qxs import QXSDataset

    dataset = QXSDataset(root="data/QXSLAB_SAROPT/QXSLAB_SAROPT")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[misc, assignment]


class QXSDataset(Dataset):
    """PyTorch dataset for QXS-SAROPT image pairs.

    Args:
        root: Path to the dataset directory containing opt_256_oc_0.2 and sar_256_oc_0.2.
        split: Optional train/val/test split. Uses an 80/10/10 deterministic split.
        sar_transform: Optional transform applied to the SAR image tensor.
        optical_transform: Optional transform applied to the optical image tensor.
        pair_transform: Optional transform applied to the (sar, optical) tuple.
        return_metadata: If True, return a dict with file paths alongside tensors.
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test"] | None = None,
        sar_transform: Callable | None = None,
        optical_transform: Callable | None = None,
        pair_transform: Callable | None = None,
        return_metadata: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.sar_transform = sar_transform
        self.optical_transform = optical_transform
        self.pair_transform = pair_transform
        self.return_metadata = return_metadata

        # QXS-SAROPT directories
        self.opt_dir = self.root / "opt_256_oc_0.2"
        self.sar_dir = self.root / "sar_256_oc_0.2"

        if not self.opt_dir.exists() or not self.sar_dir.exists():
            print(f"[warn] QXS-SAROPT directories missing at {self.root}.")
            self.pairs = []
        else:
            self.pairs = self._discover_pairs()

        # Apply deterministic split if requested
        if self.split is not None:
            self._apply_split()

    def _discover_pairs(self) -> list[tuple[Path, Path, str]]:
        """Find matching SAR/optical file pairs."""
        pairs = []
        # Look at all files in optical dir
        opt_files = sorted([f for f in self.opt_dir.iterdir() if f.is_file() and f.suffix.lower() == ".png"])
        
        for opt_file in opt_files:
            sar_file = self.sar_dir / opt_file.name
            if sar_file.exists():
                # QXS is entirely urban, so we hardcode the terrain label to "urban"
                pairs.append((sar_file, opt_file, "urban"))

        if not pairs:
            print(f"[warn] No SAR-optical pairs found in {self.root}")
            
        return pairs

    def _apply_split(self) -> None:
        """Apply a deterministic 80/10/10 train/val/test split."""
        n = len(self.pairs)
        # Use a fixed seed for reproducibility
        rng = np.random.RandomState(seed=42)
        indices = rng.permutation(n)

        train_end = int(0.8 * n)
        val_end = int(0.9 * n)

        if self.split == "train":
            selected = indices[:train_end]
        elif self.split == "val":
            selected = indices[train_end:val_end]
        elif self.split == "test":
            selected = indices[val_end:]
        else:
            return

        self.pairs = [self.pairs[i] for i in sorted(selected)]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        """Load and return a SAR-optical image pair."""
        sar_path, optical_path, terrain = self.pairs[idx]

        sar_img = self._load_image(sar_path, grayscale=True)
        optical_img = self._load_image(optical_path, grayscale=False)

        # Apply individual transforms
        if self.sar_transform is not None:
            sar_img = self.sar_transform(sar_img)
        if self.optical_transform is not None:
            optical_img = self.optical_transform(optical_img)

        # Apply pair transform
        if self.pair_transform is not None:
            sar_img, optical_img = self.pair_transform(sar_img, optical_img)

        if self.return_metadata:
            metadata = {
                "sar_path": str(sar_path),
                "optical_path": str(optical_path),
                "terrain": terrain,
                "index": idx,
            }
            return sar_img, optical_img, metadata

        return sar_img, optical_img

    @staticmethod
    def _load_image(path: Path, grayscale: bool = False) -> torch.Tensor:
        """Load an image file and return a normalized float32 tensor."""
        if Image is not None:
            mode = "L" if grayscale else "RGB"
            img = Image.open(path).convert(mode)
            arr = np.array(img, dtype=np.float32) / 255.0
        else:
            # Fallback: use OpenCV
            import cv2
            flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
            arr = cv2.imread(str(path), flags)
            if arr is None:
                raise FileNotFoundError(f"Could not load image: {path}")
            if not grayscale:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            arr = arr.astype(np.float32) / 255.0

        # Convert to (C, H, W)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]  # (1, H, W)
        else:
            arr = arr.transpose(2, 0, 1)  # (C, H, W)

        return torch.from_numpy(arr)

    def __repr__(self) -> str:
        return (
            f"QXSDataset(\n"
            f"  root={self.root},\n"
            f"  split={self.split},\n"
            f"  num_pairs={len(self.pairs):,}\n"
            f")"
        )

# -- Quick test --
if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "data/QXSLAB_SAROPT/QXSLAB_SAROPT"
    
    print("Loading QXS-SAROPT dataset...")
    ds = QXSDataset(root=root)
    print(ds)
    
    if len(ds) > 0:
        sar, optical, meta = QXSDataset(root=root, return_metadata=True)[0]
        print(f"SAR shape:     {sar.shape}  dtype: {sar.dtype}  range: [{sar.min():.3f}, {sar.max():.3f}]")
        print(f"Optical shape: {optical.shape}  dtype: {optical.dtype}  range: [{optical.min():.3f}, {optical.max():.3f}]")
        print(f"Terrain:       {meta['terrain']}")
