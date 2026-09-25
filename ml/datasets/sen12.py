"""
PyTorch Dataset for the SEN1-2 (Sentinel-1 & Sentinel-2) image pair dataset.

Loads co-registered SAR (Sentinel-1) and optical (Sentinel-2) image patches
for use in optical-SAR fusion contrastive pretraining and SAR analysis tasks.

This loader supports the Kaggle "Sentinel-1&2 Image Pairs (SAR & Optical)"
dataset which is organized by terrain type:

    raw/v_2/
        agri/         s1/ s2/
        barrenland/   s1/ s2/
        grassland/    s1/ s2/
        urban/        s1/ s2/

Usage:
    from ml.datasets.sen12 import SEN12Dataset

    dataset = SEN12Dataset(root="data/sen12/raw")
    sar_img, optical_img = dataset[0]  # Both are torch.Tensor

    # Filter by terrain type
    dataset = SEN12Dataset(root="data/sen12/raw", terrains=["urban", "agri"])
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


ALL_TERRAINS = ["agri", "barrenland", "grassland", "urban"]


class SEN12Dataset(Dataset):
    """PyTorch dataset for SEN1-2 SAR-Optical image pairs.

    Args:
        root: Path to the extracted dataset directory (e.g. ``data/sen12/raw``).
              The loader auto-detects the ``v_2/`` subdirectory if present.
        terrains: List of terrain types to include. Defaults to all four.
        split: Optional train/val/test split. Uses an 80/10/10 deterministic split.
        sar_transform: Optional transform applied to the SAR image tensor.
        optical_transform: Optional transform applied to the optical image tensor.
        pair_transform: Optional transform applied to the (sar, optical) tuple.
        return_metadata: If True, return a dict with file paths alongside tensors.
    """

    def __init__(
        self,
        root: str | Path,
        terrains: list[str] | None = None,
        split: Literal["train", "val", "test"] | None = None,
        sar_transform: Callable | None = None,
        optical_transform: Callable | None = None,
        pair_transform: Callable | None = None,
        return_metadata: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.terrains = terrains or ALL_TERRAINS
        self.split = split
        self.sar_transform = sar_transform
        self.optical_transform = optical_transform
        self.pair_transform = pair_transform
        self.return_metadata = return_metadata

        # Auto-detect v_2 subdirectory
        self._data_root = self.root
        if (self.root / "v_2").exists():
            self._data_root = self.root / "v_2"

        # Discover all SAR-optical pairs
        self.pairs: list[tuple[Path, Path, str]] = []  # (sar_path, optical_path, terrain)
        self._discover_pairs()

        # Apply deterministic split if requested
        if self.split is not None:
            self._apply_split()

    def _discover_pairs(self) -> None:
        """Walk the directory tree to find matching SAR/optical file pairs."""
        for terrain in self.terrains:
            terrain_dir = self._data_root / terrain
            if not terrain_dir.exists():
                print(f"[warn] Terrain directory not found: {terrain_dir}")
                continue

            s1_dir = terrain_dir / "s1"
            s2_dir = terrain_dir / "s2"

            if not s1_dir.exists() or not s2_dir.exists():
                print(f"[warn] Missing s1/ or s2/ in {terrain_dir}")
                continue

            # Build a lookup of optical files by name for matching
            s2_files_map = {
                f.name.replace("_s2_", "_s1_"): f
                for f in s2_dir.iterdir()
                if f.is_file() and f.suffix.lower() in (".png", ".tif", ".tiff", ".jpg", ".jpeg")
            }

            # Also try direct name matching (same filename in both dirs)
            s2_direct_map = {
                f.name: f
                for f in s2_dir.iterdir()
                if f.is_file() and f.suffix.lower() in (".png", ".tif", ".tiff", ".jpg", ".jpeg")
            }

            s1_files = sorted(
                [f for f in s1_dir.iterdir()
                 if f.is_file() and f.suffix.lower() in (".png", ".tif", ".tiff", ".jpg", ".jpeg")]
            )

            for s1_file in s1_files:
                # Try matching: s1 filename -> swap s1 to s2 in name
                s2_name_swapped = s1_file.name.replace("_s1_", "_s2_")
                s2_file = s2_dir / s2_name_swapped

                if s2_file.exists():
                    self.pairs.append((s1_file, s2_file, terrain))
                elif s1_file.name in s2_files_map:
                    # Reverse mapping match
                    self.pairs.append((s1_file, s2_files_map[s1_file.name], terrain))
                elif s1_file.name in s2_direct_map:
                    # Direct name match
                    self.pairs.append((s1_file, s2_direct_map[s1_file.name], terrain))

        if not self.pairs:
            print(
                f"[warn] No SAR-optical pairs found in {self._data_root} "
                f"for terrains {self.terrains}. Did you run the download?"
            )

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
        """Load and return a SAR-optical image pair.

        Returns:
            If return_metadata is False:
                (sar_tensor, optical_tensor) -- both float32, values in [0, 1]
            If return_metadata is True:
                (sar_tensor, optical_tensor, metadata_dict)
        """
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
        """Load an image file and return a normalized float32 tensor.

        Args:
            path: Path to the image file.
            grayscale: If True, load as single-channel (1, H, W).
                       If False, load as RGB (3, H, W).

        Returns:
            torch.Tensor with shape (C, H, W) and values in [0.0, 1.0].
        """
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
            f"SEN12Dataset(\n"
            f"  root={self.root},\n"
            f"  terrains={self.terrains},\n"
            f"  split={self.split},\n"
            f"  num_pairs={len(self.pairs):,}\n"
            f")"
        )


def visualize_pair(
    sar: torch.Tensor,
    optical: torch.Tensor,
    terrain: str = "",
    title: str = "SEN1-2 SAR-Optical Pair",
) -> None:
    """Quick matplotlib visualization of a SAR-optical pair.

    Args:
        sar: SAR tensor (1, H, W).
        optical: Optical tensor (3, H, W).
        terrain: Terrain label for the subtitle.
        title: Plot title.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # SAR -- single channel, show as grayscale
    sar_np = sar.squeeze(0).numpy()
    ax1.imshow(sar_np, cmap="gray")
    ax1.set_title("Sentinel-1 (SAR)")
    ax1.axis("off")

    # Optical -- RGB
    opt_np = optical.permute(1, 2, 0).numpy()
    ax2.imshow(opt_np)
    ax2.set_title("Sentinel-2 (Optical)")
    ax2.axis("off")

    suptitle = f"{title} [{terrain}]" if terrain else title
    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    plt.show()


# -- Quick test --
if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "data/sen12/raw"

    print("Loading SEN1-2 dataset...")
    ds = SEN12Dataset(root=root)
    print(ds)

    if len(ds) > 0:
        sar, optical, meta = SEN12Dataset(root=root, return_metadata=True)[0]
        print(f"SAR shape:     {sar.shape}  dtype: {sar.dtype}  range: [{sar.min():.3f}, {sar.max():.3f}]")
        print(f"Optical shape: {optical.shape}  dtype: {optical.dtype}  range: [{optical.min():.3f}, {optical.max():.3f}]")
        print(f"Terrain:       {meta['terrain']}")

        # Per-terrain stats
        for t in ALL_TERRAINS:
            t_ds = SEN12Dataset(root=root, terrains=[t])
            print(f"  {t}: {len(t_ds):,} pairs")
    else:
        print("No data found. Run download.py first.")
