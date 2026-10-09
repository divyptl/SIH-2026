"""
Download / prepare water-body SAR-optical pairs for the "water" terrain class.

The existing SEN1-2 dataset on Kaggle only ships four terrains
(agri, barrenland, grassland, urban).  This script creates a compatible
``water/`` directory inside ``data/sen12/raw/v_2/`` using one of three sources:

  1. BigEarthNet-MM (already in ``data/bigearthnet_mm/``)
     - Filter patches whose Corine Land Cover label includes water classes
       (e.g. "Water bodies", "Sea and ocean", "Water courses")
     - Extract co-registered S1 + S2 patches

  2. SEN12MS (external download)
     - A large-scale SAR-optical dataset that includes all land-cover types
     - Water patches are filtered by IGBP land cover label

  3. Synthetic from existing SEN1-2 + NDWI
     - Scan all existing terrains for patches where NDWI suggests water
     - Symlink/copy them into the water/ directory (useful for bootstrapping)

Usage:
    python data/sen12/download_water.py --method bigearthnet
    python data/sen12/download_water.py --method synthetic --threshold 0.3
    python data/sen12/download_water.py --method sen12ms --download
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    print("Pillow is required: pip install Pillow")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("PyTorch is required: pip install torch")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WATER_S1_DIR = PROJECT_ROOT / "data" / "sen12" / "raw" / "v_2" / "water" / "s1"
WATER_S2_DIR = PROJECT_ROOT / "data" / "sen12" / "raw" / "v_2" / "water" / "s2"


def ensure_dirs() -> None:
    WATER_S1_DIR.mkdir(parents=True, exist_ok=True)
    WATER_S2_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directories:")
    print(f"  SAR:     {WATER_S1_DIR}")
    print(f"  Optical: {WATER_S2_DIR}")


# -- Method 1: Synthetic from existing SEN1-2 data + NDWI ----------------


def create_synthetic(threshold: float = 0.25, min_water_ratio: float = 0.3) -> int:
    """Scan existing SEN1-2 terrains for patches with significant water content.

    Uses NDWI (computed from RGB) and SAR low-backscatter to identify patches
    that contain water bodies, then copies them to the water/ directory.

    Args:
        threshold: NDWI threshold above which a pixel is considered water.
        min_water_ratio: Minimum fraction of water pixels required.

    Returns:
        Number of pairs copied.
    """
    from ml.fusion.transforms import compute_ndwi, compute_sar_water_mask

    ensure_dirs()
    data_root = PROJECT_ROOT / "data" / "sen12" / "raw" / "v_2"
    existing_terrains = ["agri", "barrenland", "grassland", "urban"]
    copied = 0

    for terrain in existing_terrains:
        s1_dir = data_root / terrain / "s1"
        s2_dir = data_root / terrain / "s2"
        if not s1_dir.exists() or not s2_dir.exists():
            continue

        s1_files = sorted(s1_dir.glob("*.png")) + sorted(s1_dir.glob("*.tif"))
        print(f"  Scanning {terrain}: {len(s1_files)} SAR files...")

        for s1_file in s1_files:
            # Find matching optical file
            s2_name = s1_file.name.replace("_s1_", "_s2_")
            s2_file = s2_dir / s2_name
            if not s2_file.exists():
                s2_file = s2_dir / s1_file.name
                if not s2_file.exists():
                    continue

            # Load and compute NDWI
            try:
                opt_img = np.array(Image.open(s2_file).convert("RGB"), dtype=np.float32) / 255.0
                opt_tensor = torch.from_numpy(opt_img.transpose(2, 0, 1))
                ndwi = compute_ndwi(opt_tensor)
                water_ratio = (ndwi > threshold).float().mean().item()

                # Also check SAR
                sar_img = np.array(Image.open(s1_file).convert("L"), dtype=np.float32) / 255.0
                sar_tensor = torch.from_numpy(sar_img[np.newaxis])
                sar_water = compute_sar_water_mask(sar_tensor)
                sar_water_ratio = sar_water.mean().item()

                # Accept if either modality shows significant water
                combined_score = 0.6 * water_ratio + 0.4 * sar_water_ratio
                if combined_score >= min_water_ratio:
                    dst_name = f"water_{terrain}_{s1_file.stem}"
                    shutil.copy2(s1_file, WATER_S1_DIR / f"{dst_name}.png")
                    shutil.copy2(s2_file, WATER_S2_DIR / f"{dst_name}.png")
                    copied += 1
            except Exception as exc:
                print(f"    [skip] {s1_file.name}: {exc}")
                continue

    print(f"\n  Copied {copied} water pairs from existing data.")
    return copied


# -- Method 2: BigEarthNet-MM water patches ------------------------------


def extract_bigearthnet_water() -> int:
    """Extract water patches from BigEarthNet-MM (if available locally).

    BigEarthNet-MM contains Sentinel-1 and Sentinel-2 patches with
    Corine Land Cover labels.  Water-related CLC classes:
        - 511: Water courses
        - 512: Water bodies
        - 521: Coastal lagoons
        - 522: Estuaries
        - 523: Sea and ocean

    Returns:
        Number of pairs extracted.
    """
    ensure_dirs()
    ben_root = PROJECT_ROOT / "data" / "bigearthnet_mm"

    if not ben_root.exists():
        print(f"BigEarthNet-MM not found at {ben_root}")
        print("Download from: https://bigearth.net/downloads/BigEarthNet-MM.html")
        return 0

    # Water-related CLC labels
    WATER_LABELS = {
        "Water courses", "Water bodies", "Coastal lagoons",
        "Estuaries", "Sea and ocean",
    }

    s1_root = ben_root / "BigEarthNet-S1-v1.0"
    s2_root = ben_root / "BigEarthNet-S2-v1.0"

    if not s1_root.exists() or not s2_root.exists():
        s1_root = ben_root / "s1"
        s2_root = ben_root / "s2"

    if not s1_root.exists():
        print(f"Could not find S1/S2 subdirectories in {ben_root}")
        return 0

    import json
    extracted = 0
    s2_patches = sorted(p for p in s2_root.iterdir() if p.is_dir())

    print(f"  Scanning {len(s2_patches)} BigEarthNet patches for water...")

    for patch_dir in s2_patches:
        label_file = patch_dir / f"{patch_dir.name}_labels_metadata.json"
        if not label_file.exists():
            continue

        try:
            with open(label_file) as f:
                meta = json.load(f)
            labels = set(meta.get("labels", []))
        except Exception:
            continue

        if not labels.intersection(WATER_LABELS):
            continue

        # Find matching S1 patch
        s1_patch = s1_root / patch_dir.name.replace("_S2", "_S1")
        if not s1_patch.exists():
            continue

        tci_files = list(patch_dir.glob("*_TCI*")) or list(patch_dir.glob("*_B04*"))
        s1_files = list(s1_patch.glob("*.tif"))

        if tci_files and s1_files:
            dst = f"ben_water_{extracted:05d}"
            shutil.copy2(s1_files[0], WATER_S1_DIR / f"{dst}.tif")
            shutil.copy2(tci_files[0], WATER_S2_DIR / f"{dst}.tif")
            extracted += 1

    print(f"\n  Extracted {extracted} water pairs from BigEarthNet-MM.")
    return extracted


# -- Method 3: SEN12MS download (for reference) --------------------------


def download_sen12ms_water() -> int:
    """Guide for downloading water patches from SEN12MS."""
    print("=" * 60)
    print("  SEN12MS Water Data Download Guide")
    print("=" * 60)
    print()
    print("SEN12MS is the recommended source for high-quality water")
    print("SAR-optical training pairs. It contains Sentinel-1 & 2 patches")
    print("with IGBP land cover labels including water bodies.")
    print()
    print("Steps:")
    print("  1. Download from: https://mediatum.ub.tum.de/1474000")
    print("     (or use the Python API: pip install sen12ms)")
    print()
    print("  2. Filter for IGBP class 0 (Water Bodies) or 11 (Wetlands)")
    print()
    print("  3. Extract S1/S2 patches and save them as:")
    print(f"     {WATER_S1_DIR}/<name>.png")
    print(f"     {WATER_S2_DIR}/<name>.png")
    print()
    print("Alternative smaller datasets with water labels:")
    print("  - EuroSAT (Sentinel-2, 10 classes including River/Lake)")
    print("    https://github.com/phelber/eurosat")
    print()
    print("  - So2Sat LCZ42 (Sentinel-1 & 2, includes water class)")
    print("    https://mediatum.ub.tum.de/1454690")
    print()
    print("  - SpaceNet 8 (flood detection, SAR + optical)")
    print("    https://spacenet.ai/sn8-challenge/")
    print()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare water terrain data for the fusion model"
    )
    parser.add_argument(
        "--method",
        choices=["synthetic", "bigearthnet", "sen12ms"],
        default="synthetic",
        help="Data source method (default: synthetic)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="NDWI threshold for synthetic method (default: 0.25)",
    )
    parser.add_argument(
        "--min-water-ratio",
        type=float,
        default=0.3,
        help="Min fraction of water pixels for synthetic method (default: 0.3)",
    )
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  Water Terrain Data Preparation")
    print(f"  Method: {args.method}")
    print(f"{'=' * 60}\n")

    if args.method == "synthetic":
        count = create_synthetic(
            threshold=args.threshold,
            min_water_ratio=args.min_water_ratio,
        )
    elif args.method == "bigearthnet":
        count = extract_bigearthnet_water()
    elif args.method == "sen12ms":
        count = download_sen12ms_water()
    else:
        count = 0

    if count > 0:
        print(f"\nCreated {count} water pairs in data/sen12/raw/v_2/water/")
        print(f"  Run training with: python -m ml.fusion.train")
    elif args.method != "sen12ms":
        print(f"\nNo water pairs were created.")
        print(f"  Try adjusting --threshold (lower = more lenient)")
        print(f"  Or download dedicated water datasets (see --method sen12ms)")


if __name__ == "__main__":
    main()
