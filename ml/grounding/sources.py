"""
One entry point for the grounding datasets, and the rules for mixing them.

Datasets:
    vrsbench    splits: train, validation (the VRSBench eval file)
    dior_rsvg   splits: train, val, test

Both VRSBench and DIOR-RSVG use DIOR photos: VRSBench names its crop of DIOR
photo 05863 "05863_0000.png", DIOR-RSVG calls it "05863.jpg". Each benchmark's
own split is kept as published, but when one dataset's *training* data is used,
photos from the other benchmark's *evaluation* split are removed from it:

    VRSBench train  - photos in the DIOR-RSVG test split
    DIOR-RSVG train - photos in the VRSBench eval split

Without this, a model trained on the combination would have seen the other
benchmark's test photos (with different captions of the same objects), and its
scores there would be inflated. Evaluation splits are never filtered.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path

from torch.utils.data import ConcatDataset, Dataset

from ml.grounding.config import TrainConfig
from ml.grounding.dataset import VRSBenchGroundingDataset
from ml.grounding.dior_rsvg import DEFAULT_ROOT as DIOR_ROOT
from ml.grounding.dior_rsvg import DIORRSVGDataset

DATASETS = ("vrsbench", "dior_rsvg")
EVAL_SPLITS = {"vrsbench": "validation", "dior_rsvg": "test"}

_VRSBENCH_DIOR = re.compile(r"^(\d{5})_\d+\.png$")
_DIOR_RSVG = re.compile(r"^(\d{5})\.jpg$")


def dior_photo_id(image_name: str) -> str | None:
    """The DIOR photo an image comes from, or None (e.g. VRSBench's DOTA images)."""
    name = Path(image_name).name
    match = _VRSBENCH_DIOR.match(name) or _DIOR_RSVG.match(name)
    return match.group(1) if match else None


@functools.lru_cache(maxsize=None)
def eval_photo_ids(dataset: str) -> frozenset[str]:
    """DIOR photo ids in a dataset's evaluation split (read from annotations only)."""
    if dataset == "vrsbench":
        from huggingface_hub import hf_hub_download

        cfg = TrainConfig()
        path = cfg.val_annotations_file or hf_hub_download(
            cfg.data_name, "VRSBench_EVAL_referring.json", repo_type="dataset",
            cache_dir=cfg.data_cache_dir,
        )
        with open(path, encoding="utf-8") as f:
            names = [e.get("image_id") or e.get("image") or "" for e in json.load(f)]
    elif dataset == "dior_rsvg":
        names = [s["image_name"] for s in DIORRSVGDataset("test", DIOR_ROOT).samples]
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")
    return frozenset(i for i in map(dior_photo_id, names) if i is not None)


def held_out_photo_ids(dataset: str) -> frozenset[str]:
    """Photos that must not appear in `dataset`'s training data: the *other*
    benchmark's evaluation photos."""
    other = {"vrsbench": "dior_rsvg", "dior_rsvg": "vrsbench"}[dataset]
    return eval_photo_ids(other)


def load_split(
    dataset: str,
    split: str,
    cfg: TrainConfig | None = None,
    exclude_other_eval: bool = True,
    max_samples: int | None = None,
) -> Dataset:
    """Build one split of one dataset.

    Training splits drop photos that belong to the other benchmark's evaluation
    split (see module docstring) when `exclude_other_eval` is set; evaluation
    splits are returned untouched.
    """
    cfg = cfg or TrainConfig()
    if dataset == "vrsbench":
        is_train = split == "train"
        ds = VRSBenchGroundingDataset(
            data_name=cfg.data_name,
            split=split,
            cache_dir=cfg.data_cache_dir,
            image_dir=cfg.image_dir,
            download_images=cfg.download_images,
            auto_extract_zip=cfg.auto_extract_zip,
            extracted_image_dir=cfg.extracted_image_dir,
            annotations_file=cfg.annotations_file if is_train else cfg.val_annotations_file,
            image_zip=cfg.image_zip if is_train else cfg.val_image_zip,
        )
    elif dataset == "dior_rsvg":
        ds = DIORRSVGDataset(split, DIOR_ROOT)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of {DATASETS}.")

    if exclude_other_eval and split == "train":
        held_out = held_out_photo_ids(dataset)
        before = len(ds.samples)
        ds.samples = [s for s in ds.samples if dior_photo_id(s["image_name"]) not in held_out]
        dropped = before - len(ds.samples)
        if dropped:
            print(f"  {dataset} [{split}]: dropped {dropped} expressions on photos in the "
                  f"other benchmark's evaluation split ({len(ds.samples)} left)")

    if max_samples is not None and max_samples < len(ds.samples):
        ds.samples = ds.samples[:max_samples]
    return ds


def load_training_set(
    datasets: list[str], cfg: TrainConfig | None = None, max_samples: int | None = None,
) -> Dataset:
    """Training data from one or more datasets, concatenated.

    With a single dataset nothing is filtered, so a VRSBench-only run is
    unchanged from before; with several, the overlap rules apply.
    """
    combined = len(datasets) > 1
    parts = [
        load_split(name, "train", cfg, exclude_other_eval=combined, max_samples=max_samples)
        for name in datasets
    ]
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)
