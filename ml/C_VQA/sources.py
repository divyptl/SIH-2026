"""
Readers for the change-detection datasets the Change-VQA model is trained on.

Each reader yields ``RawPair``s: a co-registered T1/T2 image pair at its native
resolution, its change labels, and its ground sample distance. ``prepare.py``
turns them into multi-scale training tiles with generated questions.

Layouts that could not be checked against a local copy are validated as they
are read (unknown label colours or values raise), so a dataset that differs
from what a reader expects fails loudly instead of producing silent label noise.

Supported ``type`` values for a source spec (see ``prepare.py``):

    levir, s2looking, sysu, folder   binary change, one folder per split
    levir_hf                         LEVIR-CD 256 px crops from the Hugging Face hub
    second                           SECOND semantic change (from/to land cover)
    oscd                             Onera Satellite Change Detection (Sentinel-2, 10 m)
    xview2                           xView2 pre/post-disaster building damage
    dynamic_world                    pairs exported by gee_dynamic_world.py (Sentinel-2, 10 m)
"""

from __future__ import annotations

import json
import re
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ml.C_VQA.qa import (
    FOCUS_ANY,
    FOCUS_BUILDING,
    FOCUS_DAMAGE,
    FOCUS_SEMANTIC,
    LAND_COVER,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SPLITS = ("train", "val", "test")

# SECOND label colours (from the dataset's reference code) -> LAND_COVER.
SECOND_PALETTE: dict[tuple[int, int, int], str | None] = {
    (255, 255, 255): None,            # no change
    (0, 0, 255): "water",
    (128, 128, 128): "bare ground",   # non-vegetated ground surface
    (0, 128, 0): "vegetation",        # low vegetation
    (0, 255, 0): "vegetation",        # tree
    (128, 0, 0): "built-up",          # building
    (255, 0, 0): "other",             # playground / sports field
}

# Dynamic World class ids -> LAND_COVER. Flooded vegetation counts as water so
# floods over vegetated land register as new water.
DYNAMIC_WORLD = {
    0: "water", 1: "vegetation", 2: "vegetation", 3: "water", 4: "cropland",
    5: "vegetation", 6: "built-up", 7: "bare ground", 8: "other",
}
DYNAMIC_WORLD_NODATA = 255


@dataclass
class RawPair:
    """One co-registered image pair and its labels, at native resolution."""

    source: str
    pair_id: str
    split: str
    t1: Image.Image
    t2: Image.Image
    gsd_m: float
    focus: str
    change: np.ndarray
    before: np.ndarray | None = None
    after: np.ndarray | None = None
    gained: dict[str, np.ndarray] = field(default_factory=dict)
    lost: dict[str, np.ndarray] = field(default_factory=dict)
    labelled_classes: tuple[str, ...] = ()


def hash_split(key: str, val: float = 0.1, test: float = 0.1) -> str:
    """Deterministic scene-level split for datasets that ship without one."""
    bucket = zlib.crc32(key.encode("utf-8")) % 1000 / 1000
    if bucket < test:
        return "test"
    if bucket < test + val:
        return "val"
    return "train"


def _images(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES}


def _rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        if image.mode in ("I;16", "I", "F"):
            return _stretch(np.array(image, dtype=np.float32))
        return image.convert("RGB")


def _stretch(array: np.ndarray) -> Image.Image:
    """2-98 percentile stretch of a single band or (H, W, 3) array to 8-bit RGB."""
    low, high = np.percentile(array, (2, 98))
    scaled = np.clip((array - low) / max(high - low, 1e-6), 0, 1)
    image = Image.fromarray((scaled * 255).astype(np.uint8))
    return image.convert("RGB")


def _binary(path: Path) -> np.ndarray:
    with Image.open(path) as mask:
        array = np.array(mask.convert("L"))
    return array > (127 if array.max() > 1 else 0)


def _split_dirs(root: Path) -> list[tuple[str, Path]]:
    """(split, folder) pairs; a root with no split folders is one hash-split pool."""
    found = [(s, root / s) for s in SPLITS if (root / s).is_dir()]
    return found or [("hash", root)]


# ── Binary change datasets laid out as <split>/<t1>/, <split>/<t2>/, <split>/<label>/ ──


def folder_pairs(
    spec: dict[str, Any],
    *,
    t1: str,
    t2: str,
    label: str,
    gsd_m: float,
    focus: str,
    gained_dir: str | None = None,
    lost_dir: str | None = None,
) -> Iterator[RawPair]:
    root = Path(spec["root"])
    t1, t2, label = spec.get("t1", t1), spec.get("t2", t2), spec.get("label", label)
    gsd_m = float(spec.get("gsd_m", gsd_m))
    focus = spec.get("focus", focus)
    name = spec["name"]

    for split, folder in _split_dirs(root):
        if not (folder / t1).is_dir():
            raise FileNotFoundError(f"[{name}] expected '{folder / t1}' (set \"t1\" in the spec)")
        firsts, seconds, labels = (_images(folder / d) for d in (t1, t2, label))
        gains = _images(folder / gained_dir) if gained_dir and (folder / gained_dir).is_dir() else {}
        losses = _images(folder / lost_dir) if lost_dir and (folder / lost_dir).is_dir() else {}

        for stem, first in firsts.items():
            if stem not in seconds or stem not in labels:
                continue
            change = _binary(labels[stem])
            gained: dict[str, np.ndarray] = {}
            lost: dict[str, np.ndarray] = {}
            classes: tuple[str, ...] = ()
            if focus == FOCUS_BUILDING:
                classes = ("built-up",)
                if stem in gains and stem in losses:
                    gained["built-up"] = _binary(gains[stem])
                    lost["built-up"] = _binary(losses[stem])
                else:
                    # LEVIR-CD / WHU-CD mark building change without its direction;
                    # it is overwhelmingly new construction, so it counts as a gain.
                    gained["built-up"] = change
            yield RawPair(
                source=name,
                pair_id=f"{name}_{split}_{stem}",
                split=hash_split(f"{name}:{stem}") if split == "hash" else split,
                t1=_rgb(first),
                t2=_rgb(seconds[stem]),
                gsd_m=gsd_m,
                focus=focus,
                change=change,
                gained=gained,
                lost=lost,
                labelled_classes=classes,
            )


def levir(spec: dict[str, Any]) -> Iterator[RawPair]:
    """LEVIR-CD: 1024 px, 0.5 m, building change. <split>/A, <split>/B, <split>/label."""
    return folder_pairs(spec, t1="A", t2="B", label="label", gsd_m=0.5, focus=FOCUS_BUILDING)


def s2looking(spec: dict[str, Any]) -> Iterator[RawPair]:
    """S2Looking: 1024 px, ~0.5-0.8 m, new (label1) and demolished (label2) buildings."""
    return folder_pairs(
        spec, t1="Image1", t2="Image2", label="label", gsd_m=0.65, focus=FOCUS_BUILDING,
        gained_dir="label1", lost_dir="label2",
    )


def sysu(spec: dict[str, Any]) -> Iterator[RawPair]:
    """SYSU-CD: 256 px, 0.5 m, many change types (unlabelled). <split>/time1, time2, label."""
    return folder_pairs(spec, t1="time1", t2="time2", label="label", gsd_m=0.5, focus=FOCUS_ANY)


def folder(spec: dict[str, Any]) -> Iterator[RawPair]:
    """Any binary dataset in this layout; "t1", "t2", "label", "gsd_m" and "focus" are required."""
    for key in ("t1", "t2", "label", "gsd_m", "focus"):
        if key not in spec:
            raise ValueError(f"[{spec['name']}] a 'folder' source needs \"{key}\"")
    return folder_pairs(
        spec, t1=spec["t1"], t2=spec["t2"], label=spec["label"],
        gsd_m=float(spec["gsd_m"]), focus=spec["focus"],
    )


def levir_hf(spec: dict[str, Any]) -> Iterator[RawPair]:
    """LEVIR-CD 256 px crops streamed from the Hugging Face hub (needs ``datasets``)."""
    from datasets import load_dataset

    name = spec["name"]
    limit = int(spec.get("max_pairs", 10**9))
    for split in ("train", "val", "test"):
        try:
            stream = load_dataset(spec.get("dataset_id", "ericyu/LEVIRCD_Cropped_256"), split=split, streaming=True)
        except Exception as exc:  # the hub dataset may not have every split
            print(f"[{name}] split '{split}' unavailable: {exc}")
            continue
        for index, item in enumerate(stream):
            if index >= limit:
                break
            keys = list(item)
            first = item.get("image_A", item[keys[0]])
            second = item.get("image_B", item[keys[1]])
            mask = np.array(item.get("label", item[keys[2]]).convert("L"))
            change = mask > (127 if mask.max() > 1 else 0)
            yield RawPair(
                source=name, pair_id=f"{name}_{split}_{index:05d}", split=split,
                t1=first.convert("RGB"), t2=second.convert("RGB"), gsd_m=0.5,
                focus=FOCUS_BUILDING, change=change, gained={"built-up": change},
                labelled_classes=("built-up",),
            )


# ── Semantic change ──────────────────────────────────────────────────────────


def _second_classes(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(changed, class index) from a SECOND colour label."""
    rgb = np.array(Image.open(path).convert("RGB"))
    classes = np.full(rgb.shape[:2], -1, dtype=np.int16)
    known = np.zeros(rgb.shape[:2], dtype=bool)
    for colour, name in SECOND_PALETTE.items():
        hit = np.all(rgb == colour, axis=-1)
        known |= hit
        if name is not None:
            classes[hit] = LAND_COVER.index(name)
    if not known.all():
        colours = np.unique(rgb[~known].reshape(-1, 3), axis=0)[:8].tolist()
        raise ValueError(f"{path}: colours outside the SECOND palette: {colours}")
    return classes >= 0, classes


def second(spec: dict[str, Any]) -> Iterator[RawPair]:
    """SECOND: 512 px, ~0.5-3 m. <split?>/im1, im2, label1 (T1 classes), label2 (T2 classes)."""
    name = spec["name"]
    gsd_m = float(spec.get("gsd_m", 1.0))
    for split, root in _split_dirs(Path(spec["root"])):
        firsts, seconds = _images(root / "im1"), _images(root / "im2")
        labels1, labels2 = _images(root / "label1"), _images(root / "label2")
        for stem, first in firsts.items():
            if stem not in seconds or stem not in labels1 or stem not in labels2:
                continue
            changed, before = _second_classes(labels1[stem])
            _, after = _second_classes(labels2[stem])
            yield RawPair(
                source=name,
                pair_id=f"{name}_{stem}",
                split=hash_split(f"{name}:{stem}") if split == "hash" else split,
                t1=_rgb(first), t2=_rgb(seconds[stem]), gsd_m=gsd_m, focus=FOCUS_SEMANTIC,
                change=changed,
                before=np.clip(before, 0, None).astype(np.int64),
                after=np.clip(after, 0, None).astype(np.int64),
                labelled_classes=("water", "vegetation", "built-up", "bare ground"),
            )


# ── Onera Satellite Change Detection (Sentinel-2) ────────────────────────────


def _oscd_rgb(folder: Path) -> Image.Image:
    bands = []
    for band in ("B04", "B03", "B02"):
        path = folder / f"{band}.tif"
        if not path.exists():
            raise FileNotFoundError(f"missing Sentinel-2 band {path}")
        with Image.open(path) as image:
            bands.append(np.array(image, dtype=np.float32))
    return _stretch(np.stack(bands, axis=-1))


def _oscd_mask(city_labels: Path) -> np.ndarray:
    for candidate in (city_labels / "cm" / "cm.png", *city_labels.glob("cm/*.tif"), *city_labels.glob("*-cm.tif")):
        if candidate.exists():
            array = np.array(Image.open(candidate).convert("L") if candidate.suffix == ".png" else Image.open(candidate))
            values = set(np.unique(array).tolist())
            if values <= {1, 2}:      # 1 = no change, 2 = change
                return array == 2
            if values <= {0, 1, 255}:
                return array > 0
            raise ValueError(f"{candidate}: unexpected change-map values {sorted(values)[:8]}")
    raise FileNotFoundError(f"no change map under {city_labels}")


def oscd(spec: dict[str, Any]) -> Iterator[RawPair]:
    """OSCD: 10 m Sentinel-2 city pairs.

    Spec keys: "images" (the '... - Images' folder with <city>/imgs_1_rect/B0x.tif
    and train.txt/test.txt) and "labels" (one or more '... Labels' folders with
    <city>/cm/cm.png).
    """
    name = spec["name"]
    images = Path(spec["images"])
    label_roots = [Path(p) for p in (spec["labels"] if isinstance(spec["labels"], list) else [spec["labels"]])]
    listed: dict[str, str] = {}
    for split in ("train", "test"):
        listing = images / f"{split}.txt"
        if listing.exists():
            for city in listing.read_text().replace("\n", ",").split(","):
                if city.strip():
                    listed[city.strip()] = split

    for city_dir in sorted(p for p in images.iterdir() if p.is_dir()):
        city = city_dir.name
        labels = next((root / city for root in label_roots if (root / city).is_dir()), None)
        if labels is None:
            continue
        split = listed.get(city, "train")
        if split == "train":
            split = "val" if hash_split(f"{name}:{city}", val=0.15, test=0.0) == "val" else "train"
        first, second_ = _oscd_rgb(city_dir / "imgs_1_rect"), _oscd_rgb(city_dir / "imgs_2_rect")
        if first.size != second_.size:
            second_ = second_.resize(first.size, Image.BILINEAR)
        yield RawPair(
            source=name, pair_id=f"{name}_{city}", split=split, t1=first, t2=second_,
            gsd_m=10.0, focus=FOCUS_ANY, change=_oscd_mask(labels),
        )


# ── xView2 building damage ───────────────────────────────────────────────────

_POLYGON = re.compile(r"\(\(([^()]+)\)")
DAMAGED = {"minor-damage", "major-damage", "destroyed"}


def _damage_mask(label_json: Path, size: tuple[int, int]) -> np.ndarray:
    data = json.loads(label_json.read_text())
    canvas = Image.new("L", size, 0)
    draw = ImageDraw.Draw(canvas)
    for feature in data.get("features", {}).get("xy", []):
        if feature.get("properties", {}).get("subtype") not in DAMAGED:
            continue
        for ring in _POLYGON.findall(feature.get("wkt", "")):
            points = [tuple(float(v) for v in pair.split()[:2]) for pair in ring.split(",")]
            if len(points) >= 3:
                draw.polygon(points, fill=255)
    return np.array(canvas) > 0


def xview2(spec: dict[str, Any]) -> Iterator[RawPair]:
    """xView2: 1024 px, ~0.8 m. <subset>/images/*_pre|post_disaster.png, <subset>/labels/*.json.

    Subsets map to splits via "subsets" (default train/tier3 -> train, hold -> val, test -> test).
    """
    name = spec["name"]
    subsets = spec.get("subsets", {"train": "train", "tier3": "train", "hold": "val", "test": "test"})
    only = set(spec.get("disaster_types", []))
    for subset, split in subsets.items():
        root = Path(spec["root"]) / subset
        if not (root / "images").is_dir():
            continue
        for post in sorted((root / "images").glob("*_post_disaster.png")):
            base = post.name.replace("_post_disaster.png", "")
            pre = root / "images" / f"{base}_pre_disaster.png"
            label = root / "labels" / f"{base}_post_disaster.json"
            if not pre.exists() or not label.exists():
                continue
            if only:
                disaster = json.loads(label.read_text()).get("metadata", {}).get("disaster_type")
                if disaster not in only:
                    continue
            first, second_ = _rgb(pre), _rgb(post)
            yield RawPair(
                source=name, pair_id=f"{name}_{base}", split=split, t1=first, t2=second_,
                gsd_m=float(spec.get("gsd_m", 0.8)), focus=FOCUS_DAMAGE,
                change=_damage_mask(label, first.size),
            )


# ── Dynamic World pairs exported from Earth Engine ───────────────────────────


def dynamic_world(spec: dict[str, Any]) -> Iterator[RawPair]:
    """Pairs written by gee_dynamic_world.py: <id>_t1.tif, <id>_t2.tif (8-bit RGB) and
    <id>_t1_label.tif, <id>_t2_label.tif (Dynamic World class ids, 255 = no data)."""
    name = spec["name"]
    root = Path(spec["root"])
    min_valid = float(spec.get("min_valid", 0.9))
    for first in sorted(root.glob("*_t1.tif")):
        base = first.name[: -len("_t1.tif")]
        paths = [root / f"{base}_{suffix}.tif" for suffix in ("t2", "t1_label", "t2_label")]
        if not all(p.exists() for p in paths):
            continue
        labels = [np.array(Image.open(p)) for p in paths[1:]]
        valid = (labels[0] != DYNAMIC_WORLD_NODATA) & (labels[1] != DYNAMIC_WORLD_NODATA)
        if valid.mean() < min_valid:
            continue
        unknown = set(np.unique(labels[0][valid])) | set(np.unique(labels[1][valid]))
        if not unknown <= set(DYNAMIC_WORLD):
            raise ValueError(f"{base}: labels outside Dynamic World classes: {sorted(unknown)[:8]}")
        lookup = np.array([LAND_COVER.index(DYNAMIC_WORLD.get(i, "other")) for i in range(256)])
        before, after = lookup[labels[0]], lookup[labels[1]]
        yield RawPair(
            source=name, pair_id=f"{name}_{base}", split=hash_split(f"{name}:{base.split('__')[0]}"),
            t1=_rgb(first), t2=_rgb(paths[0]), gsd_m=10.0, focus=FOCUS_SEMANTIC,
            change=valid & (before != after), before=before, after=after,
            labelled_classes=("water", "vegetation", "cropland", "built-up", "bare ground"),
        )


READERS = {
    "levir": levir,
    "levir_hf": levir_hf,
    "s2looking": s2looking,
    "sysu": sysu,
    "folder": folder,
    "second": second,
    "oscd": oscd,
    "xview2": xview2,
    "dynamic_world": dynamic_world,
}


def read_source(spec: dict[str, Any]) -> Iterator[RawPair]:
    spec = {"name": spec.get("name", spec["type"]), **spec}
    if spec["type"] not in READERS:
        raise ValueError(f"unknown source type '{spec['type']}'; choose from {sorted(READERS)}")
    return READERS[spec["type"]](spec)
