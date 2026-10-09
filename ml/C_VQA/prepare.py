"""
Build the multi-source Change-VQA training set.

Reads every dataset listed in a JSON config (see ``sources.py`` for the
supported types), cuts each image pair into 256 px tiles at several scales,
generates questions from each tile's labels (``qa.py``), and writes:

    <out>/tiles/<source>/<tile>_t1.png, _t2.png, _mask.png
    <out>/{train,val,test}_annotations.json   (the format CDVQADataset reads)
    <out>/dataset_info.json                   (answer vocabulary, sources, GSD range)

Scales: a tile at scale k is cut from the image downsampled by k, so a 0.5 m
dataset also yields 1 m and 2 m tiles. Together with natively coarse sources
(OSCD, Dynamic World at 10 m) the model learns change across resolutions
instead of at a single one.

Usage:
    python -m ml.C_VQA.prepare --config sources.json --out data/change_vqa

Config example:
    {"sources": [
        {"type": "levir", "root": "/kaggle/input/levir-cd"},
        {"type": "second", "root": "/kaggle/input/second/train", "gsd_m": 1.0},
        {"type": "oscd", "images": ".../Onera Satellite Change Detection dataset - Images",
         "labels": [".../Train Labels", ".../Test Labels"]},
        {"type": "dynamic_world", "root": "/kaggle/input/dw-pairs"}
    ]}
Any source accepts "scales" and "max_tiles" to override the command-line values.
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ml.C_VQA.qa import ANSWERS, ChangeLabels, generate_questions
from ml.C_VQA.sources import RawPair, read_source

TILE = 256


def _downscale_bool(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """A coarse pixel counts as changed when at least half of it changed."""
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.array(image.resize(size, Image.BOX)) >= 128


def _downscale_classes(classes: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.array(Image.fromarray(classes.astype(np.uint8)).resize(size, Image.NEAREST)).astype(np.int64)


def _starts(length: int, tile: int) -> list[int]:
    """Tile offsets covering the full length; the last tile is shifted back to fit."""
    if length < tile:
        return []
    starts = list(range(0, length - tile + 1, tile))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def _keep(key: str, probability: float) -> bool:
    return probability >= 1.0 or zlib.crc32(key.encode()) % 10_000 < probability * 10_000


def tiles_for(pair: RawPair, scales: list[int], tile: int = TILE):
    """Yield (tile_id, scale, gsd_m, t1, t2, ChangeLabels) for every tile of a pair."""
    width, height = pair.t1.size
    if pair.t2.size != (width, height):
        raise ValueError(f"{pair.pair_id}: T1 {pair.t1.size} and T2 {pair.t2.size} differ in size")
    for scale in scales:
        size = (width // scale, height // scale)
        if min(size) < tile:
            continue
        if scale == 1:
            t1, t2, change = pair.t1, pair.t2, pair.change
            before, after = pair.before, pair.after
            gained, lost = pair.gained, pair.lost
        else:
            t1, t2 = pair.t1.resize(size, Image.BOX), pair.t2.resize(size, Image.BOX)
            change = _downscale_bool(pair.change, size)
            before = _downscale_classes(pair.before, size) if pair.before is not None else None
            after = _downscale_classes(pair.after, size) if pair.after is not None else None
            gained = {k: _downscale_bool(v, size) for k, v in pair.gained.items()}
            lost = {k: _downscale_bool(v, size) for k, v in pair.lost.items()}

        for y in _starts(size[1], tile):
            for x in _starts(size[0], tile):
                window = (slice(y, y + tile), slice(x, x + tile))
                box = (x, y, x + tile, y + tile)
                labels = ChangeLabels(
                    change=change[window],
                    focus=pair.focus,
                    before=before[window] if before is not None else None,
                    after=after[window] if after is not None else None,
                    gained={k: v[window] for k, v in gained.items()},
                    lost={k: v[window] for k, v in lost.items()},
                    labelled_classes=pair.labelled_classes,
                )
                yield (
                    f"{pair.pair_id}_s{scale}_{y}_{x}",
                    scale,
                    pair.gsd_m * scale,
                    t1.crop(box),
                    t2.crop(box),
                    labels,
                )


def prepare(config: dict[str, Any], out: Path, scales: list[int], max_tiles: int, keep_unchanged: float) -> dict:
    manifests: dict[str, list[dict]] = defaultdict(list)
    counts: dict[str, Counter] = defaultdict(Counter)
    gsds: dict[str, set[float]] = defaultdict(set)
    answers: Counter = Counter()

    specs = config["sources"] if isinstance(config, dict) else config
    for spec in specs:
        name = spec.get("name", spec["type"])
        source_scales = [int(s) for s in spec.get("scales", scales)]
        limit = int(spec.get("max_tiles", max_tiles))
        # Readers yield one split after another, so a single shared cap would be
        # used up by train and leave nothing to validate or test on.
        limits = {"train": limit, "val": max(1, limit // 5), "test": max(1, limit // 5)}
        tile_dir = out / "tiles" / name
        tile_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        print(f"[prepare] {name}: reading ({spec['type']}), scales {source_scales}")

        for pair in read_source(spec):
            if all(counts[name][split] >= cap for split, cap in limits.items()):
                break
            if counts[name][pair.split] >= limits.get(pair.split, limit):
                continue
            for tile_id, scale, gsd_m, t1, t2, labels in tiles_for(pair, source_scales):
                if not labels.change.any() and not _keep(tile_id, keep_unchanged):
                    continue
                questions = generate_questions(labels, tile_id)
                paths = {kind: f"tiles/{name}/{tile_id}_{kind}.png" for kind in ("t1", "t2", "mask")}
                t1.save(out / paths["t1"])
                t2.save(out / paths["t2"])
                Image.fromarray(labels.change.astype(np.uint8) * 255).save(out / paths["mask"])
                manifests[pair.split].append({
                    "id": tile_id,
                    "image_t1": paths["t1"],
                    "image_t2": paths["t2"],
                    "mask": paths["mask"],
                    "source": name,
                    "gsd_m": round(gsd_m, 3),
                    "scale": scale,
                    "questions": questions,
                })
                counts[name][pair.split] += 1
                gsds[name].add(round(gsd_m, 3))
                answers.update(q["answer"] for q in questions)
                written += 1
                if counts[name][pair.split] >= limits.get(pair.split, limit):
                    break
        print(f"[prepare] {name}: {written} tiles {dict(counts[name])}, GSD {sorted(gsds[name])} m")

    for split, samples in manifests.items():
        with open(out / f"{split}_annotations.json", "w", encoding="utf-8") as f:
            json.dump({"samples": samples}, f)

    all_gsd = [g for values in gsds.values() for g in values]
    info = {
        "answer_vocab": ANSWERS,
        "sources": {name: {"tiles": dict(c), "gsd_m": sorted(gsds[name])} for name, c in counts.items()},
        "gsd_range_m": [min(all_gsd), max(all_gsd)] if all_gsd else None,
        "answer_counts": dict(answers.most_common()),
        "tile_size": TILE,
    }
    with open(out / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the multi-source Change-VQA dataset")
    parser.add_argument("--config", required=True, help="JSON file listing the sources")
    parser.add_argument("--out", default="data/change_vqa", help="Output directory")
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4],
                        help="Downsampling factors to tile at (default: 1 2 4)")
    parser.add_argument("--max-tiles", type=int, default=20_000,
                        help="Training tiles per source, so one large dataset cannot dominate; "
                             "val and test get a fifth of this each")
    parser.add_argument("--keep-unchanged", type=float, default=0.5,
                        help="Share of tiles with no change to keep (default: 0.5)")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: Windows PowerShell 5 writes UTF-8 files with a byte-order mark.
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    info = prepare(config, out, args.scales, args.max_tiles, args.keep_unchanged)
    if not info["sources"]:
        print("[prepare] No tiles were written; check the source paths.", file=sys.stderr)
        return 1
    print(f"[prepare] Done. GSD range {info['gsd_range_m']} m -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
