"""
Export Sentinel-2 image pairs with Dynamic World land-cover labels from Earth Engine.

This supplies 10 m training pairs with change the aerial datasets never show:
floods, rivers shifting, vegetation loss, cropland and urban growth. For each
region and pair of date windows it exports, to Google Drive:

    <region>__<t1>_<t2>_t1.tif         8-bit RGB Sentinel-2 median composite (T1 window)
    <region>__<t1>_<t2>_t2.tif         same for the T2 window
    <region>__<t1>_<t2>_t1_label.tif   Dynamic World label (mode over the window), 255 = no data
    <region>__<t1>_<t2>_t2_label.tif

Download the Drive folder and point a ``dynamic_world`` source at it (see
prepare.py). The labels are Dynamic World's own predictions, not hand-drawn, so
treat them as noisy; the test split of hand-labelled datasets is the reference.

Requires ``pip install earthengine-api`` and an Earth Engine project
(``earthengine authenticate`` once). This script could not be run while it was
written; ``sources.dynamic_world`` validates every exported file it reads.

Usage:
    python -m ml.C_VQA.gee_dynamic_world --project my-ee-project --regions regions.json

regions.json:
    [{"name": "brahmaputra_majuli", "lon": 94.2, "lat": 26.95, "size_km": 5.12,
      "t1": ["2023-12-01", "2024-02-28"], "t2": ["2024-07-01", "2024-08-31"]}]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

S2 = "COPERNICUS/S2_SR_HARMONIZED"
DYNAMIC_WORLD = "GOOGLE/DYNAMICWORLD/V1"
NODATA = 255


def _utm_epsg(lon: float, lat: float) -> str:
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def export_region(ee, region: dict, folder: str, max_cloud: float) -> list:
    """Queue the four exports for one region; returns the started tasks."""
    lon, lat = float(region["lon"]), float(region["lat"])
    half = float(region.get("size_km", 5.12)) * 500  # metres from centre to edge
    crs = _utm_epsg(lon, lat)
    area = ee.Geometry.Point([lon, lat]).buffer(half).bounds(1, ee.Projection(crs))
    t1, t2 = region["t1"], region["t2"]
    base = f"{region['name']}__{t1[0]}_{t2[0]}"

    def window(start: str, end: str):
        optical = (
            ee.ImageCollection(S2)
            .filterBounds(area)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
            .median()
            .visualize(bands=["B4", "B3", "B2"], min=0, max=3000)
        )
        labels = (
            ee.ImageCollection(DYNAMIC_WORLD)
            .filterBounds(area)
            .filterDate(start, end)
            .select("label")
            .mode()
            .unmask(NODATA)
            .toUint8()
        )
        return optical, labels

    tasks = []
    for tag, (start, end) in (("t1", t1), ("t2", t2)):
        optical, labels = window(start, end)
        for suffix, image in ((tag, optical), (f"{tag}_label", labels)):
            task = ee.batch.Export.image.toDrive(
                image=image.clip(area),
                description=f"{base}_{suffix}"[:100],
                folder=folder,
                fileNamePrefix=f"{base}_{suffix}",
                region=area,
                scale=10,
                crs=crs,
                maxPixels=1e9,
            )
            task.start()
            tasks.append(task)
    return tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export Sentinel-2 + Dynamic World change pairs")
    parser.add_argument("--project", required=True, help="Earth Engine cloud project id")
    parser.add_argument("--regions", required=True, help="JSON list of regions (see module docstring)")
    parser.add_argument("--folder", default="change_vqa_dynamic_world", help="Google Drive folder")
    parser.add_argument("--max-cloud", type=float, default=20.0, help="Max scene cloud percentage")
    args = parser.parse_args(argv)

    try:
        import ee
    except ImportError:
        print("Install the Earth Engine client first: pip install earthengine-api", file=sys.stderr)
        return 1
    ee.Initialize(project=args.project)

    regions = json.loads(Path(args.regions).read_text())
    started = []
    for region in regions:
        started += export_region(ee, region, args.folder, args.max_cloud)
        print(f"[gee] queued {region['name']}")
    print(f"[gee] {len(started)} export tasks started; follow them at https://code.earthengine.google.com/tasks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
