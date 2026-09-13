# GeoTIFF test imagery

Real Copernicus Sentinel data, georeferenced, for exercising the GeoTIFF-only
upload path. Nothing here is synthetic — every pixel is measured radiometry on
the source product's own CRS and transform.

## Where it came from

| Set | Source | Licence |
|---|---|---|
| `flood-chips/` | [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) v1.1 hand-labeled split, via `gs://sen1floods11` | CC BY 4.0 |
| `bitemporal/` | Sentinel-2 L2A COGs via [Earth Search](https://earth-search.aws.element84.com/v1) / AWS Open Data | Copernicus open licence |

Contains modified Copernicus Sentinel data (2018–2024). Sen1Floods11:
Bonafilia, Tellman, Anderson & Issenberg, *Sen1Floods11: a georeferenced
dataset to train and test deep learning flood algorithms for Sentinel-1*,
CVPR Workshops 2020.

## What was done to it

Band selection and a **fixed physical scaling** — no per-image percentile
stretch, so chips stay radiometrically comparable to one another:

- **Sentinel-2**: DN ÷ 10000 → reflectance, mapped 0–0.22 (visible) or 0–0.45
  (NIR/SWIR) to 0–255, display gamma 1/1.4.
- **Sentinel-1**: fixed dB windows — VV `[−22, 0]`, VH `[−28, −6]`,
  VV−VH `[3, 16]`.

Georeferencing is untouched: the CRS and transform are the source product's.

## `flood-chips/` — 8 flood events, 512×512, EPSG:4326

Five products per chip:

| Suffix | What it is | Good for |
|---|---|---|
| `_s2-truecolour` | B4/B3/B2, 8-bit RGB | optical VQA, captioning |
| `_s2-falsecolour-swir` | B11/B8/B4, 8-bit RGB | **clearest water discrimination** |
| `_s1-vv-vh-rgb` | VV/VH/VV−VH, 8-bit RGB | SAR VQA, grounding |
| `_s1-vv-sigma0-db` | VV backscatter, **float32 dB**, nodata −9999 | high-bit-depth path |
| `_label-water` | hand-drawn ground truth: 0 dry, 255 water, 128 nodata | checking answers |

| Chip | Location | Water |
|---|---|---|
| `india-399883` | Brahmaputra, Assam — braided channels, heavy sediment | 52% |
| `india-900498` | Brahmaputra floodplain, Assam | 47% |
| `india-804466` | Assam — river plus settlements, partial inundation | 19% |
| `mekong-922373` | Mekong meander belt, Cambodia | 46% |
| `spain-7387658` | Murcia/Alicante — agriculture and a town | 58% |
| `usa-788696` | Kansas/Missouri — large open water | 43% |
| `sri-lanka-321316` | Eastern Sri Lanka — coastal lagoon | 37% |
| `india-103447` | Assam — **genuine cloud cover**, kept as a hard case | 10% |

Water percentages are from the hand label. Exact bounds are in
`flood-chips/manifest.json`.

The S1 and S2 members of a chip share an identical grid, so **any
`_s2-*` + `_s1-*` of the same chip is a genuinely co-registered cross-modal
pair** — that is the `fusion` case.

## `bitemporal/` — Brahmaputra before/after

Two dates of Sentinel-2 L2A over MGRS tile 46REQ, 1536×1536 at 10 m
(~15 km across), EPSG:32646. Same tile and the identical pixel window, so the
pair co-registers exactly with no resampling.

- `..._2024-01-28_dry-season_...` — low flow, channels threading exposed sand bars (0.6% cloud)
- `..._2024-07-26_monsoon-flood_...` — swollen sediment-laden channel, bars submerged (9.4% cloud)

This is the `change_vqa` / `change_description` case.

## Task coverage

| Task | Use |
|---|---|
| `vqa`, `caption`, `grounding` | any single chip product |
| `change_vqa`, `change_description` | the `bitemporal/` pair |
| `fusion` | `_s2-*` + `_s1-*` of the same chip |

## Known gaps

- **No bi-temporal SAR.** Sentinel-1 GRD on AWS is ground-range, so two dates
  do not co-register without terrain correction. Analysis-ready alternatives:
  [ASF HyP3](https://hyp3-docs.asf.alaska.edu/) RTC products, or
  Planetary Computer's `sentinel-1-rtc` collection.
- **True colour looks hazy.** Sen1Floods11 Sentinel-2 is L1C (top of
  atmosphere), so atmospheric scattering is included. Use the SWIR false colour
  for water; the `bitemporal/` scenes are L2A and look much cleaner.
- **Label files report as SAR.** `_infer_modality` sees no `s1`/`s2` hint in the
  name and falls back to a single-band → SAR heuristic. They are reference
  masks, not model inputs.

## Getting more

- **Sen1Floods11** — 446 hand-labeled chips over 11 events; this set uses 8.
  Browse: `https://storage.googleapis.com/storage/v1/b/sen1floods11/o?prefix=v1.1/data/flood_events/HandLabeled/`
- **Earth Search / AWS** — free, no account, COGs support windowed HTTP reads:
  `https://earth-search.aws.element84.com/v1/search`
- **Copernicus Data Space** — full Sentinel archive, free account:
  https://dataspace.copernicus.eu
- **Copernicus EMS Rapid Mapping** — delineation products for specific
  disasters: https://emergency.copernicus.eu/mapping
- **Microsoft Planetary Computer** — STAC plus analysis-ready
  `sentinel-1-rtc`: https://planetarycomputer.microsoft.com
