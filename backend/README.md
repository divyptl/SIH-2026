# SatQuery AI — backend

FastAPI service exposing the agentic remote-sensing analysis workflow. The
vision-language reasoning is served through [OpenRouter](https://openrouter.ai).

## Setup

```bash
cd backend
uv sync
cp .env.example .env     # then paste your OpenRouter key into .env
```

## Run

```bash
uv run uvicorn main:app --reload
```

`uv run fastapi dev main.py` also works, but on Windows its startup banner can
crash the console with a `UnicodeEncodeError`; prefix it with
`PYTHONIOENCODING=utf-8` if you prefer that command.

The API listens on <http://localhost:8000>, with interactive docs at
<http://localhost:8000/docs>.

## Configuration

All settings are environment variables, read from `backend/.env`.

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | *(required)* | Key from openrouter.ai. Without it `/api/analyse` returns 503. |
| `OPENROUTER_VISION_MODEL` | `google/gemma-4-31b-it:free` | Model that reads the imagery. Must accept `image` input. |
| `OPENROUTER_ROUTER_MODEL` | same as vision model | Model that classifies the query into a task. |
| `MAX_IMAGE_BYTES` | `20971520` (20 MB) | Per-upload size cap. |
| `MAX_IMAGE_EDGE_PX` | `1280` | Longest edge before upstream inference. |
| `OPENROUTER_TIMEOUT_S` | `120` | Upstream request timeout. |
| `CORS_ORIGINS` | `localhost:3000,127.0.0.1:3000,localhost:5173` | Comma-separated allowed origins. |

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness, and whether an API key is configured. |
| `GET` | `/api/registry` | The specialist-model registry the controller selects from. |
| `POST` | `/api/analyse` | Run an analysis. |

### `POST /api/analyse`

`multipart/form-data`:

| Field | Required | Notes |
|---|---|---|
| `prompt` | yes | The natural-language query. |
| `images` | yes | One image, or two for a pair. Repeat the field for the second. |
| `modalities` | no | Comma-separated hints, e.g. `optical,sar`. Overrides auto-detection. |
| `task` | no | Force a task instead of routing: `vqa`, `caption`, `grounding`, `change_vqa`, `change_description`, `fusion`. |

Accepted formats: GeoTIFF/TIFF for geospatial imagery, PNG/JPEG/WEBP for
benchmark data. Rasters are normalised to 8-bit RGB — 16-bit and single-band
SAR products are contrast-stretched, which a browser cannot do — and the
response carries that normalised render back as `inputs[].preview_data_uri` so
evidence boxes overlay the exact raster the model saw.

```bash
curl -X POST http://localhost:8000/api/analyse \
  -F "prompt=Has the built-up area increased between these two dates?" \
  -F "images=@t1.tif" -F "images=@t2.tif"
```

The response carries the answer, a confidence score, normalised bounding-box
evidence, and an `trace` object recording the resolved input configuration, the
routed task and why, the tools selected, and per-step timings.

## Architecture

```
main.py                    FastAPI routes, CORS, upload handling
config.py                  Environment-backed settings
schemas.py                 Pydantic API models (mirrors ml/controller/schema.py)
agent/
  controller.py            Orchestration: validate -> classify -> select -> execute -> aggregate
  registry.py              Task -> tool mapping, specialist vs. baseline
  prompts.py               Task system prompts and the response contract
services/
  images.py                Decoding, validation, pair compatibility, normalisation
  openrouter_client.py     Async OpenRouter SDK wrapper
```

> Do not add a top-level `openrouter.py` here — it shadows the installed SDK
> package and makes `from openrouter import OpenRouter` import itself.

## Wiring in a fine-tuned specialist

The problem statement is explicit that a generic VLM does not satisfy the
requirements, so every registry entry names the specialist that should own its
task. Until one is registered, the OpenRouter baseline answers instead, the
response sets `trace.domain_adapted: false`, a warning is attached, and
confidence is capped at 0.75.

To promote a task, implement the module against
`ml.controller.schema.SpecialistModel`, set `loader` on its `ToolEntry` in
[`agent/registry.py`](agent/registry.py), and implement the specialist branch of
`AgenticController._execute`.
