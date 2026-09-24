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
| `HF_TOKEN` | *(required for non-English queries)* | Hugging Face read token for the gated IndicTrans2 checkpoints. |
| `TRANSLATION_ENABLED` | `true` | Turn the Indic ⇄ English translation layer off. |
| `TRANSLATION_DEVICE` | `auto` | `auto` (CUDA if available), `cpu`, `cuda:1`, … |
| `TRANSLATION_PRELOAD` | `false` | Load both translation models at startup instead of on first use. |
| `TRANSLATION_BEAMS` / `TRANSLATION_BATCH_SIZE` | `5` / `16` | Beam search width and sentences per batch. |
| `INDICTRANS_INDIC_EN_MODEL` / `INDICTRANS_EN_INDIC_MODEL` | `ai4bharat/indictrans2-*-dist-200M` | Swap in the 1B checkpoints for higher quality. |
| `CORS_ORIGINS` | `localhost:3000,127.0.0.1:3000,localhost:5173` | Comma-separated allowed origins. |

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness, and whether an API key is configured. |
| `GET` | `/api/registry` | The specialist-model registry the controller selects from. |
| `GET` | `/api/languages` | Supported query languages and the translator's state. |
| `POST` | `/api/analyse` | Run an analysis. |

### `POST /api/analyse`

`multipart/form-data`:

| Field | Required | Notes |
|---|---|---|
| `prompt` | yes | The natural-language query. |
| `images` | yes | One image, or two for a pair. Repeat the field for the second. |
| `modalities` | no | Comma-separated hints, e.g. `optical,sar`. Overrides auto-detection. |
| `task` | no | Force a task instead of routing: `vqa`, `caption`, `grounding`, `change_vqa`, `change_description`, `fusion`. |
| `language` | no | The user's language code (`hi`, `ta`, `ur`, … — see `/api/languages`). Picks between languages that share a script and sets the answer language. |

Accepted format: georeferenced GeoTIFF (`.tif`/`.tiff`) only — a plain TIFF
with no CRS or pixel-to-world transform is rejected. Rasters are normalised to
8-bit RGB — 16-bit and single-band SAR products are contrast-stretched, which a
browser cannot do — and the response carries that normalised render back as
`inputs[].preview_data_uri` so evidence boxes overlay the exact raster the
model saw.

```bash
curl -X POST http://localhost:8000/api/analyse \
  -F "prompt=Has the built-up area increased between these two dates?" \
  -F "images=@t1.tif" -F "images=@t2.tif"
```

The response carries the answer, a confidence score, normalised bounding-box
evidence, and an `trace` object recording the resolved input configuration, the
routed task and why, the tools selected, and per-step timings.

## Multilingual queries

Every model behind the controller works in English, so
[`services/translation.py`](services/translation.py) sits in front of it:

1. The query's script is detected from the text (the `language` hint only
   disambiguates, e.g. Hindi vs Marathi in Devanagari). English queries pass
   straight through.
2. Anything else is translated to English with
   [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) (distilled 200M,
   runs locally, GPU if available), and the controller runs on the English text.
3. The English answer and evidence descriptions are translated back into the
   user's language. The response keeps the English originals in
   `answer`/`evidence` and adds the localised copies under `translation`; both
   translation hops appear in `trace.steps` with stage `translate`.

All 22 scheduled languages are supported. A failed back-translation only adds a
warning; a failed query translation returns 503.

**One-time setup** — the checkpoints are gated on Hugging Face:

1. While signed in, open
   [indictrans2-indic-en-dist-200M](https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M)
   and [indictrans2-en-indic-dist-200M](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M)
   and accept the terms on each.
2. Create a read token at <https://huggingface.co/settings/tokens> and set
   `HF_TOKEN` in `backend/.env`.

The first non-English request downloads both models (~1 GB each) into the
Hugging Face cache.

`transformers` is pinned below 5 because IndicTransToolkit and the IndicTrans2
remote code do not support 5.x. IndicTransToolkit is compiled from source on
Python 3.14, so Windows needs the MSVC build tools.

### Frontend locale files

`frontend/src/locales/en.json` is the source of truth for UI strings. To
machine-translate keys that are missing from the other locales (after adding a
string, or for a language without a file yet), run:

```bash
uv run python -m scripts.translate_locales          # all languages, missing keys only
uv run python -m scripts.translate_locales sat mni  # specific languages
```

Existing (hand-reviewed) strings are left alone unless you pass `--force`.

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
  translation.py           IndicTrans2 Indic <-> English layer around the controller
scripts/
  translate_locales.py     Fills frontend/src/locales/*.json with IndicTrans2
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
