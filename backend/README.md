# SatQuery AI — backend

FastAPI service exposing the agentic remote-sensing analysis workflow. Tasks
with a fine-tuned model from `ml/` (grounding, change VQA, optical–SAR fusion)
run it in-process; the rest use a general vision-language model through
[OpenRouter](https://openrouter.ai).

## Setup

```bash
cd backend
uv sync
cp .env.example .env     # then paste your OpenRouter key into .env
```

`backend/` and `ml/` form one uv workspace (see the root `pyproject.toml`), so
`uv sync` installs the fine-tuned specialists as the `ml` package and the
environment lives in the repository-root `.venv`. To install everything —
backend, training dependencies and the Earth Engine client — in one go, run
`uv sync --all-packages --all-extras` from the repository root. `uv sync`
removes whatever the command does not ask for, so a narrower command such as
`uv sync --package ml --extra train` uninstalls the backend's packages.

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
| `SPECIALISTS_ENABLED` | `true` | `false` sends every task to the OpenRouter baseline, for comparison. |
| `GROUNDING_CHECKPOINT` | `checkpoints/grounding/v1/best.pt` | Fine-tuned GroundingDINO. Relative paths resolve against the repository root; empty disables it. |
| `GROUNDING_RERANKERS` | `auto` | `auto` uses every `reranker*.pt` beside the grounding checkpoint; or a comma-separated list; empty for none. |
| `CHANGE_VQA_CHECKPOINT` | `checkpoints/c_vqa_best.pt` | Change-VQA model for `change_vqa` and `change_description`. Point it at your latest training run, e.g. `checkpoints/change_detection_v2/best.pt`. |
| `FUSION_CHECKPOINT` | `checkpoints/fusion_best.pt` | Optical–SAR fusion model. |
| `SPECIALIST_DEVICE` | `auto` | Device for the specialists: `auto`, `cpu`, `cuda:1`, … |
| `SPECIALIST_PRELOAD` | `false` | Load the specialists at startup instead of on their first request. |
| `NARRATION_ENABLED` | `true` | Reword Change-VQA results in plain language with the VLM (see [Plain-language narration](#plain-language-narration)). |
| `OPENROUTER_NARRATION_MODEL` | vision model | Model that writes the plain-language text. Must accept images. |

Settings are read once at startup: restart the server after editing `.env`
(`--reload` only watches code).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness, and whether an API key is configured. |
| `GET` | `/api/registry` | The specialist-model registry the controller selects from. |
| `GET` | `/api/languages` | Supported query languages and the translator's state. |
| `POST` | `/api/analyse` | Run an analysis. |
| `POST` | `/api/report` | Export an analysis result as a PDF report. |

### `POST /api/analyse`

`multipart/form-data`:

| Field | Required | Notes |
|---|---|---|
| `prompt` | yes | The natural-language query. |
| `images` | yes | One image, or two for a pair. Repeat the field for the second. |
| `modalities` | no | Comma-separated hints, e.g. `optical,sar`. Overrides auto-detection. |
| `task` | no | Force a task instead of routing: `vqa`, `caption`, `grounding`, `change_vqa`, `change_description`, `fusion`. |
| `language` | no | The user's language code (`hi`, `ta`, `ur`, … — see `/api/languages`). Picks between languages that share a script and sets the answer language. |

Accepted formats: GeoTIFF, TIFF, PNG and JPEG (`.tif`/`.tiff`/`.png`/`.jpg`/`.jpeg`).
Only a GeoTIFF carries georeferencing; for the other formats evidence is
reported in image pixels and the trace notes it. Rasters are normalised to
8-bit RGB — 16-bit and single-band SAR products are contrast-stretched, which a
browser cannot do — and the response carries that normalised render back as
`inputs[].preview_data_uri` so evidence boxes overlay the exact raster the
model saw.

```bash
curl -X POST http://localhost:8000/api/analyse \
  -F "prompt=Has the built-up area increased between these two dates?" \
  -F "images=@t1.tif" -F "images=@t2.tif"
```

The response carries the answer, a confidence score, evidence, and a `trace`
object recording the resolved input configuration, the routed task and why, the
tools selected, whether a fine-tuned model answered (`domain_adapted`), and
per-step timings. Evidence items are one of:

- `bbox` — a normalised box (`data.x_min` … `data.y_max` in 0–1), with a label
  and, for the change model, the model's name for the change inside it.
- `mask` — a change mask as a PNG data URI in `data.png`, opaque where change is
  predicted; the frontend draws it as a tinted overlay.
- `observation` — a finding with no location.

`inputs[].ground_sample_distance_m` is the upload's metres per pixel, read from
its GeoTIFF tags (null for PNG, JPEG and plain TIFF).

### `POST /api/report`

JSON body: the `AnalysisResponse` the client already holds (`result`), the
question as typed (`query`), the report language (`language`; `en` gives the
English original only) and every printed string already localised (`labels`,
see `ReportLabels` in `schemas.py`). Returns `application/pdf`. The endpoint is
stateless: nothing about a result is stored server-side.

The PDF is typeset with [Typst](https://typst.app) from `assets/report.typ`,
not printed from HTML. It has real pagination, running headers, PDF
bookmarks, evidence boxes drawn over the imagery, and HarfBuzz-grade shaping
for every Indic script, including right-to-left Urdu, Kashmiri and Sindhi.
The fonts it uses ship in `assets/fonts` (Poppins and Noto, SIL Open Font
License) and system fonts are ignored, so a report renders the same on any
machine. The labels come from the frontend's `report`, `result` and `home`
locale strings.

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
  specialists.py           Adapters that run the fine-tuned models from ml/
  translation.py           IndicTrans2 Indic <-> English layer around the controller
  report.py                Typst PDF export of an analysis result
assets/
  report.typ               Report template
  fonts/                   Poppins + Noto fonts for every supported script
scripts/
  translate_locales.py     Fills frontend/src/locales/*.json with IndicTrans2
```

> Do not add a top-level `openrouter.py` here — it shadows the installed SDK
> package and makes `from openrouter import OpenRouter` import itself.

## Fine-tuned specialists

The problem statement is explicit that a generic VLM does not satisfy the
requirements, so every registry entry names the specialist that should own its
task. A task uses its specialist when the checkpoint configured for it exists;
otherwise the OpenRouter baseline answers and the response sets
`trace.domain_adapted: false`. The UI shows which one answered.

| Task | Specialist | Default checkpoint |
|---|---|---|
| `grounding` | Fine-tuned GroundingDINO + re-ranker ensemble (`ml.grounding`) | `checkpoints/grounding/v1/best.pt` |
| `change_vqa`, `change_description` | Siamese Change-VQA (`ml.C_VQA`) | `checkpoints/c_vqa_best.pt` |
| `fusion` | Optical–SAR dual encoder with terrain head (`ml.fusion`) | `checkpoints/fusion_best.pt` |
| `vqa`, `caption` | — (baseline only) | — |

Behaviour worth knowing:

- **Resolution check.** A specialist can declare the metres per pixel it was
  trained on (`gsd_range` on its `ToolEntry`; Change-VQA reads it from the
  checkpoint). Imagery more than 2x outside that range goes to the baseline,
  with a note in `trace.warnings` saying why — a model shown imagery far from
  its training data answers confidently and wrongly. Uploads without
  georeferencing have no known resolution; they still reach the specialist,
  with a caveat.
- **Failures fall back.** If a specialist raises, the baseline answers and the
  trace records the error.
- **Confidence** is the model's own estimate (softmax probability or
  detection score), reported as-is.

### Plain-language narration

The Change-VQA model's own wording ("bare ground to water", "33.5% of the
scene, in the south sector") is exact but hard for non-specialists to read. So
after it runs, [`agent/narration.py`](agent/narration.py) has the VLM reword it:

1. The VLM is shown both images with the model's regions drawn as numbered
   boxes, plus the model's facts: its answer, the changed share and area
   (km², when the upload is georeferenced), and each region's label, size and
   direction.
2. It returns a short summary and, for each region, what the ground looks like
   before and after and one sentence on what changed.
3. The reply is checked before anything is shown. It is **discarded** if it
   describes a region number the model did not report, contains a figure that
   is not one of the model's (within rounding), or places something in a
   direction where the model found no region.

If accepted, it replaces the answer and the region descriptions (the model's
size and location stay appended to each); the response sets `narration` with
the VLM's name and the model's original answer, and the trace gains a
`narrate` step. If discarded, or if OpenRouter fails, the model's own wording
is shown and `trace.warnings` says why. `trace.domain_adapted` is unchanged:
the regions, figures and confidence all come from the fine-tuned model.

The check covers numbers written as digits, region numbers and compass
directions; it cannot verify the VLM's visual words ("sandbanks", "muddy
water"), which are exactly what it is there to add.

To add a specialist: write a runner in
[`services/specialists.py`](services/specialists.py) that takes
`(query, images)` and returns the baseline's payload shape (`answer`,
`confidence`, `evidence` with normalised `box` dicts or a `mask` data URI), add
a checkpoint setting to [`config.py`](config.py), and set `runner`,
`checkpoint` (and optionally `gsd_range`) on the task's entry in
[`agent/registry.py`](agent/registry.py).
