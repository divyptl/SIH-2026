# SatQuery AI

An agentic, query-driven vision-language assistant for analyzing remote-sensing imagery through natural-language questions.

**Smart India Hackathon 2026 — Problem Statement:** SatQuery AI (ISRO/SAC)

---

## What it does

Upload satellite images — a single image, an optical + SAR pair, or two images from different dates — and ask a question in plain English or any of 22 Indian languages (translated with IndicTrans2):

- *"Describe the land-cover and major objects visible in this image."*
- *"Highlight the water body referred to in the query."*
- *"What changed between these two dates, and where did the change occur?"*
- *"Use the optical and SAR images together to identify built-up and water-covered regions."*
- *"Has the built-up area increased, decreased, or remained unchanged?"*

Instead of routing every query through one generic model, an **agentic controller** validates the inputs, picks the task by fixed rules, and dispatches it to the right fine-tuned specialist model — then returns an answer backed by visual evidence (bounding boxes, masks, confidence scores) and a full, auditable trace of what it did.

A recorded, animated walkthrough of all three specialists is at **`/demo`** in the web app — see [Demo page](#demo-page).

### Why this matters

Domain experts who already work with satellite imagery — disaster management teams, agriculture and urban planning departments, forest departments — currently depend on GIS specialists to answer even simple questions. That doesn't scale, especially during time-critical events like floods. SatQuery AI removes that bottleneck: ask a question directly, get an answer with proof, no GIS expertise required.

---

## Architecture

```
Query + Image Input (single / cross-modal / bi-temporal)
              │
              ▼
      Agentic Controller
 (input validation, rule-based task routing,
  resolution check, specialist selection)
              │
   ┌──────────┼──────────┬──────────────┐
   ▼          ▼          ▼              ▼
 VQA /    Grounding   Change VQA   Optical-SAR
Captioning              (bi-temporal)   Fusion
   │          │          │              │
   └──────────┴──────────┴──────────────┘
              ▼
          Aggregator
 (combine outputs, confidence, evidence)
              ▼
   Evidence-grounded response
(answer + bbox/mask + confidence + execution trace)
```

### Task routing

The task is chosen by plain rules in the controller — no model call, so the
same input always gets the same task:

| Input | Rule | Task |
|---|---|---|
| One image | the question asks *where* something is (or to *find, locate, show, highlight, mark, detect* it) | Grounding |
| One image | any other question | VQA |
| Two images of one place, different dates | — | Change-VQA |
| An optical and a SAR image of one place | — | Optical–SAR fusion |

### Specialist models

| Task | Approach | Fine-tuned on | Measured |
|---|---|---|---|
| Visual Question Answering / Captioning | SkyEyeGPT on the MiniGPT-v2 runtime ([ml/vqa](ml/vqa/README.md)) | RSVQA, VRSBench | Not yet wired into the backend |
| Text-guided Region Grounding | GroundingDINO proposes 10 boxes, an ensemble of 5 re-rankers picks the described one ([details](docs/grounding-module-explained.md)) | VRSBench | 67.5% Acc@0.5 on 16,146 VRSBench validation expressions (storage tanks: 85.1%) |
| Change Detection / Change-VQA | Siamese ResNet-34 + question head; tiled change mask, each changed region named by the model ([details](ml/C_VQA/README.md)) | LEVIR-CD (0.5 m aerial) + Sentinel-2 pairs labelled with Dynamic World (10 m) | Answer accuracy 81% (Dynamic World) / 83% (LEVIR); mask F1 0.58 / 0.88 |
| Optical–SAR Fusion | ResNet-50 dual encoder, contrastive pretraining, terrain-classification head; SAR backscatter water mask | SEN1-2 (Sentinel-1 & 2) | 84.7% SAR→optical matching on validation; SAR water share within 5.7 points of the hand labels on 8 Sen1Floods11 chips |

The controller only sends imagery to a fine-tuned model when its resolution
(read from the GeoTIFF) is close to what the model was trained on; outside that
range the result carries a warning saying why. Every result is labelled
"Fine-tuned model" or not.

### Current limitations

- **Single-image VQA / captioning** has no specialist yet (SkyEyeGPT is not
  installed), and the general-VLM fallback is switched off, so a single-image
  question without a location word gets a placeholder answer. The same happens
  when imagery fails the resolution check or a specialist errors.
- **Fusion:** `checkpoints/fusion_best.pt` was trained with 4 land types
  (agricultural, barren, grassland, urban); the "water" class added in code needs
  a retrain. The optical-only water figure in fusion answers is a brightness
  proxy (the images have no infrared band) and over-counts water by 37 points on
  average; the SAR figure is the reliable one.

---

## Mandatory functional scope (per PS)

- [x] Remote-sensing domain adaptation (fine-tuned on BigEarthNet.txt / open remote-sensing data)
- [x] Single-image VQA (mandatory baseline)
- [x] One additional single-image task (captioning or grounding)
- [x] Multi-image change analysis (change description / change-VQA)
- [x] Cross-modal (optical + SAR) pair analysis
- [x] Agentic orchestration across specialist models
- [x] Auditable execution trace (task, models used, parameters)

---

## Tech stack

- **Frontend:** React + Vite + TanStack Router, shadcn/ui, Motion; evidence boxes and change masks drawn over server-rendered previews, a before/after swipe viewer, and the `/demo` walkthrough
- **Backend:** FastAPI, with the fine-tuned models served in-process and rule-based routing; OpenRouter is only used for the optional plain-language narration (`NARRATION_ENABLED`, off by default)
- **Translation:** IndicTrans2 (22 Indian languages ⇄ English), in-process
- **ML:** PyTorch, Hugging Face Transformers, one uv workspace shared with the backend
- **Report export:** PDF (Typst)

---

## Repo structure

```
/frontend              # Web app (upload, viewer, results UI)
  /src/routes          #   index.tsx (the app), demo.tsx (the /demo walkthrough)
  /src/components/demo #   one recorded run per specialist + shared animations
  /public/demo         #   images and masks the demo plays back
/backend               # FastAPI service, controller, validation, report generation
/ml
  /datasets            # PyTorch dataset loaders (SEN1-2, BigEarthNet, etc.)
  /vqa                 # VQA + captioning model, training and inference
  /grounding           # Grounding model, training and inference
  /C_VQA               # Change VQA model: data pipeline, training, evaluation, inference
  /fusion              # Optical-SAR fusion model, contrastive pretraining
  /controller          # Agentic controller — intent parsing, routing, aggregation
/data
  /bigearthnet_mm      # BigEarthNet-MM raw data (gitignored)
  /sen12               # SEN1-2 download script + raw data (gitignored)
/docs                  # Architecture notes, evaluation results
```

---

## Datasets

| Dataset | Used for |
|---|---|
| [BigEarthNet-MM](https://bigearth.net/) | Optical–SAR contrastive pretraining |
| [SEN1-2 (Sentinel-1&2 Image Pairs)](https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain) | SAR-Optical fusion, SAR analysis (16K paired SAR & optical patches, terrain-labeled) |
| RSVQA | Visual Question Answering |
| VRSBench | Captioning, grounding, VQA |
| [LEVIR-CD](https://justchenhao.github.io/LEVIR/) | Change-VQA: building change at 0.5 m |
| Sentinel-2 + [Dynamic World](https://dynamicworld.app/) | Change-VQA: land-cover and water change at 10 m, exported from Earth Engine |

Evaluation also uses an ISRO/SAC dataset of co-registered Cartosat-2S optical and RISAT SAR image pairs (annotations not disclosed to teams).

### SEN1-2 (Sentinel-1 & 2 Image Pairs)

16,000 co-registered image pairs from Sentinel-1 (SAR) and Sentinel-2 (optical), organized by terrain type (agricultural, barren land, grassland, urban). Derived from the full SEN1-2 dataset (Schmitt et al., 2018).

- **SAR (Sentinel-1):** 8-bit single-channel (sigma-nought backscatter, dB), 256x256 px
- **Optical (Sentinel-2):** 8-bit RGB (bands B4/B3/B2), 256x256 px
- **Terrains:** agri (4K pairs), barrenland (4K), grassland (4K), urban (4K)
- **Paper:** Schmitt M, Hughes LH, Zhu XX (2018). *The SEN1-2 dataset for deep learning in SAR-optical data fusion.* ISPRS Annals.

**Download:**

```bash
# 1. Set your Kaggle API token as environment variable
#    Get it from: https://www.kaggle.com/settings
export KAGGLE_API_TOKEN=your_token_here

# 2. Run the download script
pip install kaggle
cd data/sen12
python download.py                     # Full dataset (~2.7 GB)
```

**Data structure after download:**

```
data/sen12/raw/v_2/
  agri/              # Agricultural land (4,000 pairs)
    s1/              # SAR patches
    s2/              # Optical patches
  barrenland/        # Barren land (4,000 pairs)
    s1/  s2/
  grassland/         # Grassland (4,000 pairs)
    s1/  s2/
  urban/             # Urban areas (4,000 pairs)
    s1/  s2/
```

**Usage in code:**

```python
from ml.datasets.sen12 import SEN12Dataset

dataset = SEN12Dataset(root="data/sen12/raw", terrains=["urban", "agri"], split="train")
sar_img, optical_img = dataset[0]  # (1,256,256) and (3,256,256) float32 tensors
```

---

## Demo page

`/demo` (linked as **How it works** in the header) is an animated, presenter-ready
walkthrough of how a question is answered, one per input type. Each plays back a
**real recorded run** of the pipeline, so it needs no backend, GPU or network
during a presentation, and every number on screen came from the models.

| Walkthrough | Specialist | Recorded example | Checked against |
|---|---|---|---|
| Before and after | Change-VQA | Brahmaputra flood in Assam (Jan vs Jul 2024), asked in Hindi | — (change mask, 8 measured regions, Hindi answer) |
| One image | Grounding | "Where is a storage tank on the upper left?" — three tanks in one photo | DIOR-RSVG hand-labelled box: 90% overlap (IoU) |
| Optical + radar | Optical–SAR fusion | Flooded Brahmaputra chip in Assam (Sen1Floods11) | Hand-drawn water map: 47% water vs 48% from the SAR |

Each walkthrough steps through the real pipeline stages — ask (and translate),
check the images, pick the task, choose the model, run it, answer — with the
time each stage took in the recording.

**Presenter controls:** autoplay runs through all three walkthroughs; Space
pauses, ←/→ or a clicker's PageUp/PageDown step, Shift+←/→ switches
walkthrough, 1–9 jump to a step, R replays a step, F enters or leaves full
screen.

**Re-recording a walkthrough** (after retraining a model): POST the same inputs to
`/api/analyse`, save the JSON response next to the walkthrough in
`frontend/src/components/demo/` (adding the question as `query`), and move the
preview/mask data URIs to image files in `frontend/public/demo/`. Each
walkthrough file (`change.tsx`, `grounding.tsx`, `fusion.tsx`) names its inputs
in its header comment.

---

## Getting started

```bash
git clone https://github.com/divyptl/SIH-2026.git
cd SIH-2026

# Python: backend + ml share one uv workspace and the root .venv
uv sync --all-packages --all-extras

# Backend (see backend/README.md for .env settings and checkpoints)
cd backend
cp .env.example .env
uv run uvicorn main:app --reload

# Frontend
cd ../frontend
pnpm install
pnpm dev
```

Open http://localhost:3000 for the app and http://localhost:3000/demo for the
walkthrough.

- Backend setup, configuration and the specialist models: [backend/README.md](backend/README.md)
- Frontend structure and the demo page: [frontend/README.md](frontend/README.md)
- Training and evaluating the Change-VQA model: [ml/C_VQA/README.md](ml/C_VQA/README.md)
- Grounding model: [docs/grounding-module-explained.md](docs/grounding-module-explained.md)

---

## Team

| Name | Role |
|---|---|
| Divy | AI/ML lead — optical-SAR fusion, agentic controller |
| Yashvi | AI/ML — VQA / captioning |
| Jainee | AI/ML — grounding |
| Harivansh | AI/ML — change detection |
| Prayag | Frontend + backend lead |
| Aryan | Frontend + backend — validation, reporting, testing |

---

## Roadmap

Progress is tracked via [GitHub Issues](../../issues) and Milestones:

1. **Setup** — repo structure, datasets, tech stack
2. **Frontend + Backend** — upload, viewer, results UI, API layer
3. **VQA**, **Grounding**, **Change Detection**, **Fusion** — baseline → fine-tuning → controller integration (parallel tracks)
4. **Agentic Controller** — schema, routing, orchestration, execution trace
5. **Evaluation & Documentation** — benchmark evaluation, final docs

---

## License

TBD
