# SatQuery AI

An agentic, query-driven vision-language assistant for analyzing remote-sensing imagery through natural-language questions.

**Smart India Hackathon 2026 — Problem Statement:** SatQuery AI (ISRO/SAC)

---

## What it does

Upload satellite images — a single image, an optical + SAR pair, or two images from different dates — and ask a question in plain English:

- *"Describe the land-cover and major objects visible in this image."*
- *"Highlight the water body referred to in the query."*
- *"What changed between these two dates, and where did the change occur?"*
- *"Use the optical and SAR images together to identify built-up and water-covered regions."*
- *"Has the built-up area increased, decreased, or remained unchanged?"*

Instead of routing every query through one generic model, an **agentic controller** interprets the question, validates the inputs, and dispatches it to the right specialist model — then returns an answer backed by visual evidence (bounding boxes, masks, confidence scores) and a full, auditable trace of what it did.

### Why this matters

Domain experts who already work with satellite imagery — disaster management teams, agriculture and urban planning departments, forest departments — currently depend on GIS specialists to answer even simple questions. That doesn't scale, especially during time-critical events like floods. SatQuery AI removes that bottleneck: ask a question directly, get an answer with proof, no GIS expertise required.

---

## Architecture

```
Query + Image Input (single / cross-modal / bi-temporal)
              │
              ▼
      Agentic Controller
   (intent parsing, input validation, task routing)
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

### Specialist models

| Task | Approach | Fine-tuned on |
|---|---|---|
| Visual Question Answering / Captioning | Pretrained Vision-Language Model (BLIP-2 or a remote-sensing-adapted VLM like RS-LLaVA / GeoChat) | RSVQA, VRSBench |
| Text-guided Region Grounding | GroundingDINO / lightweight grounding head | VRSBench grounding subset |
| Change Detection / Change-VQA | Siamese vision encoder + VLM head | CDVQA |
| Optical–SAR Fusion | CLIP-style dual-encoder, contrastive pretraining | BigEarthNet-MM |

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

- **Frontend:** Next.js / React, Leaflet or deck.gl for GeoTIFF rendering and map-based evidence overlays
- **Backend:** FastAPI
- **ML:** PyTorch, Hugging Face Transformers
- **Model serving:** TorchServe / REST wrappers
- **Report export:** PDF + GeoJSON

---

## Repo structure

```
/frontend              # Web app (upload, viewer, results UI)
/backend               # FastAPI service, controller, validation, report generation
/ml
  /datasets            # PyTorch dataset loaders (SEN1-2, BigEarthNet, etc.)
  /vqa                 # VQA + captioning model, training and inference
  /grounding           # Grounding model, training and inference
  /change-detection    # Change VQA model, training and inference
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
| CDVQA | Change-based Visual Question Answering |

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

## Getting started

```bash
# Clone the repo
git clone <repo-url>
cd satquery-ai

# Backend setup
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend setup
cd frontend
npm install
npm run dev
```

(Full setup instructions per module to be added as components come online — see `/docs`.)

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