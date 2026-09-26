# SatQuery AI — Change Detection / Change-VQA Module

**Lead:** Harivansh (AI/ML — Change Detection)
**Task:** Bi-temporal remote-sensing change detection and visual question answering
**Model:** Siamese vision encoder + cross-attention VLM head
**Training data:** LEVIR-CD (0.5 m aerial, building change) + Sentinel-2 pairs labelled with Dynamic World (10 m, land-cover change); readers also exist for SECOND, OSCD, S2Looking, SYSU-CD and xView2

---

## 1. Overview

Given two co-registered optical images of the same place ($T_1$ before, $T_2$ after) and a question, the model:

1. Predicts a pixel-level **change mask**.
2. Answers the question by **classifying** it into a fixed answer vocabulary (60 answers: yes/no, trends, scales, counts, compass sectors, land-cover classes and `<from> to <to>` transitions such as `bare ground to water`).
3. Splits the mask into **regions** and **names the change in each one** by asking the model about a crop around it.

Everything reported comes from the network: its answer with its own softmax confidence, its mask, and its per-region answers. Nothing is filled in by keyword rules or fixed confidence values.

```
Time 1 Image (T1) ──────┐
(3, H, W)               ▼
               ┌─────────────────┐
               │ Siamese Vision  │ (Shared ResNet-18/34/50 backbone)
               │ Encoder         │ Multi-scale feature maps (S1 to S4)
               └─────────────────┘
                        ▲
Time 2 Image (T2) ──────┘
(3, H, W)
         │
         ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Bi-temporal Difference & Interaction Module             │
 │ - Absolute difference: |F1_i - F2_i|                    │
 │ - Channel concatenation: [F1, F2, |F1 - F2|]            │
 │ - Channel attention gating & temporal correlation       │
 └─────────────────────────────────────────────────────────┘
         │
         ├─────────────────────────────────────────┐
         ▼                                         ▼
┌──────────────────────────────────┐     ┌─────────────────────────────────┐
│ Change Grounding / Mask Head     │     │ Vision-Language (VLM) Head      │
│ Multi-scale decoder upsampling   │     │ - Query Text Tokenizer/Encoder  │
│ Output: (1, H, W) change heatmap │     │ - Visual-Text Cross-Attention   │
│                                  │     │ - Answer classification         │
└──────────────────────────────────┘     └─────────────────────────────────┘
         │                                         │
         ▼                                         ▼
  Change mask → regions                    Answer + softmax confidence
  (each region labelled by the model)
         └────────────────────┬────────────────────┘
                              ▼
            analyze_pair() result / ModelResponse
```

---

## 2. Directory structure

```
ml/C_VQA/
├── config.py              ModelConfig / TrainConfig dataclasses
├── model.py               Siamese backbone, difference module, mask head, VLM head, loss
├── transforms.py          Paired flips/rotations, resolution degradation, colour jitter
├── dataset.py             CDVQADataset: reads the manifests prepare.py writes
├── qa.py                  Question generation from change labels; the answer vocabulary
├── sources.py             Readers for each change-detection dataset
├── prepare.py             Builds multi-scale training tiles + manifests from the sources
├── gee_dynamic_world.py   Exports Sentinel-2 + Dynamic World pairs from Earth Engine
├── train.py               Multi-task training with per-source validation
├── evaluate.py            Test report per source, question type and resolution band
├── inference.py           ChangeVQAModel: tiled inference, regions, SpecialistModel protocol
├── test_pipeline.py       Tests for the data pipeline, tiling and regions
└── test_cvqa.py           Older model tests (several need the original CDVQA download)
```

---

## 3. Mathematical formulation

### 3.1 Siamese multi-scale feature extraction
Given $T_1, T_2 \in \mathbb{R}^{3 \times H \times W}$, a shared backbone $f_\theta$ produces multi-level features:
$$F_1^{(l)} = f_\theta^{(l)}(T_1), \quad F_2^{(l)} = f_\theta^{(l)}(T_2) \quad \text{for } l \in \{1, 2, 3, 4\}$$

### 3.2 Deep bi-temporal fusion and difference
$$F_{\text{cat}} = \left[ F_1^{(4)} \,\|\, F_2^{(4)} \,\|\, |F_1^{(4)} - F_2^{(4)}| \right]$$
$$F_{\Delta} = \text{Conv}_{3\times 3}(\text{Conv}_{1\times 1}(F_{\text{cat}})) \odot \sigma(\text{MLP}(\text{GAP}(F_{\Delta})))$$

### 3.3 Change grounding (mask head)
$$M_{\text{prob}} = \sigma(\text{Decoder}(F_{\Delta}, \{\Delta F^{(1)}, \Delta F^{(2)}, \Delta F^{(3)}\})) \in [0, 1]^{1 \times H \times W}$$

### 3.4 Vision-language cross-attention
Given question token embeddings $Q \in \mathbb{R}^{L_q \times d}$ and visual change tokens $V \in \mathbb{R}^{N_v \times d}$:
$$\text{Attention}(Q, V) = \text{softmax}\left(\frac{Q W_Q (V W_K)^T}{\sqrt{d_k}}\right) V W_V$$

### 3.5 Multi-task loss
$$\mathcal{L}_{\text{total}} = \lambda_{\text{vqa}} \mathcal{L}_{\text{CE}}(\hat{y}, y) + \lambda_{\text{bce}} \mathcal{L}_{\text{BCE}}(\hat{M}, M) + \lambda_{\text{dice}} \mathcal{L}_{\text{Dice}}(\hat{M}, M)$$

---

## 4. Inference

```python
from ml.C_VQA import ChangeVQAModel

model = ChangeVQAModel.from_checkpoint("checkpoints/change_detection_v2/best.pt")
result = model.analyze_pair("t1.tif", "t2.tif", "What has changed in these 2 images?")

result["answer"]
# "Yes. The change mask marks 69.91% of the scene as changed, in 29 regions; the 8 largest
#  are marked. By region, the model describes the change mainly as bare ground to water
#  (59% of the scene); the largest region is in the south sector."
result["bounding_boxes"][0]
# {"normalized_bbox": [...], "share": 0.335, "sector": "the south sector",
#  "label": "bare ground to water", "label_confidence": 0.59, "confidence": 0.93, ...}
```

How a pair is analysed:

- **Answer:** from the whole scene resized to 256 px, so it takes in context across the image.
- **Change mask:** pairs larger than one tile are analysed **tile by tile at their own resolution** (256 px tiles, 32 px overlap, averaged at the seams) instead of being squashed to 256 px. On a 768 px scene stitched from LEVIR-CD test pairs this raised mask IoU from 0.09 to 0.81.
- **Regions:** the mask is cleaned with a small morphological opening, so thin strands (river channels, roads) do not chain separate changes into one scene-wide blob, then split into connected regions. Patches under 0.1% of the scene are ignored. `total_regions` counts every region; the 8 largest are returned as `bounding_boxes`.
- **Region labels:** the model is asked `"What has changed in these 2 images?"` about a crop around each region (at least one tile of context), all regions in one batch. Its answer and confidence become the region's `label` and `label_confidence`.
- **Training resolution:** `model.gsd_range_m` is the range of metres per pixel the checkpoint was trained on. The backend only routes imagery within about 2x of that range to the model and uses the general VLM otherwise. Checkpoints trained before this was recorded report `(0.5, 0.5)` (LEVIR-CD).

`predict(ModelRequest)` implements the controller's `SpecialistModel` protocol and returns the same information as `ModelResponse` evidence (mask, one box per region, metrics).

CLI: `python -m ml.C_VQA.inference --checkpoint <ckpt> --t1 t1.png --t2 t2.png --query "..."`

---

## 5. Training data pipeline

### 5.1 Sources

Each dataset is listed in a JSON config and read by `sources.py`:

| `type` | Dataset | Resolution | What its labels support |
|---|---|---|---|
| `levir_hf` | LEVIR-CD 256 px crops, streamed from Hugging Face (no download) | 0.5 m | Building change |
| `levir` | LEVIR-CD, full 1024 px images (`<split>/A`, `B`, `label`) | 0.5 m | Building change |
| `s2looking` | S2Looking (`Image1`, `Image2`, `label1` new, `label2` demolished) | ~0.65 m | Building change with direction |
| `sysu` | SYSU-CD (`time1`, `time2`, `label`) | 0.5 m | Change of unlabelled type |
| `second` | SECOND (`im1`, `im2`, `label1`, `label2` colour maps) | ~0.5–3 m | From/to land cover |
| `oscd` | Onera Satellite Change Detection (Sentinel-2 bands) | 10 m | Urban change of unlabelled type |
| `xview2` | xView2 pre/post-disaster (polygon JSON) | ~0.8 m | Building damage |
| `dynamic_world` | Pairs exported by `gee_dynamic_world.py` | 10 m | From/to land cover, incl. water |
| `folder` | Any `<split>/<t1>/<t2>/<label>` layout (set `t1`, `t2`, `label`, `gsd_m`, `focus`) | any | Per `focus` |

Only `levir_hf` and `dynamic_world` have been run on real data so far. The others follow each dataset's documented layout; unexpected label colours or values raise instead of producing silent label noise, so run a dry run on a slice first.

### 5.2 Questions

`qa.py` generates each tile's questions from its labels, and only asks what the source actually labels: a building-change dataset never answers vegetation questions, and a binary dataset never names what an area became. Each question type has several paraphrases so the model sees varied wording. The answer vocabulary (`qa.ANSWERS`) is fixed and written to `dataset_info.json`.

### 5.3 Sentinel-2 pairs from Earth Engine

```bash
uv sync --all-packages --all-extras
uv run earthengine authenticate
uv run python -m ml.C_VQA.gee_dynamic_world --project <ee-project> --regions data/dw_regions.json --download data/dw_pairs
```

Each region in `regions.json` is `{"name", "lon", "lat", "size_km", "t1": [start, end], "t2": [start, end]}`. For every region it writes a median Sentinel-2 RGB composite and the Dynamic World label (mode) for each window. Check coverage before choosing windows: **monsoon-season optical imagery over India is almost always too cloudy** (0–1 clear scenes per site in July–September), so flood-related pairs use dry season (January–March) against post-monsoon (mid-September to mid-November). The labels are Dynamic World's own predictions, so treat them as noisy.

### 5.4 Build, train, evaluate

```bash
# data/sources.json: {"sources": [{"type": "levir_hf", "max_tiles": 2500},
#                                 {"type": "dynamic_world", "root": "data/dw_pairs", "scales": [1, 2]}]}
uv run python -m ml.C_VQA.prepare --config data/sources.json --out data/change_vqa_dryrun --max-tiles 30
uv run python -m ml.C_VQA.prepare --config data/sources.json --out data/change_vqa --max-tiles 5000

uv run python -m ml.C_VQA.train --data-root data/change_vqa --init-from checkpoints/c_vqa_best.pt \
    --checkpoint-dir checkpoints/change_detection_v2 --epochs 15 --batch-size 32 --lr 2e-4 --num-workers 4

uv run python -m ml.C_VQA.evaluate --checkpoint checkpoints/change_detection_v2/best.pt \
    --data-root data/change_vqa --split test --out results/change_vqa_v2_test.json
```

- `prepare` tiles every pair at scales 1, 2 and 4 (e.g. 0.5 m imagery also yields 1 m and 2 m tiles). `--max-tiles` caps training tiles per source; validation and test get a fifth of that each.
- `train` adds resolution degradation and per-image colour jitter, scores every source separately on validation (model selection uses their mean, so a small 10 m source is not drowned out), and records the covered `gsd_range_m` in the checkpoint. `--init-from` reuses every tensor whose shape still matches; the text embedding and answer head start fresh when the vocabularies change.
- On an RTX 5070 Ti laptop GPU, training runs at about 256 samples/s (about 2 min per epoch for the set above). Use `--num-workers 4`; 8 was slower.

In PowerShell, put each command on one line or continue lines with a backtick instead of `\`.

### 5.5 Current results

`checkpoints/change_detection_v2/best.pt`, trained on 2,500 LEVIR-CD and 538 Dynamic World training tiles, held-out test split (`results/change_vqa_v2_test.json`):

| Group | Answer accuracy | Mask F1 |
|---|---|---|
| LEVIR-CD, 0.5 m | 82.8% | 0.88 |
| Dynamic World, 10–20 m | 81.4% | 0.58 |
| Flooding yes/no | 91.7% | — |
| What changed | 90.2% | — |
| Water trend | 61.7% | — |
| Where (sector) | 26.5% | — |

The Dynamic World test split is small (60 tiles from about three sites), so treat its figures as rough. The "where" answer is unreliable; the region boxes and the mask show location better. More Sentinel-2 sites are the most useful next step.

---

## 6. Tests

```bash
uv run python -m unittest ml.C_VQA.test_pipeline
```

The pipeline tests use small synthetic fixtures to exercise question generation, each reader, tiling, one training step, tiled inference and region extraction. `test_cvqa.py` predates the pipeline; five of its tests need the original CDVQA download or refer to outdated defaults and fail without them.
