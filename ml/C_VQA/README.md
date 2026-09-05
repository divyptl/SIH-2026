# SatQuery AI — Change Detection / Change-VQA Module

**Lead:** Harivansh (AI/ML — Change Detection)  
**Task:** Bi-Temporal Remote-Sensing Change Detection & Visual Question Answering  
**Model:** Siamese Vision Encoder + VLM Head  
**Dataset:** CDVQA (Change Detection Visual Question Answering)

---

## 1. Overview

The **Change Detection / Change-VQA** module is a specialist model designed for multi-temporal satellite imagery analysis. Given two co-registered optical images taken at different times ($T_1$ pre-change and $T_2$ post-change) and a natural language question, the model:

1. Identifies and segments pixel-level environmental and anthropogenic changes.
2. Extracts bounding boxes around changed zones (e.g., urban development, deforestation, water bodies).
3. Synthesizes a natural language answer backed by confidence scores and visual evidence.
4. Complies with the **SatQuery AI Agentic Controller** `SpecialistModel` protocol (`predict(ModelRequest) -> ModelResponse`).

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
│ Bounding box extraction          │     │ - Answer Classification / Gen   │
└──────────────────────────────────┘     └─────────────────────────────────┘
         │                                         │
         ▼                                         ▼
  Change Mask & BBoxes                      Predicted Answer + Confidence
(Evidence: Visual Grounding)              ("Built-up area increased by 15%")
         │                                         │
         └────────────────────┬────────────────────┘
                              ▼
                     ModelResponse Schema
            (Agentic Controller Protocol Compliant)
```

---

## 2. Directory Structure

```
ml/C_VQA/
├── __init__.py               # Package exports
├── config.py                 # Architecture (ModelConfig) & training (TrainConfig) dataclasses
├── model.py                  # SiameseBackbone, DifferenceModule, MaskHead, VLM Head, Loss
├── dataset.py                # CDVQADataset, collate_fn (real-life CDVQA benchmark loader)
├── transforms.py             # Synchronized bi-temporal spatial transforms & normalization
├── inference.py              # ChangeVQAModel wrapper & SpecialistModel implementation
├── train.py                  # Multi-task training script with warmup & cosine scheduler
├── test_cvqa.py              # Complete unit and integration test suite
└── README.md                 # Module documentation

ml/change_detection/
└── __init__.py               # Alias re-exporting ml.C_VQA for naming compatibility
```

---

## 3. Mathematical Formulation

### 3.1 Siamese Multi-Scale Feature Extraction
Given $T_1, T_2 \in \mathbb{R}^{3 \times H \times W}$, a shared backbone $f_\theta$ produces multi-level features:
$$F_1^{(l)} = f_\theta^{(l)}(T_1), \quad F_2^{(l)} = f_\theta^{(l)}(T_2) \quad \text{for } l \in \{1, 2, 3, 4\}$$

### 3.2 Deep Bi-temporal Fusion & Difference
Deep visual change tokens are computed by concatenating features and their absolute difference:
$$F_{\text{cat}} = \left[ F_1^{(4)} \,\|\, F_2^{(4)} \,\|\, |F_1^{(4)} - F_2^{(4)}| \right]$$
$$F_{\Delta} = \text{Conv}_{3\times 3}(\text{Conv}_{1\times 1}(F_{\text{cat}})) \odot \sigma(\text{MLP}(\text{GAP}(F_{\Delta})))$$

### 3.3 Visual Change Grounding (Mask Head)
A multi-scale decoder reconstructs the full-resolution change probability map:
$$M_{\text{prob}} = \sigma(\text{Decoder}(F_{\Delta}, \{\Delta F^{(1)}, \Delta F^{(2)}, \Delta F^{(3)}\})) \in [0, 1]^{1 \times H \times W}$$

### 3.4 Vision-Language Cross-Attention
Given question token embeddings $Q \in \mathbb{R}^{L_q \times d}$ and visual change patch tokens $V \in \mathbb{R}^{N_v \times d}$:
$$\text{Attention}(Q, V) = \text{softmax}\left(\frac{Q W_Q (V W_K)^T}{\sqrt{d_k}}\right) V W_V$$

### 3.5 Multi-Task Loss
Joint optimization of question-answering accuracy and pixel-level segmentation:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{vqa}} \mathcal{L}_{\text{CE}}(\hat{y}, y) + \lambda_{\text{bce}} \mathcal{L}_{\text{BCE}}(\hat{M}, M) + \lambda_{\text{dice}} \mathcal{L}_{\text{Dice}}(\hat{M}, M)$$

---

## 4. Usage

### 4.1 Python API (Direct Inference)

```python
from ml.C_VQA import ChangeVQAModel

# Load model (from checkpoint or initialized)
model = ChangeVQAModel.from_checkpoint("checkpoints/change_detection/best.pt")

# Analyze a bi-temporal pair
result = model.analyze_pair(
    img_t1="data/cdvqa/train/images_t1/sample_0001_t1.png",
    img_t2="data/cdvqa/train/images_t2/sample_0001_t2.png",
    query="Has the built-up area increased, decreased, or remained unchanged?",
)

print(result["answer"])
# "The built-up area has increased. New construction activity is detected in the north-east quadrant, affecting approximately 12.4% of the analyzed region (2 distinct change clusters identified)."
print(f"Confidence: {result['confidence']:.2%}")
print(f"Change BBoxes: {result['bounding_boxes']}")
```

### 4.2 Agentic Controller Protocol

```python
from ml.controller.schema import ModelRequest
from ml.C_VQA import ChangeVQAModel

model = ChangeVQAModel.from_checkpoint()

request = ModelRequest(
    query="What changed between these two dates and where did it happen?",
    images=["path/to/t1.png", "path/to/t2.png"],
    modalities=["optical", "optical"],
    task_hint="change_detection",
)

response = model.predict(request)
# Returns ModelResponse(answer=..., confidence=..., evidence=[mask_evidence, bbox_evidence, metrics_evidence], ...)
```

### 4.3 Command Line Interface (CLI)

```bash
# Run inference on an image pair
python -m ml.C_VQA.inference \
  --t1 path/to/t1.png \
  --t2 path/to/t2.png \
  --query "Did new structures appear between the two dates?"

# Check real-life CDVQA dataset integrity
python data/cdvqa/download_cdvqa.py --check

# Train the model on real remote-sensing dataset
python -m ml.C_VQA.train --epochs 20 --batch-size 8 --backbone resnet18 --lr 1e-4

# Run test suite against real dataset
python -m unittest ml/C_VQA/test_cvqa.py -v
```
