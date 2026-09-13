> Note: These diagrams use Mermaid code fences (` ```mermaid `). Obsidian renders these natively — no plugin needed, just paste this file in as-is.

# Text-Guided Region Grounding Module (`ml/grounding/`)

Given a satellite image + a plain-English phrase (e.g. *"water body near bridge"*), this module returns bounding boxes around the matching regions. It's built on **GroundingDINO**, an open-vocabulary detector — it isn't limited to a fixed list of classes, it can localize **anything describable in language**.

VRSBench dataset: **~52K object references over ~29K remote-sensing images**.

---

## 1. High-level architecture

```mermaid
flowchart TB
    subgraph Inputs
        IMG["Satellite image<br/>(H × W × 3)"]
        TXT["Text query<br/>'buildings . roads .'"]
    end

    IMG --> BACKBONE["Swin-T backbone<br/>(vision features)<br/><i>frozen by default</i>"]
    TXT --> BERT["BERT text encoder<br/>(text features)<br/><i>trainable</i>"]

    BACKBONE --> FUSION["Cross-modal decoder<br/>(fuses image + text)"]
    BERT --> FUSION

    FUSION --> HEADS["Detection heads"]
    HEADS --> OUT["Bounding boxes + scores<br/>+ matched text spans"]
```

---

## 2. Module map — what each file does

```mermaid
flowchart LR
    config["config.py<br/>ModelConfig / TrainConfig<br/>(all hyperparameters)"]
    model["model.py<br/>GroundingModel<br/>(load + freeze + forward + post-process)"]
    dataset["dataset.py<br/>VRSBenchGroundingDataset<br/>(loads + normalizes VRSBench)"]
    transforms["transforms.py<br/>GroundingAugmentation<br/>(flip / color jitter / blur)"]
    train["train.py<br/>fine-tuning loop"]
    inference["inference.py<br/>GroundingInference<br/>(production API)"]
    test["test_grounding.py<br/>smoke tests"]

    config --> model
    config --> train
    dataset --> train
    transforms --> train
    model --> train
    model --> inference
    train -- "checkpoint .pt" --> inference
    model --> test
    dataset --> test
    transforms --> test
```

---

## 3. Training pipeline

```mermaid
flowchart TD
    A["VRSBench (HuggingFace Hub)"] --> B["VRSBenchGroundingDataset<br/>normalize boxes: 0–100 xyxy → 0–1 cx,cy,w,h"]
    B --> C["DataLoader + collate_fn<br/>(batches images/text/labels)"]
    C --> D["GroundingAugmentation<br/>(flip, color jitter, blur)"]
    D --> E["HF Processor<br/>(tokenize text + normalize pixels)"]
    E --> F["GroundingModel.forward()<br/>(backbone frozen, text encoder trainable)"]
    F --> G["Losses: loss_ce + loss_bbox + loss_giou"]
    G --> H["AdamW optimizer<br/>+ cosine LR schedule w/ warmup<br/>+ gradient clipping (max_norm=0.1)"]
    H --> I{"epoch % save_every == 0<br/>or new best val_loss?"}
    I -->|yes| J["Save checkpoint (.pt)"]
    I -->|no| K["Next epoch"]
    J --> K
    K --> F
```

---

## 4. Inference flow (what runs in production)

```mermaid
sequenceDiagram
    participant App as Backend / Agentic Controller
    participant Inf as GroundingInference
    participant Proc as HF Processor
    participant Model as GroundingModel

    App->>Inf: ground(image, "flooded areas near highway")
    Inf->>Inf: load image, ensure query ends with "."
    Inf->>Proc: preprocess(image, text)
    Proc-->>Inf: pixel_values, input_ids, attention_mask
    Inf->>Model: forward(pixel_values, input_ids, ...)
    Model-->>Inf: raw logits + pred_boxes
    Inf->>Proc: post_process_grounded_object_detection(box_threshold, text_threshold)
    Proc-->>Inf: filtered boxes, scores, matched labels
    Inf-->>App: [{box, score, label}, ...]
```

---

## 5. Loading paths

```mermaid
flowchart LR
    A["GroundingInference"] --> B["from_pretrained()<br/>zero-shot, no training needed"]
    A --> C["from_checkpoint(path)<br/>fine-tuned weights"]
    B --> D["ground() / ground_batch() /<br/>get_analysis() / visualize()"]
    C --> D
```

---

## Key numbers to remember

| Thing | Value |
|---|---|
| Base model | `IDEA-Research/grounding-dino-tiny` (~172M params) |
| Dataset | VRSBench — 52K object references / 29K images |
| Backbone frozen by default | Yes (Swin-T) |
| Text encoder frozen | No (BERT stays trainable) |
| Batch size | 4 |
| Epochs | 20 |
| LR | 1e-5 (backbone LR 1e-6 if unfrozen) |
| Gradient clip | max_norm = 0.1 |
| box_threshold / text_threshold | 0.25 / 0.25 |

## Talking points

- **Open-vocabulary, not fixed-class**: can ground phrases never seen during training (unlike classic object detectors with a hardcoded label set).
- **Backbone frozen** → cheaper fine-tuning, avoids catastrophic forgetting on a relatively small dataset.
- **Three joint losses** (`loss_ce`, `loss_bbox`, `loss_giou`) → is there an object, where exactly, how well does it overlap ground truth.
- **`get_analysis()`** is the exact dict shape the agentic controller consumes downstream.
- **`visualize()`** draws boxes directly on the image — usable for a live demo.
