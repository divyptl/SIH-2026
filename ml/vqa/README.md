# ml/vqa — Single-Image Remote-Sensing VQA

**Owner:** Yashvi (AI/ML — VQA / captioning)
**Scope:** single-image VQA only. Change detection, Change-VQA, optical-SAR
fusion, grounding, the frontend, and the main agentic controller are **not**
part of this module — see `ml/C_VQA` (owner: Harivansh) for change-related
work, and don't modify it from here.

## What this is

A specialist model for SatQuery AI (SIH26167) that answers a natural-language
question about a single remote-sensing image. It is a custom PyTorch model
trained from an ImageNet-pretrained backbone, not a wrapper around an
existing VLM:

```
Image (3, 256, 256) ──► ConvNeXt-Tiny ──► 1×1 conv projector ──► 7×7 = 49 visual tokens (256-d)
                                                                        │
Question ──► word tokenizer ──► text encoder (256-d, ≤ 32 tokens) ──► cross-attention
                                                                   (2 layers, 8 heads)
                                                                        │
                                                          pooled ──► MLP answer head
                                                                        │
                                                   one of 64 answer classes + confidence
```

All hyperparameters live in `ModelConfig` (`config.py`); the backbone can be
switched to `resnet18`, `resnet34` or `resnet50`.

**The model is a classifier, not a text generator.** It picks one answer from
a fixed vocabulary (`CANONICAL_ANSWERS` in `model.py`), which today has 24
named answers:

- `yes`, `no`
- land use / cover: `residential`, `industrial`, `agricultural`,
  `commercial`, `forest`, `water`, `bare land`, `road`
- counts: `0` – `10`, `more than 10`
- density: `sparse`, `dense`

The other 40 of the 64 output slots are unnamed placeholders
(`answer_category_24` …). "Captioning" (`--task captioning`) only sends the
fixed question "Describe this image." through the same classifier, so it
returns a single label, not a caption.

## Status — not usable yet

- **The package does not import.** Commit `558493b` rewrote `config.py` and
  `model.py`, but `__init__.py`, `inference.py`, `preprocessing.py` and
  `test_vqa.py` still import names that no longer exist:
  `VQAConfig`, `DEFAULT_CONFIG`, `SUPPORTED_MODALITIES`,
  `SUPPORTED_EXTENSIONS`, `DEFAULT_MULTISPECTRAL_RGB_BANDS` (from
  `config.py`) and `BaseVQAModel`, `SkyEyeGPTModel` (from `model.py`).
  Because `ml/vqa/__init__.py` fails, `python -m ml.vqa.training.train`
  fails too.
- **No trained checkpoint.** Nothing exists under `checkpoints/vqa/`.
- **No measured accuracy.** There is no RSVQA / VRSBench result for this
  model yet — don't quote one.
- **Not wired into the backend.** The tool registry lists `vqa` and
  `caption` with `specialist_module="ml.vqa"`, but no specialist is
  attached, so single-image questions get the deterministic fallback answer
  (see `backend/README.md`).
- **Unknown answers are labelled `yes`.** `VQADataset` maps any answer
  outside `CANONICAL_ANSWERS` to class 0. RSVQA-LR contains many such
  answers (area ranges, counts above 10), so extend the vocabulary or drop
  those questions before training.
- **Leftovers from the earlier SkyEyeGPT / MiniGPT-v2 plan:** `confidence.py`
  (not used by the new model), `training/lora_config.py` (stub), and the
  MiniGPT / `peft` / `bitsandbytes` entries in `requirements.txt`.

## File layout

```
ml/vqa/
├── __init__.py          # public API (currently broken — see Status)
├── config.py            # ModelConfig (architecture) + TrainConfig (training)
├── model.py             # VisionBackbone, SimpleTokenizer, TextEncoder,
│                        # CrossModalFusion, VQAHead, VQAModel, VQALoss
├── inference.py         # VQAModel.predict(ModelRequest) -> ModelResponse + CLI
├── preprocessing.py     # format detection, GeoTIFF via rasterio, RGB normalization
├── evaluate.py          # RSVQA-LR/HR and VRSBench accuracy harness
├── confidence.py        # unused — from the SkyEyeGPT plan
├── test_vqa.py          # unit tests (need updating for the new model)
├── requirements.txt
└── training/
    ├── dataset.py       # VQADataset over a JSONL manifest
    ├── train.py         # training loop
    └── lora_config.py   # unused stub
```

## Data

Training and evaluation both read a flat JSONL manifest at
`{data_root}/{split}.jsonl`, one record per line:

```json
{"image": "path/to/img.jpg", "question": "Is there a road?", "answer": "yes", "modality": "optical"}
```

RSVQA and VRSBench ship their own formats, and no converter exists in the
repo yet; RSVQA-LR is not in `data/` either. Write a one-off converter that
produces `train.jsonl`, `val.jsonl` and `test.jsonl`.

## Training

```bash
python -m ml.vqa.training.train --data-root data/rsvqa-lr --epochs 30 --batch-size 16 --lr 1e-4
```

Defaults (`TrainConfig`): 256×256 input with colour-jitter augmentation,
AdamW with lr 1e-4 for the heads and 2e-5 for the pretrained backbone,
weight decay 1e-4, mixed precision on CUDA, and `DataParallel` when more
than one GPU is visible. Each epoch is validated on exact-match answer
accuracy, and the best epoch is saved to `checkpoints/vqa/best.pt` with its
`ModelConfig`, which `VQAModel.from_checkpoint()` reads back.

## Inference

```bash
python -m ml.vqa.inference --image sample.jpg --query "Is there a water body?"
```

`inference.VQAModel` loads `checkpoints/vqa/best.pt` if it exists and
otherwise runs **untrained weights** (it logs a warning), so its answers mean
nothing until a checkpoint is trained. Images are resized to 256×256 and
ImageNet-normalized.

## Evaluation

```bash
python -m ml.vqa.evaluate --dataset rsvqa-lr --split test \
    --data-root data/rsvqa-lr --checkpoint checkpoints/vqa/best.pt \
    --out docs/vqa_results.md
```

## Confidence — read before trusting the number

`confidence` is the highest softmax probability over the 64 answer classes.
It is not calibrated: treat it as a ranking signal, not as "X% likely to be
correct", until it has been checked against held-out accuracy.

## Modality caveats

- **Multispectral GeoTIFF** is converted to an RGB composite using a
  documented band mapping (default `(3, 2, 1)` = R, G, B as 1-indexed band
  positions). This is a visualization choice, not multispectral
  understanding — state the mapping whenever you report results.
- **SAR** goes through the same pipeline as a 1-band grayscale image. The
  model is trained on optical RGB, so this is **not** SAR understanding;
  SAR analysis belongs to the fusion specialist.

## Error handling

`VQAModel.predict()` doesn't raise for bad input; it returns a
`ModelResponse` with `answer="ERROR: ..."` and `confidence=0.0` for: no
image, more than one image (that belongs to Change-VQA), an unsupported
modality, an unsupported file format, an unreadable image, or an unsupported
band count.

## What this module deliberately does NOT do

- Modify `ml/C_VQA` (Change-VQA — different owner, different interface).
- Implement change detection, optical-SAR fusion, grounding, the frontend,
  or the main controller.
- Commit model weights to git.
- Fabricate confidence scores or benchmark numbers.
