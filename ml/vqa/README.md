# ml/vqa — Single-Image Remote-Sensing VQA + Captioning

**Owner:** Yashvi (AI/ML — VQA / captioning)
**Scope:** single-image VQA and captioning only. Change detection, Change-VQA,
optical-SAR fusion, grounding, the frontend, and the main agentic controller
are **not** part of this module — see `ml/C_VQA` (owner: Harivansh) for
change-related work, and don't modify it from here.

## What this is

A specialist model for SatQuery AI (SIH26167) that answers natural-language
questions about a single remote-sensing image, or generates a caption for it,
using **SkyEyeGPT** (a remote-sensing-adapted VLM) run through the
**MiniGPT-v2** runtime.

> The official SkyEyeGPT GitHub repo does not currently ship a complete
> chatbot/inference codebase — it's marked "coming soon" upstream. SkyEyeGPT
> is distributed as a checkpoint meant to run on top of MiniGPT-v2. This
> module therefore = **MiniGPT-v2 runtime + SkyEyeGPT checkpoint + our
> wrapper**, not a standalone SkyEyeGPT package.

## File layout

```
ml/vqa/
├── __init__.py        # public API: VQAModel, BaseVQAModel, SkyEyeGPTModel
├── config.py           # paths, device, generation params, prompts
├── model.py            # BaseVQAModel interface + SkyEyeGPTModel backend
├── inference.py        # predict(ModelRequest) -> ModelResponse + CLI
├── preprocessing.py     # format detection, GeoTIFF/rasterio, RGB normalization
├── confidence.py        # documented, non-fabricated confidence scoring
├── evaluate.py          # RSVQA / VRSBench evaluation harness
├── test_vqa.py           # unit tests (fake backend, no GPU needed)
├── requirements.txt
├── README.md            # this file
└── training/             # Phase 4 stubs only — not wired in yet
    ├── __init__.py
    ├── dataset.py
    ├── lora_config.py
    └── train.py
```

`checkpoints/SkyEyeGPT.pth` (~680MB) is **not** included and must not be
committed — download it separately (see Setup) and add it to `.gitignore`.

## Setup

1. **Clone MiniGPT-v2** (not pip-installable):
   ```bash
   git clone https://github.com/Vision-CAIR/MiniGPT-4 third_party/MiniGPT-4
   export MINIGPT_REPO_DIR=$(pwd)/third_party/MiniGPT-4
   ```
   Follow that repo's own instructions to get its base LLM weights and
   `eval_configs/minigptv2_eval.yaml` in place.

2. **Download the SkyEyeGPT checkpoint** from its Hugging Face repo into
   `checkpoints/SkyEyeGPT.pth` (or set `SATQUERY_CHECKPOINTS_DIR`). Point
   `minigptv2_eval.yaml`'s `model.ckpt` field at this file.

3. **Install this module's requirements**, plus MiniGPT-4's own
   `requirements.txt` from inside its cloned repo:
   ```bash
   pip install -r ml/vqa/requirements.txt
   pip install -r third_party/MiniGPT-4/requirements.txt
   ```

4. Add to `.gitignore` (repo root):
   ```
   checkpoints/
   third_party/
   ```

## Roadmap (do these in order — don't skip ahead)

1. **Phase 1 — Prove basic inference works.** Load MiniGPT-v2 + the
   SkyEyeGPT checkpoint, run one image + one question through it, confirm
   you get a sane answer. No FastAPI, no training, no controller yet.
2. **Phase 2 — Wrapper.** `SkyEyeGPTModel` (`model.py`) with
   `answer()` / `caption()` / `generate()`.
3. **Phase 3 — SatQuery schema.** `VQAModel.predict()` (`inference.py`)
   implementing `ModelRequest -> ModelResponse` per
   `ml/controller/schema.py`.
4. **Phase 4 — Preprocessing.** RGB/PNG/JPEG + GeoTIFF, documented
   multispectral/SAR-to-RGB visualization (`preprocessing.py`).
5. **Phase 5 — Tests + CLI.** Already scaffolded in `test_vqa.py` and
   `inference.py`'s `__main__`.
6. **Phase 6 — Evaluation.** RSVQA-LR/HR and VRSBench, via
   `evaluate.py` and/or VRSBench's own official evaluation notebooks.
7. **Phase 7 — LoRA / domain adaptation.** Only after a baseline number
   exists. Stubs live in `training/` and deliberately raise
   `NotImplementedError` until then.
8. **Phase 8 — Controller integration.** Register `"vqa"` and
   `"captioning"` task hints against `VQAModel` in the agentic controller.

If a dependency/API from SkyEyeGPT or MiniGPT-v2 has changed since this was
written, check the actual upstream source before assuming this code is
correct — several comments in `model.py` flag exactly which calls to verify.

## CLI

```bash
python -m ml.vqa.inference --image sample.jpg --query "What type of land cover is visible?"
python -m ml.vqa.inference --image sample.jpg --task captioning
```

## Confidence — read before trusting the number

`ModelResponse.confidence` is **not** a calibrated probability of
correctness. SkyEyeGPT doesn't give us one just because it generated fluent
text. `confidence.py` either:
- rescales an average token log-probability if the backend exposes one, or
- falls back to a coarse text heuristic (empty/refusal → low, hedged →
  medium-low, direct answer → medium).

Treat it as a triage/ranking signal only. Don't quote it as "the model is
X% confident it's correct" in any report — see Section 23/28 of the
project spec for the reasoning and the roadmap toward a real calibration
method (self-consistency, answerability checks, specialist cross-validation).

## Modality caveats

- **Multispectral GeoTIFF** is converted to an RGB composite using a
  **documented** band mapping (`config.DEFAULT_MULTISPECTRAL_RGB_BANDS`,
  default `(3, 2, 1)` = R, G, B 1-indexed band positions). This is a
  visualization choice, not true multispectral understanding — state the
  mapping used whenever you report results.
- **SAR** input goes through the same pipeline as a 1-band grayscale
  visualization. This is **not** native SAR understanding. Don't claim
  "the model supports SAR" on this basis; SAR-specific analysis belongs to
  the fusion specialist in the full SatQuery pipeline.

## Error handling

`VQAModel.predict()` never raises for expected bad input — it returns a
`ModelResponse` with `answer="ERROR: ..."` and `confidence=0.0` for:
no image, more than one image (routes the user toward Change-VQA instead),
unsupported modality, unsupported file format, unreadable/corrupted image,
unsupported band count, and empty query in VQA mode. See `test_vqa.py` for
the full matrix.

## What this module deliberately does NOT do

- Train a VLM from scratch, or build a new 7B/13B model.
- Modify `ml/C_VQA` (Change-VQA — different owner, different interface).
- Implement change detection, optical-SAR fusion, grounding, the frontend,
  or the main controller.
- Assume SkyEyeGPT natively handles arbitrary multispectral/SAR data.
- Commit model weights to git.
- Invent MiniGPT-v2/SkyEyeGPT APIs — where this code makes an assumption
  about upstream API shape, it's called out in a comment for you to verify.
- Fabricate confidence scores or benchmark numbers.

## Extending to another backbone

`model.py` separates `BaseVQAModel` (interface) from `SkyEyeGPTModel`
(implementation) specifically so GeoChat, RS-LLaVA, or BLIP-2 can be
swapped in later by adding a new `BaseVQAModel` subclass — `inference.py`
and the controller integration don't need to change.
