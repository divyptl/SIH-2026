# SatQuery AI — ML

The specialist models behind the SatQuery AI controller. This directory is the
`ml` package: the backend imports it (`from ml.grounding.inference import …`),
and training scripts run as modules from the repository root
(`uv run python -m ml.C_VQA.train …`).

| Module | Task | Docs |
|---|---|---|
| `C_VQA/` | Change detection and change-VQA on bi-temporal pairs | [C_VQA/README.md](C_VQA/README.md) |
| `grounding/` | Text-guided region grounding (GroundingDINO + re-ranker) | [docs/grounding-module-explained.md](../docs/grounding-module-explained.md) |
| `fusion/` | Optical–SAR dual encoder and terrain classification | — |
| `vqa/` | Single-image VQA and captioning | [vqa/README.md](vqa/README.md) |
| `controller/` | `SpecialistModel` request/response schema shared with the backend | — |
| `datasets/` | Dataset loaders (SEN1-2, QXS-SAROPT) | — |

## Environment

`ml/` and `backend/` form one uv workspace with a single `.venv` at the
repository root. From the root:

```bash
uv sync --all-packages --all-extras
```

The `ml` package's own dependencies cover inference only. Training extras
(`train`: datasets, rasterio, torchgeo, …) and the Earth Engine client (`gee`)
are optional; the command above installs both.

Checkpoints go in `checkpoints/` at the repository root (gitignored). The
backend reads their paths from `backend/.env`; see
[backend/README.md](../backend/README.md#fine-tuned-specialists).
