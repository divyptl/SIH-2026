# SatQuery AI — Benchmark Results

### Training Summary - All Specialists

| Specialist | Backbone | Epochs | Final Train Loss | Best Val Metric | Params |
| --- | --- | --- | --- | --- | --- |
| Change-VQA | ConvNeXt-Tiny | 10 | 0.5924 | VQA 93.8% / IoU 80.9% | ~28M |
| Fusion (v1) | ConvNeXt-Tiny | 50 | 0.0212 | S2O 79.5% / O2S 77.4% | ~55M |
| Fusion (v2) | ConvNeXt-Tiny | 50 | 0.0414 | S2O 62.4% / O2S 68.8% | ~55M |

### Visual Grounding — Comparison with SOTA

| Model | Dataset | Acc@0.5 | Acc@0.7 | mIoU |
| --- | --- | --- | --- | --- |
| GeoChat (zero-shot) | VRSBench | 39.60% | — | — |
| GroundingDINO-T (zero-shot) | VRSBench | 42.30% | — | — |
| MGVLF | DIOR-RSVG | 66.20% | 47.40% | 56.80% |
| GeoGATE | VRSBench | 52.10% | 34.20% | — |
| SATGround | VRSBench | 56.80% | 38.50% | — |

### Change Detection / VQA — Comparison with SOTA

| Model | Dataset | Change F1 | VQA Acc |
| --- | --- | --- | --- |
| BIT (Chen et al.) | LEVIR-CD | 89.31% | — |
| ChangeFormer | LEVIR-CD | 90.40% | — |
| SChanger-base | LEVIR-CD | 92.87% | — |
| MTP | LEVIR-CD | 92.67% | — |
| CDVQA Baseline (Yuan et al.) | CDVQA | — | 63.30% |
| VisTA | QAG-360K | — | 78.50% |
| **SatQuery (Ours)** | **LEVIR-CD + DW** | **89.42%** | **93.78%** |

### Optical–SAR Fusion Retrieval — Comparison with SOTA

| Model | Dataset | SAR→Opt R@1 | SAR→Opt R@5 | Opt→SAR R@1 | Opt→SAR R@5 |
| --- | --- | --- | --- | --- | --- |
| MCRN (Lu et al.) | SEN1-2 | 18.70% | 44.20% | 17.90% | 43.10% |
| GaLR | SEN1-2 | 24.30% | 51.80% | 23.50% | 50.20% |
| CLIP-RS (ResNet-50) | SEN1-2 | 35.20% | 62.40% | 33.80% | 61.70% |
| DINOv2 (zero-shot) | SEN1-2 | 41.50% | 68.30% | 40.20% | 67.10% |
| **SatQuery (Ours)** | **SEN1-2 + QXS** | **79.50%** | **—** | **77.44%** | **—** |
