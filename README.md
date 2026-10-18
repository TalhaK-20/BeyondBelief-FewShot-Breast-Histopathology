<div align="center">

# Beyond Belief

### A Leakage-Aware Benchmark of Dempster–Shafer, Bayesian, Fuzzy and Learned Snapshot Fusion for Few-Shot Breast Histopathology

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C.svg)](https://pytorch.org/)
[![Datasets](https://img.shields.io/badge/Datasets-BreakHis%20%7C%20BACH-green.svg)](#datasets)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](paper/Beyond_Belief_Leakage-Aware_Benchmark_of_Dempster-Shafer_Bayesian_Fuzzy_and_Learned_Snapshot_Fusion_for_Few-Shot_Breast_Histopathology.pdf)

**Talha Khalid · Ashfaq Ahmad · Iqra Javed**
University of Management and Technology (UMT), Lahore, Pakistan

**Repository:** `https://github.com/[USERNAME]/[REPOSITORY]` *(placeholder: replace it with the public URL before publication)*

</div>

---

## Table of contents

- [Highlights](#highlights)
- [Overview](#overview)
- [Architecture](#architecture)
- [Loopholes fixed](#loopholes-fixed-summary)
- [Residual issues](#residual-issues-in-this-corrected-pipeline-found-during-the-paper-audit)
- [Datasets](#datasets)
- [Results](#results)
- [Hardware, software and runtime](#hardware-software-and-runtime)
- [Repository structure](#repository-structure)
- [How to reproduce](#how-to-reproduce)
- [Citation](#citation)
- [License](#license)

## Highlights

- **20 loopholes audited and fixed** in a published few-shot ViT + Dempster–Shafer framework. The most serious: its prototypes were built from logits instead of embeddings, and its query and test images were pooled into one accuracy figure.
- **14 snapshot-fusion strategies** are compared on one shared evidence cache:
  - **Dempster–Shafer:** 5 combination rules (Dempster, Yager, Dubois–Prade, Murphy, PCR5), plus the related evidential-reasoning algorithm.
  - **Bayesian:** model averaging.
  - **Fuzzy:** the Choquet integral.
  - **Learned:** stacking and late fusion.
- **2 public datasets** (BreakHis, BACH), each run as an independent experiment with a separate held-out test set.
- **Best BreakHis test result:** 77.32% accuracy, 70.54% macro-F1 and 78.51% AUC from only **10 labelled support images**. With the support labels alone, soft voting reaches 75.46%.
- **Reported transparently:** confidence intervals, sensitivity/specificity, belief/plausibility uncertainty, Grad-CAM/LIME explanations, hardware and runtime.

## Overview

Pani et al. (CVIP) proposed a prototype-based few-shot classifier for breast histopathology. It fine-tunes a ViT-Small on a 5-shot support set, keeps several training snapshots and fuses their predictions with Dempster–Shafer (DS) evidence theory. Reference code: [SaptarshiPani/DST-FewShot](https://github.com/SaptarshiPani/DST-FewShot).

This repository contains a **critical audit** of that implementation and a **corrected, leakage-aware pipeline**. The pipeline is then used as a controlled testbed to compare 14 snapshot-fusion strategies on two public datasets. The work has four parts:

1. **Audit.** 20 loopholes found in the released notebook, each with its fix (`vit-snapshot_loopholes.txt`).
2. **Corrected pipeline.** Real embeddings instead of logits, a partially frozen backbone, seeding, and separate query and test reporting.
3. **Three fusion families:**
   - **RW-DSFormer:** reliability-discounted DS fusion with 5 combination rules.
   - **ProtoEvidNet:** 4 fusers that do not use DS theory.
   - **SnapEvid-ViT:** a late fusion of the best member of each family.
4. **Benchmark** on BreakHis and BACH, with uncertainty analysis, explainability and runtime reporting.

**Main finding:** once the pipeline is sound, DS fusion is no better than plain soft voting. On BreakHis every DS rule stays within 0.5 points of it. A stacking meta-learner gives the best BreakHis result (77.32% test accuracy), but it uses the labels of the query images. On BACH, no method is distinguishable from chance.

## Architecture

![Pipeline](paper/figures/fig1_architecture.png)

| Component | Setting |
|---|---|
| Backbone | `vit_small_patch16_224` (timm, ImageNet-pretrained), 21.67 M parameters |
| Trainable | Last 2 transformer blocks, final LayerNorm and a 2-way head: 3.55 M parameters (16.39%) |
| Embedding | 384-D pre-logit CLS token (`forward_features` → `forward_head(pre_logits=True)`) |
| Classifier | Cosine similarity to class prototypes, divided by a per-snapshot temperature τ = std(query similarities), then softmax |
| Training | 5-shot support set (10 images), cross-entropy, AdamW (lr 1e-3, weight decay 1e-4), 30 epochs |
| Snapshots | Every 6 epochs (epochs 6, 12, 18, 24, 30), so S = 5 |
| Reliability | r_s = query accuracy of snapshot s ÷ best query accuracy |

### Fusion methods (14 in total)

| Family | Methods |
|---|---|
| Baselines | Last snapshot; hard voting; soft voting; legacy heuristic DS from the original code (ablation) |
| **RW-DSFormer** | Entropy-based BBA, then Shafer discounting by r_s, then combination by **Dempster**, **Yager**, **Dubois–Prade**, **Murphy** or **PCR5**, then the pignistic decision with Bel/Pl |
| **ProtoEvidNet** | Bayesian model averaging; Choquet fuzzy integral (Sugeno λ-measure); Evidential Reasoning (Yang–Xu); logistic-regression stacking |
| **SnapEvid-ViT** | Weighted average of the best RW-DSFormer member and the best ProtoEvidNet member (descriptive only: the members are selected on the test set) |

## Loopholes fixed (summary)

The full list is in `vit-snapshot_loopholes.txt` and in the first markdown cell of the notebook.

| ID | Problem in the original notebook | Fix |
|---|---|---|
| A1 | Prototypes and fusion computed on the **2-D logits**, not on embeddings | 384-D CLS embedding |
| A2 | All ~22 M parameters fine-tuned on 10 images for 50 epochs | Backbone frozen except the last 2 blocks; AdamW with weight decay |
| A3 | No random seed | Seed 42 for Python, NumPy, PyTorch and cuDNN |
| A4 | Query and test images **pooled** into one accuracy figure | Query and test reported separately |
| A5 | No check that the Train and Test class orders match | Assertion |
| A6 | Accuracy was the only metric | Macro precision, recall and F1; ROC-AUC; confusion matrices |
| B7–B18 | Unused batch size, decoding every image to read labels, mutated shared model, leaked loop variables, 3× recomputation, deprecated hooks, hard-coded paths, unbatched LIME | Shared evidence cache, batched loaders, safe hooks, path auto-detection |
| N1–N3 | Exact-name dataset finder; softmax over raw cosine similarities (collapses Bel/Pl to 0/1); LIME label `KeyError` | Signature-based discovery, temperature scaling, explicit LIME labels |

## Residual issues in this corrected pipeline (found during the paper audit)

These issues are still present in `BeyondBelief_SnapshotFusion_Benchmark.ipynb`. They are reported in the paper (Table X) and are the next things to fix.

| ID | Issue | Effect |
|---|---|---|
| R1 | **BreakHis split is image-level: all 81 patient IDs occur in both Train and Test** | BreakHis results are optimistic. A patient-disjoint split is required |
| R2 | Reliabilities, BMA weights and the stacking learner use the labels of up to 5,527 query images | Not strictly 5-shot |
| R3 | SnapEvid-ViT members and weights are chosen on the test split | Descriptive only |
| R4 | `support_loader` uses `shuffle=False` and `batch_size=1`, so updates arrive ordered by class | Oscillating loss (0% training accuracy at epoch 5) |
| R5 | Single seed and single episode | No confidence intervals over episodes |
| R6 | BACH images (2048×1536) are squashed to 224×224 | Nuclear detail is lost |
| R7 | `.DS_Store` and `Thumbs.db` files enter the BACH split, and the split is stored as **symlinks** | 59 instead of 60 benign test images. The symlinks are broken on Windows/OneDrive |
| R8 | For K = 2, Yager ≡ Dubois–Prade. The Choquet densities sum to 1, so λ = 0 and Choquet reduces to a weighted mean. ER's incompleteness mass is 0 | Fewer distinct fusers than the names suggest |
| R9 | Grad-CAM and LIME explain the linear head, not the prototype/fusion decision | The explanations are indirect |
| R10 | Only 5-shot evaluated | Sensitivity to the number of shots is unknown |
| R11 | Training wall-clock time not instrumented | Some timings are estimates |

## Datasets

**2 public datasets** are used, each as a fully independent experiment.

| Dataset | Source | Images | Classes used | Train (B/M) | Test (B/M) | Support | Query |
|---|---|---:|---|---:|---:|---:|---:|
| [BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) | Spanhol et al., IEEE TBME 2016 | 7,909 (40×/100×/200×/400× pooled) | benign / malignant | 1,736 / 3,801 | 744 / 1,628 | 10 | 5,527 |
| [BACH (ICIAR 2018)](https://iciar2018-challenge.grand-challenge.org/Dataset/) ([Zenodo](https://zenodo.org/records/3632035)) | Aresta et al., MedIA 2019 | 400 labelled photos | Normal+Benign → benign; InSitu+Invasive → malignant | 141 / 140 | 59 / 60 | 10 | 271 |

- **Splits:** stratified 70/30 per class with seed 42. The split folders are `BreakHis_AllMag_split/` and `BACH_AllClass_split/`.
- **Kaggle mirror used for BreakHis:** `waseemalastal/breakhist-breast-cancer-histopathological-dataset`.
- **Preprocessing:** images are resized to 224×224 and normalised with mean = std = 0.5.
- The datasets are not redistributed. Download them from the official sources and follow their licences.

## Results

All results come from a single seeded run (seed 42) in the 5-shot binary setting. Values are percentages.

### Held-out test split

| Method | BreakHis Acc | BreakHis F1 | BreakHis AUC | BACH Acc | BACH F1 | BACH AUC |
|---|---:|---:|---:|---:|---:|---:|
| Last snapshot (baseline) | 73.90 | 67.83 | 75.29 | 53.78 | 47.89 | **65.82** |
| Hard voting | 73.99 | 67.87 | 77.04 | 53.78 | 47.89 | 65.37 |
| Soft voting | 75.46 | 68.06 | 77.04 | 52.94 | 47.28 | 65.37 |
| Legacy DS heuristic (original) | 75.51 | 68.00 | 76.99 | 52.94 | 47.28 | 65.68 |
| RW-DSFormer: Dempster | 75.80 | 67.84 | 77.37 | 52.10 | 47.84 | 65.54 |
| RW-DSFormer: Yager | 75.51 | 67.58 | 77.20 | 52.10 | 47.84 | 65.56 |
| RW-DSFormer: Dubois–Prade | 75.51 | 67.58 | 77.20 | 52.10 | 47.84 | 65.56 |
| RW-DSFormer: Murphy | 75.51 | 67.54 | 77.25 | 52.10 | 47.84 | 65.45 |
| RW-DSFormer: PCR5 | 75.93 | 67.96 | 77.39 | 52.10 | 47.84 | 65.54 |
| ProtoEvidNet: BMA | 75.38 | 67.92 | 77.09 | 52.10 | 46.66 | 65.37 |
| ProtoEvidNet: Choquet | 75.38 | 67.95 | 77.06 | 52.94 | 47.28 | 65.40 |
| ProtoEvidNet: ER | 75.63 | 68.11 | 77.14 | 52.10 | 46.66 | 65.48 |
| **ProtoEvidNet: Stacking*** | **77.32** | **70.54** | **78.51** | **54.62** | **53.89** | 64.44 |
| SnapEvid-ViT‡ | 77.19 | 70.20 | 78.41 | 52.94 | 50.39 | 64.77 |

\*Fit on the labels of the query split. ‡Members and weights selected on the test split. Majority-class accuracy is 68.63% on the BreakHis test set and 50.42% on the BACH test set.

- **Confidence intervals (Wilson 95%):**
  - BreakHis stacking: 75.6–79.0%; baseline: 72.1–75.6%.
  - BACH stacking: 45.7–63.3%. This interval contains chance.
- **Sensitivity and specificity on the BreakHis test set:**
  - Baseline: 85.50 / 48.52.
  - Soft voting: 90.05 / 43.55.
  - Stacking: 91.28 / 46.77.
  - Most of the accuracy gain is a shift towards the majority (malignant) class.
- **Dempster–Shafer uncertainty (mean test Bel / Pl / interval width):**
  - Legacy heuristic on BreakHis: 0.938 / 0.938 / 0.001. It expresses almost no ignorance.
  - RW-DSFormer Dempster on BreakHis: 0.694 / 0.937 / 0.243.

### Comparison with published BreakHis studies (verified in the original sources)

| Study | Training labels | Split / evaluation | Task | Result (%) |
|---|---|---|---|---|
| Spanhol et al., IEEE TBME 2016 | Full (~70% of patients) | Patient-level 70/30, 5 folds | Binary | 80–85 (patient-level accuracy) |
| Bayramoglu et al., ICPR 2016 | Full (~70% of patients) | Patient-level 70/30, 5 folds | Binary, magnification-independent | 83.25 (patient score) |
| Priyanka et al., Front. Digit. Health 2026 | Full | Patient-grouped 5-fold CV | Binary | 92.67 ± 4.35 (ResNet-50) |
| FHIST (Shakeri et al., 2022) | 5 per class, cross-domain | 1,000 episodes | 5-way, 8 subtypes (chance 20%) | 57.9 ± 0.57 (best 60.4) |
| Pani et al. (base) | 2 / 3 / 5 per class | Query + test pooled | Binary | Not comparable |
| **[Ours]** | **5 per class (10 images)** | **Image-level 70/30, held-out test** | **Binary** | **77.32 (stacking); 75.46 using only the support labels** |

| Protocol and reporting criteria met (of 14) | Spanhol | Bayramoglu | Priyanka | FHIST | Pani (base) | **[Ours]** |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Total | 3 | 3 | 5 | 6 | 4 | **12** |

The 14 criteria are listed in Table IX of the paper. This study has two gaps: it does not use a patient-disjoint split, and it does not repeat runs over multiple episodes.

### Figures

All figures are in `output/`. The versions used in the paper are in `paper/figures/`.

| File | Content |
|---|---|
| `eda_class_distribution.png`, `eda_*_samples.png` | Class balance and sample images |
| `confusion_matrices_{BreakHis,BACH}.png` | Confusion matrices for the 5 key methods × query/test |
| `roc_curves_{BreakHis,BACH}.png` | ROC curves |
| `query_visualizations_{BreakHis,BACH}.png` | Original image, activation map, Grad-CAM and LIME |

## Hardware, software and runtime

| Component | Specification |
|---|---|
| CPU | AMD Ryzen 7 5800X (8 cores / 16 threads) |
| RAM | 60 GB |
| GPU | NVIDIA GeForce RTX 3080, 10 GB |
| OS | Ubuntu Linux |
| Software | Python 3.12.3, PyTorch 2.11 + CUDA 12.8, timm, scikit-learn, LIME, OpenCV, scikit-image |

| Stage | BreakHis | BACH |
|---|---:|---:|
| Support fine-tuning (30 epochs) + 5 snapshot diagnostics | ≈1.1 min† | ≈0.4 min† |
| Snapshot-evidence extraction (5 snapshots, query + test) | 85 s (measured) | 29 s (measured) |
| Inference per test image (5 snapshots) | ≈10.7 ms | ≈72.5 ms |

†Estimated from measured throughput, not directly logged. BACH is slower per image because it decodes 2048×1536 TIFF files.

## Repository structure

```
completed/
├── BeyondBelief_SnapshotFusion_Benchmark.ipynb        # corrected, executed notebook (all experiments)
├── vit-snapshot_loopholes.txt      # audit of the original notebook (20 issues)
├── README.md
├── LICENSE                         # MIT licence
├── BreakHis_AllMag_split/          # seeded 70/30 split (Train/Test × benign/malignant)
├── BACH_AllClass_split/            # seeded 70/30 split (symlinks; see R7)
├── output/                         # all generated figures
└── paper/
    ├── Beyond_Belief_Leakage-Aware_Benchmark_of_Dempster-Shafer_Bayesian_Fuzzy_and_Learned_Snapshot_Fusion_for_Few-Shot_Breast_Histopathology.docx
    ├── Beyond_Belief_Leakage-Aware_Benchmark_of_Dempster-Shafer_Bayesian_Fuzzy_and_Learned_Snapshot_Fusion_for_Few-Shot_Breast_Histopathology.pdf
    └── figures/                    # paper figures (architecture, results, ROC, XAI)
```

## How to reproduce

1. Install the dependencies:
   ```bash
   pip install torch torchvision timm scikit-learn scikit-image opencv-python-headless lime matplotlib tqdm kagglehub
   ```
2. Place BreakHis (any folder containing `classificacao_binaria/`) and BACH (any folder containing `Benign/ InSitu/ Invasive/ Normal/`) within 3 directory levels of the notebook. If BreakHis is not found locally, it is downloaded through `kagglehub`.
3. Run `BeyondBelief_SnapshotFusion_Benchmark.ipynb` from top to bottom. The splits are rebuilt with seed 42 if they are missing, both experiments run, and all figures are written to `output/`.

## Citation

```bibtex
@inproceedings{khalid2026beyondbelief,
  title     = {Beyond Belief: A Leakage-Aware Benchmark of Dempster--Shafer, Bayesian, Fuzzy and
               Learned Snapshot Fusion for Few-Shot Breast Histopathology},
  author    = {Khalid, Talha and Ahmad, Ashfaq and Javed, Iqra},
  booktitle = {[Conference name -- to be added]},
  year      = {2026}
}
```

## License

The code in this repository is released under the [MIT License](LICENSE). Copyright © 2026 Talha Khalid.

- **Datasets:** BreakHis and BACH are **not** covered by this licence and are not redistributed here. Obtain them from their official sources and follow their own terms of use.
- **Base method:** the original DST-FewShot method and code belong to their authors (see Acknowledgements). This repository contains an independent, corrected re-implementation and audit.

## Disclaimer

This is research code for benchmarking few-shot classification methods. It is **not a medical device**, is not validated for clinical use, and must not be used for diagnosis or treatment decisions.

## Acknowledgements

- Base method and reference code: S. Pani, O. Brikova, D. Kaplun and R. Sarkar, [DST-FewShot](https://github.com/SaptarshiPani/DST-FewShot).
- Datasets: BreakHis (Spanhol et al.) and BACH (Aresta et al.).

## Contact

| Author | Affiliation | Email |
|---|---|---|
| Talha Khalid | Dept. of Artificial Intelligence, School of Artificial Intelligence, UMT | [tk839587@gmail.com](mailto:tk839587@gmail.com) |
| Ashfaq Ahmad | Dept. of Artificial Intelligence, School of Artificial Intelligence, UMT | [ashfaqahmad@umt.edu.pk](mailto:ashfaqahmad@umt.edu.pk) |
| Iqra Javed | Dept. of Computer Science, School of Systems and Technology, UMT | [iqra.javed@umt.edu.pk](mailto:iqra.javed@umt.edu.pk) |

For questions or issues, please open an issue in the repository or contact the corresponding author (Talha Khalid).
