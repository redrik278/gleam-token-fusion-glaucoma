# Efficient structure-function token fusion for glaucoma staging

This repository contains the executable code and derived, non-identifiable results for the four-class GLEAM experiments reported in the manuscript. It does not redistribute GLEAM images, local clinical data, OCR files, or trained weights.

## Data

Download GLEAM from the distribution linked by the authors at <https://github.com/microewing/HAMM> and place it at `data/public/GLEAM`. The expected release contains 1,200 eye-level multimodal samples and the authors' fixed 720/240/240 train/validation/test split.

The published split is at eye level. Patient identifiers are absent from the public release, so fellow eyes cannot be grouped retrospectively and may cross partitions. Results must not be described as patient-independent evaluation.

## Environment

The completed runs used Windows, Python 3.13.15, PyTorch 2.11.0+cu128, torchvision 0.26.0, and an NVIDIA GeForce RTX 5060 8 GB. Install dependencies using `requirements.txt`. Exact operating-system and GPU details are recorded in `environment.yml`.

## Commands

Run the compact models and official-architecture comparison:

```powershell
python scripts/train_gleam_fourclass.py --models early_fusion late_fusion gated_token official_hamm --seeds 17 29 43 --epochs 12 --batch 12 --lr 1e-4
```

Run repeated component ablations:

```powershell
python scripts/train_gleam_fourclass.py --models token_no_gates token_no_transformer token_no_regularizers --seeds 17 29 43 --epochs 12 --batch 12 --lr 1e-4
```

Recompute aggregate statistics and figures:

```powershell
python scripts/analyze_gleam_fourclass.py
python scripts/analyze_repeated_ablations.py
python scripts/create_manuscript_figures.py
python scripts/create_extended_result_figures.py
```

The HAMM comparison requires cloning upstream commit `78b48100f212c85cfda47840ea82901ae82d98ba` and applying `hamm_patch/pytorch_amp_compatibility.patch`. The patch contains dtype casts needed for current automatic mixed precision; it does not change the graph-attention equations. The authors' masked-pretrained backbone was unavailable, so the local HAMM results are a compute-constrained reproduction.

## Outputs

`results/` contains per-eye test probabilities, validation histories, aggregate metrics, bootstrap comparisons, computational cost, repeated-seed ablations, the split-unit audit, and the post-hoc non-inferiority sensitivity analysis. SHA-256 checksums are provided in `SHA256SUMS.txt`.

## Archiving

Public source repository: <https://github.com/redrik278/gleam-token-fusion-glaucoma>. A permanent archival DOI requires verified author metadata, a rights-holder-approved license, Zenodo integration, and a tagged release.
