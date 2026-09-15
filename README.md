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

## Additional reviewer-requested analyses

The v1.1.0 analysis set adds modality-specific augmentation, a cumulative-link ordinal comparator, class-wise and macro AUPRC, stage-specific bootstrap intervals, early-versus-normal operating points, all dual/single missing-modality configurations, Grad-CAM summaries, and anatomical-region versus border perturbation.

```powershell
python scripts/train_gleam_additional.py
python scripts/analyze_additional_models.py
python scripts/analyze_clinical_robustness.py
python scripts/create_review_figures.py
```

The primary three-seed gated checkpoints are retained locally because redistributing trained weights requires confirmation that the GLEAM data terms permit derivative model-weight distribution. Per-eye probabilities and all reported aggregate outputs are public in `results/`.

## v1.2.0 reviewer-requested comparisons

This revision adds independently trained VF, VF+OCT, and VF+OCT+SLO models; learned weighted-logit fusion; attention pooling; a parameter-comparable MLP; and separate auxiliary-supervision and modality-dropout ablations. All use seeds 17, 29, and 43 and the same validation-selected checkpoint protocol.

```powershell
python scripts/train_reviewer_baselines.py
python scripts/analyze_reviewer_baselines.py
python scripts/create_reviewer_figure.py
```

`results/reviewer_baseline_comparisons.json` contains paired 5,000-resample AUC-difference intervals and separately labeled exact McNemar tests of correctness. The aggregate Bangladesh frozen-model sensitivity file contains no patient identifiers; local prediction-level files are not distributed.

## Data and complexity figures

`python scripts/create_data_complexity_figures.py` recreates the representative held-out input/output panel and performance-complexity radar chart. Radar efficiency axes use `minimum observed burden / model burden`; AUC and accuracy retain their observed 0–1 values. No aggregate radar score is calculated.

## Patient-independent public stress tests (v1.3)

The release includes a locked Harvard-GDP protocol and executable scripts for five-seed OCT, VF, late-fusion, and reliability-aware models on the official 600/400 patient split. Aggregate results and confidence intervals are under `results/external_stress_tests/`. It also includes frozen GLEAM SLO transfer evaluation on HYGD. Public source images are not redistributed; obtain Harvard-GDP and HYGD from their official repositories and comply with their licenses.
