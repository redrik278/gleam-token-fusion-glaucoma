# Locked Harvard-GDP external-benchmark protocol

Timestamp: 2026-09-16 Asia/Dhaka
Status: frozen before archive extraction or test-image inference

## Objective
Evaluate patient-independent multimodal glaucoma detection using OCT RNFL maps and 52 visual-field total-deviation values from Harvard-GDP. This is a separate external benchmark of structure-function learning and is not represented as direct four-class GLEAM validation.

## Data split
- Preserve the official 600-patient training and 400-patient test division.
- Split the 600 training patients once into 480 development and 120 validation patients using seed 20260916 and stratification by glaucoma label.
- Do not use official test images, labels, demographics, or metrics for model or threshold selection. Aggregate class counts already published in metadata may be reported.

## Models
1. OCT-only ResNet-18.
2. VF-only MLP using 52 TD values.
3. Concatenation late fusion.
4. Proposed reliability-aware fusion: modality-specific encoders and auxiliary heads, learned heteroscedastic log-variance per modality, precision-normalized logit fusion, and training-time modality dropout.

## Training
- Seeds: 17, 29, 43, 71, 101.
- ImageNet initialization for OCT encoder; AdamW; mixed precision; validation early stopping.
- Validation macro ROC AUC is the checkpoint criterion.
- No hyperparameter is changed after test evaluation.

## Primary comparison
Proposed reliability-aware fusion versus late fusion on the official 400-patient test set. Primary metric: ROC AUC. Patient bootstrap 95% CI for AUC and paired AUC difference. Secondary: AUPRC, accuracy, balanced accuracy, F1, sensitivity, specificity, PPV, NPV, Brier score, NLL, ECE, decision curves, and exact McNemar test.

## Prespecified operating point
Choose the threshold on validation data that maximizes Youden index. Also report test sensitivity at validation-selected 90% and 95% specificity.

## Missing-modality and reliability tests
Evaluate OCT only, VF only, and both modalities. Correlate predicted modality variance with branch error and image/field corruption. Report selective-prediction coverage-risk curves.

## Subgroups
Report test metrics with bootstrap intervals by sex and race where each subgroup has adequate positive and negative counts. Small groups will be descriptive and explicitly flagged.

## Acceptance criterion
Claim a discrimination advantage only if the paired 95% CI for proposed-minus-late AUC excludes zero. Otherwise frame the contribution around calibration, missing-modality robustness, reliability ranking, or efficiency only when its prespecified evidence supports that claim.
