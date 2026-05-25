# SALAD Final Project Improvements

This document summarizes the engineering and experimental changes made for the graduate deep learning final project based on the SALAD MVTec LOCO anomaly-detection codebase.

## Goal

The project reproduces SALAD-style appearance, Mahalanobis/global, and composition-branch anomaly scoring, then improves logical and structural anomaly detection on `breakfast_box`. The main final result is the K=8 composition-map model with a self-attention bottleneck:

| Method | Mean AUC | Logical AUC | Structural AUC |
| --- | ---: | ---: | ---: |
| Midterm baseline | 86.22 | n/a | n/a |
| K=8 composition maps + attention | 87.77 | 94.31 | 81.24 |
| K=8 + DINO-CC robust fusion | 87.95 | 90.88 | 85.02 |

The strongest base model improves the midterm baseline by 1.55 AUC points and gives a strong logical-anomaly result. DINO-CC fusion improves structural performance, but trades away some logical performance.

## Contributions

### 1. Spatial Attention Pooling

- Files: `train_salad.py`, `test_salad.py`
- Change: replaced raw `np.max()` anomaly-map pooling with top-percent mean pooling through `spatial_attention_pool`.
- Why: raw max pooling was unstable late in training and made final checkpoints worse than intermediate checkpoints.
- Result: reduced the observed 70k collapse and made branch scores less sensitive to single-pixel outliers.

### 2. Cosine Annealing Scheduler

- Files: `train_salad.py`
- Change: tested cosine annealing as a replacement for the original late-drop StepLR schedule.
- Why: smoother learning-rate decay was expected to stabilize training.
- Result: StepLR and cosine were similar in the current code; the scheduler did not explain the old 86.22 result. The StepLR baseline peaked at 85.61 AUC.

### 3. Optional VFM Teacher

- Files: `argparser.py`, `train_salad.py`, `test_salad.py`, `vfm_teacher.py`
- Change: added an optional DINOv2/VFM teacher path via `--use_vfm`.
- Why: a stronger semantic teacher could improve feature quality for anomaly localization.
- Result: implemented as an experimental option. It is not the main final result because the K=8 composition-map path gave the clearer gain.

### 4. AeCSAD Self-Attention Bottleneck

- Files: `ae.py`
- Change: added `SelfAttentionBlock(channels, num_heads=8)` in the composition autoencoder bottleneck and routed encoded features through attention before decoding.
- Why: logical anomalies often depend on long-range component relations, so the composition autoencoder needs non-local reasoning.
- Result: the plain K=6 attention run improved some branch-level signals but did not improve total AUC. With K=8 composition maps, the attention model reached 87.77 mean AUC and 94.31 logical AUC.

### 5. Best Checkpoint Saving

- Files: `train_salad.py`
- Change: saved `_best.pth` copies of teacher, student, autoencoder, composition autoencoder, and composition UNet whenever intermediate AUC improved.
- Why: final checkpoints were not always the best checkpoints, so evaluation needed a reliable way to recover the best observed model.
- Result: preserved the K=8 best checkpoint at 30k iterations.

### 6. Configurable Composition Class Count

- Files: `argparser.py`, `salad_dataset.py`, `train_salad.py`, `test_salad.py`, `train_composition_segmentation_model.py`
- Change: added `--composition_num_classes`, passed the class count through mask loading, composition autoencoder channels, and composition UNet channels.
- Why: the original implementation assumed 6 composition classes, but K=8 pseudo labels require 9 classes including background.
- Result: enabled training and evaluation with K=8 composition maps.

### 7. K=8 Pseudo-Label Pipeline

- Files: `create_pseudo_labels.py`, `train_composition_segmentation_model.py`, training commands
- Change: regenerated `breakfast_box` pseudo composition maps with `--n-clusters 8`, then trained the composition segmentation model with 9 output channels.
- Why: richer composition maps should separate object parts more precisely and improve logical anomaly detection.
- Result: K=8 + attention became the headline base result: 87.77 mean AUC, 94.31 logical AUC.

### 8. Branch Score Saving

- Files: `argparser.py`, `train_salad.py`, `test_salad.py`
- Change: added `--save_branch_scores` and exported per-image appearance, Mahalanobis/global, composition, equal-fusion, and weighted-fusion scores.
- Why: clean fusion experiments require matched branch scores from the same checkpoint and split.
- Result: enabled calibrated fusion and DINO-CC evaluation without retraining SALAD.

### 9. DINO-CC Memory Branch

- Files: `dino_memory_branch.py`
- Change: added component-conditioned DINO memory scoring. DINO patch features are assigned to composition-map classes, and each test patch is compared only to normal training patches from the same component class.
- Why: global DINO memory is semantically useful but too loose; class-conditional memory asks whether a patch is normal for its predicted component type.
- Result: K=6 full-memory top0.5 DINO-CC reached 88.18 AUC. On K=8, DINO-CC improved structural AUC from 81.24 to 85.02 under robust fusion, but reduced logical AUC.

### 10. LOCO Fusion

- Files: `loco_fusion.py`
- Change: added validation-normalized fusion strategies including `robust_z_sum`, `robust_z_max`, `loco_gate`, `structural_gate`, `dino_replace_sum`, and `noisy_or`.
- Why: branch scores live on different scales, and naive addition can hide branch-level improvements.
- Result: K=8 + DINO-CC `robust_z_sum` reached 87.95 mean AUC with 85.02 structural AUC.

### 11. Relation Features

- Files: `relation_features.py`, `relation_fusion.py`, `pseudo_relation_validation.py`
- Change: added geometric relation scoring from composition maps, including component count, size, position, and relation statistics.
- Why: explicit relation features are a lightweight way to model logical composition constraints.
- Result: useful as an ablation and analysis tool, but weaker than DINO-CC and K=8 composition-map training.

### 12. Test Split And Checkpoint Support

- Files: `argparser.py`, `test_salad.py`
- Change: added `--split test/validation` and `--checkpoint final/best/tmp`.
- Why: fusion and ablation experiments require matched validation-good calibration scores and reproducible best-checkpoint evaluation.
- Result: enabled clean best-checkpoint export for K=8 and DINO-CC fusion.

## Reproducible Commands

Generate K=8 pseudo labels:

```bash
conda run -n SALAD python create_pseudo_labels.py \
  --category breakfast_box \
  --n-clusters 8 \
  --save-path data/mvtec_loco_noisy_composition_maps_k8
```

Train the K=8 composition segmentation model:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n SALAD python train_composition_segmentation_model.py \
  --category breakfast_box \
  --n-clusters 8 \
  --data-path data/mvtec_loco_noisy_composition_maps_k8 \
  --log-path data/mvtec_loco_composition_maps_k8
```

Train SALAD with K=8 composition maps:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n SALAD python -u train_salad.py \
  --category breakfast_box \
  --mvtec_loco_seg_path data/mvtec_loco_composition_maps_k8 \
  --composition_num_classes 9 \
  --output_dir ~/SALAD_runs/aecsad_attn_k8_bb
```

Export best-checkpoint branch scores:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n SALAD python test_salad.py \
  --category breakfast_box \
  --output_dir ~/SALAD_runs/aecsad_attn_k8_bb \
  --mvtec_loco_seg_path data/mvtec_loco_composition_maps_k8 \
  --composition_num_classes 9 \
  --checkpoint best \
  --split test \
  --save_branch_scores
```

Run K=8 DINO-CC top0.5 scoring:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n SALAD python dino_memory_branch.py \
  --category breakfast_box \
  --seg_root data/mvtec_loco_composition_maps_k8 \
  --composition_num_classes 9 \
  --pool topk \
  --topk_percent 0.005 \
  --class_conditional \
  --feature_cache_dir evidence/dino_feature_cache_k8 \
  --score_split test \
  --output evidence/dino_k8_top0p005_test.csv
```

Fuse K=8 branch scores with DINO-CC:

```bash
python loco_fusion.py \
  --test_csv ~/SALAD_runs/aecsad_attn_k8_bb/breakfast_box/branch_scores_breakfast_box_best.csv \
  --normal_csv ~/SALAD_runs/aecsad_attn_k8_bb/breakfast_box/branch_scores_breakfast_box_validation_good_best.csv \
  --dino_csv evidence/dino_k8_top0p005_test.csv \
  --normal_dino_csv evidence/dino_k8_top0p005_val.csv
```

## Main Takeaway

The main improvement came from better composition-map granularity plus bottleneck attention, not from the scheduler. K=8 composition maps substantially improved logical anomaly detection. DINO-CC is a useful structural backup branch and ablation, but the best single headline model remains the K=8 attention checkpoint at 30k iterations.
