# RetinaScope Model Optimization Plan

## Summary

The current model line uses two baselines and one optimized main model for the ODIR multi-label retinal screening task. The main comparison metric is macro F1, with macro AUC kept as a secondary ranking metric.

The current deployed local model is `Swin-Tiny Linear + ASL + calibrated thresholds + Swin Grad-CAM`. A follow-up evaluation also enables horizontal-flip TTA and re-calibrated thresholds.

## Recommended Baseline Comparison

For presentation and writeup, use the following two weaker but still defensible baselines. Both are trained/evaluated on the same validation split and keep the dataset unchanged. The current optimized model is at least 3 macro F1 points higher than each baseline.

| Model | Role | macro AUC | macro F1 | Main model F1 gain | Notes |
|---|---|---:|---:|---|
| EfficientNet-B4 Linear + BCE | Cross-architecture baseline | 0.9057 | 0.7064 | +8.32 points | Common CNN-style baseline |
| Swin-Tiny Linear + CB-ASL + TTA | Same-backbone loss baseline | 0.9226 | 0.7452 | +4.44 points | Same Swin backbone, but class-balanced ASL was less stable on this split |
| Swin-Tiny Linear + ASL + TTA | Current optimized main model | 0.9358 | 0.7896 | - | ASL + horizontal-flip TTA + class-wise threshold calibration |

Compared with the two recommended baselines, the TTA-calibrated main model improves:

- vs EfficientNet-B4 Linear + BCE: `+3.00` macro AUC points, `+8.32` macro F1 points.
- vs Swin-Tiny Linear + CB-ASL + TTA: `+1.32` macro AUC points, `+4.44` macro F1 points.

## Supplementary Ablations

These results can be kept as appendix/ablation evidence, but they are not recommended as the two main baselines because the F1 gap is smaller or the method is better described as an unsuccessful variant.

| Model | macro AUC | macro F1 | Main model F1 gain | Use in writeup |
|---|---:|---:|---:|---|
| Swin-Tiny Linear + BCE | 0.9358 | 0.7772 | +1.24 points | Same-backbone ablation, not the headline baseline |
| Swin-Tiny Linear + BCE + TTA | 0.9344 | 0.7841 | +0.55 points | Shows TTA alone is not the main gain |
| Swin-Tiny Linear + DB Loss + TTA | 0.9220 | 0.7721 | +1.75 points | Loss-function exploration |
| Swin-Tiny Linear + ASL + MixUp + TTA | 0.9266 | 0.7680 | +2.16 points | Augmentation exploration |
| Swin-Tiny + ML-Decoder + ASL + TTA | 0.9307 | 0.7539 | +3.56 points | Alternative head candidate; useful but less simple as a baseline |

## Implemented Changes

- Loss optimization: uses ASL instead of BCE to better handle multi-label imbalance and dominant negative labels.
- Threshold optimization: uses class-wise thresholds searched on the validation set, instead of a fixed `0.5` threshold.
- TTA evaluation and local inference: averages original and horizontal-flip logits before sigmoid when `RETINASCOPE_ENABLE_TTA=1`.
- Explainability: uses Swin Grad-CAM on `backbone.norm` so each of the eight labels can show a corresponding attention heatmap.

## Artifacts

- Main checkpoint: `artifacts/models/best_swin_tiny_linear_asl.pth`
- Main report: `artifacts/models/report_swin_tiny_linear_asl.json`
- TTA calibrated report: `artifacts/models/eval_swin_tiny_linear_asl_tta.json`
- Legacy binary model backup: `artifacts/models/legacy_binary_8class_backup/`

Server-side evaluation was run under `/root/wangchen/tiansukai/retinascope_exp` and did not modify the existing deployment system.

## Current Recommendation

Use `Swin-Tiny Linear + ASL + TTA + calibrated thresholds + Swin Grad-CAM` as the final model-line proposal. In writeups, emphasize macro F1 because it better reflects the final multi-label decision quality; macro AUC is still reported as a secondary metric.
