#!/usr/bin/env bash
set -euo pipefail

cd /root/wangchen/tiansukai/retinascope_exp
export PYTHONPATH=/root/wangchen/tiansukai/retinascope_exp/.deps:${PYTHONPATH:-}

DATA_DIR=/root/wangchen/tiansukai/data
IMAGE_ROOT=/root/wangchen/tiansukai/train/images
MERGED_DIR=/root/wangchen/tiansukai/train/images/merged
OUT=/root/wangchen/tiansukai/retinascope_exp/artifacts/swin_followup_tuning
PREV=/root/wangchen/tiansukai/retinascope_exp/artifacts/side_non_convnext_model_tuning
CKPT=/root/wangchen/tiansukai/retinascope_exp/artifacts/pretrained/swin_tiny_patch4_window7_224_ms_in1k_model.safetensors

mkdir -p "$OUT" /root/wangchen/tiansukai/retinascope_exp/artifacts/logs

python3 training/evaluate_binocular_checkpoint.py \
  --val_csv "$DATA_DIR/val.csv" \
  --image_root "$IMAGE_ROOT" \
  --merged_dir "$MERGED_DIR" \
  --checkpoint "$PREV/best_swin_tiny_linear_bce.pth" \
  --output "$OUT/eval_swin_tiny_linear_bce_tta.json" \
  --batch 8 \
  --num_workers 4 \
  --tta \
  --enforce_server_root \
  2>&1 | tee /root/wangchen/tiansukai/retinascope_exp/artifacts/logs/swin_tiny_linear_bce_tta_eval.log

python3 training/train_binocular_label_graph.py \
  --train_csv "$DATA_DIR/train.csv" \
  --val_csv "$DATA_DIR/val.csv" \
  --image_root "$IMAGE_ROOT" \
  --merged_dir "$MERGED_DIR" \
  --output_dir "$OUT" \
  --epochs 30 \
  --num_workers 4 \
  --enforce_server_root \
  --preset swin_tiny_linear \
  --loss asl \
  --batch 8 \
  --lr 5e-5 \
  --pretrained_checkpoint "$CKPT" \
  2>&1 | tee /root/wangchen/tiansukai/retinascope_exp/artifacts/logs/swin_tiny_linear_asl.log
