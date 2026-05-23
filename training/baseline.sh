#!/usr/bin/env bash
# train_all_labels.sh
CSV_PATH="data/train.csv"
IMG_DIR="train/images/merged"
LABELS=(N D G C A H M O)

for i in "${!LABELS[@]}"; do
  GPU=$i               # GPU 0‑7
  LABEL=${LABELS[$i]}
  echo ">>> Launching $LABEL on GPU $GPU"
  CUDA_VISIBLE_DEVICES=$GPU \
  nohup python training/baseline.py \
        --csv "$CSV_PATH" \
        --img_dir "$IMG_DIR" \
        --label "$LABEL" \
        --epochs 20 \
        --folds 5 \
        --batch 32 \
        --lr 1e-4 \
        --lr_min 1e-6 \
        --seed 42  \
        > "artifacts/logs/log_${LABEL}.txt" 2>&1 &
done

wait   # block until all 8 jobs finish
echo "All trainings complete."
