#!/usr/bin/env bash
set -euo pipefail

SERVER_ROOT="/root/wangchen/tiansukai"
PROJECT_DIR="${PROJECT_DIR:-$SERVER_ROOT/retinascope_exp}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
IMAGE_ROOT="${IMAGE_ROOT:-$PROJECT_DIR/train/images}"
MERGED_DIR="${MERGED_DIR:-$PROJECT_DIR/train/images/merged}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/artifacts/binocular_label_graph}"
PYTHON="${PYTHON:-python3}"

case "$(realpath "$PROJECT_DIR")" in
  "$SERVER_ROOT"/*) ;;
  *) echo "PROJECT_DIR must be under $SERVER_ROOT"; exit 2 ;;
esac

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR" "$PROJECT_DIR/artifacts/logs"

"$PYTHON" training/check.py

COMMON_ARGS=(
  --train_csv "$DATA_DIR/train.csv"
  --val_csv "$DATA_DIR/val.csv"
  --image_root "$IMAGE_ROOT"
  --merged_dir "$MERGED_DIR"
  --output_dir "$OUTPUT_DIR"
  --epochs "${EPOCHS:-20}"
  --batch "${BATCH:-16}"
  --num_workers "${NUM_WORKERS:-4}"
  --enforce_server_root
)

if [ "${NO_PRETRAINED:-0}" = "1" ]; then
  COMMON_ARGS+=(--no_pretrained)
fi

"$PYTHON" training/train_binocular_label_graph.py \
  "${COMMON_ARGS[@]}" \
  --preset effnet_b4_linear \
  --loss bce \
  2>&1 | tee "$PROJECT_DIR/artifacts/logs/effnet_b4_linear.log"

"$PYTHON" training/train_binocular_label_graph.py \
  "${COMMON_ARGS[@]}" \
  --preset convnext_tiny_linear \
  --loss bce \
  2>&1 | tee "$PROJECT_DIR/artifacts/logs/convnext_tiny_linear.log"

"$PYTHON" training/train_binocular_label_graph.py \
  "${COMMON_ARGS[@]}" \
  --preset dual_branch_label_gcn_asl \
  --loss asl \
  2>&1 | tee "$PROJECT_DIR/artifacts/logs/dual_branch_label_gcn_asl.log"

tar -czf "$OUTPUT_DIR/retinascope_binocular_results.tar.gz" \
  -C "$OUTPUT_DIR" . \
  -C "$PROJECT_DIR/artifacts/logs" \
  effnet_b4_linear.log convnext_tiny_linear.log dual_branch_label_gcn_asl.log

echo "Results packed at $OUTPUT_DIR/retinascope_binocular_results.tar.gz"
