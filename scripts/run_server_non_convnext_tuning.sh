#!/usr/bin/env bash
set -euo pipefail

SERVER_ROOT="/root/wangchen/tiansukai"
PROJECT_DIR="${PROJECT_DIR:-$SERVER_ROOT/retinascope_exp}"
DATA_DIR="${DATA_DIR:-$SERVER_ROOT/data}"
IMAGE_ROOT="${IMAGE_ROOT:-$SERVER_ROOT/train/images}"
MERGED_DIR="${MERGED_DIR:-$SERVER_ROOT/train/images/merged}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/artifacts/side_non_convnext_model_tuning}"
LOG_DIR="$PROJECT_DIR/artifacts/logs"
PYTHON="${PYTHON:-python3}"

case "$(realpath "$PROJECT_DIR")" in
  "$SERVER_ROOT"/*) ;;
  *) echo "PROJECT_DIR must be under $SERVER_ROOT"; exit 2 ;;
esac

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/.deps:${PYTHONPATH:-}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

COMMON_ARGS=(
  --train_csv "$DATA_DIR/train.csv"
  --val_csv "$DATA_DIR/val.csv"
  --image_root "$IMAGE_ROOT"
  --merged_dir "$MERGED_DIR"
  --output_dir "$OUTPUT_DIR"
  --epochs "${EPOCHS:-30}"
  --num_workers "${NUM_WORKERS:-4}"
  --enforce_server_root
)

run_exp() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START $name"
  "$PYTHON" training/train_binocular_label_graph.py "${COMMON_ARGS[@]}" "$@" \
    2>&1 | tee "$LOG_DIR/${name}.log"
  echo "[$(date '+%F %T')] END $name"
}

EFFNET_CKPT="$PROJECT_DIR/artifacts/pretrained/tf_efficientnet_b4_ns_pytorch_model.bin"
EFFNET_PRETRAINED_ARGS=()
if [ -f "$EFFNET_CKPT" ]; then
  EFFNET_PRETRAINED_ARGS=(--pretrained_checkpoint "$EFFNET_CKPT")
fi

SWIN_CKPT="$PROJECT_DIR/artifacts/pretrained/swin_tiny_patch4_window7_224_ms_in1k_model.safetensors"
SWIN_PRETRAINED_ARGS=()
if [ -f "$SWIN_CKPT" ]; then
  SWIN_PRETRAINED_ARGS=(--pretrained_checkpoint "$SWIN_CKPT")
fi

run_exp "side_effnet_b4_linear_bce" \
  --preset effnet_b4_linear \
  --loss bce \
  --batch "${EFFNET_BATCH:-8}" \
  --lr "${EFFNET_LR:-1e-4}" \
  "${EFFNET_PRETRAINED_ARGS[@]}"

run_exp "side_swin_tiny_linear_bce" \
  --preset swin_tiny_linear \
  --loss bce \
  --batch "${SWIN_BATCH:-8}" \
  --lr "${SWIN_LR:-5e-5}" \
  "${SWIN_PRETRAINED_ARGS[@]}"

run_exp "side_effnet_b4_label_corr_focal_tta" \
  --preset effnet_b4_label_corr \
  --loss focal \
  --batch "${EFFNET_BATCH:-8}" \
  --lr "${EFFNET_CORR_LR:-5e-5}" \
  --tta_eval \
  "${EFFNET_PRETRAINED_ARGS[@]}"

run_exp "side_swin_tiny_label_corr_focal_tta" \
  --preset swin_tiny_label_corr \
  --loss focal \
  --batch "${SWIN_BATCH:-8}" \
  --lr "${SWIN_CORR_LR:-3e-5}" \
  --tta_eval \
  "${SWIN_PRETRAINED_ARGS[@]}"

"$PYTHON" - <<'PY'
import json
from pathlib import Path

out = Path("artifacts/side_non_convnext_model_tuning")
rows = []
for path in sorted(out.glob("report_*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    best = data.get("best", {})
    rows.append({
        "report": path.name,
        "preset": data.get("config", {}).get("preset"),
        "loss": data.get("loss"),
        "tta_eval": data.get("tta_eval"),
        "epoch": best.get("epoch"),
        "macro_auc": best.get("macro_auc"),
        "macro_f1": best.get("macro_f1"),
        "val_loss": best.get("val_loss"),
        "per_label_f1": best.get("per_label_f1"),
    })
summary = out / "summary_non_convnext_model_tuning.json"
summary.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"Summary: {summary}")
for row in sorted(rows, key=lambda x: ((x.get("macro_f1") or 0), (x.get("macro_auc") or 0)), reverse=True):
    print(row["report"], "epoch=", row["epoch"], "auc=", row["macro_auc"], "f1=", row["macro_f1"])
PY

tar -czf "$OUTPUT_DIR/non_convnext_model_tuning_results.tar.gz" \
  -C "$OUTPUT_DIR" . \
  -C "$LOG_DIR" \
  side_effnet_b4_linear_bce.log \
  side_swin_tiny_linear_bce.log \
  side_effnet_b4_label_corr_focal_tta.log \
  side_swin_tiny_label_corr_focal_tta.log

echo "Results packed at $OUTPUT_DIR/non_convnext_model_tuning_results.tar.gz"
