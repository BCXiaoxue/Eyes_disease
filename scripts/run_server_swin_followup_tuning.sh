#!/usr/bin/env bash
set -euo pipefail

SERVER_ROOT="/root/wangchen/tiansukai"
PROJECT_DIR="${PROJECT_DIR:-$SERVER_ROOT/retinascope_exp}"
DATA_DIR="${DATA_DIR:-$SERVER_ROOT/data}"
IMAGE_ROOT="${IMAGE_ROOT:-$SERVER_ROOT/train/images}"
MERGED_DIR="${MERGED_DIR:-$SERVER_ROOT/train/images/merged}"
PREV_DIR="${PREV_DIR:-$PROJECT_DIR/artifacts/side_non_convnext_model_tuning}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/artifacts/swin_followup_tuning}"
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

SWIN_TINY_CKPT="$PROJECT_DIR/artifacts/pretrained/swin_tiny_patch4_window7_224_ms_in1k_model.safetensors"
SWIN_SMALL_CKPT="$PROJECT_DIR/artifacts/pretrained/swin_small_patch4_window7_224_ms_in1k.pth"

"$PYTHON" training/evaluate_binocular_checkpoint.py \
  --val_csv "$DATA_DIR/val.csv" \
  --image_root "$IMAGE_ROOT" \
  --merged_dir "$MERGED_DIR" \
  --checkpoint "$PREV_DIR/best_swin_tiny_linear_bce.pth" \
  --output "$OUTPUT_DIR/eval_swin_tiny_linear_bce_tta.json" \
  --batch "${EVAL_BATCH:-8}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --tta \
  --enforce_server_root \
  2>&1 | tee "$LOG_DIR/swin_tiny_linear_bce_tta_eval.log"

run_exp "swin_tiny_linear_asl" \
  --preset swin_tiny_linear \
  --loss asl \
  --batch "${SWIN_TINY_BATCH:-8}" \
  --lr "${SWIN_TINY_ASL_LR:-5e-5}" \
  --pretrained_checkpoint "$SWIN_TINY_CKPT"

run_exp "swin_small_linear_bce" \
  --preset swin_small_linear \
  --loss bce \
  --batch "${SWIN_SMALL_BATCH:-4}" \
  --lr "${SWIN_SMALL_LR:-3e-5}" \
  --pretrained_checkpoint "$SWIN_SMALL_CKPT"

run_exp "swin_small_linear_asl" \
  --preset swin_small_linear \
  --loss asl \
  --batch "${SWIN_SMALL_BATCH:-4}" \
  --lr "${SWIN_SMALL_ASL_LR:-3e-5}" \
  --pretrained_checkpoint "$SWIN_SMALL_CKPT"

"$PYTHON" - <<'PY'
import json
from pathlib import Path

out = Path("artifacts/swin_followup_tuning")
rows = []
for path in sorted(out.glob("report_*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    best = data.get("best", {})
    rows.append({
        "kind": "train",
        "report": path.name,
        "preset": data.get("config", {}).get("preset"),
        "loss": data.get("loss"),
        "epoch": best.get("epoch"),
        "macro_auc": best.get("macro_auc"),
        "macro_f1": best.get("macro_f1"),
        "val_loss": best.get("val_loss"),
        "per_label_f1": best.get("per_label_f1"),
    })
eval_path = out / "eval_swin_tiny_linear_bce_tta.json"
if eval_path.exists():
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    best = data["calibrated_threshold_metrics"]
    rows.append({
        "kind": "eval",
        "report": eval_path.name,
        "preset": data.get("config", {}).get("preset"),
        "loss": data.get("config", {}).get("loss"),
        "epoch": data.get("config", {}).get("metrics", {}).get("epoch"),
        "macro_auc": best.get("macro_auc"),
        "macro_f1": best.get("macro_f1"),
        "val_loss": None,
        "per_label_f1": best.get("per_label_f1"),
    })
summary = out / "summary_swin_followup_tuning.json"
summary.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"Summary: {summary}")
for row in sorted(rows, key=lambda x: ((x.get("macro_f1") or 0), (x.get("macro_auc") or 0)), reverse=True):
    print(row["report"], "auc=", row["macro_auc"], "f1=", row["macro_f1"])
PY

tar -czf "$OUTPUT_DIR/swin_followup_tuning_results.tar.gz" \
  -C "$OUTPUT_DIR" . \
  -C "$LOG_DIR" \
  swin_tiny_linear_bce_tta_eval.log \
  swin_tiny_linear_asl.log \
  swin_small_linear_bce.log \
  swin_small_linear_asl.log

echo "Results packed at $OUTPUT_DIR/swin_followup_tuning_results.tar.gz"
