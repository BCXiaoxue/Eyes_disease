#!/usr/bin/env bash
set -euo pipefail

SERVER_ROOT="/root/wangchen/tiansukai"
PROJECT_DIR="$SERVER_ROOT/retinascope_exp"
DATA_DIR="$SERVER_ROOT/data"
IMAGE_ROOT="$SERVER_ROOT/train/images"
MERGED_DIR="$SERVER_ROOT/train/images/merged"
PRETRAINED="$PROJECT_DIR/artifacts/pretrained/swin_tiny_patch4_window7_224_ms_in1k_model.safetensors"
LOG_DIR="$PROJECT_DIR/artifacts/logs"
OUT_A="$PROJECT_DIR/artifacts/swin_ml_decoder_20260616"
OUT_B="$PROJECT_DIR/artifacts/swin_asl_tuned_20260616"

mkdir -p "$LOG_DIR" "$OUT_A" "$OUT_B"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/.deps:${PYTHONPATH:-}"

run_train() {
  local name="$1"
  local out="$2"
  shift 2
  echo "[$(date '+%F %T')] START $name"
  python3 training/train_binocular_label_graph.py \
    --train_csv "$DATA_DIR/train.csv" \
    --val_csv "$DATA_DIR/val.csv" \
    --image_root "$IMAGE_ROOT" \
    --merged_dir "$MERGED_DIR" \
    --output_dir "$out" \
    --epochs 30 \
    --batch 8 \
    --lr 5e-5 \
    --num_workers 4 \
    --pretrained_checkpoint "$PRETRAINED" \
    --enforce_server_root \
    "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
  echo "[$(date '+%F %T')] END $name"
}

run_eval() {
  local name="$1"
  local out="$2"
  local ckpt="$3"
  echo "[$(date '+%F %T')] EVAL $name"
  python3 training/evaluate_binocular_checkpoint.py \
    --val_csv "$DATA_DIR/val.csv" \
    --image_root "$IMAGE_ROOT" \
    --merged_dir "$MERGED_DIR" \
    --checkpoint "$ckpt" \
    --output "$out/eval_${name}_tta.json" \
    --batch 8 \
    --num_workers 4 \
    --tta \
    --enforce_server_root 2>&1 | tee "$LOG_DIR/${name}_tta_eval.log"
}

run_train "swin_tiny_ml_decoder_asl" "$OUT_A" \
  --preset swin_tiny_ml_decoder \
  --loss asl \
  --seed 42 \
  --asl_gamma_neg 4.0 \
  --asl_gamma_pos 1.0 \
  --asl_clip 0.05
run_eval "swin_tiny_ml_decoder_asl" "$OUT_A" "$OUT_A/best_swin_tiny_ml_decoder_asl.pth"

run_train "swin_tiny_linear_asl_tuned" "$OUT_B" \
  --preset swin_tiny_linear \
  --loss asl \
  --seed 42 \
  --asl_gamma_neg 4.0 \
  --asl_gamma_pos 0.0 \
  --asl_clip 0.05
run_eval "swin_tiny_linear_asl_tuned" "$OUT_B" "$OUT_B/best_swin_tiny_linear_asl.pth"

python3 - <<'PY'
import json
from pathlib import Path

root = Path("artifacts")
items = [
    ("baseline_effnet_b4_linear_bce", root / "side_non_convnext_model_tuning/report_effnet_b4_linear_bce.json", None),
    ("baseline_swin_tiny_linear_bce", root / "side_non_convnext_model_tuning/report_swin_tiny_linear_bce.json", root / "swin_followup_tuning/eval_swin_tiny_linear_bce_tta.json"),
    ("current_swin_tiny_linear_asl", root / "swin_followup_tuning/report_swin_tiny_linear_asl.json", root / "swin_followup_tuning/eval_swin_tiny_linear_asl_tta.json"),
    ("candidate_swin_tiny_ml_decoder_asl", root / "swin_ml_decoder_20260616/report_swin_tiny_ml_decoder_asl.json", root / "swin_ml_decoder_20260616/eval_swin_tiny_ml_decoder_asl_tta.json"),
    ("candidate_swin_tiny_linear_asl_tuned", root / "swin_asl_tuned_20260616/report_swin_tiny_linear_asl.json", root / "swin_asl_tuned_20260616/eval_swin_tiny_linear_asl_tuned_tta.json"),
]
rows = []
for name, report, eval_path in items:
    row = {"name": name}
    if report.exists():
        data = json.loads(report.read_text(encoding="utf-8"))
        best = data.get("best", {})
        row.update(
            {
                "train_macro_auc": best.get("macro_auc"),
                "train_macro_f1": best.get("macro_f1"),
                "best_epoch": best.get("epoch"),
                "loss": data.get("loss"),
                "loss_params": data.get("loss_params"),
                "config": data.get("config"),
            }
        )
    if eval_path and eval_path.exists():
        data = json.loads(eval_path.read_text(encoding="utf-8"))
        best = data.get("calibrated_threshold_metrics", {})
        row.update(
            {
                "tta_macro_auc": best.get("macro_auc"),
                "tta_macro_f1": best.get("macro_f1"),
                "tta_thresholds": best.get("thresholds"),
                "tta_per_label_f1": best.get("per_label_f1"),
            }
        )
    rows.append(row)

out = root / "summary_swin_next_improvements_20260616.json"
out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(out)
for row in rows:
    print(row["name"], "train_f1=", row.get("train_macro_f1"), "tta_f1=", row.get("tta_macro_f1"), "tta_auc=", row.get("tta_macro_auc"))
PY
