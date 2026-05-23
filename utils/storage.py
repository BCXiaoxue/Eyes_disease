import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
REPORT_DIR = ARTIFACTS_DIR / "reports"
FEEDBACK_LOG = ARTIFACTS_DIR / "feedback" / "feedback_records.jsonl"
REPORT_LOG = REPORT_DIR / "diagnosis_reports.jsonl"


def ensure_storage():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    next_id = _count_jsonl(path) + 1
    payload = {"id": next_id, **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return next_id


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def save_diagnosis_report(patient_id, age, sex, probs, preds, report_text):
    ensure_storage()
    return _append_jsonl(
        REPORT_LOG,
        {
            "patient_id": str(patient_id),
            "age": "" if age is None else str(age),
            "sex": "" if sex is None else str(sex),
            "probabilities": [float(x) for x in probs],
            "predictions": [int(x) for x in preds],
            "report_text": report_text,
            "created_at": _now_text(),
        },
    )


def save_feedback_record(patient_id, probs, preds, correct, confidence=None, comment="", flag_for_review=False):
    ensure_storage()
    return _append_jsonl(
        FEEDBACK_LOG,
        {
            "patient_id": str(patient_id),
            "probabilities": [float(x) for x in probs],
            "predictions": [int(x) for x in preds],
            "correct": bool(correct),
            "confidence": confidence,
            "comment": comment,
            "flag_for_review": bool(flag_for_review),
            "created_at": _now_text(),
        },
    )
