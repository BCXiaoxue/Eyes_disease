from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any, Dict, List
import uuid


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
FEEDBACK_LOG = ARTIFACTS_DIR / "feedback" / "feedback_records.jsonl"
_WRITE_LOCK = threading.Lock()


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    record_id = str(payload.get("record_id") or uuid.uuid4())
    record = {"record_id": record_id, **payload}
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    return record_id


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
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                records.append(item)
    return records


def save_feedback_record(
    patient_id,
    probs,
    preds,
    correct,
    *,
    confidence=None,
    comment="",
    flag_for_review=False,
    doctor_id=None,
    doctor_name="",
    created_at="",
    path: Path | None = None,
) -> str:
    from datetime import datetime

    target = path or FEEDBACK_LOG
    return _append_jsonl(
        target,
        {
            "img_id": str(patient_id),
            "patient_id": str(patient_id),
            "probs": [float(x) for x in probs],
            "preds": [int(x) for x in preds],
            "correct": bool(correct),
            "confidence": confidence,
            "comment": str(comment or ""),
            "flag_for_review": bool(flag_for_review),
            "doctor_id": doctor_id,
            "doctor_name": str(doctor_name or ""),
            "created_at": created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "file",
        },
    )


def list_feedback_records(path: Path | None = None) -> List[Dict[str, Any]]:
    records = _read_jsonl(path or FEEDBACK_LOG)
    seen = set()
    deduplicated = []
    for record in reversed(records):
        record_id = str(record.get("record_id") or "")
        if record_id and record_id in seen:
            continue
        if record_id:
            seen.add(record_id)
        deduplicated.append(record)
    return list(reversed(deduplicated))
