from __future__ import annotations

import json

from utils.paths import FEEDBACK_DIR
from utils.storage import list_feedback_records, save_feedback_record


FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)


def save_feedback(
    img_id: str,
    probs,
    preds,
    doctor_ok: bool,
    confidence=None,
    comment="",
    flag_for_review=False,
    doctor_id=None,
    doctor_name="",
) -> dict:
    record_id = save_feedback_record(
        img_id,
        probs,
        preds,
        doctor_ok,
        confidence=confidence,
        comment=comment,
        flag_for_review=flag_for_review,
        doctor_id=doctor_id,
        doctor_name=doctor_name,
    )
    return {"record_id": record_id, "storage_source": "file", "error": ""}


def load_local_feedback() -> list[dict]:
    records = list_feedback_records()
    if records:
        return records

    legacy_records = []
    for file_path in sorted(FEEDBACK_DIR.glob("*.json")):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            data.setdefault("source", "legacy_file")
            legacy_records.append(data)
    return legacy_records
