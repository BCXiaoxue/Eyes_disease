import json
from utils.paths import FEEDBACK_DIR
from utils.storage import save_feedback_record

FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

def save_feedback(img_id: str, probs, preds, doctor_ok: bool, confidence=None, comment="", flag_for_review=False):
    fname = FEEDBACK_DIR / f"{img_id}.json"
    payload = {
        "img_id": str(img_id),
        "probs": probs.tolist(),
        "preds": preds.tolist(),
        "correct": doctor_ok,
        "confidence": confidence,
        "comment": comment,
        "flag_for_review": flag_for_review,
    }
    fname.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_feedback_record(img_id, probs, preds, doctor_ok, confidence=confidence, comment=comment, flag_for_review=flag_for_review)
