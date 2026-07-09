import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from utils.logger import save_feedback
from utils.storage import list_feedback_records, save_feedback_record


class StorageTests(unittest.TestCase):
    def test_feedback_jsonl_uses_unique_ids_and_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feedback.jsonl"
            first = save_feedback_record("1", [0.1, 0.9], [0, 1], False, comment="复核", path=path)
            second = save_feedback_record("1", [0.2, 0.8], [0, 1], True, path=path)
            self.assertNotEqual(first, second)
            records = list_feedback_records(path)
            self.assertEqual([record["record_id"] for record in records], [first, second])
            self.assertEqual(records[0]["comment"], "复核")

    def test_feedback_reader_skips_bad_lines_and_deduplicates_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feedback.jsonl"
            record = {"record_id": "same", "patient_id": "1"}
            path.write_text(json.dumps(record) + "\nnot-json\n" + json.dumps(record) + "\n", encoding="utf-8")
            records = list_feedback_records(path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["record_id"], "same")

    def test_save_feedback_uses_local_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feedback.jsonl"
            with patch("utils.storage.FEEDBACK_LOG", path):
                result = save_feedback("1", [0.1], [0], True, doctor_name="reviewer")
                records = list_feedback_records(path)
            self.assertEqual(result["storage_source"], "file")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["doctor_name"], "reviewer")


if __name__ == "__main__":
    unittest.main()
