from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from utils.model import LABELS, _apply_thresholds, _normalise_device, get_model_fingerprint


class ModelUtilityTests(unittest.TestCase):
    def test_thresholds_enforce_normal_exclusivity(self):
        thresholds = np.full(len(LABELS), 0.5, dtype=np.float32)
        abnormal = _apply_thresholds(np.array([0.9, 0.8, 0, 0, 0, 0, 0, 0]), thresholds)
        self.assertEqual(abnormal[0], 0)
        self.assertEqual(abnormal[1], 1)
        normal = _apply_thresholds(np.zeros(len(LABELS), dtype=np.float32), thresholds)
        self.assertEqual(normal[0], 1)
        self.assertEqual(normal[1:].sum(), 0)

    def test_model_fingerprint_changes_with_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "best_swin_tiny_linear_asl.pth"
            evaluation = root / "eval_swin_tiny_linear_asl_tta.json"
            model.write_bytes(b"model-v1")
            evaluation.write_text("{}", encoding="utf-8")
            first = get_model_fingerprint(root)
            model.write_bytes(b"model-v2-longer")
            second = get_model_fingerprint(root)
            self.assertNotEqual(first, second)

    def test_requested_cuda_does_not_silently_fall_back(self):
        with patch("utils.model.torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                _normalise_device("cuda")


if __name__ == "__main__":
    unittest.main()
