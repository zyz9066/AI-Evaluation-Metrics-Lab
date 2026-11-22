import math
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from eval_lab.metrics import (
    binary_classification_metrics,
    normalized_edit_similarity,
    regression_metrics,
)


class ClassificationMetricsTests(unittest.TestCase):
    def test_confusion_metrics_and_ranking_metrics(self):
        result = binary_classification_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
        self.assertEqual(result["confusion_matrix"], {"tp": 1, "fp": 0, "tn": 2, "fn": 1})
        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["roc_auc"], 0.75)
        self.assertAlmostEqual(result["average_precision"], 5 / 6)

    def test_ties_receive_half_credit_for_roc_auc(self):
        result = binary_classification_metrics([0, 1], [0.5, 0.5])
        self.assertEqual(result["roc_auc"], 0.5)

    def test_rejects_invalid_probability_and_mismatched_lengths(self):
        with self.assertRaisesRegex(ValueError, "probabilities"):
            binary_classification_metrics([0], [1.1])
        with self.assertRaisesRegex(ValueError, "same length"):
            binary_classification_metrics([0, 1], [0.2])

    def test_cli_records_gate_limits_and_failure_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "predictions.csv"
            output_path = root / "report.json"
            input_path.write_text("y_true,y_score\n1,0.2\n0,0.1\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable, "-m", "eval_lab",
                    "--input", str(input_path), "--output", str(output_path),
                    "--min-recall", "0.6", "--max-brier", "0.9",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(report["quality_gate"]["thresholds"], {
                "min_recall": 0.6, "max_brier": 0.9,
            })
            self.assertEqual(report["quality_gate"]["checks"], {
                "min_recall": False, "max_brier": True,
            })
            self.assertFalse(report["quality_gate"]["passed"])


class OtherMetricTests(unittest.TestCase):
    def test_regression_error_and_fit(self):
        result = regression_metrics([1, 2, 3], [1, 2, 4])
        self.assertAlmostEqual(result["mae"], 1 / 3)
        self.assertAlmostEqual(result["rmse"], math.sqrt(1 / 3))
        self.assertAlmostEqual(result["r_squared"], 0.5)

    def test_r_squared_is_undefined_for_constant_or_single_target(self):
        self.assertIsNone(regression_metrics([2, 2], [2, 2])["r_squared"])
        self.assertIsNone(regression_metrics([2, 2], [3, 3])["r_squared"])
        self.assertIsNone(regression_metrics([2], [2])["r_squared"])

    def test_normalized_edit_similarity(self):
        self.assertAlmostEqual(normalized_edit_similarity("kitten", "sitting"), 4 / 7)
        self.assertEqual(normalized_edit_similarity("", ""), 1.0)
        self.assertEqual(normalized_edit_similarity("id", ""), 0.0)


if __name__ == "__main__":
    unittest.main()
