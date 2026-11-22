import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from eval_lab.rag import evaluate_rag
from eval_lab.track_mlflow import main as log_to_mlflow


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "rag_examples.json"


class RagEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.dataset = json.loads(DATA.read_text(encoding="utf-8"))

    def test_fixture_metrics_and_disagreement(self):
        report = evaluate_rag(self.dataset)
        metrics = report["metrics"]
        self.assertEqual(metrics["schema_valid_rate"], 0.75)
        self.assertEqual(metrics["account_id_exact_rate"], 0.75)
        self.assertEqual(metrics["retrieval_micro_precision"], 0.8)
        self.assertEqual(metrics["retrieval_micro_recall"], 0.8)
        self.assertEqual(metrics["judge_human_agreement"], 0.75)
        self.assertFalse(report["examples"][1]["account_id_exact"])
        self.assertFalse(report["examples"][2]["judge_agrees"])
        self.assertFalse(report["examples"][3]["schema_valid"])

    def test_undefined_retrieval_and_invalid_data(self):
        dataset = copy.deepcopy(self.dataset)
        dataset["examples"] = dataset["examples"][:1]
        dataset["examples"][0]["relevant_context_ids"] = []
        dataset["examples"][0]["retrieved_context_ids"] = []
        metrics = evaluate_rag(dataset)["metrics"]
        self.assertIsNone(metrics["retrieval_micro_precision"])
        self.assertIsNone(metrics["retrieval_micro_recall"])
        dataset["examples"][0]["human_faithful"] = "true"
        with self.assertRaisesRegex(ValueError, "boolean"):
            evaluate_rag(dataset)
        dataset["examples"][0]["human_faithful"] = True
        dataset["examples"][0]["retrieved_context_ids"] = ["same", "same"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            evaluate_rag(dataset)

    def test_invalid_json_is_reported_as_failed_schema(self):
        self.dataset["examples"][0]["response"] = "{"
        result = evaluate_rag(self.dataset)["examples"][0]
        self.assertFalse(result["schema_valid"])
        self.assertFalse(result["account_id_exact"])

    def test_cli_writes_passing_and_failing_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            command = [
                sys.executable, "-m", "eval_lab.rag", "--input", str(DATA),
                "--output", str(output), "--min-retrieval-recall",
            ]
            passed = subprocess.run(command + ["0.8"], capture_output=True, text=True)
            self.assertEqual(passed.returncode, 0, passed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["quality_gate"]["passed"])
            self.assertEqual(report["quality_gate"]["thresholds"]["retrieval_micro_recall"], 0.8)
            failed = subprocess.run(command + ["0.9"], capture_output=True, text=True)
            self.assertEqual(failed.returncode, 2, failed.stderr)
            self.assertFalse(json.loads(output.read_text(encoding="utf-8"))["quality_gate"]["passed"])

    def test_undefined_recall_fails_configured_gate(self):
        self.dataset["examples"] = self.dataset["examples"][:1]
        self.dataset["examples"][0]["relevant_context_ids"] = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            output = Path(directory) / "report.json"
            path.write_text(json.dumps(self.dataset), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable, "-m", "eval_lab.rag", "--input", str(path),
                    "--output", str(output), "--min-retrieval-recall", "0",
                ],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertIsNone(report["metrics"]["retrieval_micro_recall"])
            self.assertFalse(report["quality_gate"]["passed"])

    def test_mlflow_logging_accepts_genai_report(self):
        report = evaluate_rag(self.dataset)
        report["quality_gate"] = {"passed": None}
        mlflow = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with patch.dict(sys.modules, {"mlflow": mlflow}), patch.object(
                sys, "argv", ["eval-lab-log-mlflow", "--report", str(path)]
            ):
                self.assertEqual(log_to_mlflow(), 0)
        self.assertEqual(mlflow.log_params.call_args.args[0]["task"], "synthetic_rag_evaluation")
        self.assertEqual(mlflow.log_metrics.call_args.args[0]["judge_human_agreement"], 0.75)
        mlflow.log_artifact.assert_called_once_with(str(path))


if __name__ == "__main__":
    unittest.main()
