"""Command-line evaluation of a CSV containing labels and predicted probabilities."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from eval_lab.metrics import binary_classification_metrics


def _read_predictions(path: Path) -> tuple[list[int], list[float]]:
    labels: list[int] = []
    scores: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"y_true", "y_score"}.issubset(reader.fieldnames):
            raise ValueError("CSV must contain y_true and y_score columns")
        for row_number, row in enumerate(reader, start=2):
            try:
                labels.append(int(row["y_true"]))
                scores.append(float(row["y_score"]))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid label or score on CSV row {row_number}") from error
    return labels, scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV with y_true and y_score")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-recall", type=float)
    parser.add_argument("--max-brier", type=float)
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    args = parser.parse_args()

    if args.min_recall is not None and not 0.0 <= args.min_recall <= 1.0:
        parser.error("--min-recall must be in [0, 1]")
    if args.max_brier is not None and not 0.0 <= args.max_brier <= 1.0:
        parser.error("--max-brier must be in [0, 1]")

    labels, scores = _read_predictions(args.input)
    metrics = binary_classification_metrics(labels, scores, args.threshold)
    limits = {
        "min_recall": args.min_recall,
        "max_brier": args.max_brier,
    }
    checks: dict[str, bool] = {}
    if args.min_recall is not None:
        checks["min_recall"] = (
            metrics["recall"] is not None and metrics["recall"] >= args.min_recall
        )
    if args.max_brier is not None:
        checks["max_brier"] = metrics["brier_score"] <= args.max_brier
    gate_passed = all(checks.values()) if checks else None
    report = {
        "task": "binary_classification",
        "model_id": "provided_predictions",
        "dataset_id": args.input.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "threshold": args.threshold,
        "sample_count": len(labels),
        "metrics": metrics,
        "quality_gate": {
            "thresholds": {name: limit for name, limit in limits.items() if limit is not None},
            "checks": checks,
            "passed": gate_passed,
        },
        "note": "Demonstration data only; select thresholds on validation data.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 2 if gate_passed is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
