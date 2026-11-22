"""Log a previously generated JSON evaluation report to MLflow."""

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment", default="ai-evaluation-metrics-lab")
    args = parser.parse_args()

    import mlflow

    report = json.loads(args.report.read_text(encoding="utf-8"))
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run():
        params = {
            "task": report["task"],
            "model_id": report["model_id"],
            "dataset_id": report["dataset_id"],
            "sample_count": report["sample_count"],
        }
        if "threshold" in report:
            params["threshold"] = report["threshold"]
        mlflow.log_params(params)
        mlflow.log_metrics(
            {
                name: value
                for name, value in report["metrics"].items()
                if isinstance(value, (int, float)) and value is not None
            }
        )
        mlflow.log_param(
            "quality_gate_passed", str(report["quality_gate"]["passed"])
        )
        mlflow.log_artifact(str(args.report))
    print(f"Logged evaluation report to MLflow experiment: {args.experiment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
