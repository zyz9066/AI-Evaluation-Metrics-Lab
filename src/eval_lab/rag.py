"""Evaluate synthetic RAG outputs with deterministic checks and recorded labels."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _string_list(example: dict, key: str) -> list[str]:
    value = example.get(key)
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{key} must be a list of distinct non-empty strings")
    return value


def evaluate_rag(dataset: dict) -> dict:
    """Evaluate the documented fixture format; no model or judge is invoked."""
    for key in ("dataset_id", "model_id"):
        if not isinstance(dataset.get(key), str) or not dataset[key]:
            raise ValueError(f"{key} must be a non-empty string")
    examples = dataset.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("examples must be a non-empty list")

    results = []
    seen_ids: set[str] = set()
    retrieved_count = relevant_count = relevant_retrieved = 0
    for example in examples:
        if not isinstance(example, dict):
            raise ValueError("each example must be an object")
        for key in ("id", "question", "context", "response", "expected_account_id"):
            if not isinstance(example.get(key), str) or not example[key]:
                raise ValueError(f"{key} must be a non-empty string")
        if example["id"] in seen_ids:
            raise ValueError("example IDs must be unique")
        seen_ids.add(example["id"])
        for key in ("human_faithful", "judge_faithful"):
            if not isinstance(example.get(key), bool):
                raise ValueError(f"{key} must be a boolean")
        relevant = set(_string_list(example, "relevant_context_ids"))
        retrieved = set(_string_list(example, "retrieved_context_ids"))
        relevant_count += len(relevant)
        retrieved_count += len(retrieved)
        relevant_retrieved += len(relevant & retrieved)

        try:
            response = json.loads(example["response"])
        except json.JSONDecodeError:
            response = None
        schema_valid = (
            isinstance(response, dict)
            and isinstance(response.get("account_id"), str)
            and bool(response["account_id"])
            and isinstance(response.get("answer"), str)
            and bool(response["answer"])
        )
        id_exact = (
            isinstance(response, dict)
            and response.get("account_id") == example["expected_account_id"]
        )
        results.append(
            {
                "id": example["id"],
                "schema_valid": schema_valid,
                "account_id_exact": id_exact,
                "relevant_retrieved": len(relevant & retrieved),
                "retrieved_count": len(retrieved),
                "relevant_count": len(relevant),
                "human_faithful": example["human_faithful"],
                "judge_faithful": example["judge_faithful"],
                "judge_agrees": example["human_faithful"] == example["judge_faithful"],
            }
        )

    count = len(results)
    return {
        "task": "synthetic_rag_evaluation",
        "model_id": dataset["model_id"],
        "dataset_id": dataset["dataset_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": count,
        "metrics": {
            "schema_valid_rate": sum(row["schema_valid"] for row in results) / count,
            "account_id_exact_rate": sum(row["account_id_exact"] for row in results) / count,
            "retrieval_micro_precision": (
                relevant_retrieved / retrieved_count if retrieved_count else None
            ),
            "retrieval_micro_recall": (
                relevant_retrieved / relevant_count if relevant_count else None
            ),
            "judge_human_agreement": sum(row["judge_agrees"] for row in results) / count,
        },
        "examples": results,
        "note": "Synthetic responses and illustrative labels; no model or judge was run.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="RAG evaluation JSON")
    parser.add_argument("--output", type=Path, default=Path("reports/genai_evaluation.json"))
    parser.add_argument("--min-schema-valid-rate", type=float)
    parser.add_argument("--min-retrieval-recall", type=float)
    parser.add_argument("--min-judge-agreement", type=float)
    args = parser.parse_args()

    limits = {
        "schema_valid_rate": args.min_schema_valid_rate,
        "retrieval_micro_recall": args.min_retrieval_recall,
        "judge_human_agreement": args.min_judge_agreement,
    }
    for name, limit in limits.items():
        if limit is not None and not 0 <= limit <= 1:
            parser.error(f"minimum for {name} must be in [0, 1]")

    dataset = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be a JSON object")
    report = evaluate_rag(dataset)
    checks = {
        name: report["metrics"][name] is not None and report["metrics"][name] >= limit
        for name, limit in limits.items()
        if limit is not None
    }
    report["quality_gate"] = {
        "thresholds": {name: limit for name, limit in limits.items() if limit is not None},
        "checks": checks,
        "passed": all(checks.values()) if checks else None,
    }
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    return 2 if report["quality_gate"]["passed"] is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
