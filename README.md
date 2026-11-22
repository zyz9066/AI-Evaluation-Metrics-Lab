# AI Evaluation Metrics Lab

A practical companion to **Evaluation Metrics Across Software Engineering, AI, and MLOps**. It connects the presentation's metric-selection and release-gate ideas to two reproducible examples: binary classification and a small customer-support RAG evaluation. Everything runs without an API key or third-party packages; MLflow logging is optional.

**Scope:** All included data, responses, references, and annotations are synthetic and handwritten. No model was trained, no RAG application or LLM judge was run, and the example human/judge labels are not collected judgments. Results are teaching examples, not evidence of model performance, judge calibration, safety, or production readiness. The repository supplements, but does not replace, the required presentation.

## Run the lab (Windows PowerShell)

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m unittest discover -s tests -v

python -m eval_lab --input data\validation_predictions.csv --threshold 0.50 --min-recall 0.60 --max-brier 0.20
python -m eval_lab.rag --input data\rag_examples.json
```

The commands write `reports\evaluation.json` and `reports\genai_evaluation.json`. Each prints the report as well. Both reports record configured gate limits and checks. A configured gate returns exit code `0` when it passes and `2` when it fails; with no configured gates, `passed` is `null`, not a release approval. Invalid inputs produce an error.

**Presentation sync:** slide 28 summarizes this workflow; use the commands above to reproduce it.

## Exercise 1: classification decisions (slides 5, 22-23)

The 20-row synthetic validation CSV contains `y_true` (0 or 1) and `y_score` (probability between 0 and 1). Additional columns are ignored. The classification CLI predicts positive when `y_score >= threshold`.

Compare two thresholds on **the same validation set**:

```powershell
python -m eval_lab --input data\validation_predictions.csv --threshold 0.50 --output reports\threshold-050.json
python -m eval_lab --input data\validation_predictions.csv --threshold 0.35 --output reports\threshold-035.json
```

| Threshold | TP | FP | FN | TN | Precision | Recall | Cost at $2/FP and $25/FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 5 | 3 | 3 | 9 | 0.625 | 0.625 | $81 |
| 0.35 | 7 | 5 | 1 | 7 | 0.583 | 0.875 | $35 |

Lowering the threshold catches two additional positives but creates two more false alerts. The illustrative cost favors 0.35, assuming those costs and review capacity are appropriate; it is not a universal recommendation. ROC-AUC, average precision, and Brier score remain unchanged because the probabilities did not change.

**Metric selection:** precision = TP/(TP+FP) answers how reliable alerts are; recall = TP/(TP+FN) measures how many positives are found. Specificity = TN/(TN+FP). F1 balances precision and recall but ignores TN. Accuracy can hide rare-class failures. ROC-AUC measures ranking across thresholds; average precision summarizes the precision-recall curve using score groups and step-wise recall increments (not trapezoidal PR-AUC). Brier score is mean squared probability error: lower is better, but it is not a complete calibration diagnosis. Undefined precision, recall, specificity, ROC-AUC, or average precision is represented as `null` where applicable.

The default 0.50 and the example `--min-recall 0.60 --max-brier 0.20` limits are demonstrations, not validated release criteria. Tune the threshold and limits on representative validation data, freeze the choices, and evaluate once on an independent final test set. This tiny synthetic dataset has no independent test set; do not claim a final-test result.

## Exercise 2: RAG evaluation (slides 12-15 and 21)

`data\rag_examples.json` holds four fictional questions, contexts, response strings, expected account IDs, relevant and retrieved context IDs, and illustrative binary faithfulness labels. A response should be a JSON object with nonempty string fields `account_id` and `answer`; additional fields are allowed. The program performs these deliberately narrow checks:

| Measure | Calculation on the example set | Why it matters / limitation |
|---|---|---|
| JSON shape validity | 3/4 = 0.75 | Checks the two required field types, not a complete JSON Schema or factual correctness. |
| Account ID exact match | 3/4 = 0.75 | Compares the parsed field exactly, without normalization; an ID can match even when another field makes the response invalid. Never accept a near-match for an identifier. |
| Retrieval micro-precision | 4/5 = 0.80 | Total relevant retrieved IDs divided by total retrieved IDs, counting each example separately. Does not account for ranking. |
| Retrieval micro-recall | 4/5 = 0.80 | Total relevant retrieved IDs divided by total relevant IDs; assumes the listed relevant IDs are complete. |
| Judge-human agreement | 3/4 = 0.75 | Fraction of matching pre-recorded binary labels. This is not proof of judge quality or real human agreement. |

Read `examples` in the generated report: case 2 has a near-match but incorrect account ID; case 3 contains an unsupported date and a simulated judge disagreement; case 4 has an invalid answer type. `human_faithful` and `judge_faithful` are **provided labels**, not inferred from the answer. The suggested rubric is “every response claim, including the account identifier, is supported by the supplied context”; a non-text answer is not a faithful text answer. These labels are contrived to show agreement and disagreement, not to represent an actual review.

Try a passing gate, then a failing one:

```powershell
python -m eval_lab.rag --input data\rag_examples.json --min-retrieval-recall 0.80 --output reports\rag-pass.json
python -m eval_lab.rag --input data\rag_examples.json --min-retrieval-recall 0.90 --output reports\rag-fail.json
```

Other optional limits are `--min-schema-valid-rate` and `--min-judge-agreement`, each between 0 and 1. They are stored in the report. An undefined retrieval ratio is `null` and fails a gate that requires it. No example threshold, including those in the slides, should be treated as a production standard.

To turn this into a real evaluation, collect representative questions, trusted references and chunk IDs; run and version the actual retriever and generator; define the annotation rubric; have qualified reviewers label a sufficiently large, independent set; and compare a separate judge's per-case judgments against those labels, including false approvals and disagreements. Add checks for answer correctness, relevance, completeness, safety and operational limits. Human review remains essential, especially for consequential outcomes. Agreement alone can conceal systematic errors or class imbalance.

## MLflow tracking (slides 19, 20 and 27)

Optional installation and logging:

```powershell
python -m pip install -e ".[tracking]"
eval-lab-log-mlflow --report reports\evaluation.json
eval-lab-log-mlflow --report reports\genai_evaluation.json
mlflow ui
```

The logger records the task, model and dataset identifiers, sample count, classification threshold when present, numeric metrics, gate outcome, and the full report as an artifact. It logs **completed reports only**; it does not instrument application calls, create MLflow traces, run a judge, or automatically capture prompt, dataset, evaluator, or code versions. The full report preserves the RAG gate limits and case-level results. Configure an appropriate tracking URI for shared use, and avoid sending private data to an unapproved server. Consult [MLflow GenAI evaluation and monitoring](https://mlflow.org/docs/latest/genai/eval-monitor/) for actual tracing, evaluation, and version-specific APIs.

## Presentation-to-lab map

| Slides | Topic | What this repository provides |
|---|---|---|
| 2-4, 24-26 | Task selection, splits, uncertainty and slices | Decision workflow and limitations above; no automated split, confidence interval, or subgroup analysis. |
| 5-6, 22-23 | Classification, regression, threshold and calibration | Classification CLI and tests; `regression_metrics` provides MAE, RMSE and R² as Python functions. R² is `None` when the target is constant or only one observation is available. |
| 7-9 | Clustering, anomalies and deep learning | Discussed in the slides only; no clustering, detection, neural-model training, or latency benchmark is implemented. |
| 10-12 | String, semantic and structured outputs | `normalized_edit_similarity` demonstrates character similarity; RAG exercise checks two JSON fields and exact IDs. No embedding or semantic metric is run. |
| 13-15, 21 | RAG and evaluator comparison | The offline synthetic RAG exercise above; no real human or LLM evaluation, safety classifier, or tracing. |
| 16-18, 27 | Software, DevOps and MLOps | Unit tests, a GitHub Actions test workflow, optional report logging and illustrative CLI gates. No live DORA metrics, deployment, monitoring, or promotion pipeline. |
| 19-20, 28 | Frameworks and project architecture | Dependency-free examples and optional MLflow logging. Other frameworks are references, not dependencies. |

## Discussion prompts for the presentation

1. What failure is more costly in the classification scenario, and does the reviewer have capacity for five alerts?
2. Why is the wrong account ID in case 2 unacceptable even if edit similarity is high? Why doesn't a valid JSON shape establish correctness?
3. Case 3's judge accepts an unsupported claim. What changes to the rubric, independent labels, and failure analysis would help assess a real judge?
4. Which operational measures (p95 latency, cost, error rate), subgroup results, and safety checks would block a real release? Who would approve the limits?

## References and responsible use

- Course assignment and **AI Evaluation Metrics & Frameworks Guide**, supplied with the presentation.
- [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html), [RapidFuzz](https://rapidfuzz.github.io/RapidFuzz/), [Hugging Face Evaluate](https://huggingface.co/docs/evaluate/), [DeepEval](https://deepeval.com/docs/), [Ragas](https://docs.ragas.io/), and [MLflow](https://mlflow.org/docs/latest/genai/eval-monitor/).
- [DORA research](https://dora.dev/capabilities/) for software delivery and operational performance.

For real data, document provenance, dataset versions, split logic, error costs, evaluator and prompt versions, and relevant slices. Protect personal data and credentials. Aggregate metrics and small samples can conceal failure, drift, bias, leakage, and uncertainty.
