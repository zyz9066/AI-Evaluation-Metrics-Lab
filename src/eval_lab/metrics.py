"""Small, explicit metric implementations for binary classification."""

from collections.abc import Sequence
from math import sqrt


def _validate_binary_inputs(
    y_true: Sequence[int], y_score: Sequence[float], threshold: float
) -> None:
    if not y_true:
        raise ValueError("y_true and y_score must not be empty")
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")
    if any(label not in (0, 1) for label in y_true):
        raise ValueError("y_true values must be binary 0 or 1")
    if any(not 0.0 <= score <= 1.0 for score in y_score):
        raise ValueError("y_score values must be probabilities in [0, 1]")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")


def binary_classification_metrics(
    y_true: Sequence[int], y_score: Sequence[float], threshold: float = 0.5
) -> dict[str, int | float | None | dict[str, int]]:
    """Return threshold and ranking metrics for binary labels and probabilities."""
    _validate_binary_inputs(y_true, y_score, threshold)

    tp = fp = tn = fn = 0
    for label, score in zip(y_true, y_score):
        prediction = int(score >= threshold)
        if label == 1 and prediction == 1:
            tp += 1
        elif label == 0 and prediction == 1:
            fp += 1
        elif label == 0 and prediction == 0:
            tn += 1
        else:
            fn += 1

    total = len(y_true)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall
        else 0.0
    )

    grouped: dict[float, list[int]] = {}
    for label, score in zip(y_true, y_score):
        grouped.setdefault(score, []).append(label)

    positives = sum(y_true)
    negatives = total - positives
    if positives and negatives:
        true_positives = false_positives = 0
        previous_tpr = previous_fpr = roc_auc = 0.0
        for score in sorted(grouped, reverse=True):
            labels_at_score = grouped[score]
            true_positives += sum(labels_at_score)
            false_positives += len(labels_at_score) - sum(labels_at_score)
            tpr = true_positives / positives
            fpr = false_positives / negatives
            roc_auc += (fpr - previous_fpr) * (tpr + previous_tpr) / 2
            previous_tpr, previous_fpr = tpr, fpr
    else:
        roc_auc = None

    if positives:
        previous_recall = 0.0
        average_precision = 0.0
        seen = true_positives = 0
        for score in sorted(grouped, reverse=True):
            group = grouped[score]
            true_positives += sum(group)
            seen += len(group)
            current_recall = true_positives / positives
            precision_at_cutoff = true_positives / seen
            average_precision += (current_recall - previous_recall) * precision_at_cutoff
            previous_recall = current_recall
    else:
        average_precision = None

    brier_score = sum((score - label) ** 2 for label, score in zip(y_true, y_score)) / total
    return {
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "brier_score": brier_score,
    }


def regression_metrics(
    actual: Sequence[float], predicted: Sequence[float]
) -> dict[str, float | None]:
    """Return MAE, RMSE, and R-squared, or None when R-squared is undefined."""
    if not actual:
        raise ValueError("actual and predicted must not be empty")
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    errors = [prediction - observed for observed, prediction in zip(actual, predicted)]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = sqrt(sum(error**2 for error in errors) / len(errors))
    mean_actual = sum(actual) / len(actual)
    total_sum_squares = sum((observed - mean_actual) ** 2 for observed in actual)
    residual_sum_squares = sum(error**2 for error in errors)
    r_squared = (
        1 - residual_sum_squares / total_sum_squares
        if len(actual) >= 2 and total_sum_squares
        else None
    )
    return {"mae": mae, "rmse": rmse, "r_squared": r_squared}


def normalized_edit_similarity(reference: str, prediction: str) -> float:
    """Return 1 - normalized Levenshtein distance; this is not correctness."""
    if not reference and not prediction:
        return 1.0
    if not reference or not prediction:
        return 0.0
    previous = list(range(len(prediction) + 1))
    for row, ref_char in enumerate(reference, start=1):
        current = [row]
        for column, pred_char in enumerate(prediction, start=1):
            current.append(
                min(
                    current[column - 1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref_char != pred_char),
                )
            )
        previous = current
    return 1.0 - previous[-1] / max(len(reference), len(prediction))
