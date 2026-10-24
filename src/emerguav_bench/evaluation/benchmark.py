from __future__ import annotations

from typing import Any

from emerguav_bench.config.vocab import TASK_ID_TO_NAME
from emerguav_bench.taxonomy import DIAGNOSTICS
from emerguav_bench.evaluation.metrics import (
    compute_mad,
    compute_pacs,
)
from emerguav_bench.evaluation.semantic_metrics import compute_semantic_report


def evaluate_benchmark(
    samples: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute protocol, safety, and twelve-task diagnostic aggregates."""
    if len(samples) != len(predictions):
        raise ValueError(f"Sample count ({len(samples)}) and prediction count ({len(predictions)}) mismatch")
    if not samples:
        raise ValueError("Cannot evaluate an empty run; leave unmeasured paper results null")
    sample_ids = [sample.get("sample_id") for sample in samples]
    prediction_ids = [prediction.get("sample_id") for prediction in predictions]
    for name, identifiers in (("sample", sample_ids), ("prediction", prediction_ids)):
        if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
            raise ValueError(f"Each {name} must have a nonempty string sample_id")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"Duplicate {name} sample_id")
    if set(sample_ids) != set(prediction_ids):
        raise ValueError("Sample and prediction IDs must match exactly")

    # Run base semantic evaluator
    semantic_report = compute_semantic_report(samples, predictions)
    task_scores = semantic_report["semantic_task_metrics"]
    evaluated_count = int(task_scores["evaluated_samples"])
    if evaluated_count == 0:
        raise ValueError("No schema-valid predictions were evaluated; no numerical result is available")

    # Paper-facing D1-D6 are indexed projections of the unchanged base tasks.
    diagnostic_scores = {
        diagnostic["diagnostic_id"]: float(task_scores[f"{diagnostic['task_id']}_score"])
        for diagnostic in DIAGNOSTICS
    }
    # The three-term protocol metric is unchanged: T7 + T8 + T10.
    s_proto = sum(diagnostic_scores[key] for key in ("D4", "D5", "D6")) / 3.0

    # Diagnostic Safety Metrics
    strict_diagnostics = semantic_report["strict_diagnostics"]
    acc_safe = strict_diagnostics["safe_action_accuracy"]
    rate_unsafe = strict_diagnostics["unsafe_action_rate"]
    pacs = compute_pacs(samples, predictions)
    mad = compute_mad(samples, predictions)

    # Calculate Normalization Failure Count
    norm_failures = 0
    for pred in predictions:
        out = pred.get("output", {})
        if isinstance(out, dict) and out.get("_normalization_failed", False):
            norm_failures += 1

    # The base report keeps strict accuracies separately from semantic scores.
    # Average the same twelve tasks used by Semantic_Avg, with no fallback data.
    strict_avg = sum(
        strict_diagnostics[f"{task_id}_strict_accuracy"] for task_id in TASK_ID_TO_NAME
    ) / len(TASK_ID_TO_NAME)
    semantic_avg = float(task_scores["semantic_avg_task_score"])

    return {
        "benchmark_name": "EmergUAV-Bench",
        "paper": "ICASSP 2027",
        "sample_count": len(samples),
        "evaluated_sample_count": evaluated_count,
        "excluded_prediction_count": len(samples) - evaluated_count,
        "S_proto": round(s_proto, 4),
        "Semantic_Avg": round(semantic_avg, 4),
        "Acc_safe": round(acc_safe, 4),
        "Rate_unsafe": round(rate_unsafe, 4),
        "PACS": round(pacs, 4),
        "MAD": round(mad, 4),
        "Strict_Avg": round(strict_avg, 4),
        "Norm_Failures": norm_failures,
        "Diagnostic_Scores": {key: round(value, 4) for key, value in diagnostic_scores.items()},
        "Diagnostic_Index": {
            diagnostic["diagnostic_id"]: {
                key: diagnostic[key] for key in ("label", "group_id", "legacy_alias", "task_id")
            }
            for diagnostic in DIAGNOSTICS
        },
        # Existing report consumers can continue using their original keys.
        "Core_Dimensions": {
            diagnostic["legacy_alias"]: round(diagnostic_scores[diagnostic["diagnostic_id"]], 4)
            for diagnostic in DIAGNOSTICS
        },
        "Dimension_Descriptions": {
            diagnostic["legacy_alias"]: diagnostic["score"] for diagnostic in DIAGNOSTICS
        },
        "raw_semantic_report": semantic_report,
    }
