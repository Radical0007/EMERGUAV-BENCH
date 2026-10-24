import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from emerguav_bench.evaluation.metrics import (
    compute_grouped_metrics,
    compute_ifr,
    compute_mad,
    compute_pacs,
    compute_summary_metrics,
    compute_task_metrics,
)
from emerguav_bench.evaluation.semantic_metrics import compute_semantic_report, compute_semantic_task_metrics
from emerguav_bench.truth.action import derive_action_truth
from emerguav_bench.truth.observation_protocol import derive_observation_protocol_truth
from emerguav_bench.truth.intent import derive_intent_truth
from emerguav_bench.uavscenes.indexer import UAVScenesIndexer
from scripts.build_samples import build_prediction_stub, build_seed_samples_from_manifest, save_json as save_seed_json
from scripts.evaluate_predictions import evaluate_predictions, load_json, save_json


def make_workspace_temp_dir(prefix: str) -> Path:
    base_dir = Path.cwd() / ".test_tmp"
    base_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = base_dir / f"{prefix}_{uuid4().hex}"
    temp_dir.mkdir()
    return temp_dir


class TruthBuilderTests(unittest.TestCase):
    def test_derive_observation_protocol_truth_raises_risk_with_overlay(self) -> None:
        truth = derive_observation_protocol_truth(
            degradation_labels=["dust_veil"],
            degradation_severity="medium",
            source_domain_bucket="airport_like",
            protocol_context="Airport Perimeter Monitoring",
            policy_family="Government / Critical Facility Sensitivity",
            enable_nfz_overlay=True,
        )
        self.assertEqual(truth["primary_degradation"], "dust_veil")
        self.assertEqual(truth["risk_level"], "high")
        self.assertEqual(truth["risk_zone_label"], "nfz_edge")

    def test_derive_intent_truth_preserves_constraints(self) -> None:
        truth = derive_intent_truth(
            clean_instruction="Proceed slowly.",
            mission_constraints=["avoid_nfz", "slow_navigation"],
            protocol_context="Smart-City Boulevard Patrol",
        )
        self.assertEqual(truth["constraints"], ["avoid_nfz", "slow_navigation"])
        self.assertTrue(truth["target_goal"])

    def test_derive_action_truth_uses_conservative_overlay_action(self) -> None:
        truth = derive_action_truth(
            source_domain_bucket="town_like",
            risk_level="high",
            trusted_modality="lidar",
            protocol_context="Smart-City Boulevard Patrol",
            enable_nfz_overlay=True,
        )
        self.assertIn(truth["action_token_gt"], {"Hover_and_Alert", "Abort_Mission", "proceed", "hover"})
        self.assertIn("Descend_Cautiously", truth["forbidden_actions"])


class EvaluationTests(unittest.TestCase):
    def _build_samples_and_predictions(self):
        manifests = UAVScenesIndexer().build_manifest(sequence_ids=["SEQ_AIRPORT_01", "SEQ_TOWN_01"])
        samples = build_seed_samples_from_manifest(manifests, enable_nfz_overlay=False)
        predictions = build_prediction_stub(samples)
        for prediction in predictions:
            prediction["output"]["Action_Output"]["confidence"] = 0.9
            prediction["output"]["Rationale_Short"] = "placeholder rationale"
        return samples, predictions

    def test_compute_task_metrics_on_perfect_predictions(self) -> None:
        samples, predictions = self._build_samples_and_predictions()
        metrics = compute_task_metrics(samples, predictions)
        self.assertEqual(metrics["degradation_accuracy"], 1.0)
        self.assertEqual(metrics["action_exact_match"], 1.0)
        self.assertEqual(metrics["safe_action_accuracy"], 1.0)
        self.assertEqual(metrics["T1_accuracy"], 1.0)
        self.assertEqual(metrics["T12_accuracy"], 1.0)
        self.assertEqual(metrics["protocol_decision_accuracy"], 1.0)
        self.assertEqual(metrics["decision_task_accuracy"], 1.0)

    def test_protocol_decision_accuracy_excludes_task_context_identification(self) -> None:
        samples, predictions = self._build_samples_and_predictions()
        for prediction in predictions:
            prediction["output"]["Observation_Protocol_Assessment"]["protocol_context_profile"] = "wrong protocol context"

        metrics = compute_task_metrics(samples, predictions)

        self.assertEqual(metrics["T1_accuracy"], 0.0)
        self.assertEqual(metrics["protocol_decision_accuracy"], 1.0)
        self.assertEqual(metrics["decision_task_accuracy"], 1.0)

    def test_t1_recognition_accepts_compound_profile_and_keeps_exact_diagnostic(self) -> None:
        samples, predictions = self._build_samples_and_predictions()
        for sample, prediction in zip(samples, predictions):
            prediction["output"]["Observation_Protocol_Assessment"][
                "protocol_context_profile"
            ] = f"{sample['protocol_context']} under General Flight Operations policy"

        task_metrics = compute_task_metrics(samples, predictions)
        summary_metrics = compute_summary_metrics(samples, predictions)

        self.assertEqual(task_metrics["T1_accuracy"], 1.0)
        self.assertEqual(task_metrics["protocol_context_exact_accuracy"], 0.0)
        self.assertEqual(task_metrics["protocol_context_inclusion_accuracy"], 1.0)
        self.assertEqual(summary_metrics["Protocol Context Exact Match"], 0.0)
        self.assertEqual(summary_metrics["Protocol Context Inclusion"], 1.0)
        self.assertEqual(task_metrics["protocol_decision_accuracy"], 1.0)

    def test_compute_mad_and_pacs_on_perfect_predictions(self) -> None:
        samples, predictions = self._build_samples_and_predictions()
        self.assertEqual(compute_mad(samples, predictions), 0.0)
        self.assertEqual(compute_pacs(samples, predictions), 1.0)
        self.assertEqual(compute_ifr(1.0, 1.0), 1.0)

    def test_compute_summary_and_grouped_metrics_on_perfect_predictions(self) -> None:
        manifests = UAVScenesIndexer().build_manifest(sequence_ids=["SEQ_TOWN_01"])
        samples = build_seed_samples_from_manifest(
            manifests,
            protocol_context="Sensitive-Site Respect",
            policy_family="Worship-Site Respect",
            language_entropy_labels=["code_switching"],
        )
        predictions = build_prediction_stub(samples)

        summary = compute_summary_metrics(samples, predictions)
        grouped = compute_grouped_metrics(samples, predictions)

        self.assertEqual(summary["MMTR"], 0.0)
        self.assertIn("Sensitive-Site Respect", grouped["by_protocol_context"])

    def test_evaluate_predictions_returns_report(self) -> None:
        samples, predictions = self._build_samples_and_predictions()
        report = evaluate_predictions(samples, predictions)
        self.assertIn("task_metrics", report)
        self.assertIn("summary_metrics", report)
        self.assertEqual(report["summary_metrics"]["MAD"], 0.0)
        self.assertIn("grouped_results", report)


if __name__ == "__main__":
    unittest.main()
