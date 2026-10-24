"""Regression checks for benchmark diagnostics and sample accounting."""
import copy
import unittest

from emerguav_bench.evaluation.benchmark import evaluate_benchmark
from emerguav_bench.uavscenes.indexer import UAVScenesIndexer
from scripts.build_samples import build_prediction_stub, build_seed_samples_from_manifest


class EmergUAVEvaluationReportingTests(unittest.TestCase):
    def setUp(self):
        manifests = UAVScenesIndexer().build_manifest(sequence_ids=["SEQ_AIRPORT_01"])
        self.samples = build_seed_samples_from_manifest(manifests, enable_nfz_overlay=False)
        self.predictions = build_prediction_stub(self.samples)

    def test_public_indices_preserve_legacy_scores_and_protocol_formula(self):
        report = evaluate_benchmark(self.samples, self.predictions)
        aliases = ("D_vis", "D_lidar", "D_trust", "D_intent", "D_constr", "D_action")
        task_ids = ("T4", "T3", "T5", "T7", "T8", "T10")
        self.assertEqual(list(report["Diagnostic_Scores"]), [f"D{i}" for i in range(1, 7)])
        raw_scores = report["raw_semantic_report"]["semantic_task_metrics"]
        for index, (alias, task_id) in enumerate(zip(aliases, task_ids), 1):
            diagnostic_id = f"D{index}"
            self.assertEqual(report["Diagnostic_Scores"][diagnostic_id], report["Core_Dimensions"][alias])
            self.assertEqual(report["Diagnostic_Scores"][diagnostic_id], round(raw_scores[f"{task_id}_score"], 4))
            self.assertEqual(report["Diagnostic_Index"][diagnostic_id]["task_id"], task_id)
        expected = sum(raw_scores[f"{task}_score"] for task in ("T7", "T8", "T10")) / 3
        self.assertEqual(report["S_proto"], round(expected, 4))

    def test_strict_average_tracks_actual_predictions(self):
        perfect = evaluate_benchmark(self.samples, self.predictions)
        self.assertEqual(perfect["Strict_Avg"], 1.0)
        for prediction in self.predictions:
            prediction["output"]["Intent_Extraction"]["cleaned_instruction"] = "wrong instruction"
        changed = evaluate_benchmark(self.samples, self.predictions)
        self.assertAlmostEqual(changed["Strict_Avg"], 11 / 12, places=4)
        self.assertLess(changed["S_proto"], perfect["S_proto"])

    def test_empty_run_is_not_a_zero_result(self):
        with self.assertRaisesRegex(ValueError, "empty run"):
            evaluate_benchmark([], [])

    def test_unknown_ids_do_not_silently_reduce_denominator(self):
        self.predictions[0]["sample_id"] = "unmatched"
        with self.assertRaisesRegex(ValueError, "IDs must match"):
            evaluate_benchmark(self.samples, self.predictions)

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluate_benchmark(self.samples * 2, self.predictions * 2)

    def test_no_valid_predictions_are_not_a_zero_result(self):
        predictions = [{"sample_id": s["sample_id"], "output": {}} for s in self.samples]
        with self.assertRaisesRegex(ValueError, "No schema-valid"):
            evaluate_benchmark(self.samples, predictions)

    def test_excluded_predictions_remain_visible(self):
        samples = [copy.deepcopy(self.samples[0]) for _ in range(2)]
        samples[1]["sample_id"] += "_second"
        predictions = build_prediction_stub(samples)
        predictions[1]["output"] = {}
        report = evaluate_benchmark(samples, predictions)
        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(report["evaluated_sample_count"], 1)
        self.assertEqual(report["excluded_prediction_count"], 1)


if __name__ == "__main__":
    unittest.main()
