import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from emerguav_bench.uavscenes.indexer import UAVScenesIndexer
from scripts.build_samples import build_prediction_stub, build_seed_samples_from_manifest, save_json
from scripts.evaluate_leaderboard import evaluate_leaderboard


def make_workspace_temp_dir(prefix: str) -> Path:
    base_dir = Path.cwd() / ".test_tmp"
    base_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = base_dir / f"{prefix}_{uuid4().hex}"
    temp_dir.mkdir()
    return temp_dir


class LeaderboardReportRecomputeTests(unittest.TestCase):
    def test_evaluate_leaderboard_writes_strict_semantic_and_tables(self) -> None:
        tmp_dir = make_workspace_temp_dir("leaderboard_report_recompute")
        try:
            manifests = UAVScenesIndexer().build_manifest(sequence_ids=["SEQ_AIRPORT_01"])
            samples = build_seed_samples_from_manifest(manifests, enable_nfz_overlay=True)
            predictions = build_prediction_stub(samples)
            predictions[0]["output"]["Observation_Protocol_Assessment"][
                "protocol_context_profile"
            ] = f"{samples[0]['protocol_context']} under General Flight Operations"

            samples_path = tmp_dir / "samples.json"
            leaderboard_root = tmp_dir / "leaderboard"
            run_dir = leaderboard_root / "model_a" / "strict" / "zero_shot" / "seed_0"
            save_json(samples_path, samples)
            save_json(run_dir / "predictions.json", predictions)
            save_json(run_dir / "run_metadata.json", {"status": "passed", "runtime": "live", "precision": "bf16"})

            summary = evaluate_leaderboard(
                samples_path=samples_path,
                leaderboard_root=leaderboard_root,
                prompt_mode="zero_shot",
                seed=0,
            )

            strict_report = json.loads((run_dir / "evaluation_report.json").read_text(encoding="utf-8"))
            semantic_report = json.loads((run_dir / "semantic_evaluation_report.json").read_text(encoding="utf-8"))
            main_table = (leaderboard_root / "semantic_main_leaderboard.md").read_text(encoding="utf-8")
            task_table = (leaderboard_root / "semantic_task_breakdown.md").read_text(encoding="utf-8")

            self.assertEqual(summary["recomputed_model_count"], 1)
            self.assertEqual(strict_report["task_metrics"]["T1_accuracy"], 1.0)
            self.assertEqual(strict_report["summary_metrics"]["Protocol Context Exact Match"], 0.0)
            self.assertEqual(semantic_report["metric_profile"], "benchmark_semantic_v2")
            self.assertIn("Semantic Protocol Decision Score", semantic_report["semantic_summary_metrics"])
            self.assertIn("model_a", main_table)
            self.assertIn("Semantic Avg Task Score", main_table)
            self.assertIn("T1 Exact", task_table)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_recompute_leaderboard_tables_sort_by_semantic_protocol_decision_score(self) -> None:
        tmp_dir = make_workspace_temp_dir("leaderboard_report_sort")
        try:
            manifests = UAVScenesIndexer().build_manifest(sequence_ids=["SEQ_AIRPORT_01", "SEQ_TOWN_01"])
            samples = build_seed_samples_from_manifest(manifests, enable_nfz_overlay=True)
            good_predictions = build_prediction_stub(samples)
            weak_predictions = build_prediction_stub(samples)
            for prediction in weak_predictions:
                prediction["output"]["Observation_Protocol_Assessment"]["protocol_context_profile"] = "unknown"
                prediction["output"]["Intent_Extraction"]["constraints"] = []
                prediction["output"]["Intent_Extraction"]["target_goal"] = "unknown"
                prediction["output"]["Action_Output"]["action_token"] = "Descend_Cautiously"
                prediction["output"]["Action_Output"]["abstention_decision"] = "continue_execution"
                prediction["output"]["Action_Output"]["collaboration_request"] = {
                    "required": False,
                    "request_type": "none",
                    "trusted_modality": "rgb",
                }

            samples_path = tmp_dir / "samples.json"
            leaderboard_root = tmp_dir / "leaderboard"
            save_json(samples_path, samples)
            for model_id, predictions in (("aa_weak_model", weak_predictions), ("zz_good_model", good_predictions)):
                run_dir = leaderboard_root / model_id / "strict" / "zero_shot" / "seed_0"
                save_json(run_dir / "predictions.json", predictions)
                save_json(run_dir / "run_metadata.json", {"status": "passed", "runtime": "live", "precision": "bf16"})

            evaluate_leaderboard(
                samples_path=samples_path,
                leaderboard_root=leaderboard_root,
                prompt_mode="zero_shot",
                seed=0,
            )

            lines = (leaderboard_root / "semantic_main_leaderboard.md").read_text(encoding="utf-8").splitlines()
            first_data_row = lines[2]
            self.assertIn("zz_good_model", first_data_row)
            task_lines = (leaderboard_root / "semantic_task_breakdown.md").read_text(encoding="utf-8").splitlines()
            first_task_row = task_lines[2]
            self.assertIn("zz_good_model", first_task_row)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
