from __future__ import annotations

import unittest
from pathlib import Path

from emerguav_bench.config.vocab import ACTION_TOKENS, LANGUAGE_CONDITION_TAXONOMY
from emerguav_bench.evaluation.benchmark import evaluate_benchmark
from emerguav_bench.language.entropy import (
    compute_normalized_shannon_entropy,
    generate_language_entropy_variant,
)
from emerguav_bench.schemas.output_schema import validate_model_output
from emerguav_bench.schemas.sample_schema import validate_sample_record
from scripts.build_benchmark import build_benchmark


class TestEmergUAVBench(unittest.TestCase):

    def test_language_entropy_generator(self) -> None:
        raw_instruction = "Proceed carefully along the western corridor and avoid restricted zone edge."
        for level in ("L0", "L1", "L2", "L3", "L4", "L5"):
            res = generate_language_entropy_variant(raw_instruction, level=level, seed=42)
            self.assertEqual(res["entropy_level"], level)
            self.assertTrue(len(res["perturbed_instruction"]) > 0)
            self.assertEqual(res["canonical_instruction"], raw_instruction)

    def test_normalized_shannon_entropy(self) -> None:
        counts = {"proceed": 10, "hover": 10, "reobserve": 10, "abort": 10}
        h_norm = compute_normalized_shannon_entropy(counts)
        self.assertAlmostEqual(h_norm, 1.0, places=4)

        single_count = {"proceed": 40}
        h_norm_zero = compute_normalized_shannon_entropy(single_count)
        self.assertAlmostEqual(h_norm_zero, 0.0, places=4)

    def test_output_schema_canonical_tokens(self) -> None:
        sample_output = {
            "Observation_Protocol_Assessment": {
                "protocol_context_profile": "EmergUAV Protocol",
                "primary_degradation": "dust_veil",
                "usability": "degraded_usable",
                "trusted_modality": "visual_rgb",
                "risk_level": "high",
                "risk_zone_label": "nfz_edge",
                "viewpoint_summary": "aerial_view",
                "cause_label": "airborne_visibility_loss",
            },
            "Intent_Extraction": {
                "cleaned_instruction": "Proceed carefully along the corridor.",
                "target_object": None,
                "constraints": ["avoid_nfz"],
                "target_goal": "maintain_corridor_safety",
            },
            "Action_Output": {
                "action_token": "proceed",
                "confidence": 0.95,
                "abstention_decision": "continue_execution",
                "collaboration_request": {
                    "required": False,
                    "request_type": "none",
                    "trusted_modality": "multi",
                },
            },
            "Rationale_Short": "Visual observation allows safe forward execution.",
        }
        errors = validate_model_output(sample_output)
        self.assertEqual(errors, [])

    def test_emerguav_dataset_builder_and_evaluator(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            report = build_benchmark(
                uavscenes_root="data/UAVscenes",
                output_root=tmp_dir,
                target_total_samples=12,
                seed=42,
            )
            self.assertEqual(report["sample_count"], 12)
            self.assertTrue(Path(report["output_files"]["samples"]).exists())

            from emerguav_bench.llm.io import load_json
            samples = load_json(report["output_files"]["samples"])
            predictions = load_json(report["output_files"]["predictions"])

            self.assertEqual(len(samples), 12)
            eval_report = evaluate_benchmark(samples, predictions)
            self.assertEqual(eval_report["benchmark_name"], "EmergUAV-Bench")
            self.assertIn("S_proto", eval_report)
            self.assertIn("Core_Dimensions", eval_report)
            self.assertIn("D_vis", eval_report["Core_Dimensions"])
            self.assertIn("D_lidar", eval_report["Core_Dimensions"])
            self.assertIn("D_trust", eval_report["Core_Dimensions"])
            self.assertIn("D_intent", eval_report["Core_Dimensions"])
            self.assertIn("D_constr", eval_report["Core_Dimensions"])
            self.assertIn("D_action", eval_report["Core_Dimensions"])

    def test_emerguav_orthogonality_and_facets(self) -> None:
        import tempfile
        from emerguav_bench.llm.io import load_json

        with tempfile.TemporaryDirectory() as tmp_dir:
            report = build_benchmark(
                uavscenes_root="data/UAVscenes",
                output_root=tmp_dir,
                target_total_samples=1280,
                seed=2027,
            )
            self.assertEqual(report["sample_count"], 1280)
            self.assertEqual(report["split_information"]["split_name"], "emerguav_benchmark")

            # Verify Orthogonality & Decoupled Conditions
            ortho = report["orthogonality_audit"]
            self.assertTrue(ortho["is_orthogonal"])
            self.assertFalse(ortho["condition_binding_detected"])
            self.assertLessEqual(ortho["max_cell_discrepancy_entropy_vs_pillar"], 2)

            # Check that every cell in entropy vs pillar is populated and balanced (~71)
            for level, pillar_dict in ortho["contingency_entropy_vs_pillar"].items():
                for pillar, count in pillar_dict.items():
                    self.assertGreaterEqual(count, 70, f"Cell ({level}, {pillar}) underpopulated")
                    self.assertLessEqual(count, 73, f"Cell ({level}, {pillar}) overpopulated")

            # Verify Frame Repetition Metrics
            provenance = report["frame_provenance_and_repetition"]
            self.assertEqual(provenance["total_samples"], 1280)
            self.assertGreater(provenance["unique_base_frames"], 0)
            self.assertGreater(provenance["frame_repetition_rate"], 0)
            self.assertEqual(len(report["source_sequences"]["sequence_breakdown"]), 4)

            # Verify 16 Decision Facets Coverage
            facets_audit = report["decision_facets_audit"]
            self.assertTrue(facets_audit["all_16_facets_covered"])
            self.assertEqual(facets_audit["missing_facets"], [])
            for row in facets_audit["facet_coverage_report"]:
                self.assertGreater(row["sample_count"], 0, f"Facet {row['facet_id']} has 0 samples")

            # Check individual sample facet fields
            samples = load_json(report["output_files"]["samples"])
            for s in samples[:20]:
                self.assertIn("diagnostic_facets", s)
                self.assertIn("primary_diagnostic_focus", s)
                self.assertIn("diagnostic_facet_breakdown", s)
                self.assertGreaterEqual(len(s["diagnostic_facets"]), 6)


if __name__ == "__main__":
    unittest.main()
