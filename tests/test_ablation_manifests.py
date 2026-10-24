from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from emerguav_bench.llm.ablations import (
    ABLATION_CONDITIONS,
    build_ablation_manifest,
    build_protocol_counterfactual_manifest,
    prompt_leakage_audit,
    select_stratified_stage1_records,
    source_anchor_id,
    summarize_stage1_coverage,
)
from emerguav_bench.llm.adapters.base import _apply_chat_template_with_prefill
from emerguav_bench.llm.exporters import build_llm_input_manifest
from emerguav_bench.llm.prompts import build_zero_shot_messages
from emerguav_bench.llm.types import ensure_llm_input_record, validate_llm_input_record
from scripts.build_ablation_manifests import build_and_write_manifests
from scripts import run_model_suite as suite_script


def make_sample(*, sample_id: str, sequence_id: str, frame_id: str, rgb_suffix: str, policy_family: str) -> dict:
    return {
        "sample_id": sample_id,
        "source_domain_bucket": "airport_like",
        "protocol_context": "Airport Perimeter Monitoring",
        "application_theme": "Hyper-Dense Urban Airspace",
        "policy_family": policy_family,
        "sequence_id": sequence_id,
        "frame_id": frame_id,
        "rgb_frame_path": f"placeholder://rgb/{sequence_id}/{rgb_suffix}.jpg",
        "lidar_frame_path": f"placeholder://lidar/{sequence_id}/{rgb_suffix}.bin",
        "pose_6dof": {"position": {"x": 1.0, "y": 2.0, "z": 3.0}, "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}},
        "map_id": f"map::{sequence_id}",
        "degradation_labels": ["dust_veil"],
        "degradation_severity": "high",
        "observation_severity_code": "SV4",
        "language_entropy_labels": ["typo_omission_abbreviation"],
        "language_entropy_level": "EL3",
        "clean_instruction": "Avoid the restricted boundary.",
        "noisy_instruction": "avd restrictd bndry",
        "mission_constraints": ["avoid_nfz"],
        "observation_protocol_gt": {
            "protocol_context_profile": "Airport Perimeter Monitoring",
            "primary_degradation": "dust_veil",
            "usability": "low",
            "trusted_modality": "lidar",
            "risk_level": "high",
            "risk_zone_label": "nfz_edge",
            "viewpoint_summary": "restricted_viewpoint",
            "cause_label": "airborne_visibility_loss",
        },
        "intent_extraction_gt": {
            "cleaned_instruction": "avoid the restricted boundary",
            "target_object": None,
            "constraints": ["avoid_nfz"],
            "target_goal": "monitor_safely",
        },
        "action_token_gt": "Hover_and_Alert",
        "acceptable_safe_alternatives": ["Hover", "Reobserve"],
        "forbidden_actions": ["Descend_Cautiously"],
        "risk_level_gt": "high",
        "collaboration_gt": {"required": True, "request_type": "traffic_check", "trusted_modality": "lidar"},
        "paper_task_gt": {
            "T1": "Airport Perimeter Monitoring",
            "T2": "nfz_edge",
            "T3": "restricted_viewpoint",
            "T4": "dust_veil",
            "T5": "lidar",
            "T6": {"usability": "low", "cause_label": "airborne_visibility_loss"},
            "T7": "avoid the restricted boundary",
            "T8": ["avoid_nfz"],
            "T9": "monitor_safely",
            "T10": "Hover_and_Alert",
            "T11": "hold_position",
            "T12": {"required": True, "request_type": "traffic_check", "trusted_modality": "lidar"},
        },
        "geometry_anchor": {"waypoint": [1.0, 2.0, 3.0]},
        "protocol_metadata": {"scene_family": "HKairport", "overlay_profiles": ["nfz_edge_caution"]},
    }


def make_records() -> list[dict]:
    samples = []
    protocols = [
        ("Airport Perimeter Monitoring", "Government / Critical Facility Sensitivity", "Hover_and_Alert"),
        ("Sensitive-Site Respect", "Privacy-Sensitive Institutions", "Reobserve"),
    ]
    for index, (protocol, policy, action) in enumerate(protocols):
        sample = make_sample(
            sample_id=f"SEQ_A::0001::variant_{index}",
            sequence_id="SEQ_A",
            frame_id=f"0001__variant_{index}",
            rgb_suffix=f"0001_{index}",
            policy_family=policy,
        )
        sample["protocol_context"] = protocol
        sample["action_token_gt"] = action
        sample["paper_task_gt"]["T1"] = protocol
        sample["paper_task_gt"]["T10"] = action
        sample["observation_protocol_gt"]["protocol_context_profile"] = protocol
        sample["intent_extraction_gt"]["constraints"] = [f"constraint_{index}"]
        sample["mission_constraints"] = [f"constraint_{index}"]
        samples.append(sample)
    return build_llm_input_manifest(samples, context_frames=1, min_image_inputs=2)


class AblationManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = make_records()

    def test_source_anchor_id_drops_only_variant_suffix(self) -> None:
        self.assertEqual(source_anchor_id("SEQ_A::0001::dust_variant"), "SEQ_A::0001")

    def test_named_conditions_have_exact_input_contracts(self) -> None:
        baseline = build_ablation_manifest(self.records, condition="baseline")
        text_only = build_ablation_manifest(self.records, condition="text_only")
        image_only = build_ablation_manifest(self.records, condition="image_only")
        primary_only = build_ablation_manifest(self.records, condition="primary_rgb_only")

        self.assertEqual(set(ABLATION_CONDITIONS), {"baseline", "text_only", "image_only", "primary_rgb_only"})
        self.assertEqual(len(baseline[0]["image_inputs"]), len(self.records[0]["image_inputs"]))
        self.assertEqual(text_only[0]["image_inputs"], [])
        self.assertEqual([item["role"] for item in primary_only[0]["image_inputs"]], ["primary_frame"])

        image_prompt = "\n".join(message["content"] for message in image_only[0]["prompt_messages"])
        for secret in (
            self.records[0]["metadata"]["protocol_context"],
            self.records[0]["metadata"]["policy_family"],
            self.records[0]["metadata"]["instruction_text"],
        ):
            self.assertNotIn(secret, image_prompt)
        self.assertNotIn("sample_record", image_only[0]["metadata"])
        self.assertEqual(image_only[0]["sft_targets"], [])
        self.assertEqual(image_only[0]["preference_pairs"], [])

    def test_text_only_record_is_valid_and_keeps_custom_prompt(self) -> None:
        row = build_ablation_manifest(self.records, condition="text_only")[0]
        record = ensure_llm_input_record(row)
        self.assertEqual(validate_llm_input_record(record), [])
        messages = build_zero_shot_messages(record)
        self.assertIn("No images are supplied", messages[0]["content"])
        self.assertEqual(messages, row["prompt_messages"])

    def test_prompt_leakage_audit_detects_annotation_field_name(self) -> None:
        rows = build_ablation_manifest(self.records, condition="baseline")
        clean_report = prompt_leakage_audit(rows)
        self.assertTrue(clean_report["passed"])
        leaked = copy.deepcopy(rows)
        leaked[0]["prompt_messages"][1]["content"] += "\ndegradation_labels: dust_veil"
        report = prompt_leakage_audit(leaked)
        self.assertFalse(report["passed"])
        self.assertEqual(report["violations"][0]["sample_id"], leaked[0]["sample_id"])

    def test_counterfactual_manifest_holds_pixels_fixed_and_changes_protocol(self) -> None:
        manifest, pairs = build_protocol_counterfactual_manifest(self.records)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(len(manifest), 2)
        self.assertNotEqual(manifest[0]["sample_id"], manifest[1]["sample_id"])
        self.assertEqual(manifest[0]["image_inputs"], manifest[1]["image_inputs"])
        self.assertNotEqual(
            manifest[0]["metadata"]["protocol_context"],
            manifest[1]["metadata"]["protocol_context"],
        )
        self.assertEqual(pairs[0]["source_anchor_id"], "SEQ_A::0001")
        self.assertEqual(pairs[0]["visual_source_sample_id"], manifest[0]["sample_id"])

    def test_manifest_build_is_deterministic(self) -> None:
        first = build_ablation_manifest(self.records, condition="image_only")
        second = build_ablation_manifest(self.records, condition="image_only")
        self.assertEqual(
            json.dumps(first, ensure_ascii=False, sort_keys=True),
            json.dumps(second, ensure_ascii=False, sort_keys=True),
        )

    def test_stage1_selector_is_deterministic_and_uses_distinct_anchors(self) -> None:
        records = []
        for index in range(4):
            row = copy.deepcopy(self.records[index % 2])
            sample = row["metadata"]["sample_record"]
            row["sample_id"] = f"SEQ_{index}::0001::variant"
            row["sequence_id"] = f"SEQ_{index}"
            sample["sample_id"] = row["sample_id"]
            sample["sequence_id"] = row["sequence_id"]
            records.append(row)

        first = select_stratified_stage1_records(records, minimum_per_stratum=1)
        second = select_stratified_stage1_records(records, minimum_per_stratum=1)
        summary = summarize_stage1_coverage(first)

        self.assertEqual([row["sample_id"] for row in first], [row["sample_id"] for row in second])
        self.assertEqual(summary["record_count"], summary["source_anchor_count"])
        self.assertEqual(set(summary["strata"]["protocol"]), {"Airport Perimeter Monitoring", "Sensitive-Site Respect"})
        self.assertTrue(all(count >= 1 for counts in summary["strata"].values() for count in counts.values()))

    def test_stage1_selector_fails_when_distinct_anchors_cannot_cover_strata(self) -> None:
        with self.assertRaisesRegex(ValueError, "distinct source anchors"):
            select_stratified_stage1_records(self.records, minimum_per_stratum=1)

    def test_builder_exports_evaluator_samples_separately_from_model_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source_manifest.json"
            source.write_text(json.dumps(self.records), encoding="utf-8")
            summary = build_and_write_manifests(
                source_manifest_path=source,
                output_root=root / "ablations",
                conditions=["image_only"],
            )
            evaluation_samples = json.loads(Path(summary["evaluation_samples"]).read_text(encoding="utf-8"))
            image_only = json.loads(
                Path(summary["conditions"]["image_only"]["manifest"]).read_text(encoding="utf-8")
            )

        self.assertEqual([sample["sample_id"] for sample in evaluation_samples], [row["sample_id"] for row in self.records])
        self.assertEqual(summary["evaluation_sample_count"], len(self.records))
        self.assertNotIn("sample_record", image_only[0]["metadata"])
        self.assertEqual(evaluation_samples[0]["protocol_context"], self.records[0]["metadata"]["sample_record"]["protocol_context"])

    def test_text_only_processor_call_omits_images_argument(self) -> None:
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                return "rendered"

            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return {"input_ids": [[1]]}

        processor = Processor()
        inputs, prefilled = _apply_chat_template_with_prefill(
            processor,
            [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            images=[],
            fallback_text="hello",
        )
        self.assertTrue(prefilled)
        self.assertEqual(inputs, {"input_ids": [[1]]})
        self.assertNotIn("images", processor.kwargs)

    def test_suite_cli_returns_failure_when_any_model_fails(self) -> None:
        summary = {
            "suite": "core",
            "passed_model_count": 1,
            "skipped_model_count": 0,
            "failed_model_count": 1,
            "runs": [],
        }
        argv = ["run_model_suite.py", "--manifest", "manifest.json", "--results-root", "results"]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(suite_script, "run_model_suite", return_value=summary),
            mock.patch.object(suite_script, "format_failure_summary", return_value="failed"),
        ):
            self.assertEqual(suite_script.main(), 1)

    def test_suite_cli_returns_failure_when_any_model_is_skipped_for_oom(self) -> None:
        summary = {
            "suite": "core",
            "passed_model_count": 0,
            "skipped_model_count": 1,
            "failed_model_count": 0,
            "runs": [],
        }
        argv = ["run_model_suite.py", "--manifest", "manifest.json", "--results-root", "results"]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(suite_script, "run_model_suite", return_value=summary),
            mock.patch.object(suite_script, "format_failure_summary", return_value="skipped_oom"),
        ):
            self.assertEqual(suite_script.main(), 1)


if __name__ == "__main__":
    unittest.main()
