import json
import shutil
import unittest
from collections import Counter
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageChops

from emerguav_bench.degradation.image_ops import generate_degraded_image
from emerguav_bench.degradation.variants import (
    STRICT_VARIANT_MATRIX,
    allocate_strict_variant_cases,
    build_strict_variant_samples,
    summarize_variant_coverage,
)
from emerguav_bench.validation.json_validator import assert_valid_sample_record
from scripts.build_variants import build_variants


def make_workspace_temp_dir(prefix: str) -> Path:
    base_dir = Path.cwd() / ".test_tmp"
    base_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = base_dir / f"{prefix}_{uuid4().hex}"
    temp_dir.mkdir()
    return temp_dir


def make_rgb_image(path: Path, color: tuple[int, int, int] = (80, 120, 160)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (96, 64), color)
    image.save(path, format="JPEG")


def make_base_sample(tmp_dir: Path) -> dict:
    rgb_path = tmp_dir / "assets" / "AMtown03" / "000000" / "rgb" / "frame.jpg"
    lidar_path = tmp_dir / "assets" / "AMtown03" / "000000" / "lidar" / "frame.txt"
    cam_label_path = tmp_dir / "assets" / "AMtown03" / "000000" / "labels" / "camera" / "frame.png"
    lidar_label_path = tmp_dir / "assets" / "AMtown03" / "000000" / "labels" / "lidar" / "frame.txt"
    frame_metadata_path = tmp_dir / "assets" / "AMtown03" / "000000" / "metadata" / "frame_metadata.json"
    make_rgb_image(rgb_path)
    lidar_path.parent.mkdir(parents=True, exist_ok=True)
    lidar_path.write_text("1 2 3\n", encoding="utf-8")
    make_rgb_image(cam_label_path, color=(0, 0, 0))
    lidar_label_path.parent.mkdir(parents=True, exist_ok=True)
    lidar_label_path.write_text("0 0 0\n", encoding="utf-8")
    frame_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    frame_metadata_path.write_text("{}", encoding="utf-8")
    return {
        "sample_id": "AMtown03::000000",
        "source_domain_bucket": "town_like",
        "protocol_context": "Smart-City Boulevard Patrol",
        "application_theme": "Hyper-Dense Urban Airspace",
        "policy_family": "General Flight Operations",
        "sequence_id": "AMtown03",
        "frame_id": "000000",
        "rgb_frame_path": str(rgb_path),
        "lidar_frame_path": str(lidar_path),
        "pose_6dof": {
            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
            "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        },
        "map_id": "uavscenes://map/AMtown03",
        "degradation_labels": ["clean_visual"],
        "degradation_severity": "low",
        "observation_severity_code": "SV1",
        "language_entropy_labels": ["clean_prompt"],
        "language_entropy_level": "EL1",
        "clean_instruction": "Proceed carefully.",
        "noisy_instruction": "Proceed carefully.",
        "mission_constraints": [],
        "observation_protocol_gt": {
            "protocol_context_profile": "Smart-City Boulevard Patrol",
            "primary_degradation": "clean_visual",
            "usability": "high",
            "trusted_modality": "multi",
            "risk_level": "low",
            "risk_zone_label": "urban_canyon",
            "viewpoint_summary": "mission_favorable_viewpoint",
            "cause_label": "clean_observation",
        },
        "intent_extraction_gt": {
            "cleaned_instruction": "Proceed carefully.",
            "target_object": None,
            "constraints": [],
            "target_goal": "maintain_boulevard_patrol_coverage",
        },
        "action_token_gt": "Proceed_Slow",
        "acceptable_safe_alternatives": ["Hover"],
        "forbidden_actions": ["Descend_Cautiously"],
        "risk_level_gt": "low",
        "collaboration_gt": {"required": False, "request_type": "none", "trusted_modality": "multi"},
        "paper_task_gt": {
            "T1": "Smart-City Boulevard Patrol",
            "T2": "urban_canyon",
            "T3": "mission_favorable_viewpoint",
            "T4": "clean_visual",
            "T5": "multi",
            "T6": {"usability": "high", "cause_label": "clean_observation"},
            "T7": "Proceed carefully.",
            "T8": [],
            "T9": "maintain_boulevard_patrol_coverage",
            "T10": "Proceed_Slow",
            "T11": "continue_execution",
            "T12": {"required": False, "request_type": "none", "trusted_modality": "multi"},
        },
        "geometry_anchor": {"type": "uavscenes_frame"},
        "protocol_metadata": {
            "scene_family": "AMtown",
            "uavscenes": {
                "is_placeholder": False,
                "sequence_id": "AMtown03",
                "scene_family": "AMtown",
                "source_paths": {
                    "rgb": str(rgb_path),
                    "lidar": str(lidar_path),
                    "camera_label": str(cam_label_path),
                    "lidar_label": str(lidar_label_path),
                },
                "materialized_paths": {
                    "rgb": str(rgb_path),
                    "lidar": str(lidar_path),
                    "camera_label": str(cam_label_path),
                    "lidar_label": str(lidar_label_path),
                    "frame_metadata": str(frame_metadata_path),
                },
                "sample_info": {"OriginalSceneName": "AMtown03"},
            },
        },
    }


class StrictVariantTests(unittest.TestCase):
    def test_generate_degraded_image_writes_changed_image_and_metadata(self) -> None:
        tmp_dir = make_workspace_temp_dir("degrade_image")
        try:
            source = tmp_dir / "source.jpg"
            target = tmp_dir / "dust.jpg"
            make_rgb_image(source)
            params = generate_degraded_image(
                source,
                target,
                degradation_label="dust_veil",
                severity="high",
                seed=7,
            )

            self.assertTrue(target.exists())
            self.assertEqual(params["degradation_label"], "dust_veil")
            diff = ImageChops.difference(Image.open(source).convert("RGB"), Image.open(target).convert("RGB"))
            self.assertIsNotNone(diff.getbbox())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_strict_variant_allocation_uses_moderate_curve_profile(self) -> None:
        plan = allocate_strict_variant_cases([{} for _ in range(252)], STRICT_VARIANT_MATRIX)
        counts = Counter(item.case.variant_id for item in plan)

        self.assertEqual(sum(counts.values()), 3024)
        self.assertEqual(
            counts,
            {
                "clean_general_el1": 212,
                "dust_low_visibility_typo": 229,
                "heat_low_visibility_codeswitch": 225,
                "glare_nfz_hesitation": 235,
                "blur_sensitive_operator": 219,
                "small_target_infra_clean": 206,
                "occlusion_crowd_mixed": 292,
                "noise_landmark_codeswitch": 285,
                "masking_privacy_mixed": 265,
                "data_loss_critical_nfz_mixed": 278,
                "occlusion_memorial_cross_culture": 272,
                "noise_multilingual_mixed": 306,
            },
        )
        self.assertGreater(max(counts.values()) - min(counts.values()), 80)
        self.assertLess(max(counts.values()) - min(counts.values()), 120)
        self.assertNotIn(252, counts.values())
        repeated_slots = [item for item in plan if item.repeat_index > 0]
        self.assertTrue(repeated_slots)

    def test_build_strict_variant_samples_covers_matrix_and_validates(self) -> None:
        tmp_dir = make_workspace_temp_dir("strict_variants")
        try:
            base_sample = make_base_sample(tmp_dir)
            variants = build_strict_variant_samples([base_sample], output_root=tmp_dir / "strict")

            self.assertEqual(len(variants), len(STRICT_VARIANT_MATRIX))
            coverage = summarize_variant_coverage(variants)
            self.assertEqual(set(coverage["by_degradation"]), {case.degradation_label for case in STRICT_VARIANT_MATRIX})
            self.assertIn("Sensitive-Site Respect", coverage["by_protocol_context"])
            self.assertIn("Worship-Site Respect", coverage["by_policy_family"])
            self.assertIn("mixed_entropy", coverage["by_language_profile"])
            for sample in variants:
                assert_valid_sample_record(sample)
                self.assertTrue(Path(sample["rgb_frame_path"]).exists())
                self.assertIn("degradation_generation", sample["protocol_metadata"])
                self.assertIn("table_profile", sample["protocol_metadata"])
                table_profile = sample["protocol_metadata"]["table_profile"]
                self.assertEqual(table_profile["allocation_mode"], "nonlinear_semantic_curve")
                self.assertIn("allocation_index", table_profile)
                self.assertIn("repeat_index", table_profile)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_build_variants_writes_offline_inference_outputs(self) -> None:
        tmp_dir = make_workspace_temp_dir("strict_script")
        try:
            base_sample = make_base_sample(tmp_dir)
            base_samples_path = tmp_dir / "base_samples.json"
            base_samples_path.write_text(json.dumps([base_sample], ensure_ascii=False), encoding="utf-8")

            report = build_variants(
                base_samples_path=base_samples_path,
                output_root=tmp_dir / "strict_out",
                run_offline_inference=True,
                context_frames=1,
                min_image_inputs=2,
                hardest_fraction=1.0,
            )

            self.assertEqual(report["variant_sample_count"], len(STRICT_VARIANT_MATRIX))
            self.assertTrue(Path(report["output_files"]["samples"]).exists())
            self.assertTrue(Path(report["output_files"]["offline_predictions"]).exists())
            self.assertTrue(Path(report["output_files"]["offline_evaluation_report"]).exists())
            evaluation = json.loads(Path(report["output_files"]["offline_evaluation_report"]).read_text(encoding="utf-8"))
            self.assertEqual(evaluation["task_metrics"]["evaluated_samples"], float(len(STRICT_VARIANT_MATRIX)))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()


