import unittest

from emerguav_bench.schemas.output_schema import validate_model_output
from emerguav_bench.schemas.sample_schema import validate_sample_record
from emerguav_bench.validation.json_validator import (
    SchemaValidationError,
    assert_valid_model_output,
    assert_valid_sample_record,
)


def make_valid_sample() -> dict:
    return {
        "sample_id": "emerguav-0001",
        "source_domain_bucket": "airport_like",
        "protocol_context": "Airport Perimeter Monitoring",
        "application_theme": "Hyper-Dense Urban Airspace",
        "policy_family": "Government / Critical Facility Sensitivity",
        "sequence_id": "HKairport01",
        "frame_id": "000123",
        "rgb_frame_path": "rgb/HKairport01/000123.jpg",
        "lidar_frame_path": "lidar/HKairport01/000123.bin",
        "pose_6dof": {
            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
            "orientation": {"roll": 0.0, "pitch": 0.1, "yaw": 0.2},
        },
        "map_id": "hkairport-map-v1",
        "degradation_labels": ["dust_veil"],
        "degradation_severity": "high",
        "observation_severity_code": "SV4",
        "language_entropy_labels": ["typo_omission_abbreviation"],
        "language_entropy_level": "EL3",
        "clean_instruction": "Descend cautiously to 30m and avoid NFZ edge.",
        "noisy_instruction": "descnd cautoisly 30m avoid nfz edg",
        "mission_constraints": ["avoid_nfz", "slow_descent"],
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
            "cleaned_instruction": "descend cautiously to 30m and avoid NFZ edge",
            "target_object": None,
            "constraints": ["avoid_nfz", "slow_descent"],
            "target_goal": "monitor_airport_perimeter_safely",
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
            "T7": "descend cautiously to 30m and avoid NFZ edge",
            "T8": ["avoid_nfz", "slow_descent"],
            "T9": "monitor_airport_perimeter_safely",
            "T10": "Hover_and_Alert",
            "T11": "hold_position",
            "T12": {"required": True, "request_type": "traffic_check", "trusted_modality": "lidar"},
        },
        "geometry_anchor": {"waypoint": [1.0, 2.0, 3.0]},
        "protocol_metadata": {"scene_family": "HKairport"},
    }


def make_valid_output() -> dict:
    return {
        "Observation_Protocol_Assessment": {
            "protocol_context_profile": "Airport Perimeter Monitoring",
            "primary_degradation": "dust_veil",
            "usability": "low",
            "trusted_modality": "lidar",
            "risk_level": "high",
            "risk_zone_label": "nfz_edge",
            "viewpoint_summary": "restricted_viewpoint",
            "cause_label": "airborne_visibility_loss",
        },
        "Intent_Extraction": {
            "cleaned_instruction": "descend cautiously to 30m and avoid NFZ edge",
            "target_object": None,
            "constraints": ["avoid_nfz", "slow_descent"],
            "target_goal": "monitor_airport_perimeter_safely",
        },
        "Action_Output": {
            "action_token": "Hover_and_Alert",
            "confidence": 0.71,
            "abstention_decision": "hold_position",
            "collaboration_request": {
                "required": True,
                "request_type": "traffic_check",
                "trusted_modality": "lidar",
            },
        },
        "Rationale_Short": "Observation input is unreliable near NFZ edge; safe descent cannot be confirmed.",
    }


class SchemaValidationTests(unittest.TestCase):
    def test_validate_sample_record_accepts_valid_fixture(self) -> None:
        errors = validate_sample_record(make_valid_sample())
        self.assertEqual(errors, [])

    def test_validate_sample_record_rejects_invalid_action_and_forbidden_conflict(self) -> None:
        sample = make_valid_sample()
        sample["action_token_gt"] = "Fly_Fast"
        sample["forbidden_actions"] = ["Fly_Fast"]

        errors = validate_sample_record(sample)
        fields = {error.field for error in errors}

        self.assertIn("action_token_gt", fields)
        self.assertIn("forbidden_actions", fields)

    def test_validate_sample_record_rejects_unknown_policy_family_and_invalid_codes(self) -> None:
        sample = make_valid_sample()
        sample["policy_family"] = "Unknown Policy"
        sample["observation_severity_code"] = "SV8"
        sample["language_entropy_level"] = "EL9"

        errors = validate_sample_record(sample)
        fields = {error.field for error in errors}

        self.assertIn("policy_family", fields)
        self.assertIn("observation_severity_code", fields)
        self.assertIn("language_entropy_level", fields)

    def test_assert_valid_sample_record_raises_schema_error(self) -> None:
        sample = make_valid_sample()
        del sample["sample_id"]

        with self.assertRaises(SchemaValidationError) as exc_info:
            assert_valid_sample_record(sample)

        self.assertTrue(any(error.field == "sample_id" for error in exc_info.exception.errors))

    def test_validate_model_output_accepts_valid_fixture(self) -> None:
        errors = validate_model_output(make_valid_output())
        self.assertEqual(errors, [])

    def test_validate_model_output_rejects_bad_confidence_and_action(self) -> None:
        output = make_valid_output()
        output["Action_Output"]["action_token"] = "Unsafe_Action"
        output["Action_Output"]["confidence"] = 1.5

        errors = validate_model_output(output)
        fields = {error.field for error in errors}

        self.assertIn("Action_Output.action_token", fields)
        self.assertIn("Action_Output.confidence", fields)

    def test_validate_model_output_rejects_missing_collaboration_request_shape(self) -> None:
        output = make_valid_output()
        output["Action_Output"]["collaboration_request"] = {"required": False, "request_type": "traffic_check"}

        errors = validate_model_output(output)
        fields = {error.field for error in errors}

        self.assertIn("Action_Output.collaboration_request.trusted_modality", fields)

    def test_assert_valid_model_output_raises_schema_error(self) -> None:
        output = make_valid_output()
        del output["Rationale_Short"]

        with self.assertRaises(SchemaValidationError) as exc_info:
            assert_valid_model_output(output)

        self.assertTrue(any(error.field == "Rationale_Short" for error in exc_info.exception.errors))


if __name__ == "__main__":
    unittest.main()
