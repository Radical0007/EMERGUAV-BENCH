import unittest

from emerguav_bench.evaluation.metrics import compute_grouped_metrics
from emerguav_bench.llm.normalizer import normalize_model_response
from emerguav_bench.llm.prompts import render_output_contract
from emerguav_bench.schemas.output_schema import validate_model_output
from emerguav_bench.uavscenes.manifest import make_placeholder_sequence
from scripts.build_samples import build_prediction_stub, build_seed_samples_from_manifest


def make_protocol_assessment() -> dict:
    return {
        "protocol_context_profile": "Sensitive-Site Respect",
        "primary_degradation": "dust_veil",
        "usability": "low",
        "trusted_modality": "lidar",
        "risk_level": "high",
        "risk_zone_label": "policy_sensitive_buffer",
        "viewpoint_summary": "policy_limited_viewpoint",
        "cause_label": "airborne_visibility_loss",
    }


def make_valid_output() -> dict:
    return {
        "Observation_Protocol_Assessment": make_protocol_assessment(),
        "Intent_Extraction": {
            "cleaned_instruction": "Proceed carefully while respecting mission constraints.",
            "target_object": None,
            "constraints": ["respect_sensitive_site_policy"],
            "target_goal": "maintain_sensitive_site_buffer",
        },
        "Action_Output": {
            "action_token": "Hover",
            "confidence": 0.8,
            "abstention_decision": "hold_position",
            "collaboration_request": {
                "required": False,
                "request_type": "none",
                "trusted_modality": "lidar",
            },
        },
        "Rationale_Short": "Protocol context requires conservative motion near a policy buffer.",
    }


class ProtocolContractTests(unittest.TestCase):
    def test_model_output_contract_uses_protocol_assessment(self) -> None:
        output = make_valid_output()

        self.assertEqual(validate_model_output(output), [])
        self.assertIn("Observation_Protocol_Assessment", output)
        self.assertIn("protocol_context_profile", output["Observation_Protocol_Assessment"])
        self.assertNotIn("scene_profile", output["Observation_Protocol_Assessment"])

    def test_prompt_contract_uses_protocol_assessment(self) -> None:
        contract = render_output_contract()

        self.assertIn("Observation_Protocol_Assessment (object, required)", contract)
        self.assertIn("protocol_context_profile", contract)
        self.assertNotIn("Visual_Diagnosis", contract)
        self.assertNotIn("scene_profile", contract)

    def test_normalizer_outputs_only_protocol_assessment_section(self) -> None:
        normalized = normalize_model_response(
            {
                "Observation_Protocol_Assessment": {
                    "risk_level": "high",
                    "trusted_modality": "lidar",
                },
                "Intent_Extraction": {},
                "Action_Output": {},
            }
        )

        self.assertIn("Observation_Protocol_Assessment", normalized)
        self.assertNotIn("Visual_Diagnosis", normalized)
        self.assertIn("protocol_context_profile", normalized["Observation_Protocol_Assessment"])

    def test_generated_samples_and_predictions_use_protocol_contract(self) -> None:
        manifests = [make_placeholder_sequence(sequence_id="SEQ_PROTOCOL", scene_family="AMtown", frame_count=1)]
        samples = build_seed_samples_from_manifest(
            manifests,
            protocol_context="Sensitive-Site Respect",
            policy_family="Worship-Site Respect",
        )
        predictions = build_prediction_stub(samples)

        self.assertIn("protocol_context", samples[0])
        self.assertIn("observation_severity_code", samples[0])
        self.assertNotIn("scenario_template", samples[0])
        self.assertNotIn("visual_severity_code", samples[0])
        self.assertIn("observation_protocol_gt", samples[0])
        self.assertNotIn("visual_diagnosis_gt", samples[0])
        self.assertIn("protocol_context_profile", samples[0]["observation_protocol_gt"])
        self.assertNotIn("scene_profile", samples[0]["observation_protocol_gt"])
        self.assertIn("Observation_Protocol_Assessment", predictions[0]["output"])
        self.assertNotIn("Visual_Diagnosis", predictions[0]["output"])

        grouped = compute_grouped_metrics(samples, predictions)
        self.assertIn("Sensitive-Site Respect", grouped["by_protocol_context"])
        self.assertIn("dust_veil", grouped["by_degradation_profile"])
        self.assertNotIn("by_scenario_family", grouped)
        self.assertNotIn("by_visual_profile", grouped)


if __name__ == "__main__":
    unittest.main()

