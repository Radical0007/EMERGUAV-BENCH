import unittest

import emerguav_bench.protocol_context.mapper as protocol_mapper
import emerguav_bench.protocol_context.templates as protocol_templates
from emerguav_bench.protocol_context.mapper import assign_protocol_context, get_source_domain_bucket
from emerguav_bench.protocol_context.templates import PROTOCOL_CONTEXT_TEMPLATES
from emerguav_bench.uavscenes.indexer import UAVScenesIndexer
from scripts.build_samples import build_seed_samples_from_manifest


class ProtocolContextMappingTests(unittest.TestCase):
    def test_protocol_context_api_does_not_export_legacy_names(self) -> None:
        self.assertFalse(hasattr(protocol_mapper, "assign_scenario"))
        self.assertFalse(hasattr(protocol_mapper, "ScenarioAssignment"))
        self.assertFalse(hasattr(protocol_templates, "SCENARIO_TEMPLATES"))
        self.assertFalse(hasattr(protocol_templates, "ScenarioTemplate"))

    def test_scene_family_maps_to_expected_bucket(self) -> None:
        self.assertEqual(get_source_domain_bucket("HKairport"), "airport_like")
        self.assertEqual(get_source_domain_bucket("AMtown"), "town_like")

    def test_assign_protocol_context_returns_primary_protocol_context(self) -> None:
        assignment = assign_protocol_context(sequence_id="SEQ_AIRPORT_01", scene_family="HKairport")
        self.assertEqual(assignment.source_domain_bucket, "airport_like")
        self.assertEqual(assignment.protocol_context, "Airport Perimeter Monitoring")
        self.assertIn("avoid_nfz", assignment.mission_constraints)
        self.assertEqual(assignment.policy_family, "Government / Critical Facility Sensitivity")

    def test_assign_protocol_context_can_attach_nfz_overlay(self) -> None:
        assignment = assign_protocol_context(sequence_id="SEQ_TOWN_01", scene_family="AMtown", enable_nfz_overlay=True)
        self.assertIn("NFZ Edge Caution", assignment.protocol_metadata["overlay_protocol_contexts"])
        self.assertNotIn("overlay_templates", assignment.protocol_metadata)
        self.assertIn("respect_no_approach_sector", assignment.mission_constraints)
        self.assertIn("nfz_edge_caution", assignment.overlay_profiles)

    def test_assign_protocol_context_can_select_sensitive_protocol_context_and_policy_pack(self) -> None:
        assignment = assign_protocol_context(
            sequence_id="SEQ_TOWN_01",
            scene_family="AMtown",
            protocol_context="Sensitive-Site Respect",
            policy_family="Privacy-Sensitive Institutions",
        )
        self.assertEqual(assignment.protocol_context, "Sensitive-Site Respect")
        self.assertEqual(assignment.application_theme, "Cultural & Religious Sensitivity")
        self.assertEqual(assignment.policy_family, "Privacy-Sensitive Institutions")
        self.assertIn("privacy_sensitive_context", assignment.overlay_profiles)

    def test_overlay_protocol_context_is_marked_overlay(self) -> None:
        overlay = PROTOCOL_CONTEXT_TEMPLATES["NFZ Edge Caution"]
        self.assertTrue(overlay.is_overlay)
        self.assertEqual(overlay.source_domain_bucket, "overlay_on_any_bucket")
        self.assertEqual(overlay.protocol_context_id, "S5")


class SeedSampleBuilderTests(unittest.TestCase):
    def test_build_seed_samples_from_placeholder_manifest(self) -> None:
        manifests = UAVScenesIndexer().build_manifest(sequence_ids=["SEQ_AIRPORT_01"])
        samples = build_seed_samples_from_manifest(manifests)

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["source_domain_bucket"], "airport_like")
        self.assertEqual(sample["protocol_context"], "Airport Perimeter Monitoring")
        self.assertEqual(sample["protocol_metadata"]["protocol_context_id"], "S1")
        self.assertNotIn("scenario_id", sample["protocol_metadata"])
        self.assertTrue(sample["rgb_frame_path"].startswith("placeholder://rgb/"))
        self.assertIn(sample["action_token_gt"], ["Hover_and_Alert", "Hover", "Reobserve"])
        self.assertIn("paper_task_gt", sample)
        self.assertEqual(sample["paper_task_gt"]["T1"], sample["protocol_context"])

    def test_build_seed_samples_with_overlay_updates_metadata(self) -> None:
        manifests = UAVScenesIndexer().build_manifest(sequence_ids=["SEQ_TOWN_01"])
        samples = build_seed_samples_from_manifest(manifests, enable_nfz_overlay=True)

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertIn("NFZ Edge Caution", sample["protocol_metadata"]["overlay_protocol_contexts"])
        self.assertNotIn("overlay_templates", sample["protocol_metadata"])
        self.assertIn("respect_no_approach_sector", sample["mission_constraints"])
        self.assertTrue(sample["protocol_metadata"]["placeholder_source"])

    def test_build_seed_samples_supports_sensitive_template_and_entropy_fields(self) -> None:
        manifests = UAVScenesIndexer().build_manifest(sequence_ids=["SEQ_TOWN_01"])
        samples = build_seed_samples_from_manifest(
            manifests,
            protocol_context="Sensitive-Site Respect",
            policy_family="Worship-Site Respect",
            language_entropy_labels=["code_switching"],
        )

        sample = samples[0]
        self.assertEqual(sample["protocol_context"], "Sensitive-Site Respect")
        self.assertEqual(sample["policy_family"], "Worship-Site Respect")
        self.assertEqual(sample["language_entropy_level"], "EL4")
        self.assertEqual(sample["paper_task_gt"]["T9"], "maintain_sensitive_site_buffer")


if __name__ == "__main__":
    unittest.main()


