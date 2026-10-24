from __future__ import annotations

import ast
import unittest
from pathlib import Path

from emerguav_bench.allocation import TARGET_DIAGNOSTIC_FOCUS_COUNT, dataset_allocation_profile
from emerguav_bench.taxonomy import DIAGNOSTIC_TO_LEGACY, DIAGNOSTICS, taxonomy_facets, taxonomy_implementation_details, taxonomy_rows


class AllocationTests(unittest.TestCase):
    def test_public_indices_preserve_evaluator_mapping(self):
        self.assertEqual([row["task_id"] for row in DIAGNOSTICS], ["T4", "T3", "T5", "T7", "T8", "T10"])
        self.assertEqual(list(DIAGNOSTIC_TO_LEGACY.values()), ["D_vis", "D_lidar", "D_trust", "D_intent", "D_constr", "D_action"])
        facets = taxonomy_facets()
        self.assertEqual([row["facet_id"] for row in facets], [f"F{i:02d}" for i in range(1, 17)])
        self.assertEqual([sum(row["diagnostic_id"] == diagnostic["diagnostic_id"] for row in facets) for diagnostic in DIAGNOSTICS], [3, 3, 2, 3, 3, 2])

    def test_compact_mapping_preserves_each_implementation_detail_once(self):
        facets = taxonomy_facets()
        details = taxonomy_implementation_details()
        member_ids = [member for facet in facets for member in facet["member_ids"]]
        self.assertEqual(len(member_ids), 28)
        self.assertEqual(set(member_ids), {f"I{i:02d}" for i in range(1, 29)})
        expected = [(1,), (2, 3, 4), (5, 6), (7,), (8, 10), (9,), (11, 12, 13), (14,),
                    (15,), (16, 17, 18, 19), (20,), (21,), (23, 24), (22, 25), (26, 27), (28,)]
        for facet, indices in zip(facets, expected):
            self.assertEqual(facet["legacy_member_ids"], [f"F{i:02d}" for i in indices])
            self.assertTrue(all(member["diagnostic_id"] == facet["diagnostic_id"] for member in facet["members"]))
        self.assertEqual([row["legacy_facet_id"] for row in details], [f"F{i:02d}" for i in range(1, 29)])
        self.assertTrue(all(row["definition"] and row["prompt"] for row in details))

    def test_public_taxonomy_has_only_task_and_focus_levels(self):
        self.assertEqual([row["label"] for row in DIAGNOSTICS], ["Scene condition", "Mission viewpoint", "Evidence selection", "Instruction recovery", "Emergency constraints", "Safe action"])
        rows = taxonomy_rows()
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(len(row) == 4 and row[0].startswith("D") and row[1].startswith("F") for row in rows))
        profile = dataset_allocation_profile()
        self.assertEqual(profile["schema_version"], 3)
        self.assertNotIn("groups", profile)
        for level in ("diagnostics", "facets", "implementation_details"):
            self.assertTrue(all("group_id" not in row and "group_label" not in row for row in profile[level]))
        self.assertTrue(all(len(row["hierarchy_index"].split(".")) == 2 for row in taxonomy_facets()))

    def test_allocation_totals_aggregate_at_every_level(self):
        for total in [1, 31, 1280, 2500]:
            profile = dataset_allocation_profile(total)
            for level in ["diagnostics", "facets", "implementation_details"]:
                self.assertEqual(sum(row["count"] for row in profile[level]), total)
                self.assertAlmostEqual(sum(row["proportion"] for row in profile[level]), 1)
            for diagnostic in profile["diagnostics"]:
                self.assertEqual(diagnostic["count"], sum(row["count"] for row in profile["facets"] if row["diagnostic_id"] == diagnostic["diagnostic_id"]))
            details = {row["detail_id"]: row for row in profile["implementation_details"]}
            for facet in profile["facets"]:
                self.assertEqual(facet["count"], sum(details[key]["count"] for key in facet["member_ids"]))
                self.assertTrue(all(details[key]["public_facet_id"] == facet["facet_id"] for key in facet["member_ids"]))
                self.assertNotIn("weight", facet)

    def test_compaction_keeps_original_diagnostic_and_detail_budgets(self):
        profile = dataset_allocation_profile()
        self.assertEqual([row["count"] for row in profile["diagnostics"]], [276, 180, 182, 268, 233, 141])
        self.assertEqual([row["count"] for row in profile["facets"]], [37, 143, 96, 39, 90, 51, 128, 54, 36, 179, 53, 47, 83, 103, 89, 52])
        self.assertEqual([row["count"] for row in profile["implementation_details"]],
                         [37, 53, 48, 42, 50, 46, 39, 47, 51, 43, 49, 41, 38, 54, 36, 43, 50, 46, 40, 53, 47, 54, 44, 39, 49, 48, 41, 52])

    def test_reproducible_mild_variation_and_planning_provenance(self):
        profile = dataset_allocation_profile()
        self.assertEqual(profile, dataset_allocation_profile())
        counts = [row["count"] for row in profile["implementation_details"]]
        self.assertGreater(len(set(counts)), 8)
        self.assertGreater(min(counts), 0)
        self.assertLess(max(counts) / min(counts), 1.6)
        self.assertEqual(profile["status"], "planned_design_allocation")
        self.assertIsNone(profile["provenance"]["observed_samples"])
        self.assertIn("prediction-dependent", profile["provenance"]["action_branches"])
        self.assertIn("not applied", profile["provenance"]["scope"])

    def test_design_target_matches_current_builder(self):
        builder = Path(__file__).resolve().parents[1] / "scripts" / "build_benchmark.py"
        tree = ast.parse(builder.read_text(encoding="utf-8"))
        target = next(node.value.value for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(name, ast.Name) and name.id == "TARGET_EMERGUAV_SAMPLE_COUNT" for name in node.targets))
        self.assertEqual(TARGET_DIAGNOSTIC_FOCUS_COUNT, target)

    def test_invalid_totals_rejected(self):
        for total in [0, -1, 1.5, True, "1280"]:
            with self.subTest(total=total), self.assertRaises(ValueError):
                dataset_allocation_profile(total)


if __name__ == "__main__":
    unittest.main()
