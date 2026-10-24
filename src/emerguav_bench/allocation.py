"""Reproducible design quotas for benchmark diagnostic-focus figures.

This allocates test-case *focus slots*, not observed dataset samples or outcomes.
It neither edits the dataset builder nor fills measured result statistics.
"""
from __future__ import annotations

from typing import Any

from emerguav_bench.taxonomy import DIAGNOSTICS, taxonomy_facets, taxonomy_implementation_details
from emerguav_bench.allocation_profiles import paper_curved_count_profile


# Same target as scripts/build_benchmark.py. No generated data is claimed.
TARGET_DIAGNOSTIC_FOCUS_COUNT = 1280
ALLOCATION_EXPONENT = 1.20
# Explicit bounded design weights, not inferred frequencies or difficulty scores.
# Sequence preserves the original 28-detail allocation, now indexed I01..I28.
# Compact F01..F16 quotas are sums of these integer allocations, never re-curved.
IMPLEMENTATION_FOCUS_WEIGHTS = dict(zip(
    (f"I{index:02d}" for index in range(1, 29)),
    (0.86, 1.16, 1.08, 0.96, 1.12, 1.04,
     0.90, 1.06, 1.14, 0.98,
     1.10, 0.94, 0.88, 1.18,
     0.84, 0.98, 1.12, 1.04, 0.92, 1.16,
     1.06, 1.18, 1.00, 0.90, 1.10,
     1.08, 0.94, 1.14),
))


def dataset_allocation_profile(total: int = TARGET_DIAGNOSTIC_FOCUS_COUNT) -> dict[str, Any]:
    """Allocate an exclusive primary focus to each *planned* test-case slot.

    Counts sum exactly at both public levels and in the detail appendix.
    Existing samples can support several
    diagnostics or constraints; those observed coverages must be counted later.
    D6 quotas request tests exercising judgment branches, not model success rates.
    """
    if type(total) is not int or total <= 0:
        raise ValueError("total must be a positive integer")
    facets = taxonomy_facets()
    implementation_details = taxonomy_implementation_details()
    counts = paper_curved_count_profile(
        IMPLEMENTATION_FOCUS_WEIGHTS, total=total, exponent=ALLOCATION_EXPONENT,
    )
    public_ids = {member: facet["facet_id"] for facet in facets for member in facet["member_ids"]}
    for detail in implementation_details:
        detail_id = detail["detail_id"]
        count = counts[detail_id]
        detail.update(weight=IMPLEMENTATION_FOCUS_WEIGHTS[detail_id], count=count,
                      proportion=count / total, public_facet_id=public_ids[detail_id])
    for facet in facets:
        count = sum(counts[key] for key in facet["member_ids"])
        facet.update(count=count, proportion=count / total)
    diagnostics = []
    for diagnostic in DIAGNOSTICS:
        count = sum(row["count"] for row in facets if row["diagnostic_id"] == diagnostic["diagnostic_id"])
        public_diagnostic = {key: value for key, value in diagnostic.items() if key not in ("group_id", "group_label")}
        diagnostics.append({**public_diagnostic, "count": count, "proportion": count / total})
    return {
        "schema_version": 3,
        "status": "planned_design_allocation",
        "unit": "primary_diagnostic_focus_slot",
        "total": total,
        "diagnostics": diagnostics,
        "facets": facets,
        "implementation_details": implementation_details,
        "method": {
            "name": "bounded_weights_curved_largest_remainder",
            "exponent": ALLOCATION_EXPONENT,
            "weight_range": [min(IMPLEMENTATION_FOCUS_WEIGHTS.values()), max(IMPLEMENTATION_FOCUS_WEIGHTS.values())],
            "weight_basis": "Explicit mildly nonuniform design weights; no empirical frequency or difficulty estimate.",
            "rounding": "Floor normalized powered implementation-detail weights, then allocate largest fractional remainders; ties use raw share then descending detail ID.",
            "aggregation": "Allocate the 28 implementation-detail quotas once; sum them into 16 public focuses and six emergency tasks. Implementation details are a traceability appendix, not a third public level. Do not exponentiate merged weights.",
        },
        "provenance": {
            "target_source": "Default 1280 from scripts/build_benchmark.py:TARGET_EMERGUAV_SAMPLE_COUNT; total can be overridden explicitly.",
            "allocation_source": "src/emerguav_bench/allocation_profiles.py:paper_curved_count_profile",
            "weights_source": "src/emerguav_bench/allocation.py:IMPLEMENTATION_FOCUS_WEIGHTS",
            "taxonomy_source": "src/emerguav_bench/taxonomy.py:DIAGNOSTICS/FOCUS_GROUPS/IMPLEMENTATION_DETAILS",
            "indices": "Two public levels: D1-D6 emergency tasks and F01-F16 focuses. I01-I28 are implementation-appendix references; legacy_facet_id records the superseded 28-focus index.",
            "observed_samples": None,
            "scope": "Preparation-only planning; not applied to the dataset builder and not measured dataset coverage.",
            "overlap": "Actual diagnostics and T8 constraints can overlap. Exactly one primary focus is assigned only in this planned slot budget.",
            "action_branches": "F15-F16 group tests of prediction-dependent T10 scoring branches (I26-I28); their quotas are not success, failure, or action-class frequencies.",
        },
    }
