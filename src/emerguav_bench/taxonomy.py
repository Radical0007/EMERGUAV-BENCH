"""Decision facets and diagnostic indices for EmergUAV-Bench.

Sources: evaluation/semantic_metrics.py (T3/T4/T8/T10), evaluation/metrics.py
(T5/T7 exact matching), config/vocab.py (modality vocabulary), and
language/entropy.py (the six implemented instruction transformations).
"""
from __future__ import annotations

from typing import Any

# Public presentation indices are independent of the evaluator's historical T IDs
# and report aliases. The order is shared by tables, plots, and report adapters.
# group_id/group_label are compatibility metadata for older evaluator reports;
# public taxonomy and allocation exports have only task and focus levels.
DIAGNOSTICS = (
    dict(group_id="G1", group_label="Emergency perception", diagnostic_id="D1", label="Scene condition", legacy_alias="D_vis", task_id="T4", output="Degradation label", score="Semantic degradation-label agreement", prompt="Report degradation affecting the emergency-scene observation."),
    dict(group_id="G1", group_label="Emergency perception", diagnostic_id="D2", label="Mission viewpoint", legacy_alias="D_lidar", task_id="T3", output="Viewpoint summary", score="Semantic viewpoint-summary agreement; not clearance scoring", prompt="Report viewpoint limitations affecting emergency observation."),
    dict(group_id="G2", group_label="Evidence trust", diagnostic_id="D3", label="Evidence selection", legacy_alias="D_trust", task_id="T5", output="Trusted modality", score="Exact match of submitted trusted-modality label", prompt="Which supplied evidence should guide the emergency decision?"),
    dict(group_id="G3", group_label="Operator command understanding", diagnostic_id="D4", label="Instruction recovery", legacy_alias="D_intent", task_id="T7", output="Cleaned instruction", score="Exact match of submitted recovered instruction", prompt="Recover the emergency operator's intended instruction."),
    dict(group_id="G3", group_label="Operator command understanding", diagnostic_id="D5", label="Emergency constraints", legacy_alias="D_constr", task_id="T8", output="Constraint set", score="Semantic constraint-set F1", prompt="Extract the emergency mission's route, safety, and site constraints."),
    dict(group_id="G4", group_label="Safe emergency action", diagnostic_id="D6", label="Safe action", legacy_alias="D_action", task_id="T10", output="Safe action", score="Safe-action agreement; forbidden actions score zero", prompt="Select the next safe action under the supplied emergency mission constraints."),
)
DIAGNOSTIC_IDS = tuple(row["diagnostic_id"] for row in DIAGNOSTICS)
DIAGNOSTIC_TO_LEGACY = {row["diagnostic_id"]: row["legacy_alias"] for row in DIAGNOSTICS}
LEGACY_TO_DIAGNOSTIC = {legacy: public for public, legacy in DIAGNOSTIC_TO_LEGACY.items()}
TASK_TO_DIAGNOSTIC = {row["task_id"]: row["diagnostic_id"] for row in DIAGNOSTICS}


# Each tuple contains a concrete focus, its scope, and an illustrative prompt.
# These are explanatory facets, not an exclusive sample partition or new scores.
IMPLEMENTATION_DETAILS = {
    "T4": [
        ("Clean observation",
         "Recognize the clean_visual reference class when no significant degradation is described.",
         "Is the observation clear, or does it show a specific degradation?"),
        ("Airborne visibility loss",
         "Identify dust_veil and its implemented dust, haze, fog, or smoke descriptions.",
         "Which degradation best describes the obscured aerial observation?"),
        ("Thermal distortion",
         "Identify heat_shimmer from heat, shimmer, mirage, or thermal-distortion descriptions.",
         "Does the apparent distortion indicate heat shimmer?"),
        ("Glare and saturation",
         "Identify high_glare from glare, overexposure, reflection, or saturation descriptions.",
         "What degradation accounts for the bright or washed-out region?"),
        ("Blur, noise, and scale loss",
         "Distinguish blur, noise, and small_target_long_distance as separate reference labels.",
         "Is lost image detail associated with blur, noise, or a distant small target?"),
        ("Occlusion, masking, and dropout",
         "Distinguish occlusion, partial_masking, and data_loss using the existing label and family rules.",
         "Is the missing evidence blocked, masked, or affected by data loss?"),
    ],
    "T3": [
        ("Mission-favorable viewpoint",
         "Match a favorable viewpoint summary using the implemented clear, usable, and aerial-view rules.",
         "Describe whether the current viewpoint supports the mission observation."),
        ("Constrained viewpoint",
         "Describe a limited, obstructed, or degraded viewpoint under the constrained_viewpoint rules.",
         "Which observation limitations constrain this viewpoint?"),
        ("Dangerous viewpoint",
         "Match dangerous_viewpoint descriptions involving an unsafe view, collision risk, or boundary proximity.",
         "Does the supplied context indicate a dangerous viewpoint?"),
        ("Low-visibility viewpoint",
         "Match low_visibility_viewpoint descriptions involving haze, fog, dust, or poor visibility.",
         "How does the reported visibility limit the current view?"),
    ],
    "T5": [
        ("RGB evidence selection",
         "Report the RGB trust label in the output vocabulary; scoring compares the submitted label exactly.",
         "Should RGB be the trusted evidence source in this observation?"),
        ("LiDAR evidence selection",
         "Report the LiDAR trust label from the supplied evidence; the task scores source selection, not point-cloud error.",
         "Should the supplied LiDAR evidence be trusted for this decision?"),
        ("Pose evidence selection",
         "Report pose as the trusted source when indicated by the annotated case; no pose-estimation error is computed.",
         "Does the pose evidence provide the trusted context for this decision?"),
        ("Combined evidence selection",
         "Report the combined-evidence label; multi and fused remain distinct submitted labels for exact matching.",
         "Should the decision use combined evidence rather than one source?"),
    ],
    "T7": [
        ("L0: canonical instruction",
         "Recover the reference instruction from the unperturbed text condition; score cleaned_instruction exactly.",
         "Restate the canonical operator instruction without adding requirements."),
        ("L1: orthographic perturbation",
         "Recover the reference instruction from generated character omissions and transposition errors.",
         "Recover the intended instruction from the misspelled operator text."),
        ("L2: lexical code-switching",
         "Recover the reference instruction after dictionary-based English and regional-term substitutions.",
         "Recover the canonical directive from this mixed-language command."),
        ("L3: hesitation and revision",
         "Recover the annotated reference under generated hesitation, correction, and reobservation phrases.",
         "Resolve the hesitation and revision wording into the reference directive."),
        ("L4: operator shorthand",
         "Recover the reference instruction from dictionary abbreviations and truncated long words.",
         "Expand this operator shorthand into the canonical instruction."),
        ("L5: compound perturbation",
         "Recover the reference instruction under the implemented combination of language transformations.",
         "Recover the directive despite the combined operator-text perturbations."),
    ],
    "T8": [
        ("Mission and route constraints",
         "Extract recorded mission, lane, corridor, and dispatch requirements using the existing constraint aliases.",
         "Which mission and route requirements must be retained?"),
        ("Restricted-zone constraints",
         "Extract NFZ avoidance, boundary restrictions, and no-approach sectors from the recorded constraint set.",
         "Which restricted-zone and no-approach constraints apply?"),
        ("Clearance and safety buffers",
         "Extract corridor clearance, standoff, and facility-buffer constraints as text labels, not measured geometry.",
         "Which clearance and buffer requirements must the instruction preserve?"),
        ("Visibility and conservative operation",
         "Extract slow_navigation, maintain_visibility_margin, and prefer_conservative_operation when annotated.",
         "Which speed, visibility, and conservative-operation constraints apply?"),
        ("People and sensitive-site constraints",
         "Extract crowd-overflight, privacy, filming, permit, and sensitive-site requirements supported by the alias map.",
         "Which crowd, privacy, or site-policy requirements constrain the action?"),
    ],
    "T10": [
        ("Reference-action agreement",
         "Check agreement with action_token_gt, subject to the forbidden-action check taking precedence.",
         "Select the next action under the supplied mission and safety constraints."),
        ("Acceptable alternative action",
         "Credit membership in acceptable_safe_alternatives when the predicted action is not forbidden.",
         "Can the selected action satisfy the annotated safe-alternative set?"),
        ("Forbidden or unaccepted action",
         "Give zero credit to a forbidden action or an action outside the annotated safe set.",
         "Does the selected action violate the recorded allowed and forbidden sets?"),
    ],
}


# Compact paper-facing focuses group the existing details; scoring is unchanged.
# Member integers refer to the original 28-detail order, now named I01-I28.
FOCUS_GROUPS = {
    "T4": [
        ("Clear emergency-scene observation", "Clear scene\nobservation", (1,),
         "Recognize the clean_visual reference class when the emergency-scene observation has no described significant degradation.",
         "Is the emergency-scene observation clear?"),
        ("Environmental and photometric degradation", "Environment /\nphotometric effects", (2, 3, 4),
         "Identify airborne visibility loss, thermal distortion, or glare and saturation affecting emergency observation using existing degradation-label rules.",
         "Is the emergency view degraded by smoke or dust, thermal distortion, or glare?"),
        ("Detail loss and missing evidence", "Detail loss /\nmissing evidence", (5, 6),
         "Distinguish blur, noise, small-target scale loss, occlusion, partial masking, and data loss in supplied emergency-scene evidence.",
         "Is the supplied emergency-scene evidence degraded in detail, obscured, masked, or missing?"),
    ],
    "T3": [
        ("Mission-favorable viewpoint", "Favorable\nviewpoint", (7,),
         "Match a viewpoint supporting emergency mission observation using the implemented clear, usable, and aerial-view rules.",
         "Does this viewpoint support the emergency mission observation?"),
        ("Constrained or low-visibility viewpoint", "Constrained /\nlow visibility", (8, 10),
         "Describe limited, obstructed, degraded, or low-visibility emergency observation using existing viewpoint-summary rules.",
         "Which viewpoint or visibility limitations constrain emergency observation?"),
        ("Dangerous viewpoint", "Dangerous\nviewpoint", (9,),
         "Match dangerous_viewpoint descriptions involving an unsafe view, collision risk, or boundary proximity during the supplied emergency mission.",
         "Does the emergency mission context indicate a dangerous viewpoint?"),
    ],
    "T5": [
        ("Single-source evidence selection", "Single-source\nevidence", (11, 12, 13),
         "Select the annotated RGB, LiDAR, or pose source for the emergency decision; scoring remains exact label matching, not sensor-estimation error.",
         "Which supplied source should guide the emergency decision?"),
        ("Combined evidence selection", "Combined\nevidence", (14,),
         "Select the annotated combined-evidence label for the emergency decision; multi and fused remain distinct labels for exact matching.",
         "Should this emergency decision use combined evidence?"),
    ],
    "T7": [
        ("Canonical instruction (L0)", "L0 · Canonical\ninstruction", (15,),
         "Recover the emergency operator's reference instruction in the unperturbed L0 condition; cleaned_instruction is scored exactly.",
         "Restate the emergency operator's canonical instruction without adding requirements."),
        ("Single perturbation family (L1-L4)", "L1-L4 · Single\nperturbation family", (16, 17, 18, 19),
         "Recover an emergency operator's instruction under one implemented text family: orthographic error, lexical code-switching, hesitation and revision, or shorthand. Urgency is illustrative context, not a measured variable.",
         "Recover the emergency operator's directive from this perturbed instruction."),
        ("Compound perturbation (L5)", "L5 · Compound\nperturbation", (20,),
         "Recover the emergency operator's reference instruction under the implemented L5 combination of text transformations.",
         "Recover the emergency directive despite combined operator-text perturbations."),
    ],
    "T8": [
        ("Mission and route constraints", "Mission /\nroute", (21,),
         "Extract recorded emergency mission, lane, corridor, and dispatch requirements using existing constraint aliases.",
         "Which emergency mission and route requirements must be retained?"),
        ("Operating-safety constraints", "Operating\nsafety", (23, 24),
         "Extract clearance, standoff, facility-buffer, speed, visibility-margin, and conservative-operation requirements for the emergency mission as annotated text labels.",
         "Which clearance, visibility, and conservative-operation requirements apply during this emergency mission?"),
        ("Restricted-zone and site constraints", "Restricted zones /\npeople / sites", (22, 25),
         "Extract recorded NFZ, boundary, no-approach, crowd, privacy, filming, permit, and sensitive-site restrictions relevant to emergency access and people safety.",
         "Which zone restrictions and people or sensitive-site protections must the emergency instruction preserve?"),
    ],
    "T10": [
        ("Reference or acceptable safe action", "Reference /\nacceptable action", (26, 27),
         "Credit the emergency mission's reference action or an annotated acceptable alternative only after the forbidden-action check.",
         "Is the emergency action the annotated reference or an acceptable safe alternative?"),
        ("Forbidden or unaccepted action", "Forbidden /\nunaccepted action", (28,),
         "Give zero credit to an emergency action forbidden by the recorded constraints or outside the annotated safe set.",
         "Does this emergency action violate the recorded allowed and forbidden sets?"),
    ],
}
TASK_FACETS = {
    task_id: [(name, definition, prompt) for name, _, _, definition, prompt in focuses]
    for task_id, focuses in FOCUS_GROUPS.items()
}


def taxonomy_implementation_details() -> list[dict[str, str]]:
    """Preserve all 28 original definitions under a distinct implementation ID."""
    rows = []
    for diagnostic in DIAGNOSTICS:
        for name, definition, prompt in IMPLEMENTATION_DETAILS[diagnostic["task_id"]]:
            index = len(rows) + 1
            public_diagnostic = {key: value for key, value in diagnostic.items() if key not in ("group_id", "group_label")}
            rows.append({**public_diagnostic, "detail_id": f"I{index:02d}",
                         "legacy_facet_id": f"F{index:02d}", "detail_label": name,
                         "definition": definition, "prompt": prompt})
    return rows


def taxonomy_facets() -> list[dict[str, Any]]:
    """Return 16 public focuses with all 28 implementation details traceable.

    T8 constraints can co-occur; T10 focuses remain prediction-dependent branches.
    Neither compact focuses nor their members introduce additional scores.
    """
    details = {row["detail_id"]: row for row in taxonomy_implementation_details()}
    rows = []
    for diagnostic in DIAGNOSTICS:
        public_diagnostic = {key: value for key, value in diagnostic.items() if key not in ("group_id", "group_label")}
        for local_index, (name, display_label, indices, definition, prompt) in enumerate(FOCUS_GROUPS[diagnostic["task_id"]], 1):
            member_ids = [f"I{index:02d}" for index in indices]
            rows.append({
                **public_diagnostic,
                "facet_id": f"F{len(rows) + 1:02d}",
                "facet_label": name,
                "display_label": display_label,
                "definition": definition,
                "prompt": prompt,
                "member_ids": member_ids,
                "legacy_member_ids": [details[key]["legacy_facet_id"] for key in member_ids],
                "member_count": len(member_ids),
                "members": [details[key] for key in member_ids],
                "hierarchy_index": f"{diagnostic['diagnostic_id'][1:]}.{local_index}",
            })
    return rows


FACET_INDEX = {row["facet_id"]: row for row in taxonomy_facets()}


def taxonomy_rows(dimensions: list[tuple] | None = None) -> list[list[str]]:
    """Render the task/focus hierarchy using D/F indices; accept legacy callers."""
    selected_tasks = {row[3] for row in dimensions} if dimensions is not None else None
    return [
        [f"{row['diagnostic_id']} {row['label']}",
         f"{row['facet_id']} {row['facet_label']}", row["definition"], row["prompt"]]
        for row in taxonomy_facets()
        if selected_tasks is None or row["task_id"] in selected_tasks
    ]


def sample_to_facet_breakdown(sample: dict[str, Any]) -> dict[str, list[str]]:
    """Map a sample to active decision facets for each diagnostic dimension."""
    breakdown: dict[str, list[str]] = {f"D{i}": [] for i in range(1, 7)}

    # D1 (T4) Scene condition: F01, F02, F03
    degradation = (
        sample.get("observation_protocol_gt", {}).get("primary_degradation")
        or sample.get("paper_task_gt", {}).get("T4")
        or (sample.get("degradation_labels") or ["clean_visual"])[0]
    )
    deg_str = str(degradation).lower()
    if deg_str in ("clean_visual", "clean", "none", "clean_observation"):
        breakdown["D1"].append("F01")
    elif deg_str in ("dust_veil", "heat_shimmer", "high_glare") or any(k in deg_str for k in ("dust", "heat", "shimmer", "glare")):
        breakdown["D1"].append("F02")
    else:
        breakdown["D1"].append("F03")

    # D2 (T3) Mission viewpoint: F04, F05, F06
    viewpoint = (
        sample.get("observation_protocol_gt", {}).get("viewpoint_summary")
        or sample.get("paper_task_gt", {}).get("T3")
        or "constrained_viewpoint"
    )
    vp_str = str(viewpoint).lower()
    if any(k in vp_str for k in ("favorable", "clear", "aerial", "open")):
        breakdown["D2"].append("F04")
    elif any(k in vp_str for k in ("danger", "hazard", "unsafe", "critical")):
        breakdown["D2"].append("F06")
    else:
        breakdown["D2"].append("F05")

    # D3 (T5) Evidence selection: F07, F08
    modality = (
        sample.get("observation_protocol_gt", {}).get("trusted_modality")
        or sample.get("paper_task_gt", {}).get("T5")
        or "lidar"
    )
    mod_str = str(modality).lower()
    if mod_str in ("multi", "fused", "combined"):
        breakdown["D3"].append("F08")
    else:
        breakdown["D3"].append("F07")

    # D4 (T7) Instruction recovery: F09, F10, F11
    entropy_level = (
        sample.get("icassp_entropy_level")
        or sample.get("language_entropy_level")
        or "L0"
    )
    ent_str = str(entropy_level).upper()
    if ent_str in ("L0", "EL1", "CLEAN_PROMPT"):
        breakdown["D4"].append("F09")
    elif ent_str in ("L5", "EL5", "MIXED_ENTROPY"):
        breakdown["D4"].append("F11")
    else:
        breakdown["D4"].append("F10")

    # D5 (T8) Emergency constraints: F12, F13, F14 (co-occurring)
    constraints = (
        sample.get("mission_constraints")
        or sample.get("intent_extraction_gt", {}).get("constraints")
        or sample.get("paper_task_gt", {}).get("T8")
        or []
    )
    constraints_str = " ".join(str(c).lower() for c in constraints)
    if any(k in constraints_str for k in ("corridor", "route", "lane", "dispatch", "patrol", "target", "hotspot")):
        breakdown["D5"].append("F12")
    if any(k in constraints_str for k in ("clearance", "standoff", "slow", "visibility", "buffer", "conservative", "speed", "margin")):
        breakdown["D5"].append("F13")
    if any(k in constraints_str for k in ("nfz", "hazard", "personnel", "crowd", "privacy", "sensitive", "site", "no_approach", "protect")):
        breakdown["D5"].append("F14")
    if not breakdown["D5"]:
        breakdown["D5"].append("F12")

    # D6 (T10) Safe action: F15 (reference/safe action), F16 (forbidden boundary)
    breakdown["D6"].append("F15")
    forbidden = sample.get("forbidden_actions") or []
    if forbidden or sample.get("action_token_gt"):
        breakdown["D6"].append("F16")

    return breakdown


def sample_to_facets(sample: dict[str, Any]) -> list[str]:
    """Return the sorted active decision-facet IDs (F01-F16) for a sample."""
    breakdown = sample_to_facet_breakdown(sample)
    active_facets: list[str] = []
    for facet_list in breakdown.values():
        for facet_id in facet_list:
            if facet_id not in active_facets and facet_id in FACET_INDEX:
                active_facets.append(facet_id)
    return sorted(active_facets)


def classify_primary_focus(sample: dict[str, Any]) -> str:
    """Classify the primary decision focus facet (F01-F16) for a given sample."""
    if sample.get("primary_diagnostic_focus") in FACET_INDEX:
        return sample["primary_diagnostic_focus"]

    breakdown = sample_to_facet_breakdown(sample)

    # Prioritize specialized conditions
    if "F11" in breakdown["D4"]:
        return "F11"
    if "F06" in breakdown["D2"]:
        return "F06"
    if "F08" in breakdown["D3"]:
        return "F08"
    if "F01" in breakdown["D1"]:
        return "F01"
    if "F09" in breakdown["D4"]:
        return "F09"
    if "F04" in breakdown["D2"]:
        return "F04"
    if "F02" in breakdown["D1"]:
        return "F02"
    if "F03" in breakdown["D1"]:
        return "F03"
    if "F10" in breakdown["D4"]:
        return "F10"
    if "F05" in breakdown["D2"]:
        return "F05"
    if "F14" in breakdown["D5"]:
        return "F14"
    if "F13" in breakdown["D5"]:
        return "F13"
    if "F12" in breakdown["D5"]:
        return "F12"
    return "F15"


def summarize_dataset_facets(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit facet representation across the sample set, asserting all 16 decision facets are covered."""
    facet_counts = {facet["facet_id"]: 0 for facet in taxonomy_facets()}
    primary_counts = {facet["facet_id"]: 0 for facet in taxonomy_facets()}
    dim_counts: dict[str, dict[str, int]] = {f"D{i}": {} for i in range(1, 7)}

    for sample in samples:
        facets = sample_to_facets(sample)
        for f in facets:
            facet_counts[f] = facet_counts.get(f, 0) + 1

        primary = classify_primary_focus(sample)
        primary_counts[primary] = primary_counts.get(primary, 0) + 1

        breakdown = sample_to_facet_breakdown(sample)
        for dim, f_list in breakdown.items():
            for f in f_list:
                dim_counts[dim][f] = dim_counts[dim].get(f, 0) + 1

    total = max(1, len(samples))
    coverage_report = []
    for facet in taxonomy_facets():
        fid = facet["facet_id"]
        coverage_report.append({
            "facet_id": fid,
            "diagnostic_id": facet["diagnostic_id"],
            "task_id": facet["task_id"],
            "facet_label": facet["facet_label"],
            "sample_count": facet_counts[fid],
            "coverage_percentage": round(100.0 * facet_counts[fid] / total, 2),
            "primary_focus_count": primary_counts[fid],
        })

    missing_facets = [fid for fid, cnt in facet_counts.items() if cnt == 0]

    return {
        "total_samples": len(samples),
        "facet_counts": facet_counts,
        "facet_coverage_report": coverage_report,
        "primary_focus_counts": primary_counts,
        "dimension_counts": dim_counts,
        "all_16_facets_covered": len(missing_facets) == 0,
        "missing_facets": missing_facets,
    }
