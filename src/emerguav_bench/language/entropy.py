from __future__ import annotations

import math
import random
from typing import Any, Sequence

# Controlled 6-level Language Entropy Taxonomy (L0 ~ L5)
# Paper Reference: Section II-B & Eq. (2)-(3)
ENTROPY_TAXONOMY_MAP = {
    "L0": "L0_canonical_baseline",
    "L1": "L1_orthographic_perturbation",
    "L2": "L2_lexical_code_switching",
    "L3": "L3_temporal_disfluency",
    "L4": "L4_tactical_abbreviation",
    "L5": "L5_compound_high_entropy",
}

# L2 Dialectal terms mapping (Arabic/English domain jargon)
L2_DIALECT_DICTIONARY = {
    "proceed": "proceed shwaya",
    "carefully": "b-tani",
    "avoid": "tajanab",
    "restricted zone": "nfz edge",
    "constraints": "qawaid",
    "mission": "wajib",
    "maintain": "hafiz",
    "corridor": "masar",
    "safety": "alaman",
}

# L4 Operator shorthand abbreviations
L4_SHORTHAND_DICTIONARY = {
    "proceed": "proc",
    "carefully": "slow",
    "along": "alg",
    "corridor": "corr",
    "avoid": "chk",
    "restricted": "restr",
    "zone": "zn",
    "edge": "edg",
    "maintain": "keep",
    "mission": "msn",
    "constraints": "constr",
    "safety": "safe",
    "reobserve": "reobs",
    "collaboration": "collab",
    "escalation": "escal",
}


def compute_normalized_shannon_entropy(output_counts: dict[str, int]) -> float:
    """Computes Normalized Shannon Entropy H_norm(Y_d) in [0, 1] as defined in Eq. (3):

    H_norm(Y_d) = - (1 / log2 |Y_d|) * sum_{y in Y_d} p_d(y) * log2 p_d(y)
    """
    total = sum(output_counts.values())
    num_classes = len(output_counts)
    if total == 0 or num_classes <= 1:
        return 0.0

    entropy = 0.0
    for count in output_counts.values():
        if count > 0:
            prob = count / total
            entropy -= prob * math.log2(prob)

    max_entropy = math.log2(num_classes)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def apply_l1_orthographic_perturbation(text: str, rng: random.Random) -> str:
    """L1: Character omissions, transposition errors, typos."""
    words = text.split()
    perturbed = []
    for word in words:
        if len(word) > 4 and rng.random() < 0.35:
            op = rng.choice(["omit", "transpose", "swap"])
            if op == "omit":
                idx = rng.randint(1, len(word) - 2)
                word = word[:idx] + word[idx + 1 :]
            elif op == "transpose" and len(word) > 3:
                idx = rng.randint(1, len(word) - 3)
                word = word[:idx] + word[idx + 1] + word[idx] + word[idx + 2 :]
        perturbed.append(word)
    return " ".join(perturbed)


def apply_l2_lexical_code_switching(text: str, rng: random.Random) -> str:
    """L2: Lexical code-switching mixing domain jargon and regional dialect terms."""
    words = text.split()
    res = []
    for word in words:
        clean = word.lower().strip(",.")
        if clean in L2_DIALECT_DICTIONARY and rng.random() < 0.6:
            res.append(L2_DIALECT_DICTIONARY[clean])
        else:
            res.append(word)
    return " ".join(res)


def apply_l3_temporal_disfluency(text: str, rng: random.Random) -> str:
    """L3: Speech hesitation, real-time self-corrections, dynamic overrides."""
    prefixes = [
        "uh, wait, hold on... ",
        "ah correction... ",
        "wait, revise instruction... ",
        "uh, confirm status first... ",
    ]
    hesitations = ["... um ... ", ", I mean, ", "... wait ... "]
    prefix = rng.choice(prefixes) if rng.random() < 0.7 else ""
    mid = rng.choice(hesitations) if rng.random() < 0.5 else " "
    parts = text.split(",", 1)
    if len(parts) == 2:
        return f"{prefix}{parts[0]}{mid}reobserve first, then {parts[1].strip()}"
    return f"{prefix}{text}{mid}reobserve first"


def apply_l4_tactical_abbreviation(text: str) -> str:
    """L4: Extreme operational truncation, telegraphic syntax, acronym compression."""
    words = text.split()
    shorthand = []
    for word in words:
        clean = word.lower().strip(",.")
        if clean in L4_SHORTHAND_DICTIONARY:
            shorthand.append(L4_SHORTHAND_DICTIONARY[clean])
        elif len(clean) > 4:
            shorthand.append(clean[:3])
        else:
            shorthand.append(clean)
    return " ".join(shorthand)


def generate_language_entropy_variant(
    instruction: str,
    level: str = "L0",
    seed: int = 42,
) -> dict[str, Any]:
    """Generates an instruction variant under controlled language entropy level L0~L5."""
    rng = random.Random(seed)
    canonical = instruction.strip()
    level_key = level.upper()
    if level_key not in ENTROPY_TAXONOMY_MAP:
        level_key = "L0"

    entropy_name = ENTROPY_TAXONOMY_MAP[level_key]

    if level_key == "L0":
        perturbed = canonical
    elif level_key == "L1":
        perturbed = apply_l1_orthographic_perturbation(canonical, rng)
    elif level_key == "L2":
        perturbed = apply_l2_lexical_code_switching(canonical, rng)
    elif level_key == "L3":
        perturbed = apply_l3_temporal_disfluency(canonical, rng)
    elif level_key == "L4":
        perturbed = apply_l4_tactical_abbreviation(canonical)
    elif level_key == "L5":
        # Compound high-entropy: blend L1, L2, L4
        step1 = apply_l2_lexical_code_switching(canonical, rng)
        step2 = apply_l1_orthographic_perturbation(step1, rng)
        perturbed = apply_l4_tactical_abbreviation(step2)
    else:
        perturbed = canonical

    return {
        "entropy_level": level_key,
        "entropy_name": entropy_name,
        "canonical_instruction": canonical,
        "perturbed_instruction": perturbed,
        "seed": seed,
    }


def make_placeholder_noisy_instruction(
    instruction: str,
    language_entropy_labels: Sequence[str] | None = None,
) -> str:
    """Helper for pipeline backwards compatibility."""
    labels = list(language_entropy_labels or ["clean_prompt"])
    primary_label = labels[0] if labels else "clean_prompt"
    if primary_label == "clean_prompt":
        return instruction
    elif primary_label == "typo_omission_abbreviation":
        return generate_language_entropy_variant(instruction, "L1")["perturbed_instruction"]
    elif primary_label == "code_switching":
        return generate_language_entropy_variant(instruction, "L2")["perturbed_instruction"]
    elif primary_label == "hesitation_revision":
        return generate_language_entropy_variant(instruction, "L3")["perturbed_instruction"]
    elif primary_label == "operator_shorthand":
        return generate_language_entropy_variant(instruction, "L4")["perturbed_instruction"]
    elif primary_label in ("mixed_entropy", "compound"):
        return generate_language_entropy_variant(instruction, "L5")["perturbed_instruction"]
    return instruction
