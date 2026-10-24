from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image

from emerguav_bench.llm.io import load_json, save_json


def source_anchor_id(sample_id: str) -> str:
    return "::".join(str(sample_id).split("::")[:2])


def _resolve_path(raw_path: str, *, repo_root: Path) -> Path:
    normalized = raw_path.replace("\\", "/")
    path = Path(normalized)
    if path.is_file():
        return path
    candidate = repo_root / normalized
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(raw_path)


def _frame_number(frame_id: str) -> int | None:
    match = re.match(r"^(\d+)", str(frame_id))
    return int(match.group(1)) if match else None


def _timestamp_from_path(path: Path) -> float | None:
    matches = re.findall(r"(\d{10}\.\d+)", path.name)
    return float(matches[0]) if matches else None


def _dhash(path: Path, *, width: int = 16, height: int = 16) -> int:
    with Image.open(path) as image:
        resized = image.convert("L").resize((width + 1, height))
        pixel_source = getattr(resized, "get_flattened_data", resized.getdata)
        pixels = list(pixel_source())
    value = 0
    row_width = width + 1
    for y in range(height):
        row_start = y * row_width
        for x in range(width):
            value = (value << 1) | int(pixels[row_start + x] > pixels[row_start + x + 1])
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gap_summary(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"minimum": None, "median": None, "maximum": None}
    return {"minimum": min(values), "median": median(values), "maximum": max(values)}


def analyze_dependence(
    *,
    base_samples: list[dict[str, Any]],
    strict_samples: list[dict[str, Any]],
    repo_root: Path,
) -> dict[str, Any]:
    base_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in base_samples:
        base_by_sequence[str(sample["sequence_id"])].append(sample)

    strict_anchor_counts = Counter(source_anchor_id(sample["sample_id"]) for sample in strict_samples)
    strict_sequence_counts = Counter(str(sample["sequence_id"]) for sample in strict_samples)
    report: dict[str, Any] = {
        "source_anchor_count": len(strict_anchor_counts),
        "strict_variant_count": len(strict_samples),
        "sequence_count": len(base_by_sequence),
        "variants_per_anchor": _gap_summary(list(strict_anchor_counts.values())),
        "sequences": {},
        "perceptual_hash": {
            "algorithm": "256-bit difference hash over 17x16 grayscale resize",
            "threshold_counts": {"0": 0, "2": 0, "4": 0, "8": 0},
            "pair_count": 0,
            "most_similar_pairs": [],
        },
    }

    similar_pairs: list[dict[str, Any]] = []
    exact_hash_to_samples: dict[str, list[str]] = defaultdict(list)
    for sequence_id in sorted(base_by_sequence):
        rows = sorted(base_by_sequence[sequence_id], key=lambda item: (_frame_number(item["frame_id"]) or -1, item["frame_id"]))
        resolved = [(row, _resolve_path(str(row["rgb_frame_path"]), repo_root=repo_root)) for row in rows]
        frame_numbers = [value for value in (_frame_number(row["frame_id"]) for row, _ in resolved) if value is not None]
        timestamps = [value for value in (_timestamp_from_path(path) for _, path in resolved) if value is not None]
        frame_gaps = [second - first for first, second in zip(frame_numbers, frame_numbers[1:])]
        timestamp_gaps = [second - first for first, second in zip(timestamps, timestamps[1:])]
        hashes: list[tuple[dict[str, Any], Path, int]] = []
        for row, path in resolved:
            digest = _sha256(path)
            exact_hash_to_samples[digest].append(str(row["sample_id"]))
            hashes.append((row, path, _dhash(path)))
        for (first, _, first_hash), (second, _, second_hash) in combinations(hashes, 2):
            distance = (first_hash ^ second_hash).bit_count()
            report["perceptual_hash"]["pair_count"] += 1
            for threshold in (0, 2, 4, 8):
                if distance <= threshold:
                    report["perceptual_hash"]["threshold_counts"][str(threshold)] += 1
            similar_pairs.append(
                {
                    "sequence_id": sequence_id,
                    "first_sample_id": first["sample_id"],
                    "second_sample_id": second["sample_id"],
                    "hamming_distance": distance,
                }
            )
        report["sequences"][sequence_id] = {
            "source_anchor_count": len(rows),
            "strict_variant_count": strict_sequence_counts[sequence_id],
            "frame_index_gap": _gap_summary(frame_gaps),
            "timestamp_gap_seconds": _gap_summary(timestamp_gaps),
        }

    exact_duplicate_groups = [sample_ids for sample_ids in exact_hash_to_samples.values() if len(sample_ids) > 1]
    report["exact_duplicate_image_group_count"] = len(exact_duplicate_groups)
    report["exact_duplicate_sample_ids"] = exact_duplicate_groups
    report["perceptual_hash"]["most_similar_pairs"] = sorted(
        similar_pairs,
        key=lambda row: (row["hamming_distance"], row["sequence_id"], row["first_sample_id"], row["second_sample_id"]),
    )[:30]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit source-frame dependence and near-duplicate risk.")
    parser.add_argument("--base-samples", required=True)
    parser.add_argument("--strict-samples", required=True)
    parser.add_argument("--report-out", required=True)
    args = parser.parse_args()
    report = analyze_dependence(
        base_samples=load_json(args.base_samples),
        strict_samples=load_json(args.strict_samples),
        repo_root=Path.cwd(),
    )
    save_json(args.report_out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
