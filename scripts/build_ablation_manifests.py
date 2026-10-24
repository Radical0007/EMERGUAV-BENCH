from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from emerguav_bench.llm.ablations import (
    ABLATION_CONDITIONS,
    ABLATION_CONTRACT_VERSION,
    build_ablation_manifest,
    build_protocol_counterfactual_manifest,
    prompt_leakage_audit,
    sha256_file,
    stable_json_sha256,
)
from emerguav_bench.llm.io import load_json, save_json


def build_and_write_manifests(
    *,
    source_manifest_path: str | Path,
    output_root: str | Path,
    conditions: list[str] | None = None,
    max_records: int | None = None,
) -> dict[str, Any]:
    source_path = Path(source_manifest_path)
    output_path = Path(output_root)
    source_records = load_json(source_path)
    if not isinstance(source_records, list):
        raise ValueError("Source manifest must be a JSON array")
    if max_records is not None:
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        source_records = source_records[:max_records]

    selected_conditions = conditions or [*ABLATION_CONDITIONS, "protocol_counterfactual"]
    unknown = sorted(set(selected_conditions) - {*ABLATION_CONDITIONS, "protocol_counterfactual"})
    if unknown:
        raise ValueError(f"Unknown conditions: {', '.join(unknown)}")

    output_path.mkdir(parents=True, exist_ok=True)
    evaluation_samples: list[dict[str, Any]] = []
    for record in source_records:
        sample = record.get("metadata", {}).get("sample_record")
        if not isinstance(sample, dict):
            raise ValueError(f"Manifest record {record.get('sample_id')} is missing metadata.sample_record")
        if sample.get("sample_id") != record.get("sample_id"):
            raise ValueError(f"Manifest/sample id mismatch for {record.get('sample_id')}")
        evaluation_samples.append(copy.deepcopy(sample))
    evaluation_samples_path = output_path / "evaluation_samples.json"
    save_json(evaluation_samples_path, evaluation_samples)
    summary: dict[str, Any] = {
        "contract_version": ABLATION_CONTRACT_VERSION,
        "source_manifest": str(source_path),
        "source_manifest_sha256": sha256_file(source_path),
        "source_record_count": len(source_records),
        "evaluation_samples": str(evaluation_samples_path),
        "evaluation_samples_sha256": sha256_file(evaluation_samples_path),
        "evaluation_sample_count": len(evaluation_samples),
        "conditions": {},
    }

    for condition in selected_conditions:
        if condition == "protocol_counterfactual":
            rows, pairs = build_protocol_counterfactual_manifest(source_records)
            pairs_path = output_path / "protocol_counterfactual_pairs.json"
            save_json(pairs_path, pairs)
        else:
            rows = build_ablation_manifest(source_records, condition=condition)
            pairs = None

        audit = prompt_leakage_audit(rows)
        if not audit["passed"]:
            raise ValueError(f"Prompt leakage audit failed for {condition}: {audit['violations'][:3]}")
        manifest_path = output_path / f"llm_manifest_{condition}.json"
        save_json(manifest_path, rows)
        condition_summary: dict[str, Any] = {
            "record_count": len(rows),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "stable_payload_sha256": stable_json_sha256(rows),
            "prompt_leakage_audit": audit,
        }
        if pairs is not None:
            condition_summary["pair_count"] = len(pairs)
            condition_summary["pairs"] = str(pairs_path)
            condition_summary["pairs_sha256"] = sha256_file(pairs_path)
        summary["conditions"][condition] = condition_summary

    summary_path = output_path / "manifest_build_summary.json"
    save_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build immutable EmergUAV-Bench multimodal-ablation manifests.")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--conditions",
        default=",".join([*ABLATION_CONDITIONS, "protocol_counterfactual"]),
        help="Comma-separated conditions",
    )
    parser.add_argument("--max-records", type=int, help="Smoke-test prefix only; omit for the complete strict manifest")
    args = parser.parse_args()
    selected = [item.strip() for item in args.conditions.split(",") if item.strip()]
    summary = build_and_write_manifests(
        source_manifest_path=args.source_manifest,
        output_root=args.output_root,
        conditions=selected,
        max_records=args.max_records,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
