from __future__ import annotations

import argparse
import json
from pathlib import Path

from emerguav_bench.llm.ablations import (
    select_stratified_stage1_records,
    sha256_file,
    summarize_stage1_coverage,
)
from emerguav_bench.llm.io import load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a deterministic, source-anchor-distinct Stage 1 ablation subset."
    )
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--minimum-per-stratum", type=int, default=2)
    args = parser.parse_args()

    source_path = Path(args.source_manifest)
    records = load_json(source_path)
    if not isinstance(records, list):
        raise ValueError("Source manifest must be a JSON array")
    selected = select_stratified_stage1_records(
        records,
        minimum_per_stratum=args.minimum_per_stratum,
    )
    output_path = Path(args.output_manifest)
    save_json(output_path, selected)
    summary = {
        "source_manifest": str(source_path),
        "source_manifest_sha256": sha256_file(source_path),
        "selection_method": "deterministic_greedy_multicover_distinct_source_anchors",
        "minimum_per_stratum": args.minimum_per_stratum,
        "selected_manifest": str(output_path),
        "selected_manifest_sha256": sha256_file(output_path),
        "sample_ids": [record["sample_id"] for record in selected],
        **summarize_stage1_coverage(selected),
    }
    save_json(Path(args.summary), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
