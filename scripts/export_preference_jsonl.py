from __future__ import annotations

import argparse

from emerguav_bench.llm.exporters import build_preference_records
from emerguav_bench.llm.io import load_json, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Export EmergUAV-Bench LLM manifest to preference-pair JSONL.")
    parser.add_argument("--manifest", required=True, help="Input LLM manifest JSON path")
    parser.add_argument("--preference-out", required=True, help="Output preference JSONL path")
    parser.add_argument("--hardest-fraction", type=float, default=0.3, help="Fraction of hardest records to retain")
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    save_jsonl(args.preference_out, build_preference_records(manifest, hardest_fraction=args.hardest_fraction))


if __name__ == "__main__":
    main()
