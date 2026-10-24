from __future__ import annotations

import argparse

from emerguav_bench.llm.exporters import build_sft_records
from emerguav_bench.llm.io import load_json, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Export EmergUAV-Bench LLM manifest to SFT JSONL.")
    parser.add_argument("--manifest", required=True, help="Input LLM manifest JSON path")
    parser.add_argument("--sft-out", required=True, help="Output SFT JSONL path")
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    save_jsonl(args.sft_out, build_sft_records(manifest))


if __name__ == "__main__":
    main()
