from __future__ import annotations

import argparse

from emerguav_bench.llm.exporters import build_llm_input_manifest
from emerguav_bench.llm.io import load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Export EmergUAV-Bench samples into an LLM/VLM manifest.")
    parser.add_argument("--samples", required=True, help="Input sample JSON path")
    parser.add_argument("--manifest-out", required=True, help="Output LLM manifest JSON path")
    parser.add_argument("--split-name", default="dev", help="Split name recorded into the manifest")
    parser.add_argument("--instruction-source", choices=["clean", "noisy"], default="noisy")
    parser.add_argument("--context-frames", type=int, default=2, help="Neighbor frame count per side")
    parser.add_argument("--min-image-inputs", type=int, default=3, help="Minimum ordered image inputs per record")
    args = parser.parse_args()

    samples = load_json(args.samples)
    manifest = build_llm_input_manifest(
        samples,
        split_name=args.split_name,
        instruction_source=args.instruction_source,
        context_frames=args.context_frames,
        min_image_inputs=args.min_image_inputs,
    )
    save_json(args.manifest_out, manifest)


if __name__ == "__main__":
    main()
