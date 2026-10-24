from __future__ import annotations

import argparse

from emerguav_bench.llm.training import SFTTrainingConfig, SFTTrainingWrapper


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or dry-run EmergUAV-Bench SFT training.")
    parser.add_argument("--train-jsonl", required=True, help="Input SFT JSONL path")
    parser.add_argument("--model-id", required=True, help="Registry model id")
    parser.add_argument("--output-dir", required=True, help="Output directory for training summaries")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--weight-scale", type=int, default=2)
    parser.add_argument("--disable-qlora", action="store_true")
    args = parser.parse_args()

    wrapper = SFTTrainingWrapper(
        SFTTrainingConfig(
            model_id=args.model_id,
            output_dir=args.output_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_qlora=not args.disable_qlora,
            lora_r=args.lora_r,
            weight_scale=args.weight_scale,
        )
    )
    wrapper.dry_run(args.train_jsonl)


if __name__ == "__main__":
    main()
