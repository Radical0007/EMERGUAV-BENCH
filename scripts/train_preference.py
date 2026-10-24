from __future__ import annotations

import argparse

from emerguav_bench.llm.training import PreferenceTrainingConfig, PreferenceTrainingWrapper


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or dry-run EmergUAV-Bench preference training.")
    parser.add_argument("--preference-jsonl", required=True, help="Input preference JSONL path")
    parser.add_argument("--model-id", required=True, help="Registry model id")
    parser.add_argument("--output-dir", required=True, help="Output directory for training summaries")
    parser.add_argument("--algorithm", choices=["DPO", "ORPO", "dpo", "orpo"], default="DPO")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    wrapper = PreferenceTrainingWrapper(
        PreferenceTrainingConfig(
            model_id=args.model_id,
            output_dir=args.output_dir,
            algorithm=args.algorithm,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )
    )
    wrapper.dry_run(args.preference_jsonl)


if __name__ == "__main__":
    main()
