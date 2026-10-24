from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from emerguav_bench.llm.io import load_jsonl, save_json
from emerguav_bench.llm.registry import MODEL_PRESETS
from emerguav_bench.utils.progress import progress_iter


def _check_optional_dependency(module_name: str) -> bool:
    try:
        __import__(module_name)
    except Exception:
        return False
    return True


@dataclass(slots=True)
class SFTTrainingConfig:
    model_id: str
    output_dir: str
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    use_qlora: bool = True
    lora_r: int = 16
    weight_scale: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PreferenceTrainingConfig:
    model_id: str
    output_dir: str
    algorithm: str = "DPO"
    epochs: int = 2
    learning_rate: float = 1e-5
    batch_size: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SFTTrainingWrapper:
    def __init__(self, config: SFTTrainingConfig) -> None:
        if config.model_id not in MODEL_PRESETS:
            raise KeyError(f"Unsupported model_id for SFT: {config.model_id}")
        self.config = config

    def load_records(self, path: str | Path) -> list[dict[str, Any]]:
        rows = load_jsonl(path)
        for row in progress_iter(rows, desc="validate sft rows", total=len(rows), unit="row"):
            required_fields = {"messages", "image_inputs", "target_text", "training_weight"}
            missing = required_fields - set(row)
            if missing:
                raise ValueError(f"SFT row missing required fields: {sorted(missing)}")
        return rows

    def build_weighted_records(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        weighted_rows: list[dict[str, Any]] = []
        for row in progress_iter(rows, desc="weight sft rows", total=len(rows), unit="row"):
            repeat_count = max(1, round(float(row["training_weight"]) * self.config.weight_scale))
            for _ in range(repeat_count):
                weighted_rows.append(dict(row))
        return weighted_rows

    def dry_run(self, path: str | Path) -> dict[str, Any]:
        rows = self.load_records(path)
        weighted_rows = self.build_weighted_records(rows)
        variants: dict[str, int] = {}
        for row in progress_iter(rows, desc="summarize sft rows", total=len(rows), unit="row"):
            variant = row.get("variant", "unknown")
            variants[variant] = variants.get(variant, 0) + 1
        summary = {
            "backend": "transformers+peft+accelerate",
            "model_id": self.config.model_id,
            "record_count": len(rows),
            "weighted_record_count": len(weighted_rows),
            "variants": variants,
            "dependencies": {
                "transformers": _check_optional_dependency("transformers"),
                "accelerate": _check_optional_dependency("accelerate"),
                "peft": _check_optional_dependency("peft"),
                "datasets": _check_optional_dependency("datasets"),
            },
            "config": self.config.to_dict(),
        }
        output_path = Path(self.config.output_dir) / "sft_dry_run_summary.json"
        save_json(output_path, summary)
        return summary


class PreferenceTrainingWrapper:
    def __init__(self, config: PreferenceTrainingConfig) -> None:
        if config.model_id not in MODEL_PRESETS:
            raise KeyError(f"Unsupported model_id for preference training: {config.model_id}")
        normalized_algorithm = config.algorithm.upper()
        if normalized_algorithm not in {"DPO", "ORPO"}:
            raise ValueError("Preference algorithm must be DPO or ORPO")
        config.algorithm = normalized_algorithm
        self.config = config

    def load_pairs(self, path: str | Path) -> list[dict[str, Any]]:
        rows = load_jsonl(path)
        for row in progress_iter(rows, desc="validate preference rows", total=len(rows), unit="row"):
            required_fields = {"prompt_text", "chosen_text", "rejected_text"}
            missing = required_fields - set(row)
            if missing:
                raise ValueError(f"Preference row missing required fields: {sorted(missing)}")
        return rows

    def dry_run(self, path: str | Path) -> dict[str, Any]:
        rows = self.load_pairs(path)
        pair_types: dict[str, int] = {}
        for row in progress_iter(rows, desc="summarize preference rows", total=len(rows), unit="row"):
            pair_type = row.get("pair_type", "unknown")
            pair_types[pair_type] = pair_types.get(pair_type, 0) + 1
        summary = {
            "backend": "trl",
            "algorithm": self.config.algorithm,
            "model_id": self.config.model_id,
            "pair_count": len(rows),
            "pair_types": pair_types,
            "dependencies": {
                "trl": _check_optional_dependency("trl"),
                "transformers": _check_optional_dependency("transformers"),
                "datasets": _check_optional_dependency("datasets"),
            },
            "config": self.config.to_dict(),
        }
        output_path = Path(self.config.output_dir) / "preference_dry_run_summary.json"
        save_json(output_path, summary)
        return summary
