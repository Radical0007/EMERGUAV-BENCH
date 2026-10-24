from __future__ import annotations

from typing import Any

from emerguav_bench.schemas.output_schema import validate_model_output
from emerguav_bench.schemas.sample_schema import ValidationErrorDetail, validate_sample_record


class SchemaValidationError(ValueError):
    def __init__(self, message: str, errors: list[ValidationErrorDetail]) -> None:
        super().__init__(message)
        self.errors = errors


def _format_errors(errors: list[ValidationErrorDetail]) -> str:
    return "; ".join(f"{error.field}: {error.message}" for error in errors)


def assert_valid_sample_record(sample: dict[str, Any]) -> None:
    errors = validate_sample_record(sample)
    if errors:
        raise SchemaValidationError(f"Invalid sample record: {_format_errors(errors)}", errors)


def assert_valid_model_output(output: dict[str, Any]) -> None:
    errors = validate_model_output(output)
    if errors:
        raise SchemaValidationError(f"Invalid model output: {_format_errors(errors)}", errors)
