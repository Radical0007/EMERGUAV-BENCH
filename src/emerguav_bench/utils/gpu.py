"""GPU allocation helpers for multi-GPU inference without NVLink."""

from __future__ import annotations

import gc
import math
import os
import re
from typing import Any

OOM_SKIP_RETURN_CODE = 75


def _lazy_import_torch() -> Any:
    import torch

    return torch


def gpu_count() -> int:
    try:
        torch = _lazy_import_torch()
    except ImportError:
        return 0
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def detect_visible_cuda_vram_gb(*, fallback_gb: int = 40) -> int:
    try:
        torch = _lazy_import_torch()
    except ImportError:
        return fallback_gb
    try:
        if not torch.cuda.is_available():
            return fallback_gb
        device_count = torch.cuda.device_count()
        if device_count <= 0:
            return fallback_gb
        max_bytes = max(int(torch.cuda.get_device_properties(index).total_memory) for index in range(device_count))
    except Exception:
        return fallback_gb
    return max(1, math.ceil(max_bytes / 1024**3))


def estimate_model_billions(model_id: str) -> float | None:
    decimal_matches: list[str] = re.findall(r"(\d+_\d+)[bB]", model_id)
    whole_matches: list[str] = re.findall(r"(\d+)[bB]", model_id)
    candidates: list[float] = []
    for raw in decimal_matches + whole_matches:
        try:
            candidates.append(float(raw.replace("_", ".")))
        except ValueError:
            pass
    if not candidates:
        return None
    return max(candidates)


def model_fits_single_gpu(model_id: str, *, threshold_b: float = 20.0) -> bool:
    estimated = estimate_model_billions(model_id)
    if estimated is None:
        return True
    return estimated <= threshold_b


def gpu_affinity_env(gpu_id: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def dual_gpu_env() -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    return env


def classify_models(model_ids: list[str]) -> tuple[list[str], list[str]]:
    single: list[str] = []
    dual: list[str] = []
    for model_id in model_ids:
        if model_fits_single_gpu(model_id):
            single.append(model_id)
        else:
            dual.append(model_id)
    return single, dual


def recommended_inference_batch_size(model_id: str, *, vram_gb: int | None = None) -> int:
    resolved_vram_gb = detect_visible_cuda_vram_gb() if vram_gb is None else vram_gb
    estimated = estimate_model_billions(model_id)
    if estimated is None:
        return 1
    if resolved_vram_gb >= 80:
        if estimated <= 3:
            return 16
        if estimated <= 6:
            return 8
        if estimated <= 10:
            return 4
        if estimated <= 20:
            return 2
        return 1
    if resolved_vram_gb >= 40:
        if estimated <= 3:
            return 8
        if estimated <= 6:
            return 4
        if estimated <= 10:
            return 2
        return 1
    if resolved_vram_gb >= 24:
        if estimated <= 3:
            return 4
        if estimated <= 6:
            return 2
        return 1
    if resolved_vram_gb >= 16 and estimated <= 3:
        return 2
    return 1


def is_cuda_out_of_memory_error(exc: BaseException) -> bool:
    if exc.__class__.__name__ in {"OutOfMemoryError", "CudaOutOfMemoryError"}:
        return True
    text = str(exc).lower()
    return "cuda" in text and "out of memory" in text


def clear_cuda_memory() -> None:
    gc.collect()
    try:
        torch = _lazy_import_torch()
    except ImportError:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        return
