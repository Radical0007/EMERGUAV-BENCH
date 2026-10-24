from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from emerguav_bench.utils.progress import progress_iter


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: Any) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def save_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    row_list = list(rows)
    with target_path.open("w", encoding="utf-8") as handle:
        for row in progress_iter(row_list, desc=f"write {target_path.name}", total=len(row_list), unit="row"):
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
