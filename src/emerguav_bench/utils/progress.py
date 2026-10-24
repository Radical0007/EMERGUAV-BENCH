from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


class NullProgressBar:
    def __init__(
        self,
        *,
        total: int | None = None,
        desc: str | None = None,
        unit: str | None = None,
        bar_format: str | None = None,
    ) -> None:
        self.total = total
        self.desc = desc
        self.unit = unit
        self.bar_format = bar_format

    def __enter__(self) -> "NullProgressBar":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return False

    def update(self, amount: int = 1) -> None:
        del amount

    def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
        del refresh
        self.desc = desc

    def set_postfix(self, ordered_dict: object | None = None, refresh: bool = True, **kwargs: object) -> None:
        del ordered_dict, refresh, kwargs


def _progress_disabled() -> bool:
    return os.environ.get("EMERGUAV_DISABLE_TQDM", "").strip().lower() in {"1", "true", "yes", "on"}


def _load_tqdm():
    if _progress_disabled():
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm


def progress_iter(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    unit: str | None = None,
) -> Iterable[T]:
    tqdm = _load_tqdm()
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit=unit, dynamic_ncols=True, position=0, file=sys.stderr)


def make_progress_bar(
    *,
    total: int | None,
    desc: str,
    unit: str | None = None,
    bar_format: str | None = None,
):
    tqdm = _load_tqdm()
    if tqdm is None:
        return NullProgressBar(total=total, desc=desc, unit=unit, bar_format=bar_format)
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True, position=0, file=sys.stderr, bar_format=bar_format)


def iter_with_progress(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    unit: str | None = None,
) -> Iterator[T]:
    yield from progress_iter(iterable, desc=desc, total=total, unit=unit)
