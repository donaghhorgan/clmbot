from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import List, Tuple, TypeVar

from pydantic import validator
from pydantic.dataclasses import dataclass

T = TypeVar("T")


def split(data: List[T], ratio: float) -> Tuple[List[T], List[T]]:
    n_data = len(data)

    if ratio < 0.5:
        n_split = math.floor(ratio * n_data)
    else:
        n_split = math.ceil(ratio * n_data)

    if n_split == n_data:
        n_split -= 1
    elif n_split == 0:
        n_split += 1

    indices = list(range(n_data))
    random.shuffle(indices)

    a = [data[i] for i in indices[:n_split]]
    b = [data[i] for i in indices[n_split:]]

    return a, b


@dataclass(frozen=True)
class SplitterConfig:
    ratio: float


@dataclass(frozen=True)
class Splitter(Callable[List[T], Tuple[List[T], List[T]]]):
    ratio: float

    def __call__(self, data: List[T] = None) -> Tuple[List[T], List[T]]:
        return split(data, self.ratio)

    @classmethod
    def from_config(cls, config: SplitterConfig) -> Splitter:
        return Splitter(ratio=config.ratio)

    @validator("ratio", always=True)
    def validate_ratio(cls, ratio):
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(
                f"Split ratio must be in the range [0, 1], but got {ratio}"
            )
        return ratio
