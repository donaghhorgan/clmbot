from __future__ import annotations

import math
import random
from typing import List, Tuple, TypeVar

import torch.utils.data
from pydantic.dataclasses import dataclass

T = TypeVar("T")


def split(data: List[T], p_split: float) -> Tuple[List[T], List[T]]:
    n_data = len(data)

    if p_split < 0.5:
        n_split = math.floor(p_split * n_data)
    else:
        n_split = math.ceil(p_split * n_data)

    if n_split == n_data:
        n_split -= 1
    elif n_split == 0:
        n_split += 1

    indices = list(range(n_data))
    random.shuffle(indices)

    a = [data[i] for i in indices[:n_split]]
    b = [data[i] for i in indices[n_split:]]

    return a, b


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings: List[List[int]]):
        super(Dataset, self).__init__()
        self.encodings = encodings

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, i) -> List[int]:
        return self.encodings[i]


@dataclass(frozen=True)
class DatasetShaper:
    p_train: float

    def __call__(
        self, encodings: List[int], max_seq_len: int
    ) -> Tuple[Dataset, Dataset]:
        chunks = []
        for i in range(0, len(encodings) - max_seq_len + 1, max_seq_len):
            chunks.append(encodings[i : i + max_seq_len])

        train_encodings, eval_encodings = split(chunks, self.p_train)

        return Dataset(train_encodings), Dataset(eval_encodings)
