from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict

import datasets
from datasets import DatasetDict
from pydantic.dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetLoader:
    path: pathlib.Path
    type: str
    p_train: float
    parameters: Dict[str, Any]

    def __call__(self) -> DatasetDict:
        path = str(self.path)
        ds = datasets.load_dataset(path=path, **self.parameters)

        if "validation" not in ds.keys():
            ds["train"] = datasets.load_dataset(
                path=path, split=f"train[:{self.p_train:.0%}]", **self.parameters
            )
            ds["validation"] = datasets.load_dataset(
                path=path, split=f"train[{self.p_train:.0%}:]", **self.parameters
            )

        logger.info(f"Loaded {len(ds)} documents from {self.path}")

        return ds
