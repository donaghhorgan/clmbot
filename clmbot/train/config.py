from __future__ import annotations

import pathlib
from dataclasses import field
from typing import Any, Dict

from clmbot.util.config import Config
from pydantic import validator
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class DataLoaderConfig(Config):
    path: pathlib.Path
    pattern: str = "*.txt"
    sep: str = "\n"


@dataclass(frozen=True)
class TokenizerConfig(Config):
    type: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetConfig(Config):
    p_train: float

    @classmethod
    @validator("p_train", always=True)
    def validate_p_train(cls, p_train):
        if p_train < 0.0 or p_train > 1.0:
            raise ValueError(
                f"The split proportion must be in the range [0, 1], but got {p_train}"
            )
        return p_train


@dataclass(frozen=True)
class ModelConfig(Config):
    type: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainConfig(Config):
    input: DataLoaderConfig
    tokenizer: TokenizerConfig
    dataset: DatasetConfig
    model: ModelConfig
    encoding_args: Dict[str, Any] = field(default_factory=dict)
    training_args: Dict[str, Any] = field(default_factory=dict)
