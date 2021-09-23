from __future__ import annotations

import pathlib
from dataclasses import field
from typing import Any, Dict

from clmbot.util.config import Config
from pydantic import validator
from pydantic.dataclasses import dataclass

from .datasets import TYPES


@dataclass(frozen=True)
class DatasetConfig(Config):
    path: pathlib.Path
    type: str = "text"
    p_train: float = 0.8
    parameters: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    @validator("p_train", always=True)
    def validate_p_train(cls, p_train):
        if p_train < 0.0 or p_train > 1.0:
            raise ValueError(
                f"The split proportion must be in the range [0, 1], but got {p_train}"
            )
        return p_train

    @classmethod
    @validator("type", always=True)
    def validate_type(cls, type_):
        if type_ not in TYPES:
            raise ValueError(
                f'Unsupported dataset type "{type_}", should be one of: {TYPES}'
            )
        return type_


@dataclass(frozen=True)
class TokenizerConfig(Config):
    type: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig(Config):
    type: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainConfig(Config):
    dataset: DatasetConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    block_size: int = None
    encoding_args: Dict[str, Any] = field(default_factory=dict)
    training_args: Dict[str, Any] = field(default_factory=dict)
