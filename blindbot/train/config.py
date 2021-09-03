from __future__ import annotations

import dataclasses
from dataclasses import field
from abc import ABC

import pathlib
from typing import Dict, Any

import yaml
from pydantic.dataclasses import dataclass
from pydantic import validator


@dataclass(frozen=True)
class BaseConfig(ABC):

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class DataLoaderConfig(BaseConfig):
    path: pathlib.Path
    pattern: str = "*.txt"
    sep: str = "\n"


@dataclass(frozen=True)
class TokenizerConfig(BaseConfig):
    type: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetConfig(BaseConfig):
    p_train: float

    @validator("p_train", always=True)
    def validate_p_train(cls, p_train):
        if p_train < 0.0 or p_train > 1.0:
            raise ValueError(
                f"The split proportion must be in the range [0, 1], but got {p_train}"
            )
        return p_train


@dataclass(frozen=True)
class ModelConfig(BaseConfig):
    type: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Config(BaseConfig):
    input: DataLoaderConfig
    tokenizer: TokenizerConfig
    dataset: DatasetConfig
    model: ModelConfig

    encoding_args: Dict[str, Any] = field(default_factory=dict)
    training_args: Dict[str, Any] = field(default_factory=dict)

    @validator('encoding_args', always=True)
    def set_default_encoding_args(cls, encoding_args: Dict[str, Any]):
        encoding_args.setdefault("return_tensors", "pt")
        return encoding_args

    @classmethod
    def from_yaml(cls, file: pathlib.Path) -> Config:
        with file.open() as fp:
            config = yaml.safe_load(fp)

        if not config:
            raise ValueError(f"The configuration in {file} is empty")

        return cls(**config)
