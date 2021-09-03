from __future__ import annotations

import pathlib

import yaml
from pydantic.dataclasses import dataclass

from .stage import (
    EncoderConfig,
    LoaderConfig,
    SaverConfig,
    SplitterConfig,
    TrainerConfig,
)


@dataclass(frozen=True)
class Config:
    input: LoaderConfig
    split: SplitterConfig
    encoder: EncoderConfig
    trainer: TrainerConfig
    output: SaverConfig

    @classmethod
    def from_yaml(cls, file: pathlib.Path) -> Config:
        with file.open() as fp:
            config = yaml.safe_load(fp)

        if not config:
            raise ValueError(f"The configuration in {file} is empty")

        return cls(**config)
