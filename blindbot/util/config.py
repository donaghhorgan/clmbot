from __future__ import annotations

import dataclasses
import pathlib
from abc import ABC
from typing import Any, Dict

import yaml
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class Config(ABC):
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_yaml(cls, file: pathlib.Path) -> Config:
        with file.open() as fp:
            config = yaml.safe_load(fp)

        if not config:
            raise ValueError(f"The configuration in {file} is empty")

        return cls(**config)
