from __future__ import annotations

import pathlib
from dataclasses import field
from typing import Any, Dict

from clmbot.util.config import Config
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class ClientConfig(Config):
    type: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenizerConfig(Config):
    path: pathlib.Path


@dataclass(frozen=True)
class ModelConfig(Config):
    path: pathlib.Path


@dataclass(frozen=True)
class DeployConfig(Config):
    client: ClientConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    generation_args: Dict[str, Any] = field(default_factory=dict)
