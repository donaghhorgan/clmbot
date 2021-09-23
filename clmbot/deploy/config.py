from __future__ import annotations

import pathlib
from dataclasses import field
from typing import Any, Dict, Optional

from clmbot.util.config import Config
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class ClientConfig(Config):
    type: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HfApiConfig(Config):
    url: str
    token: str
    options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.options["use_cache"] = False  # Text generation is non-deterministic


@dataclass(frozen=True)
class ModelConfig(Config):
    path: Optional[pathlib.Path] = None
    api: Optional[HfApiConfig] = None

    def __post_init__(self):
        if self.path is None and self.api is None:
            raise ValueError(
                "Must specify either a local path or HuggingFace API details for model"
            )
        elif self.path is not None and self.api is not None:
            raise ValueError(
                "Cannot specify both a local path and HuggingFace API details for model"
            )


@dataclass(frozen=True)
class DeployConfig(Config):
    client: ClientConfig
    model: ModelConfig
    generation_args: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.generation_args["do_sample"] = True  # Text generation is non-deterministic
