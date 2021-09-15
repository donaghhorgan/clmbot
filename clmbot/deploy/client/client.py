from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

import clmbot.deploy.client
from clmbot.util.importlib import get_module_type
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline

from ..config import DeployConfig


@dataclass(frozen=True)
class Client(ABC):
    pipeline: TextGenerationPipeline
    generation_args: Dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def __call__(self):
        pass

    @classmethod
    def from_config(cls, config: DeployConfig) -> Client:
        client = get_module_type(clmbot.deploy.client, config.client.type, Client)
        return client(
            pipeline=TextGenerationPipeline(
                model=AutoModelForCausalLM.from_pretrained(config.model.path),
                tokenizer=AutoTokenizer.from_pretrained(
                    config.tokenizer.type, **config.tokenizer.parameters
                ),
            ),
            generation_args=config.generation_args,
            **config.client.parameters
        )
