from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import clmbot.deploy.client
import requests
from clmbot.util.importlib import get_module_type
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline

from ..config import DeployConfig, HfApiConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Client(ABC):
    pipeline: Optional[TextGenerationPipeline]
    api: Optional[HfApiConfig]
    generation_args: Dict[str, Any] = field(default_factory=dict)

    def generate(self, prompt: str = None) -> str:
        try:
            if self.pipeline is not None:
                return self.pipeline(prompt, **self.generation_args)[0][
                    "generated_text"
                ]
            elif self.api is not None:
                if not prompt:
                    return ""

                response = requests.request(
                    "POST",
                    self.api.url,
                    headers={"Authorization": f"Bearer {self.api.token}"},
                    data=json.dumps(
                        {
                            "inputs": prompt,
                            "parameters": self.generation_args,
                            "options": self.api.options,
                        }
                    ),
                )

                response.raise_for_status()

                result = response.json()
                if "error" in result:
                    raise ValueError(f"API error: {result}")

                return result[0]["generated_text"]
            else:
                raise ValueError("No pipeline or API details have been defined")
        except Exception as e:
            logging.exception("Text generation failed")
            return f"Text generation failed: {e}"

    @abstractmethod
    def __call__(self):
        pass

    @classmethod
    def from_config(cls, config: DeployConfig) -> Client:
        client = get_module_type(clmbot.deploy.client, config.client.type, Client)

        if config.model.path is not None:
            model_path = config.model.path.expanduser()
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer,)

            if pipeline.model.config.pad_token_id is None:
                pipeline.model.config.pad_token_id = tokenizer.pad_token_id
        else:
            pipeline = None

        return client(
            pipeline=pipeline,
            api=config.model.api,
            generation_args=config.generation_args,
            **config.client.parameters,
        )
