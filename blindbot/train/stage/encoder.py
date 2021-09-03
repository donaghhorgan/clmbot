from __future__ import annotations

from dataclasses import field
from typing import Any, Dict, List, Tuple

from pydantic.dataclasses import dataclass
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizer


@dataclass(frozen=True)
class TokenizerConfig:
    type: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EncoderConfig:
    tokenizer: TokenizerConfig
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Encoder:
    tokenizer: PreTrainedTokenizer
    parameters: Dict[str, Any]

    def __call__(
        self, train_texts: List[str], eval_texts: List[str]
    ) -> Tuple[BatchEncoding, BatchEncoding]:
        train_encodings = self.tokenizer(train_texts, **self.parameters)
        eval_encodings = self.tokenizer(eval_texts, **self.parameters)
        return train_encodings, eval_encodings

    @classmethod
    def from_config(cls, config: EncoderConfig) -> Encoder:
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer.type, **config.tokenizer.parameters
        )
        return Encoder(tokenizer=tokenizer, parameters=config.parameters)
