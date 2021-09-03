from __future__ import annotations

from dataclasses import field
from typing import Any, Dict

import torch.utils.data
import transformers
from pydantic.dataclasses import dataclass
from transformers import (
    AutoModelWithLMHead,
    BatchEncoding,
    PreTrainedModel,
    TrainingArguments,
)


@dataclass(frozen=True)
class ModelConfig:
    type: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainerConfig:
    model: ModelConfig
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Trainer:
    model: PreTrainedModel
    training_args: TrainingArguments

    def __call__(
        self, train_encodings: BatchEncoding, eval_encodings: BatchEncoding
    ) -> PreTrainedModel:
        train_dataset = Dataset(train_encodings)
        eval_dataset = Dataset(eval_encodings)

        trainer = transformers.Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        trainer.train()

        return self.model

    @classmethod
    def from_config(cls, config: TrainerConfig) -> Trainer:
        model = AutoModelWithLMHead.from_pretrained(
            config.model.type, **config.model.parameters
        )
        training_args = TrainingArguments(**config.parameters)
        return Trainer(model=model, training_args=training_args)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings: BatchEncoding):
        super(Dataset, self).__init__()
        self.encodings = encodings

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return {key: torch.tensor(val[i]) for key, val in self.encodings.items()}
