import logging
from dataclasses import dataclass
from typing import Any, Dict

from clmbot.train.config import TrainConfig
from clmbot.util.timer import Timer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from .data_loader import DataLoader
from .dataset_shaper import DatasetShaper

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pipeline:
    data_loader: DataLoader
    tokenizer: PreTrainedTokenizer
    encoding_args: Dict[str, Any]
    dataset_shaper: DatasetShaper
    model: PreTrainedModel
    training_args: TrainingArguments

    def __call__(self):
        with Timer() as timer:
            text = self.data_loader()
        logger.info(f"Loaded data in {timer.duration:.2f} seconds")

        with Timer() as timer:
            encodings = self.tokenizer.encode(text, **self.encoding_args)
        logger.info(f"Encoded data in {timer.duration:.2f} seconds")

        with Timer() as timer:
            train_dataset, eval_dataset = self.dataset_shaper(
                encodings, self.tokenizer.model_max_length
            )
        logger.info(f"Shaped dataset in {timer.duration:.2f} seconds")

        with Timer() as timer:
            data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )

            trainer.train()
        logger.info(f"Trained model in {timer.duration:.2f} seconds")

        with Timer() as timer:
            trainer.save_model()
        logger.info(f"Saved model in {timer.duration:.2f} seconds")

    @classmethod
    def from_config(cls, config: TrainConfig):
        return cls(
            data_loader=DataLoader(**config.input.to_dict()),
            tokenizer=AutoTokenizer.from_pretrained(
                config.tokenizer.type, **config.tokenizer.parameters
            ),
            encoding_args=config.encoding_args,
            dataset_shaper=DatasetShaper(**config.dataset.to_dict()),
            model=AutoModelForCausalLM.from_pretrained(
                config.model.type, **config.model.parameters
            ),
            training_args=TrainingArguments(**config.training_args),
        )
