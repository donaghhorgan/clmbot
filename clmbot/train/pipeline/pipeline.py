import itertools
import logging
from dataclasses import dataclass
from typing import Any, Dict

from clmbot.train.config import TrainConfig
from clmbot.util.timer import Timer
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from .dataset_loader import DatasetLoader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pipeline:
    dataset_loader: DatasetLoader
    tokenizer: PreTrainedTokenizer
    encoding_args: Dict[str, Any]
    block_size: int
    model: PreTrainedModel
    training_args: TrainingArguments

    def __post_init__(self):
        self.model.resize_token_embeddings(len(self.tokenizer))

    def __call__(self):
        with Timer() as timer:
            dataset = self.dataset_loader()
        logger.info(f"Loaded dataset in {timer.duration:.2f} seconds")

        with Timer() as timer:
            text_col = dataset["train"].column_names[0]
            with self.training_args.main_process_first(desc="Encoding dataset"):
                dataset = dataset.map(
                    lambda batch: self.encode(batch, text_col),
                    batched=True,
                    remove_columns=text_col,
                    desc="Encoding dataset",
                )
        logger.info(f"Encoded dataset in {timer.duration:.2f} seconds")

        with Timer() as timer:
            block_size = self.tokenizer.model_max_length if self.block_size is None else self.block_size
            with self.training_args.main_process_first(desc="Reshaping dataset"):
                dataset = dataset.map(
                    lambda batch: self.reshape(batch, block_size),
                    batched=True,
                    desc="Reshaping dataset",
                )
        logger.info(f"Reshaped dataset in {timer.duration:.2f} seconds")

        with Timer() as timer:
            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["validation"],
                data_collator=default_data_collator,
            )

            result = trainer.train()
        logger.info(f"Trained model in {timer.duration:.2f} seconds")

        with Timer() as timer:
            self.tokenizer.save_pretrained(self.training_args.output_dir)
            trainer.save_model()
        logger.info(f"Saved model in {timer.duration:.2f} seconds")

        with Timer() as timer:
            train_metrics = result.metrics
            trainer.log_metrics("train", train_metrics)
            trainer.save_metrics("train", train_metrics)

            eval_metrics = trainer.evaluate()
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)
        logger.info(f"Saved metrics in {timer.duration:.2f} seconds")

    def encode(self, batch, text_col: str):
        encodings = self.tokenizer(batch[text_col], **self.encoding_args)
        encodings["labels"] = encodings["input_ids"].copy()
        return encodings

    def reshape(self, batch, n_block: int):
        # Concatenate the values in the batch
        concatenated = {k: list(itertools.chain(*v)) for k, v in batch.items()}

        # Compute the length of the concatenated values
        n_total = len(next(iter(concatenated.values())))

        # Reshape the concatenated values to the nearest multiple of n_block
        if n_total > n_block:
            n_total = (n_total // n_block) * n_block

        return {
            k: [v[i : i + n_block] for i in range(0, n_total, n_block)]
            for k, v in concatenated.items()
        }

    @classmethod
    def from_config(cls, config: TrainConfig):
        return cls(
            dataset_loader=DatasetLoader(**config.dataset.to_dict()),
            tokenizer=AutoTokenizer.from_pretrained(
                config.tokenizer.type, **config.tokenizer.parameters
            ),
            encoding_args=config.encoding_args,
            block_size=config.block_size,
            model=AutoModelForCausalLM.from_pretrained(
                config.model.type,
                config=AutoConfig.from_pretrained(config.model.type),
                **config.model.parameters,
            ),
            training_args=TrainingArguments(**config.training_args),
        )
