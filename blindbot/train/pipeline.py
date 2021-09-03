import logging

from blindbot.util import Timer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .config import Config
from .data_loader import DataLoader
from .dataset_shaper import DatasetShaper

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: Config):
        self.data_loader = DataLoader(**config.input.to_dict())
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer.type, **config.tokenizer.parameters
        )
        self.encoding_args = config.encoding_args
        self.dataset_shaper = DatasetShaper(**config.dataset.to_dict())
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model.type, **config.model.parameters
        )
        self.training_args = TrainingArguments(**config.training_args)

    def __call__(self):
        with Timer() as timer:
            text = self.data_loader()
        logger.info(f"Loaded data in {timer.duration:.2f} seconds")

        with Timer() as timer:
            encodings = self.tokenizer.encode(text, **self.encoding_args).squeeze()
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
