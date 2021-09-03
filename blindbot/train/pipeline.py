import logging
from collections.abc import Callable

from blindbot.util import Timer
from pydantic.dataclasses import dataclass

from .config import Config
from .stage import Encoder, Loader, Saver, Splitter, Trainer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pipeline(Callable[[], None]):
    loader: Loader
    splitter: Splitter
    encoder: Encoder
    trainer: Trainer
    saver: Saver

    def __call__(self):
        # Load documents
        with Timer() as timer:
            documents = self.loader()

        logging.info(f"Loaded {len(documents)} in {timer:.2f} seconds")

        # Split text
        with Timer() as timer:
            train_texts, eval_texts = self.splitter(documents)

        logging.info(
            f"Split documents into subsets of size {len(train_texts)} "
            f"and {len(eval_texts)} in {timer:.2f} seconds"
        )

        # Encode text
        with Timer() as timer:
            encodings = self.encoder(train_texts, eval_texts)

        logging.info(f"Encoded text in {timer:.2f} seconds")

        # Train a model
        with Timer() as timer:
            model = self.trainer(*encodings)

        logging.info(f"Trained model in {timer:.2f} seconds")

    @classmethod
    def from_config(cls, config: Config):
        loader = Loader.from_config(config.input)
        splitter = Splitter.from_config(config.split)
        encoder = Encoder.from_config(config.encoder)
        trainer = Trainer.from_config(config.trainer)

        return Pipeline(loader, splitter, encoder, trainer)
