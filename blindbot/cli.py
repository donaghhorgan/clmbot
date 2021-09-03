import pathlib

import blindbot.logging as logging
import typer
from blindbot.train import Config, Pipeline

cli = typer.Typer()

CONFIG_FILE = typer.Option(
    "config.yml",
    "--config",
    "-c",
    help="The configuration file to use",
    exists=True,
    dir_okay=False,
)

LOG_LEVEL = typer.Option(
    logging.LogLevel.info,
    "--log-level",
    case_sensitive=False,
    help="Set the logging level.",
)


@cli.command(help="Train a new model.")
def train(
    config_file: pathlib.Path = CONFIG_FILE, log_level: logging.LogLevel = LOG_LEVEL,
):
    logging.basicConfig(log_level)

    config = Config.from_yaml(config_file)
    pipeline = Pipeline.from_config(config)

    pipeline()
