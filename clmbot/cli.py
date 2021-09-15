import pathlib

import clmbot.logging as logging
import typer
from clmbot.deploy import Client, DeployConfig
from clmbot.train import Pipeline, TrainConfig

cli = typer.Typer()

LOG_LEVEL = typer.Option(
    logging.LogLevel.info,
    "--log-level",
    case_sensitive=False,
    help="Set the logging level.",
)


@cli.command(help="Train a new model.")
def train(
    config_file: pathlib.Path = typer.Option(
        "train.yml",
        "--config",
        "-c",
        help="The configuration file to use.",
        exists=True,
        dir_okay=False,
    ),
    log_level: logging.LogLevel = LOG_LEVEL,
):
    logging.basicConfig(log_level)

    config = TrainConfig.from_yaml(config_file)
    pipeline = Pipeline.from_config(config)

    pipeline()


@cli.command(help="Deploy a model.")
def deploy(
    config_file: pathlib.Path = typer.Option(
        "deploy.yml",
        "--config",
        "-c",
        help="The configuration file to use.",
        exists=True,
        dir_okay=False,
    ),
    log_level: logging.LogLevel = LOG_LEVEL,
):
    logging.basicConfig(log_level)

    config = DeployConfig.from_yaml(config_file)
    client = Client.from_config(config)

    client()
