import pathlib

import clmbot.util.logging as logging
import typer
from clmbot.deploy import Client, ClientConfig, DeployConfig
from clmbot.train import Pipeline, TrainConfig

app = typer.Typer()

TRAIN_CONFIG_FILE = typer.Option(
    "train.yml",
    "--config",
    "-c",
    help="The configuration file to use.",
    exists=True,
    dir_okay=False,
)

DEPLOY_CONFIG_FILE = typer.Option(
    "deploy.yml",
    "--config",
    "-c",
    help="The configuration file to use.",
    exists=True,
    dir_okay=False,
)

LOG_LEVEL = typer.Option(
    logging.LogLevel.info,
    "--log-level",
    case_sensitive=False,
    help="Set the logging level.",
)


@app.command(help="Train a model.")
def train(
    config_file: pathlib.Path = TRAIN_CONFIG_FILE,
    log_level: logging.LogLevel = LOG_LEVEL,
):
    logging.basicConfig(log_level)

    config = TrainConfig.from_yaml(config_file)
    pipeline = Pipeline.from_config(config)

    pipeline()


@app.command(help="Deploy a model.")
def deploy(
    config_file: pathlib.Path = DEPLOY_CONFIG_FILE,
    log_level: logging.LogLevel = LOG_LEVEL,
):
    logging.basicConfig(log_level)

    config = DeployConfig.from_yaml(config_file)
    client = Client.from_config(config)

    client()


@app.command(help="Deploy a model CLI.")
def cli(
    config_file: pathlib.Path = DEPLOY_CONFIG_FILE,
    log_level: logging.LogLevel = LOG_LEVEL,
):
    logging.basicConfig(log_level)

    config = DeployConfig.from_yaml(config_file)

    # Patch the config, replacing the client
    object.__setattr__(config, "client", ClientConfig(type="cli"))

    client = Client.from_config(config)

    client()
