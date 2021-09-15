import pathlib

from clmbot.deploy import Client, DeployConfig
from clmbot.deploy.client import CLIClient


def test_load_from_config():
    path = pathlib.Path(__file__).parent.joinpath("./config.yml")
    config = DeployConfig.from_yaml(path)
    client = Client.from_config(config)

    assert isinstance(client, CLIClient)
    assert getattr(client.pipeline.tokenizer, "name_or_path") == "gpt2"
    assert getattr(client.pipeline.model, "name_or_path") == "gpt2"
    assert client.generation_args == {"do_sample": True}
