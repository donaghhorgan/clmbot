import pathlib

from blindbot.train import Config


def test_load_config():
    path = pathlib.Path(__file__).parent.joinpath("./config.yml")
    config = Config.from_yaml(path)
    assert config.tokenizer.type == "gpt2"
