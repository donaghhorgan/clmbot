import pathlib

from blindbot.train import Pipeline, TrainConfig


def test_load_from_config():
    path = pathlib.Path(__file__).parent.joinpath("./config.yml")
    config = TrainConfig.from_yaml(path)
    pipeline = Pipeline.from_config(config)

    assert isinstance(pipeline, Pipeline)
    assert str(pipeline.data_loader.path) == "/inputs/data"
    assert pipeline.dataset_shaper.p_train == 0.8
    assert pipeline.encoding_args == {}
    assert getattr(pipeline.tokenizer, "name_or_path") == "gpt2"
    assert getattr(pipeline.model, "name_or_path") == "gpt2"
    assert pipeline.training_args.output_dir == "/outputs/model"
