import json

from loguru import logger

from builders.pipeline_builder import PipelineBuilder

with open("ressources/config.json", "r") as c:
    config = json.load(c)

logger.info("Beginning experiment {}".format(config["experiment_name"]))
pipeline_builder = PipelineBuilder(config)
pipeline = pipeline_builder.get_pipeline()


if pipeline.get("preprocessing"):
    pipeline["preprocessing"].run()


if pipeline.get("train_test_split"):
    pipeline["train_test_split"].run()

    train = pipeline["train_test_split"].get_train()
    test = pipeline["train_test_split"].get_test()

if pipeline.get("train"):
    model = pipeline["train"]
    model.train(train)

if pipeline.get("test"):
    model = pipeline["test"]
    model.test(test)

if pipeline.get("result_analysis"):
    pipeline["result_analysis"].run()
