import json

from config_loader.data_source_processor import DataSourceProcessor
from loguru import logger

logger.info("Loading config")

with open("ressources/config.json", "r") as c:
    config = json.load(c)

random = config["randomstate"]
# Defining test and train
process_source = DataSourceProcessor(config["datasource"], random)


train_sequence = process_source.get_training_set()
test_sequence = process_source.get_test_set()

# Loading model
