from loguru import logger


class BaseModel:
    def __init__(self):
        logger.info("Building model")

    def build_new(self):
        logger.info("constructing model")

    def train(self, train):
        logger.info("Beginning train")

    def test(self, test):
        logger.info("Testing model. Results saved in {}".format(self.results_path))

    def set_saving_file(self, saving_file):
        self.file = saving_file

    def set_config(self, config):
        self.config = config

    def set_results_path(self, path):
        self.results_path = path

    def load(self):
        logger.info("Loading model at {}".format(self.file))
