from loguru import logger


class BaseModel:
    def __init__(self):
        logger.info("Building model")

    def construct(self):
        logger.info("constructing model")
        

    def train(self, train):
        logger.info("Beginning train")

    def test(self, test):
        logger.info("test model")
