from loguru import logger


class BaseSplitter(object):
    def __init__(self):
        logger.info("Building split train")

    def run(self):
        logger.info("Running Splitter")
        self.test = None
        self.train = None

    def get_test(self):
        return self.test

    def get_train(self):
        return self.train

    def set_randomstate(self, rstate):
        self.seed = rstate

    def set_working_dir(self, d):
        self.working_dir = d

    def set_train_class(self, tc):
        self.train_class = tc

    def set_test_class(self, tc):
        self.test_class = tc

    def set_split_size(self, split):
        self.split = split
