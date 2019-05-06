from loguru import logger


class BaseResutsAnalysis(object):
    def __init__(self):
        logger.info("Building results analysis")

    def set_data_path(self, data):
        self.data = data

    def set_save_path(self, path):
        self.save_path = path

    def run(self):
        logger.info("Running analysis")
