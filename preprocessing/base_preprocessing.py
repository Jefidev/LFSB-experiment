from loguru import logger

from utils.files_util import make_dirs


class BasePreprocessing(object):
    def __init__(self):
        logger.info("Building preprocessing")

    def set_input_dir(self, path):
        self.input_dir = path

    def set_output_dir(self, path):
        make_dirs(path)
        self.output_dir = path

    def set_config(self, config):
        self.config = config

    def run(self):
        logger.info("Executing preprocessing")
