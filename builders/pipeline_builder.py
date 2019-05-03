from loguru import logger

from utils.files_util import load_class, make_dirs


class PipelineBuilder(object):
    """
    PipelineBuilder parse the configuration file in order to create the pipeline it describe.
    The builder also create a folder for storing experiment results.
    """

    def __init__(self, config):
        logger.info("Building pipeline")

        self.experiment_path = "./experiments/{}".format(config["experiment_name"])
        self.randomstate = config["randomstate"]

        make_dirs(self.experiment_path)

        self.pipeline = {}

        self._init_preprocessing(config["data_processing"])
        self._init_train_test_split(config["train_test_split"])

    def _init_preprocessing(self, config):
        if config is None:
            return

        preprocessing = load_class(config["classpath"])()

        preprocessing.set_input_dir(config["input_dir"])
        preprocessing.set_output_dir(config["output_dir"])

        self.pipeline["preprocessing"] = preprocessing

    def _init_train_test_split(self, config):
        if config is None:
            return

        splitter = load_class(config["splitter_class"])()

        splitter.set_working_dir(config["data_directory"])
        splitter.set_split_size(config["train_split"])
        splitter.set_randomstate(self.randomstate)

        splitter.set_train_class(load_class(config["train_classpath"]))
        splitter.set_train_class(load_class(config["test_classpath"]))

        self.pipeline["train_test_split"] = splitter

    def _init_model(self, config):
        if config is None:
            return

    def get_pipeline(self):
        return self.pipeline
