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

        self._init_preprocessing(config.get("data_processing"))
        self._init_train_test_split(config.get("train_test_split"))
        self._init_model(config.get("model"))
        self._init_result_analysis(config.get("result_analysis"))

    def _init_preprocessing(self, config):
        if config is None:
            return

        preprocessing = load_class(config["classpath"])()

        preprocessing.set_input_dir(config["input_dir"])
        preprocessing.set_output_dir(config["output_dir"])
        preprocessing.set_config(config)

        self.pipeline["preprocessing"] = preprocessing

    def _init_train_test_split(self, config):
        if config is None:
            return

        splitter = load_class(config["splitter_class"])()

        splitter.set_working_dir(config["data_directory"])
        splitter.set_split_size(config["train_split"])
        splitter.set_randomstate(self.randomstate)
        splitter.set_other_config(config)

        splitter.set_train_class(load_class(config["train_classpath"]))
        splitter.set_train_class(load_class(config["test_classpath"]))

        self.pipeline["train_test_split"] = splitter

    def _init_model(self, config):
        if config is None:
            return

        model_path = "{}/{}".format(self.experiment_path, "models")
        make_dirs(model_path)
        model_file = "{}/{}.h5".format(model_path, config["name"])

        results_path = "{}/{}".format(self.experiment_path, "results")

        model = load_class(config["classpath"])()
        model.set_saving_file(model_file)
        model.set_results_path(results_path)
        model.set_config(config)

        if config["train"]:
            self.pipeline["train"] = model

        if config["test"]:
            self.pipeline["test"] = model

    def _init_result_analysis(self, config):
        if config is None:
            return

        result_folder = "{}/{}".format(self.experiment_path, "analysis")
        make_dirs(result_folder)

        analyser = load_class(config["classpath"])()

        analyser.set_save_path(result_folder)
        analyser.set_data_path(config["data_path"])
        analyser.set_config(config)

        self.pipeline["result_analysis"] = analyser

    def get_pipeline(self):
        return self.pipeline
