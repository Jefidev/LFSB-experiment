import numpy as np

import pandas as pd
from data_sequence.image_sequence import ImageSequence
from loguru import logger


class DataSourceProcessor():
    def __init__(self, data, random=None):
        self.base_dir = data["base_dir"]
        self.labels = data["labels"]
        self.train_size = data["train_size"]

        self._generate_set(random)

    def get_training_set(self):
        return self.train_sequence

    def get_test_set(self):
        return self.test_sequence

    def _load_labels(self):
        label_df = pd.read_csv(self.labels)

        if "path" in label_df.columns:
            label_df["path"] = label_df["path"].apply(
                lambda x: "{}/{}".format(self.base_dir, x.split("/")[-1]))

        return label_df

    def _generate_set(self, random):
        data = self._load_labels()

        np.random.seed(random)
        msk = np.random.rand(len(data)) < self.train_size

        train = data[msk]
        test = data[~msk]

        logger.info("Loading data. Train size {}  Test size {}".format(
            len(train), len(test)))

        self._generate_test_train_sequences(train, test)

    def _generate_test_train_sequences(self, train, test):

        self.train_sequence = ImageSequence(train["path"], train["label"], 15)
        self.test_sequence = ImageSequence(test["path"], test["label"], 15)
