from loguru import logger

from data_sequence.image_sequence import ImageSequence
from data_sequence.triplet_sequence import TripletSequence
from train_test_split.base_splitter import BaseSplitter
import pandas as pd
import numpy as np

class TripletLossSplitter(BaseSplitter):
    def __init__(self):
        BaseSplitter.__init__(self)

    def run(self):
        self._generate_set()

    def _load_labels(self):
        label_file = self.working_dir + "/labels.csv"
        label_df = self._load_sign_dataframe(label_file)

        if "path" in label_df.columns:
            label_df["path"] = label_df["path"].apply(
                lambda x: "{}/{}".format(self.working_dir, x.split("/")[-1])
            )

        return label_df

    def _generate_set(self):
        data = self._load_labels()

        np.random.seed(self.seed)
        msk = np.random.rand(len(data)) < self.split_size

        train = data[msk]
        test = data[~msk]

        logger.info(
            "Loading data. Train size {}  Test size {}".format(len(train), len(test))
        )

        self._generate_test_train_sequences(train, test)

    def _generate_test_train_sequences(self, train, test):
        batch = self.other_config["batch_size"]
        scale = self.other_config["image_scale"]

        self.train = TripletSequence(train, batch, resize=(scale, scale))
        self.test = ImageSequence(
            test["path"], test["label"], batch, resize=(scale, scale)
        )

    def _load_sign_dataframe(self, path):
        data = pd.read_csv(path)

        # Filtering descriptive sign + undefined signs
        data = data[data["label"].isin(["1", "2", "3", "4", "5"])]

        if self.other_config.get("balance"):
            logger.info("Balancing data")
            g = data.groupby("label")
            data = pd.DataFrame(
                g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
            )

        return data
