import random

import cv2
import numpy as np
from keras.utils import Sequence
from loguru import logger


class TripletSequence(Sequence):
    def __init__(self, df, batch_size, resize=None):

        # Keeping only the sign with more than 5 examples
        df["nbr_examples"] = df.apply(
            lambda row: self._get_number_occurence(row, df), axis=1
        )

        logger.info("Dataframe before filtering : {}".format(len(df)))
        df = df[df["nbr_examples"] > 5]
        logger.info("Dataframe after filtering : {}".format(len(df)))

        self.data = df

        self.batch = batch_size
        self.resize = resize

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch)))

    def __getitem__(self, idx):

        i = 0
        triplets = []
        pos = []
        neg = []
        anch = []

        while i < self.batch:
            try:
                choosen = self.data.sample(1)

                path = choosen["path"].tolist()[0]
                label = choosen["label"].tolist()[0]

                anchor_path = path
                negative_path = self._get_negative(label)
                positive_path = self._get_positive(label, path)

                anchor = self._load_images(anchor_path)
                negative = self._load_images(negative_path)
                positive = self._load_images(positive_path)

                pos.append(positive)
                neg.append(negative)
                anch.append(anchor)

                i += 1
            except ValueError as e:
                pass
        X = [np.array(anch), np.array(pos), np.array(neg)]
        y = np.ones(len(anch))

        return X, y

    def _get_positive(self, label, already_taken):
        pos_data = self.data[self.data["label"] == label]
        pos_data = pos_data[pos_data["path"] != already_taken]

        if len(pos_data) <= 1:
            raise ValueError

        pos_list = pos_data["path"].tolist()
        return random.choice(pos_list)

    def _get_negative(self, label):
        neg_data = self.data[self.data["label"] != label]
        neg_list = neg_data["path"].tolist()

        return random.choice(neg_list)

    def _load_images(self, img):
        img = cv2.imread(img) / 255

        if self.resize:
            fx = self.resize[0]
            fy = self.resize[1]

            img = cv2.resize(img, (0, 0), fx=fx, fy=fy)

        return img

    def _get_number_occurence(self, row, df):
        df = df[df["label"] == row["label"]]
        return len(df)
