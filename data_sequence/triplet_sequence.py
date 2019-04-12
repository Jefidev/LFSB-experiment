import random
from random import shuffle

import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
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
        self.model = None

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch)))

    def __getitem__(self, idx):

        i = 0
        triplets = []
        pos = []
        neg = []
        anch = []

        while i < self.batch:
            # sampling random label
            choosen = self.data.sample(1)
            label = choosen["label"].tolist()[0]

            # Get duet for this label
            duets = self._construct_duet(label, self.batch - i)

            # Get semi hard negative
            anchor, posi, nega = self._construct_triplet(duets, label)

            anch += anchor
            pos += posi
            neg += nega

            i += len(anch)

        X = [np.array(anch), np.array(pos), np.array(neg)]
        y = np.ones(len(anch))

        return X, y

    def _construct_duet(self, label, remaining):
        duet_list = []
        label_data = self.data[self.data["label"] == label]

        paths = label_data["path"].tolist()
        shuffle(paths)

        while len(paths) >= 2:
            a = paths.pop()
            p = paths.pop()

            duet_list.append((a, p))

        return duet_list[:remaining]

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

    def set_model(self, model):
        self.model = model

    def _construct_triplet(self, duets, label):

        anchors = []
        positives = []
        negatives = []

        neg_data = self.data[self.data["label"] != label]
        neg_list = neg_data["path"].tolist()
        shuffle(neg_list)

        for elem in duets:

            img_a = self._load_images(elem[0])
            anchors.append(img_a)

            img_p = self._load_images(elem[1])
            positives.append(img_p)

            if self.model:
                neg, neg_list = self._get_semi_hard([img_a, img_p], neg_list)
                negatives.append(neg)
            else:
                negatives.append(self._load_images(neg_list.pop()))

        return anchors, positives, negatives

    def _get_semi_hard(self, ref, neg_list):

        anch_pos_embed = self.model.predict(np.array(ref))
        a = anch_pos_embed[0]
        p = anch_pos_embed[1]

        for i, neg in enumerate(neg_list):
            img = self._load_images(neg)

            n = self.model.predict(img)[0]

            if self._is_semi_hard(a, p, n):
                neg_list.pop(i)
                return img, neg_list

        logger.info("No semi hard examples found")
        img = self._load_images(neg_list.pop())
        return img, neg_list

    def _is_semi_hard(self, a, p, n):
        a = tf.reshape(a, [-1, 1024])
        p = tf.reshape(p, [-1, 1024])
        n = tf.reshape(n, [-1, 1024])

        p_dist = K.sum(K.square(a - p), axis=-1)
        n_dist = K.sum(K.square(a - n), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + 0.2, 0), axis=0) > 0


def update_model(seq, model):
    embed_input = model.layers[3].get_input_at(-1)
    embed_output = model.layers[3].get_output_at(-1)

    embedding_model = Model(embed_input, embed_output)

    seq.set_model(embedding_model)
