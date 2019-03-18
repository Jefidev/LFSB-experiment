import numpy as np

import cv2
from keras.utils import Sequence


class ImageSequence(Sequence):

    def __init__(self, x, y, batch_size, resize=None):
        self.x = x
        self.y = y

        self.batch = batch_size
        self.resize = resize

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch)))

    def _load_images(self, img):
        img = cv2.imread(img) / 255

        if self.resize:
            fx = self.resize[0]
            fy = self.resize[1]

            img = cv2.resize(img, (0, 0), fx=fx, fy=fy)

        return img

    def __getitem__(self, idx):
        batch_id = self.x[idx * self.batch:(idx + 1) * self.batch]
        batch_y = self.y[idx * self.batch:(idx + 1) * self.batch]

        # Creating images
        X = np.array([self._load_images(id_img) for id_img in batch_id])

        return X, np.array(batch_y)
