import cv2
import numpy as np
from keras.utils import Sequence


class TripletSequence(Sequence):
    def __init__(self, df, batch_size, resize=None):
        self.data = df

        self.batch = batch_size
        self.resize = resize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pass

    def _get_positive(self):
        pass

    def _get_negative(self):
        pass

    def _construct_triplet(self):
        pass
