import numpy as np
import math
import cv2

from tensorflow.keras.utils import Sequence
from operator import itemgetter


class Sequence(Sequence):
    def __init__(self, data_x, data_y, dec_odometry=None, batch_size=128, shuffle=True):
        self.x, self.y = data_x, data_y
        self.batch_size = batch_size
        self.dec_odometry = dec_odometry

        self.shuffle = shuffle
        self.perm = None
        if shuffle:
            self.shuffle_dataset()

    def __len__(self):
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, idx):

        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.dec_odometry is not None:
            batch_dec_odometry = self.dec_odometry[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_x = [batch_x, batch_dec_odometry]

        return batch_x, batch_y

    def on_epoch_end(self):
        #do shuffling
        if self.shuffle:
            self.shuffle_dataset()

    def shuffle_dataset(self):
        self.perm = perm = np.random.permutation(self.x.shape[0])
        self.x = self.x[perm]
        self.y = self.y[perm]
        if self.dec_odometry is not None:
            self.dec_odometry = self.dec_odometry[perm]
