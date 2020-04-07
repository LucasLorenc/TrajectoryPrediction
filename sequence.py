import numpy as np
import math
import cv2

from tensorflow.keras.utils import Sequence
from operator import itemgetter


class Sequence(Sequence):
    def __init__(self, data_x, data_y, img_paths=None, batch_size=128, shuffle=True):
        self.x, self.y = data_x, data_y
        self.batch_size = batch_size
        self.img_paths = img_paths

        self.shuffle = shuffle
        self.perm = None
        if shuffle:
            self.shuffle_dataset()

    def __len__(self):
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, idx):

        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.img_paths is not None:
            imgs = self.load_images(idx)
            batch_x = [batch_x, imgs]

        return batch_x, batch_y

    def on_epoch_end(self):
        #do shuffling
        if self.shuffle:
            self.shuffle_dataset()

    def load_images(self, idx):
        images = []
        end = min((idx + 1) * self.batch_size, self.x.shape[0])
        for i in range(idx * self.batch_size, end):
            sequence = self.img_paths[i]
            img_path = sequence[self.x.shape[1]]
            img = cv2.imread(img_path)
            images.append(img)

        images = np.asarray(images) / 127.5
        images -= 1
        return images.astype(dtype='float32')

    def shuffle_dataset(self):
        self.perm = perm = np.random.permutation(self.x.shape[0])
        self.x = self.x[perm]
        self.y = self.y[perm]
        if self.img_paths is not None:
            self.img_paths = [self.img_paths[i].copy() for i in list(perm)]