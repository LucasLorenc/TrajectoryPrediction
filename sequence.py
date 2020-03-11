import numpy as np
import math
import cv2

from tensorflow.keras.utils import Sequence


class Sequence(Sequence):
    def __init__(self, data_x, data_y, img_paths=None, batch_size=128):
        self.x, self.y = data_x, data_y
        self.batch_size = batch_size
        self.img_paths = img_paths

    def __len__(self):
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, idx):

        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.img_paths is not None:
            imgs = self.load_images(idx, self.x.shape[1])
            batch_x = [batch_x, imgs]

        return batch_x, batch_y

    def on_epoch_end(self):
        #do shuffle
        pass

    def load_images(self, idx, img_idx):
        images = []
        end = min((idx + 1) * self.batch_size, self.x.shape[0])
        for i in range(idx * self.batch_size, end):
            sequence = self.img_paths[i]
            img_path = sequence[img_idx]
            img = cv2.imread(img_path)
            images.append(img)

        return np.asarray(images, dtype='uint8')