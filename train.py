import numpy as np
import os
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from utils import *




# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.7
    epochs_drop = 3
    lrate = initial_lrate * np.power(drop, np.floor((1+epoch)/epochs_drop))
    return lrate


def cross_validation(data_x, data_y):
    from sklearn.model_selection import KFold
    data_train = []
    data_test = []
    kf = KFold(n_splits=3, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(data_x, data_y):
        data_train.append((data_x[train_index], data_y[train_index]))
        data_test.append((data_x[test_index], data_y[test_index]))

    return data_train, data_test


def train():
    in_frames = 8
    out_frames = 15
    batch_size = 512
    min_seq_len = 12
    epochs = 80

    #DATA
    #odometry
    odometry_pickle_path_test = 'data/pickles/test_odometry.p'
    odometry_pickle_path_train = 'data/pickles/train_odometry.p'

    odometry_x_test, odometry_y_test = get_odometry('data/tracks/tracks_test.h5', 'data/odometry/test',
                                                    in_frames, out_frames)
    odometry_x_train, odometry_y_train = get_odometry('data/tracks/tracks_train.h5', 'data/odometry/train',
                                                      in_frames, out_frames)

    save_pickle(odometry_pickle_path_test, (odometry_x_test, odometry_y_test))
    save_pickle(odometry_pickle_path_train, (odometry_x_train, odometry_y_train))

    odometry_x_test, odometry_y_test = load_pickle(odometry_pickle_path_test)
    odometry_x_train, odometry_y_train = load_pickle(odometry_pickle_path_train)
    print('[+] Odometry loaded train shapes x: %s y: %s' % (odometry_x_train.shape, odometry_y_train.shape))


    #loading tracks
    tracks_pickle_path_test = 'data/pickles/test_tracks.p'
    tracks_pickle_path_train = 'data/pickles/train_tracks.p'
    tracks_pickle_path_inverse_train = 'data/pickles/inverse_train_tracks.p'

    test_x, test_y = get_data_set(in_frames, out_frames, 'data/tracks/tracks_test.h5', diff_fn=get_diff_array_v2)
    train_x, train_y = get_data_set(in_frames, out_frames, 'data/tracks/tracks_train.h5', diff_fn=get_diff_array_v2)
    inverse_train_x, inverse_train_y = get_data_set(in_frames, out_frames, 'data/tracks/tracks_train.h5',
                                                    diff_fn=get_diff_array_v2)

    save_pickle(tracks_pickle_path_test, (test_x, test_y))
    save_pickle(tracks_pickle_path_train, (train_x, train_y))
    save_pickle(tracks_pickle_path_inverse_train, (inverse_train_x, inverse_train_y))

    test_x, test_y = load_pickle(tracks_pickle_path_test)
    train_x, train_y = load_pickle(tracks_pickle_path_train)
    inverse_train_x, inverse_train_y = load_pickle(tracks_pickle_path_inverse_train)
    print('[+] Tracks loaded train shapes x: %s y: %s' % (train_x.shape, train_y.shape))

    # concatenate train inverse bbs with normal bbs
    train_x = np.concatenate([train_x, inverse_train_x], axis=0)
    train_y = np.concatenate([train_y, inverse_train_y], axis=0)

    #concatenate odometry with itself because inverse bbs were added
    inverse_odometry_x_train = np.copy(odometry_x_train)
    inverse_odometry_x_train[:, :, 1] = inverse_odometry_x_train[:, :, 1] * -1 #iversion of steering angle
    inverse_odometry_y_train = np.copy(odometry_y_train)
    inverse_odometry_y_train[:, :, 1] = inverse_odometry_y_train[:, :, 1] * -1

    odometry_x_train = np.concatenate([odometry_x_train, inverse_odometry_x_train], axis=0)
    odometry_y_train = np.concatenate([odometry_y_train, inverse_odometry_y_train], axis=0)


    #concatanete bbs with odometry
    test_x = np.concatenate([test_x, odometry_x_test], axis=-1)
    train_x = np.concatenate([train_x, odometry_x_train], axis=-1)


if __name__ == '__main__':
    np.random.seed(1444)  # random seed for train data shuffling
    train()
