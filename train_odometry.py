import numpy as np
import sys
import tensorflow as tf
import cv2
import argparse
import matplotlib.pyplot as plt

from utils import *
from model.model import *
from train import predict

from tensorflow.keras.losses import mse
from os.path import join
from distutils.util import strtobool


def train(model_name, model_path='data/model', in_frames=8, out_frames=15, normalize=False,
          batch_size=512, epochs=10, evaluate=True, mc_samples=10, use_inverse_data=False, force_load_data=False,
          train_model=True, load_model=False, **model_kwargs):

    model_kwargs = dict(locals(), **model_kwargs)
    del model_kwargs['model_kwargs'] # del duplicate values
    print(model_kwargs)

    #DATA
    #odometry
    odometry_pickle_path_test = 'data/pickles/test_odometry.p'
    odometry_pickle_path_train = 'data/pickles/train_odometry.p'

    if force_load_data:
        odometry_x_test, odometry_y_test = get_odometry('data/tracks/tracks_test.h5', 'data/odometry/test',
                                                        in_frames, out_frames)
        odometry_x_train, odometry_y_train = get_odometry('data/tracks/tracks_train.h5', 'data/odometry/train',
                                                          in_frames, out_frames)

        save_pickle(odometry_pickle_path_test, (odometry_x_test, odometry_y_test))
        save_pickle(odometry_pickle_path_train, (odometry_x_train, odometry_y_train))

    odometry_x_test, odometry_y_test = load_pickle(odometry_pickle_path_test)
    odometry_x_train, odometry_y_train = load_pickle(odometry_pickle_path_train)

    #concatenate odometry with itself because inverse bbs were added
    if use_inverse_data:
        inverse_odometry_x_train = np.copy(odometry_x_train)
        inverse_odometry_x_train[:, :, 1] = inverse_odometry_x_train[:, :, 1] * -1 #iversion of steering angle
        inverse_odometry_y_train = np.copy(odometry_y_train)
        inverse_odometry_y_train[:, :, 1] = inverse_odometry_y_train[:, :, 1] * -1

        odometry_x_train = np.concatenate([odometry_x_train, inverse_odometry_x_train], axis=0)
        odometry_y_train = np.concatenate([odometry_y_train, inverse_odometry_y_train], axis=0)

    print('[+] Odometry train shapes x: %s y: %s' % (odometry_x_train.shape, odometry_y_train.shape))

    if normalize:
        # norm each speed and steer angle separately
        speed_mean = np.asarray([odometry_x_test[:, :, 0].mean(), odometry_x_train[:, :, 0].mean()]).mean()
        speed_std = np.asarray([odometry_x_test[:, :, 0].std(), odometry_x_train[:, :, 0].std()]).mean()
        odometry_x_test[:, :, 0] = standardization(odometry_x_test[:, :, 0], speed_mean, speed_std)
        odometry_x_train[:, :, 0] = standardization(odometry_x_train[:, :, 0], speed_mean, speed_std)
        odometry_y_test[:, :, 0] = standardization(odometry_y_test[:, :, 0], speed_mean, speed_std)
        odometry_y_train[:, :, 0] = standardization(odometry_y_train[:, :, 0], speed_mean, speed_std)

        steer_mean = np.asarray([odometry_x_test[:, :, 1].mean(), odometry_x_train[:, :, 1].mean()]).mean()
        steer_std = np.asarray([odometry_x_test[:, :, 1].std(), odometry_x_train[:, :, 1].std()]).mean()
        odometry_x_test[:, :, 1] = standardization(odometry_x_test[:, :, 1], steer_mean, steer_std)
        odometry_x_train[:, :, 1] = standardization(odometry_x_train[:, :, 1], steer_mean, steer_std)
        odometry_y_test[:, :, 1] = standardization(odometry_y_test[:, :, 1], steer_mean, steer_std)
        odometry_y_train[:, :, 1] = standardization(odometry_y_train[:, :, 1], steer_mean, steer_std)

    model_kwargs['model_input_shape'] = odometry_x_test.shape[1:]
    model_kwargs['model_output_dim'] = odometry_y_test.shape[-1]

    model = LstmPredictor.load_model(join(model_path, model_name)) if load_model else build_model(**model_kwargs)
    if train_model:
        model.fit(odometry_x_train, odometry_y_train, batch_size=batch_size, epochs=epochs, shuffle=True)

        # saving weights and model_kwargs separately
        model.save_model()

    if evaluate:
        epistemic = None
        pred, log_var = predict(model, odometry_x_test, predict_var=model.predict_variance, mc_samples=mc_samples)

        if log_var is not None:
            aletoric = log_var.mean(axis=0)

        if mc_samples > 1:
            epistemic = pred.var(axis=0)

        pred = pred.mean(axis=0)

        mse = np.square(pred - odometry_y_test).mean()
        print('MSE: %f Aletoric unc.: %s Epistemic unc: %s' % (mse,
                                                               str(aletoric.mean()) if aletoric is not None else '-',
                                                               str(epistemic.mean()) if epistemic is not None else '-'))

    return model


def get_kwargs_from_cli():

    def get_fn(loss_fn):
        return globals()[loss_fn]

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='model')
    parser.add_argument('--model_path', type=str, default='data/model_odometry')
    parser.add_argument('--num_prediction_steps', type=int, default=15)
    parser.add_argument('--in_frames', type=int, default=8)
    parser.add_argument('--out_frames', type=int, default=15)
    parser.add_argument('--weight_dropout', type=float, default=0.)
    parser.add_argument('--unit_dropout', type=float, default=0.)
    parser.add_argument('--lam', type=float, default=0.0001)
    parser.add_argument('--predict_variance', type=strtobool, default=False, choices=[True, False])
    parser.add_argument('--use_mc_dropout', type=strtobool, default=False, choices=[True, False])
    parser.add_argument('--mc_samples', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_units', type=int, default=256)
    parser.add_argument('--normalize', type=strtobool, default=True, choices=[True, False])
    parser.add_argument('--loss_fn', type=get_fn, default=mse,
                        choices=[mse, heteroskedastic_loss_v2, heteroskedastic_loss])
    parser.add_argument('--diff_fn', type=get_fn, default=get_diff_array, choices=[get_diff_array])
    parser.add_argument('--evaluate', type=strtobool, default=True, choices=[True, False])
    parser.add_argument('--train_model', type=strtobool, default=True, choices=[True, False])
    parser.add_argument('--load_model', type=strtobool, default=False, choices=[True, False])
    parser.add_argument('--shuffle', type=strtobool, default=True, choices=[True, False])
    parser.add_argument('--force_load_data', type=strtobool, default=False, choices=[True, False])
    args = parser.parse_args()

    kwargs = dict(args._get_kwargs())

    print('[+] Odometry model parameters')
    for k, v in kwargs.items():
        print(k + " -> " + str(v))

    return kwargs


if __name__ == '__main__':
    np.random.seed(1444)  # random seed for train data shuffling

    kwargs = dict(arg.split('=') for arg in sys.argv[1:])
    kwargs = get_kwargs_from_cli(kwargs)
    model = train(**kwargs)