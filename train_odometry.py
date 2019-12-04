import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from utils import *
from model.model import build_model, heteroskedastic_loss, heteroskedastic_loss_v2
from train import predict

from tensorflow.keras.losses import mse
from distutils.util import strtobool


def train(model_name, model_path='data/model', in_frames=8, out_frames=15, normalize=False,
          batch_size=512, epochs=10, evaluate=True, mc_samples=10, **model_kwargs):

    model_kwargs = dict(locals(), **model_kwargs)
    del model_kwargs['model_kwargs'] # del duplicate values
    print(model_kwargs)

    #DATA
    #odometry
    odometry_pickle_path_test = 'data/pickles/test_odometry.p'
    odometry_pickle_path_train = 'data/pickles/train_odometry.p'

    # odometry_x_test, odometry_y_test = get_odometry('data/tracks/tracks_test.h5', 'data/odometry/test',
    #                                                 in_frames, out_frames)
    # odometry_x_train, odometry_y_train = get_odometry('data/tracks/tracks_train.h5', 'data/odometry/train',
    #                                                   in_frames, out_frames)
    #
    # save_pickle(odometry_pickle_path_test, (odometry_x_test, odometry_y_test))
    # save_pickle(odometry_pickle_path_train, (odometry_x_train, odometry_y_train))

    odometry_x_test, odometry_y_test = load_pickle(odometry_pickle_path_test)
    odometry_x_train, odometry_y_train = load_pickle(odometry_pickle_path_train)


    #concatenate odometry with itself because inverse bbs were added
    inverse_odometry_x_train = np.copy(odometry_x_train)
    inverse_odometry_x_train[:, :, 1] = inverse_odometry_x_train[:, :, 1] * -1 #iversion of steering angle
    inverse_odometry_y_train = np.copy(odometry_y_train)
    inverse_odometry_y_train[:, :, 1] = inverse_odometry_y_train[:, :, 1] * -1

    odometry_x_train = np.concatenate([odometry_x_train, inverse_odometry_x_train], axis=0)
    odometry_y_train = np.concatenate([odometry_y_train, inverse_odometry_y_train], axis=0)
    print('[+] Odometry loaded train shapes x: %s y: %s' % (odometry_x_train.shape, odometry_y_train.shape))

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

    model = build_model(**model_kwargs)
    model.fit(odometry_x_train, odometry_y_train, batch_size=batch_size, epochs=epochs, shuffle=True)

    # cant use model save method because of bug https://github.com/tensorflow/tensorflow/issues/34028
    # model.save(os.path.join(base_path, model_name))

    # saving weights and model_kwargs separately
    model.save_model()

    if evaluate:
        pred, aletoric, epistemic = predict(model, odometry_x_test,
                                            predict_var=model_kwargs['predict_variance'],
                                            use_cum_sum=False,
                                            mc_samples=mc_samples)
        if normalize:
            pred[:, :, 1] = inverse_standardization(pred[:, :, 1], steer_mean, steer_std)
            pred[:, :, 0] = inverse_standardization(pred[:, :, 0], speed_mean, speed_std)

            odometry_y_test[:, :, 1] = inverse_standardization(odometry_y_test[:, :, 1], steer_mean, steer_std)
            odometry_y_test[:, :, 0] = inverse_standardization(odometry_y_test[:, :, 0], speed_mean, speed_std)
            if aletoric is not None:
                # scale aletoric uncertainty
                aletoric[:, :, 1] = aletoric[:, :, 1] * steer_std
                aletoric[:, :, 0] = speed_std[:, :, 0] * speed_std
            if epistemic is not None:
                epistemic[:, :, 1] = inverse_standardization(epistemic[:, :, 1], steer_mean, steer_std)
                epistemic[:, :, 0] = inverse_standardization(epistemic[:, :, 0], speed_mean, speed_std)

        mse = np.square(pred - odometry_y_test).mean()
        print('MSE: %f Aletoric unc.: %s Epistemic unc: %s' % (mse,
                                                               str(aletoric.mean()) if aletoric is not None else '-',
                                                               str(epistemic.mean()) if epistemic is not None else '-'))

    return model


def get_kwargs_from_cli(kwargs):
    kwargs['model_name'] = kwargs.get('model_name', 'model')
    kwargs['model_path'] = kwargs.get('model_path', 'data/model_odometry')
    kwargs['num_prediction_steps'] = int(kwargs.get('num_prediction_steps', 15))
    kwargs['weight_dropout'] = float(kwargs.get('weight_dropout', 0.))
    kwargs['unit_dropout'] = float(kwargs.get('unit_dropout', 0.))
    kwargs['lam'] = float(kwargs.get('lam', 0.0001))
    kwargs['predict_variance'] = strtobool(kwargs.get('predict_variance', 'False'))
    kwargs['use_mc_dropout'] = strtobool(kwargs.get('use_mc_dropout', 'False'))
    kwargs['mc_samples'] = int(kwargs.get('mc_samples', 1))
    kwargs['epochs'] = int(kwargs.get('epochs', 10))
    kwargs['batch_size'] = int(kwargs.get('batch_size', 512))
    kwargs['num_units'] = int(kwargs.get('num_units', 256))
    kwargs['normalize'] = strtobool(kwargs.get('normalize', 'True'))
    kwargs['loss_fn'] = kwargs.get('loss_fn', 'mse')
    _losses = ['mse', heteroskedastic_loss.__name__, heteroskedastic_loss_v2.__name__]
    if kwargs['loss_fn'] not in _losses:
        raise ValueError('Unknown loss_fn  use one  these {}'.format(_losses))
    kwargs['loss_fn'] = globals()[kwargs['loss_fn']]
    kwargs['evaluate'] = strtobool(kwargs.get('evaluate', 'True'))

    return kwargs


if __name__ == '__main__':
    np.random.seed(1444)  # random seed for train data shuffling

    kwargs = dict(arg.split('=') for arg in sys.argv[1:])
    kwargs = get_kwargs_from_cli(kwargs)
    model = train(**kwargs)