import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from utils import *
from model.model import build_model, heteroskedastic_loss, save_model, load_model
from tensorflow.keras.losses import mse
from distutils.util import strtobool

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.7
    epochs_drop = 3
    lrate = initial_lrate * np.power(drop, np.floor((1+epoch)/epochs_drop))
    return lrate


def predict(model, test_x, mc_samples=10, predict_var=True, use_cum_sum=True):
    aletoric_unc = None
    epistemic_unc = None

    pred = [model.predict(test_x) for _ in range(mc_samples)]
    pred = np.asarray(pred)

    if predict_var:
        pred, logvar = np.split(pred, 2, axis=-1)
        # aletoric_unc = np.exp(logvar.mean(axis=0))**0.5
        aletoric_unc = logvar.mean(axis=0)

    mean_prediction = np.cumsum(pred.mean(axis=0), axis=1) if use_cum_sum else pred.mean(axis=0)

    if mc_samples > 1:
        epistemic_unc = pred.std(axis=0)

    return mean_prediction, aletoric_unc, epistemic_unc


def cross_validation(data_x, data_y):
    from sklearn.model_selection import KFold
    data_train = []
    data_test = []
    kf = KFold(n_splits=3, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(data_x, data_y):
        data_train.append((data_x[train_index], data_y[train_index]))
        data_test.append((data_x[test_index], data_y[test_index]))

    return data_train, data_test


def train(model_name, model_path='data/model', in_frames=8, out_frames=15, diff_fn=get_diff_array_v2, normalize=False,
          batch_size=512, epochs=10, evaluate=True, mc_samples=10, **model_kwargs):

    model_kwargs = dict(locals(), **model_kwargs)
    del model_kwargs['model_kwargs'] # del duplicate values
    print(model_kwargs)

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

    test_x, test_y = get_data_set(in_frames, out_frames, 'data/tracks/tracks_test.h5', diff_fn=diff_fn)
    train_x, train_y = get_data_set(in_frames, out_frames, 'data/tracks/tracks_train.h5', diff_fn=diff_fn)
    inverse_train_x, inverse_train_y = get_data_set(in_frames, out_frames, 'data/tracks/tracks_train.h5',
                                                    diff_fn=diff_fn, use_inverse_bbs=True)

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

    if normalize:
        tracks_mean = np.asarray([test_x.mean(), train_x.mean(),
                                test_y.mean(), train_y.mean()]).mean()
        tracks_std = np.asarray([test_x.std(), train_x.std(),
                                test_y.std(), train_y.std()]).mean()

        test_x = standardization(test_x, tracks_mean, tracks_std)
        test_y = standardization(test_y, tracks_mean, tracks_std)
        train_x = standardization(train_x, tracks_mean, tracks_std)
        train_y = standardization(train_y, tracks_mean, tracks_std)

        speed_mean = np.asarray([odometry_x_test[:, :, 0].mean(), odometry_x_train[:, :, 0].mean()]).mean()
        speed_std = np.asarray([odometry_x_test[:, :, 0].std(), odometry_x_train[:, :, 0].std()]).mean()
        odometry_x_test[:, :, 0] = standardization(odometry_x_test[:, :, 0], speed_mean, speed_std)
        odometry_x_train[:, :, 0] = standardization(odometry_x_train[:, :, 0], speed_mean, speed_std)

        steer_mean = np.asarray([odometry_x_test[:, :, 1].mean(), odometry_x_train[:, :, 1].mean()]).mean()
        steer_std = np.asarray([odometry_x_test[:, :, 1].std(), odometry_x_train[:, :, 1].std()]).mean()
        odometry_x_test[:, :, 1] = standardization(odometry_x_test[:, :, 1], steer_mean, steer_std)
        odometry_x_train[:, :, 1] = standardization(odometry_x_train[:, :, 1], steer_mean, steer_std)

    #concatanete bbs with odometry
    test_x = np.concatenate([test_x, odometry_x_test], axis=-1)
    train_x = np.concatenate([train_x, odometry_x_train], axis=-1)

    model_kwargs['input_shape'] = train_x.shape[1:]
    model_kwargs['output_dim'] = train_y.shape[-1]

    model = build_model(**model_kwargs)
    model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, shuffle=True)
    # cant use model save method because of bug https://github.com/tensorflow/tensorflow/issues/34028
    # model.save(os.path.join(base_path, model_name))

    # saving weights and model_kwargs separately
    save_model(os.path.join(model_path, model_name), model, model_kwargs)

    if evaluate:
        pred, aletoric, epistemic = predict(model, test_x,
                                            predict_var=model_kwargs['predict_variance'],
                                            use_cum_sum=True if diff_fn == get_diff_array_v2 else False,
                                            mc_samples=mc_samples)
        if normalize:
            test_y = inverse_standardization(test_y, tracks_mean, tracks_std)
            pred = inverse_standardization(pred, tracks_mean, tracks_std)
            if aletoric is not None:
                aletoric = aletoric * tracks_std # scale aletoric uncertainty
            if epistemic is not None:
                epistemic = inverse_standardization(epistemic, tracks_mean, tracks_std)

        test_y = np.cumsum(test_y, axis=1) if diff_fn == get_diff_array_v2 else test_y
        mse = np.square(pred - test_y).mean()
        print('MSE: %f Aletoric unc.: %s Epistemic unc: %s' % (mse,
                                                               str(aletoric.mean()) if aletoric is not None else '-',
                                                               str(epistemic.mean()) if epistemic is not None else '-'))

    return model


def get_kwargs_from_cli(kwargs):
    kwargs['model_name'] = kwargs.get('model_name', 'model')
    kwargs['model_path'] = kwargs.get('model_path', 'data/model')
    kwargs['num_prediction_steps'] = int(kwargs.get('num_prediction_steps', 15))
    kwargs['weight_dropout'] = float(kwargs.get('weight_dropout', 0.))
    kwargs['unit_dropout'] = float(kwargs.get('unit_dropout', 0.25))
    kwargs['lam'] = float(kwargs.get('lam', 0.0001))
    kwargs['predict_variance'] = strtobool(kwargs.get('predict_variance', 'True'))
    kwargs['use_mc_dropout'] = strtobool(kwargs.get('use_mc_dropout', 'True'))
    kwargs['mc_samples'] = int(kwargs.get('mc_samples', 10))
    kwargs['epochs'] = int(kwargs.get('epochs', 1))
    kwargs['batch_size'] = int(kwargs.get('batch_size', 512))
    kwargs['num_units'] = int(kwargs.get('num_units', 256))
    kwargs['normalize'] = strtobool(kwargs.get('normalize', 'True'))
    kwargs['diff_fn'] = kwargs.get('diff_fn', 'get_diff_array')
    kwargs['loss_fn'] = kwargs.get('loss_fn', 'heteroskedastic_loss')
    _losses = ['mse', heteroskedastic_loss.__name__]
    if kwargs['loss_fn'] not in _losses:
        raise ValueError('Unknown loss_fn  use one  these {}'.format(_losses))
    kwargs['loss_fn'] = globals()[kwargs['loss_fn']]
    kwargs['diff_fn'] = globals()[kwargs['diff_fn']] \
        if kwargs['diff_fn'] in [get_diff_array_v2.__name__, get_diff_array.__name__] else get_diff_array
    kwargs['evaluate'] = strtobool(kwargs.get('evaluate', 'False'))

    return kwargs


if __name__ == '__main__':
    np.random.seed(1444)  # random seed for train data shuffling

    kwargs = dict(arg.split('=') for arg in sys.argv[1:])
    kwargs = get_kwargs_from_cli(kwargs)
    model = train(**kwargs)