import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from utils import *
from model.model import build_model, heteroskedastic_loss, save_model, load_model


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
        aletoric_unc = np.exp(logvar.mean(axis=0))**0.5

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


def train(model_name, base_path='data/model', in_frames=8, out_frames=15, diff_fn=get_diff_array_v2, normalize=False,
          batch_size=512 , epochs=10, evaluate=True, mc_samples=10, **model_kwargs):

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

    model_kwargs['input_shape'] = train_x.shape[1:]
    model_kwargs['output_dim'] = train_y.shape[-1]

    model = build_model(**model_kwargs)
    model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs)
    # cant use model save method because of bug https://github.com/tensorflow/tensorflow/issues/34028
    # model.save(os.path.join(base_path, model_name))

    # saving weights and model_kwargs separately
    save_model(os.path.join(base_path, model_name), model, model_kwargs)

    if evaluate:
        pred, aletoric, epistemic = predict(model, test_x,
                                            predict_var=model_kwargs['predict_variance'],
                                            use_cum_sum=True if diff_fn == get_diff_array_v2 else False,
                                            mc_samples=mc_samples)
        test_y = np.cumsum(test_y, axis=1) if diff_fn == get_diff_array_v2 else test_y
        mse = np.square(pred - test_y).mean()
        print('MSE: %f Aletoric unc.: %s Epistemic unc: %s' % (mse,
                                                               str(aletoric.mean()) if aletoric is not None else '-',
                                                               str(epistemic.mean()) if epistemic is not None else '-'))

    return model


if __name__ == '__main__':
    np.random.seed(1444)  # random seed for train data shuffling

    kwargs = dict(arg.split('=') for arg in sys.argv[1:])
    kwargs['num_prediction_steps'] = kwargs.get('num_prediction_steps', 15)
    kwargs['weight_dropout'] = kwargs.get('weight_dropout', 0.)
    kwargs['unit_dropout'] = kwargs.get('unit_dropout', 0.35)
    kwargs['lam'] = kwargs.get('lam', 0.0001)
    kwargs['loss_fn'] = kwargs.get('loss_fn', heteroskedastic_loss)
    kwargs['predict_variance'] = kwargs.get('predict_variance', True)
    kwargs['use_mc_dropout'] = kwargs.get('use_mc_dropout', True)
    kwargs['mc_samples'] = kwargs.get('mc_samples', 10)
    kwargs['evaluate'] = kwargs.get('evaluate', True)
    kwargs['epochs'] = kwargs.get('epochs', 10)
    kwargs['num_units'] = kwargs.get('num_units', 256)
    model = train(**kwargs)