import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import argparse
import matplotlib.pyplot as plt
from utils import *
from sequence import Sequence
from model.model import *
from layers.dropconnect_rnn import DropConnectLSTM
from layers.dropconnect_dense import DropConnectDense
from tensorflow.keras.losses import mse
from distutils.util import strtobool

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.7
    epochs_drop = 3
    lrate = initial_lrate * np.power(drop, np.floor((1+epoch)/epochs_drop))
    return lrate


def predict(model, test_x, mc_samples=10, predict_var=True):
    logvar = None

    pred = [model.predict(test_x) for _ in range(mc_samples)]
    pred = np.asarray(pred)

    if predict_var:
        pred, logvar = np.split(pred, 2, axis=-1)

    return pred, logvar


def train(model_name, model_path='data/model', in_frames=8, out_frames=15, diff_fn=get_diff_array_v2, normalize=False,
          batch_size=128, epochs=20, evaluate=True, train_model=True, mc_samples=10, use_visual_features=False,
          predict_variance=True, odometry_model_path='data/model_odometry', odometry_model_name='model',
          load_model=False, shuffle=True, use_inverse_data=False, force_load_data=False, log_dir='data/logs/',
          imgs_base_path_test = 'data/imgs/test', tracks_base_path_test = 'data/tracks/tracks_test.h5',
          odometry_base_path_test = 'data/odometry/test', imgs_base_path_train = 'data/imgs/train',
          tracks_base_path_train = 'data/tracks/tracks_train.h5', odometry_base_path_train = 'data/odometry/train',
          **model_kwargs):

    model_kwargs = dict(locals(), **model_kwargs)
    del model_kwargs['model_kwargs'] # del duplicate values

    #DATA
    dataset_pickle_path_test = 'data/pickles/dataset_test.p'
    dataset_pickle_path_train = 'data/pickles/dataset_train.p'

    if force_load_data:
        # get data set
        test_dataset= get_whole_data_set(in_frames,
                                         out_frames,
                                         diff_fn=diff_fn,
                                         tracks_base_path=tracks_base_path_test,
                                         odometry_base_path=odometry_base_path_test,
                                         img_sequences_base_path=imgs_base_path_test,
                                         load_img_paths=True if use_visual_features else False)

        test_x, test_y, _, image_sequence_paths_test, odometry_x_test, odometry_y_test, _ = test_dataset
        save_pickle(dataset_pickle_path_test, test_dataset)

        train_dataset = get_whole_data_set(in_frames,
                                           out_frames,
                                           diff_fn=diff_fn,
                                           tracks_base_path=tracks_base_path_train,
                                           odometry_base_path=odometry_base_path_train,
                                           img_sequences_base_path=imgs_base_path_train,
                                           load_img_paths=True if use_visual_features else False)

        train_x, train_y, _, image_sequence_paths_train, odometry_x_train, odometry_y_train, _ = train_dataset
        save_pickle(dataset_pickle_path_train, train_dataset)

    else:
        test_x, test_y, _, image_sequence_paths_test, odometry_x_test, odometry_y_test, _ \
            = load_pickle(dataset_pickle_path_test)
        train_x, train_y, _, image_sequence_paths_train, odometry_x_train, odometry_y_train, _ \
            = load_pickle(dataset_pickle_path_train)

    print('[+] Tracks  train shapes x: %s y: %s' % (train_x.shape, train_y.shape))
    print('[+] Odometry  train shapes x: %s y: %s' % (odometry_x_train.shape, odometry_y_train.shape))

    # normalizing data features separately with standard scaling
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
        odometry_y_test[:, :, 0] = standardization(odometry_y_test[:, :, 0], speed_mean, speed_std)
        odometry_y_train[:, :, 0] = standardization(odometry_y_train[:, :, 0], speed_mean, speed_std)

        steer_mean = np.asarray([odometry_x_test[:, :, 1].mean(), odometry_x_train[:, :, 1].mean()]).mean()
        steer_std = np.asarray([odometry_x_test[:, :, 1].std(), odometry_x_train[:, :, 1].std()]).mean()
        odometry_x_test[:, :, 1] = standardization(odometry_x_test[:, :, 1], steer_mean, steer_std)
        odometry_x_train[:, :, 1] = standardization(odometry_x_train[:, :, 1], steer_mean, steer_std)
        odometry_y_test[:, :, 1] = standardization(odometry_y_test[:, :, 1], steer_mean, steer_std)
        odometry_y_train[:, :, 1] = standardization(odometry_y_train[:, :, 1], steer_mean, steer_std)

    #concatanete bbs with odometry
    test_x = np.concatenate([test_x, odometry_x_test], axis=-1)
    train_x = np.concatenate([train_x, odometry_x_train], axis=-1)

    model_kwargs['model_input_shape'] = train_x.shape[1:]
    model_kwargs['model_visual_input_shape'] = [300, 150, 3]
    model_kwargs['model_output_dim'] = train_y.shape[-1]

    if load_model:
        model = tf.keras.models.load_model(os.path.join(model_path, model_name),
                                           custom_objects={'DropConnectDense': DropConnectDense,
                                                           'DropConnectLSTM': DropConnectLSTM})
    else:
        model = get_model_visual(**model_kwargs) if use_visual_features else get_model(**model_kwargs)

    if train_model:
        #split train data to val and train
        validation_split = 0.2
        val_size = int(train_x.shape[0] * (1 - validation_split))
        val_x = train_x[val_size:]
        val_image_sequence_paths = image_sequence_paths_train[val_size:]
        val_y = train_y[val_size:]
        train_x = train_x[:val_size]
        train_image_sequence_paths = image_sequence_paths_train[:val_size]
        train_y = train_y[:val_size]

        callbacks = []
        #call_back for validation with mc_sampling
        # callbacks.append(TrainEvalCallback(predict, train_x, train_y, tracks_mean, tracks_std, mc_samples, False))
        log_dir = log_dir + model_name
        if not os.path.isdir(log_dir): os.mkdir(log_dir)
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=0))


        train_gen = Sequence(train_x, train_y, train_image_sequence_paths if use_visual_features else None,
                             batch_size=batch_size)
        val_gen = Sequence(val_x, val_y, val_image_sequence_paths if use_visual_features else  None,
                           batch_size=batch_size)

        model.fit(train_gen, epochs=epochs, callbacks=callbacks, validation_data=val_gen)

        # saving model
        model.save(os.path.join(model_path, model_name))

    if evaluate:
        epistemic = None
        aletoric = None

        test_gen = Sequence(test_x, test_y, image_sequence_paths_test if use_visual_features else None,
                            batch_size=batch_size, shuffle=False)
        pred, log_var = predict(model, test_gen, predict_var=predict_variance, mc_samples=mc_samples)

        if log_var is not None:
            aletoric = log_var.mean(axis=0)

        if normalize:
            test_y = inverse_standardization(test_y, tracks_mean, tracks_std)
            pred = inverse_standardization(pred, tracks_mean, tracks_std)
            if log_var is not None:
                aletoric = np.exp(aletoric)
                # scale aletoric uncertainty
                aletoric = (aletoric - aletoric.min()) / (aletoric.max() - aletoric.min())
                aletoric = aletoric * tracks_std**2

        if mc_samples > 1:
            epistemic = pred.var(axis=0)

        pred = pred.mean(axis=0)

        test_y = np.cumsum(test_y, axis=1) if diff_fn == get_diff_array_v2 else test_y
        pred = np.cumsum(pred, axis=1) if diff_fn == get_diff_array_v2 else pred
        mse = np.square(pred - test_y).mean()
        print('MSE: %f Aletoric unc.: %s Epistemic unc: %s' % (mse,
                                                               str(aletoric.mean()) if aletoric is not None else '-',
                                                               str(epistemic.mean()) if epistemic is not None else '-'))

    return model


def get_kwargs_from_cli():

    def get_fn(loss_fn):
        return globals()[loss_fn]

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='model')
    parser.add_argument('--model_path', type=str, default='data/model')

    parser.add_argument('--imgs_base_path_test', type=str, default='data/imgs/test')
    parser.add_argument('--imgs_base_path_train', type=str, default='data/imgs/train')

    parser.add_argument('--odometry_base_path_test', type=str, default='data/odometry/test')
    parser.add_argument('--odometry_base_path_train', type=str, default='data/odometry/train')
    parser.add_argument('--tracks_base_path_train', type=str, default='data/tracks/tracks_train.h5')
    parser.add_argument('--tracks_base_path_test', type=str, default='data/tracks/tracks_test.h5')

    parser.add_argument('--use_visual_features', type=strtobool, default=False, choices=[True, False])
    parser.add_argument('--num_prediction_steps', type=int, default=15)
    parser.add_argument('--in_frames', type=int, default=8)
    parser.add_argument('--out_frames', type=int, default=15)
    parser.add_argument('--weight_dropout', type=float, default=0.25)
    parser.add_argument('--unit_dropout', type=float, default=0.)
    parser.add_argument('--lam', type=float, default=0.0001)
    parser.add_argument('--predict_variance', type=strtobool, default=True, choices=[True, False])
    parser.add_argument('--use_mc_dropout', type=strtobool, default=True, choices=[True, False])
    parser.add_argument('--mc_samples', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_units', type=int, default=256)
    parser.add_argument('--normalize', type=strtobool, default=True, choices=[True, False])
    parser.add_argument('--loss_fn', type=get_fn, default=heteroskedastic_loss_v2,
                        choices=[mse, heteroskedastic_loss_v2, heteroskedastic_loss])
    parser.add_argument('--diff_fn', type=get_fn, default=get_diff_array, choices=[get_diff_array_v2, get_diff_array])
    parser.add_argument('--evaluate', type=strtobool, default=True, choices=[True, False])
    parser.add_argument('--train_model', type=strtobool, default=True, choices=[True, False])
    parser.add_argument('--load_model', type=strtobool, default=False, choices=[True, False])
    parser.add_argument('--shuffle', type=strtobool, default=True, choices=[True, False])
    parser.add_argument('--force_load_data', type=strtobool, default=False, choices=[True, False])
    args = parser.parse_args()

    kwargs = dict(args._get_kwargs())

    print('[+] Model parameters')
    for k, v in kwargs.items():
        print(k + " -> " + str(v))

    return kwargs


if __name__ == '__main__':
    np.random.seed(1444)  # random seed for train data shuffling

    kwargs = get_kwargs_from_cli()
    model = train(**kwargs)