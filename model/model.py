import tensorflow as tf
import numpy as np
import os

from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Dense, LSTM, RepeatVector

from layers.dropconnect_rnn import DropConnectLSTM
from layers.dropconnect_dense import DropConnectDense
from utils import save_pickle, load_pickle


def heteroskedastic_loss(y, pred):
    mean, log_var= tf.split(pred, num_or_size_splits=2, axis=-1)
    loss1 = tf.reduce_mean(tf.exp(-log_var) * (tf.square((mean - y))))
    loss2 = tf.reduce_mean(log_var)
    loss = .5 * (loss1 + loss2)
    return loss


# computing loss for x and y coordinates separately
def heteroskedastic_loss_v2(y, pred):
    mean, log_var= tf.split(pred, num_or_size_splits=2, axis=-1)

    c1 = 0.75
    c2 = 1 - c1

    y_x1 = tf.gather(y, [0], axis=-1)
    mean_x1 = tf.gather(mean, [0], axis=-1)
    log_var_x1 = tf.gather(log_var, [0], axis=-1)

    y_x2 = tf.gather(y, [2], axis=-1)
    mean_x2 = tf.gather(mean, [2], axis=-1)
    log_var_x2 = tf.gather(log_var, [2], axis=-1)

    y_y1 = tf.gather(y, [1], axis=-1)
    mean_y1 = tf.gather(mean, [1], axis=-1)
    log_var_y1 = tf.gather(log_var, [1], axis=-1)

    y_y2 = tf.gather(y, [3], axis=-1)
    mean_y2 = tf.gather(mean, [3], axis=-1)
    log_var_y2 = tf.gather(log_var, [3], axis=-1)

    loss_x1 = tf.reduce_mean(c1 * (tf.exp(-log_var_x1) * (tf.square((mean_x1 - y_x1)))) + c2 * log_var_x1)
    loss_x2 = tf.reduce_mean(c1 * (tf.exp(-log_var_x2) * (tf.square((mean_x2 - y_x2)))) + c2 * log_var_x2)
    loss_y1 = tf.reduce_mean(c1 * (tf.exp(-log_var_y1) * (tf.square((mean_y1 - y_y1)))) + c2 * log_var_y1)
    loss_y2 = tf.reduce_mean(c1 * (tf.exp(-log_var_y2) * (tf.square((mean_y2 - y_y2)))) + c2 * log_var_y2)

    return loss_x1 + loss_y1 + loss_x2 + loss_y2


# calculate mse from mean when predicting mean and variance
def mse_metric(y, pred):
    mean, log_var = tf.split(pred, num_or_size_splits=2, axis=-1)
    return tf.reduce_mean(tf.square(y - mean))


def mean_variance(y, pred):
    mean, log_var = tf.split(pred, num_or_size_splits=2, axis=-1)
    return tf.reduce_mean(log_var)


def get_model(model_name, model_path, model_input_shape, model_output_dim, loss_fn, predict_variance,
              num_prediction_steps=15, weight_dropout=0., unit_dropout=0.35, lam=0.0001, use_mc_dropout=True,
              num_units=256, **kwargs):

    inputs = tf.keras.Input(model_input_shape)

    w_in = Dense(64,
                     kernel_regularizer=tf.keras.regularizers.l2(lam),
                     bias_regularizer=tf.keras.regularizers.l2(lam),
                     activation=tf.nn.relu)(inputs)

    encoder = DropConnectLSTM(num_units,
                                  dropout=unit_dropout,
                                  recurrent_dropout=unit_dropout,
                                  recurrent_kernel_dropout=weight_dropout,
                                  kernel_dropout=weight_dropout,
                                  kernel_regularizer=tf.keras.regularizers.l2(lam),
                                  recurrent_regularizer=tf.keras.regularizers.l2(lam),
                                  bias_regularizer=tf.keras.regularizers.l2(lam),
                                  use_mc_dropout=use_mc_dropout)(w_in)

    repeat_vector = RepeatVector(num_prediction_steps)(encoder)

    decoder = DropConnectLSTM(num_units,
                                  dropout=unit_dropout,
                                  recurrent_dropout=unit_dropout,
                                  recurrent_kernel_dropout=weight_dropout,
                                  kernel_dropout=weight_dropout,
                                  kernel_regularizer=tf.keras.regularizers.l2(lam),
                                  recurrent_regularizer=tf.keras.regularizers.l2(lam),
                                  bias_regularizer=tf.keras.regularizers.l2(lam),
                                  return_sequences=True,
                                  use_mc_dropout=use_mc_dropout)(repeat_vector)

    y = DropConnectDense(model_output_dim,
                                kernel_dropout=weight_dropout,
                                unit_dropout=unit_dropout,
                                use_mc_dropout=use_mc_dropout,
                                kernel_regularizer=tf.keras.regularizers.l2(lam),
                                bias_regularizer=tf.keras.regularizers.l2(lam),
                                activation=None)(decoder)
    if predict_variance:
        log_var = Dense(model_output_dim,
                            kernel_regularizer=tf.keras.regularizers.l2(lam),
                            bias_regularizer=tf.keras.regularizers.l2(lam),
                            activation=tf.nn.relu)(decoder)
        y = Concatenate()([y, log_var])

    model = tf.keras.models.Model(inputs=inputs, outputs=y)
    model.compile(optimizer='adam', loss=loss_fn)

    return model


def get_model_visual(model_name, model_path, model_input_shape, model_output_dim, model_visual_input_shape, loss_fn,
                     predict_variance, num_prediction_steps=15, weight_dropout=0., unit_dropout=0.35, lam=0.0001,
                     use_mc_dropout=True, num_units=256, cnn_extractor=tf.keras.applications.InceptionResNetV2,
                     **kwargs):

    inputs = tf.keras.Input(model_input_shape)
    visual_inputs = tf.keras.Input(model_visual_input_shape)

    # visual features
    cnn_model = cnn_extractor(input_shape=model_visual_input_shape, include_top=False)
    cnn_ext = cnn_model(visual_inputs)
    visual_features_flat = tf.keras.layers.Flatten()(cnn_ext)
    w_vis = DropConnectDense(128, kernel_dropout=weight_dropout,
                             unit_dropout=unit_dropout,
                             use_mc_dropout=use_mc_dropout,
                             kernel_regularizer=tf.keras.regularizers.l2(lam),
                             bias_regularizer=tf.keras.regularizers.l2(lam),
                             activation=None)(visual_features_flat)

    w_in = Dense(64,
                     kernel_regularizer=tf.keras.regularizers.l2(lam),
                     bias_regularizer=tf.keras.regularizers.l2(lam),
                     activation=tf.nn.relu)(inputs)

    encoder = DropConnectLSTM(num_units,
                                  dropout=unit_dropout,
                                  recurrent_dropout=unit_dropout,
                                  recurrent_kernel_dropout=weight_dropout,
                                  kernel_dropout=weight_dropout,
                                  kernel_regularizer=tf.keras.regularizers.l2(lam),
                                  recurrent_regularizer=tf.keras.regularizers.l2(lam),
                                  bias_regularizer=tf.keras.regularizers.l2(lam),
                                  use_mc_dropout=use_mc_dropout)(w_in)

    z_enc = Concatenate()([encoder, w_vis])

    repeat_vector = RepeatVector(num_prediction_steps)(z_enc)

    decoder = DropConnectLSTM(num_units,
                                  dropout=unit_dropout,
                                  recurrent_dropout=unit_dropout,
                                  recurrent_kernel_dropout=weight_dropout,
                                  kernel_dropout=weight_dropout,
                                  kernel_regularizer=tf.keras.regularizers.l2(lam),
                                  recurrent_regularizer=tf.keras.regularizers.l2(lam),
                                  bias_regularizer=tf.keras.regularizers.l2(lam),
                                  return_sequences=True,
                                  use_mc_dropout=use_mc_dropout)(repeat_vector)

    y = DropConnectDense(model_output_dim,
                                kernel_dropout=weight_dropout,
                                unit_dropout=unit_dropout,
                                use_mc_dropout=use_mc_dropout,
                                kernel_regularizer=tf.keras.regularizers.l2(lam),
                                bias_regularizer=tf.keras.regularizers.l2(lam),
                                activation=None)(decoder)
    if predict_variance:
        log_var = Dense(model_output_dim,
                            kernel_regularizer=tf.keras.regularizers.l2(lam),
                            bias_regularizer=tf.keras.regularizers.l2(lam),
                            activation=tf.nn.relu)(decoder)
        y = Concatenate()([y, log_var])

    # freeze weights of visual features pretrained cnn extractor
    for layer in cnn_model.layers:
        layer.trainable = False

    model = tf.keras.models.Model(inputs=[inputs, visual_inputs], outputs=y)
    model.compile(optimizer='adam', loss=loss_fn)

    return model