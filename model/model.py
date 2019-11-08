import tensorflow as tf
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Dense, LSTM, RepeatVector

from layers.dropconnect_rnn import DropConnectLSTM
from layers.dropconnect_dense import DropConnectDense


def heteroskedasticit_loss(y, pred):
    mean, log_var= tf.split(pred, num_or_size_splits=2, axis=-1)
    loss1 = tf.reduce_mean(tf.exp(-log_var) * (tf.square((mean - y))))
    loss2 = tf.reduce_mean(log_var)
    loss = .5 * (loss1 + loss2)
    return loss


# calculate mse from mean when predicting mean and variance
def mse_metric(y, pred):
    mean, log_var = tf.split(pred, num_or_size_splits=2, axis=-1)
    return tf.reduce_mean(tf.square(y - mean))


def mean_variance(y, pred):
    mean, log_var = tf.split(pred, num_or_size_splits=2, axis=-1)
    return tf.reduce_mean(log_var)


def build_model(input_shape, output_dim, num_prediction_steps=15, weight_dropout=0., unit_dropout=0.35, lam=0.0001,
                loss_fn=heteroskedasticit_loss, predict_variance=True, use_mc_dropout=True):

        inputs = tf.keras.Input(shape=input_shape)

        x = Dense(128,
                  kernel_regularizer=tf.keras.regularizers.l2(lam),
                  bias_regularizer=tf.keras.regularizers.l2(lam),
                  activation=tf.nn.relu)(inputs)

        encoder = DropConnectLSTM(256,
                            dropout=unit_dropout,
                            recurrent_dropout=unit_dropout,
                            recurrent_kernel_dropout=weight_dropout,
                            kernel_dropout=weight_dropout,
                            kernel_regularizer=tf.keras.regularizers.l2(lam),
                            recurrent_regularizer=tf.keras.regularizers.l2(lam),
                            use_mc_dropout=use_mc_dropout)(x)

        encoder = RepeatVector(num_prediction_steps)(encoder)

        decoder = DropConnectLSTM(256,
                                  dropout=unit_dropout,
                                  recurrent_dropout=unit_dropout,
                                  recurrent_kernel_dropout=weight_dropout,
                                  kernel_dropout=weight_dropout,
                                  kernel_regularizer=tf.keras.regularizers.l2(lam),
                                  recurrent_regularizer=tf.keras.regularizers.l2(lam),
                                  return_sequences=True,
                                  use_mc_dropout=use_mc_dropout)(encoder)

        mean = Dense(output_dim,
                                kernel_regularizer=tf.keras.regularizers.l2(lam),
                                bias_regularizer=tf.keras.regularizers.l2(lam),
                                activation=None)(decoder)
        if predict_variance:
            log_var = Dense(output_dim,
                            kernel_regularizer=tf.keras.regularizers.l2(lam),
                            bias_regularizer=tf.keras.regularizers.l2(lam),
                            activation=tf.nn.relu,)(decoder)
            x = Concatenate()([mean, log_var])
        else:
            x = mean

        model = Model(inputs=[inputs], outputs=x)
        metrics = [mse_metric if predict_variance else 'mse']
        if predict_variance: metrics.append(mean_variance)
        model.compile(optimizer='adam', loss=loss_fn, metrics=metrics)

        return model