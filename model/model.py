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

    c1 = 0.6
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

    loss_x1 = c1 * tf.reduce_mean(tf.exp(-log_var_x1) * (tf.square((mean_x1 - y_x1)))) + c2 * tf.reduce_mean(log_var_x1)
    loss_x2 = c1 * tf.reduce_mean(tf.exp(-log_var_x2) * (tf.square((mean_x2 - y_x2)))) + c2 * tf.reduce_mean(log_var_x2)
    loss_y1 = c1 * tf.reduce_mean(tf.exp(-log_var_y1) * (tf.square((mean_y1 - y_y1)))) + c2 * tf.reduce_mean(log_var_y1)
    loss_y2 = c1 * tf.reduce_mean(tf.exp(-log_var_y2) * (tf.square((mean_y2 - y_y2)))) + c2 * tf.reduce_mean(log_var_y2)

    return loss_x1 + loss_y1 + loss_x2 + loss_y2


# calculate mse from mean when predicting mean and variance
def mse_metric(y, pred):
    mean, log_var = tf.split(pred, num_or_size_splits=2, axis=-1)
    return tf.reduce_mean(tf.square(y - mean))


def mean_variance(y, pred):
    mean, log_var = tf.split(pred, num_or_size_splits=2, axis=-1)
    return tf.reduce_mean(log_var)


class BaseModel(Model):

    def __init__(self, model_name, model_path, model_input_shape, model_output_dim,
                 loss_fn, predict_variance, **kwargs):
        super(BaseModel, self).__init__(name=model_name)
        self.model_name = model_name
        self.model_path = model_path
        self.model_input_shape = model_input_shape
        self.model_output_dim = model_output_dim
        self.loss_fn = loss_fn
        self.predict_variance = predict_variance
        self.kwargs = kwargs

    def call(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    @staticmethod
    def load_model(path):
        raise NotImplementedError

    def get_config(self):
        self.kwargs['model_name'] = self.model_name
        self.kwargs['model_path'] = self.model_path
        self.kwargs['model_input_shape'] = self.model_input_shape
        self.kwargs['model_output_dim'] = self.model_output_dim
        self.kwargs['loss_fn'] = self.loss_fn
        self.kwargs['predict_variance'] = self.predict_variance
        return self.kwargs


class LstmPredictor(BaseModel):

    def __init__(self, num_prediction_steps=15, weight_dropout=0., unit_dropout=0.35, lam=0.0001,
                 use_mc_dropout=True, num_units=256, **model_kwargs):
        super(LstmPredictor, self).__init__(**model_kwargs)

        self.num_prediction_steps = num_prediction_steps
        self.weight_dropout = weight_dropout
        self.unit_dropout = unit_dropout
        self.lam = lam
        self.use_mc_dropout = use_mc_dropout
        self.num_units = num_units

        self.w_in = Dense(64,
                         kernel_regularizer=tf.keras.regularizers.l2(lam),
                         bias_regularizer=tf.keras.regularizers.l2(lam),
                         activation=tf.nn.relu,
                         input_shape=self.model_input_shape)

        self.encoder = DropConnectLSTM(num_units,
                                      dropout=unit_dropout,
                                      recurrent_dropout=unit_dropout,
                                      recurrent_kernel_dropout=weight_dropout,
                                      kernel_dropout=weight_dropout,
                                      kernel_regularizer=tf.keras.regularizers.l2(lam),
                                      recurrent_regularizer=tf.keras.regularizers.l2(lam),
                                      use_mc_dropout=use_mc_dropout)

        self.repeat_vector = RepeatVector(num_prediction_steps)

        self.decoder = DropConnectLSTM(num_units,
                                      dropout=unit_dropout,
                                      recurrent_dropout=unit_dropout,
                                      recurrent_kernel_dropout=weight_dropout,
                                      kernel_dropout=weight_dropout,
                                      kernel_regularizer=tf.keras.regularizers.l2(lam),
                                      recurrent_regularizer=tf.keras.regularizers.l2(lam),
                                      return_sequences=True,
                                      use_mc_dropout=use_mc_dropout)

        self.mean = DropConnectDense(self.model_output_dim,
                                    kernel_dropout=weight_dropout,
                                    unit_dropout=unit_dropout,
                                    use_mc_dropout=use_mc_dropout,
                                    kernel_regularizer=tf.keras.regularizers.l2(lam),
                                    bias_regularizer=tf.keras.regularizers.l2(lam),
                                    activation=None)
        if self.predict_variance:
            self.log_var = Dense(self.model_output_dim,
                                kernel_regularizer=tf.keras.regularizers.l2(lam),
                                bias_regularizer=tf.keras.regularizers.l2(lam),
                                activation=tf.nn.relu)
            self.concat = Concatenate()

    @tf.function
    def call(self, inputs):
        x = self.w_in(inputs)
        enc = self.encoder(x)
        enc = self.repeat_vector(enc)
        dec = self.decoder(enc)
        mean = self.mean(dec)
        if self.predict_variance:
            log_var = self.log_var(dec)
            mean = self.concat([mean, log_var])

        return mean

    def save_model(self):
        path = os.path.join(self.model_path, self.model_name)
        save_pickle(path +"_kwargs.p", self.get_config())
        self.save_weights(path)

    @staticmethod
    def load_model(path):
        kwargs = load_pickle(path + "_kwargs.p")
        model = LstmPredictor(**kwargs)
        metrics = [mse_metric if model.predict_variance else 'mse']
        if model.predict_variance: metrics.append(mean_variance)
        model.compile(optimizer='adam', loss=model.loss_fn, metrics=metrics)
        # dummy training, can't properly load weight without using dummy fit
        x_shape = [1] + list(model.model_input_shape)
        y_shape = [1, model.num_prediction_steps, model.model_output_dim]
        model.fit(np.ones(x_shape), np.ones(y_shape))
        model.load_weights(path)
        return model

    def get_config(self):
        self.kwargs['num_prediction_steps'] = self.num_prediction_steps
        self.kwargs['weight_dropout'] = self.weight_dropout
        self.kwargs['unit_dropout'] = self.unit_dropout
        self.kwargs['lam'] = self.lam
        self.kwargs['use_mc_dropout'] = self.use_mc_dropout
        self.kwargs['num_units'] = self.num_units
        base_config = super(LstmPredictor, self).get_config()
        return dict(list(base_config.items()) + list(self.kwargs.items()))


def build_model(model_cls=LstmPredictor, **kwargs):
    model = model_cls(**kwargs)
    metrics = [mse_metric if model.predict_variance else 'mse']
    if model.predict_variance: metrics.append(mean_variance)
    model.compile(optimizer='adam', loss=model.loss_fn, metrics=metrics)

    return model


class TwoStreamLstmPredictor(BaseModel):

    def __init__(self, num_prediction_steps=15, weight_dropout=0., unit_dropout=0.35, lam=0.0001,
                 use_mc_dropout=True, num_units=256, input_2_dim=2, **model_kwargs):
        super(TwoStreamLstmPredictor, self).__init__(**model_kwargs)

        self.num_prediction_steps = num_prediction_steps
        self.weight_dropout = weight_dropout
        self.unit_dropout = unit_dropout
        self.lam = lam
        self.use_mc_dropout = use_mc_dropout
        self.num_units = num_units
        self.input_2_dim = input_2_dim

        self.w_in = Dense(64,
                         kernel_regularizer=tf.keras.regularizers.l2(lam),
                         bias_regularizer=tf.keras.regularizers.l2(lam),
                         activation=tf.nn.relu,
                         input_shape=self.model_input_shape)

        self.encoder = DropConnectLSTM(num_units,
                                      dropout=unit_dropout,
                                      recurrent_dropout=unit_dropout,
                                      recurrent_kernel_dropout=weight_dropout,
                                      kernel_dropout=weight_dropout,
                                      kernel_regularizer=tf.keras.regularizers.l2(lam),
                                      recurrent_regularizer=tf.keras.regularizers.l2(lam),
                                      use_mc_dropout=use_mc_dropout)

        self.repeat_vector = RepeatVector(num_prediction_steps)

        self.decoder = DropConnectLSTM(num_units,
                                      dropout=unit_dropout,
                                      recurrent_dropout=unit_dropout,
                                      recurrent_kernel_dropout=weight_dropout,
                                      kernel_dropout=weight_dropout,
                                      kernel_regularizer=tf.keras.regularizers.l2(lam),
                                      recurrent_regularizer=tf.keras.regularizers.l2(lam),
                                      return_sequences=True,
                                      use_mc_dropout=use_mc_dropout)

        self.mean = DropConnectDense(self.model_output_dim,
                                    kernel_dropout=weight_dropout,
                                    unit_dropout=unit_dropout,
                                    use_mc_dropout=use_mc_dropout,
                                    kernel_regularizer=tf.keras.regularizers.l2(lam),
                                    bias_regularizer=tf.keras.regularizers.l2(lam),
                                    activation=None)
        if self.predict_variance:
            self.log_var = Dense(self.model_output_dim,
                                kernel_regularizer=tf.keras.regularizers.l2(lam),
                                bias_regularizer=tf.keras.regularizers.l2(lam),
                                activation=tf.nn.relu)
            self.concat = Concatenate()

        self.concat_odometry = Concatenate()
        self.w_odo = DropConnectDense(64,
                                     kernel_dropout=weight_dropout,
                                     unit_dropout=unit_dropout,
                                     use_mc_dropout=use_mc_dropout,
                                     kernel_regularizer=tf.keras.regularizers.l2(lam),
                                     bias_regularizer=tf.keras.regularizers.l2(lam),
                                     activation=tf.nn.relu)

    @tf.function
    def call(self, inputs):
        inputs, inputs_odometry = inputs
        x = self.w_in(inputs)
        enc = self.encoder(x)
        enc = self.repeat_vector(enc)
        inputs_odometry = self.w_odo(inputs_odometry)
        enc  = self.concat_odometry([enc, inputs_odometry])
        dec = self.decoder(enc)
        mean = self.mean(dec)

        if self.predict_variance:
            log_var = self.log_var(dec)
            mean = self.concat([mean, log_var])

        return mean

    def save_model(self):
        path = os.path.join(self.model_path, self.model_name)
        save_pickle(path +"_kwargs.p", self.get_config())
        self.save_weights(path)

    @staticmethod
    def load_model(path):
        kwargs = load_pickle(path + "_kwargs.p")
        model = TwoStreamLstmPredictor(**kwargs)
        metrics = [mse_metric if model.predict_variance else 'mse']
        if model.predict_variance: metrics.append(mean_variance)
        model.compile(optimizer='adam', loss=model.loss_fn, metrics=metrics)
        # dummy training, can't properly load weight without using dummy fit
        x_shape = [1] + list(model.model_input_shape)
        x2_shape = [1, model.num_prediction_steps, model.input_2_dim]
        y_shape = [1, model.num_prediction_steps, model.model_output_dim]
        model.fit([np.ones(x_shape), np.ones(x2_shape)], np.ones(y_shape))
        model.load_weights(path)
        return model

    def get_config(self):
        self.kwargs['num_prediction_steps'] = self.num_prediction_steps
        self.kwargs['weight_dropout'] = self.weight_dropout
        self.kwargs['unit_dropout'] = self.unit_dropout
        self.kwargs['lam'] = self.lam
        self.kwargs['use_mc_dropout'] = self.use_mc_dropout
        self.kwargs['num_units'] = self.num_units
        self.kwargs['input_2_dim'] = self.input_2_dim
        base_config = super(TwoStreamLstmPredictor, self).get_config()
        return dict(list(base_config.items()) + list(self.kwargs.items()))