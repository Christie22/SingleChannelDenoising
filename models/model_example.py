# Example of convolutional autoencoder model

from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPool2D, BatchNormalization, Flatten, Reshape, Dropout
from keras.models import Model
from keras import backend as K
from libs.model_utils import LossLayer
import numpy as np


# model
class AEModelFactory(object):
    def __init__(
            self,
            input_shape,
            kernel_size,
            n_filters,
            n_intermediate_dim=128,
            n_latent_dim=2
            ):
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.n_tot_dim = np.prod(input_shape)
        self.n_inter_dim = n_intermediate_dim
        self.n_latent_dim = n_latent_dim
        self._encoder = None
        self._decoder = None
        self._model = None

    def get_encoder(self):
        return self._encoder

    def get_decoder(self):
        return self._decoder

    def get_model(self):
        if self._model is None:
            self.gen_model()
        return self._model

    def gen_encoder(self):
        inputs = Input(shape=self.input_shape)
        x = Conv2D(
            self.n_filters // 4,
            kernel_size=self.kernel_size,
            padding='same',
            strides=(2, 2),
            activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(
            self.n_filters // 2,
            kernel_size=self.kernel_size,
            padding='same',
            strides=(2, 2),
            activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(
            self.n_filters,
            kernel_size=self.kernel_size,
            padding='same',
            strides=(2, 2),
            activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(
            self.n_filters,
            kernel_size=self.kernel_size,
            padding='same',
            strides=(2,2))(x)
        x = BatchNormalization()(x)
        self.conv_shape = K.int_shape(x)
        flat = Flatten()(x)
        dense = Dense(
            self.n_inter_dim,
            activation='tanh')(flat)
        dense = Dropout(0.4)(dense)
        z = Dense(self.n_latent_dim)(dense)
        self._encoder = Model(inputs, z)
        self._encoder.summary()
        self._encoder.name = 'encoder'

    def gen_decoder(self):
        inputs = Input(shape=(self.n_latent_dim,))
        dense = Dense(
            self.n_inter_dim,
            activation='relu')(inputs)
        dense = BatchNormalization()(dense)
        dense = Dense(
            np.prod(self.conv_shape[1:]),
            activation='relu')(inputs)
        dense = Dropout(0.4)(dense)
        dense = BatchNormalization()(dense)
        x = Reshape(self.conv_shape[1:])(dense)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(
            self.n_filters,
            kernel_size=self.kernel_size,
            padding='same',
            activation='relu',
            strides=(2,2))(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(
            self.n_filters // 2,
            kernel_size=self.kernel_size,
            padding='same',
            activation='relu',
            strides=(2,2))(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(
            self.n_filters // 4,
            kernel_size=self.kernel_size,
            padding='same',
            activation='relu',
            strides=(2,1))(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(
            2,
            kernel_size=1,
            padding='same',
            strides=1)(x)
        self._decoder = Model(inputs, x)
        self._decoder.summary()
        self._decoder.name = 'decoder'

    def gen_model(self):
        self.gen_encoder()
        self.gen_decoder()
        # TODO can we do something like here?
        # https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py#L159
        x_true = Input(shape=self.input_shape, name='input')
        z = self._encoder(x_true)
        x_pred = self._decoder(z)
        loss = LossLayer(name='loss')(
            [x_true, x_pred, z])
        self._model = Model(inputs=[x_true], outputs=[loss])
