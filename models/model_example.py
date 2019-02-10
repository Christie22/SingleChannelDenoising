# adapted from cifar10 from https://github.com/chaitanya100100/VAE-for-Image-Generation/

from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPool2D, BatchNormalization, Flatten, Reshape, Dropout
from keras.models import Model
from keras import backend as K
from model_utils import SampleNormal, VAELossLayer


# model
class VaeModel(object):
    def __init__(
            self,
            input_shape,
            kernel_size,
            n_filters,
            n_intermediate_dim=128,
            n_latent_dim=2,
            epsilon_std=1.0):
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.n_tot_dim = input_shape[0] * input_shape[1] * input_shape[2]
        self.n_inter_dim = n_intermediate_dim
        self.n_latent_dim = n_latent_dim
        self.epsilon_std = epsilon_std
        self.model_enc = None
        self.model_dec = None
        self.model_vae = None

    def get_encoder(self):
        return self.model_enc

    def get_decoder(self):
        return self.model_dec

    def get_model(self):
        if self.model_vae is None:
            self.gen_model()
        return self.model_vae

    def gen_encoder(self):
        inputs = Input(shape=self.input_shape)
        x = Conv2D(
            self.n_filters // 4,
            kernel_size=self.kernel_size,
            padding='valid',
            activation='relu',
            dilation_rate=(64,8))(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(
            self.n_filters // 2,
            kernel_size=self.kernel_size,
            padding='valid',
            activation='relu',
            dilation_rate=(32,8))(x)
        x = BatchNormalization()(x)
        x = Conv2D(
            self.n_filters,
            kernel_size=self.kernel_size,
            padding='valid',
            activation='relu',
            dilation_rate=(16,8))(x)
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
        z_mean = Dense(self.n_latent_dim)(dense)
        z_log_var = Dense(self.n_latent_dim)(dense)
        self.model_enc = Model(inputs, [z_mean, z_log_var])
        self.model_enc.name = 'vae_encoder'

    def gen_decoder(self):
        inputs = Input(shape=(self.n_latent_dim,))
        dense = Dense(
            self.n_inter_dim,
            activation='relu')(inputs)
        dense = BatchNormalization()(dense)
        dense = Dense(
            self.conv_shape[1] *
            self.conv_shape[2] *
            self.conv_shape[3],
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
        self.model_dec = Model(inputs, x)
        self.model_dec.name = 'vae_decoder'

    def gen_model(self):
        self.gen_encoder()
        self.gen_decoder()
        # TODO can we do something like here?
        # https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py#L159
        x_true = Input(shape=self.input_shape, name='vae_input')
        z_mean, z_log_var = self.model_enc(x_true)
        z = SampleNormal(name='vae_sampling')([z_mean, z_log_var])
        x_pred = self.model_dec(z)
        vae_loss = VAELossLayer(name='vae_loss')(
            [x_true, x_pred, z_mean, z_log_var])
        self.model_vae = Model(inputs=[x_true], outputs=[vae_loss])
