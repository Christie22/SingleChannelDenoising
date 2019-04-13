import math
import numpy as np

from keras.callbacks import TensorBoard
from keras.layers import Layer
from keras import backend as K


# sampling from normal dist. + reparametrization trick
class SampleNormal(Layer):
    __name__ = 'sample_normal'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(SampleNormal, self).__init__(**kwargs)

    def _sample_normal(self, z_mean, z_log_var):
        # batch_size = K.shape(z_mean)[0]
        # z_dims = K.shape(z_mean)[1]
        eps = K.random_normal(shape=K.shape(z_mean), mean=0.0, stddev=1.0)
        return z_mean + K.exp(z_log_var / 2.0) * eps

    def call(self, inputs):
        z_mean, z_log_var = inputs
        return self._sample_normal(z_mean, z_log_var)


# loss function dummy layer for variational autoencoder
class VAELossLayer(Layer):
    __name__ = 'vae_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(VAELossLayer, self).__init__(**kwargs)

    def lossfun(self, x_true, x_pred, z_mean, z_log_var):
        # log power spectrum loss
        # lps_true = K.log(K.pow(x_true[:,:,:,0], 2) + K.pow(x_true[:,:,:,1], 2))
        # lps_pred = K.log(K.pow(x_pred[:,:,:,0], 2) + K.pow(x_pred[:,:,:,1], 2))
        # lps_loss = K.mean(K.square(lps_true - lps_pred))

        rec_loss = K.mean(K.square(x_true - x_pred))
        kl_loss = K.mean(-0.5 * K.sum(1.0 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1))
        return rec_loss + kl_loss

    def call(self, inputs):
        x_true, x_pred, z_mean, z_log_var = inputs
        loss = self.lossfun(x_true, x_pred, z_mean, z_log_var)
        self.add_loss(loss, inputs=inputs)
        return x_true


# loss function dummy layer
class LossLayer(Layer):
    __name__ = 'loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(LossLayer, self).__init__(**kwargs)

    def lossfun(self, x_true, x_pred, z):
        # log power spectrum loss
        # lps_true = K.log(K.pow(x_true[:,:,:,0], 2) + K.pow(x_true[:,:,:,1], 2))
        # lps_pred = K.log(K.pow(x_pred[:,:,:,0], 2) + K.pow(x_pred[:,:,:,1], 2))
        # lps_loss = K.mean(K.square(lps_true - lps_pred))

        rec_loss = K.mean(K.square(x_true - x_pred))
        return rec_loss

    def call(self, inputs):
        x_true, x_pred, z = inputs
        loss = self.lossfun(x_true, x_pred, z)
        self.add_loss(loss, inputs=inputs)
        return x_true

# tensorboard callback with LR tracking
class ExtendedTensorBoard(TensorBoard):
    def __init__(self, data_generator, **kwargs):
        super().__init__(**kwargs)
        self.data_generator = data_generator

    def on_epoch_end(self, epoch, logs=None):
        # add learning rate to logs
        logs.update({'lr': K.eval(self.model.optimizer.lr)})

        # adds stuff to validation_data (only 1 batch)
        s_noisy, s_true = None, None
        # NOTE loop over batches here
        s_noisy, s_true = self.data_generator[0]
        self.validation_data = [s_noisy, s_true]

        # call parent's func
        super().on_epoch_end(epoch, logs)


# learning rate scheduling function
def lr_schedule_func(initial_lr, drop_rate, drop_epochs):
    def func(epoch):
        return initial_lr * drop_rate ** math.floor((1+epoch)//drop_epochs)
    return func
