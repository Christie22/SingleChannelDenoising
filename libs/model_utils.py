from keras.layers import Layer
from keras import backend as K
import tensorflow as tf


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

# loss function dummy layer
class VAELossLayer(Layer):
    __name__ = 'vae_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(VAELossLayer, self).__init__(**kwargs)

    def lossfun(self, x_true, x_pred, z_mean, z_log_var):
        # log power spectrum loss
        lps_true = K.log(K.pow(x_true[:,:,:,0], 2) + K.pow(x_true[:,:,:,1], 2))
        lps_pred = K.log(K.pow(x_pred[:,:,:,0], 2) + K.pow(x_pred[:,:,:,1], 2))
        lps_loss = K.mean(K.square(lps_true - lps_pred))

        rec_loss = K.mean(K.square(x_true - x_pred))
        kl_loss = K.mean(-0.5 * K.sum(1.0 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1))
        return rec_loss + kl_loss

    def call(self, inputs):
        x_true, x_pred, z_mean, z_log_var = inputs
        loss = self.lossfun(x_true, x_pred, z_mean, z_log_var)
        self.add_loss(loss, inputs=inputs)
        return x_true
