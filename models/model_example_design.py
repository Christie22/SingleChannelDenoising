# Example of convolutional autoencoder model

from keras.layers import Input, Dense, \
    Conv2D, Conv2DTranspose, \
        MaxPool2D, BatchNormalization, \
        Flatten, Reshape, Dropout, Activation, ELU, \
        LSTM, ConvLSTM2D
from keras.models import Model
from keras import backend as K
import numpy as np

# TODO: change strings of the parsefile into int/floats/whatever is needed.
# TODO; see how to inject the good constant into Dense layers
# TODO; see how to integrate n_intermediate_dim & n_latent_dim
# model
class AEModelFactory(object):
    dict_layers = {
        'conv': Conv2D,
        'convt': Conv2DTranspose,
        'dense': Dense,
        'activation': Activation,
        'batchnorm': BatchNormalization,
        'dropout': Dropout,
        'flatten': Flatten,
        'reshape': Reshape,
        'lstm': LSTM,
        'lstmconv': ConvLSTM2D,
        'elu': ELU
    }
    def __init__(
            self,
            input_shape,
            architecture):
        self.input_shape = input_shape
        self.architecture = architecture
        self._arch= None
        self._model= None


    @staticmethod
    def get_lossfunc(time_slice):
        def lossfunc(x_true, x_pred):
            return K.mean(K.square(x_true - x_pred))
        return lossfunc

    def get_model(self):
        if self._model is None:
            self.gen_model()
        return self._model


    def gen_arch(self): 
        # ordered processing: every step is considered as a layer; the layers are ordered; 
        # the type of layer is now at the lowest level of the structure, along with the other params.

        x = None
        inputs = Input(shape=self.input_shape)
        print(inputs)
        print(K.int_shape(inputs))
        conv_shape = np.zeros(K.int_shape(inputs)[1:])

        for layer in self.architecture:
            # get layer data
            layer_type = layer['layer_type']
            layer_args = layer['layer_args']

            # use conv_shape if needed
            if layer_type == 'dense' and layer_args['units'] == - 1:
                layer_args['units'] = np.prod(conv_shape)
            if layer_type == 'reshape' and layer_args['target_shape'] == - 1:
                layer_args['target_shape'] = conv_shape

            # debug info
            print('[] adding layer: {}  -  {}'.format(layer_type, layer_args))

            # if first layer, use input
            if x is None:
                x = inputs

            # declare layer with functional API
            x = AEModelFactory.dict_layers[layer_type](**layer_args)(x)

            # calculate shape each time we compute this special type of layer even though we need only the last occurrence:
            if layer_type == 'conv':
                conv_shape = K.int_shape(x)[1:]

        self._arch = Model(inputs, x)
        #self._encoder.summary()
        self._arch.name = 'arch'


    def gen_model(self):
        self.gen_arch()
        # NOTE no need to wrap model into another model
        #x_true = Input(shape=self.input_shape, name='input')
        #x_pred = self._arch(x_true)
        #self._model = Model(inputs=[x_true], outputs=[x_pred])
        # NOTE this is for compatibility sake
        self._model = self._arch
