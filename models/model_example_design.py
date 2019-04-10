# Example of convolutional autoencoder model

import json
import os.path as osp
import numpy as np
from keras.layers import Input, Dense, \
        Conv1D, Conv2D, Conv2DTranspose, \
        MaxPool2D, BatchNormalization, \
        Flatten, Reshape, Permute, \
        Dropout, Activation, ELU, \
        LSTM, ConvLSTM2D, GRU
from keras.models import Model
from keras import backend as K

from libs.utilities import hash_args


# model factory class
class AEModelFactory(object):
    dict_layers = {
        'conv1d': Conv1D,
        'conv2d': Conv2D,
        'conv2dt': Conv2DTranspose,
        'dense': Dense,
        'activation': Activation,
        'batchnorm': BatchNormalization,
        'dropout': Dropout,
        'flatten': Flatten,
        'reshape': Reshape,
        'permute': Permute,
        'lstm': LSTM,
        'lstmconv2d': ConvLSTM2D,
        'gru': GRU,
        'elu': ELU,
    }
    def __init__(
            self,
            input_shape,
            arch_path,
            template_args={}):
        self.input_shape = input_shape
        self._model= None
        # open model arch json file
        with open(arch_path) as f:
            str_data = f.read()
        # if template, process it
        if osp.splitext(arch_path)[1] == '.jsont':
            str_data = self.process_template(str_data, **template_args)
        # store data as dict
        self._architecture = json.loads(str_data)['architecture']

    def process_template(self, str_data, **kwargs):
        print('[m] Processing template...')
        # swap single and double curlies
        str_data = str_data.replace('{{', '__')
        str_data = str_data.replace('{', '{{')
        str_data = str_data.replace('__', '{')
        str_data = str_data.replace('}}', '__')
        str_data = str_data.replace('}', '}}')
        str_data = str_data.replace('__', '}')
        # replace template args
        return str_data.format(**kwargs)

    @staticmethod
    def get_lossfunc(time_slice):
        def lossfunc(x_true, x_pred):
            return K.mean(K.square(x_true[..., time_slice, :] - x_pred[..., time_slice, :]))
        return lossfunc

    def get_model(self):
        if self._model is None:
            self.gen_model()
        return self._model


    def gen_model(self): 
        # ordered processing: every step is considered as a layer; the layers are ordered; 
        # the type of layer is now at the lowest level of the structure, along with the other params.

        # store some initial values
        x = None
        inputs = Input(shape=self.input_shape)
        conv_shape = np.zeros(len(K.int_shape(inputs)[1:]), dtype=int)

        for layer in self._architecture:
            # get layer data
            layer_type = layer['layer_type']
            layer_args = layer['layer_args']

            # use conv_shape if needed
            if layer_type == 'dense' and layer_args['units'] == - 1:
                layer_args['units'] = np.prod(conv_shape)
            if layer_type == 'reshape' and layer_args['target_shape'] == - 1:
                layer_args['target_shape'] = conv_shape

            # debug info
            print('[m] Adding layer {}  -  {}'.format(layer_type.upper(), layer_args))

            # if first layer, use input
            if x is None:
                x = inputs

            # declare layer with functional API
            x = AEModelFactory.dict_layers[layer_type](**layer_args)(x)

            # calculate shape each time we compute this special type of layer even though we need only the last occurrence:
            if layer_type == 'conv2d' or layer_type == 'conv1d':
                conv_shape = K.int_shape(x)[1:]

        self._model = Model(inputs, x)
        self._model.name = hash_args(self._architecture)
