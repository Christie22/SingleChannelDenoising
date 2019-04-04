# Example of convolutional autoencoder model

from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPool2D, BatchNormalization, Flatten, Reshape, Dropout
from keras.models import Model
from keras import backend as K
import numpy as np
import ast

# TODO: change strings of the parsefile into int/floats/whatever is needed.
# TODO; see how to inject the good constant into Dense layers
# TODO; see how to integrate n_intermediate_dim & n_latent_dim
# model
class AEModelFactory(object):
    def __init__(
            self,
            input_shape,
            model_args):
        self.input_shape = input_shape
        self.architecture = model_args['architecture']
        self._encoder = None
        self._decoder = None
        self._model = None

    @staticmethod
    def get_lossfunc():
        def lossfunc(x_true, x_pred):
            return K.mean(K.square(x_true - x_pred))
        return lossfunc

    def get_encoder(self):
        return self._encoder

    def get_decoder(self):
        return self._decoder

    def get_model(self):
        if self._model is None:
            self.gen_model()
        return self._model


    def gen_encoder(self): 
        # ordered processing: every step is considered as a layer; the layers are ordered; 
        # the type of layer is now at the lowest level of the structure, along with the other params.
        print('[m] Just entered gen_ENcoder')
        inputs = Input(shape=self.input_shape)

        encoder = self.architecture['encoder']
        # n_layers = encoder['n_layers']
        all_layers = np.array([typ for typ in encoder]) #ex: Layer1, Layer2, Layer3
        # nb_layers = all_layers.shape
        print(type(all_layers))

        for i, layer in enumerate(all_layers):
            # print(type(all_layers))
            # print(i)
            print(layer) # layer = 'Layer'+str(i)
            attr = encoder[layer]
            # print('2. attr: {}'.format(attr))
            type_layer = attr['type_layer']
            print(type_layer)
            print(type(type_layer))
            # print('1. layers: {}'.format(type_layer))
            #del attr['type_layer']
            if i==0: # init 
                x = eval(type_layer + '(**attr)(inputs)' )
                print('3. x: {}'.format(x))
            else:
                x = eval(type_layer + '(**layer_attr)(x)' )
            if type_layer == 'Conv2D': 
                #calculate 'conv_shape'each time we compute this special type of layer even though we need only the last occurrence:
                self.conv_shape = K.int_shape(x)
        self._encoder = Model(inputs, x)
        #self._encoder.summary()
        self._encoder.name = 'encoder'

    def gen_decoder(self):
        # ordered processing: every step is considered as a layer; the layers are ordered; 
        # the type of layer is now at the lowest level of the structure, along with the other params.
        print('[m] Just entered gen_DEcoder')
        inputs = Input(shape=(self.n_latent_dim,))
        decoder = self.architecture['decoder']
        all_layers = np.array([typ for typ in decoder]) #ex: Layer1, Layer2, Layer3
        # nb_layers = all_layers.shape

        for i, layer in enumerate(all_layers):
            attr = decoder[layer]
            # print('2. attr: '.format(attr))
            type_layer = attr['type_layer']
            # print('1. layers'.format(type_layer))
            del attr['type_layer']
            if i==0: # init
                x = eval(type_layer + '(**attr)(inputs)' )
                print('3. x: '.format(x))
            else:
                x = eval(type_layer + '(**layer_attr)(x)' )

        self._decoder = Model(inputs, x)
        #self._decoder.summary()
        self._decoder.name = 'decoder'

    def gen_model(self):
        self.gen_encoder()
        self.gen_decoder()
        # TODO can we do something like here?
        # https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py#L159
        x_true = Input(shape=self.input_shape, name='input')
        z = self._encoder(x_true)
        x_pred = self._decoder(z)
        self._model = Model(inputs=[x_true], outputs=[x_pred])
