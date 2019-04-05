# Example of convolutional autoencoder model

from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPool2D, BatchNormalization, Flatten, Reshape, Dropout
from keras.models import Model
from keras import backend as K
import numpy as np

# TODO: change strings of the parsefile into int/floats/whatever is needed.
# TODO; see how to inject the good constant into Dense layers
# TODO; see how to integrate n_intermediate_dim & n_latent_dim
# model
class AEModelFactory(object):
    def __init__(
            self,
            input_shape,
            architecture):
        self.input_shape = input_shape
        self.architecture = architecture
        self._arch= None
        self._model= None

    @staticmethod
    def get_lossfunc():
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
        print('[m] Just entered gen_arch')
        # print('architecture: {}'.format(self.architecture))

        inputs = Input(shape=self.input_shape)

        dict_layers = {'conv': Conv2D, 'dense': Dense, 'batch': BatchNormalization, 'dropout': Dropout, 'flat': Flatten}

        all_layers = np.array([layer for layer in self.architecture]) #ex: Layer1, Layer2, Layer3
        # nb_layers = all_layers.shape
        print(type(all_layers))

        for i, layer in enumerate(all_layers):
            # print(i)
            print(layer) # layer = 'Layer'+str(i)
            layer_type = layer['layer_type']
            attr = layer['layer_args']
            print('[m] 20. layer_type: {}'.format(layer_type))
            print('[m] 21. attr: {}'.format(attr))
            # print('[m] '+layer_type)
            # print('[m] '+type(layer_type))
            x = inputs if i==0 else x # init 
            x = dict_layers[layer_type](**attr)
            print('[m], '+str(i) +', x: {}'.format(x))

            #calculate 'conv_shape'each time we compute this special type of layer even though we need only the last occurrence:
            self.conv_shape = K.int_shape(x) if layer_type == 'Conv2D' else self.conv_shape
        self._arch = Model(inputs, x)
        #self._encoder.summary()
        self._arch.name = 'arch'


    def gen_model(self):
        self.gen_arch()
        # TODO can we do something like here?
        # https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py#L159
        x_true = Input(shape=self.input_shape, name='input')
        x_pred = self._arch(x_true)
        self._model = Model(inputs=[x_true], outputs=[x_pred])
