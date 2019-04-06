# Utilities for various data handling

### Libs
import os
import pandas as pd
from keras.models import load_model
from keras import backend as K
import json

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.io.wavfile import read
import librosa
import librosa.display
import librosa.feature as ftr
import librosa.onset as onst
import sys
import glob

def load_dataset(dataset_path):
    # TODO implement actual data handling
    # (requires figuring out data format)
    print('[u] Loading all wav files from {}'.format(dataset_path))
    filelist = glob.glob(os.path.join(dataset_path, '*.wav'))
    print('[u] Loaded {} files'.format(len(filelist)))
    return filelist


def create_autoencoder_model(model_source, input_shape, **kwargs):
    print('[u] Creating autoencoder model from {}'.format(model_source))
    # import model factory
    from models.model_example_design import AEModelFactory
    # load model architecture configuration as dictionary
    model_architecture = json.load(open(model_source))
    print('[u] Model factory parameters: {}'.format({
        'model_architecture': model_architecture,
        'input_shape': input_shape,
        **kwargs
    }))
    # calc input shape and enforce it
    K.set_image_data_format('channels_last')
    # generate model
    obj = AEModelFactory(input_shape, model_architecture['architecture'])
    model = obj.get_model()
    # return model and loss
    return model, AEModelFactory.get_lossfunc(**kwargs)


def load_autoencoder_model(model_path, custom_objects=None):
    print('[u] Loading autoencoder model from {}'.format(model_path))
    model = load_model(model_path, custom_objects=custom_objects)
    # extract encoder from main model
    #encoder = model.get_layer('encoder')
    #decoder = model.get_layer('decoder')
    # NOTE compatibility sake
    return None, None, model


def load_autoencoder_lossfunc(time_slice):
    print('[u] Loading loss function for model')
    from models.model_example import AEModelFactory
    # return loss function
    return AEModelFactory.get_lossfunc(time_slice)
