# Utilities for various data handling

### Libs
import os
import glob
import json
import hashlib
import io
import types
import os.path as osp

import pandas as pd
import numpy as np
import librosa as lr
from keras.models import load_model
from keras import backend as K

import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.io.wavfile import read


def load_dataset(dataset_path):
    # TODO implement actual data handling
    # (requires figuring out data format)
    print('[u] Loading all wav files from {}'.format(dataset_path))
    filelist = glob.glob(osp.join(dataset_path, '*.wav'))
    print('[u] Loaded {} files'.format(len(filelist)))
    return filelist


def create_autoencoder_model(model_source, input_shape, template_args, return_descr=False, **kwargs):
    print('[u] Creating autoencoder model from {}'.format(model_source))
    print('[u] Model factory parameters: {}'.format({
        'input_shape': input_shape,
        'template_args': template_args,
        **kwargs
    }))
    # calc input shape and enforce it
    K.set_image_data_format('channels_last')
    # generate model
    from models.model_example_design import AEModelFactory
    obj = AEModelFactory(
        input_shape, model_source, template_args)
    model = obj.get_model()
    arch = obj.architecture
    descr = obj.description
    lossfunc = AEModelFactory.get_lossfunc(**kwargs)
    # return model and loss
    return (model, lossfunc, arch, descr) if return_descr else (model, lossfunc)


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


# calcualate md5 hash of input arguments
def hash_args(args):
    m = hashlib.md5()
    for x in args:
        m.update(str(x).encode())
    return m.hexdigest()[:6]


def store_logs(logs_path, new_log):
    try:
        if not osp.exists(logs_path):
            print('[u] Creating log file {}'.format(logs_path))
            with open(logs_path, 'w+') as f:
                json.dump([new_log], f)
            return

        with open(logs_path) as f:
            logs = json.load(f)

        os.remove(logs_path)
        logs.append(new_log)

        with open(logs_path, 'w+') as f:
            json.dump(logs, f)
    except Exception as e:
        print('[u] Exception while writing log: {} - {}'.format(type(e), e))
    

# return model.summary() as string
def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string
    

# get name of function or class from instance
def get_func_name(f):
    if isinstance(f, types.FunctionType) or hasattr(f, '__name__'):
        return f.__name__
    else:
        return f.__class__.__name__

