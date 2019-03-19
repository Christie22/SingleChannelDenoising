# test and evaluate trained model
# TODO calculate matrics on training and testing datasets and display/store them
# metrics: [165 from overview]
#  - SDR (source-to-distortion ratio) START FROM THIS!!
#  - STOI
#  - SIR (source-to- interference ratio)
#  - SAR (source-to-artifact ratio) 

import os
import pandas as pd
import numpy as np
import pickle
import itertools
import matplotlib.pyplot as plt
from keras.models import load_model

# custom modules
from libs.updated_utils import load_dataset, load_autoencoder_lossfunc, load_autoencoder_model
from libs.model_utils import LossLayer
from libs.data_generator import DataGenerator
from libs.processing import white_noise, s_to_reim
from libs.metrics import calc_metrics


def results(model_name, model_path, 
            dataset_path, sr, 
            rir_path, noise_snrs,
            n_fft, hop_length, win_length, frag_hop_length, frag_win_length,
            batch_size, cuda_device):
    print('[r] Calculating results for model {} on dataset {}'.format(model_path, dataset_path))
    print('[r] Parameters: {}'.format({
        'model_path': model_path,
        'dataset_path': dataset_path,
        'cuda_device': cuda_device,
    }))

    # set GPU devices
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    # load dataset filenames and split in train and validation
    filepath_list = load_dataset(dataset_path)

    # store DataGenerator args
    generator_args = {
        # dataset cfg
        'sr': sr,
        'cache_path': None,
        # noising/reverberation cfg
        'rir_path': rir_path,
        'noise_funcs': [None],
        'noise_snrs': noise_snrs,
        # stft cfg
        'n_fft': n_fft,
        'hop_length': hop_length,
        'win_length': win_length,
        # processing cfg
        'proc_func': s_to_reim,
        'proc_func_label': s_to_reim,
        # fragmenting cfg
        'frag_hop_length': frag_hop_length,
        'frag_win_length': frag_win_length,
        # general cfg
        'shuffle': False,
        'label_type': 'clean',
        'batch_size': batch_size,
        'disable_cacheinit': True
    }
    print('[t] Data generator parameters: {}'.format(generator_args))

    # create DataGenerator objects
    testing_generator = DataGenerator(filepath_list[:4], **generator_args)
    test_steps_per_epoch = len(testing_generator)
    print('[t] Test steps per epoch: ', test_steps_per_epoch)

    # load model
    print('[t] Loading model from {}...'.format(model_path))
    lossfunc = load_autoencoder_lossfunc(model_name)
    encoder, decoder, model = load_autoencoder_model(model_path, {'lossfunc': lossfunc})

    # predict data and calculate metrics, one batch at a time
    for i in range(len(testing_generator)):
        data_batch = testing_generator[i]
        y_true = data_batch[1]
        y_est = model.predict(data_batch,
            steps=test_steps_per_epoch,
            use_multiprocessing=True,
            workers=16
        )

        print("batch ", i)
        print(y_true.shape)
        print(y_est.shape)

    

    print('Done!')

