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
from libs.updated_utils import load_dataset, load_autoencoder_model
from libs.model_utils import LossLayer
from libs.data_generator import DataGenerator
from libs.processing import white_noise, s_to_reim


def results(model_path, dataset_path,
            sr, n_fft, hop_length, win_length, frag_hop_length, frag_win_length,
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
    filepath_list_train, filepath_list_valid = train_test_split(
        filepath_list, test_size=0.2, random_state=1337)

    # store DataGenerator args
    generator_args = {
        # dataset cfg
        'sr': sr,
        'cache_path': None,
        # noising/reverberation cfg
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
        'shuffle': True,
        'label_type': 'clean',
        'batch_size': batch_size,
    }
    print('[t] Data generator parameters: {}'.format(generator_args))

    # create DataGenerator objects
    training_generator = DataGenerator(filepath_list_train, **generator_args)
    train_steps_per_epoch = len(training_generator)
    print('[t] Train steps per epoch: ', train_steps_per_epoch)

    # load encoder
    print('loading encoder from {}...'.format(model_path))
    encoder, decoder, model = load_autoencoder_model(
        model_path, {
            'LossLayer': LossLayer
        })

    # run predictions
    encoded_train_data, _ = encoder.predict_generator(
        generator=testing_generator,
        steps=test_steps_per_epoch,
        #use_multiprocessing=True,
        #workers=8
    )

    print('Done! Check content of {} and {}'.format(args['latent_space_paths'], args['history_plot_path']))

