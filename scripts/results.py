# test and evaluate trained model
# TODO calculate metrics on training and testing datasets and display/store them
# metrics: [165 from overview]
#  - SDR (source-to-distortion ratio) START FROM THIS!!
#  - STOI
#  - SIR (source-to- interference ratio)
#  - SAR (source-to-artifact ratio) 

import os
import pandas as pd
import numpy as np
import librosa as lr
import matplotlib.pyplot as plt
from tqdm import trange
from keras.models import load_model
from mir_eval.separation import bss_eval_sources

# suppress mir_eval warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# custom modules
from libs.utilities import load_dataset, load_autoencoder_lossfunc, load_autoencoder_model
from libs.model_utils import LossLayer
from libs.data_generator import DataGenerator
from libs.processing import white_noise, s_to_reim, reim_to_s, unmake_fragments
from libs.metrics import calc_metrics, sample_metric


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
        'noise_funcs': [white_noise],
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
        'batch_size': batch_size
    }
    print('[r] Data generator parameters: {}'.format(generator_args))

    # create DataGenerator objects
    testing_generator = DataGenerator(filepath_list[:4], **generator_args)
    test_steps_per_epoch = len(testing_generator)
    print('[r] Test steps per epoch: ', test_steps_per_epoch)

    # load model
    print('[r] Loading model from {}...'.format(model_path))
    lossfunc = load_autoencoder_lossfunc(model_name)
    _, _, model = load_autoencoder_model(model_path, {'lossfunc': lossfunc})

    # metrics data structure
    n_batches = len(testing_generator)
    metrics = {
        'mse': np.zeros(n_batches),
        'sdr': np.zeros(n_batches),
        'sar': np.zeros(n_batches),
        'sir': np.zeros(n_batches),
    }

    # loop through batches
    for batch_index in trange(len(testing_generator)):
        data_batch = testing_generator[batch_index]
        y_noisy_batch = data_batch[0]
        y_true_batch = data_batch[1]
        y_pred_batch = model.predict(y_noisy_batch)

        # convert to complex spectrogram
        s_noisy_batch = reim_to_s(y_noisy_batch)
        s_true_batch = reim_to_s(y_true_batch)
        s_pred_batch = reim_to_s(y_pred_batch)

        # merge batches
        s_noisy = unmake_fragments(s_noisy_batch, frag_hop_len=frag_hop_length, frag_win_len=frag_win_length)
        s_true = unmake_fragments(s_true_batch, frag_hop_len=frag_hop_length, frag_win_len=frag_win_length)
        s_pred = unmake_fragments(s_pred_batch, frag_hop_len=frag_hop_length, frag_win_len=frag_win_length)

        ## TODO remove?   loop through steps per batch
        ##for step_index, y_noisy, y_pred, y_true in zip(range(batch_size), y_noisy_batch, y_pred_batch, y_true_batch):
        
        # get absolute spectrogram
        s_noisy = np.abs(s_noisy) ** 2
        s_true = np.abs(s_true) ** 2
        s_pred = np.abs(s_pred) ** 2

        # get waveform
        x_noisy = lr.istft(s_noisy, hop_length=hop_length, win_length=win_length)
        x_true = lr.istft(s_true, hop_length= hop_length, win_length=win_length)
        x_pred = lr.istft(s_pred, hop_length=hop_length, win_length=win_length)
        
        # METRIC 1: mean squared error
        mse = sample_metric(s_pred, s_true)

        # METRIC 2: sdr, sir, sar
        src_true = np.array([
            x_true,        # true clean
            x_noisy-x_true # true noise (-ish)
        ])
        src_pred = np.array([
            x_pred,         # predicted clean
            x_noisy-x_pred  # predicted noise (-ish)
        ])
        sdr, sir, sar, _ = bss_eval_sources(src_true, src_pred)

        # store metrics
        metrics['mse'][batch_index] = mse
        metrics['sdr'][batch_index] = sdr[0]
        metrics['sir'][batch_index] = sir[0]
        metrics['sar'][batch_index] = sar[0]

    print('[r] Results:')
    print('[r]   Average MSE: {}'.format(metrics['mse'].mean()))
    print('[r]   Average SDR: {}'.format(metrics['sdr'].mean()))
    print('[r]   Average SIR: {}'.format(metrics['sir'].mean()))
    print('[r]   Average SAR: {}'.format(metrics['sar'].mean()))
    print('[r] Done!')


