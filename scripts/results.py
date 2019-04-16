# test and evaluate trained model
# TODO calculate metrics on training and testing datasets and display/store them
# metrics: [165 from overview]
#  - SDR (source-to-distortion ratio) START FROM THIS!!
#  - STOI
#  - SIR (source-to- interference ratio)
#  - SAR (source-to-artifact ratio) 

import os
import os.path as osp
import itertools
import time
import pandas as pd
import numpy as np
import librosa as lr
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.models import load_model
from mir_eval.separation import bss_eval_sources

# suppress mir_eval warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# custom modules
from libs.utilities import load_dataset, load_autoencoder_model, get_func_name
from libs.model_utils import LossLayer
from libs.data_generator import DataGenerator
from libs.processing import pink_noise, s_to_exp, exp_to_s, unmake_fragments
from libs.metrics import calc_metrics, sample_metric


def results(model_source, 
            dataset_path, sr, 
            rir_path, noise_snrs,
            n_fft, hop_length, win_length, frag_hop_length, frag_win_length,
            batch_size, force_cacheinit, output_path, cuda_device):
    print('[r] Calculating results for model {} on dataset {}'.format(
        model_source, dataset_path))
    print('[r] Parameters: {}'.format({
        'model_source': model_source,
        'output_path': output_path,
        'dataset_path': dataset_path,
        'cuda_device': cuda_device,
        'force_cacheinit': force_cacheinit
    }))

    # processing/unprocessing functions
    # TODO un-hardcode
    noise_funcs = [
        pink_noise 
    ]
    proc_func = s_to_exp(1.0/6)
    unproc_func = exp_to_s(1.0/6)
    rir_filepaths = [None]

    # set GPU devices
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    # load dataset filenames
    filepath_list = load_dataset(dataset_path)

    # calculate noise variations
    noise_variations = list(itertools.product(
        noise_funcs, noise_snrs, rir_filepaths))

    # store DataGenerator args
    generator_args = {
        # dataset cfg
        'sr': sr,
        'cache_path': None,
        # stft cfg
        'n_fft': n_fft,
        'hop_length': hop_length,
        'win_length': win_length,
        # processing cfg
        'proc_func': proc_func,  # TODO un-hardcode
        'proc_func_label': proc_func,  # TODO un-hardcode
        # fragmenting cfg
        'frag_hop_length': frag_hop_length,
        'frag_win_length': frag_win_length,
        # general cfg
        'shuffle': False,
        'label_type': 'clean',
        'normalize': False,
        'batch_size': batch_size,
        'force_cacheinit': force_cacheinit,
    }
    print('[r] Data generator parameters: {}'.format(generator_args))

    # loss function: data slice under consideration
    time_slice = frag_win_length // 2

    # load model
    model, _ = load_autoencoder_model(
        model_source, time_slice=time_slice)
    # print model summary
    model.summary()

    # list of filepath-noise_variation combinations
    file_noisevariation_prod = list(itertools.product(filepath_list, noise_variations))
    
    # metrics dataframe vars
    df_index = pd.MultiIndex.from_tuples(
        file_noisevariation_prod, names=['filepath', 'noise_variation'])
    df_columns = ['mse', 'sdr', 'sir', 'sar']
    df = pd.DataFrame(np.empty((len(df_index), len(df_columns))),
                      index=df_index, columns=df_columns)

    # loop for filepath-noise_variation combinations
    pbar = tqdm(file_noisevariation_prod)
    for file_noisevariation in pbar:
        # unpack data
        filepath, noise_variation = file_noisevariation
        noise_func, snr, _ = noise_variation
        # create DataGenerator objects (one file at a time!)
        pbar.set_description('{} @ {}'.format(osp.basename(filepath), noise_variation))
        testing_generator = DataGenerator(
            filepaths=[filepath], 
            noise_funcs=[noise_func],
            noise_snrs=[snr],
            rir_path=rir_path,  # TODO un-hardcode
            **generator_args)
        n_batches = len(testing_generator)

        # data temp variables
        y_noisy = np.empty((n_batches, batch_size, *testing_generator.data_shape))
        y_true = np.empty((n_batches, batch_size, *testing_generator.data_shape))
        y_pred = np.empty((n_batches, batch_size, *testing_generator.data_shape))
        # loop through batches
        for batch_index in range(n_batches):
            pbar.set_description('predicting {}/{} '.format(
                batch_index, n_batches))
            # predict data and sort out noisy and clean
            y_noisy_batch, y_true_batch = testing_generator[batch_index]
            y_noisy[batch_index] = y_noisy_batch
            y_true[batch_index] = y_true_batch
            y_pred[batch_index] = model.predict(y_noisy_batch)

        # flatten along batches
        pbar.set_description('reshaping')
        y_noisy = y_noisy.reshape(-1, *testing_generator.data_shape)
        y_true = y_true.reshape(-1, *testing_generator.data_shape)
        y_pred = y_pred.reshape(-1, *testing_generator.data_shape)

        # convert to complex spectrogram
        pbar.set_description('post-proc')
        s_noisy = unproc_func(y_noisy)
        s_true = unproc_func(y_true)
        s_pred = unproc_func(y_pred)

        # merge batches
        pbar.set_description('merge frags')
        s_noisy = unmake_fragments(s_noisy, frag_hop_len=frag_hop_length, frag_win_len=frag_win_length)
        s_true = unmake_fragments(s_true, frag_hop_len=frag_hop_length, frag_win_len=frag_win_length)
        s_pred = unmake_fragments(s_pred, frag_hop_len=frag_hop_length, frag_win_len=frag_win_length)

        # get waveform
        pbar.set_description('istft')
        x_noisy = lr.istft(s_noisy, hop_length=hop_length, win_length=win_length)
        x_true = lr.istft(s_true, hop_length= hop_length, win_length=win_length)
        x_pred = lr.istft(s_pred, hop_length=hop_length, win_length=win_length)
            
        # METRIC 1: mean squared error
        pbar.set_description('metrics (1)')
        mse = sample_metric(s_pred, s_true)

        # METRIC 2: sdr, sir, sar
        pbar.set_description('metrics (2)')
        src_true = np.array([
            x_true,         # true clean
            x_noisy-x_true  # true noise (-ish)
        ])
        src_pred = np.array([
            x_pred,         # predicted clean
            x_noisy-x_pred  # predicted noise (-ish)
        ])
        sdr, sir, sar, _ = bss_eval_sources(src_true, src_pred)

        # store metrics
        df.loc[file_noisevariation, 'mse'] = mse
        df.loc[file_noisevariation, 'sdr'] = sdr[0]
        df.loc[file_noisevariation, 'sir'] = sir[0]
        df.loc[file_noisevariation, 'sar'] = sar[0]
    
    # store results in pandas dataframe
    timestamp = time.strftime('%y%m%d_%H%M')
    output_path = output_path.format(model_name=model.name, ts=timestamp)
    df.to_pickle(output_path)



    print('[r] Results:')
    print(df)
    print('[r] Done!')


