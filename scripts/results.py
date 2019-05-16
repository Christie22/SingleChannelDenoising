# test and evaluate trained model

import os
import os.path as osp
import itertools
import contextlib
import time
import pandas as pd
import numpy as np
import librosa as lr
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.models import load_model

from sklearn.metrics import mean_squared_error
from mir_eval.separation import bss_eval_sources
from pystoi.stoi import stoi as eval_stoi

# suppress mir_eval warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# custom modules
from libs.utilities import load_dataset, load_autoencoder_model, get_func_name
from libs.model_utils import LossLayer
from libs.data_generator import DataGenerator
from libs.processing import pink_noise, take_file_as_noise
from libs.processing import s_to_exp, s_to_reim, s_to_db
from libs.processing import exp_to_s, reim_to_s, db_to_s
from libs.processing import make_fragments, unmake_fragments, unmake_fragments_slice
from libs.metrics import sample_metric


def results(model_source, dataset_path, 
            sr, rir_path, noise_snrs,
            n_fft, hop_length, win_length, frag_hop_length, frag_win_length,
            batch_size, force_cacheinit, output_path, store_wavs, cuda_device):
    print('[r] Calculating results for model {} on dataset {}'.format(
        model_source, dataset_path))
    print('[r] Parameters: {}'.format({
        'model_source': model_source,
        'output_path': output_path,
        'dataset_path': dataset_path,
        'cuda_device': cuda_device,
        'force_cacheinit': force_cacheinit,
        'store_wavs': store_wavs
    }))

    # set GPU devices
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    # load dataset filenames
    filepath_list = load_dataset(dataset_path)
    rirpath_list = [None] # TODO do something with rir_path

    ## hyper-parameters (TODO un-hardcode some?)
    # noising functions
    noise_paths = [
        '/data/riccardo_datasets/demand/STRAFFIC/ch01.wav',
        '/data/riccardo_datasets/demand/TMETRO/ch01.wav',
        '/data/riccardo_datasets/demand/PCAFETER/ch01.wav',
        '/data/riccardo_datasets/demand/PRESTO/ch01.wav',
    ]
    noise_funcs = [
        pink_noise,
        #*[take_file_as_noise(f) for f in noise_paths]
    ]  
    # data processing function
    exponent = 1
    slice_width = 3
    proc_func = s_to_db
    unproc_func = db_to_s
    # loss function slice
    time_slice = slice((frag_win_length - slice_width) // 2,
                       (frag_win_length + slice_width) // 2)
    print('[t] Varius hyperparameters: {}'.format({
        'noise_paths': noise_paths,
        'noise_funcs': noise_funcs,
        'exponent': exponent,
        'proc_func': proc_func,
        'proc_func_label': proc_func,
        'time_slice': time_slice
    }))

    # calculate noise variations
    noise_variations = list(itertools.product(
        noise_funcs, noise_snrs, rirpath_list))

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
        'proc_func': proc_func,
        'proc_func_label': proc_func,
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

    # load model
    model, _ = load_autoencoder_model(
        model_source, time_slice=time_slice)
    # print model summary
    #model.summary()

    # list of filepath-noise_variation combinations
    file_noisevariation_prod = list(itertools.product(filepath_list, noise_variations))
    
    # metrics dataframe vars
    df_index = pd.MultiIndex.from_tuples(
        file_noisevariation_prod, names=['filepath', 'noise_variation'])
    df_columns = ['mse', 'sdr', 'sir', 'sar', 'stoi']
    df = pd.DataFrame(np.empty((len(df_index), len(df_columns))),
                      index=df_index, columns=df_columns)

    # generate folder structure
    subfolder = 'model_{}'.format(model.name)
    output_dir = osp.join(output_path, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    # generate complete path
    timestamp = time.strftime('%y%m%d_%H%M')
    filename = 'results_{}.pkl'.format(timestamp)
    output_filepath = osp.join(output_dir, filename)

    if store_wavs:
        wavs_list = []
        # add wav subfolder to results path (NOTE same name as the output pickle)
        output_dir_wav = osp.join(output_dir, 'results_{}_wav'.format(timestamp))
        os.makedirs(output_dir_wav, exist_ok=True)
        # path for wav files descriptor
        wavlist_filename = 'results_{}_wavlist.txt'.format(timestamp)
        wavlist_filepath = osp.join(output_dir_wav, wavlist_filename)

    # loop for filepath-noise_variation combinations
    pbar = tqdm(file_noisevariation_prod)
    for file_index, file_noisevariation in enumerate(pbar):
        # unpack data
        filepath, noise_variation = file_noisevariation
        noise_func, snr, _ = noise_variation

        # load speech 
        pbar.set_description('loading')
        x, _ = lr.load(filepath, sr=sr)
        # apply noise
        pbar.set_description('noising')
        x_noisy = noise_func(x=x, sr=sr, snr=snr)
        # convert both to TF-domain
        pbar.set_description('stft')
        s_clean = lr.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        s_noisy = lr.stft(x_noisy, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        # split into fragments
        pbar.set_description('split frags')
        s_frags_clean = make_fragments(s_clean, frag_hop_len=frag_hop_length, frag_win_len=frag_win_length)
        s_frags_noisy = make_fragments(s_noisy, frag_hop_len=frag_hop_length, frag_win_len=frag_win_length)
        # apply pre-processing (data representation)
        pbar.set_description('pre-proc')
        s_frags_noisy_proc = proc_func(s_frags_noisy)

        # predict data
        pbar.set_description('predict')
        s_frags_pred = model.predict(s_frags_noisy_proc)
        
        # unprocess (data representation)
        pbar.set_description('un-proc')
        s_frags_pred_unproc = unproc_func(s_frags_pred, s_noisy=s_frags_noisy)
        
        # handle variable names... TODO fix!
        s_noisy = s_frags_noisy
        s_true = s_frags_clean
        s_pred = s_frags_pred_unproc

#        # data temp variables
#        y_noisy = np.empty((n_batches, batch_size, *testing_generator.data_shape))
#        y_true = np.empty((n_batches, batch_size, *testing_generator.data_shape))
#        y_pred = np.empty((n_batches, batch_size, *testing_generator.data_shape))
#        # loop through batches
#        for batch_index in range(n_batches):
#            pbar.set_description('predicting {}/{} '.format(
#                batch_index, n_batches))
#            # predict data and sort out noisy and clean
#            y_noisy_batch, y_true_batch = testing_generator[batch_index]
#            y_noisy[batch_index] = y_noisy_batch
#            y_true[batch_index] = y_true_batch
#            y_pred[batch_index] = model.predict(y_noisy_batch)
#
#        # flatten along batches
#        pbar.set_description('reshaping')
#        y_noisy = y_noisy.reshape(-1, *testing_generator.data_shape)
#        y_true = y_true.reshape(-1, *testing_generator.data_shape)
#        y_pred = y_pred.reshape(-1, *testing_generator.data_shape)

#        # convert to complex spectrogram
#        pbar.set_description('post-proc')
#        s_noisy = unproc_func(y_noisy)
#        s_true = unproc_func(y_true)
#        s_pred = unproc_func(y_pred)

        # merge fragments
        pbar.set_description('merge frags')
        s_noisy = unmake_fragments_slice(
            s_noisy, frag_hop_len=frag_hop_length, 
            frag_win_len=frag_win_length, time_slice=time_slice)
        s_true = unmake_fragments_slice(
            s_true, frag_hop_len=frag_hop_length, 
            frag_win_len=frag_win_length, time_slice=time_slice)
        s_pred = unmake_fragments_slice(
            s_pred, frag_hop_len=frag_hop_length, 
            frag_win_len=frag_win_length, time_slice=time_slice)

        # trim spectrograms
        trim_slice = slice(2*frag_win_length, s_noisy.shape[1]-2*frag_win_length)
        s_noisy = s_noisy[:, trim_slice]
        s_true = s_true[:, trim_slice]
        s_pred = s_pred[:, trim_slice]

        # get waveforms
        pbar.set_description('istft')
        x_noisy = lr.istft(s_noisy, hop_length=hop_length, win_length=win_length)
        x_true = lr.istft(s_true, hop_length= hop_length, win_length=win_length)
        x_pred = lr.istft(s_pred, hop_length=hop_length, win_length=win_length)
            
        # store audio
        if store_wavs:
            pbar.set_description('storing')
            # store files
            for x, type in [(x_noisy, 'x_noisy'), (x_true, 'x_true'), (x_pred, 'x_pred')]:
                filename_wav = '{}_{}.wav'.format(file_index, type)
                filepath_wav = osp.join(output_dir_wav, filename_wav)
                lr.output.write_wav(filepath_wav, y=x, sr=sr)
                wav_list_entry = '{:3}: {} {}'.format(
                    file_index, filepath, noise_variation)
                wavs_list.append(wav_list_entry)
            # store descriptor
            with open(wavlist_filepath, 'w') as f:
                print('\n'.join(wavs_list), file=f)
            
        # METRIC 1: mean squared error
        pbar.set_description('metrics (mse)')
        mse = mean_squared_error(np.abs(s_pred), np.abs(s_true))

        # METRIC 2: sdr, sir, sar
        pbar.set_description('metrics (bss)')
        src_true = np.array([
            x_true,         # true clean
            x_noisy-x_true  # true noise (-ish)
        ])
        src_pred = np.array([
            x_pred,         # predicted clean
            x_noisy-x_pred  # predicted noise (-ish)
        ])
        sdr, sir, sar, _ = bss_eval_sources(src_true, src_pred)

        # METRIC 3: stoi
        pbar.set_description('metrics (stoi)')
        stoi = eval_stoi(x=x_true, y=x_pred, fs_sig=sr, extended=False)

        # store metrics
        df.loc[file_noisevariation, 'mse'] = mse
        df.loc[file_noisevariation, 'sdr'] = sdr[0]
        df.loc[file_noisevariation, 'sir'] = sir[0]
        df.loc[file_noisevariation, 'sar'] = sar[0]
        df.loc[file_noisevariation, 'stoi'] = stoi
    
    
    # store dataframe as pickle (NOTE load with pd.read_pickle)
    df.to_pickle(output_filepath)

    print('[r] Results stored in {}'.format(output_filepath))
    print(df)
    print('[r] Done!')


