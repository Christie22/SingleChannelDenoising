#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 10:24:20 2019

@author: test
"""

###### Merging utilities.py & sm_utils.py ######

### Libs
import os
import pandas as pd
from keras.models import load_model
import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt
from scipy.stats import norm
import librosa
import librosa.display
import librosa.feature as ftr
import librosa.onset as onst


### Functions
# Create
def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

# Objects' Generation 
def gen_filename(params, extention=None, extra=None):
    params['extention'] = extention
    params['extra'] = extra
    return '_'.join([
        '{kernel_size}',
        '{n_filters}',
        '{n_intermediate_dim}',
        '{n_latent_dim}',
        '{epochs}',
        '{batch_size}' +
            ('_{extra}' if extra is not None else '') +
            ('.{extention}' if extention is not None else '')
    ]).format(**params)
    
def gen_model_filepath(params):
    model_dir = os.path.join('models', '{pre_processing}_{model}'.format(**params))
    create_folder(model_dir)
    model_name = gen_filename(params, 'h5')
    model_path = os.path.join(model_dir, model_name)
    return model_path

def gen_logs_dir(params):
    logs_dir = os.path.join('logs', '{pre_processing}_{model}'.format(**params))
    create_folder(logs_dir)
    return logs_dir

def gen_output_dir(params):
    output_dir = os.path.join('output', '{pre_processing}_{model}'.format(**params))
    output_curr_dir_name = gen_filename(params)
    output_curr_dir = os.path.join(output_dir, output_curr_dir_name)
    create_folder(output_curr_dir)
    return output_curr_dir

# Loading 
def load_dataset(params):
    annotation_path = os.path.expanduser(os.path.join(params['annotation_path']))
    print('Reading annotation files from {}'.format(annotation_path))
    df = pd.read_csv(annotation_path, engine='python')
    # filenames = df['filepath'].values.tolist()
    # labels = df.drop(columns='filepath').to_dict('list')
    return df

def load_model_vae(custom_objects, params):
    model_dir = os.path.join('models', '{pre_processing}_{model}'.format(**params))
    model_name = gen_filename(params, 'h5')
    model_path = os.path.join(model_dir, model_name)
    model = load_model(model_path, custom_objects=custom_objects)
    # extract encoder from main model
    encoder = model.get_layer('vae_encoder')
    decoder = model.get_layer('vae_decoder')
    return encoder, decoder, model

def load_test_data(params):
    test_dir = os.path.join('test_data', '{pre_processing}_{model}'.format(**params))
    test_name = gen_filename(params, 'pkl')
    test_path = os.path.join(test_dir, test_name)
    df = pd.read_pickle(test_path)
    return df

def load_all_as_test_data(params):
    annotation_path = os.path.expanduser(os.path.join(params['annotation_path']))
    print('Reading annotation files from {}'.format(annotation_path))
    df = pd.read_csv(annotation_path, engine='python')
    return df

def load_history_data(params):
    history_dir = os.path.join('history', '{pre_processing}_{model}'.format(**params))
    history_name = gen_filename(params, 'pkl')
    history_path = os.path.join(history_dir, history_name)
    return pd.read_pickle(history_path)

# Storing
def store_history_data(history, params):
    # build path
    history_dir = os.path.join('history', '{pre_processing}_{model}'.format(**params))
    create_folder(history_dir)
    history_name = gen_filename(params, 'pkl')
    history_path = os.path.join(history_dir, history_name)
    # store data
    df = pd.DataFrame(history.history)
    df.to_pickle(history_path)
    print('')
    
def store_history_plot(fig, params):
    output_dir = gen_output_dir(params)
    output_history_path = os.path.join(output_dir, 'history.png')
    fig.savefig(output_history_path)

def store_latent_space_plot(fig, params):
    output_dir = gen_output_dir(params)
    output_latent_space_path = os.path.join(output_dir, 'latent_space.png')
    fig.savefig(output_latent_space_path)

def store_manifold_plot(fig, params):
    output_dir = gen_output_dir(params)
    output_manifold_path = os.path.join(output_dir, 'manifold.png')
    fig.savefig(output_manifold_path)

# Arbitrarily chosen constant:
norm_coeff = 100

# Input's processing
def norm_channel(ch):
    #ch = 1.0 / (1.0 + np.exp(-norm_coeff*ch))
    return ch

def process_data(s):
    # remove a bin if odd number
    if s.shape[0] % 2 != 0:
        s = s[:-1]
    # split re/im
    re = np.real(s)
    im = np.imag(s)
    re = norm_channel(re)
    im = norm_channel(im)
    s = np.dstack((re, im))
    return s

# Output's inverse processing
def denorm_channel(ch):
    #ch = np.log(-ch / (ch - 1.0)) / norm_coeff
    return ch

def unprocess_data(s):
    # should return a complex spectrum
    # convert to complex:
    re = s[:,:,0]
    im = s[:,:,1]
    re = denorm_channel(re)
    im = denorm_channel(im)
    s = re + 1j*im
    # adding previously removed bin
    padding = np.zeros((1, *s.shape[1:]))
    s = np.concatenate((s, padding))
    return s

# Plotting

# Import predictions
def import_spectrogram(filepath):
    if os.path.isfile(filepath):
        print('Reading data from `{}`...'.format(filepath))
        data = np.load(filepath)
        print('Data loaded. Shape = {}'.format(data.shape))
        return data
    else:
        print('Wrong path: `{}`'.format(filepath))
        quit()


def plot_spectrogram(spect,title_name):
    spect_db = librosa.amplitude_to_db(np.abs(spect), ref=np.max) #should it be power_to_db?
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(spect_db, y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title_name)
    plt.savefig('outputs/spectrogram_'+title_name+'.png')
    
def plot_manifold(n, latent_dim_vals, decoder):
    fig, ax = plt.subplots(n, n, tight_layout=True, figsize=(20,20))
    z_sample = latent_dim_vals
    x = z_sample[0,0]
    y = z_sample[0,1]
    grid_x = norm.ppf(np.linspace(x - 0.5, x + 0.5, n))
    grid_y = norm.ppf(np.linspace(y - 0.5, y + 0.5, n))
    print('Figure shape : ', ax.shape)
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample[0,0] = xi
            z_sample[0,1] = yi
            #print(z_sample.shape)
            x_decoded = decoder.predict(z_sample)
            #print(x_decoded)
            x_decoded = unprocess_data(x_decoded[0])
            #np.set_printoptions(threshold=np.nan)
            img = librosa.amplitude_to_db(np.abs(x_decoded))
            x_spec = librosa.display.specshow(img, y_axis='log', ax=ax[i,j])
            print(x_spec)
            ax[i,j] = x_spec

    fig.savefig('outputs/manifold.png')

# Calculate all features
def calc_features_rand(dataset_params, specs):
    # for now it generates just random features
    L = len(specs)
    features = {
        'harmon':[],
        'sth':[],
        'what':[]
        }
    for i in features:
        features[i].extend(np.random.rand(L))

    return features


def calc_features(dataset_params, input_spectogram):
    n_fft = dataset_params['n_fft']
    hop_length = dataset_params['hop_length']
    samplerate = dataset_params['samplerate']
    centroids = []
    centroids_dev = []
    contrasts = []
    flatnesses = []
    zeroCrossings = []
    mfccs = []
    onset_strengths = []

    for spec in input_spectogram:
        #compute magnitude spectrum
        y = unprocess_data(spec)
        magSpec = np.abs(y)
        t = librosa.core.istft(y, hop_length=hop_length, win_length=n_fft)

        spectral_centroid = ftr.spectral_centroid(S=magSpec, sr=samplerate, n_fft=n_fft, hop_length=hop_length)
        spectral_contrast = ftr.spectral_contrast(S=magSpec, sr=samplerate, n_fft=n_fft, hop_length=hop_length)
        spectral_flatness = ftr.spectral_flatness(S=magSpec, n_fft=n_fft, hop_length=hop_length)
        spectral_flatness = 10*np.log10(spectral_flatness) #spectral flatness in dB
        zero_crossing_rate = ftr.zero_crossing_rate(y=t)
        mfcc = ftr.mfcc(y=t, sr=samplerate)
        onset_strength = onst.onset_strength(y=t, sr=samplerate) #in case we want to test it with drums

        centroids.append(np.mean(spectral_centroid))
        centroids_dev.append(np.mean(np.std(spectral_centroid)))
        contrasts.append(spectral_contrast)
        flatnesses.append(np.mean(spectral_flatness)) #in dB
        zeroCrossings.append(np.mean(zero_crossing_rate))
        mfccs.append(mfcc)
        onset_strengths.append(onset_strength)

        
    output = {'centroid_frequency' : centroids, 'centroid_frequency_dev' : centroids_dev, 'spectral_contrasts' : contrasts, 'spectral_flatness' : flatnesses, 'zero_crossing_rate' : zeroCrossings, 'MFCC' : mfccs, 'onset_strength' : onset_strengths}
    return output




def calc_metrics(params, yest, y):
    # calc SDR and NRR
    "SDR: ref: C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen, 'An Algorithm for Intelligibility Prediction of Time-Frequency Weighted Noisy Speech,' IEEE/ACM"
    
    beta = -15 #lower SDR bound
  
    # evaluation parametres
    tBands= params['tbands']
    fBands= params['fbands']
    norm_yest = params['norm_yest']
    clip_yest= params['clip_yest']
    
    
    if fBands <= 1: #time domain
        SDR,NRR = np.zeros(tBands),np.zeros(tBands)
        if norm_yest:
            yest =[LA.norm(y)/LA.norm(yest) * el_yest for el_yest in yest]
        if clip_yest:
            lowerBound =[(1+ 10**(-beta/20)) * ely for ely in y]
            yest =[np.min([elyest, ellowerBound]) for elyest, ellowerBound in zip(yest,lowerBound)]
        
        if tBands > 0: #cut y into pieces
            linscale= np.round(np.linspace(0,y.shape[0],tBands))
            
        for n in range(tBands): 
            
            # SIgnal 2 Distorsion Ratio:
            diffTrue2Est= [a-b for a,b in zip( y[np.int(linscale[n]) : np.int(linscale[n+1])],\
                                                yest[np.int(linscale[n]) : np.int(linscale[n+1])] )]
            numSDR = np.mean(np.square(diffTrue2Est))    
            denomSDR= np.mean(np.square(yest[np.int(linscale[n]) : np.int(linscale[n+1])]))
            # formula uses inverse ratio inside the log, compared to the paper cited 
            SDR[n]=10* np.log10(numSDR / denomSDR)
            
            # Noise Reduction Ratio:
            numNR =np.mean(np.square( y[np.int(linscale[n]) : np.int(linscale[n+1])] ))
            denomNR= np.mean(np.square( yest[np.int(linscale[n]) : np.int(linscale[n+1])] ))
            NRR[n]= 10* np.log10(numNR / denomNR)
        
        
    else: #fBands > 1, do it in the complex domain
        
        SDR,NRR = np.zeros(fBands,tBands),np.zeros(fBands,tBands)

        n_fft = params['n_fft']
        hop_length = params['hop_length']
        samplerate = params['samplerate']
        
        ## compute stft
        Yest= librosa.core.stft(yest, hop_length=hop_length, win_length=n_fft)
        Y= librosa.core.stft(y, hop_length=hop_length, win_length=n_fft)
          
        ## calculate the grid for the calculation of the metrics
        # Frequence: just to get a log scale:
        logscale = librosa.mel_frequencies(n_mels=fBands,fmin=0,fmax=samplerate/2)
        #librosa.fft_frequencies(sr=22050, n_fft=subbands)
        stepsF = np.floor(logscale/logscale[-1]* Y.shape[0])
        
        # Time:
        if tBands > 0:
            stepsT = np.round(np.linspace(0,y.shape[1],tBands))
        else:
            stepsT= np.array([0,y.shape[0]])
            
        #Ytemp=Y
        for nf in range(fBands):
            for nt in range(tBands):
                #if norm_yest and clip_yest:
                #Ytemp[nf][nt] = np.min([LA.norm(Y[nf][nt])/LA.norm(Yest[nf][nt]) * Yest[nf][nt], (1+ 10**(-beta/20) * Y[j][m] )])

                # SIgnal 2 Distorsion Ratio:
                diffTrue2Est= [a-b for a,b in zip( \
                        Y[ np.int(stepsF[n]) : np.int(stepsF[n+1]) ][ np.int(stepsT[n]) : np.int(stepsT[n+1]) ],\
                        Yest[ np.int(stepsF[n]) : np.int(stepsF[n+1]) ][ np.int(stepsT[n]) : np.int(stepsT[n+1])]) ]
                numSDR = np.mean(np.square(diffTrue2Est))    
                denomSDR= np.mean( np.square( Yest[np.int(stepsF[n]):np.int(stepsF[n+1])][np.int(stepsT[n]):np.int(stepsT[n+1])]))
                # formula uses inverse ratio inside the log, compared to the paper cited 
                SDR[nf][nt]= 10*np.log10(numSDR / denomSDR)
                
                # Noise Reduction Ratio:
                numNR =np.mean(np.square( y[np.int(linscale[n]) : np.int(linscale[n+1])] ))
                denomNR= np.mean(np.square( yest[np.int(linscale[n]) : np.int(linscale[n+1])] ))
                NRR[nf][nt]= 10* np.log10(numNR / denomNR) 
    
    
    # calc Noise Reduction ratio NR
    
    output = {'Signal-To-Distorsion Ratio (SDR)' : SDR, 'Noise Reduction Ratio' : NRR}
    return output

    






