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


######### v FUNCTIONS THAT ARE ACTUALLY USED v ############

def load_dataset(dataset_path):
    # TODO implement actual data handling
    # (requires figuring out data format)
    print('Loading dataset!')
    with open(dataset_path, 'rb') as f:
        print(f)
    return None

def create_autoencoder_model(custom_objects, model_name, input_shape):
    # import model factory
    if model_name == 'lstm':
        print('Using model `{}` from {}'.format(model_name, 'model_lstm'))
    elif model_name == 'conv':
        print('Using model `{}` from {}'.format(model_name, 'model_conv'))
    else:
        print('importing example model :D')
        import models.model_example as m

    # calc input shape and enforce it
    K.set_image_data_format('channels_last')
    # generate model
    obj = m.AEModelFactory(input_shape=input_shape)
    model = obj.get_model()
    return model



######### v NOT VERIFIED OR USED v ############

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


#def calc_features(dataset_params, input_spectogram):
#    n_fft = dataset_params['n_fft']
#    hop_length = dataset_params['hop_length']
#    samplerate = dataset_params['samplerate']
#    centroids = []
#    centroids_dev = []
#    contrasts = []
#    flatnesses = []
#    zeroCrossings = []
#    mfccs = []
#    onset_strengths = []
#
#    for spec in input_spectogram:
#        #compute magnitude spectrum
#        y = unprocess_data(spec)
#        magSpec = np.abs(y)
#        t = librosa.core.istft(y, hop_length=hop_length, win_length=n_fft)
#
#        spectral_centroid = ftr.spectral_centroid(S=magSpec, sr=samplerate, n_fft=n_fft, hop_length=hop_length)
#        spectral_contrast = ftr.spectral_contrast(S=magSpec, sr=samplerate, n_fft=n_fft, hop_length=hop_length)
#        spectral_flatness = ftr.spectral_flatness(S=magSpec, n_fft=n_fft, hop_length=hop_length)
#        spectral_flatness = 10*np.log10(spectral_flatness) #spectral flatness in dB
#        zero_crossing_rate = ftr.zero_crossing_rate(y=t)
#        mfcc = ftr.mfcc(y=t, sr=samplerate)
#        onset_strength = onst.onset_strength(y=t, sr=samplerate) #in case we want to test it with drums
#
#        centroids.append(np.mean(spectral_centroid))
#        centroids_dev.append(np.mean(np.std(spectral_centroid)))
#        contrasts.append(spectral_contrast)
#        flatnesses.append(np.mean(spectral_flatness)) #in dB
#        zeroCrossings.append(np.mean(zero_crossing_rate))
#        mfccs.append(mfcc)
#        onset_strengths.append(onset_strength)
#
#        
#    output = {'centroid_frequency' : centroids, 'centroid_frequency_dev' : centroids_dev, 'spectral_contrasts' : contrasts, 'spectral_flatness' : flatnesses, 'zero_crossing_rate' : zeroCrossings, 'MFCC' : mfccs, 'onset_strength' : onset_strengths}
#    return output
#



def calc_metrics(y, yest, **kwargs):
    # calc SDR and NRR
    
    # checking the type of the input
    sizY = y.shape
    sizYest = yest.shape
    
    if min(sizY) > 2 and min(sizYest)>2: #spectograms
        inputType = 'F'
    else:
        inputType = 'T'
    
    # retrieve params' values
    keys = kwargs.keys()
    
    if 'tBands' in keys:
        tBands = kwargs.pop('tBands', '')
    else:
        tBands = 1
        
    if 'fBands' in keys:
        fBands = kwargs.pop('fBands', '')
    else:
        fBands=1
     
        
    # cqse: we wanr to calculate the metrics from the T-F representations of y and yest   
    if fBands > 1:   
        if 'samplerate' in keys:
            samplerate = kwargs.pop('samplerate','')
        else:
            samplerate = 44100 # or 10k # or display an error
        
        if 'n_fft' in keys:
            n_fft = kwargs.pop('n_fft','')
        else:
            n_fft = 256 # 0-padded to 512
            
        if 'hop_length' in keys:
            hop_length = kwargs.pop('hop_length','')
        else:
            hop_length = n_fft/2 
        
           
        if inputType == 'T': # need to perform the STFT first
            Yest = librosa.core.stft(yest, hop_length=hop_length, win_length=n_fft,window='hann')
            Y    = librosa.core.stft(y,    hop_length=hop_length, win_length=n_fft,window='hann')
                
        elif inputType == 'F':
            Yest = yest
            Y    = y
            
        ## calculate the grid for the calculation of the metrics
        # Frequence: :
        logscale = librosa.mel_frequencies(n_mels=fBands,fmin=0,fmax=samplerate/2) #just to get a log scale
        #librosa.fft_frequencies(sr=22050, n_fft=subbands)        
        stepsF = np.floor(logscale/logscale[-1]* Y.shape[0])
        
        # Time:
        if tBands > 0:
            stepsT = np.round(np.linspace(0,Y.shape[1],tBands))
        else:
            tBands = 1
            stepsT = np.array([0,Y.shape[1]])
    
   
    else: #if fBands <= 1: (and  inputType == 'T' or 'F') : time domain
        fBands = 1
        stepsF = np.array([0,1])
        
        Yest = yest
        Y    = y
        
        if tBands > 0: #grid to cut y and yest into pieces
            stepsT = np.round(np.linspace(0, y.shape[0], tBands))
        else:
            stepsT = np.array([0,sizY[0]])
            
    
    SDR, NRR = np.zeros(fBands,tBands), np.zeros(fBands,tBands)
    
    for nf in range(fBands):
        for nt in range(tBands):

            yestSelec = Yest[ np.int(stepsF[nf]) : np.int(stepsF[nf+1]) , np.int(stepsT[nt]) : np.int(stepsT[nt+1])]
            ySelec    = Y[    np.int(stepsF[nf]) : np.int(stepsF[nf+1]) , np.int(stepsT[nt]) : np.int(stepsT[nt+1]) ]
            
#            Normalisation?
#            if norm_yest and clip_yest:
#            Ytemp[nf][nt] = np.min([LA.norm(Y[nf][nt])/LA.norm(Yest[nf][nt]) * Yest[nf][nt], (1+ 10**(-beta/20) * Y[j][m] )])
#            OR
#            if norm_yest:
#                yest =[LA.norm(y)/LA.norm(yest) * el_yest for el_yest in yest]
#            if clip_yest:
#                lowerBound =[(1+ 10**(-beta/20)) * ely for ely in y]
#                yest =[np.min([elyest, ellowerBound]) for elyest, ellowerBound in zip(yest,lowerBound)]

    
            # SIgnal 2 Distorsion Ratio:
            diffTrue2Est= [ a-b for a,b in zip(ySelec, yestSelec) ]
                    
            numSDR   = np.mean(diffTrue2Est @ diffTrue2Est.T)    
            denomSDR = np.mean(yestSelec @ yestSelec.T)
            # formula uses inverse ratio inside the log, compared to the paper cited 
            SDR[nf][nt] = 10 * np.log10(numSDR / denomSDR)
            
            # Noise Reduction Ratio:
            numNR   = np.mean(ySelec @ ySelec.T)
            denomNR = denomSDR
            NRR[nf][nt] = 10 * np.log10(numNR / denomNR)

        
    # Calculation of STOI
    STOI = calc_STOI(y,yest,**kwargs)
                
    output = {'Signal-To-Distorsion Ratio (SDR)' : SDR, 'Noise Reduction Ratio' : NRR, 'STOI' : STOI}
    return output

    

def calc_STOI(y, yest, **kwargs):
    "SDR: ref: C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen, 'An Algorithm for Intelligibility Prediction of Time-Frequency Weighted Noisy Speech,' IEEE/ACM"
    
    keys = kwargs.keys()
    
#    if 'norm_yest' in keys:
#        norm_yest=kwargs.pop('norm_yest', '')
#    else:
#        norm_yest=True
#        
#    if 'clip_yest' in keys:
#        clip_yest=kwargs.pop('clip_yest', '')
#    else:
#        clip_yest=True
        
    # Parameters
    if 'beta' in keys:
        beta = kwargs.pop('beta', '')
    else:
        beta = -15 #lower SDR bound
    
    if 'STOIsamplerate' in keys:
        STOIsamplerate = kwargs.pop('STOIsamplerate', '')
    else:
        STOIsamplerate = 10000
        
    if 'n_fft' in keys:
        n_fft = kwargs.pop('n_fft','')
    else:
        n_fft = 256 # 0-padded to 512
        
    if 'hop_length' in keys:
        hop_length = kwargs.pop('hop_length','')
    else:
        hop_length = n_fft/2 
            
    if 'STOIdurFrame' in keys:
        STOIdurFrame = kwargs.pop('STOIdurFrame','')
    else:
        STOIdurFrame = .386 # in sec, optimal for STOI according to the ref
    STOInFrame= np.int(STOIdurFrame * STOIsamplerate)
    
    # stft of time-domain signals / can be done with specs but then includes uncertainties about the parameters' values used
    Yest = librosa.core.stft(yest, hop_length=hop_length, win_length=n_fft,window='hann')
    Y    = librosa.core.stft(y,    hop_length=hop_length, win_length=n_fft,window='hann')
    
    Y_dB = librosa.core.amplitude_to_db(Y)
    
    NRJ_Y = np.sum(Y_dB, axis = 0) / Y_dB.shape[0] 
    indMaxEnergyFrameInCleanSpeech = [ii for ii,Yii in enumerate(NRJ_Y) if Yii == np.max(NRJ_Y) ]
    
    # find sequences without speech (energy < 40 dB)
    maxEnergyFrameInCleanSpeech = NRJ_Y[indMaxEnergyFrameInCleanSpeech[0]]
    framesToDiscard = [ii for ii in range(Y_dB.shape[0]) if NRJ_Y[ii] < maxEnergyFrameInCleanSpeech - 40]
    framesToKeep = set(range(Y_dB.shape[0]), framesToDiscard)
    
    # eliminate sequences without speech (energy < 40 dB) 
    Y = [Y[:,ff] for ff in framesToKeep]
    Y_dB = librosa.core.amplitude_to_db(Y)
    Y_power = librosa.core.db_to_power(Y_dB, ref=1.0)
    
    Yest = [Yest[:,ff] for ff in framesToKeep]
    Yest_dB = librosa.core.amplitude_to_db(Yest)
    Yest_power = librosa.core.db_to_power(Yest_dB, ref=1.0)
    
    # reconstruction of the trimmed signals
#    y = librosa.core.istft(Y, hop_length=hop_length, win_length=n_fft,window='hann')
#    yest = librosa.core.istft(Yest, hop_length=hop_length, win_length=n_fft,window='hann')   
    
    # a one-third octave band analysis by grouping DFT-bins. In total 15 one-third octave bands > 150Hz and < 4.3kHz (center of the highest band)
    logscale = librosa.mel_frequencies(n_mels=16, fmin=100, fmax=5000) 
    #16, so as to get 15 bands,  5000 = sr/2
    stepsF = np.floor(logscale/logscale[-1]* Y.shape[0]) - 1
    stepsF[0] = 0
    #stepsF[-1] = nBands or floor(nfft/2) (+ 1)

    # calculate T-F units
    TF_units = np.empty(Y.shape[1])
    for t in range(Y.shape[1]):
        TF_units[t,] = [np.sum(Y_power[np.int(stepsF[f]):np.int(stepsF[f+1]), :]) for f in range(len(stepsF)-1)]
    
    ## TODO   
    # short-term segmentation
    short_term_segments = [TF_units[:,t] ]
    # normalise + clip yest
    # correlation coeff
    # average
    STOI = -1
    
    return STOI




