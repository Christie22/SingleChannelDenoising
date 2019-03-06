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
from keras import backend as K

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt
from scipy.stats import norm
import librosa
import librosa.display
import librosa.feature as ftr
import librosa.onset as onst
import sys
import glob


######### v FUNCTIONS THAT ARE ACTUALLY USED v ############

def load_dataset(dataset_path):
    # TODO implement actual data handling
    # (requires figuring out data format)
    print('[i] Loading all wav files from {}'.format(dataset_path))
    filelist = glob.glob(os.path.join(dataset_path, '*.wav'))
    print('[i] Loaded {} files'.format(len(filelist)))
    return filelist


def create_autoencoder_model(custom_objects, model_name, input_shape):
    # import model factory
    if model_name == 'lstm':
        print('Using model `{}` from {}'.format(model_name, 'model_lstm'))
        return None
    elif model_name == 'conv':
        print('Using model `{}` from {}'.format(model_name, 'model_conv'))
        return None
    else:
        print('importing example model :D')
        import models.model_example as m

    # calc input shape and enforce it
    K.set_image_data_format('channels_last')
    # generate model
    obj = m.AEModelFactory(input_shape=input_shape)
    model = obj.get_model()
    return model


def load_autoencoder_model(model_path, custom_objects):
    model = load_model(model_path, custom_objects=custom_objects)
    # extract encoder from main model
    encoder = model.get_layer('encoder')
    decoder = model.get_layer('decoder')
    return encoder, decoder, model


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



def calc_metrics(y, yest, **kwargs):
    # calc SDR and NRR

    # checking the type of the input
    sizY = y.shape
    sizYest = yest.shape

    if min(sizY) > 2 and min(sizYest)>2: #spectrograms
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


    # cqse: we want to calculate the metrics from the T-F representations of y and yest
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
    print('length of y: {0}'.format(y.shape))

    keys = kwargs.keys()

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
        hop_length = np.int(n_fft/2)

    if 'STOIframe_t' in keys:
        STOIframe_t = kwargs.pop('STOIframe_t','')
    else:
        STOIframe_t = .384 # in sec, optimal for STOI, according to the ref
    STOIframe_n = np.int(STOIframe_t * STOIsamplerate)
    print('STOIframe_n: {0}'.format(STOIframe_n))

    nbFramesToGetA384ms_longFrame = np.int(STOIframe_n / n_fft) # .384/(256 / (fs*2)) = 30.0
    print('nbFramesToGetA384ms_longFrame: {0}'.format(nbFramesToGetA384ms_longFrame,n_fft))

    fBands = 15 # cf literature

    # stft of time-domain signals / can be done with specs but then includes uncertainties about the parameters' values used
    Yest = librosa.core.stft(yest, hop_length=hop_length, win_length=n_fft, window='hann')
    Y    = librosa.core.stft(y,    hop_length=hop_length, win_length=n_fft, window='hann')
    Y_dB = librosa.core.amplitude_to_db(np.abs(Y))
    #librosa.display.specshow(Y_dB)
    print('dimensions of Y: {0}'.format(Y.shape))

    NRJ_Y = np.sum(Y_dB,axis=0)/Y_dB.shape[0] #20*np.log10(y**2/2e-5)
    #t =  np.linspace(0,y.shape[0], y.shape[0] )
    #plt.figure, plt.plot(t, NRJ_Y)

    # find sequences without speech (energy < 40 dB) and eliminate them
    maxEnergyFrameInCleanSpeech = NRJ_Y[NRJ_Y==np.max(NRJ_Y)][0]
    print('np.max(NRJ_Y) is {0}'.format(np.max(NRJ_Y) ))
    print('maxEnergyFrameInCleanSpeech is: {0}'.format(maxEnergyFrameInCleanSpeech) )

    framesToKeep = [ii for ii in range(Y_dB.shape[1]) if NRJ_Y[ii:ii+1] >= maxEnergyFrameInCleanSpeech - 40]
    tShape = len(framesToKeep)
    print('{0} frames to keep (tShape)'.format(tShape))

    Y =np.array([Y[:,tt] for tt in framesToKeep]).T #[ Y[:,tt] for tt in framesToKeep] #
    #Y = Y[:,[NRJ_Y >= maxEnergyFrameInCleanSpeech - 40]]
    Y_dB = librosa.core.amplitude_to_db(np.abs(Y))
    Y_power = librosa.core.db_to_power(Y_dB, ref=1.0)
    print('new Y.Shape: {0}'.format(Y.shape))

    Yest = np.array([Yest[:,tt] for tt in framesToKeep]).T #transpose to keep the original disposition (freq * time)
    Yest_dB = librosa.core.amplitude_to_db(np.abs(Yest))
    Yest_power = librosa.core.db_to_power(Yest_dB, ref=1.0)

    # reconstruction of the trimmed signals
#    y = librosa.core.istft(Y, hop_length=hop_length, win_length=n_fft,window='hann')
#    yest = librosa.core.istft(Yest, hop_length=hop_length, win_length=n_fft,window='hann')

    # a one-third octave band analysis by grouping DFT-bins. In total 15 one-third octave bands > 150Hz and < 4.3kHz (center of the highest band)
    logscale = librosa.mel_frequencies(n_mels=fBands+1, fmin=100, fmax=5000)
    #16, so as to get 15 bands (16 edges),  5000 = sr/2 (coincidence?)

    stepsF = np.floor(logscale/logscale[-1]* Y.shape[0]) - 1 #mapping onto the dim of Y (axis freq)
    stepsF[0] = 0
    print('length stepsF: {0}'.format(stepsF.shape ))
    print('boundaries stepsF: {0} and {1}'.format(stepsF[0],stepsF[-1]))

    # calculate T-F units
    Y_TF_units = np.empty(( fBands, tShape)) # time * fBands
    Yest_TF_units = np.empty(( fBands, tShape)) # time * fBands
    for t in range(tShape):
        Y_TF_units[:,t]    = [np.sqrt(np.sum(Y_power[   np.int(stepsF[f]):np.int(stepsF[f+1]), t])) for f in range(len(stepsF)-1)]
        Yest_TF_units[:,t] = [np.sqrt(np.sum(Yest_power[np.int(stepsF[f]):np.int(stepsF[f+1]), t])) for f in range(len(stepsF)-1)]

    # Short-term segments: group nbFramesToGetA384ms_longFrame TF-units to create 384ms(ish)-long frames
    # I'm not sure if the frames are successive or if they are supposed to overlap...
    #### case: they are successive
    totNbTimeFrames = np.int(Yest.shape[1] / nbFramesToGetA384ms_longFrame) # 1+np.int((Yest.shape[1] - nbFramesToGetA384ms_longFrame) / (nbFramesToGetA384ms_longFrame))
    print('totNbTimeFrames (without overlap): {0}'.format(totNbTimeFrames) )

    #### case: they overlap
    #n_overlap = nbFramesToGetA384ms_longFrame-1
    #totNbTimeFrames = 1+np.int((Yest.shape[1] - nbFramesToGetA384ms_longFrame) / (nbFramesToGetA384ms_longFrame-n_overlap)) # np.int((Yest.shape[1] - nbFramesToGetA384ms_longFrame) / (STOIframe_n-1))
    #print('totNbTimeFrames (with overlap): {0}'.format(totNbTimeFrames) )

    Y_short_term_segments = np.empty((totNbTimeFrames, fBands, nbFramesToGetA384ms_longFrame))
    Yest_short_term_segments = np.empty((totNbTimeFrames, fBands, nbFramesToGetA384ms_longFrame))

    for nn in range(totNbTimeFrames-1):
        Y_short_term_segments[nn,]    = Y_TF_units[   :,nbFramesToGetA384ms_longFrame*nn : nbFramesToGetA384ms_longFrame *(nn+1)]
        Yest_short_term_segments[nn,] = Yest_TF_units[:,nbFramesToGetA384ms_longFrame*nn : nbFramesToGetA384ms_longFrame *(nn+1)]
    # dim: totNbTimeFrames * fBands * nbFramesToGetA384ms_longFrame
    print('dim of Y_short_term_segments: {0}'.format(Y_short_term_segments.shape ))

    # normalise + clip yest
    Yest_normalised_clipped = np.empty((totNbTimeFrames,fBands,nbFramesToGetA384ms_longFrame))
    for tt in range(totNbTimeFrames):
        for ff in range(fBands):
            norm_coeff = LA.norm(Y_short_term_segments[tt,ff,:]) / LA.norm(Yest_short_term_segments[tt,ff,:])
            #print('norm_coeff: {0} '.format(norm_coeff * Yest_short_term_segments[tt,ff,0]))
            #print('(1+10**(-beta/20)*Y_short_term_segments[tt,ff,n=0])) : {0}'.format((1+10**(-beta/20)*Y_short_term_segments[tt,ff,0])))
            #print('value qui coince: {0}'.format(np.min((norm_coeff * Yest_short_term_segments[tt,ff,0], (1+10**(-beta/20)*Y_short_term_segments[tt,ff,0]))))

            Yest_normalised_clipped[tt,ff,:] = [np.min((norm_coeff * Yest_short_term_segments[tt,ff,nn], (1+10**(-beta/20)*Y_short_term_segments[tt,ff,nn])) ) for nn in range(nbFramesToGetA384ms_longFrame) ]

    # correlation coeff
    d = np.empty((totNbTimeFrames,fBands))
    for tt in range(totNbTimeFrames):
        for ff in range(fBands):
            Yjm = Y_short_term_segments[tt,ff,:]
            Yestbar = Yest_normalised_clipped[tt,ff,:]
            #print('Yjm.shape: {0}'.format(Yjm.shape) )

            Y_mu = np.mean(Yjm)
            Yest_n_c_mu = np.mean(Yestbar)

            num = (Yjm - Y_mu).T.dot((Yestbar - Yest_n_c_mu))
            #print('d num: {0}'.format(num) )
            denom = LA.norm(Yjm - Y_mu) * LA.norm(Yestbar - Yest_n_c_mu)
            #print('d denom: {0}'.format(denom) )
            d[tt,ff] = num / (denom + sys.float_info.epsilon)
    # average
    #print('d sum: {0}'.format(np.sum(d)) )
    STOI = 1/ (totNbTimeFrames*fBands) * np.sum(d)
    print('STOI: {0}'.format(STOI) )
    print('0: non-intelligible, 1: very intelligible')

    return STOI, framesToKeep




