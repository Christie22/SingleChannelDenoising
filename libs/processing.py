# DSP functions such as applying noise, RIRs, or data representation conversions

import numpy as np
import pandas as pd
import random as rnd
import librosa as lr
import time
import os
from os import path
import scipy

# generate seed from the time at which this script is run
rnd.seed(int(time.time()))


### FRAGMENTING AND RECONSTRUCTING FROM FRAGMENTS
def make_fragments(s, frag_hop_len, frag_win_len):
    # convert T-F data into fragments
    n_frags = int((s.shape[1] - frag_win_len) / frag_hop_len + 1)
    
    def get_slice(i):
        lower_bound = i*frag_hop_len
        upper_bound = i*frag_hop_len+frag_win_len
        return s[:, lower_bound:upper_bound]
    frags = [get_slice(i) for i in range(n_frags)]
    return np.array(frags)


def unmake_fragments(s_frag, frag_hop_len, frag_win_len):
    # store input shape
    in_shape = s_frag.shape
    # calculate output spectrogram length in frames
    spec_length = (in_shape[0]-1) * frag_hop_len + frag_win_len
    # calculate output shape based on number of dims
    output_shape = (in_shape[1], spec_length, in_shape[-1]) if len(in_shape) == 4 else (in_shape[1], spec_length)
    s = np.zeros(output_shape, dtype=s_frag.dtype)
    for i, frag in enumerate(s_frag):
        # NOTE this uses the initial portion of each fragment
        lower_bound = i*frag_hop_len
        upper_bound = i*frag_hop_len+frag_win_len
        s[:, lower_bound:upper_bound] = frag
    return s



### PRE/POST PROCESSING FUNCTIONS
# convert complex spectrograms to absolute power spectrum
def s_to_power(s):
    # remove a bin if odd number
    if s.shape[0] % 2 != 0:
        s = s[:-1]
    s_power = np.abs(s) ** 2
    return np.expand_dims(s_power, axis=2)

def power_to_s(power, s_noisy=None):
    s = np.sqrt(power)
    # TODO might require noisy signal as input for phase
    # add previously removed bin
    pad_shape = list(s.shape)
    pad_shape[-2] = 1
    pad_shape = tuple(pad_shape)
    padding = np.zeros(pad_shape)
    s = np.concatenate((s, padding), axis=-2)
    return s

# convert complex spectrograms to Re/Im representation
def s_to_reim(s):
    # remove a bin if odd number
    if s.shape[0] % 2 != 0:
        s = s[:-1]
    # split re/im
    re = np.real(s)
    im = np.imag(s)
    # stack
    reim = np.dstack((re, im))
    return reim

# convert Re/Im representation to complex spectrograms
def reim_to_s(reim):
    # extract real and imaginary components
    re = reim[..., 0]
    im = reim[..., 1]
    # combine into complex values
    s = re + 1j * im
    # add previously removed bin
    pad_shape = list(s.shape)
    pad_shape[-2] = 1
    pad_shape = tuple(pad_shape)
    padding = np.zeros(pad_shape)
    s = np.concatenate((s, padding), axis=-2)
    return s



### NOISING FUNCTIONS  
# sum s(ignal) and n(oise) at a given SNR (in dB)
def sum_with_snr(s, n, snr):
    # calculate SNR as linear ratio
    snr_lin = 10.0 ** (snr / 10.0)
    # calculate signals RMS (standard deviation in AC signals)
    s_rms = s.std()
    n_rms = n.std()
    # calculate scaling factor for noise
    scaling = s_rms / (snr_lin * n_rms)
    # sum scaled signals
    out = s + (n * scaling)
    # TODO normalize?
    return out

# add white gaussian noise
def white_noise(x, sr, snr):
    n = np.random.randn(*x.shape)
    return sum_with_snr(x, n, snr)

# add pink (1/f) noise using Voss-McCartney algorithm
def pink_noise(x, sr, snr):
    # number of values to generate
    nrows = len(x) #x.shape
    # number of random sources to add
    ncols=16
    
    array = np.empty((nrows, ncols))
    array.fill(np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)
    
    # the total number of changes is nrows
    n = nrows
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=n)
    array[rows, cols] = np.random.random(n)

    df = pd.DataFrame(array)
    df.fillna(method='ffill', axis=0, inplace=True)
    total = df.sum(axis=1)

    sigma = np.sqrt( (x @ x.T) / (nrows * 10**(snr/10)) )
    noise= sigma*(total.values-np.mean(total.values)) / (max(total.values) - np.mean(total.values))
    
    return noise


def velvet_noise(x, SNR):
    print('Using velvet noise')
    
    N = max(x.shape)
    # N = len(x) alternatively
    sigma = np.sqrt( (x @ x.T) / (N * 10**(SNR/10)) )
    print('sigma = {0}'.format(sigma))
    
    def createVelvetNoise(rate_zero = .95):
        """
        #### Role: create a vector of velvet noise (containing exclusively {-1,0,1})
        # Input:
         rate_zero(optional): pourcentage (between 0 and 1) of "0" in the output vector. 
        # output: velvet noise
         V: standard vector
         params(optional) (struct): parametres (nb of zeros, indices, values) TODO
        """
        myVelvetNoise = [rnd.uniform(-1, 1) for k in range( N) ] #random numbers between -1 and 1
        noise = [sigma * ((vv> rate_zero) - (vv < -rate_zero)) for vv in myVelvetNoise]
        
        #        params.NonZeros = np.sum(np.abs(noise))
        #        params.realZeroRate = 1-params.NonZeros/noise.shape[0];
        #        [params.indNonZeros, ~,params.valNonZeros] = find(SV);
        #        params.sizeVN = N;
        return noise#, params
  


def take_file_as_noise(x, SNR):
    # checking TODO
    N = len(x)
    sigma = np.sqrt( (x @ x.T) / (N * 10**(SNR/10)) )
    def noising_prototype( filepath):
        print('Using the following file as noise: {0}'.format(filepath))
#        path = os.path.join(filepath + '.wav')
        load_noise = np.load(filepath)
        noise =  sigma * (load_noise - np.mean(load_noise)) + np.mean(load_noise) 
        return noise
    return noising_prototype

#print('compiled')
#
#
#aa = np.array([[[1,2,12,212],[3,4,34, 434],[0,1,10, 110]],[[5,6,56,656],[7,8,78,878] ,[9,0,90, 990]]])
#bb=10*aa
#xx = np.array([aa,bb])
#xx = xx.reshape(3,2,4,2)
#xxx=xx[:,:,:,1]+xx[:,:,:,0]  
#xxx
#y=unmake_fragments(xx,1,0)
#
#y
#
#print('done')
