# DSP functions such as applying noise, RIRs, or data representation conversions

import numpy as np
import pandas as pd
import random as rnd
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
    return frags


def unmake_fragments(s_frag, frag_hop_len, frag_win_len):
    # TODO get to work with arbitrary input shape?
    spec_length = (s_frag.shape[0]-1) * frag_hop_len + frag_win_len
    s = np.zeros((s_frag.shape[1], spec_length), dtype=s_frag.dtype)
    for i, frag in enumerate(s_frag):
        # TODO does this use the oldest or newest part?
        lower_bound = i*frag_hop_len
        upper_bound = i*frag_hop_len+frag_win_len
        s[:, lower_bound:upper_bound] = frag
    return s



### PRE/POST PROCESSING FUNCTIONS
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
def white_noise(x, snr):
    print('Using white noise')
    
    N = max(x.shape)
    # N = len(x) alternatively
    sigma = np.sqrt( (x @ x.T) / (N * 10**(snr/10)) )
    noise = [sigma * rnd.uniform(-1,1) for k in range( N) ]
    
    return x+noise


def pink_noise(x, snr):
    """Generates pink noise using the Voss-McCartney algorithm.
        
    nrows: number of values to generate
    rcols: number of random sources to add
    
    returns: NumPy array
    """
    print('Using pink noise')
    
    nrows = len(x) #x.shape
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
    
    return x+noise


def velvet_noise(x, SNR):
    print('Using velvet noise')
    
    N = max(x.shape)
    # N = len(x) alternatively
    sigma = np.sqrt( (x @ x.T) / (N * 10**(SNR/10)) )
    print('sigma = {0}'.format(sigma))
    
    myVelvetNoise = [rnd.uniform(-1, 1) for k in range( N) ] #random numbers between -1 and 1
    rate_zero=.95 # could be parametrized
    noise = [sigma * ((vv> rate_zero) - (vv < -rate_zero)) for vv in myVelvetNoise]
    return x+noise

# def take_file_as_noise(x, SNR):
#     N = len(x)
#     sigma = np.sqrt( (x @ x.T) / (N * 10**(SNR/10)) )
#     def noising_prototype( filepath):
#         print('Using the following file as noise: {0}'.format(filepath))
# #        path = os.path.join(filepath + '.wav')
#         load_noise = np.load(filepath)
#         noise =  sigma * (load_noise - np.mean(load_noise)) + np.mean(load_noise) 
#         return noise
#     return noising_prototype

def take_file_as_noise(filepath):
    # checking TODO
    print('Using the following file as noise: {0}'.format(filepath))
    # path = os.path.join(filepath + '.wav')
    load_noise = np.load(filepath)
    
    def noising_prototype(x, SNR):
        N = len(x)
        sigma = np.sqrt( (x @ x.T) / (N * 10**(SNR/10)) )
        noise =  sigma * (load_noise - np.mean(load_noise)) + np.mean(load_noise) 
        return noise
    return noising_prototype

