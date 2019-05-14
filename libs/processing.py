# DSP functions such as applying noise, RIRs, or data representation conversions

import numpy as np
import pandas as pd
import random as rnd
import librosa as lr
import time
import os
import os.path as osp
import scipy
from libs.colored_noise import powerlaw_psd_gaussian

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


def unmake_fragments_slice(s_frag, frag_hop_len, frag_win_len, time_slice):
    # store input shape
    in_shape = s_frag.shape
    # multiple input shape support
    spec_length = (in_shape[0]-1) * frag_hop_len + frag_win_len
    output_shape = (in_shape[1], spec_length, in_shape[-1]
                    ) if len(in_shape) == 4 else (in_shape[1], spec_length)
    # if slice is integer, use it as single slice
    # NOTE: indexing [i] instead of slicing [x:y] cause dimension to collapse
    if isinstance(time_slice, int) or isinstance(time_slice, np.generic):
        time_slice = slice(time_slice, time_slice+frag_hop_len)
    # if time slice is larger than frag_hop, actual slice will be of size hop
    slice_width = time_slice.stop - time_slice.start
    if slice_width > frag_hop_len:
        slice_center = (time_slice.stop + time_slice.start) // 2
        time_slice_curr = slice(slice_center - frag_hop_len // 2, 
                                slice_center + frag_hop_len // 2+1)
    else:
        time_slice_curr = time_slice
    # initialize recipient
    s = np.zeros(output_shape, dtype=s_frag.dtype)
    for i, frag in enumerate(s_frag):
        frag = frag[..., time_slice_curr, :] if len(
            frag.shape) == 3 else frag[..., time_slice_curr]
        lower_bound = i*frag_hop_len
        upper_bound = (i+1)*frag_hop_len
        #upper_bound = i*frag_hop_len+frag_win_len
        s[:, lower_bound:upper_bound] = frag
    return s


### PRE/POST PROCESSING FUNCTIONS
## helper funcs
def rem_dc_bin(s):
    if s.shape[-2] % 2 != 0:
        s = s[..., :-1, :]
    return s

def add_dc_bin(s):
    pad_shape = list(s.shape)
    pad_shape[-2] = 1
    pad_shape = tuple(pad_shape)
    padding = np.zeros(pad_shape)
    s = np.concatenate((s, padding), axis=-2)
    return s


## convert complex spectrograms to/from magnitude^exponent
# NOTE implementation based on callable class rather than nested functions
#      due to `fit_generator` requiring data_generator arguments to be 
#      picklable (nested functions aren't)
class s_to_exp(object):
    def __init__(self, exponent):
        self.exponent = exponent
        self.__name__ = 's_to_exp({:.3f})'.format(exponent)

    def __call__(self, s):
        s = rem_dc_bin(s)
        # complex -> magnitude -> power/amplitude/etc
        s_power = np.abs(s) ** self.exponent
        return s_power[..., np.newaxis]

def exp_to_s(exponent):
    def func(power, s_noisy=None):
        # power/amplitude/etc -> magnitude
        s = power[..., 0] ** (1.0/exponent)
        # use phase from noisy signal: magnitude -> complex
        if s_noisy is not None:
            s_noisy = s_noisy[..., :-1, :]
            angles = np.angle(s_noisy)
            s = s * np.exp(1j * angles)
        s = add_dc_bin(s)
        return s
    return func


## convert complex spectrograms to/from absolute power spectrum
def s_to_power(s):
    return s_to_exp(2)(s)

def power_to_s(power, s_noisy=None):
    return exp_to_s(2)(power, s_noisy)


## convert complex spectrograms to/from decibel-spectrum
def s_to_db(s):
    s = rem_dc_bin(s)
    # complex -> magnitude -> decibels
    s_db = lr.amplitude_to_db(np.abs(s))
    return s_db[..., np.newaxis]

def db_to_s(db, s_noisy=None):
    # decibels -> magnitude
    s = lr.db_to_amplitude(db[..., 0])
    # use phase from noisy signal: magnitude -> complex
    if s_noisy is not None:
        s_noisy = s_noisy[..., :-1, :]
        angles = np.angle(s_noisy)
        s = s * np.exp(1j * angles)
    s = add_dc_bin(s)
    return s


# convert complex spectrograms to Re/Im representation
# NOTE unmaintained!
def s_to_reim(s):
    s = rem_dc_bin(s)
    # split re/im
    re = np.real(s)
    im = np.imag(s)
    # stack
    reim = np.stack([re, im], axis=-1)
    return reim

# convert Re/Im representation to complex spectrograms
def reim_to_s(reim):
    # extract real and imaginary components
    re = reim[..., 0]
    im = reim[..., 1]
    # combine into complex values
    s = re + 1j * im
    s = add_dc_bin(s)
    return s


## normalization
def normalize_spectrum(s):
    s_std = np.std(s)
    s_avg = np.mean(s)
    return (s - s_avg) / s_std, (s_avg, s_std)

def normalize_spectrum_clean(s, norm_factors):
    s_avg, s_std = norm_factors
    return (s - s_avg) / s_std

def unnormalize_spectrum(s, norm_factors):
    s_avg, s_std = norm_factors
    return (s * s_std) + s_avg


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

def pink_noise(x, sr, snr):
    n = powerlaw_psd_gaussian(1, len(x))
    return sum_with_snr(x, n, snr)

    
# add white gaussian noise
def white_noise(x, sr, snr):
    n = np.random.randn(*x.shape)
    return sum_with_snr(x, n, snr)


class take_file_as_noise(object):
    def __init__(self, filepath):
        self.filepath = filepath
        self.__name__ = 'take_file_as_noise({})'.format(osp.basename(osp.dirname(filepath)))

    def __call__(self, x, sr, snr):  
        xn, _ = lr.load(self.filepath, sr = sr)
        dur_speech = x.shape[0]
        dur_noise = xn.shape[0]

        # Create Fade-in & fade-out 
        dur_fade = 1# in sec
        p100_fade = dur_fade*sr / dur_noise # proportion
        fade_len = np.int(p100_fade * dur_noise)
        fadein = np.cos(np.linspace(-np.pi/2,0,fade_len))**2
        fadeout = np.cos(np.linspace(0, np.pi/2,fade_len))**2

        # apply fading to copy of xn
        noise = xn[:]
        noise[ :fade_len] = fadein * noise[ :fade_len]
        noise[-fade_len: ] = fadeout * noise[-fade_len: ] 

        # Draw random proportion for the beginning of the noise, in the interval [fade_len, dur_noise-fade_len]
        rnd_beg_ind = np.int(np.random.random(1) * (dur_noise - 2*fade_len)) + fade_len
        # init
        out = np.zeros((dur_speech)) 
        portion_noise = dur_noise - rnd_beg_ind # always <dur_noise
    
        # Checking if the remaining portion of noise can fit into out
        if portion_noise >= dur_speech: 
            n_noise_next = rnd_beg_ind+dur_speech
            out[:] += noise[rnd_beg_ind : n_noise_next]
            # and that's is!

        else:
            n_noise_next = 0
            n_out_next = dur_noise - rnd_beg_ind
            out[ :n_out_next] += noise[rnd_beg_ind:]
            # Looping
            n_out_beg = n_out_next - fade_len
            n_out_end = n_out_beg + dur_noise
            #n = 0
            while n_out_end < dur_speech:
                #print('n: {}, nb_out_end: {}'.format(n, n_out_end))
                out[n_out_beg: n_out_end] += noise[:]
                n_out_next = n_out_end
                n_out_beg = n_out_next - fade_len
                n_out_end = n_out_beg + dur_noise
                #n +=1
                
            #Last iteration: The noise may be too long for the remaining of the speech file. Trimmed 
            portion_out = dur_speech-n_out_next
            out[n_out_next: ] += noise[n_noise_next : n_noise_next + portion_out]    
            
        return sum_with_snr(x,out,snr)


# add pink (1/f) noise using Voss-McCartney algorithm
def pink_noise2(x, sr, snr):
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
    # TODO return signal + noise at given SNR
    return noise






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
