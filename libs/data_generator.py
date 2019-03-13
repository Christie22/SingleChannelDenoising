# TODO
# - multithreaded caching?
# - verify (x, target) matching
# - implement other label_type modes

import os
import glob
import itertools
import keras
import librosa as lr
import numpy as np
import pandas as pd
import os.path as osp
from scipy.signal import fftconvolve
from tqdm import tqdm
#from scipy.io.wavfile import write

import libs.updated_utils
import libs.processing as processing
import libs.rir_simulator_python.roomsimove_single as room


class DataGenerator(keras.utils.Sequence):
    # util.frame : split a vector in overlapping windows
    def __init__(self, filepaths, cache_path=None, sr=16000,
                 rir_path=None, noise_funcs=[None], noise_snrs=[0],
                 n_fft=512, hop_length=128, win_length=512,
                 proc_func=None, proc_func_label=None,
                 frag_hop_length=64, frag_win_length=32,
                 shuffle=True, label_type='clean', batch_size=32, disable_cacheinit=True):

        # dataset cfg
        self.filepaths = filepaths
        self.cache_path = osp.expanduser(cache_path) if cache_path else osp.join(
            osp.dirname(filepaths[0]), 'cache')
        self.sr = sr
        # reverberation cfg
        self.rir_path = osp.expanduser(rir_path) if rir_path else None
        # noising cfg
        self.noise_funcs = noise_funcs
        self.noise_snrs = noise_snrs
        # stft cfg
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        # processing cfg
        self.proc_func = proc_func
        self.proc_func_label = proc_func_label
        # fragmenting cfg
        self.frag_hop_length = frag_hop_length
        self.frag_win_length = frag_win_length
        # general cfg
        self.shuffle = shuffle
        self.label_type = label_type
        self.batch_size = batch_size
        # computed vars
        self._data_shape = None
        self.rir_filepaths = self.load_rirs()
        self.noise_variations = list(itertools.product(
            self.noise_funcs, self.noise_snrs, self.rir_filepaths))
        # cached vars
        self.fragments_x = None
        self.fragments_y = None
        self.indexes = []
        # init stuff up
        if disable_cacheinit:
            self.load_cache()
        else:
            self.init_cache()
        self.on_epoch_end()

    # load list of RIR files
    def load_rirs(self):
        print('[d] Loading all RIRs files from {}'.format(self.rir_path))
        filelist = glob.glob(osp.join(self.rir_path, '*.npy'))
        print('[d] Loaded {} files'.format(len(filelist)))
        return filelist or [None]

    # init cache
    def init_cache(self):
        print('[d] Initializing cache...')
        self.fragments_x = []
        self.fragments_y = []
        for i, filepath in enumerate(tqdm(self.filepaths)):
            #print('[d] Loading file {}/{}: {}'.format(i, len(self.filepaths), filepath))
            # load data
            x, _ = lr.core.load(filepath, sr=self.sr, mono=True)
            # apply variations of noise parameters + clean (labels)
            for noise_variation in self.noise_variations + ['clean']:
                #print('[d]  Applying noise variation {}'.format(noise_variation))
                if noise_variation == 'clean':
                    # convert to TF-domain
                    s = lr.core.stft(
                        x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
                    # apply label preprocessing
                    s_proc = self.proc_func_label(
                        s) if self.proc_func_label else np.reshape(s, (*s.shape, 1))

                else:
                    noise_func, snr, rir_filepath = noise_variation
                    # apply room
                    x_rev = self.apply_reverb(
                        x, rir_filepath) if rir_filepath else x
                    # apply noise function
                    x_noise = noise_func(
                        x_rev, sr=self.sr, snr=snr) if noise_func else x_rev
                    # convert to TF-domain
                    s_noise = lr.core.stft(
                        x_noise, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
                    # apply data repr processing
                    s_proc = self.proc_func(s_noise) if self.proc_func else np.reshape(
                        s_noise, (*s_noise.shape, 1))

                # fragment data
                s_frags = self.make_fragments(
                    s_proc, self.frag_hop_length, self.frag_win_length)
                # store shape!
                if not self._data_shape:
                    self._data_shape = s_frags[0].shape

                # store fragments as numpy arrays
                for i, frag in enumerate(s_frags):
                    # generate filepath from parameters
                    frag_path = self.gen_cache_path(
                        self.cache_path, filepath, noise_variation,
                        self.proc_func if noise_variation != 'clean' else self.proc_func_label, i)
                    #print('[d]   Storing frag {} in {}'.format(i, frag_path))
                    os.makedirs(osp.dirname(frag_path), exist_ok=True)
                    np.save(frag_path, frag)
                    # append fragment path to proper list (labels or processed)
                    if noise_variation == 'clean':
                        self.fragments_y.append(frag_path)
                    else:
                        self.fragments_x.append(frag_path)
        # done
        print('[d] Cache ready, generated {} noisy and {} clean fragments'.format(
            len(self.fragments_x), len(self.fragments_y)))

    # load a pre-initialized cache (use with caution)
    def load_cache(self):
        print('[d] Cache generation disabled. Indexing cache...')
        self.fragments_x = []
        self.fragments_y = []
        for i, filepath in enumerate(tqdm(self.filepaths)):
            # get audio duration in sample length
            n_samples = lr.time_to_samples(
                lr.get_duration(filename=filepath), sr=self.sr)
            # calculate number of stft frames
            n_frames = int((n_samples - self.win_length) / self.hop_length + 1)
            # calculate number of fragments
            n_frags = int((n_frames - self.frag_win_length) /
                          self.frag_hop_length + 1)
            # apply variations of noise parameters + clean (labels)
            for noise_variation in self.noise_variations + ['clean']:
                for i in range(n_frags):
                    # generate filepath from parameters
                    frag_path = self.gen_cache_path(
                        self.cache_path, filepath, noise_variation,
                        self.proc_func if noise_variation != 'clean' else self.proc_func_label, i)
                    # append fragment path to proper list (labels or processed)
                    if noise_variation == 'clean':
                        self.fragments_y.append(frag_path)
                    else:
                        self.fragments_x.append(frag_path)
        # load last one to get shape
        self._data_shape = np.load(frag_path).shape
        # done
        print('[d] Cache ready, indexed {} noisy and {} clean fragments'.format(
            len(self.fragments_x), len(self.fragments_y)))

    # generate filepath for individual fragments

    def gen_cache_path(self, cache_path, filepath, noise_variation, proc_func, frag_index):
        filepath_dir = osp.splitext(osp.basename(filepath))[
            0].replace(' ', '_')
        path = osp.join(cache_path, filepath_dir)
        if noise_variation == 'clean':
            noise_variation_str = noise_variation
        else:
            noise_func, snr, rir_filepath = noise_variation
            noise_variation_str = '{}_{}_{}'.format(
                noise_func.__name__ if noise_func else 'none',
                snr,
                osp.splitext(rir_filepath)[
                    0][-6:] if rir_filepath else 'none'
            )
        path = osp.join(path, noise_variation_str)
        path = osp.join(path, proc_func.__name__ if proc_func else 'none')
        path = osp.join(path, 'frag_{}.npy'.format(frag_index))
        return path

    def apply_reverb(self, x, rir_filepath):
        rir = np.load(rir_filepath)
        x = fftconvolve(rir, x)
        return x

    # convert T-F data into fragments
    # frag_hop_len, frag_win_len provided in seconds?
    def make_fragments(self, s, frag_hop_len, frag_win_len):
        n_frags = int((s.shape[1] - frag_win_len) / frag_hop_len + 1)

        def get_slice(i):
            lower_bound = i*frag_hop_len
            upper_bound = i*frag_hop_len+frag_win_len
            return s[:, lower_bound:upper_bound]
        frags = [get_slice(i) for i in range(n_frags)]
        return frags

    # callback at each epoch (shuffles batches)
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.fragments_x))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # number of batches
    def __len__(self):
        if not self.fragments_x:
            self.init_cache()
        return len(self.fragments_x) // self.batch_size

    # return one batch of data
    def __getitem__(self, index):
        # generate indexes
        lower_bound = index * self.batch_size
        upper_bound = (index+1)*self.batch_size
        indexes = self.indexes[lower_bound:upper_bound]
        # generate list of fragments
        filepaths = [self.fragments_x[i] for i in indexes]
        # initializing arrays
        x = np.empty((self.batch_size, *self.data_shape))
        # load data
        for i, filepath in enumerate(filepaths):
            #print('[d] loading file {}'.format(filepath))
            x[i, ] = np.load(filepath)

        # handle labels
        if self.label_type == 'clean':
            y = np.empty((self.batch_size, *self.data_shape))
            for i, filepath in enumerate(filepaths):
                # TODO find a way and use gen_cache_path?
                filename = osp.basename(filepath)
                basedir = osp.dirname(osp.dirname(osp.dirname(filepath)))
                proc_str = self.proc_func_label.__name__ if self.proc_func_label else 'none'
                filepath_y = osp.join(basedir, 'clean', proc_str, filename)
                # laad data
                #print('[d] loading file {}'.format(filepath_y))
                y[i, ] = np.load(filepath_y)
        elif self.label_type == 'x':
            y = x
        else:
            print('[d] Label type unsupported, y = empty!')
            y = np.empty((self.batch_size))

        return x, y

    # return shape of data
    @property
    def data_shape(self):
        return self._data_shape

    # return number of individual audio fragments
    @property
    def n_fragments(self):
        return len(self.fragments_x) + len(self.fragments_y)
