import os
import glob
import itertools
import keras
import librosa as lr
import numpy as np
import pandas as pd
#from scipy.io.wavfile import write

import libs.updated_utils
import libs.processing as processing


class DataGenerator(keras.utils.Sequence):
    # util.frame : split a vector in overlapping windows
    def __init__(self, filenames,
                 dataset_path, sr, rir_path, cache_path=None,
                 noise_funcs=[None], noise_snrs=[0],
                 n_fft=512, hop_length=128, win_length=512, 
                 proc_func=None, proc_func_label=None,
                 frag_hop_length=64, frag_win_length=32, 
                 shuffle=True, labels='clean', batch_size=32):

        # dataset cfg
        self.filenames = filenames
        self.dataset_path = os.path.expanduser(dataset_path)
        self.cache_path = os.path.expanduser(cache_path) if cache_path else os.path.join(self.dataset_path, 'cache')
        self.sr = sr
        # reverberation cfg
        self.rir_path = os.path.expanduser(rir_path)
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
        self.labels = labels
        self.batch_size = batch_size
        # computed vars
        self._data_shape = (256, 64, 2) # TODO calculate based on n_fft, processing, and fragment
        self.rir_filenames = self.load_rirs()
        self.noise_variations = list(itertools.product(self.noise_funcs, self.noise_snrs, self.rir_filenames))
        # local vars
        self.n_fragments = None
        self.indexes = []
        self.batch_buffer = None
        self.leftover_buffer = None
        self.pointer = {
            'index': 0,
            'variation': 0,
            'frag': 0,
        }
        # init random indexes
        self.init_cache()
        self.on_epoch_end()

    # load list of RIR files
    def load_rirs(self):
        print('[d] Loading all RIRs files from {}'.format(self.rir_path))
        filelist = glob.glob(os.path.join(self.rir_path, '*.wav'))
        print('[d] Loaded {} files'.format(len(filelist)))
        return filelist or [None]

    # init cache
    def init_cache(self):
        print('[d] Initializing cache...')
        self.n_fragments = 0
        for filename in self.filenames:
            print('[d] Loading file {}'.format(filename))
            # load data
            filepath = os.path.join(self.dataset_path, filename)
            x, _ = lr.core.load(filepath, sr=self.sr, mono=True)
            # apply variations of noise + clean (labels)
            for noise_variation in self.noise_variations + ['clean']:
                print('[d]  Applying noise variation {}'.format(noise_variation))
                if noise_variation == 'clean':
                    # convert to TF-domain
                    s = lr.core.stft(
                        x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
                    # apply label preprocessing
                    s_proc = self.proc_func_label(s) if self.proc_func_label else s 
                else:
                    noise_func, snr, rir_filename = noise_variation
                    # apply room
                    x_rev = self.apply_reverb(x, rir_filename) if rir_filename else x
                    # apply noise function
                    x_noise = noise_func(x_rev, sr=self.sr, snr=snr) if noise_func else x_rev
                    # convert to TF-domain
                    s_noise = lr.core.stft(
                        x_noise, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
                    # apply data repr processing
                    s_proc = self.proc_func(s_noise) if self.proc_func else s_noise 

                # fragment data
                s_frags = self.make_fragments(s_proc, self.frag_hop_length, self.frag_win_length)
                # store
                for i, frag in enumerate(s_frags):
                    frag_path = self.gen_cache_path(
                        self.cache_path, filename, noise_variation, 
                        self.proc_func if noise_variation != 'clean' else self.proc_func_label, i)
                    print('[d]   Storing frag {} in {}'.format(i, frag_path))
                    #self.store_frag(frag_path, frag)
                    self.n_fragments += 1
        # done
        print('[d] Cache ready.')
    
    # generate filepath for individual fragments
    def gen_cache_path(self, cache_path, filename, noise_variation, proc_func, frag_index):
        print('[d] gen_cache_path chache_path: {}'.format(cache_path))
        path = os.path.join(cache_path, os.path.splitext(filename)[0].replace(' ', '_'))
        print('[d] gen_cache_path path: {}'.format(path))
        if noise_variation == 'clean':
            noise_variation_str = noise_variation
        else:
            noise_func, snr, rir_filename = noise_variation
            noise_variation_str = '{}_{}_{}'.format(
                noise_func.__name__ if noise_func else 'none',
                snr,
                os.path.splitext(rir_filename)[
                    0][-6:] if rir_filename else 'none'
            )
        path = os.path.join(path, noise_variation_str)
        path = os.path.join(path, proc_func.__name__ if proc_func else 'none')
        path = os.path.join(path, 'frag_{}.npy'.format(frag_index))
        return path



    # return overall number of fragments
    def get_n_fragments(self):
        file_durations = [lr.core.get_duration(filename=f) for f in self.filenames]
        file_fragments = lr.core.time_to_frames(
            file_durations, 
            sr=self.sr, 
            hop_length=self.frag_hop_length,
            n_fft=self.frag_win_length) - 1
        return file_fragments.sum()

    # store processed audio fragment into cache
    def store_frag(self, filepath, s):
        np.save(filepath, s)

    # load processed audio fragment from cache
    def load_frag(self, filepath):
        return np.load(filepath)

    def apply_reverb(self, x, rir_filename):
        # TODO
        return x

    # convert T-F data into fragments
    def make_fragments(self, s, frag_hop_len, frag_win_len):
        return [s, s]

    # callback at each epoch (shuffles batches)
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.filenames))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # number of batches
    def __len__(self):
        if not self.n_fragments:
            self.init_cache()
        return self.n_fragments // self.batch_size

    # return one batch of data
    def __getitem__(self, index):
        
        


        #x, y = self.__data_generation(filenames_batch)
        
        return None
    
    # return shape of data
    @property
    def data_shape(self):
        return self._data_shape

