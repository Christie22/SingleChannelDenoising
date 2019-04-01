# TODO
# - multithreaded caching?
# - verify (x, target) matching
# - implement other label_type modes

import os
import glob
import hashlib
import itertools
import keras
import librosa as lr
import numpy as np
import pandas as pd
import os.path as osp
from scipy.signal import fftconvolve
from tqdm import tqdm

from libs.processing import make_fragments


class DataGenerator(keras.utils.Sequence):
    # util.frame : split a vector in overlapping windows
    def __init__(self, filepaths, cache_path=None, sr=16000,
                 rir_path=None, noise_funcs=[None], noise_snrs=[0],
                 n_fft=512, hop_length=128, win_length=512,
                 proc_func=None, proc_func_label=None,
                 frag_hop_length=64, frag_win_length=32,
                 shuffle=True, label_type='clean', batch_size=32, force_cacheinit=False):
        # arguments for chache folder hash
        proc_args = tuple([
            sr,
            n_fft,
            hop_length,
            win_length,
            frag_hop_length,
            frag_win_length,
            label_type])
        # dataset cfg
        self.filepaths = filepaths
        self.cache_path = osp.expanduser(cache_path) if cache_path else osp.join(
            osp.dirname(filepaths[0]), 'cache_{}'.format(self.hash_args(proc_args)))
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
        # setup cache
        if force_cacheinit or not osp.exists(self.cache_path):
            self.init_cache()
        else:
            try:
                self.load_cache()
                if not self.test_cache():
                    print( '[d] Cache indexing test failed. Attempting to initialize it...')
                    self.init_cache()
            except Exception as e:
                print('[d] Cache indexing caused an exception. Attempting to initialize it...')
                self.init_cache()
        # shuffle batches if needed
        self.on_epoch_end()
        # print some debugging info
        print('[d] Frame hop    (hop_length) (ms): {:.0f}'.format(
            lr.samples_to_time(hop_length, sr=sr)*1e3))
        print('[d] Frame length (win_length) (ms): {:.0f}'.format(
            lr.samples_to_time(win_length, sr=sr)*1e3))
        print('[d] Fragment hop    (frag_hop_length) (ms): {:.0f}'.format(
            lr.frames_to_time(frag_hop_length, sr=sr, n_fft=win_length, hop_length=hop_length)*1e3))
        print('[d] Fragment length (frag_win_length) (ms): {:.0f}'.format(
            lr.frames_to_time(frag_win_length, sr=sr, n_fft=win_length, hop_length=hop_length)*1e3))

    # calcualate md5 hash of input arguments
    def hash_args(self, args):
        m = hashlib.md5()
        for x in args:
            m.update(str(x).encode())
        return m.hexdigest()[:6]

    # load list of RIR files
    def load_rirs(self):
        print('[d] Loading all RIRs files from {}'.format(self.rir_path))
        try:
            filelist = glob.glob(osp.join(self.rir_path, '*.npy'))
            print('[d] Loaded {} files'.format(len(filelist)))
        except Exception as e:
            filelist = [None]
            print('[d] Loaded no files')
        return filelist or [None]

    # init cache
    def init_cache(self):
        print('[d] Initializing cache in {}...'.format(self.cache_path))
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
                    x_noised = noise_func(
                        x_rev, sr=self.sr, snr=snr) if noise_func else x_rev
                    # convert to TF-domain
                    s_noised = lr.core.stft(
                        x_noised, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
                    # apply data repr processing
                    s_proc = self.proc_func(s_noised) if self.proc_func else np.reshape(
                        s_noised, (*s_noised.shape, 1))

                # fragment data
                s_frags = make_fragments(
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
        print('[d] Cache ready, generated {} noisy and {} clean fragments of shape {}'.format(
            len(self.fragments_x), len(self.fragments_y), self.data_shape))

    # load a pre-initialized cache (use with caution)
    def load_cache(self):
        print('[d] Some cache exists! Indexing cache in {}...'.format(self.cache_path))
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
        print('[d] Cache ready, indexed {} noisy and {} clean fragments of shape {}'.format(
            len(self.fragments_x), len(self.fragments_y), self.data_shape))

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

    # test few cache fragments to make sure they exist
    def test_cache(self):
        try:
            [np.load(f) for f in np.random.choice(self.fragments_x, 5)]
        except Exception as e:
            return False
        return True

    def apply_reverb(self, x, rir_filepath):
        rir = np.load(rir_filepath)
        y = fftconvolve(rir, x)
        return y


    # callback at each epoch (shuffles batches)
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.fragments_x))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # number of batches
    def __len__(self):
        if self.fragments_x is None:
            self.init_cache()
        return len(self.fragments_x) // self.batch_size

    # returns one batch of data
    def __getitem__(self, index):
        # generate indexes
        lower_bound = index * self.batch_size
        upper_bound = (index+1)*self.batch_size
        indexes = self.indexes[lower_bound:upper_bound]
        # generate list of fragments
        filepaths = [self.fragments_x[i] for i in indexes]
        # initializing arrays
        x = np.empty((self.batch_size, *self.data_shape))
        std_files = np.empty(self.batch_size)
        # load data
        for i, filepath in enumerate(filepaths):
            loaded_file = np.load(filepath)
            #print('[d] loading file {}'.format(filepath))
            std_files[i] = np.std(loaded_file)
            x[i, ] = (loaded_file - np.mean(loaded_file)) / std_files[i]

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
                # loaded_file_y = np.load(filepath_y)
                # y[i, ] = (loaded_file_y-np.mean(loaded_file_y)) / std_files[i] # not sure
        elif self.label_type == 'x':
            y = x
        else:
            print('[d] Label type unsupported, y = empty!')
            y = np.empty((self.batch_size))

        return x, y, std_files

    # return shape of data
    @property
    def data_shape(self):
        return self._data_shape

    # return number of individual audio fragments
    @property
    def n_fragments(self):
        return len(self.fragments_x)
