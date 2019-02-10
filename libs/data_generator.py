"""
Created on 31.10.2018

@author: Tomas Gajarsky
@modified by: Riccardo Miccini

Data generator for fitting data on Keras models.

Based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""

import os
import keras
import librosa
import numpy as np
from utilities import *

# NOTES:
# If `labels` is None, it's going to return (x, None) or (x, x) depending on `y_value`.
# In order to return (x, y), set `labels` to a dictionary {'list_ID': label} and `y_value` to 'label'.
class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, data_path, batch_size, dim, n_channels,
                 n_classes, pre_processing='Standard', shuffle=True, y_value='label', norm_factor=142):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.data_path = os.path.expanduser(data_path)
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.y_value = y_value
        self.pre_processing = pre_processing
        self.norm_factor = norm_factor
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_IDs_temp)

        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # Initialization
        x = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty(self.batch_size)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            path = os.path.join(self.data_path, ID + '.npy')

            if self.pre_processing == 'log_mel':
                s = np.load(path)
                s = self.compute_log_mel_s(s)
                s = np.reshape(s, (s.shape[0], s.shape[1], 1))
                x[i, ] = s
                s = None
            elif self.pre_processing == 'reim':
                s = np.load(path)
                s = self.compute_reim_s(s)
                # check for odd number of bins
                if s.shape[0] % 2 != 0:
                    s = s[:-1]
                x[i, ] = s
                s = None
            elif self.pre_processing == 'reim_norm':
                s = np.load(path)
                s = self.compute_reim_norm_s(s)
                # check for odd number of bins
                if s.shape[0] % 2 != 0:
                    s = s[:-1]
                x[i, ] = s
                s = None
            elif self.pre_processing == 'reim_raw':
                s = np.load(path)
                s = self.compute_reim_raw_s(s)
                x[i, ] = s
                s = None
            elif self.pre_processing == 'large_log_mel':
                s = np.load(path)
                s = self.compute_large_log_mel_s(s)
                s = np.reshape(s, (s.shape[0], s.shape[1], 1))
                x[i, ] = s
                s = None
            elif self.pre_processing == 'standard':
                s = np.load(path)
                s = process_data(s)
                x[i, ] = s
            else:
                x[i, ] = None

            # Store class
            if self.labels is not None:
                y[i] = self.labels[ID]

        #x = (x - x.mean()) / x.std()

        if self.y_value == 'label' and self.labels is not None:
            yy = keras.utils.to_categorical(y, num_classes=self.n_classes)
        elif self.y_value == 'x':
            yy = x
        else:
            yy = None

        return x, yy

    def compute_log_mel_s(self, x):
        power_s = np.abs(x) ** 2
        mel_s = librosa.feature.melspectrogram(S=power_s, n_mels=128)
        log_mel_s = np.log10(1 + 10 * mel_s)
        log_mel_s = log_mel_s.astype(np.float32)
        return log_mel_s

    def compute_reim_s(self, x):
        # get real and imag parts
        x_re = librosa.power_to_db(np.real(x) ** 2)
        x_im = librosa.power_to_db(np.imag(x) ** 2)
        return np.dstack((x_re, x_im))

    def compute_reim_norm_s(self, x):
        # get real and imag parts
        s_abs = np.max(np.abs(x)) + 1e-20
        s_re = np.real(x)/s_abs
        s_im = np.imag(x)/s_abs
        s_re_db = librosa.power_to_db(s_re ** 2)
        s_im_db = librosa.power_to_db(s_im ** 2)
        #hardcorded value, obtained by running a script over available data
        s_re_norm = s_re_db / self.norm_factor + 1
        s_im_norm = s_im_db / self.norm_factor + 1

        if np.max(s_re_norm) > 1:
            raise NameError('s_re_norm > 1')
        if np.max(s_im_norm) > 1:
            raise NameError('s_im_norm > 1')
        if np.min(s_re_norm) < 0:
            raise NameError('s_re_norm < 0')
        if np.min(s_im_norm) < 0:
            raise NameError('s_im_norm < 0')

        return np.dstack((s_re_norm, s_im_norm))

    def compute_reim_raw_s(self, x):
        # get real and imag parts
        x_re = np.real(x)
        x_im = np.imag(x)
        return np.dstack((x_re, x_im))

    def compute_large_log_mel_s(self, x):
        power_s = np.abs(x) ** 2
        mel_s = librosa.feature.melspectrogram(S=power_s, n_mels=513)
        log_mel_s = np.log10(1 + 10 * mel_s)
        log_mel_s = log_mel_s.astype(np.float32)
        return log_mel_s
