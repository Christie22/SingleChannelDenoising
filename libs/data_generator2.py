"""
preprocess data! edit `__init__`, `__len__`, `__data_generation`, etc to support:
- input files as spectrograms / time-domain signals
- (DONE) fragmentation of input files (long file => multiple chunks)
- pre-processing function as input argument! (accepts time-domain data and 
extra parameters (sr, n_mels, etc...))
- (DONE) clean up the label mess 
- _noising_ functions as input argument! i.e. pass list of functions to DataGenerator 
(each function take time-domain data and SNR value __only__)

"""
import os
import keras
import librosa
import numpy as np
from updated_utils import *
import random as rnd
rnd.seed(576)

class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_path, batch_size, dim,
                 noising_func ='white_noising', shuffle=True, SNR):
        self.dim = dim
        self.batch_size = batch_size
        self.data_path = os.path.expanduser(data_path)

        self.shuffle = shuffle
        self.pre_processing = pre_processing
        self.nBatch = None
        self.on_epoch_end()
        self.SNR = SNR

    def __len__(self):
        # estimation of the number of chunks that we'll get
        if not self.nBatch:
            nBatch = int(np.floor(self.length_OrigSignal / self.batch_size))
        self.indexes = range(nBatch)
        
        return nBatch

    def __getitem__(self):
#        if not self.length_OrigSignal:
        path = os.path.join(self.data_path + '.wav')
        s = librosa.load(path)
        
        self.length_OrigSignal = len(s)
        
        # Generate indexes of the batch
#        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

#        self.indexes = range(self.length_OrigSignal )
#        
        # Generate data
        x = self.__data_generation(s)

        return x

#    def on_epoch_end(self):
#        self.indexes = np.arange(len(self.list_IDs))
#        if self.shuffle:
#            np.random.shuffle(self.indexes)

    def __data_generation(self, s):
        # Initialization
        x = np.empty(self.batch_size, *self.dim)

        # Generate data
        for i in self.indexes:
            # Store sample
            x[i,] = self._noising_(self.pre_processing)
            
            
            
            
            if self.pre_processing == 'white':

                s = self.white_noise(s)
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
            
            else:
                x[i, ] = None


        return x
      
            
    def white_noising(x, SNR):
        N = max(x.shape);
        sigma = np.sqrt( (x @ x.T) / (N * 10**(SNR/10)) )
        x_noised = [x[k] + sigma * rnd.uniform(-1,1) for k in range( N) ]
        return x_noised
    
    def pink_noising(x, SNR):
        # TODO
        x_noised = x
        return x_noised
    
    def music_noising(x, SNR):
        musicFromFile = load()
        # TODO
        x_noised = x
        return x_noised
    
    def speech_noising(x, SNR):
        # TODO
        x_noised = x
        return x_noised
    
    
    def add_noising_from_file(filepath):
        def noising_prototype(x, SNR):
            noise = load(filepath)
            if len(noise) > len(x):
                noise = noise[:len(x)]
            elif len(noise) < len(x):
                noise = 
                
        return x + noise
    return noising_prototype
    

    def conv_noising_from_file(filepath, durConv, offset):
        def noising_prototype(x, SNR):
            noise = librosa.load(filepath, offset = offset, duration = durConv)
            
        return np.convolve(x,  noise, 'same')
    
    return noising_prototype