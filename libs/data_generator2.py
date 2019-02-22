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
    def __init__(self, data_path, batch_size, dim, process_data_func, shuffle=True,
                 createNoise_funcs = [white_noise, pink_noise, take_file_as_noise], SNR):
        self.dim = dim
        self.batch_size = batch_size
        self.createNoise_funcs = createNoise_funcs
        self.data_path = os.path.expanduser(data_path)
        self.process_data_func = process_data_func
        self.shuffle = shuffle
        self.pre_processing = pre_processing
        self.nBatch = None
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
    
        # Generate data
        x = self.__data_generation(s)

        return x

    def __data_generation(self, s):
        noise = createNoise_funcs[0](self, s) #create the noise wanted
        x = apply_noise(s, noise) #apply it to the original sound
        # write them somewhere TODO
        return x
      
                
    """ functions creating different types of noise """    
    def white_noise(x, SNR):
        N = max(x.shape);
        sigma = np.sqrt( (x @ x.T) / (N * 10**(SNR/10)) )
        noise = [rnd.uniform(-1,1) for k in range( N) ]
        return noise
    
    def pink_noise(x, SNR):
        """Generates pink noise using the Voss-McCartney algorithm.
            
        nrows: number of values to generate
        rcols: number of random sources to add
        
        returns: NumPy array
        """
	#TODO take into account SNR
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
        noise= total.values
        return noise
    
    
    def take_file_as_noise(self, filepath):
        def noising_prototype(self, s, SNR):
            noise = load(filepath)
            
            if len(noise) >= len(s):
                noise = noise[:len(s)]
                # randomization of the noising track
                rdnOrder = [np.argsort([rnd.uniform(0,1) for i in range(self.nBatch)])]
                noise_temp = [noise[rdni * self.nBatch : (rdni+1) * self.nBatch ] for rdni in rdnOrder ]
                
            elif len(noise) < len(s):
                
                noise = 1# TODO     
                
        return noise
    return noising_prototype
    
#
#    def conv_noising_from_file(filepath, durConv, offset):
#        def noising_prototype(x, SNR):
#            noise = librosa.load(filepath, offset = offset, duration = durConv)
#            
#        return np.convolve(x,  noise, 'same')
#    
#    return noising_prototype


    """ function applying the noise previously created to the original one"""
    def apply_noise(self, s, noise):
        # Initialisation
        x = np.empty(self.batch_size, *self.dim)
        # Generate data
        for i in self.indexes:
            x[i,] = s[i*self.nBatch : (i+1)*self.nBatch] + noise[i*self.nBatch : (i+1)*self.nBatch]
        return x
         