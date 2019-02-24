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
import pandas as pd
from updated_utils import *
import random as rnd
import time
rnd.seed(int(time.time())) # generate seed from the time at which this script is run

""" functions creating different types of noise """    
def white_noise(x, SNR):
    print('Using white noise')
    
    N = max(x.shape);
    # N = len(x) alternatively
    sigma = np.sqrt( (x @ x.T) / (N * 10**(SNR/10)) )
    noise = [sigma * rnd.uniform(-1,1) for k in range( N) ]
    
    return noise

def pink_noise(x, SNR):
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

    sigma = np.sqrt( (x @ x.T) / (nrows * 10**(SNR/10)) )
    noise= sigma*(total.values-np.mean(total.values)) / (max(total.values) - np.mean(total.values))
    
    return noise

def velvet_noise(x, SNR):
    print('Using velvet noise')
    
    N = max(x.shape);
    # N = len(x) alternatively
    sigma = np.sqrt( (x @ x.T) / (N * 10**(SNR/10)) )
    print('sigma = {0}'.format(sigma))
    def createVelvetNoise(rate_zero = .95):
        ##### Role: create a vector of velvet noise (containing exclusively {-1,0,1})
        ## Input:
        # rate_zero(optional): pourcentage (between 0 and 1) of "0" in the output vector. 
        ## ouput: velvet noise 
        # V: standard vector
        # params(optional) (struct): parametres (nb of zeros, indices, values) TODO



        # should be equally destributed between (-1) and 1.
        myVelvetNoise = [rnd.uniform(-1, 1) for k in range( N) ] #random numbers between -1 and 1
        noise = [sigma * ((vv> rate_zero) - (vv < -rate_zero)) for vv in myVelvetNoise]
        
        #        params.NonZeros = np.sum(np.abs(noise))
        #        params.realZeroRate = 1-params.NonZeros/noise.shape[0];
        #        [params.indNonZeros, ~,params.valNonZeros] = find(SV);
        #        params.sizeVN = N;
        return noise
    return createVelvetNoise() #??
  


def take_file_as_noise(x, SNR):
    N = len(x)
    sigma = np.sqrt( (x @ x.T) / (N * 10**(SNR/10)) )
    def noising_prototype( filepath):
        print('Using the following file as noise: {0}'.format(filepath))
#        path = os.path.join(filepath + '.wav')
        load_noise = np.load(filepath)
        noise =  sigma * (load_noise - np.mean(load_noise)) + np.mean(load_noise) #??? TODO
        return noise
    return noising_prototype

    
class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_path, batch_size, dim, noise_choice = 0, shuffle=True, noisingFile_path = None,
            createNoise_funcs = [white_noise, pink_noise, velvet_noise, take_file_as_noise], convNoise = False, SNR = [0,5,10,15]):
        self.dim = dim
        self.batch_size = batch_size
        self.createNoise_funcs = createNoise_funcs
        self.data_path = os.path.expanduser(data_path)
        self.noisingFile_path = os.path.expanduser(noisingFile_path)
        self.shuffle = shuffle
        self.nBatch = None
        self.SNR = SNR
        self.noise_choice = noise_choice
        self.convNoise = convNoise
        

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
        # trim s to a multiple of batch_size:
        keptS_length = len(s) - (len(s) % self.batch_size)
        s_short = s[ :keptS_length] 
        
        # create noise
        noise = self.createNoise_funcs[self.noise_choice](s_short, self.SNR) #create the noise wanted
        
        if self.noise_choice == 3: #i.e. if take_file_as_noise
            noise = self.rdn_noise(s, noise) 
        
        if self.convNoise == True or type(self.convNoise)==int :
            if type(self.convNoise)==int :
                conv_length = self.convNoise
            else:
                conv_length = self.batch_size
                
            x = np.convolve(s_short, noise[ :conv_length], 'same')
            
        else: # additive noising
            x = s_short + noise
           
            
        x = np.reshape(x, (self.nBatch ,self.batch_size ))
        
        return x
    # write them somewhere TODO
      
    
    def rdn_noise(self, s, noise):
        # Function: randomize the order of the chunks of the file that is used as noise
        
        # Generate data
        if len(noise) >= len(s):
            noise = noise[:len(s)]
            # randomization of the noising track
            rdnOrder = [np.argsort([rnd.uniform(0,1) for i in range(self.nBatch)])]
            new_noise = []
            new_noise.append( [noise[rdni * self.batch_size : (rdni+1) * self.batch_size ] for rdni in rdnOrder ])
            
        elif len(noise) < len(s): # need to create some noise in addition: white_noising / interpolation / ...
            # here: white_noising
            keptNoise_length = len(noise) - (len(noise) % self.batch_size)
            noise_noised = noise[ :(len(s) - keptNoise_length)] + [rnd.uniform(-1 ,1) for k in range((len(s) - keptNoise_length))]
            new_noise = noise[ :keptNoise_length].append( noise_noised )
#            new_noise = np.reshape( noise[ :keptNoise_length].append( noise_noised ), (1, self.nBatch * self.batch_size ))
        
        # TODO: sonme normalisation?
        return new_noise         
