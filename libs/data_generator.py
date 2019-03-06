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
import random as rnd
import time
rnd.seed(int(time.time())) # generate seed from the time at which this script is run
#from scipy.io.wavfile import write

import libs.updated_utils

""" functions creating different types of noise """    
def white_noise(x, SNR):
    print('Using white noise')
    
    N = max(x.shape)
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
    
    N = max(x.shape)
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

        myVelvetNoise = [rnd.uniform(-1, 1) for k in range( N) ] #random numbers between -1 and 1
        noise = [sigma * ((vv> rate_zero) - (vv < -rate_zero)) for vv in myVelvetNoise]
        
        #        params.NonZeros = np.sum(np.abs(noise))
        #        params.realZeroRate = 1-params.NonZeros/noise.shape[0];
        #        [params.indNonZeros, ~,params.valNonZeros] = find(SV);
        #        params.sizeVN = N;
        return noise#, params
    return createVelvetNoise() # ??
  


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

    
class DataGenerator(keras.utils.Sequence):
    # util.frame : split a vector in overlapping windows
    def __init__(self, filenames, dataset_path, channels, batch_size, fragment_size, hop_length, dim=1, noise_choice = 0, sr=22050, noisingFile_path = None,
            createNoise_funcs = [white_noise, pink_noise, velvet_noise, take_file_as_noise], convNoise = False, SNR = 5):
        # christie@purwins-SYS-7048GR-TR:/data/riccardo_datasets/cnn_news/newsday031513.wav
        self.filenames = filenames
        self.batch_size = batch_size
        self.channels = channels
        self.sr = sr
        self.dim = dim
        self.fragment_size = fragment_size
        self.hop_length = hop_length
        self.createNoise_funcs = createNoise_funcs
        self.dataset_path = os.path.expanduser(dataset_path)
        self.noisingFile_path = os.path.expanduser(noisingFile_path)
        self.nFragment = None
        self.SNR = SNR
        self.noise_choice = noise_choice
        self.convNoise = convNoise
        

    def __len__(self):
        # estimation of the number of chunks that we'll get
        if not self.nFragment:
            nFragment = 1 + int((np.floor(self.length_OrigSignal - self.fragment_size)) / self.hop_length) #hop_length = 1
            #n_frames = 1 + int((len(y) - frame_length) / hop_length)
        self.indexes = range(nFragment)
        
        return nFragment

    def __getitem__(self):
#        path = os.path.join(self.dataset_path + '.wav')
#        s = librosa.load(path, self.sr)
        
        # TODO need to consider batch size
        batch, batches = [], []
        for file in self.filenames:

            s = librosa.input.read_wav(file, self.sr)
            self.length_OrigSignal = len(s)
        
            # Generate data
            x = self.__data_generation(s)
            if batch.shape > self.batch_size:
                batches.append(batch)
                batch = []
            batch.append(x)
        return batches 

    def __data_generation(self, s):
        
        
        # trim s to a multiple number of fragment_size:
        keptS_length = len(s) - (len(s) % self.fragment_size)
        s_short = s[ :keptS_length] 
        
        # create noise        
        if self.convNoise == False:# additive noising
            noise = self.createNoise_funcs[self.noise_choice](s_short, self.SNR) #create the noise wanted
            x = s_short + noise
            
        else: # self.convNoise == True or type(self.convNoise)==int :
            s_veryshort = s[ :self.fragment_size]
            noise = self.createNoise_funcs[self.noise_choice](s_veryshort, self.SNR) #create the noise wanted
            
            if type(self.convNoise)==int :
                conv_length = self.convNoise
            else:
                conv_length = self.fragment_size
            if self.noise_choice == 3: #i.e. if take_file_as_noise
                noise = self.rdn_noise(s, noise) 
            
            x = np.convolve(s_short, noise[ :conv_length], 'same')
                 
            
        x = np.reshape(x, (self.nFragment ,self.fragment_size ))
        
        write_path = os.path.join(self.dataset_path + 'noise_' + self.SNR + 'dB.wav')
        librosa.output.write_wav(write_path, s, self.sr)
#        write(write_path, self.sample_rate, x)
        return x

      
    
    def rdn_noise(self, s, noise):
        # Function: randomize the order of the chunks of the file that is used as noise
        
        # Generate data
        if len(noise) >= len(s):
            noise = noise[:len(s)]
            # randomization of the noising track
            rdnOrder = [np.argsort([rnd.uniform(0,1) for i in range(self.nFragment)])]
            new_noise = []
            new_noise.append( [noise[rdni * self.fragment_size : (rdni+1) * self.fragment_size ] for rdni in rdnOrder ])
            
        elif len(noise) < len(s): # need to create some noise in addition: white_noising / interpolation / ...
            # here: white_noising
            keptNoise_length = len(noise) - (len(noise) % self.fragment_size)
            noise_noised = noise[ :(len(s) - keptNoise_length)] + [rnd.uniform(-1 ,1) for k in range((len(s) - keptNoise_length))]
            new_noise = noise[ :keptNoise_length].append( noise_noised )
#            new_noise = np.reshape( noise[ :keptNoise_length].append( noise_noised ), (1, self.nFragment * self.fragment_size ))
        
        # TODO: sonme normalisation?
        return new_noise         
