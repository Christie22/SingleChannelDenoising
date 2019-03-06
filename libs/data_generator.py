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
import glob
import keras
import librosa as lr
import numpy as np
import pandas as pd
import random as rnd
#from scipy.io.wavfile import write

import libs.updated_utils
import libs.processing as processing


    
class DataGenerator(keras.utils.Sequence):
    # util.frame : split a vector in overlapping windows
    def __init__(self, filenames,
                 dataset_path, sr, rir_path, 
                 noise_types=[], noise_snrs=[0],
                 n_fft=512, hop_length=128, win_length=512, 
                 proc_type=None,
                 frag_hop_length=64, frag_win_length=32, 
                 shuffle=True, labels='clean', batch_size=32):

        # dataset cfg
        self.filenames = filenames
        self.dataset_path = os.path.expanduser(dataset_path)
        self.sr = sr
        # reverberation cfg
        self.rir_path = os.path.expanduser(rir_path)
        # noising cfg
        self.noise_types = noise_types
        self.noise_snrs = noise_snrs
        # stft cfg
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        # processing cfg
        self.proc_type = proc_type
        # fragmenting cfg
        self.frag_hop_length = frag_hop_length
        self.frag_win_length = frag_win_length
        # general cfg
        self.shuffle = shuffle
        self.labels = labels
        self.batch_size = batch_size
        # local vars
        self.data_shape = (256, 64, 2) # TODO calculate based on n_fft, processing, and fragment
        self.rir_filenames = self.load_rirs()


    def load_rirs(self):
        print('[d] Loading all RIRs files from {}'.format(self.rir_path))
        filelist = glob.glob(os.path.join(self.rir_path, '*.wav'))
        print('[d] Loaded {} files'.format(len(filelist)))
        return filelist


    def get_data_shape(self):
        return self.data_shape


    def __len__(self):
        print('[d] Calculating total number of input fragments')
        variations = len(self.noise_types) * len(self.noise_snrs) * len()
        file_durations = [lr.core.get_duration(filename=f) for f in self.filenames]
        file_fragments = lr.core.time_to_frames(
            file_durations, 
            sr=self.sr, 
            hop_length=self.frag_hop_length,
            n_fft=self.frag_win_length) - 1
        return file_fragments.sum() * variations

    def __getitem__(self):
#        path = os.path.join(self.dataset_path + '.wav')
#        s = lr.load(path, self.sr)
        
        # TODO need to consider batch size
        batch, batches = [], []
        for file in self.filenames:

            s = lr.input.read_wav(file, self.sr)
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
        lr.output.write_wav(write_path, s, self.sr)
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
