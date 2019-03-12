# DSP functions such as applying noise, RIRs, or data representation conversions

import numpy as np
import pandas as pd
import random as rnd
import time

#import libs.roomsimove_single 
import libs.rir_simulator_python.roomsimove_single as roomsimove_single 

# generate seed from the time at which this script is run
rnd.seed(int(time.time()))


### PRE-POST PROCESSING FUNCTIONS
def s_to_reim(s):
    # remove a bin if odd number
    if s.shape[0] % 2 != 0:
        s = s[:-1]
    # split re/im
    re = np.real(s)
    im = np.imag(s)
    # stack
    reim = np.dstack((re, im))
    return reim



### NOISING FUNCTIONS

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


### CREATE REVERB FILTERS
def create_RIR(config_file_or_dict=None):
    if config_file_or_dict is None:
        return None
    

    if isinstance(config_file_or_dict, str):
        #room_sensor_config.txt
        # config_file
        print('Reading room configuration from a config file')
        sim_rir = roomsimove_single.RoomSim.init_from_config_file(config_file_or_dict)
        source_pos = [1,1,1] #[rnd.random()*room_dim[ii] for ii in range(3)]
        rir = sim_rir.create_rir(source_pos)
        
    elif isinstance(config_file_or_dict, dict):
        # retrieve params' values
        print('Reading room configuration from a dict')
        keys = config_file_or_dict.keys()
        
        room_dim = config_file_or_dict.pop('room_dim', '') if 'room_dim' in keys else [10, 7, 3]
        rt60 = config_file_or_dict.pop('rt60', '')  if 'rt60' in keys else 0.3
        absorption = roomsimove_single.rt60_to_absorption(room_dim, rt60)
        
        room = roomsimove_single.Room(room_dim, abs_coeff=absorption)

        mic_pos = config_file_or_dict.pop('mic_pos', '') if 'mic_pos' in keys else [2, 2, 2]
#            mics = []
#            for p in range(mic_pos.shape):
#                mics.append(roomsimove_single.Microphone(mic_pos[p,], 1,  \
#                                    orientation=[0.0, 0.0, 0.0], direction='omnidirectional'))
        mics = roomsimove_single.Microphone(mic_pos, 1,  \
                                orientation=[0.0, 0.0, 0.0], direction='omnidirectional')
        

        sampling_rate = config_file_or_dict.pop('sampling_rate', '') if 'sampling_rate' in keys else 16000
        
        sim_rir = roomsimove_single.RoomSim(sampling_rate, room, mics, rt60)
        
        source_pos = config_file_or_dict.pop('source_pos', '') if 'source_pos' in keys else [2,2,2]
        #[rnd.random()*room_dim[ii] for ii in range(3)]
        #    checkDistance = scipy.linalg.norm(np.array(source_pos)- np.array(mic_pos))
        #source_pos = source_pos/np.linalg.norm(source_pos) + mic_pos #normalise to have 1m distance between source and mic
        rir = roomsimove_single.do_everything(room_dim, mic_pos, source_pos, rt60)
    
 
    for ii in range(rir.shape[1]):
        np.save('rirs/rir'+str(ii) ,rir[:,ii])
        
    print('Creation of the RIRs completed')    
    
    return rir
