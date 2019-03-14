# DSP functions such as applying noise, RIRs, or data representation conversions

import numpy as np
import pandas as pd
import random as rnd
import time
import os
from os import path

import roomsimove_single
#import tools.roomsimove_single as roomsimove_single

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
        """
        #### Role: create a vector of velvet noise (containing exclusively {-1,0,1})
        # Input:
         rate_zero(optional): pourcentage (between 0 and 1) of "0" in the output vector. 
        # ouput: velvet noise 
         V: standard vector
         params(optional) (struct): parametres (nb of zeros, indices, values) TODO
        """
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
    """ takes a dictionary or a file as input and returns '.npy' files, RIR_params.txt file that explicits the parameters used for every RIR produced
    
    Source: Room Impulse Response Generator, https://github.com/sunits/rir_simulator_python
    
    # Input Arguments: 
    - room size ([int], 3*1)
    - sampling rate (int, default 16000)
    - tr60 ([float])
    - mics positions ( [[float], 3*1] )
    - source positions ( [[float], 3*1] )
    
    # if many sources, the algorithm computes RIRs for each source separately (loops on the array).
    # if many mics' positions, the algorithm computes RIRs for all positions in a single loop.
    # if many tr60, the algorithm computes RIRs for each tr60 separately (loops on the array).
    # the room is supposed to be homogenous: the absorption calculated fron tr60 is the same on all the walls
    """
    
    bool_begin = 1
    
    if config_file_or_dict is None:
        return None

    if isinstance(config_file_or_dict, dict):
        # retrieve params' values
        print('Reading room configuration from a dict')
        rir_dict = config_file_or_dict
        keys = rir_dict.keys()
        
        # Reading input arguments
        sampling_rate = rir_dict.pop('sampling_rate', '') if 'sampling_rate' in keys else 16000
        room_dim = rir_dict.pop('room_dim', '') if 'room_dim' in keys else [10, 7, 3]
        rt60 = rir_dict.pop('rt60', '')  if 'rt60' in keys else 0.3
        rt60 = np.rray([rt60]) if isinstance(rt60, float) else rt60
        mic_pos = rir_dict.pop('mic_pos', '') if 'mic_pos' in keys else [2, 2, 2]
        source_pos = rir_dict.pop('source_pos', '') if 'source_pos' in keys else [1, 1, 1]
        
        # Calculating nb of rirs that will be calculated
        n_room = 1 if len(np.array(room_dim).shape) == 1 else np.array(room_dim).shape[0]
        n_absorption = 1 if isinstance(rt60,float) else np.array(rt60).shape[0]
        n_source = 1 if len(np.array(source_pos).shape) == 1 else np.array(source_pos).shape[0]
        n_mics = 1 if len(np.array(mic_pos).shape) == 1 else np.array(mic_pos).shape[0]
        n_rirs = n_room * n_absorption * n_mics * n_source
         
        print('%d RIRs to be created' % n_rirs)               
        # TODO add constraints on the position of the source relatively to the mikes if necessary
        
        #[rnd.random()*room_dim[ii] for ii in range(3)]
        #    checkDistance = scipy.linalg.norm(np.array(source_pos)- np.array(mic_pos))
        #source_pos = source_pos/np.linalg.norm(source_pos) + mic_pos #normalise to have 1m distance between source and mic
        
        Adim = ['Ax1','Ax2','Ay1','Ay2','Az1','Az2']
        config_file = os.path.join(os.path.abspath('../rirs')+'/config_file.txt')
        
        for r in range(n_room):
            ro = room_dim if n_room==1 else room_dim[r]
            
            
            absorption = roomsimove_single.rt60_to_absorption(ro, rt60)
            absorption = [absorption] if isinstance(absorption, float) else absorption
            

            
            if r>0:
                f = open(config_file, "r+")
                content = f.read()
                f.seek(0) #plqcing the cursor at the beginning of the doc
                f.truncate() # deleting what is after the cursor
                f.write(content.replace('room_size \t%d\t%d\t%d\n\n' % (room_dim[r-1][0],room_dim[r-1][1],room_dim[r-1][2]), 'room_size \t%d\t%d\t%d\n\n' % (ro[0],ro[1],ro[2])))

                f.close()

                
            for s in range(n_source):
                s_p = source_pos if n_source==1 else source_pos[s]
                    
                for ia in range(n_absorption):
                    a = absorption[ia]
                    A = tuple([a for i in range(7)]) # 7: arbitrary, is the number of frequence bands in the original config_file
                    print('absorption: '+str(a))
                    
                    
                    
                    if bool_begin == 1:
                        f = open(config_file, "w+")
                        
                        f.write('% Sampling frequency (in hertz)\n')
                        f.write("Fs \t%d\n\n" % sampling_rate) #f.seek(5) 
                        f.write('% Room size (in meters)\n')
                        if n_room == 1:
                            f.write('room_size \t%d\t%d\t%d\n\n' % (room_dim[0], room_dim[1], room_dim[2]))
                        else:
                            f.write('room_size \t%d\t%d\t%d\n\n' % (room_dim[0][0],room_dim[0][1],room_dim[0][2]))
                        
                        f.write('% Frequency-dependent absorption for surfaces x=0, x=Lx, y=0, y=Ly, z=0, z=Lz\n')
                        f.write("F_abs \t%d\t%d\t%d\t%d\t%d\t%d\t%d\n" % (125,250,500,1000,	2000,4000,8000))
                        for ad in Adim:
                            f.write(ad+ "\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n" % A)
                        f.write('\n')
                        
                        f.write('% Sensor positions (in meters)\n')
                        
                        if n_mics == 1:
                            f.write("sp"+str(1)+"\t%.2f\t%.2f\t%.2f\n\n" % (mic_pos[0],mic_pos[1],mic_pos[2]))
                            
                            f.write("% Sensor orientations (azimuth, elevation and roll offset in degrees, positive for slew left, nose up or right wing down)\n" )
                            f.write("so"+str(1)+"\t%.2f\t%.2f\t%.2f\n\n" % (0,0,0))
                            
                            f.write("% Sensor direction-dependent impulse responses (e.g. gains or HRTFs)\n")
                            f.write("sd"+str(1)+"\t%s\n" % 'omnidirectional\n')
                        
                        else:
                            for n in range(n_mics):
                                f.write("sp"+str(n+1)+"\t%.2f\t%.2f\t%.2f\n" % (mic_pos[n][0],mic_pos[n][1],mic_pos[n][2]))
                            f.write('\n')
                            f.write("% Sensor orientations (azimuth, elevation and roll offset in degrees, positive for slew left, nose up or right wing down)\n" )
                            for n in range(n_mics):
                                f.write("so"+str(n+1)+"\t%.2f\t%.2f\t%.2f\n" % (0,0,0))
                            f.write('\n')
                            f.write("% Sensor direction-dependent impulse responses (e.g. gains or HRTFs)\n")
                            for n in range(n_mics):
                                f.write("sd"+str(n+1)+"\t%s\n" % 'omnidirectional')
                            f.write('\n')
                            
#                        f.close()
                        bool_begin = 0
                        
                        
                    else:
                    
                        f = open(config_file, "r+")
                        content = f.read()
                        f.seek(0) #plqcing the cursor at the beginning of the doc
                        f.truncate() # deleting what is after the cursor
                        f.write(content.replace('%.2f' % absorption[ia-1], '%.2f' % a))
                    f.close()
        
                    
                    print('Reading room configuration from a config file')
                    sim_rir = roomsimove_single.RoomSim.init_from_config_file(config_file)

                    rir = sim_rir.create_rir(s_p)
            
                    for m in range(rir.shape[1]): #for m in range(n_mics)
                        np.save(os.path.abspath('../rirs')
                        +'/rir_tr60_'+'%.2f' % rt60[ia]
                        +'_mic_'+str(m)
                        +'_source_'+str(s) 
                        +'_room_'+str(r) , rir[:,m])

        fp = open(os.path.join(os.path.abspath('../rirs')+'/RIR_params.txt'), "w+")
        fp.write('%d RIRs were created with all possible combinations of the following parameters:\n\n' % n_rirs)
        fp.write('Sampling_rate (in Hz): \t%d\n\n' % 16000)
        
        if n_room==1:
            fp.write("Room dimensions (in m): "+str(0)+"\t%.2f\t%.2f\t%.2f\n" % (room_dim[0],room_dim[1],room_dim[2]))
        else:
            [fp.write("Room dimensions (in m): "+str(r)+"\t%.2f\t%.2f\t%.2f\n" % (room_dim[r][0],room_dim[r][1],room_dim[r][2])) for r in range(n_room)]
        fp.write('\n')
        fp.write('Attenuation time [-60dB] (in sec):' + '%.2f\t'*n_absorption % tuple(list(rt60)) +'\n' )
        fp.write('Corresponding absorptions:'+ '%.2f\t'*n_absorption % tuple(absorption) +'\n\n' )
    
        if n_mics==1:
            fp.write("Position of the mike ("+str(0)+"): \t%.2f\t%.2f\t%.2f\n" % (mic_pos[0],mic_pos[1],mic_pos[2])) 
        else:
            [fp.write("Position of the mike ("+str(m)+"): \t%.2f\t%.2f\t%.2f\n" % (mic_pos[m][0],mic_pos[m][1],mic_pos[m][2])) for m in range(n_mics)]
        fp.write('\n')
        
        if n_source==1:
            fp.write("Position of the source ("+str(0)+"): \t%.2f\t%.2f\t%.2f\n" % (source_pos[0],source_pos[1],source_pos[2])) 
        else:
            [fp.write("Position of the source ("+str(s)+"): \t%.2f\t%.2f\t%.2f\n" % (source_pos[s][0],source_pos[s][1],source_pos[s][2])) for s in range(n_source)]
        

    fp.close()

    print('Creation of %d RIRs completed and available at %s' % (n_rirs, os.path.abspath('../rirs')) +
          '.\n Their description is available in %s' % os.path.join(os.path.abspath('../rirs')+'/RIR_params.txt'))    
    
    return rir
