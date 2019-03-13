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
    
    bool_begin = 1
    
    if config_file_or_dict is None:
        return None

    if isinstance(config_file_or_dict, dict):
        # retrieve params' values
        print('Reading room configuration from a dict')
        rir_dict = config_file_or_dict
        keys = rir_dict.keys()
        
        sampling_rate = rir_dict.pop('sampling_rate', '') if 'sampling_rate' in keys else 16000
        room_dim = rir_dict.pop('room_dim', '') if 'room_dim' in keys else [10, 7, 3]
        rt60 = rir_dict.pop('rt60', '')  if 'rt60' in keys else 0.3
        absorption = roomsimove_single.rt60_to_absorption(room_dim, rt60)
        
        mic_pos = rir_dict.pop('mic_pos', '') if 'mic_pos' in keys else [2, 2, 2]
        source_pos = rir_dict.pop('source_pos', '') if 'source_pos' in keys else [2,2,2]
        
        #print('absorption is {}'.format(absorption))
                
        # TODO add constraints on the position of the source relatively to the mikes if necessary
        
        #[rnd.random()*room_dim[ii] for ii in range(3)]
        #    checkDistance = scipy.linalg.norm(np.array(source_pos)- np.array(mic_pos))
        #source_pos = source_pos/np.linalg.norm(source_pos) + mic_pos #normalise to have 1m distance between source and mic
        Adim = ['Ax1','Ax2','Ay1','Ay2','Az1','Az2']
        for ia, a in enumerate(absorption):
            A = tuple([a for i in range(7)])
            print('aborpt:'+str(a))
            if bool_begin == 1:
                config_file = 'config_file.txt'
                f = open(config_file, "w+")
                f.write('% Sampling frequency (in hertz)\n')
                f.write("Fs \t%d\n\n" % sampling_rate) #f.seek(5) 
                f.write('% Room size (in meters)\n')
                f.write('room_size \t%d\t%d\t%d\n\n' % (room_dim[0], room_dim[1], room_dim[2]))
                f.write('% Frequency-dependent absorption for surfaces x=0, x=Lx, y=0, y=Ly, z=0, z=Lz\n')
                f.write("F_abs \t%d\t%d\t%d\t%d\t%d\t%d\t%d\n" % (125,250,500,1000,	2000,4000,8000))
                for ad in Adim:
                    f.write(ad+ "\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n" % A)

                f.write('\n')
                f.write('% Sensor positions (in meters)\n')
                if len(np.array(mic_pos).shape) == 1:
                    f.write("sp"+str(1)+"\t%.2f\t%.2f\t%.2f\n" % (mic_pos[0],mic_pos[1],mic_pos[2]))
                    f.write("% Sensor orientations (azimuth, elevation and roll offset in degrees, positive for slew left, nose up or right wing down)\n" )
                    f.write("so"+str(1)+"\t%.2f\t%.2f\t%.2f\n" % (0,0,0))
                    f.write("% Sensor direction-dependent impulse responses (e.g. gains or HRTFs)\n")
                    f.write("sd"+str(1)+"\t%s\n" % 'omnidirectional\n')
                else:
                    for n in range(np.array(mic_pos).shape[0]):
                        f.write("sp"+str(n+1)+"\t%.2f\t%.2f\t%.2f\n" % (mic_pos[n][0],mic_pos[n][1],mic_pos[n][2]))
                    f.write("% Sensor orientations (azimuth, elevation and roll offset in degrees, positive for slew left, nose up or right wing down)\n" )
                    for n in range(np.array(mic_pos).shape[0]):
                        f.write("so"+str(n+1)+"\t%.2f\t%.2f\t%.2f\n" % (0,0,0))
                    f.write("% Sensor direction-dependent impulse responses (e.g. gains or HRTFs)\n")
                    for n in range(np.array(mic_pos).shape[0]):
                        f.write("sd"+str(n+1)+"\t%s\n" % 'omnidirectional\n')
        
                f.close()
#                bool_begin = 0
                #config_file_or_dict = config_file
                
                
            else:
                f = open(config_file, "r+")
                for line in f:
                    print(line)
                    line = line.strip()
                    if line.startswith('A') :
                        # TODO replace
                        content = f.readlines()
                        for ad in Adim:
#                            f.write(ad + "\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n" % A)
                            f.write(content.replace(absorption[ia-1], a))

                        break

                f.close()
                #content = f.read()
                #f.seek(0)
                #f.truncate()
                #f.write(content.replace('replace this', 'with this'))
            
            print('Reading room configuration from a config file')
            sim_rir = roomsimove_single.RoomSim.init_from_config_file(config_file)
            source_pos = [1,1,1] #[rnd.random()*room_dim[ii] for ii in range(3)]
            rir = sim_rir.create_rir(source_pos)
        
            for ii in range(rir.shape[1]):
                np.save('rirs/rir_absor_'+'%.2f' % a +'_mic_'+str(ii) ,rir[:,ii])

            fp = open('RIR_params.txt','a')
            fp.write('sampling_rate: \t%d\n' % 16000)
            fp.write("room_dim: "+str(1)+"\t%.2f\t%.2f\t%.2f\n" % (room_dim[0],room_dim[1],room_dim[2]))
            fp.write('rt60: %.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t\n' % tuple(list(rt60)))
            fp.write('corresp absorptions: %.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t\n' % tuple(list(absorption)))
            fp.write("mic_pos: "+str(1)+"\t%.2f\t%.2f\t%.2f\n" % (mic_pos[0],mic_pos[1],mic_pos[2]))
            fp.write("source_pos: "+str(1)+"\t%.2f\t%.2f\t%.2f\n" % (source_pos[0],source_pos[1],source_pos[2]))
            fp.write('\n\n')
            fp.close()

    print('Creation of the RIRs completed')    
    
    return rir
/data/riccardo_datasets/rirs
