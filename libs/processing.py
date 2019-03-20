# DSP functions such as applying noise, RIRs, or data representation conversions

import numpy as np
import pandas as pd
import random as rnd
import time
import os
from os import path
import scipy

# generate seed from the time at which this script is run
rnd.seed(int(time.time()))

### PRE/POST PROCESSING FUNCTIONS
# convert complex spectrograms to Re/Im representation
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

# convert Re/Im representation to complex spectrograms
def reim_to_s(reim):
    # extract real and imaginary components
    re = reim[..., 0]
    im = reim[..., 1]
    # combine into complex values
    s = re + 1j * im
    # add previously removed bin
    pad_shape = list(s.shape)
    pad_shape[-2] = 1
    pad_shape = tuple(pad_shape)
    padding = np.zeros(pad_shape)
    s = np.concatenate((s, padding), axis=-2)
    return s


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
        
        # Reading input arguments and converting into array
        sampling_rate = rir_dict.pop('sampling_rate', '') if 'sampling_rate' in keys else 16000
        room_dim = np.array(rir_dict.pop('room_dim', '')) if 'room_dim' in keys else [10, 7, 3]
        rt60 = rir_dict.pop('rt60', '')  if 'rt60' in keys else [0.3]
        if isinstance(rt60, float):
            rt60 = np.array([rt60]) 
        elif isinstance(rt60, list):
            rt60 = np.array(rt60) 
        else:
            rt60
        mic_pos = np.array(rir_dict.pop('mic_pos', '')) if 'mic_pos' in keys else np.array([])
        source_pos = np.array(rir_dict.pop('source_pos', '')) if 'source_pos' in keys else np.array([])
        check_options= 1 if (len(mic_pos.shape)==0 or len(source_pos.shape)==0) else 0
        
        # Optional parameters: constraints on the position of the source relatively to the mikes 
        if 'options' in keys:
            options = rir_dict.pop('options','')  
            opt_keys = options.keys()
            dist_mic_source = options.pop('dist_mic_source','') if 'dist_mic_source' in opt_keys else []
            if isinstance(dist_mic_source,float) or isinstance(dist_mic_source,int):
                dist_mic_source= np.array([dist_mic_source])
            elif isinstance(dist_mic_source,list):
                dist_mic_source= np.array(dist_mic_source)
            n_gen_rdn = np.array(options.pop('gen_rdn_sources_and_mics','')) if 'gen_rdn_sources_and_mics' in opt_keys else np.array([])

            if n_gen_rdn.shape[0]>0 and n_gen_rdn.shape[0]!=2: 
                raise AssertionError('2 integers are needed for the creation of randomized positions: \n 1st is the number of sources, 2nd is the number of mics.\n If you want to randomize only the sources, put argument 2 to 0.')
                    
            if dist_mic_source.shape[0]>0 and dist_mic_source.shape[0]>2 : 
                raise AssertionError('1 or 2 floats are needed to constrain the mics relatively to each source: \n - if 1 argument: all mics are located on the circle of radius arg and centered on the source;\n - if 2 arg: 1st arg is the minimal radius tolerated (default: 0), 2nd is the maximal radius (default: max dim of the room).')
                
            [source_pos, mic_pos] = generate_rdn_positions(room_dim, mic_pos, source_pos, dist_mic_source, n_gen_rdn)

        elif check_options==1:
            raise AssertionError('You didn''t give instructions either for the mics positions, or for the source positions')

        
        # Calculating nb of rirs that will be calculated
        n_room = 1 if len(np.array(room_dim).shape) == 1 else np.array(room_dim).shape[0]
        n_absorption = 1 if isinstance(rt60,float) else np.array(rt60).shape[0]
        n_source = 1 if len(np.array(source_pos).shape) == 1 else np.array(source_pos).shape[0]
        n_mics = 1 if len(np.array(mic_pos).shape) == 1 else np.array(mic_pos).shape[0]
        n_rirs = n_room * n_absorption * n_mics * n_source
         
        print('%d RIRs to be created' % n_rirs)               

        Adim = ['Ax1','Ax2','Ay1','Ay2','Az1','Az2']
        config_file = os.path.join(os.path.abspath('../rirs')+'/config_file.txt')
        
        for r in range(n_room):
            ro = room_dim if n_room==1 else room_dim[r]
            
            
            absorption = None #roomsimove_single.rt60_to_absorption(ro, rt60)
            absorption = np.array([absorption])if isinstance(absorption, float) else absorption
            

            
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
                            
                        bool_begin = 0
                        
                        
                    else:
                    
                        f = open(config_file, "r+")
                        content = f.read()
                        f.seek(0) #plqcing the cursor at the beginning of the doc
                        f.truncate() # deleting what is after the cursor
                        f.write(content.replace('%.2f' % absorption[ia-1], '%.2f' % a))
                    f.close()
        
                    
                    print('Reading room configuration from a config file')
                    sim_rir = None #roomsimove_single.RoomSim.init_from_config_file(config_file)

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
            fp.write("Room dimensions (in m) ("+str(0)+"): \t%.2f\t%.2f\t%.2f\n" % (room_dim[0],room_dim[1],room_dim[2]))
        else:
            [fp.write("Room dimensions (in m) )"+str(r)+"): \t%.2f\t%.2f\t%.2f\n" % (room_dim[r][0],room_dim[r][1],room_dim[r][2])) for r in range(n_room)]
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

def generate_rdn_positions(room_dim, mic_pos, source_pos, dist_mic_source, n_gen_rdn):
    #generate_rdn_positions(rir_dict):
#    keys = rir_dict.keys()
    
    #[rnd.random()*room_dim[ii] for ii in range(3)]
    #    checkDistance = scipy.linalg.norm(np.array(source_pos)- np.array(mic_pos))
    #source_pos = source_pos/np.linalg.norm(source_pos) + mic_pos #normalise to have 1m distance between source and mic
    
    diagdiag = np.sqrt(room_dim[0]**2 + room_dim[1]**2 +room_dim[2]**2)    
    
    # Initialisation and verifications
    d_constr = len(np.array(dist_mic_source).shape)>0 # constraint on the distance is activated
    if np.array(dist_mic_source).shape[0] == 1:
        d_min = np.min((np.max((0,dist_mic_source[0] )), diagdiag))
        d_max = d_min
    elif np.array(dist_mic_source).shape[0] == 2:
        d_min = dist_mic_source[0] if d_constr else 0
        d_max = dist_mic_source[1] if d_constr else diagdiag
        d_max = np.min((d_max, diagdiag))
        if d_min > d_max :
            d_min, d_max = d_max, d_min 
    
    n_s, n_m = n_gen_rdn[0], n_gen_rdn[1] 
    if n_s == 0:
        # means that  we don't want to simulate random positions for the sources but take the source(s) whose position(s) is/are given in source_pos
        assert source_pos.shape[0]>0 #we need at least one source
        n_source =  1 if len(source_pos.shape)==1 else source_pos.shape[0]
    else:
        n_source = n_s
        # generate positions:
        source_pos = np.array([[rnd.uniform(0,room_dim[0]), rnd.uniform(0,room_dim[1]), rnd.uniform(0,room_dim[2]) ] for s in range(n_source)])

     
    if n_m == 0:
        # means that  we don't want to simulate random positions for the mics but take the mic(s) whose position(s) is/are given in mics_pos
        assert mic_pos.shape[0]>0 #we need at least one source
        n_mics = 1 if len(mic_pos.shape)==1 else mic_pos.shape[0]
    else:
        n_mics = n_m 
        # generate positions:
        mic_pos = np.array([[rnd.uniform(0,room_dim[0]), rnd.uniform(0,room_dim[1]), rnd.uniform(0,room_dim[2]) ] for m in range(n_mics)])
                    
    if d_constr and d_min==d_max:#projection of all positions on the circle of radius d_min=d_max
        if n_s == 0:
            mic_pos = [source_pos + (source_pos-mm)/scipy.linalg.norm(source_pos-mm)*d_min for mm in mic_pos]
        elif n_m == 0:
            source_pos = [mic_pos + (mic_pos-ss)/scipy.linalg.norm(ss-mic_pos)*d_min for ss in source_pos]
        else:
            all_pos = np.zeros((n_source, n_mics))
            for ss in source_pos:
                for mm in mic_pos:
                    m_s = mic_pos[mm]-source_pos[ss]
                    all_pos[ss,mm] = ss + m_s/scipy.linalg.norm(m_s)*d_min
                    
           
                
    elif d_constr and d_min<d_max:   
        # search max and min
        Max_norm_m_s = np.max([scipy.linalg.norm(mm-ss) for ss in source_pos for mm in mic_pos])
        min_norm_m_s = np.min([scipy.linalg.norm(mm-ss) for ss in source_pos for mm in mic_pos])
        
        
        if n_s == 0:
            for mind,mval in enumerate(n_mics):
                norm_m_s = scipy.linalg.norm(mval-source_pos) 
                new_norm = (norm_m_s -min_norm_m_s)/(Max_norm_m_s-min_norm_m_s)*(d_max-d_min)+d_min
                mic_pos[mind] = source_pos + (mval-source_pos)/scipy.linalg.norm(norm_m_s)*new_norm
#                if norm_m_s < d_min:
#                    mic_pos[mm] = source_pos + norm_m_s/scipy.linalg.norm(norm_m_s)*min_norm_m_s
#                elif norm_m_s > d_max:
#                    mic_pos[mm] = source_pos + norm_m_s/scipy.linalg.norm(norm_m_s)*Max_norm_m_s

        elif n_m == 0:
            for sind,sval in enumerate(source_pos):
                
                norm_m_s = scipy.linalg.norm(mic_pos-sval) 
                new_norm = (norm_m_s -min_norm_m_s)/(Max_norm_m_s-min_norm_m_s)*(d_max-d_min)+d_min
                source_pos[sind] = sval + norm_m_s/scipy.linalg.norm(norm_m_s)*new_norm
#                if norm_m_s < d_min:
#                    source_pos[ss] = ss + norm_m_s/scipy.linalg.norm(norm_m_s)*min_norm_m_s
#                elif norm_m_s > d_max:
#                    source_pos[ss] = ss + norm_m_s/scipy.linalg.norm(norm_m_s)*Max_norm_m_s


        else:
            print('todo! (or not)')
#            all_pos = np.zeros((n_source, n_mics))
#            for ss in source_pos:
#                for mm in mic_pos:
#                    norm_m_s = scipy.linalg.norm(mm-ss) 
#                    if norm_m_s < d_min:
#                    
#                    elif norm_m_s > d_max:
#                

    # check that all items are still in the room   
    if n_s == 0:      
        for j in range(3):
            if n_mics==1:
                mic_pos[j] = np.min((np.max((mic_pos[j],0)), room_dim[j]))
            else:
                for m in range(n_mics):
                    mic_pos[m][j] = np.min((np.max((mic_pos[m][j],0)), room_dim[j]))
    if n_m == 0:      
        for j in range(3):
            if n_source==1:
                source_pos[j] = np.min((np.max((source_pos[j],0)), room_dim[j]))
            else:
                for s in range(n_source):
                    source_pos[s][j] = np.min((np.max((source_pos[s][j],0)), room_dim[j]))

    return source_pos, mic_pos
        
        
