import numpy as np

from processing import create_RIR
 
#import scipy.io

rt60 = np.linspace(0.1,1.0,10)
source_pos = [1,1,1]
mic_pos = [2,2,1]
dict_RIR = {'sampling_rate':16000, 'room_dim': [10,7,3], 'source_pos':source_pos, 'mic_pos': mic_pos ,'rt60': rt60 }

rir = create_RIR('room_sensor_config.txt') #dict_RIR, 'room_sensor_config.txt'



#for ii in range(rir.shape[1]):
#    np.save('rirs/rir'+str(ii) ,rir[:,ii])

    
