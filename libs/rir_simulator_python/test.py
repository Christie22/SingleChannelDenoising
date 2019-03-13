import numpy as np

from processing import create_RIR
 
#import scipy.io

rt60 = np.linspace(0.1,1.0,10)
source_pos = [1,1,1]
mic_pos = [2,2,1]
room_dim = [10,7,3]
dict_RIR = {'sampling_rate':16000, 'room_dim': room_dim, 'source_pos':source_pos, 'mic_pos': mic_pos ,'rt60': rt60 }

rir = create_RIR(dict_RIR) #dict_RIR, 'room_sensor_config.txt'



f = open('RIR_params.txt','w+')
f.write('sampling_rate: \t%d\n' % 16000)
f.write("room_dim: "+str(1)+"\t%.2f\t%.2f\t%.2f\n" % (room_dim[0],room_dim[1],room_dim[2]))
f.write('rt60: %.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t\n' % tuple(list(rt60)))
f.write("mic_pos: "+str(1)+"\t%.2f\t%.2f\t%.2f\n" % (mic_pos[0],mic_pos[1],mic_pos[2]))
f.write("source_pos: "+str(1)+"\t%.2f\t%.2f\t%.2f\n" % (source_pos[0],source_pos[1],source_pos[2]))

f.close()