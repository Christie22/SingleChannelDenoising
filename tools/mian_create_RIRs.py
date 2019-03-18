import numpy as np

from processing import create_RIR


rt60 = 0.1#np.linspace(0.1,0.9,9)
source_pos = np.array([1,1,1])
mic_pos = np.array([1,2,1])
room_dim = np.array([10,7,3])

N_sources, N_mics = 10, 0 # at the moment, put one of these coeff at 0
dist_min, dist_max = 1, 1
#[dist_min, dist_max]
options = {'dist_mic_source' : 1, 'gen_rdn_sources_and_mics' : [N_sources, N_mics] }

dict_RIR = {'sampling_rate':16000, 'room_dim': room_dim, 'source_pos':source_pos, 'mic_pos': mic_pos ,'rt60': rt60, 'options':options}

create_RIR(dict_RIR)

