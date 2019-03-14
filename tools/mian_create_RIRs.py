import numpy as np

from processing import create_RIR


rt60 = np.linspace(0.1,0.9,9)
source_pos = np.array([[1,1,1], [5, 3.5, 1.5]])
mic_pos = np.array([ [2,1,1], [8,5,1] ])
room_dim = np.array([10,7,3])

N_items = 10
options = {'cst_dist_mic_source':False, 'gen_rdn_sources_or_mics': N_items }

dict_RIR = {'sampling_rate':16000, 'room_dim': room_dim, 'source_pos':source_pos, 'mic_pos': mic_pos ,'rt60': rt60}# , 'options':options}

rir = create_RIR(dict_RIR)

