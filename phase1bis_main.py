import os.path as osp
import itertools
import keras.backend as K
import numpy as np

from scripts.train_fullparam import train
from libs.utilities import get_func_name
from libs.processing import s_to_exp, s_to_reim, s_to_db


# default arguments
defaults = {
    'model_source_no': 'models/conv3_small.jsont',
    'model_source_norm': 'models/conv3_small_norm.jsont',
    'dataset_path': '/data/riccardo_datasets/npr_news/ds0/train',
    'sr': 16000,
    'rir_path': None,
    'noise_snrs': [15],
    'n_fft': 512,
    'hop_length': 128,
    'win_length': 512,
    'frag_hop_length': 30,
    'frag_win_length': 32,
    'time_slice': None,
    'batch_size': 128,
    'epochs': 250,
    'normalize': False,
    'model_destination_base': '/data/riccardo_models/denoising/phase1/',
    'logs_path': './train_logs/phase1.json',
    'cuda_device': '0',
}

normalize = defaults['normalize']
frag_win_length = defaults['frag_win_length']
# construct iterator based on tweakable args
proc_func_args = [
    s_to_exp(1.0),
    s_to_exp(1.0/6),
    s_to_reim
]
p_time_slice_args = [0, 0.1, 0.5, 1]
time_slice_args = [slice(np.int(frag_win_length//2-p/2*frag_win_length),\
 np.int(frag_win_length//2+p/2*frag_win_length)) if p!=0 \
else slice(np.int(frag_win_length//2), np.int(frag_win_length//2)+1) \
for p in p_time_slice_args ]

print('[coucou], time_slice_args:{}'.format(time_slice_args[0]))

# run the thing
if __name__ == "__main__":
    for i, time_slice in enumerate(time_slice_args):
        for j, proc_func in enumerate(proc_func_args):
            print('#### PHASE 1-BIS TRAINING - {}/{}'.format(i+1, len(proc_func_args) * np.int(p_time_slice_args[i]*frag_win_length) if p_time_slice_args[i] != 0 else 1))
    
            # construct remaining arguments
            model_destination_name = 'phase1bis_{}_{}.h5'.format(
                get_func_name(proc_func), 
                np.int(p_time_slice_args[i]*frag_win_length) if p_time_slice_args[i] != 0 else 1
                
            )
            model_destination = osp.join(defaults['model_destination_base'], model_destination_name)
            model_source = defaults['model_source_norm'] if proc_func ==s_to_reim \
            else defaults['model_source_no']
    
            K.clear_session()
            train(
                model_source,
                defaults['dataset_path'],
                defaults['sr'],
                defaults['rir_path'],
                defaults['noise_snrs'],
                normalize, 
                proc_func,
                proc_func,
                defaults['n_fft'],
                defaults['hop_length'],
                defaults['win_length'],
                defaults['frag_hop_length'],
                defaults['frag_win_length'],
                time_slice,
                defaults['batch_size'],
                defaults['epochs'],
                model_destination,
                defaults['logs_path'],
                False,
                defaults['cuda_device'])
