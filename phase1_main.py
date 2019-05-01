import os.path as osp
import itertools

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
    'frag_hop_length': 16,
    'frag_win_length': 32,
    'batch_size': 128,
    'epochs': 1,
    'model_destination_base': '/data/riccardo_models/denoising/phase1/',
    'logs_path': './train_logs/phase1.json',
    'cuda_device': '0'
}

# construct iterator based on tweakable args
normalize_args = [True, False]
proc_func_args = [
    s_to_exp(1.0),
    s_to_exp(2.0),
    s_to_exp(1.0/6),
    s_to_reim,
    s_to_db,
]
args_list = itertools.product(normalize_args, proc_func_args)


# run the thing
if __name__ == "__main__":
    for i, args in enumerate(args_list):
        print('#### PHASE 1 TRAINING - {}/{}'.format(i, len(args_list)))

        # construct remaining arguments
        normalize, proc_func = args
        model_destination_name = '{}_{}.h5'.format(
            get_func_name(proc_func),
            'norm' if normalize else 'no'
        )
        model_destination = osp.join(defaults['model_destination_base'], model_destination_name)

        train(
            defaults['model_source_norm'] if normalize else defaults['model_source_no'],
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
            defaults['batch_size'],
            defaults['epochs'],
            model_destination,
            defaults['logs_path'],
            defaults['force_cacheinit'],
            defaults['cuda_device'])
