import os

import pandas as pd
import numpy as np
import matplotlib as plt
import pickle
from keras.models import load_model
from libs.model_utils import LossLayer
from libs.data_generator import DataGenerator
from libs.utilities import load_autoencoder_model, import_spectrogram


def encode(args):
    # set GPU device(s)
    os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda_device']

    # load encoder
    encoder, _, _ = load_autoencoder_model(
        args['model_path'], {
        'LossLayer': LossLayer
    })

    # run predictions
    if(args['dataset_path'][-3:] == 'csv'):
        dataset_params ={
            'dim': [args['n_rows'], args['n_cols']],
            'data_path': args['data_path'],
            'n_channels': args['n_channels'],
            'batch_size': args['batch_size'],
            'pre_processing': args['pre_processing'],
            'shuffle': False,
            'labels': None,
            'n_classes': None,
            'y_value': None,
        }
        filepaths = pd.read_csv(args['spec_path'])
        filepaths = filepaths['filepath'].values
        steps_per_epoch = int(len(filepaths) / args['batch_size'])
        s = DataGenerator(filepaths, **dataset_params)
        s_latent_space,_ = encoder.predict_generator(
            generator=s
            , steps=steps_per_epoch
            #, workers=8
            #, use_multiprocessing=True
        )
    else:
        s = import_spectrogram(args['spec_path'])
        print('specs shape: ', s.shape)
        s_latent_space,_ = encoder.predict(s)

    np.save(args['latent_vals_path'], s_latent_space)
    print('Done! Latent space values stored at:')
    print(args['latent_vals_path'])


# run the thing
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Get latent space from a given spectrogram')

    parser.add_argument('spec_path', type=str, help='npy file containing re/im spectrogram')

    parser.add_argument('-m', '-model_path', '--model_path', type=str, help='model path')
    parser.add_argument('-g', '-cuda_device', '--cuda_device', type=str, help='cuda device')
    parser.add_argument('-l', '-latent_vals_path', '--latent_vals_path', type=str, help='Latent vals path')

    from params import params
    parser.set_defaults(
        model_path = params['model_path'],
        cuda_device = params['cuda_device'],
	latent_vals_path = params['latent_vals_path'],
    )

    args = parser.parse_args()
    main(vars(args))
