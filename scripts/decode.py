import os
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from scipy.stats import norm
from libs.model_utils import LossLayer
from libs.utilities import load_autoencoder_model, unprocess_data, plot_spectrogram

def main(args):
    # set GPU device(s)
    os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda_device']

    # load decoder
    _, decoder, _ = load_autoencoder_model(
        args['model_path'], {
        'LossLayer': LossLayer
    })

    # generate spectrogram
    print('Latent values path: {}'.format(args['latent_vals_path']))
    z_sample = np.load(args['latent_vals_path'])
    print(z_sample.shape)
    #generate manifold
    #plot_manifold(args['mani_size'], z_sample, decoder)

    """
    z_sample = np.array([
        args['latent_dim_vals']
    ])
    """
    x_decoded = decoder.predict(z_sample)
    np.save(args['spec_path'], x_decoded)

    print('Done! Check content of {}'.format(args['spec_path']))

    x_decoded = unprocess_data(x_decoded[0])
    plot_spectrogram(x_decoded,'generated_spectogram')




# run the thing
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate spectogram from 2d latent space')

    # add parameters
    #parser.add_argument('latent_dim_vals', type=float, nargs='+', help='Latent dimension values')
    parser.add_argument('-m', '-model_path', '--model_path', type=str, help='model path')
    parser.add_argument('-g', '-cuda_device', '--cuda_device', type=str, help='number of training epochs')
    parser.add_argument('-p', '-path', '--spec_path', type=str, help='npy output file with re/im spectrogram')
    parser.add_argument('-r', '-rows', '--n_rows', type=int, help='Number of rows in spectrogram')
    parser.add_argument('-c', '-cols', '--n_cols', type=int, help='Number of columns in spectrogram')
    parser.add_argument('-ch', '-chans', '--n_channels', type=int, help='Number of channels in spectrogram')
    parser.add_argument('-b', '-batch_size', '--batch_size', type=int, help='Batch size')
    parser.add_argument('-ms', '-mani_size', '--mani_size', type=int, help = 'Dimension of manifold across each axis')
    parser.add_argument('-l', '-latent_vals_path', '--latent_vals_path', type=str, help = 'Latent vals path')

    # read defaults from params.py
    from params import params
    parser.set_defaults(
        model_path = params['model_path'],
        cuda_device = params['cuda_device'],
        spec_path = 'outputs/generated_spectrogram.npy',
        n_rows = params['n_rows'],
        n_cols = params['n_cols'],
        n_channels = params['n_channels'],
        batch_size = params['batch_size'],
        mani_size = params['mani_size'],
        latent_vals_path = params['latent_vals_path'])

    # run
    args = parser.parse_args()
    main(vars(args))
