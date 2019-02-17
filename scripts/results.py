import os
import pandas as pd
import numpy as np
import pickle
import itertools
import matplotlib.pyplot as plt
from keras.models import load_model

from libs.data_generator import DataGenerator
from libs.utilities import load_autoencoder_model, load_data, load_all_as_test_data
from libs.model_utils import LossLayer


def main(args):
    # set GPU device(s)
    os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda_device']
    # plot training history
    try:
        # load training history
        history_df = pd.read_pickle(args['history_path'])

        if(len(args['latent_space_paths']) != len(args['label_types'])):
            raise TypeError('Latent space path and label_types have to have the same size.')

        print('plotting trainig history...')
        ax_hist = history_df.plot(logy=True, title='History', grid=True)
        fig_hist = ax_hist.get_figure()
        fig_hist.savefig(args['history_plot_path'])
    except Exception as e:
        print('Exception while plotting training')
        print(e)

    # load data
    if(args['dataset_path'][-3:] == 'csv'):
        print('Loading real dataset from {}...'.format(args['dataset_path']))
        test_df = load_all_as_test_data(args)
        train_data = test_df['filepath'].values
        print(test_df.head())
        df_labels = test_df.drop(columns='filepath')
        dataset_params ={
            'dim': [args['n_rows'], args['n_cols']],
            'data_path': args['data_path'],
            'n_channels': args['n_channels'],
            'batch_size': args['batch_size'],
            'pre_processing': args['pre_processing'],
            'shuffle': False,
            'labels': None,
            'n_classes': None,
        }
        print(dataset_params)
        testing_generator = DataGenerator(train_data, **dataset_params)
        test_steps_per_epoch = int(len(train_data) / args['batch_size'])
    else:
        datasetPath = args['dataset_path']
        print('loading dataset from {}...'.format(datasetPath))
        train_data, train_labels = load_data(datasetPath)
        df_labels = pd.DataFrame(list(train_labels))

    # load encoder
    print('loading encoder from {}...'.format(args['model_path']))
    encoder, _, _ = load_autoencoder_model(
        args['model_path'], {
            'LossLayer': LossLayer
        })

    # run predictions
    if(args['dataset_path'][-3:] == 'csv'):
        encoded_train_data, _ = encoder.predict_generator(
            generator=testing_generator,
            steps=test_steps_per_epoch,
            #use_multiprocessing=True,
            #workers=8
        )
    else:
        encoded_train_data, _ = encoder.predict(train_data)

    for plot_path, label in zip(args['latent_space_paths'], args['label_types']):
        # choose label
        print('choosing label...')
        df_labels = df_labels.head(encoded_train_data.shape[0])
        labels = df_labels[label].values

        # plot data
        fig_latent = plot_latent_space(encoded_train_data, labels, args['n_latent_dim'], label, 4)
        fig_latent.savefig(plot_path)
        print("{} saved".format(plot_path))

    print('Done! Check content of {} and {}'.format(args['latent_space_paths'], args['history_plot_path']))


# create latent space plot
def plot_latent_space(encoded_train_data, train_labels, n_latent_dim, label_type, ax_rows = 3):
    data_frame = pd.DataFrame(encoded_train_data)

    ax_pairs = list(itertools.combinations(range(n_latent_dim), 2))
    ax_cols = int(np.ceil(len(ax_pairs) / ax_rows))
    fig, ax_latent = plt.subplots(
        nrows=ax_rows,
        ncols=ax_cols,
        figsize=(ax_cols*8, ax_rows*7),
        tight_layout=False,
        squeeze=False
        )
    title = 'Latent Space, Color: ' + label_type
    fig.suptitle(title, fontsize=20, fontweight='heavy')

    print('plotting {} dimension combinations into a subplot of size {}...'.format(len(ax_pairs), ax_latent.shape))

    ax_i = 0
    for r in range(ax_rows):
        for c in range(ax_cols):
            pair = ax_pairs[ax_i]
            ax = data_frame.plot.scatter(
                pair[0],
                pair[1],
                c=train_labels,
                colormap='viridis',
                colorbar=True,
                alpha=0.5,
                ax=ax_latent[r][c])
            xlabel = 'Latent Dimension {}'.format(pair[0])
            ylabel = 'Latent Dimension {}'.format(pair[1])
            ax.set_xlabel(xlabel, fontsize='large', fontweight='heavy')
            ax.set_ylabel(ylabel, fontsize='large', fontweight='heavy')
            ax_i += 1
            if ax_i >= len(ax_pairs):
                fig.tight_layout(rect=[0, 0.03, 1, 0.97])
                return fig
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig


# run the thing
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot training history and dataset in latent space')

    # add parameters
    parser.add_argument('-m', '-model_path', '--model_path', type=str, help='model path')
    parser.add_argument('-g', '-cuda_device', '--cuda_device', type=str, help='number of training epochs')
    parser.add_argument('-l', '-ld', '-latent_dim', '--n_latent_dim', type=int, help='number of latent dimensions')
    parser.add_argument('-t', '-label_types', '--label_types', type=str, help='model path')
    parser.add_argument('-lp', '-latent_space_paths', '--latent_space_paths', type=str, help='model path')
    parser.add_argument('-hp', '-history_path', '--history_path', type=str, help='history data path')
    parser.add_argument('-hpp', '-history_plot_path', '--history_plot_path', type=str, help='history plot path')
    parser.add_argument('dataset_path', type=str, help='datset path')

    # read defaults from params.py
    from params import params
    parser.set_defaults(
        model_path = params['model_path'],
        cuda_device=params['cuda_device'],
        n_latent_dim = params['n_latent_dim'],
        label_types = params['label_types'],
        latent_space_paths = params['latent_space_paths'],
        history_path = params['history_path'],
        history_plot_path = params['history_plot_path'],
        dataset_path = params['dataset_path'],
    )

    # run
    args = parser.parse_args()
    main(vars(args))

