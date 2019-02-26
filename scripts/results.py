# test and evaluate trained model

import os
import pandas as pd
import numpy as np
import pickle
import itertools
import matplotlib.pyplot as plt
from keras.models import load_model

# custom modules
from libs.updated_utils import load_autoencoder_model
from libs.model_utils import LossLayer
from libs.data_generator import DataGenerator


def results(model_path, dataset_path, rows, cols, channels, epochs, batch_size, history_path, cuda_device):
    # set GPU devices
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    # plot training history
    try:
        # load training history
        history_df = pd.read_pickle(history_path)

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
    print('Loading real dataset from {}...'.format(args['dataset_path']))
    test_df = load_all_as_test_data(args)
    train_data = test_df['filepath'].values
    print(test_df.head())
    df_labels = test_df.drop(columns='filepath')
    generator_args ={
        'dim': [rows, cols],
        'dataset_path': dataset_path,
        'channels': channels,
        'batch_size': batch_size,
        # NOTE insert other DataGenerator args
    }
    testing_generator = DataGenerator(train_data, **generator_args)
    test_steps_per_epoch = int(len(train_data) / args['batch_size'])

    # load encoder
    print('loading encoder from {}...'.format(args['model_path']))
    encoder, _, _ = load_autoencoder_model(
        args['model_path'], {
            'LossLayer': LossLayer
        })

    # run predictions
    encoded_train_data, _ = encoder.predict_generator(
        generator=testing_generator,
        steps=test_steps_per_epoch,
        #use_multiprocessing=True,
        #workers=8
    )

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
