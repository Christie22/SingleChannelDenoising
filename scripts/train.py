# train a model using settings in `params.py` for basic denoising AE
# not tested yet

# internal and external modules
import os
import time
import pickle
import numpy as np
import pandas as pd
from keras import backend as K
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN
from sklearn.model_selection import train_test_split
#from keras.datasets import mnist

# 
from libs.utilities import load_autoencoder_model, load_data, load_dataset
from libs.model_utils import LossLayer
from libs.data_generator import DataGenerator


def main(args):
    if(args['dataset_path'][-3:] == 'csv'):
        # set GPU device(s)
        dataset_params ={
            'dim': [args['n_rows'], args['n_cols']],
            'data_path': args['data_path'],
            'n_channels': args['n_channels'],
            'batch_size': args['batch_size'],
            'pre_processing': args['pre_processing'],
            'shuffle': True,
            'labels': None,
            'n_classes': None,
            'y_value': None,
        }
    os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda_device']

    if(args['dataset_path'][-3:] == 'csv'):
        print('Splitting data into train, validation and test subsets 80:20:20')
        dataset_df = load_dataset(args)
        dataset_df_train, dataset_df_valid = train_test_split(
            dataset_df, test_size=0.2, random_state=1337)

        filenames_train = dataset_df_train['filepath'].values
        filenames_valid = dataset_df_valid['filepath'].values
        train_steps_per_epoch = int(len(filenames_train) / args['batch_size'])
        valid_steps_per_epoch = int(len(filenames_valid) / args['batch_size'])
        print('train steps per epoch: ', train_steps_per_epoch)
        print('valid steps per epoch: ', valid_steps_per_epoch)
        training_generator = DataGenerator(filenames_train, **dataset_params)
        validation_generator = DataGenerator(filenames_valid, **dataset_params)

    else:
        # IMPORT DATA
        (train_X, train_Y) = load_data(args['dataset_path'])
        print('Training data shape : ', train_X.shape, train_Y.shape)

        train_X, valid_X = train_test_split(
            train_X, test_size=0.2, random_state=12345)

    # CREATE MODEL
    model = create_model({
        'LossLayer': LossLayer
    }, args)

    # compile model (loss function must be set in the model class)
    # TODO add metrics https://keras.io/metrics/
    model.compile(optimizer='adam', loss=None, metrics=['mse'])
    # print model summaries
    # TODO is the structure with layers still relevant?
    model.get_layer('encoder').summary()
    model.get_layer('decoder').summary()
    model.summary()

    # TRAIN MODEL
    # training callback functions
    Callbacks = [
        # conclude training if no improvement after N epochs
        EarlyStopping(monitor='val_loss', patience=8),
        # save model after each epoch if improved
        ModelCheckpoint(filepath=args['model_path'],
                        monitor='val_loss',
                        save_best_only=True,
                        save_weights_only=False),
        TerminateOnNaN()
        # save logs for tensorboard
        # TODO prolly needs some fixing to include metrics?
        #TrainValTensorBoard(log_dir=logs_dir, write_graph=False)
    ]

    # train the model!
    if(args['dataset_path'][-3:] == 'csv'):
        print('Using real data')
        history = model.fit_generator(
            generator=training_generator,
            validation_data=validation_generator,
            steps_per_epoch=train_steps_per_epoch,
            validation_steps=valid_steps_per_epoch,
            epochs=args['epochs'],
            callbacks=Callbacks,
            use_multiprocessing=True,
            workers=8
            )
    else:
        history = model.fit(
            train_X,
            None,
            batch_size=args['batch_size'],
            epochs=args['epochs'],
            callbacks=Callbacks,
            validation_data=(valid_X, None)
        )

    # save training history
    df = pd.DataFrame(history.history)
    df.to_pickle(args['history_path'])

    # byeeee
    print('Done! What to do next:')
    print('- Run `python results.py` to show training history and data points in latent space')
    print('- Run `python generate.py` to generate a latent space manifold with novel outputs')

# create the entire model (encoder + decoder)
def create_model(custom_objects, args):
    # import model
    model_name = args['model_name']
    if model_name == 'conv_dilation_leaky':
        print('Using model `{}` from {}'.format(model_name, 'model_conv_dilation_leaky'))
#        import model_conv_dilation_leaky as m
    elif model_name == 'conv_dilation':
        print('Using model `{}` from {}'.format(model_name, 'model_conv_dilation'))
#        import model_conv_dilation as m
    elif model_name == 'dense_leaky':
        print('Using model `{}` from {}'.format(model_name, 'model_dense_leaky'))
#        import model_dense_leaky as m
    elif model_name == 'dense':
        print('Using model `{}` from {}'.format(model_name, 'model_dense'))
#        import model_dense as m
    elif model_name == 'hsu_glass':
        print('Using model `{}` from {}'.format(model_name, 'model_hsu_glass'))
#        import model_hsu_glass as m
    elif model_name == 'vanilla_resnet':
        print('Using model `{}` from {}'.format(model_name, 'model_vanilla_resnet'))
#        import model_vanilla_resnet as m
    else:
        print('importing example model :D')
        import models.model_example as m

    # calc input shape and enforce it
    input_shape = (args['n_rows'], args['n_cols'], args['n_channels'])
    K.set_image_data_format('channels_last')
    # generate model
    obj = m.AEModelFactory(
        input_shape=input_shape,
        kernel_size=args['kernel_size'],
        n_filters=args['n_filters'],
        n_intermediate_dim=args['n_intermediate_dim'],
        n_latent_dim=args['n_latent_dim'])
    model = obj.get_model()
    return model

# run the thing
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Trains a denoising AE with given parameters')

    # add parameters
    parser.add_argument('dataset_path', type=str, help='dataset path')

    parser.add_argument('-r', '-rows', '--n_rows', type=int, help='number of rows')
    parser.add_argument('-c', '-cols', '--n_cols', type=int, help='number of cols')
    parser.add_argument('-n', '-channels', '--n_channels', type=int, help='number of channels')

    parser.add_argument('-m', '-model_name', '--model_name', type=str, help='model names')
    parser.add_argument('-m', '-model_path', '--model_path', type=str, help='model path')
    parser.add_argument('-e', '-epochs', '--epochs', type=int, help='gpu cuda device')
    parser.add_argument('-g', '-cuda_device', '--cuda_device', type=str, help='number of training epochs')
    parser.add_argument('-b', '-batch_size', '--batch_size', type=int, help='size of training batch')
    parser.add_argument('-hp', '-history_path', '--history_path', type=str, help='history path')

    parser.add_argument('-k', '-kernel_size', '--kernel_size', type=int, help='convolutional kernel size')
    parser.add_argument('-f', '-filters', '--n_filters', type=int, help='number of convolutional filters')
    parser.add_argument('-i', '-id', '-intermediate_dim', '--n_intermediate_dim', type=int, help='number of intermediate dense layer dimensions')
    parser.add_argument('-l', '-ld', '-latent_dim', '--n_latent_dim', type=int, help='number of latent dimensions')

    # read defaults from params.py
    from params import params
    parser.set_defaults(
        n_rows = params['n_rows'],
        n_cols = params['n_cols'],
        n_channels = params['n_channels'],

        model_name = params['model_name'],
        model_path = params['model_path'],
        epochs = params['epochs'],
        cuda_device = params['cuda_device'],
        batch_size = params['batch_size'],
        history_path = params['history_path'],

        kernel_size = params['kernel_size'],
        n_filters = params['n_filters'],
        n_intermediate_dim = params['n_intermediate_dim'],
        n_latent_dim = params['n_latent_dim']
    )

    # run
    args = parser.parse_args()
    main(vars(args))

