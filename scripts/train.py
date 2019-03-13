# train an ANN autoencoder model

import os
import time
import pickle
import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN
from sklearn.model_selection import train_test_split

# custom modules
from libs.updated_utils import load_dataset, create_autoencoder_model
from libs.model_utils import LossLayer
from libs.data_generator import DataGenerator
from libs.processing import white_noise, s_to_reim


def train(model_name, 
          dataset_path, sr, 
          rir_path, noise_snrs, 
          n_fft, hop_length, win_length, frag_win_length, frag_hop_length, 
          batch_size, epochs, model_path, history_path, cuda_device):
    print('[t] Training model {} on dataset {}'.format(model_name, dataset_path))
    print('[t] Training parameters: {}'.format({
        'epochs': epochs,
        'model_path': model_path,
        'history_path': history_path,
        'cuda_device': cuda_device
    }))

    # set GPU devices
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    # load dataset filenames and split in train and validation
    print('[t] Splitting data into train and validation subsets 80:20')
    filepath_list = load_dataset(dataset_path)
    filepath_list_train, filepath_list_valid = train_test_split(
        filepath_list, test_size=0.2, random_state=1337)
    
    # store DataGenerator args
    generator_args = {
        # dataset cfg
        'sr': sr,
        'cache_path': None,
        # noising/reverberation cfg
        'rir_path': rir_path,
        'noise_funcs': [None],
        'noise_snrs': noise_snrs,
        # stft cfg
        'n_fft': n_fft,
        'hop_length': hop_length,
        'win_length': win_length,
        # processing cfg
        'proc_func': s_to_reim,
        'proc_func_label': s_to_reim,
        # fragmenting cfg
        'frag_hop_length': frag_hop_length,
        'frag_win_length': frag_win_length,
        # general cfg
        'shuffle': True,
        'label_type': 'clean',
        'batch_size': batch_size,
    }
    print('[t] Data generator parameters: {}'.format(generator_args))

    # create DataGenerator objects
    training_generator = DataGenerator(filepath_list_train, **generator_args)
    validation_generator = DataGenerator(filepath_list_valid, **generator_args)
    train_steps_per_epoch = len(training_generator)
    valid_steps_per_epoch = len(validation_generator)
    print('[t] Train steps per epoch: ', train_steps_per_epoch)
    print('[t] Valid steps per epoch: ', valid_steps_per_epoch)

    # create model
    model_args = {
        'input_shape': training_generator.data_shape,
        'kernel_size': 3,
        'n_filters': 64,
    }
    print('[t] Model factory parameters: {}'.format(model_args))
    model, lossfunc = create_autoencoder_model(model_name, model_args)

    # compile model (loss function must be set in the model class)
    # TODO add metrics https://keras.io/metrics/
    model.compile(optimizer='adam', loss=lossfunc, metrics=['mse'])
    # print model summaries
    model.get_layer('encoder').summary()
    model.get_layer('decoder').summary()
    model.summary()

    # training callback functions
    Callbacks = [
        # conclude training if no improvement after N epochs
        EarlyStopping(monitor='val_loss', patience=8),
        # save model after each epoch if improved
        ModelCheckpoint(filepath=model_path,
                        monitor='val_loss',
                        save_best_only=True,
                        save_weights_only=False),
        TerminateOnNaN()
        # save logs for tensorboard
        # TODO prolly needs some fixing to include metrics?
        #TrainValTensorBoard(log_dir=logs_dir, write_graph=False)
    ]

    # train model
    print('[t] Begin training process...')
    history = model.fit_generator(
        generator=training_generator,
        validation_data=validation_generator,
        steps_per_epoch=train_steps_per_epoch,
        validation_steps=valid_steps_per_epoch,
        epochs=epochs,
        callbacks=Callbacks,
        use_multiprocessing=True,
        workers=8
        )

    # save training history
    if history_path is not None:
        print('[t] Storing training history to {}...'.format(history_path))
        df = pd.DataFrame(history.history)
        df.to_pickle(history_path)

    # end
    print('[t] Done!')
