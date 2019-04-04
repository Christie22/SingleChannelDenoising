# train an ANN autoencoder model

import os
import configparser as cp
import time
import pickle
import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN, TensorBoard
from sklearn.model_selection import train_test_split

# custom modules
from libs.utilities import load_dataset, create_autoencoder_model
from libs.model_utils import LossLayer
from libs.data_generator import DataGenerator
from libs.processing import pink_noise, s_to_power


def train(model_source,
          dataset_path, sr, 
          rir_path, noise_snrs, 
          n_fft, hop_length, win_length, frag_hop_length, frag_win_length, 
          batch_size, epochs, model_path, history_path, force_cacheinit, cuda_device):
    print('[t] Training model {} at {} on dataset {}'.format(model_source, model_path, dataset_path))
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
        'noise_funcs': [pink_noise],  # TODO un-hardcode
        'noise_snrs': noise_snrs,
        # stft cfg
        'n_fft': n_fft,
        'hop_length': hop_length,
        'win_length': win_length,
        # processing cfg
        'proc_func': s_to_power,    # TODO un-hardcode
        'proc_func_label': s_to_power,    # TODO un-hardcode
        # fragmenting cfg
        'frag_hop_length': frag_hop_length,
        'frag_win_length': frag_win_length,
        # general cfg
        'shuffle': True,
        'label_type': 'clean',
        'batch_size': batch_size,
        'force_cacheinit': force_cacheinit,
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
    model, lossfunc = create_autoencoder_model(model_source, input_shape, time_slice)

    # compile model (loss function must be set in the model class)
    # TODO add metrics https://keras.io/metrics/
    model.compile(optimizer='adam', loss=lossfunc)
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
        TerminateOnNaN(),
        # save logs for tensorboard
        TensorBoard()
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
        workers=8)

    # save training history
    # TODO directly plot training history
    if history_path is not None:
        print('[t] Storing training history to {}...'.format(history_path))
        df = pd.DataFrame(history.history)
        df.to_pickle(history_path)

    # end
    print('[t] Done!')
