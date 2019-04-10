# train an ANN autoencoder model

import os
import os.path as osp
import time
import pickle
import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN, TensorBoard, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# custom modules
from libs.utilities import load_dataset, create_autoencoder_model, load_autoencoder_model, load_autoencoder_lossfunc
from libs.model_utils import LossLayer
from libs.data_generator import DataGenerator
from libs.processing import pink_noise, s_to_power


def train(model_source, dataset_path, 
          sr, rir_path, noise_snrs, 
          n_fft, hop_length, win_length, frag_hop_length, frag_win_length, 
          batch_size, epochs, model_destination, force_cacheinit, cuda_device):
    print('[t] Training model on dataset {}'.format(dataset_path))
    print('[t] Training parameters: {}'.format({
        'epochs': epochs,
        'model_source': model_source,
        'model_destination': model_destination,
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
        'normalize': 'batch',
        'label_type': 'clean',
        'normalize': 'local',
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
    
    # loss function: data slice under consideration
    #time_slice = frag_win_length // 2
    input_shape = training_generator.data_shape
    model_template_args = {
        'time': input_shape[1],
        'channels': input_shape[2]
    }
    time_slice = slice(None)

    # set initial epoch to its most obvious value
    initial_epoch = 0

    # extract extension from model source file (.h5 or .json)
    model_source_ext = osp.splitext(model_source)[1]

    # if model source is a pre-trained model, load and resume training
    print('[t] Loading model source from {}...'.format(model_source))
    if model_source_ext == '.h5':
        print('[t] Model source is a pre-trained model!')
        # load stuff
        lossfunc = load_autoencoder_lossfunc(time_slice)
        _, _, model = load_autoencoder_model(model_source, {'lossfunc': lossfunc})
        # figure out number of already-trained epochs
        initial_epoch = int(osp.splitext(
            osp.basename(model_source))[0].split('_e')[-1])

    # if model source is a config file, create model
    elif model_source_ext in ['.json', '.jsont']:
        print('[t] Model source is a configuration file!')
        # create stuff
        model, lossfunc = create_autoencoder_model(
            model_source, input_shape, model_template_args, time_slice=time_slice)
    
    # if model source isn't either, well, *shrugs*
    else:
        print('[t] Model source can\'t be recognized: {}'.format(model_source))
        return

    # use separate training epochs index to get correct trained model filename 
    # and tensorboard (see keras docs on fit_generator)
    max_epochs = initial_epoch + epochs

    # compile model (loss function must be set in the model class)
    model.compile(optimizer='adam', loss=lossfunc)
    # print model summaries
    #model.get_layer('encoder').summary()
    #model.get_layer('decoder').summary()
    model.summary()

    # training callback functions
    Callbacks = [
        # conclude training if no improvement after N epochs
        EarlyStopping(monitor='loss', patience=8),
        # save model after each epoch if improved
        ModelCheckpoint(filepath=model_destination,
                        monitor='loss',
                        save_best_only=True,
                        save_weights_only=False,
                        verbose=1),
        TerminateOnNaN(),
        # save logs for tensorboard
        TensorBoard(
            log_dir='logs/model_{}/'.format(model.name),
            #histogram_freq=5,
            batch_size=batch_size,
            write_graph=True,
            write_grads=True,
            write_images=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0,
            verbose=1)
    ]

    # train model
    print('[t] Begin training process...')
    model.fit_generator(
        generator=training_generator,
        validation_data=validation_generator,
        steps_per_epoch=train_steps_per_epoch,
        validation_steps=valid_steps_per_epoch,
        initial_epoch=initial_epoch,
        epochs=max_epochs,
        callbacks=Callbacks,
        use_multiprocessing=True,
        workers=8)

    # end
    print('[t] Done! Trained models are stored somewheres in {}'.format(model_destination))
