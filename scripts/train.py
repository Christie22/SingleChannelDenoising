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


def train(model_name, dataset_path, rows, cols, channels, epochs, batch_size, model_path, history_path, cuda_device):
    # set GPU devices
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    
    # store DataGenerator args
    generator_args ={
        'dim': [rows, cols],
        'dataset_path': dataset_path,
        'channels': channels,
        'batch_size': batch_size,
        # NOTE insert other DataGenerator args
    }

    # load dataset filenames and split in train and validation
    print('Splitting data into train, validation and test subsets 80:20:20')
    dataset_df = load_dataset(dataset_path)
    dataset_df_train, dataset_df_valid = train_test_split(
        dataset_df, test_size=0.2, random_state=1337)

    # create DataGenerator objects
    filenames_train = dataset_df_train['filepath'].values
    filenames_valid = dataset_df_valid['filepath'].values
    train_steps_per_epoch = int(len(filenames_train) / batch_size)
    valid_steps_per_epoch = int(len(filenames_valid) / batch_size)
    print('train steps per epoch: ', train_steps_per_epoch)
    print('valid steps per epoch: ', valid_steps_per_epoch)
    training_generator = DataGenerator(filenames_train, **generator_args)
    validation_generator = DataGenerator(filenames_valid, **generator_args)

    # create model
    input_shape = (rows, cols, channels)
    model = create_autoencoder_model({
        'LossLayer': LossLayer
    }, model_name, input_shape)

    # compile model (loss function must be set in the model class)
    # TODO add metrics https://keras.io/metrics/
    model.compile(optimizer='adam', loss=None, metrics=['mse'])
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
    print('Training model...')
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
    df = pd.DataFrame(history.history)
    df.to_pickle(history_path)

    # end
    print('Done! Training history stored at {}'.format(history_path))
