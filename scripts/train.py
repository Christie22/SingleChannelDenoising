# train a model using settings in `params.py` for basic denoising AE
# TODO:
# - test
# - remove args
# - create_model into utils


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

# custom modules
from libs.utilities import load_data, load_dataset
from libs.model_utils import LossLayer
from libs.data_generator import DataGenerator


def train(model_name, dataset_path, rows, cols, channels, epochs, batch_size, model_path, history_path, cuda_device):
    # set GPU device(s)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    
    # store DataGenerator args
    dataset_params ={
        'dim': [rows, cols],
        'dataset_path': dataset_path,
        'channels': channels,
        'batch_size': batch_size,
        # NOTE insert other DataGenerator args
    }

    # split dataset in train and validation
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
    training_generator = DataGenerator(filenames_train, **dataset_params)
    validation_generator = DataGenerator(filenames_valid, **dataset_params)

    # create model
    input_shape = (rows, cols, channels)
    model = create_model({
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

    # byeeee
    print('Done! Training history stored at {}'.format(history_path))

# create the entire model (encoder + decoder)
def create_model(custom_objects, model_name, input_shape):
    # import model
    model_name = model_name
    if model_name == 'lstm':
        print('Using model `{}` from {}'.format(model_name, 'model_lstm'))
    elif model_name == 'conv':
        print('Using model `{}` from {}'.format(model_name, 'model_conv'))
    else:
        print('importing example model :D')
        import models.model_example as m

    # calc input shape and enforce it
    K.set_image_data_format('channels_last')
    # generate model
    obj = m.AEModelFactory(input_shape=input_shape)
    model = obj.get_model()
    return model

