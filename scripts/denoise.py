# get a trained model, clean data that we noise and feed the latter into the former

import os
import numpy as np
from keras.models import load_model

from libs.data_generator import DataGenerator
from libs.utilities import load_autoencoder_model, import_spectrogram, load_data
from libs.model_utils import LossLayer
from libs.processing import white_noise

def denoise(model_path, data_path,
        sr, rir_path, n_fft, hop_length, win_length, frag_hop_length, frag_win_length,
        loss_func, batch_size, cuda_device):

    print('[t] Applying model in {} on data in {}'.format(model_path, data_path))
    print('[t] Denoising parameters: {}'.format({
        'model_path': model_path,
        'dataset_path': data_path,
        'cuda_device': cuda_device
    }))

    # set GPU devices
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    # load data filenames
    filepath_list, _ = load_data(data_path)

    # store DataGenerator args
    generator_args = {
        # dataset cfg
        'sr': sr,
        'cache_path': None,
        # noising/reverberation cfg
        'rir_path': rir_path,
        'noise_funcs': [white_noise],
        'noise_snrs': [0, 5],
        # stft cfg
        'n_fft': n_fft,
        'hop_length': hop_length,
        'win_length': win_length,
        # # processing cfg
        'proc_func': s_to_reim,
        'proc_func_label': s_to_reim,
        # fragmenting cfg
        'frag_hop_length': frag_hop_length,
        'frag_win_length': frag_win_length,
        # general cfg
        'shuffle': False,
        'label_type': 'clean',
        'batch_size': batch_size,
        'loss_func': 'mean_squared_error',
    }
    print('[t] Data generator parameters: {}'.format(generator_args))

    # create DataGenerator object
    test_data_gen = DataGenerator(filepath_list, **generator_args)

    # load trained model
    trained_model = load_model(model_path, {
        'LossLayer': LossLayer
    })

    ## OR: create model
    # model_name =
    # model_args = {
    #     'input_shape': test_data_gen.data_shape,
    #     'kernel_size': 3,
    #     'n_filters': 64,
    # }
    # print('[t] Model factory parameters: {}'.format(model_args))
    # trained_model, lossfunc = create_autoencoder_model(model_name, model_args)

    # compile model (loss function must be set in the model class
    trained_model.compile(optimizer='adam', loss=loss_func, metrics=['mse'])

    # print model summary
    trained_model.summary()

    # prediction on test data
    print('[t] Begin testing process...')
    cleaned_data_pred, _ = trained_model.predict(test_data_gen)

    np.save('cleaned_data_pred', cleaned_data_pred)
    print('Done! Cleaned data stored at:' + data_path)
    print('cleaned_data_pred')
