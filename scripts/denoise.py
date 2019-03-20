# get a trained model, clean data that we noise and feed the latter into the former

import os
import numpy as np
from keras.models import load_model

from libs.data_generator import DataGenerator
from libs.utilities import load_dataset, load_autoencoder_lossfunc, load_autoencoder_model
from libs.model_utils import LossLayer
from libs.processing import white_noise, s_to_reim

def denoise(model_name, model_path, data_path,
        sr, n_fft, hop_length, win_length, frag_hop_length, frag_win_length, 
        batch_size, cuda_device):

    print('[t] Applying model in {}  at {} on data in {}'.format(model_name, model_path, data_path))
    print('[t] Denoising parameters: {}'.format({
        'model_name': model_name,
        'model_path': model_path,
        'data_path': data_path,
        'cuda_device': cuda_device
    }))

    # set GPU devices
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    # load data in file data_path
    data = None # load_data(data_path)


    # create DataGenerator object

    # load trained model
    print('[d] Loading model from {}...'.format(model_path))
    lossfunc = load_autoencoder_lossfunc(model_name)
    _, _, model = load_autoencoder_model(model_path, {'lossfunc': lossfunc})

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
    
    # print model summary
    model.summary()

    # prediction on test data
    print('[t] Begin testing process...')
    cleaned_data_pred, _ = model.predict(data)

    np.save('cleaned_data_pred', cleaned_data_pred)
    print('Done! Cleaned data stored at:' + data_path)
    print('cleaned_data_pred')
