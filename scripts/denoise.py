# get a trained model, clean one piece of data (or frag) that we noise and feed the latter into the former

import os
import numpy as np
from keras.models import load_model
import librosa

from libs.data_generator import DataGenerator
from libs.utilities import load_autoencoder_model, import_spectrogram, load_data
from libs.model_utils import LossLayer
from libs.processing import white_noise

def denoise(model_name, model_path, file_path,
        sr, rir_path, n_fft, hop_length, win_length, frag_hop_length, frag_win_length,
        loss_func, batch_size,
        cuda_device):
    # assume that the input is noised for instance, and call another file if it is not the case: go find them in load_cache
    # just need to fragment
    print('[t] Applying model in {} on data in {}'.format(model_path, file_path))
    print('[t] Denoising parameters: {}'.format({
        'model_path': model_path,
        'file_path': file_path,
        'cuda_device': cuda_device
    }))

    # set GPU devices
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    # load data filenames
    x = librosa.core.load(file_path, sr=sr)

    # convert to TF-domain
    s = lr.core.stft(x, n_fft= n_fft, hop_length= hop_length, win_length= win_length)
#    s = reim_to_s(reim)

    # if one path, not necessary to call the data generator: but see DataGenrator for  process
    # create DataGenerator object
    #test_file_gen = DataGenerator(filepath_list, **generator_args)

    # load model
    print('[r] Loading model from {}...'.format(model_path))
    lossfunc = load_autoencoder_lossfunc(model_name)
    _, _, model = load_autoencoder_model(model_path, {'lossfunc': lossfunc})

    # compile model (loss function must be set in the model class
    trained_model.compile(optimizer='adam', loss=loss_func, metrics=['mse'])

    # print model summary
    trained_model.summary()

    # prediction on test data
    print('[t] Begin testing process...')
    cleaned_data_pred, _ = trained_model.predict(test_data_gen)
    # very slow at the beginning then very fast (real-time possible)
    np.save('cleaned_data_pred', cleaned_data_pred)
    print('Done! Cleaned data stored at:' + data_path)
    print('cleaned_data_pred')
