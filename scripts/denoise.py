# get a trained model, clean one piece of data (or frag) that we noise and feed the latter into the former

import os
import numpy as np
from keras.models import load_model
import librosa

from libs.data_generator import DataGenerator
from libs.utilities import load_dataset, load_autoencoder_lossfunc, load_autoencoder_model
from libs.model_utils import LossLayer
from libs.processing import s_to_reim, reim_to_s, make_fragments, unmake_fragments

def denoise(model_name, model_path, input_path, output_path,
        sr, n_fft, hop_length, win_length, frag_hop_length, frag_win_length, 
        batch_size, cuda_device):

    print('[n] Applying model in {} at {} on data in {}'.format(model_name, model_path, input_path))
    print('[n] Denoising parameters: {}'.format({
        'model_name': model_name,
        'model_path': model_path,
        'input_path': input_path,
        'output_path': output_path,
        'cuda_device': cuda_device
    }))

    # set GPU devices
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    ## input data handling
    print('[n] Loading data from {}...'.format(input_path))
    # load data from file name
    x, _ = librosa.core.load(input_path, sr=sr)
    # convert to TF-domain
    s = librosa.core.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    # apply pre-processing (data representation)
    s_proc = s_to_reim(s)
    # split into fragments
    s_frags = make_fragments(s_proc, frag_hop_len=frag_hop_length, frag_win_len=frag_win_length)
    s_frags = np.array(s_frags)
    print('[n] Generated {} fragments with shape {}'.format(len(s_frags), s_frags[0].shape))

    # load trained model
    print('[n] Loading model from {}...'.format(model_path))
    lossfunc = load_autoencoder_lossfunc(model_name)
    _, _, model = load_autoencoder_model(model_path, {'lossfunc': lossfunc})
    # print model summary
    #model.summary()

    # prediction on data
    print('[n] Predicting with trained model...')
    s_frags_pred = model.predict(s_frags)
    print('[n] Prediction finished!')

    # TODO remove this part (debugging only)
    print('shape of output: ', s_frags_pred.shape)

    # perform inverse operations on data
    # TODO

    # store clean audio as wav file
    # TODO

    # very slow at the beginning then very fast (real-time possible)
    #np.save('cleaned_data_pred', cleaned_data_pred)
    print('Cleaned data reconstructed and stored at {}'.format(output_path))


    print('[n] Done!')
