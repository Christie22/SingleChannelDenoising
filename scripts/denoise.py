# get a trained model, a noisy piece of data, and feed the latter into the former

import os
import numpy as np
#from keras.models import load_model
import librosa

#from libs.data_generator import DataGenerator
from libs.utilities import load_autoencoder_lossfunc, load_autoencoder_model
#from libs.model_utils import LossLayer
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

    ## Input data handling
    print('[n] Loading data from {}...'.format(input_path))
    # load data from file name
    x_noisy, _ = librosa.core.load(input_path, sr=sr)
    # convert to TF-domain
    s = librosa.core.stft(x_noisy, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    # apply pre-processing (data representation)
    y_proc = s_to_reim(s)
    # split into fragments
    y_frags_noisy = make_fragments(y_proc, frag_hop_len=frag_hop_length, frag_win_len=frag_win_length)
    y_frags_noisy = np.array(y_frags_noisy)
    print('[n] Generated {} fragments with shape {}'.format(len(y_frags_noisy), y_frags_noisy[0].shape))
    # Normalization per fragment
    std_frag = np.empty(len(y_frags_noisy))
    for i, yy in enumerate(y_frags_noisy):
        std_frag[i] = np.std(yy)
        yy = (yy - np.mean(yy))/std_frag[i]

    # load trained model
    print('[n] Loading model from {}...'.format(model_path))
    lossfunc = load_autoencoder_lossfunc(model_name)
    _, _, model = load_autoencoder_model(model_path, {'lossfunc': lossfunc})
    # print model summary
    #model.summary()

    # prediction on data
    print('[n] Predicting with trained model...')
    y_frags_pred = model.predict(y_frags_noisy)
    print('[n] Prediction finished!')

    ## Perform inverse operations on data
    # Inverse normalization
    for i, yy in enumerate(y_frags_pred):
        yy = yy *std_frag[i] # + np.mean(ss)
    # convert to complex spectrogram
    s_pred = reim_to_s(y_frags_pred)
    # undo batches
    s_pred = unmake_fragments(s_pred, frag_hop_len=frag_hop_length, frag_win_len=frag_win_length)
    # get absolute spectrogram
    s_pred = np.abs(s_pred) ** 2
    # get waveform
    x_pred = librosa.istft(s_pred, hop_length=hop_length, win_length=win_length)
    
    # store cleaned audio as wav file
    librosa.output.write_wav(output_path, x_pred, sr=sr)

    # very slow at the beginning then very fast (real-time possible)
    #np.save('cleaned_data_pred', cleaned_data_pred)
    print('Cleaned data is reconstructed and stored at {}'.format(output_path))


    print('[n] Done!')
