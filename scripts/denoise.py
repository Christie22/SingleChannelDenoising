# get a trained model, a noisy piece of data, and feed the latter into the former

import os
import numpy as np
#from keras.models import load_model
import librosa

from libs.utilities import load_autoencoder_lossfunc, load_autoencoder_model
from libs.processing import s_to_power, power_to_s, make_fragments, unmake_fragments, normalize_spectrum, unnormalize_spectrum

def denoise(model_name, model_path, input_path, output_path,
        sr, n_fft, hop_length, win_length, frag_hop_length, frag_win_length, 
        batch_size, cuda_device):

    print('[dn] Applying model in {} at {} on data in {}'.format(model_name, model_path, input_path))
    print('[dn] Denoising parameters: {}'.format({
        'model_name': model_name,
        'model_path': model_path,
        'input_path': input_path,
        'output_path': output_path,
        'cuda_device': cuda_device
    }))

    # set GPU devices
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    ## Input data handling
    print('[dn] Loading data from {}...'.format(input_path))
    # load data from file name
    x_noisy, _ = librosa.core.load(input_path, sr=sr)
    # convert to TF-domain
    s = librosa.core.stft(x_noisy, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    # apply pre-processing (data representation)
    y_proc = s_to_power(s)
    # split into fragments
    y_frags_noisy = make_fragments(y_proc, frag_hop_len=frag_hop_length, frag_win_len=frag_win_length)
    print('[dn] Generated {} fragments with shape {}'.format(len(y_frags_noisy), y_frags_noisy[0].shape))
    # normalize fragments
    std_frags_noisy = np.empty(len(y_frags_noisy))
    for i in range(len(y_frags_noisy)):
        frag_normalized, frag_std = normalize_spectrum(y_frags_noisy[i])
        y_frags_noisy[i] = frag_normalized
        std_frags_noisy[i] = frag_std

    # load trained model
    print('[dn] Loading model from {}...'.format(model_path))
    lossfunc = load_autoencoder_lossfunc(model_name)
    _, _, model = load_autoencoder_model(model_path, {'lossfunc': lossfunc})
    # print model summary
    model.summary()

    # prediction on data
    print('[dn] Predicting with trained model...')
    y_frags_pred = model.predict(y_frags_noisy)
    print('[dn] Prediction finished! Generated {} fragments'.format(len(y_frags_pred)))

    ## Perform inverse operations on data
    # inverse normalization
    for i in range(len(y_frags_pred)):
        y_frags_pred[i] = unnormalize_spectrum(
            y_frags_pred[i], std_frags_noisy[i])
    # convert to complex spectrogram
    s_pred = power_to_s(y_frags_pred)
    # undo batches
    s_pred = unmake_fragments(s_pred, frag_hop_len=frag_hop_length, frag_win_len=frag_win_length)
    # get waveform
    x_pred = librosa.istft(s_pred, hop_length=hop_length, win_length=win_length)
    
    # store cleaned audio as wav file
    librosa.output.write_wav(output_path, x_pred, sr=sr)

    # done
    print('[dn] Done! Cleaned data is reconstructed and stored at {}'.format(output_path))
