# tool for converting spectrograms to audio

import os
import numpy as np
import pandas as pd
import librosa
from utilities import *

# ISTFT
def generate_wave(s, n_fft=1024, hop_length=512):
    print('Performing istft...')
    return librosa.core.istft(s, 
        win_length=n_fft,
        hop_length=hop_length)

# Save and play audio
def store_audio(x, filepath, sr):
    print('Storing audio in `{}`...'.format(filepath))
    librosa.output.write_wav(filepath, x, sr=sr)


def main(args):
    # read spectrogram
    spectrogram = args['spec_path']
    s = import_spectrogram(spectrogram)

    # perform inverse processing
    s = unprocess_data(s[0])

    # generate audiowave
    n_fft = args['n_fft']
    hop_length = args['hop_length']
    x = generate_wave(s, n_fft, hop_length)

    # store audio
    filepath = args['output_path']
    sr = args['samplerate']
    store_audio(x, filepath, sr)

    # byeeeeee
    print('Done! check content of `{}`'.format(args['output_path']))
    return



# run the thing
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert re/im spectrogram to audio file.')

    parser.add_argument('spec_path', type=str, help='npy file containing re/im spectrogram')
    parser.add_argument('output_path', type=str, help='wav output file')

    parser.add_argument('-sr', '-samplerate', '--samplerate', type=int, help='Samplerate')
    parser.add_argument('-fft', '-n_fft', '--n_fft', type=int, help='Num FFT bins')
    parser.add_argument('-hl', '-hop_length', '--hop_length', type=int, help='hop length')

    # read defaults from params.py
    from params import params
    parser.set_defaults(
        samplerate = params['samplerate'],
        n_fft = params['n_fft'],
        hop_length = params['hop_length'],
    )

    args = parser.parse_args()
    main(vars(args))
