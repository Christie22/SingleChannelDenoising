import numpy as np
import librosa
from sklearn.metrics import mean_squared_error
# from pystoi.stoi import stoi


# example evaluation metric; takes 2 complex spectrograms as input (true, pred) and outputs a scalar
def sample_metric(y_pred, y_true):
    return mean_squared_error(y_pred, y_true)


# TODO refactor other metrics to follow interface from sample_metric()


# TODO consider using this: https://github.com/craffel/mir_eval/blob/master/mir_eval/separation.py

def calc_metrics(y, yest, **kwargs):
    # calc SDR and NRR

    # checking the type of the input
    sizY = y.shape
    sizYest = yest.shape

    #spectrograms
    inputType = 'F' if min(sizY) > 2 and min(sizYest)>2 else 'T'

    # retrieve params' values
    keys = kwargs.keys()
    
    tBands = kwargs.pop('tBands', '')  if 'tBands' in keys else 1
 
    fBands = kwargs.pop('fBands', '') if 'fBands' in keys else 1

    # cqse: we want to calculate the metrics from the T-F representations of y and yest
    if fBands > 1:
        samplerate = kwargs.pop('samplerate','') if 'samplerate' in keys else 44100 # or 10k # or display an error

        n_fft = kwargs.pop('n_fft','') if 'n_fft' in keys else 256 # 0-padded to 512

        hop_length = kwargs.pop('hop_length','') if 'hop_length' in keys else np.int(n_fft*.5)


        if inputType == 'T': # need to perform the STFT first
            Yest = librosa.core.stft(yest, hop_length=hop_length, win_length=n_fft,window='hann')
            Y    = librosa.core.stft(y,    hop_length=hop_length, win_length=n_fft,window='hann')

        elif inputType == 'F':
            Yest = yest
            Y    = y

        ## calculate the grid for the calculation of the metrics
        # Frequence: :
        logscale = librosa.mel_frequencies(n_mels=fBands,fmin=0,fmax=samplerate/2) #just to get a log scale
        #librosa.fft_frequencies(sr=22050, n_fft=subbands)
        stepsF = np.floor(logscale/logscale[-1]* Y.shape[0])

        # Time:
        if tBands > 0:
            stepsT = np.round(np.linspace(0,Y.shape[1],tBands))
        else:
            tBands = 1
            stepsT = np.array([0,Y.shape[1]])


    else: #if fBands <= 1: (and  inputType == 'T' or 'F') : time domain
        fBands = 1
        stepsF = np.array([0,1])

        Yest = yest
        Y    = y

    #grid to cut y and yest into pieces
    stepsT = np.round(np.linspace(0, y.shape[0], tBands)) if tBands > 0 else np.array([0,sizY[0]])


    SDR, NRR = np.zeros((fBands,tBands)), np.zeros((fBands,tBands))

    for nf in range(fBands):
        for nt in range(tBands):

            yestSelec = Yest[ np.int(stepsF[nf]) : np.int(stepsF[nf+1]) , np.int(stepsT[nt]) : np.int(stepsT[nt+1])]
            ySelec    = Y[    np.int(stepsF[nf]) : np.int(stepsF[nf+1]) , np.int(stepsT[nt]) : np.int(stepsT[nt+1]) ]



            # SIgnal 2 Distorsion Ratio:
            diffTrue2Est= np.array([ a-b for a,b in zip(ySelec, yestSelec) ])

            numSDR   = np.mean(diffTrue2Est @ diffTrue2Est.T)
            denomSDR = np.mean(yestSelec @ yestSelec.T)
            # formula uses inverse ratio inside the log, compared to the paper cited
            SDR[nf][nt] = 10 * np.log10(numSDR / denomSDR)

            # Noise Reduction Ratio:
            numNR   = np.mean(ySelec @ ySelec.T)
            denomNR = denomSDR
            NRR[nf][nt] = 10 * np.log10(numNR / denomNR)


    #### Calculation of STOI
    # Clean and den should have the same length, and be 1D
    d = None # stoi(y, yest, sr, extended=False)

    output = {'Signal-To-Distorsion Ratio (SDR)' : SDR, 'Noise Reduction Ratio' : NRR, 'STOI' : d}
    return output
