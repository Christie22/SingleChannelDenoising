# Literature
> Papers on LSTM applied to audio

- `[GHu2018]` Single-Channel Speech Enhancement using Deep Learning
- `[HEr2015]` Phase-sensitive and recognition-boosted speech separation using deep recurrent neural networks
- `[EMa2015]` A novel approach for automatic acoustic novelty detection using a denoising autoencoder with bidirectional LSTM neural networks
- `[FWe2014]` Discriminatively trained recurrent neural networks for single-channel speech separation

Other resources:
- http://slazebni.cs.illinois.edu/spring17/lec26_audio.pdf
- https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
- https://gist.github.com/naotokui/12df40fa0ea315de53391ddc3e9dc0b9


## `[GHu2018]`
- Data representation:
- Target:
- Metrics/Loss:
- Architecture:
- Training:
- Dataset:
- Baseline:
- Evaluation:
- Notes:
- Useful refs:

## `[HEr2015]`
- Data representation:
- Target: phase-sensitive mask
- Metrics/Loss: phase-sensitive spectrum approximation
- Architecture: bidirectional LSTM
- Training: multi-stage training, adding layers
- Dataset: CHiME-2 challenge
- Baseline: 2 lstm layers, 256 nodes, 100 bin log-mel spectogram
- Evaluation: SDR, SIR
- Notes: good baseline!
- Useful refs:

## `[EMa2015]`
- Data representation: 10ms power spectrogram -> 26-bin mel -> log
- Target: novel events
- Metrics/Loss: mean quare error, positive first order differences, frame energy
- Architecture: denoising autoencoder with bidirectional LSTM
- Training: training set = background environmental sounds, test set = ‘abnormal’ sounds
- Dataset: PASCAL CHiME challenge
- Baseline: GMM, HMM, and compression autoencoder models
- Evaluation: precision, recall, f-measure
- Notes: different purpose than enhancemnt, interesting structure
- Useful refs:

## `[FWe2014]`
- Data representation: 100 mel ^2/3 auditory spectrum
- Target: estimated clean speech spectrograms
- Metrics/Loss: SNR
- Architecture: LSTM-DRNN, DNN
- Training:
- Dataset: evaluation: CHiME-2 challenge, training: Wall Street Journal (WSJ-0) corpus of read
- Baseline: discriminative NMF
- Evaluation: SDR
- Notes: nice experimental setup, results & discussion,
- Useful refs:
