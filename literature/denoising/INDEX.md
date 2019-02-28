# Literature
> Papers on Denoising techniques

- `[XLu2013]` Speech enhancement based on deep denoising autoencoder
- `[MKa2015]` Denoising Convolutional Autoencoders for Noisy Speech Recognition
- `[EGr2017]` Single Channel Audio Source Separation using Convolutional Denoising Autoencoders

## `[XLu2013]`
- Data representation: Mel scale power spectrum (40 filters * 7 frames of 8ms each), tot = 56ms
- Target: estimated clean speech spectrograms
- Metrics/Loss: mean square error
- Architecture: deep autoencoders
- Training: noisy-clean speech pairs, greedy layer wised pretraining + fine tuning training
- Dataset: continuous japanese speech data, 2 types of noise (factory and car), 3 SNRs (0, 5, 10 dB)
- Baseline: MMSE
- Evaluation: noise reduction, noise distortion, PESQ
- Notes: results are positive, very brief explanation
- Useful refs: [7]

## `[MKa2015]`
- Data representation: stft(10ms) -> element-wise log-transform -> zero-mean unit-variance
- Target: element-wise log-transform, estimated clean speech spectrograms
- Metrics/Loss: mean square error
- Architecture: 6 models: long skinny, long skinny, cascaded 3x3, 8-layer 7x7, network-in-network, 5-layer 7x
- Training: noisy-clean speech pairs, regular training
- Dataset: CHiME dataset (clean and corresponding artificially mixed noisy audio tracks)
- Baseline: single-layer, 1x1 affine conv-net w/t nonlinearities
- Evaluation: reconstruction MSE vs baseline
- Notes: no further evaluation metrics, lots of models
- Useful refs: N/A

## `[EGr2017]`
- Data representation: 15x1025 magnitude spectrum, tot = 370ms
- Target: estimated clean speech spectrograms
- Metrics/Loss: mean square error
- Architecture: fully convolutional autoencoders
- Training: noisy-clean speech pairs, source separation training
- Dataset: SiSEC-2015-MUS-task dataset
- Baseline: deep fully connected feedforward neural networks
- Evaluation: SDR, SIR, SAR
- Notes: training process is weird, implemented in keras
- Useful refs: