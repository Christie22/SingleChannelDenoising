# Literature
> Papers on audio Dereverberation

- `[KHa2014]` Learning spectral mapping for speech dereverberation
- `[KHa2015]` Learning spectral mapping for speech dereverberation and denoising,
- `[BWu2017]` A reverberation-time-aware approach to speech dereverberation based on deep neural networks
- `[YZh2019]` Two-stage deep learning for noisy-reverberant peech enhancement 
- `[XLu2013]` Speech enhancement based on deep denoising autoencoder - Interloan


Other resources:
- blabla

## `[KHa2014]`
- Data representation: cochleagram (64-channel gammatone * (20-ms time frames*11 frames)) (+ spectrogram) 
- Target: estimated cochleagram of the corresponding anechoic speech
- Metrics/Loss: mean square error (2 regularization terms: KL-divergence + L2-norm of the weights )
- Architecture: DNN
- Training: pre-training: stack of RBMs + finetuning of the weights thus obtained + reverbered-clean speech pairs 
- Dataset: TIMIT + IEEE database for various T60
- Baseline: (5) IBM based on single variance-based feature + estimated inverse filters + dereverb IBM + unprocessed + using spectrograms instead
- Evaluation: speech-to-reverberation ratio (SRR)
- Notes: cochleagram is better than spectrogram
- Useful refs: [3]

## `[KHa2015]`
- Data representation: log-magnitude spectrogram(161 freq bins* (20-ms time frames*11 frames)) 
- Target: log magnitude spectrogram of clean speech
- Metrics/Loss: mean square error
- Architecture: DNN
- Training: reverberant, or reverberant and noisy to clean speech , regular training with cross-validation
- Dataset:  IEEE (training),  TIMIT (testing), CHiME-2 for various {T60, RIRs (various azimuths), noise types}
- Baseline: (5) IBM based on single variance-based feature + estimated inverse filters + dereverb IBM + unprocessed 
- Evaluation: SNR, STOI, PESQ 
- Notes: post-processing stage to reconstruct the signals (iterative procedure)
- Useful refs: reconstruction [4]

## `[BWu2017]`
- Data representation: log-power spectra (LPS)
- Target: LPS of corresponding clean speech
- Metrics/Loss: MMSE
- Architecture: nonlinear DNN-based regression model
- Training: 2 stages: pairs of reverberant and anechoic speech (training) + LPS features to get enhanced LPS features and estimation of T60 (dereverb).
- Dataset: TIMIT for various {T60, RIRs (various azimuths)} - different training times
- Baseline: HWW-DNN [KHa2015], Reverberation-Time-Aware DNN (RTA-DNN) "Oracle" or not,
- Evaluation: PESQ, frequency-weighted segmental signal-to-noise ratio (fwSegSNR), STOI
- Notes: 
    - phase is directly extracted from the reverberant speech; 
    - they also propose a T60-dependent method for the dereverberation stage using 2 parameters (frame length and hop length) , that seems to be the best eventually


## `[YZh2019]`
- Data representation: log-magnitude spectrum (161 freq bins* (20-ms Hamming windows*11 frames)) 
- Target: normalized training target ( LPS of clean-anechoic speech) + IRM-processed magnitude spectrum of noisy-reverberant speech
- Metrics/Loss: new objective function that incorporates clean phase to calculate the MSE in the time domain, PSM
- Architecture: unsupervised DNN
- Training: two-stage strategy (denoising and dereverberation are conducted sequentially)
- Dataset: IEEE corpus and Diverse Environments Multichannel Acoustic Noise Database (DEMAND) for various  {untrained T60s, untrained SNRs, RIRs (various azimuths), noise types, different speakers}
- Baseline: a bunch: one-stage masking method, spectral mapping method, with PSM as obj. func., supervised masking, supervised mapping, 2-stage without TDR (see Notes)
- Evaluation: PESQ, STOI
- Notes: 
    - system = 3 modules: denoising <-> dereverb <-> time-domain signal reconstruction (TDR)
    - use exponential linear units (ELUs) instead of ReLU
    - masking seems to work better than mapping
- Useful refs:


## `[XLu2013]`
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
