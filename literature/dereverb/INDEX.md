# Literature
> Papers on audio Dereverberation

- `[KHa2014]` Learning spectral mapping for speech dereverberation
- `[KHa2015]` Learning spectral mapping for speech dereverberation and denoising,
- `[BWu2017]` A reverberation-time-aware approach to speech dereverberation based on deep neural networks
- `[YZh2019]` Two-stage deep learning for noisy-reverberant speech enhancement 

Other resources:
- `[YZh2017]` A two-stage algorithm for noisy and reverberant speech enhancement 

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
- Architecture: supervised DNN
- Training: pre-training: none + joint dereverberation and denoising training + regular training with cross-validation
- Dataset:  IEEE (training),  TIMIT (testing), CHiME-2 for various {T60, RIRs (various azimuths), noise types}
- Baseline: (5) IBM based on single variance-based feature + estimated inverse filters + dereverb IBM + unprocessed 
- Evaluation: SNR, STOI, PESQ 
- Notes: post-processing stage to reconstruct the signals (iterative procedure). Have quickly tested RNN and encourage readers to use it in future work
- Useful refs: reconstruction [4], pitch-based studies for separating reverberant voiced speech [30], [15],


## `[BWu2017]`
- Data representation: log-power spectra (LPS) (frame size and hop length depend on estimated T60 ) 
- Target: LPS of corresponding clean speech
- Metrics/Loss: MMSE
- Architecture: nonlinear DNN-based regression model
- Training: 2 stages: pairs of reverberant and anechoic speech (training) + LPS features to get enhanced LPS features and estimation of T60 (dereverb).
- Dataset: TIMIT for various {T60, RIRs (various azimuths)} - different training times
- Baseline: HWW-DNN [KHa2015], Reverberation-Time-Aware DNN (RTA-DNN) "Oracle" or not,
- Evaluation: PESQ, frequency-weighted segmental signal-to-noise ratio (fwSegSNR), STOI
- Notes: 
    - phase is directly extracted from the reverberant speech; 
    - dereverberation only



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


## `[YUe2016]`
- Data representation:  cepstrum (9 frames * 39 MFCCs) with Cepstral mean normalization (CMN) 
were used as input)
- Target:  clean speech features
- Metrics/Loss: MSE
- Architecture: AE
- Training: pretraining (RBM) + standard training on pair of clean speech and corresponding reverberant speech
- Dataset: different for training (WSJCAM0 ) and testing ( MC-WSJ-AV + WSJCAM0) with different RIR
- Baselines: spectral domaim AE, Multi-step linear prediction (MSLP), temporal structure normalization (TSN), CMN, DAE+TSN for simulated and real distant-talking environments
- Evaluation: Word Error Rates (WERs)
- Notes: TSN is a post-processing method. Seems to stand apart from the other papers... 
- Useful refs: MLSP [14], TSN [31]
