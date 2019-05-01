### PARAMETERS ###

# MODELS
- conv AE
- conv RNN
- conv RNN baseline
- conv TCN

# DATASET
- nb of clean samples
- nb of noise variations (nb of types * nb of snr)
- type of noise
==> see below our 4 datasets

# DATA PROCESSONG
- exponent, Re/Im, dB
- normalisation
- fragment size

# LOSS FUNCTION
- window size

# TRAIN PARAMS
- dropout rate
- learning rate

# COMPLEXITY (100K - 20M)
- nb of layers: exception / inception
- conv parames: nb of filter / kernel size / strides / dilations
- other tuning: 
  - TCN: residual stacks
  - RNN: timesteps



### DATASETS ###

# DATASET 0 - baby one
- ~30' speech
- pink noise
- 2 snr (15-25 dB)

# DATASET 1 - little one 1
- ~1h speech
- pink noise
- 3 snr (5-25 dB)

# DATASET 2 - little one 2
- ~1h speech
- 3 stationary noises
- 3 snr (5-25 dB)

# DATASET 3 - big one 
- ~1h speech
- 3 non-stationary noises
- 4 snr ((-5)-10 dB)