### PARAMETERS ###

- conv AE
# DATA PROCESSONG
- exponent, Re/Im, dB
- normalisation
- fragment size

# LOSS FUNCTION
- window size

# DATASET
- nb of clean samples
- nb of noise variations (nb of types * nb of snr)
- type of noise

# MODELS
- conv AE
- conv RNN
- conv RNN baseline
- conv TCN

# TRAIN PARAMS
- dropout rate
- learning rate

# COMPLEXITY (100K - 20M)
- nb of layers: exception / inception
- conv parames: nb of filter / kernel size / strides / dilations
- other tuning: 
  - TCN: residual stacks
  - RNN: timesteps
