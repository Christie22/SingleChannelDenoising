# MODELS_IDEAS

## Already implemented
### Auto-encoder: 'model_example.py'

## Propositions
### Fully-connected CNN 


### Based on [KHa2015]
Architecture:
- training: non-linear DNN-based regression model
    * normalization in [0,1] of both target (by hand) and DNN output (sigmoid func)
    * MMSE between target and DNN output (average on frames and batches)
- expected results: mediocre

### Based on [BWu2017]
Step 1: 
Concept: improve [KHa2015]'s model by normalizing globally instead of locally
Architecture:
- pre-training:
    * RBM: 1 epoch
    * learning rate: 0.4
- Fine-tuning:
    * 30 epoch
    * learning rate: 8e-5
- training: non-linear DNN-based regression model
    * global mean (0) variance (1) normalization in of both target features (by hand) and DNN output (ReLU func)
    * MMSE between target and DNN output (average on frames and batches)
    * 3 hidden layers, 2048 nodes/layer, 7 frames of input feature expansion, mini-batch size: 128
- expected results: better than previous model for low and medium freq

Step 2:
Concept: take into consideration the impact of rt60 by relevantly parametrizing the model
- step 2.a: frame-shift-aware DNN: RT60-dependent frame shift size at the DNN input 
- step 2.b: acoustic-context-aware DNN: RT60-dependent acoustic context window size at the DNN input 
- combination of steps 2.a and 2.b to design a reverberation-time-aware DNN based on the estimation of the RT60 and on the choice of best frame (R) and window (N) sizes based on previous DNNs. Read R and N into a table designed after many experiments and based on PESQ best performances.
Architecture: same but R and N values during training and testing.

Possible extensions? 
- not LSTM (i feel like it would be difficult to fit a RNN with adaptive windowing). Online model needed


### Based on [YZh2019]
Concept: conducting denoising and dereverberation sequentially using DNNs
- step 1: denoising DNN: 
- step 2: dereverberation DNN: 
- step 3: joint training



### Based on [DWi2017]
Concept: time-frequency masking in the complex domain: using complex ideal ratio mask (cIRM) and clean-anechoic speech as the desired signal for DNN-based enhancement
