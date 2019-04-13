# Models
> Collection of models as Templated JSON (JSONT) files

See usage example in `notebooks/experiment_model_designer.ipynb`.

## `rnn_conv.jsont`
### Description
This model is based on the paper ["Convolutional-Recurrent Neural Networks for Speech Enhancement" by Han Zhao et al](https://arxiv.org/abs/1805.00579).   
It comprises a convolutional layer, a 2-layers recurrent network, and a fully-connected layer for reconstruction. 
The original paper uses LSTM instead of GRU recurrent layers.

### Arguments
- `n_conv`: # of convolutional filters,
- `n_recurrent`: # of recurrent units,
- `n_dense`: # of dense units (typically F*C),
- `timesteps`: # of time steps (typically T),
- `channels`: # of channels (typically C),
- `dropout_rate`: dropout rate

### Example
```py
template_args = {
    'n_conv': 256,
    'n_recurrent': 128,
    'n_dense': input_shape[0]*input_shape[2],
    'timesteps': input_shape[1],
    'channels': input_shape[2],
    'dropout_rate': 0.2
}
```


## `model_name.jsont`
### Description
...

### Arguments
- `name`: description,

### Example
```py
template_args = {
}
```

