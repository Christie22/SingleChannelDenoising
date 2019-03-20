# Questions

- `train.py`
  - `model_name, rows, cols, etc`: should these and other model-related variables (number of convolutional filters, kernel size, intermediate dimensions...) be passed as a config-file? they are easy to parse and would make propagating settings easier

- `results.py`
  - What should it actually do? Plot training history (could be done into train.py) and calculate metrics on test data? What are the metrics?

- reconstruction
- how do we deal with the phase? Do we take it from the clean signal or do we try to estimate it from the denoised signal?
  


