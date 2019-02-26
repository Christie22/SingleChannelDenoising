# Questions

- `train.py`
  - `dataset_path`: should this be a single audio file, a series of audio files, a text file containing lists of files, or a directory name containing a list of files?
  - `model_name, rows, cols, etc`: should these and other model-related variables (number of convolutional filters, kernel size, intermediate dimensions...) be passed as a config-file? they are easy to parse and would make propagating settings easier

- `results.py`
  - What should it actually do? Plot training history (could be done into train.py) and calculate metrics on test data? What are the metrics?

  


