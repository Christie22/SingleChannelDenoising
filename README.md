# ML-driven 1-channel denoising system
> Semester project for *Sound and Music Computing* (AAU SMC 2018)

**Link to report: https://www.overleaf.com/project/5c781f4804b72d43bf4ded71**
**Link to datasets: https://drive.google.com/drive/u/0/folders/1MmtiRQF-33Gkx1ZJZ7vSDCzy-vqjaUw6/**


## Proposal
Implementation of a single-channel denoising system in voice detection applications.
The system would be based on deep learning techniques (e.g. autoencoders).

Initially, a baseline model shall be chosen, and an evaluation procedure established.
Subsequently, improvements comprising techniques drawn from relevant literature will be implemented into the baseline, and evaluated accordingly.
The project will be carried out in an iterative fashion.

Should satisfactory performances be achieved within a subset of the timeframe, the following further developments could be considered:
- Embedded implementation
- De-reverberation
- Multiple channels (e.g. with headset)


## Instructions
### Setup
- Setup remote environment:
```sh
conda create --name ml_env1 --file spec-file.txt
source activate ml_env1
```
### Model definition
- Create new model as in `models/` as `model_<model_name>.py`
  - Implement a custom class named `AEModelFactory`
  - Constructor takes model parameters
  - Expose `get_model()` methos which returns a `keras.Model` object
  - Define encoder and decoder as separate models
  - Name explicitly all layers in autoencoder models
  - Define loss function in a dummy layer
  - See `model_example.py` for reference
  - List the source, reference paper, etc!
- In `train.py`, inside of `create_model_vae`, add clause in `# import model` section (stuff in <> needs to be filled):
```py
  elif model_name == '<MODEL_NAME>':
    print('Using model `{}` from {}'.format(model_name, '<MODEL_FILENAME>'))
    import <MODEL_FILENAME> as model_vae
```
### Usage
- train model: `python train.py <options>`
- get training results and show scatter plots of latent space dimensions: `python results.py <options>`
- get latent space representations for given spectrograms: `python encode.py <options>`
- generate spectrograms from latent space representations: `python decode.py <options>`
- convert spectrograms to audio files: `python playAudio.py <options>`


## Structure
- `scripts/`: scripts for training a model, viewing results, and using encoder and decoder
- `libs/`: code dependencies for scripts
- `models/`: model architecture implementations
- `tools/`: miscellaneous software tools
- `spec-file.txt`: list of dependencies, used for setting up conda environment
- `Pipfile`: list of dependencies, used for setting up pipenv environment (local)


### Christie Laurent & Riccardo Miccini, 2019

