# ML-driven 1-channel denoising system
> Semester project for *Sound and Music Computing* (AAU SMC 2018)

**Link to report: https://www.overleaf.com/project/5c94e7edf54b1f21e4c1c066 ( https://www.overleaf.com/project/5c781f4804b72d43bf4ded71)**

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
  conda create --name <env_name> --file spec-file.txt
  source activate <env_name>
  ```
- Every time the `spec-file.txt` is modified, update environment:
  ```sh
  conda install --name <env_name> --file spec-file.txt
  ```
### Model definition
Create new model as in `models/` as `model_<model_name>.py`.
- Implement a custom class
- Constructor takes model parameters
- Expose `get_model()` method which returns a `keras.Model` object
- Expose `get_lossfunc()` method with return a loss function taking `x_pred` and `x_true` as arguments
- Define encoder and decoder as separate models
- Name explicitly all layers in autoencoder model
- See `model_example.py` for reference
- Cite the source (paper, repo, etc) on top
### Usage
- List available commands: `python main.py --help`


## Structure
### Code
- `main.py`: scripts entry point
- `scripts/`: scripts for training a model, viewing results, and using encoder and decoder
- `libs/`: code dependencies for scripts
- `models/`: model architecture implementations
- `notebooks/`: jupyter notebooks for experiments and tests
- `tools/`: miscellaneous software tools
- `Pipfile`, `Pipfile.lock`, `environment.yml`: list of dependencies, used for setting up pipenv (local) and conda (remote) environments
### Text
- `notes/`: minutes from group meetings
- `literature/`: relevant papers sorted by category
- `ext/`: unsorted, mixed stuff


### Christie Laurent & Riccardo Miccini, 2019

