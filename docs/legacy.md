# Legacy notes (2018 codebase)

This repository preserves the original 2018 thesis code, organised into the modern researcher-style layout. The code itself was written against:

- Python 2.7
- Keras 2.1.5
- TensorFlow 1.4 backend

Modernising the codebase to Python 3 / PyTorch is on the roadmap. Until then, the following notes apply.

## Running the original code

To exactly reproduce the thesis pipeline:

1. Create a Python 2.7 virtual environment.
2. Install the pinned dependencies from `requirements-legacy.txt`.
3. Run `notebooks/archive/original_main.ipynb`, which orchestrates the full pipeline using the modules under `src/violin_melody/`.

The original module names have been preserved (`chooseRepresentation.py`, `preprocessing.py`, `training.py`, `postprocessing.py`, `predicting.py`, `evaluation.py`) and only moved into appropriate subpackages. Imports inside those files have **not** been rewritten yet, so re-running the original notebook may require adjusting `sys.path` or using PYTHONPATH:

```bash
PYTHONPATH=src/violin_melody:src/violin_melody/features:src/violin_melody/data:src/violin_melody/models:src/violin_melody/training:src/violin_melody/inference:src/violin_melody/postprocessing:src/violin_melody/evaluation jupyter notebook notebooks/archive/original_main.ipynb
```

## Borrowed code

`src/violin_melody/inference/predict_on_audio.py` is borrowed from [rabitt/ismir2017-deepsalience](https://github.com/rabitt/ismir2017-deepsalience) and used together with the pretrained `weights/melody2.h5` from the same repository, for evaluation of the violin melody extractor. The borrowed code retains its original license and authorship.

## Roadmap

- [ ] Port the multi-task CNN to PyTorch.
- [ ] Replace the Keras `.h5` model checkpoints with PyTorch `state_dict` files hosted on HuggingFace Hub.
- [ ] Add training callbacks for Weights & Biases logging.
- [ ] Provide a `pip install -e .` editable install via `pyproject.toml`.
- [ ] Add smoke tests under `tests/` that load the demo audio and assert basic output shapes.
- [ ] Write up the results tables from `outputs.txt` into `docs/results.md`.
