# Violin Melody Extraction with Multi-Task CNNs on HCQTs

> Deep multi-task convolutional neural networks operating on Harmonic Constant-Q Transform (HCQT) representations, with Hidden Markov Model post-processing for temporal smoothing, applied to violin melody extraction from polyphonic music signals.

**Authors:** David W. Fong, Patrick A. Naylor (Imperial College London)
**Venue:** Submitted to the 20th International Society for Music Information Retrieval Conference (ISMIR 2019)
**Paper:** [`paper/multitask_cnns_violin_melody_extraction.pdf`](paper/multitask_cnns_violin_melody_extraction.pdf)

This repository contains the code, models, and example data accompanying my BEng final-year project at Imperial College London, supervised by Prof. Patrick A. Naylor. The thesis was awarded First Class with prize. The work was submitted to ISMIR 2019.

## TL;DR

We treat violin melody extraction from polyphonic audio as a multi-task learning problem on Harmonic Constant-Q Transform inputs:

- **Input representation:** HCQT, which stacks constant-Q transforms of the audio at integer harmonic multiples of a reference frequency. This gives the convolutional layers a built-in inductive bias for harmonic structure.
- **Architecture:** Multi-task CNN jointly predicting framewise f0 and related melody targets that share useful structure.
- **Post-processing:** A Hidden Markov Model is applied to the framewise output to enforce track-level temporal coherence.
- **Output:** A monophonic melody transcription that can be written to MIDI.

## Repository structure

```
hcqt-vmecnn/
├── paper/                                # ISMIR 2019 submission PDF
├── data/
│   ├── examples/                         # ViolinRecitalDavid.{wav,mid,csv} for demo
│   └── README.md                         # how to obtain training datasets (MedleyDB etc.)
├── configs/                              # YAML/JSON configs for model and training
├── src/violin_melody/                    # importable package
│   ├── features/                         # HCQT + spectrogram representations
│   ├── data/                             # dataset loading + preprocessing
│   ├── models/                           # CNN architectures
│   ├── training/                         # training loops
│   ├── inference/                        # prediction
│   ├── postprocessing/                   # HMM smoothing
│   ├── evaluation/                       # metrics
│   └── utils/
├── scripts/                              # CLI entry points (train, evaluate, predict, etc.)
├── notebooks/                            # exploratory + reporting notebooks
│   └── archive/original_main.ipynb       # the 2018 notebook preserved as-is
├── tests/                                # smoke tests
├── docs/                                 # extended documentation
└── results/                              # figures, tables, model outputs
```

## Quick start (demo)

```bash
# 1. Clone
git clone https://github.com/wwfong/hcqt-vmecnn.git
cd hcqt-vmecnn

# 2. Download pretrained model weights (~600MB total, not in git)
bash scripts/download_weights.sh

# 3. Run the violin melody extractor on the included example
python scripts/ViolinMelodyExtractor.py data/examples/ViolinRecitalDavid.wav
```

The extractor produces a MIDI file with the estimated violin melody and a CSV of framewise f0 estimates.

## Reproducing training

> Note: the original codebase is Python 2.7 + Keras 2.1.5 (2018-era). A Python 3 / PyTorch port is on the roadmap; see [Issues](https://github.com/wwfong/hcqt-vmecnn/issues) or the legacy notes in `docs/legacy.md`.

1. Obtain the training data. See [`data/README.md`](data/README.md) for the datasets used and how to download them.
2. Preprocess the audio into HCQT representations:
   ```bash
   python scripts/preprocess.py --config configs/default.yaml
   ```
3. Train a model:
   ```bash
   python scripts/train.py --config configs/default.yaml
   ```
4. Evaluate on a held-out set:
   ```bash
   python scripts/evaluate.py --config configs/default.yaml
   ```

## Results

Headline results from the ISMIR 2019 submission. See [`docs/results.md`](docs/results.md) for the full tables, ablations, and figures.

| Model | Pitch accuracy | Voicing recall | Voicing false alarm | Overall accuracy |
|---|---|---|---|---|
| MonoMECNN | TBD | TBD | TBD | TBD |
| PolyMECNN | TBD | TBD | TBD | TBD |
| **MTMECNN (ours)** | **TBD** | **TBD** | **TBD** | **TBD** |

(Numbers to be populated from `outputs.txt` and the paper.)

## Citation

If you use this work, please cite:

```bibtex
@unpublished{fong2019multitask,
  title  = {Multi-task Convolutional Neural Networks for Violin Melody Extraction from Polyphonic Music Signals},
  author = {Fong, David W. and Naylor, Patrick A.},
  year   = {2019},
  note   = {Submitted to the 20th International Society for Music Information Retrieval Conference (ISMIR)},
  url    = {https://github.com/wwfong/hcqt-vmecnn}
}
```

A machine-readable citation is also available in [`CITATION.cff`](CITATION.cff).

## Acknowledgments

- The HCQT representation and the broader research direction draw on Bittner et al.'s "Deep Salience Representations for f0 Estimation in Polyphonic Music" (ISMIR 2017). The `predict_on_audio.py` script and the supporting `weights/melody2.h5` are borrowed from [rabitt/ismir2017-deepsalience](https://github.com/rabitt/ismir2017-deepsalience) for evaluation purposes.
- Supervised by Prof. Patrick A. Naylor at Imperial College London, Department of Electrical and Electronic Engineering.

## License

MIT. See [`LICENSE`](LICENSE).
