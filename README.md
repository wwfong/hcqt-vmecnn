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

Headline results from the ISMIR 2019 submission, evaluated on a violin sonata subset of the Su Dataset against two state-of-the-art baselines (Melodia and the Deep Salience Map CNN of Bittner et al.). See [`docs/results.md`](docs/results.md) for the full tables, ablations, and figures.

Five standard melody extraction metrics are reported, where higher is better except for VFA (Voicing False Alarm Rate, lower is better):

- **VR** — Voicing Recall Rate
- **VFA** — Voicing False Alarm Rate (lower is better)
- **RPA** — Raw Pitch Accuracy
- **RCA** — Raw Chroma Accuracy
- **OA** — Overall Accuracy (headline metric)

| System | VR | VFA ↓ | RPA | RCA | **OA** |
|---|---:|---:|---:|---:|---:|
| Melodia [Salamon & Gómez, 2012] (baseline) | 50 | 36 | 6 | 15 | 21 |
| DSM CNN [Bittner et al., 2017] (baseline) | 56 | 50 | 29 | 32 | 34 |
| Single CNN, raw output | 83 | 28 | 73 | 73 | 73 |
| Single CNN, with HMM smoothing | 98 | 56 | 82 | 83 | 73 |
| Multi-Task CNN (γ=0.7/0.3), raw output | 77 | 25 | 68 | 68 | 70 |
| **Multi-Task CNN (γ=0.7/0.3), with HMM smoothing** | **93** | **43** | **79** | **80** | **74** |

All values are percentages, read from Figure 2 of the paper.

### Headline findings

- The best system is the **Multi-Task CNN with γ_Poly = 0.7, γ_Mono = 0.3 and HMM post-processing**, achieving an **Overall Accuracy of 74.94%** on the Su Dataset violin sonata test set.
- All proposed representation-specific CNN systems outperform both state-of-the-art baselines by a wide margin. The worst-performing variant (MT CNN raw) still beats the DSM CNN's OA by **+32.36 pp** and Melodia's by **+45.95 pp**.
- HMM-based post-processing lifts OA by an average of **+3.47 pp**, ranging from −0.12 pp for the Single CNN (whose voicing was already strong) to +7.49 pp for the worst Multi-Task variant.
- Multi-Task CNNs achieve a lower VFA than Single CNNs in every condition, suggesting a more robust timbral representation. They achieve higher OA than Single CNNs **only when paired with HMM smoothing**.

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
