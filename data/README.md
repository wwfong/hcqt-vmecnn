# Data

Training data is not committed to this repository because of licence and size constraints. The `examples/` subdirectory contains a single short violin recording (`ViolinRecitalDavid.{wav,mid,csv}`) used for end-to-end demo and sanity checks.

## Layout

```
data/
├── raw/                # gitignored. Place datasets here after downloading.
├── processed/          # gitignored. HCQT features and labels land here after preprocessing.
└── examples/           # a single demo file, included in the repo.
```

## Datasets used for training

The models in this repository were trained on a combination of:

- **MedleyDB** (Bittner et al., 2014): multi-track music dataset with f0 annotations. Used for training the multi-task targets.
- **Bach10** (Duan et al., 2010): chamber music multi-track recordings. Used as an out-of-domain evaluation set.
- **MIR-1K, RWC**, and select internal recordings, used for ablations.

You will need to obtain these datasets separately from their respective sources and place them under `data/raw/`.

## Producing the preprocessed features

After downloading the raw data, run:

```bash
python scripts/preprocess.py --config configs/default.yaml
```

This computes the HCQT representations and stores them under `data/processed/` along with framewise f0 labels (`f0TrainingLabels.npy` in the legacy code).
