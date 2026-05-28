# Model Architecture

This document describes the multi-task CNN architecture used for violin melody extraction in this repository. See [`paper/multitask_cnns_violin_melody_extraction.pdf`](../paper/multitask_cnns_violin_melody_extraction.pdf) for the full treatment.

## Input representation: HCQT

The Harmonic Constant-Q Transform (HCQT) stacks constant-Q transforms of the audio computed at integer multiples of a reference frequency. Concretely:

```
HCQT[h, k, t] = |CQT(audio scaled by h, fmin)|
```

for harmonic indices h ∈ {1, 2, ..., H}, frequency bin k, and time frame t. Stacking the harmonic axis as input channels gives the convolutional filters direct access to the harmonic alignment of pitched content, which is the right inductive bias for monophonic and polyphonic melody extraction.

In this work:
- H = 6 harmonics
- 60 bins per octave, 6 octaves (360 bins total)
- fmin = 32.7 Hz (C1)

## Variants

Three model variants are studied:

- **MonoMECNN**: single-task CNN predicting framewise f0 for monophonic violin.
- **PolyMECNN**: single-task CNN predicting framewise f0 for the violin in a polyphonic mixture.
- **MTMECNN (ours)**: multi-task CNN jointly predicting f0 of the violin, f0 of the dominant melodic source, and a voicing indicator. The shared backbone learns harmonic features useful across all three tasks; the heads specialise.

## Backbone

A stack of 2D convolutional layers over (frequency, time) with the harmonic axis as input channels. Filter counts ramp from 16 to 128 across the stack; kernel sizes are 5x5 with batch normalisation and dropout (p=0.25).

## Heads

Each task head is a small 1x1 convolutional projection followed by a sigmoid activation, producing a (frequency-bin, time-frame) probability map.

## Loss

The total loss is a weighted sum of per-task binary cross-entropies between the predicted activation map and a Gaussian-blurred ground-truth pitch contour. Weights were tuned on a held-out validation set.

## Post-processing

The framewise activation peaks are converted into a per-frame pitch track. To enforce temporal coherence, a Hidden Markov Model is applied over the pitch states with a high self-transition probability and an emission distribution derived from the network's activation map. The HMM Viterbi decode produces the final monophonic melody contour.
