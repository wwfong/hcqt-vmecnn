# Results

Results from the ISMIR 2019 submission paper, extracted from Figure 2 of [`paper/multitask_cnns_violin_melody_extraction.pdf`](../paper/multitask_cnns_violin_melody_extraction.pdf).

## Setup

- **Test set:** a violin sonata subset of the Su Dataset (Su & Yang, 2015). Two violin sonata excerpts, 52 s total, 136 violin note labels, 5180 frames after feature extraction.
- **Training (target task, polyphonic):** 20 solo violin sonata movements (Beethoven) from MusicNet, 129 minutes, 22,971 violin note labels, augmented with pitch shifts up to +3 semitones. 285,036 training frames, 31,671 validation frames.
- **Training (auxiliary task, monophonic):** ~30 minutes of solo violin Bach sonatas and partitas from MusicNet, 7,792 violin note labels, augmented 4×. 289,125 training frames, 32,126 validation frames.
- **Baselines:** Melodia (Salamon & Gómez, 2012); Deep Salience Map CNN (Bittner et al., 2017).
- **Metrics:** Voicing Recall (VR), Voicing False Alarm (VFA, lower is better), Raw Pitch Accuracy (RPA), Raw Chroma Accuracy (RCA), Overall Accuracy (OA).

## Headline numbers

| System | VR | VFA ↓ | RPA | RCA | **OA** |
|---|---:|---:|---:|---:|---:|
| Melodia (Salamon & Gómez, 2012) | 50 | 36 | 6 | 15 | 21 |
| DSM CNN (Bittner et al., 2017) | 56 | 50 | 29 | 32 | 34 |
| Single CNN, raw output | 83 | 28 | 73 | 73 | 73 |
| Single CNN, with HMM smoothing | 98 | 56 | 82 | 83 | 73 |
| Multi-Task CNN (γ=0.7/0.3), raw output | 77 | 25 | 68 | 68 | 70 |
| **Multi-Task CNN (γ=0.7/0.3), with HMM smoothing** | **93** | **43** | **79** | **80** | **74** |

All values are percentages. Numbers from Figure 2 of the paper.

### Higher-precision values reported in the text

Where the paper text gives more precise numbers, they are:

- **Melodia OA:** 20.56%
- **DSM CNN OA:** 34.15% (computed from "13.59 pp margin" over Melodia)
- **Best system (MT CNN 3 + HMM):** OA = 74.94%
- **Worst proposed system (MT CNN 4 raw) OA:** 66.51% (computed from "+32.36 pp over DSM CNN")
- **Single CNN, HMM-smoothed:** RPA = 82.26%
- **MT CNN 4 raw:** RPA = 62.36%
- **Single CNN, HMM-smoothed:** VR = 97.78%
- **MT CNN 4 raw:** VR = 69.96%
- **Mean OA across all proposed VME CNN variants:** 71.89% (range of 8.43 pp)

## Voicing metrics

### Voicing Recall (VR)

- Every proposed system's VR exceeds both baselines by at least **+14.02 pp**.
- HMM smoothing lifts VR by an average of **+16.34 pp**, confirming that voicing recall errors dominate the raw CNN outputs.
- Range across VME CNN variants: **69.96% (MT CNN 4 raw) to 97.78% (Single CNN smoothed).**

### Voicing False Alarm (VFA)

- Lower is better.
- Melodia (36%) outperforms most CNN variants on this metric in isolation. The DSM CNN (50%) has the second worst VFA after the Single CNN with HMM smoothing (56%).
- HMM smoothing **raises** VFA by an average of **+19.57 pp** across our systems, because HMMs prefer to predict melodic note labels over `None`. This is the cost of the voicing recall lift.
- Multi-Task CNNs have lower VFA than their Single CNN counterparts in **every** condition. The MT-based representation appears to discriminate violin timbre from accompaniment more robustly.

## Pitch accuracy metrics

### Raw Pitch Accuracy (RPA)

- All proposed systems achieve **at least double** the RPA of the baselines (DSM CNN: 29.07%, Melodia: 6.32%).
- HMM smoothing raises RPA by an average of **+10.94 pp** across the proposed systems by correcting anomalously labelled frames against their temporal context.
- Range across proposed systems: **62.36% (MT CNN 4 raw) to 82.26% (Single CNN smoothed).**

### Raw Chroma Accuracy (RCA)

- Melodia is heavily susceptible to **octave errors**: its RCA (15%) exceeds its RPA (6%) by **+8.51 pp**.
- The CNN-based systems are only mildly affected: RCA - RPA is at most **+2.43 pp** across the proposed systems.

## Overall Accuracy (OA)

- The headline metric, combining VR, VFA, RPA, and RCA.
- **Best system: Multi-Task CNN 3 with HMM smoothing, OA = 74.94%.**
- Difference between best proposed and best baseline: **+40.79 pp** (vs. DSM CNN at 34.15%).
- Even the worst proposed system (MT CNN 4 raw) beats the DSM CNN by **+32.36 pp**.
- Multi-Task CNNs only achieve higher OA than Single CNNs **when paired with HMM smoothing**, suggesting that raw MT CNN errors are particularly amenable to temporal correction.

## Discussion

### Representation-specific vs. non-discriminatory

Representation-specific systems (which target a particular instrument by timbre) substantially outperform non-discriminatory melody extractors. The trade-off is reduced versatility, but the approach is straightforward to transfer to any polyphonic music signal given a suitably annotated dataset.

### Multi-Task CNN vs. Single CNN

- Multi-Task CNNs **overfit less** than the Single CNN, suggesting the auxiliary monophonic task improves generalisation.
- However, the test accuracy of Multi-Task CNNs falls short of their validation accuracy by **11.06 pp** on average, vs. zero gap for the Single CNN, indicating the Multi-Task variants' reliability could be improved with a larger and more varied training set.
- Single CNN has higher RPA and RCA than Multi-Task CNNs. So while MT is the best end-to-end OA when combined with HMM smoothing, the Multi-Task formulation is not strictly dominant on every metric.

## Loss-weight ablation

Five Multi-Task CNN variants were trained with different (γ_Poly, γ_Mono) weights. All satisfy γ_Poly ≥ 0.5 since the polyphonic violin melody task is the target.

| Variant | γ_Poly | γ_Mono | Notes |
|---|---:|---:|---|
| MT CNN 1 | 0.5 | 0.5 | equal weighting |
| MT CNN 2 | 0.6 | 0.4 | |
| **MT CNN 3** | **0.7** | **0.3** | **best OA when smoothed (74.94%)** |
| MT CNN 4 | 0.8 | 0.2 | worst raw OA (66.51%), largest gain from HMM (+7.49 pp) |
| MT CNN 5 | 0.9 | 0.1 | most polyphonic-weighted |

## Qualitative examples

The repository ships with [`data/examples/ViolinRecitalDavid.wav`](../data/examples/ViolinRecitalDavid.wav), a short violin recital recording, plus its ground-truth MIDI and CSV. To extract the violin melody from this clip with the best system:

```bash
bash scripts/download_weights.sh
python scripts/ViolinMelodyExtractor.py data/examples/ViolinRecitalDavid.wav
```

Output is a MIDI file with the estimated violin melody and a CSV of framewise f0 estimates.

## Future work

From the paper's conclusion:

- **Cross-stitch networks** (Misra et al., 2016) would let the concatenated CNNs in the Multi-Task architecture learn the degree of layer-wise information sharing.
- **Higher-order HMM** smoothing operating over notes rather than frames could further reduce voicing recall errors without inflating VFA as aggressively.
- **Larger and more varied training set** to capture violin sounds across their full amplitude envelope would address the validation-test accuracy gap observed in the Multi-Task variants.
