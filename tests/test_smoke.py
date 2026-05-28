"""
Smoke tests for hcqt-vmecnn.

These tests are deliberately lightweight: they confirm the repo's structural
invariants (package importable, example audio loadable, HCQT feature path
runs) without requiring the legacy 2018 ML stack (Keras 2.1.5, TensorFlow 1.4,
Essentia 2.1.b4). They are designed to run in CI on Python 3.10+ with only
NumPy, SciPy, librosa, and soundfile installed.

The full inference pipeline (CNN forward pass, HMM smoothing, MIDI export)
requires the pretrained model weights and the legacy stack, and is therefore
not exercised here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
EXAMPLE_WAV = REPO_ROOT / "data" / "examples" / "ViolinRecitalDavid.wav"

# Make the package importable without requiring a `pip install -e .`.
sys.path.insert(0, str(SRC_DIR))


def test_package_importable():
    """The top-level package should import without triggering heavy ML imports."""
    import violin_melody

    assert violin_melody.__version__ == "0.1.0"
    assert "data" in violin_melody.__all__
    assert "features" in violin_melody.__all__


def test_example_wav_exists():
    """The bundled demo recording should be present and a valid WAV file."""
    assert EXAMPLE_WAV.exists(), f"Missing example audio at {EXAMPLE_WAV}"
    assert EXAMPLE_WAV.stat().st_size > 0
    # Quick header check: RIFF...WAVE marker.
    with EXAMPLE_WAV.open("rb") as f:
        header = f.read(12)
    assert header[:4] == b"RIFF", "Example wav is not a valid RIFF file"
    assert header[8:12] == b"WAVE", "Example wav is missing WAVE chunk"


def test_example_wav_loadable_with_librosa():
    """librosa should load the example wav at 16 kHz, mono."""
    librosa = pytest.importorskip("librosa")

    y, sr = librosa.load(str(EXAMPLE_WAV), sr=16000, mono=True)

    # The recording is roughly 2 minutes; assert it is non-trivial in length.
    assert sr == 16000
    assert y.ndim == 1
    assert y.shape[0] > 16000  # at least one second of audio
    assert np.isfinite(y).all()
    # Signal should not be flat silence.
    assert float(np.abs(y).mean()) > 1e-4


def test_hcqt_feature_shape():
    """
    Compute a Harmonic Constant-Q Transform on a slice of the example audio
    using the same parameters as Section 2.1 of the ISMIR paper, and assert
    the output shape is sane.

    Paper config:
      - fs = 16 kHz, hop = 10 ms (R = 160 samples)
      - HCQT harmonics h in {1, 2}
      - fmin = 196.00 Hz (G3)
      - B = 48 bins per octave, K = 192 bins per channel (~4 octaves)
    """
    librosa = pytest.importorskip("librosa")

    y, sr = librosa.load(str(EXAMPLE_WAV), sr=16000, mono=True, duration=5.0)
    assert sr == 16000

    hop_length = 160              # 10 ms at 16 kHz
    bins_per_octave = 48
    n_bins = 192
    fmin = 196.00                 # G3
    harmonics = (1, 2)

    cqts = []
    for h in harmonics:
        c = np.abs(
            librosa.cqt(
                y=y,
                sr=sr,
                hop_length=hop_length,
                fmin=h * fmin,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
            )
        )
        cqts.append(c)

    hcqt = np.stack(cqts, axis=-1)

    # Expected shape: (n_bins, n_frames, n_harmonics)
    assert hcqt.shape[0] == n_bins
    assert hcqt.shape[2] == len(harmonics)
    # 5 seconds of audio at 10 ms hop is roughly 500 frames; allow a wide band.
    assert 100 < hcqt.shape[1] < 1000
    # Energy should be non-trivially non-zero in violin-relevant region.
    assert hcqt.mean() > 0.0
    assert np.isfinite(hcqt).all()


def test_repo_structural_invariants():
    """Sanity-check the directory layout the README documents."""
    expected_dirs = [
        "paper",
        "data/examples",
        "configs",
        "src/violin_melody/features",
        "src/violin_melody/data",
        "src/violin_melody/models",
        "src/violin_melody/training",
        "src/violin_melody/inference",
        "src/violin_melody/postprocessing",
        "src/violin_melody/evaluation",
        "scripts",
        "notebooks/archive",
        "tests",
        "docs",
        "results",
    ]
    for rel in expected_dirs:
        assert (REPO_ROOT / rel).is_dir(), f"Missing directory: {rel}"

    expected_files = [
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "Makefile",
        "requirements-legacy.txt",
        ".gitignore",
        "paper/multitask_cnns_violin_melody_extraction.pdf",
        "data/examples/ViolinRecitalDavid.wav",
        "scripts/ViolinMelodyExtractor.py",
        "scripts/download_weights.sh",
        "notebooks/archive/original_main.ipynb",
    ]
    for rel in expected_files:
        assert (REPO_ROOT / rel).is_file(), f"Missing file: {rel}"
