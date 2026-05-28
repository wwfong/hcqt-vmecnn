"""
violin_melody - HCQT-based Multi-Task CNNs for violin melody extraction.

The submodules below carry the original 2018 thesis code. Most depend on the
legacy stack (Keras 2.1.5, TensorFlow 1.4, Librosa 0.6, Essentia 2.1.b4) and
will fail to import on a modern environment without those installed.

Lightweight introspection of the package itself (``import violin_melody``,
``violin_melody.__version__``) does not pull in any of the heavy ML imports
and is safe in CI smoke tests.
"""

__version__ = "0.1.0"
__author__ = "David W. Fong"

# Submodule names are documented but not eagerly imported, so consumers can
# `import violin_melody` without triggering Keras/TensorFlow imports.
__all__ = [
    "data",
    "features",
    "models",
    "training",
    "inference",
    "postprocessing",
    "evaluation",
    "utils",
]
