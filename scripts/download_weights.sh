#!/usr/bin/env bash
# Download pretrained model weights.
#
# The original ~600MB of .h5 model files (MTMECNN_1..5, MonoMECNN_1, PolyMECNN_1)
# and the borrowed weights/melody2.h5 from rabitt/ismir2017-deepsalience
# are NOT committed to this repo.
#
# TODO: Replace the URLs below with a hosted location (HuggingFace Hub, Zenodo,
# Google Drive, S3) once weights are re-uploaded.
#
# Usage:
#   bash scripts/download_weights.sh

set -euo pipefail

WEIGHTS_DIR="${WEIGHTS_DIR:-./weights}"
mkdir -p "$WEIGHTS_DIR"

echo "Downloading pretrained weights into $WEIGHTS_DIR ..."
echo
echo "Pretrained weights are not yet hosted. The original files were:"
echo "  - MTMECNN_1.h5 ... MTMECNN_5.h5  (~94 MB each)"
echo "  - MonoMECNN_1.h5                 (~47 MB)"
echo "  - PolyMECNN_1.h5                 (~47 MB)"
echo "  - weights/melody2.h5             (~1.6 MB, borrowed from rabitt/ismir2017-deepsalience)"
echo
echo "Until they are re-uploaded, you can request a copy from the author"
echo "(see contact details in the README) or retrain from scratch:"
echo
echo "    python scripts/train.py --config configs/default.yaml"
echo
exit 0
