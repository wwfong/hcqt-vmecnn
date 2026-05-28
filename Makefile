.PHONY: help install preprocess train evaluate predict download-weights clean

help:
	@echo "Targets:"
	@echo "  install            Install Python dependencies (legacy 2.7 stack)"
	@echo "  download-weights   Fetch pretrained model weights"
	@echo "  preprocess         Compute HCQT features from raw audio"
	@echo "  train              Train a model from the default config"
	@echo "  evaluate           Evaluate a trained model"
	@echo "  predict FILE=...   Run the violin melody extractor on FILE"
	@echo "  clean              Remove processed data and run artefacts"

install:
	pip install -r requirements-legacy.txt

download-weights:
	bash scripts/download_weights.sh

preprocess:
	python scripts/preprocess.py --config configs/default.yaml

train:
	python scripts/train.py --config configs/default.yaml

evaluate:
	python scripts/evaluate.py --config configs/default.yaml

predict:
	@if [ -z "$(FILE)" ]; then echo "Usage: make predict FILE=path/to/audio.wav"; exit 1; fi
	python scripts/ViolinMelodyExtractor.py $(FILE)

clean:
	rm -rf data/processed/*
	rm -rf results/runs/*
	find . -name __pycache__ -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
