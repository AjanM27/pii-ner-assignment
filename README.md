# PII Named-Entity Recognition for STT Transcripts

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-modeling-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

Token-level named-entity recognition for detecting personally identifiable information (PII) in speech-to-text-style transcripts. The repository includes dataset loading, DistilBERT token classification, BIO-to-span decoding, rule-based validation, exact-span evaluation, and batch-size-1 latency measurement.

## Pipeline

```text
JSONL transcript data -> tokenizer -> DistilBERT token classifier
       -> BIO decoding and PII validation -> JSON spans and metrics
```

## Repository layout

- `src/train.py` - fine-tune and save model/tokenizer
- `src/predict.py` - predict validated character spans
- `src/eval_span_f1.py` - exact span precision/recall/F1
- `src/measure_latency.py` - batch-size-1 p50/p95 timing
- `src/dataset.py` and `src/labels.py` - data and label definitions
- `data/*.jsonl` - train/dev/test examples

## Setup

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first model run downloads the selected Hugging Face checkpoint. Do not commit private transcript data, credentials, or generated model output containing sensitive information.

## Train, predict, and evaluate

```bash
python src/train.py \\
  --model_name distilbert-base-uncased \\
  --train data/train.jsonl \\
  --dev data/dev.jsonl \\
  --out_dir out

python src/predict.py \\
  --model_dir out \\
  --input data/dev.jsonl \\
  --output out/dev_pred.json

python src/eval_span_f1.py \\
  --gold data/dev.jsonl \\
  --pred out/dev_pred.json
```

Training defaults are five epochs, batch size 8, learning rate `3e-5`, maximum sequence length 256, and automatic CUDA selection when available. Evaluation reports per-label exact-span metrics, macro-F1, PII-only metrics, and non-PII metrics.

## Latency measurement

```bash
python src/measure_latency.py \\
  --model_dir out \\
  --input data/dev.jsonl \\
  --runs 50
```

The assignment target is p95 latency below approximately 20 ms per utterance at batch size 1 on a reasonably modern CPU. This is an evaluation target, not a result guaranteed by the repository.

## Privacy and status

PII detection software can itself process sensitive information, and model checkpoints may memorize training examples. Review data and outputs before sharing. This is an assignment skeleton, not production-grade anonymization or a legal compliance guarantee. See [assignment.md](assignment.md) for the original task framing.
