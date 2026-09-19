# Verification Matrix & Quality Assurance Guide

This checklist documents the verification procedures and pass criteria for all 10 core engineering contracts.

---

## Core Requirements Checklist

| ID | Requirement | Key Artifact | Pass Criteria | Status |
|---|---|---|---|---|
| **CR-1** | `dataset-generation` | `data/train.csv`, `data/val.csv` | Train & val CSV files exist; every target string equals reversed source string. | Verified |
| **CR-2** | `tokenization-pipeline` | `src/data/tokenizer.py` | Special tokens (`<PAD>=0`, `<SOS>=1`, `<EOS>=2`), variable padding, no external NLP tokenizers. | Verified |
| **CR-3** | `transformer-architecture` | `src/model/transformer.py` | Raw PyTorch `nn.Module`, deterministic Sine/Cosine PE, causal & padding masks, < 1M parameters. | Verified |
| **CR-4** | `training-pipeline` | `src/train/train.py` | $\ge 3$ epochs, saves `output/metrics.csv` & `output/checkpoint.pt`. | Verified |
| **CR-5** | `loss-convergence` | `output/metrics.csv` | Final `val_loss` < First `val_loss`; zero `NaN` values. | Verified |
| **CR-6** | `evaluation-suite` | `src/eval/evaluate.py`, `output/eval_results.json` | Autoregressive greedy decoding; `trained_em_score > random_em_score + 0.1`. | Verified |
| **CR-7** | `docker-compose-setup` | `docker-compose.yml`, `Dockerfile` | `docker-compose up --build` completes with exit code 0; produces all output artifacts under 5 min. | Verified |
| **CR-8** | `submission-config` | `submission.json` | JSON format with `model_config` and `train_config` overrides. | Verified |
| **CR-9** | `env-example-file` | `.env.example` | Environment variable definitions for paths, seeds, devices. | Verified |
| **CR-10** | `code-quality-and-formatting` | `src/`, `tests/` | Standard Python typing, docstrings, `__init__.py` packaging, unit tests pass. | Verified |

---

## Verification Commands

### 1. Run Complete Python Pipeline
```bash
# 1. Generate dataset
python src/data/generate.py

# 2. Execute training
python src/train/train.py

# 3. Run autoregressive evaluation
python src/eval/evaluate.py
```

### 2. Run Test Suite
```bash
pytest -v tests/
```

### 3. Run Containerized Pipeline via Docker Compose
```bash
docker-compose up --build
```
Verify that the output directory contains:
- `output/metrics.csv`
- `output/checkpoint.pt`
- `output/eval_results.json`
