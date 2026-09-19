# Tiny Seq2Seq Transformer: PyTorch Training Pipeline

A complete, self-contained sequence-to-sequence (Seq2Seq) Transformer training pipeline built entirely from scratch with raw PyTorch. This pipeline demonstrates dataset generation, custom character-level tokenization, sinusoidal positional encoding, attention masking, teacher-forcing training, and autoregressive greedy decoding evaluation on a synthetic sequence-reversal task (`ABC` $\rightarrow$ `CBA`).

---

## Key Features

- **Raw PyTorch Architecture**: Uses `torch.nn` Transformer modules without relying on high-level wrappers like Hugging Face `transformers` or `Trainer`.
- **Deterministic Positional Encoding**: Implements the canonical Sine/Cosine positional encodings registered as persistent PyTorch buffers.
- **Attention & Causal Masking**: Handles source key padding masks, target key padding masks, and upper-triangular causal attention masks.
- **Custom Tokenization**: Zero third-party NLP tokenizers. Custom character vocabulary with `<PAD>=0`, `<SOS>=1`, `<EOS>=2`, and characters `A-Z`.
- **Reproducible Pipeline**: Multi-seed determinism across Python, NumPy, and PyTorch.
- **Autoregressive Evaluation**: Greedy token-by-token sequence generation evaluated against an untrained random-weight baseline on exact-match (EM) accuracy.
- **Fully Containerized**: Ready for single-command end-to-end execution via Docker & Docker Compose.

---

## Repository Structure

```
tiny-seq2seq-transformer/
├── README.md                 # Project documentation
├── .gitignore                # Git ignore rules
├── .env                      # Local environment configuration
├── .env.example              # Environment variables template
├── requirements.txt          # Python dependencies
├── submission.json           # Model & training hyperparameter configuration
├── Dockerfile                # Docker build specification
├── docker-compose.yml        # Multi-stage container runner
│
├── data/                     # Dataset storage (generated)
│   ├── train.csv             # Training dataset (source, target)
│   └── val.csv               # Validation dataset (source, target)
│
├── output/                   # Evaluation & artifact outputs
│   ├── metrics.csv           # Epoch-by-epoch loss tracking
│   ├── checkpoint.pt         # Saved model weights & metadata
│   └── eval_results.json     # EM accuracy and baseline comparison
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── data/                 # Data generation, tokenizer, dataset
│   │   ├── __init__.py
│   │   ├── generate.py       # Synthetic reversal data generator
│   │   ├── tokenizer.py      # CharacterTokenizer
│   │   └── dataset.py        # SequenceReversalDataset & collate_fn
│   ├── model/                # Model architecture
│   │   ├── __init__.py
│   │   └── transformer.py    # PositionalEncoding & TinySeq2SeqTransformer
│   ├── train/                # Training orchestration
│   │   ├── __init__.py
│   │   └── train.py          # Training loop with teacher forcing
│   └── eval/                 # Evaluation & decoding
│       ├── __init__.py
│       └── evaluate.py       # Autoregressive decoding & baseline benchmark
│
├── tests/                    # Pytest test suite
│   ├── __init__.py
│   ├── test_tokenizer.py     # Tokenizer & dataset collation tests
│   ├── test_model.py         # Model architecture & AST checks
│   └── test_pipeline.py      # Contract & output verification tests
│
└── docs/                     # Detailed documentation
    ├── architecture.md       # Architecture & tensor diagrams
    └── verification.md       # Quality assurance & verification checklist
```

---

## Quickstart & Local Setup

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/rakeshchinni77/tiny-seq2seq-transformer.git
cd tiny-seq2seq-transformer

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline

```bash
# Step 1: Generate synthetic sequence reversal dataset
python src/data/generate.py

# Step 2: Train the Seq2Seq Transformer model
python src/train/train.py

# Step 3: Run autoregressive evaluation
python src/eval/evaluate.py
```

### 3. Run Unit Tests

```bash
pytest -v tests/
```

---

## Docker Execution

To build and run the entire pipeline end-to-end inside Docker:

```bash
docker-compose up --build
```

The container will automatically:
1. Generate the dataset in `./data/`.
2. Train the model and log loss metrics to `./output/metrics.csv` and `./output/checkpoint.pt`.
3. Perform autoregressive greedy evaluation and write `./output/eval_results.json`.
4. Exit cleanly with status code `0`.

---

## Configuration & Hyperparameters

Hyperparameters can be adjusted via `submission.json`:

```json
{
  "model_config": {
    "d_model": 64,
    "nhead": 4,
    "num_layers": 2
  },
  "train_config": {
    "epochs": 5,
    "batch_size": 64
  }
}
```

---

## Architecture Summary

```
Source Sequence: "ABC"
   │
   ▼
[<SOS>, 'A', 'B', 'C', <EOS>]  (Tokens: [1, 3, 4, 5, 2])
   │
   ▼
Embedding Layer (29 -> 64) * sqrt(64)
   │
   ▼
Sinusoidal Positional Encoding (Fixed PE table)
   │
   ▼
Transformer Encoder (2 layers, 4 heads, d_ff=128)
   │
   ▼ Memory Representation
   │
Transformer Decoder (2 layers, 4 heads, Causal Masked)
   │
   ▼
Linear Projection (64 -> 29)
   │
   ▼
CrossEntropyLoss (ignore_index=0) / Greedy Next-Token Argmax
```

---

## Benchmark Results

| Metric | Trained Checkpoint | Random-Weight Baseline | Pass Bar Requirement |
|---|---|---|---|
| **Exact Match (EM)** | $\ge 95.0\%$ | $\approx 0.0\%$ | `trained_em > random_em + 0.1` (Passed) |
| **Loss Convergence** | Monotonically decreasing | N/A | `final_val_loss < initial_val_loss` (Passed) |
| **Model Size** | ~170,000 parameters | ~170,000 parameters | `< 1,000,000 parameters` (Passed) |
