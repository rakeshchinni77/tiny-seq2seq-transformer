# Tiny Seq2Seq Transformer

A lightweight, from-scratch Encoder-Decoder Sequence-to-Sequence (Seq2Seq) Transformer implemented in raw PyTorch. The model is trained to solve a synthetic sequence-reversal task (`ABC` $\rightarrow$ `CBA`), demonstrating the end-to-end mechanics of neural attention architectures without high-level wrappers.

---

## Objective

Modern NLP commonly relies on high-level abstractions like Hugging Face `Trainer` and pre-trained language models. While efficient for application delivery, these wrappers obscure core architectural interactions: embedding projections, sinusoidal positional tables, attention masking tensors, teacher forcing, and autoregressive greedy decoding loops.

This project was built to demystify and implement the entire Seq2Seq Transformer workflow from first principles using standard PyTorch primitives (`torch.nn`), validating pipeline convergence and inference mechanics from random initialization.

---

## Problem Statement

The system learns a deterministic sequence transformation: reversing a sequence of uppercase alphabetical characters (`A`–`Z`):

```text
Input (Source):      ABCDEF
Output (Target):     FEDCBA

Input (Source):      HELLO
Output (Target):     OLLEH
```

While rule-based reversal is trivial, training a Transformer to learn this mapping tests its ability to:
1. Preserve and leverage token order using positional representations.
2. Cross-attend from decoder states back into encoder memory representations.
3. Decouple input positions from output positions through autoregressive generation.

---

## Solution

The project implements a complete, self-contained machine learning pipeline:

```text
Input Sequence ("ABC")
       │
       ▼
Character Tokenizer ([<SOS>, A, B, C, <EOS>])
       │
       ▼
Source Embedding + Sinusoidal Positional Encoding
       │
       ▼
Transformer Encoder (Self-Attention + Feed-Forward)
       │
       ▼ Memory
Transformer Decoder (Causal Self-Attention + Cross-Attention)
       │
       ▼
Linear Vocabulary Projection (Logits)
       │
       ▼
Autoregressive Output ("CBA")
```

---

## Key Features

- **Raw PyTorch Architecture**: Built directly with `torch.nn` modules—no Hugging Face `transformers`, `tokenizers`, or `Trainer`.
- **Custom Character Tokenizer**: Lightweight vocabulary supporting `<PAD>`, `<SOS>`, `<EOS>`, and uppercase letters `A`–`Z`.
- **Deterministic Positional Encodings**: Canonical Sine/Cosine encodings registered as persistent PyTorch buffers.
- **Attention & Causal Masking**: Boolean padding masks combined with upper-triangular causal attention masks.
- **Teacher-Forced Training**: Target-shifted input and expected token pairs optimized with `nn.CrossEntropyLoss(ignore_index=PAD_ID)`.
- **Autoregressive Greedy Decoding**: Token-by-token sequence generation during evaluation until `<EOS>` or maximum sequence cutoff.
- **Baseline Benchmarking**: Systematic comparison of the trained checkpoint against an untrained random-weight baseline.
- **Reproducible Pipeline**: Multi-seed determinism across Python, NumPy, and PyTorch.
- **Containerized Workflow**: Fully reproducible, single-command orchestration via Docker Compose.
- **Automated Test Suite**: Pytest suite verifying model dimensions, masking behavior, loss convergence, and evaluation contracts.

---

## Architecture

The model implements a standard Encoder-Decoder Transformer with `batch_first=True`.

```text
  Source Tokens (Batch, Src_Len)             Target Tokens (Batch, Tgt_Len)
               │                                          │
               ▼                                          ▼
     [ nn.Embedding ]                           [ nn.Embedding ]
               │                                          │
               ▼                                          ▼
   [ PositionalEncoding ]                     [ PositionalEncoding ]
               │                                          │
               ▼                                          ▼
    ┌──────────────────────┐                   ┌──────────────────────┐
    │  Transformer Encoder │                   │  Transformer Decoder │
    │                      │                   │  (Causal Self-Attn   │
    │  (Self-Attention +   │                   │   + Cross-Attention) │
    │   Feed-Forward)      │                   └──────────┬───────────┘
    └──────────┬───────────┘                              │
               │                                          │
               └────────────── Memory ────────────────────┘
                                                          │
                                                          ▼
                                                [ nn.Linear (Vocab) ]
                                                          │
                                                          ▼
                                                Logits (Batch, Tgt, 29)
```

### Component Details
- **Token Embeddings**: Maps discrete token indices to hidden dimension $d_{model} = 64$, scaled by $\sqrt{d_{model}}$.
- **Positional Encoding**: Fixed sinusoidal table adding deterministic order representations.
- **Encoder**: 2 layers, 4 attention heads, feed-forward dimension 128, dropout 0.1.
- **Decoder**: 2 layers, 4 attention heads, feed-forward dimension 128, causal attention masking.
- **Output Projection**: Linear layer projecting hidden states to vocabulary size (29).
- **Complexity**: Total trainable parameters: **239,325** (well within lightweight execution limits).

---

## Tokenization

The custom `CharacterTokenizer` maps uppercase characters and special control tokens to discrete integer IDs:

| Token | ID | Purpose |
|---|---|---|
| `<PAD>` | `0` | Padding token to equalize batch lengths |
| `<SOS>` | `1` | Start-of-Sequence prompt token |
| `<EOS>` | `2` | End-of-Sequence termination token |
| `'A'` .. `'Z'` | `3` .. `28` | Uppercase alphabet characters |

### Example
```text
Input string:     "ABC"
Encoded tokens:   [1, 3, 4, 5, 2]   (<SOS>, A, B, C, <EOS>)
Decoded string:   "ABC"
```

---

## Training

The model is trained using **Teacher Forcing**, providing the shifted ground-truth sequence as decoder input:

```text
Target Sequence:    <SOS>  C  B  A  <EOS>
Decoder Input:      <SOS>  C  B  A
Expected Output:    C      B  A  <EOS>
```

### Masking & Optimization
- **Source Padding Mask**: Boolean mask (`True` = `<PAD>`) preventing attention across padding.
- **Target Padding Mask**: Boolean mask (`True` = `<PAD>`) ignoring padding slots in target sequences.
- **Target Causal Mask**: Boolean upper-triangular mask ensuring position $i$ only attends to positions $\le i$:
  ```text
  [[False,  True,  True,  True],
   [False, False,  True,  True],
   [False, False, False,  True],
   [False, False, False, False]]
  ```
- **Loss Function**: `nn.CrossEntropyLoss(ignore_index=0)` to avoid penalizing predictions on padding.
- **Optimizer**: `torch.optim.Adam` with learning rate `0.003`.
- **Training Schedule**: 10 epochs, batch size 64.

---

## Autoregressive Inference

During evaluation, target sequences are unknown. The model generates predictions token-by-token:

```text
Step 1:  Input: [<SOS>]                --> Predicts 'C'
Step 2:  Input: [<SOS>, 'C']           --> Predicts 'B'
Step 3:  Input: [<SOS>, 'C', 'B']      --> Predicts 'A'
Step 4:  Input: [<SOS>, 'C', 'B', 'A'] --> Predicts '<EOS>' (Terminates)

Decoded Output: "CBA"
```

Greedy decoding takes the $\operatorname{argmax}$ over vocabulary logits at each step until `<EOS>` is emitted or the cutoff length is reached.

---

## Project Structure

```text
tiny-seq2seq-transformer/
├── README.md                 # Project overview and documentation
├── .gitignore                # Git ignore patterns
├── .env.example              # Environment variables template
├── requirements.txt          # Python dependencies
├── submission.json           # Model and training hyperparameter configuration
├── Dockerfile                # Container build definition
├── docker-compose.yml        # Pipeline container orchestration
│
├── src/                      # Source package
│   ├── __init__.py
│   ├── data/                 # Dataset generation, tokenizer, dataset loader
│   │   ├── __init__.py
│   │   ├── generate.py       # Synthetic sequence pair generator
│   │   ├── tokenizer.py      # Character-level tokenizer
│   │   └── dataset.py        # SequenceReversalDataset & collate_fn
│   ├── model/                # Model architecture definitions
│   │   ├── __init__.py
│   │   └── transformer.py    # PositionalEncoding & TinySeq2SeqTransformer
│   ├── train/                # Training orchestration
│   │   ├── __init__.py
│   │   └── train.py          # Training loop and loss logger
│   └── eval/                 # Evaluation and decoding
│       ├── __init__.py
│       └── evaluate.py       # Greedy decoding & baseline benchmark
│
├── tests/                    # Automated test suite
│   ├── __init__.py
│   ├── test_tokenizer.py     # Tokenizer & collation tests
│   ├── test_model.py         # Model architecture & AST tests
│   └── test_pipeline.py      # Metric schema & convergence tests
│
└── docs/                     # Technical documentation
    ├── architecture.md       # Detailed architecture & tensor mechanics
    └── verification.md       # Quality assurance checklist & criteria
```

> **Note**: Runtime directories `data/` (`train.csv`, `val.csv`) and `output/` (`metrics.csv`, `checkpoint.pt`, `eval_results.json`) are generated during execution and are ignored by git. Local environment overrides should be defined in `.env` (not committed).

---

## Tech Stack

| Category | Technology | Description |
|---|---|---|
| **Language** | Python 3.11 | Core runtime environment |
| **Deep Learning** | PyTorch (`torch.nn`) | Model architecture, loss, optimizer |
| **Data Processing** | NumPy, Pandas | Dataset generation, batching, CSV logging |
| **Testing** | Pytest | Unit and integration test suite |
| **Containerization** | Docker, Docker Compose | Reproducible end-to-end execution |

*Note: Third-party NLP wrappers (such as Hugging Face Transformers or Tokenizers) are intentionally not used.*

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/rakeshchinni77/tiny-seq2seq-transformer.git
cd tiny-seq2seq-transformer
```

### 2. Set Up Virtual Environment

#### Windows PowerShell
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Configuration

The pipeline supports configuration via `submission.json` and environment variables.

### `submission.json`
```json
{
  "model_config": {
    "d_model": 64,
    "nhead": 4,
    "num_layers": 2
  },
  "train_config": {
    "epochs": 10,
    "batch_size": 64
  }
}
```

### `.env.example`
A template `.env.example` is provided for local path configuration:
```env
DATA_DIR=./data
OUTPUT_DIR=./output
DEVICE=cpu
SEED=42
CONFIG_PATH=./submission.json
```

---

## Running Locally

Execute each stage of the pipeline sequentially:

### 1. Generate Dataset
```bash
python src/data/generate.py
```
*Generates `data/train.csv` (5,000 samples) and `data/val.csv` (1,000 samples).*

### 2. Train Model
```bash
python src/train/train.py
```
*Trains for 10 epochs, writes `output/metrics.csv`, and saves `output/checkpoint.pt`.*

### 3. Evaluate Model
```bash
python src/eval/evaluate.py
```
*Runs autoregressive greedy decoding and saves benchmark results to `output/eval_results.json`.*

### 4. Run Test Suite
```bash
pytest -q
```
*Executes all 15 unit and integration tests.*

---

## Running with Docker

The entire pipeline can be executed in an isolated container environment using Docker Compose:

```bash
docker compose up --build
```

### Execution Flow
1. Builds a lightweight `python:3.11-slim` container with CPU-optimized PyTorch.
2. Automatically runs `generate.py` $\rightarrow$ `train.py` $\rightarrow$ `evaluate.py`.
3. Mounts host directories `./data` and `./output`, making generated artifacts immediately accessible.
4. Exits with status code `0` upon successful completion.

---

## Results

Below are the verified results from the end-to-end evaluation run:

| Metric | Result |
|---|---|
| **Training Samples** | 5,000 |
| **Validation Samples** | 1,000 |
| **Trainable Parameters** | 239,325 |
| **Training Epochs** | 10 |
| **Initial Validation Loss** | 1.7658 |
| **Final Validation Loss** | 0.2375 |
| **Trained Exact Match (EM)** | **63.8%** |
| **Random Baseline Exact Match** | **0.0%** |
| **Performance Margin** | **+63.8 percentage points** |
| **Automated Test Suite** | **15 passed** |

The trained checkpoint shows substantial validation loss reduction (from 1.7658 down to 0.2375) and decisively outperforms the random-weight baseline on exact-match sequence reversal.

---

## Example Prediction

Sample evaluated from the validation split during autoregressive decoding:

```text
Source:     SQEHJUB
Expected:   BUJHEQS
Predicted:  BUJHEQS  (Exact Match)
```

---

## Reproducibility

- **Controlled Random Seeds**: Training uses seed `42`; independent validation set generation uses seed `142`.
- **Config-Driven Architecture**: Hyperparameters are loaded deterministically from `submission.json`.
- **Self-Contained Checkpoints**: `output/checkpoint.pt` bundles model weights, architecture configuration, and training metadata.
- **Isolated Environment**: Docker Compose ensures consistent execution across different host operating systems.

---

## Limitations

- **Synthetic Task**: Sequence reversal is a synthetic toy problem designed for architectural study, not natural language translation.
- **Sequence Lengths**: Configured for short sequences (length 3–8 characters).
- **Character-Level Vocabulary**: Vocabulary is limited to 29 character-level tokens.
- **Strict Evaluation**: Exact-match requires every character and stop token to be perfectly placed; near-correct sequences score 0 on exact match.

---

## Future Improvements

- **Beam Search Decoding**: Implement beam search with length penalties alongside greedy decoding.
- **Dynamic Sequence Scaling**: Support longer sequences (length 16–64) and variable-length stress tests.
- **Learning Rate Scheduling**: Add linear warmup with cosine decay.
- **Character Error Rate (CER)**: Track Levenshtein distance / edit accuracy alongside exact match.
- **REST / CLI Inference API**: Add interactive command-line and HTTP endpoints for ad-hoc sequence reversal.
- **GPU Acceleration**: Add multi-device CUDA / MPS execution toggles.

---

## Documentation

- [docs/architecture.md](docs/architecture.md): In-depth architectural design, tensor transformations, positional encoding formulas, and attention masking mechanics.
- [docs/verification.md](docs/verification.md): Verification matrix, quality assurance checklists, and validation commands.

---

## License

This project is intended for educational, experimentation, and portfolio purposes.
