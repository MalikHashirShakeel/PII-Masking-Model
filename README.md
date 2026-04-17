# 🛡️ PII Masking Pipeline

> **Detecting and masking Personally Identifiable Information (PII) — names and emails — using a fine-tuned BERT NER model and a zero-shot LLM approach.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [CLI Usage](#cli-usage)
- [Pipeline Details](#pipeline-details)
- [Results & Comparison](#results--comparison)
- [Error Analysis](#error-analysis)
- [Future Improvements](#future-improvements)

---

## Overview

This project implements a complete PII masking pipeline that identifies and redacts **person names** and **email addresses** from text. Two complementary approaches are implemented and compared:

| Approach | Model | Method | Strengths |
|----------|-------|--------|-----------|
| **Model 1** | BERT (`bert-base-uncased`) | Supervised fine-tuning on WikiNeural + synthetic emails | High precision & recall (F1 ≈ 0.97) |
| **Model 2** | Qwen2.5-1.5B-Instruct | Zero-shot prompting with structured JSON extraction | No training needed, generalizable |

### Key Contributions

1. **Synthetic Email Augmentation** — The WikiNeural dataset lacks email entities. We inject realistic synthetic emails with proper BIO tagging to enable email detection training.
2. **Hybrid Inference Pipeline** — Combines regex-based email detection with BERT NER for names, ensuring high reliability.
3. **Robust LLM Extraction** — Uses structured JSON output parsing with hallucination filtering and regex fallback.
4. **Comprehensive Evaluation** — Reports accuracy, precision, recall, F1, FPR, FNR, and per-entity breakdowns.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PII Masking Pipeline                        │
├───────────────────────────────┬─────────────────────────────────────┤
│     BERT (Fine-Tuned)         │        LLM (Zero-Shot)             │
│                               │                                     │
│  ┌─────────────────────┐      │  ┌──────────────────────────┐      │
│  │  Regex Email Mask   │      │  │  Few-Shot Prompt Builder │      │
│  │  (Guaranteed Catch) │      │  │  (System + 2 Examples)   │      │
│  └────────┬────────────┘      │  └───────────┬──────────────┘      │
│           ▼                   │              ▼                      │
│  ┌─────────────────────┐      │  ┌──────────────────────────┐      │
│  │  Smart Tokenizer    │      │  │  Qwen2.5-1.5B Generation │      │
│  │  (Preserve [EMAIL]) │      │  │  (Structured JSON Output)│      │
│  └────────┬────────────┘      │  └───────────┬──────────────┘      │
│           ▼                   │              ▼                      │
│  ┌─────────────────────┐      │  ┌──────────────────────────┐      │
│  │  BERT NER Inference │      │  │  Robust JSON Parser      │      │
│  │  (Token Classif.)   │      │  │  (Regex + AST Fallback)  │      │
│  └────────┬────────────┘      │  └───────────┬──────────────┘      │
│           ▼                   │              ▼                      │
│  ┌─────────────────────┐      │  ┌──────────────────────────┐      │
│  │  Confidence Filter  │      │  │  Post-Processing         │      │
│  │  + BIO Correction   │      │  │  (Hallucination Filter)  │      │
│  └────────┬────────────┘      │  └───────────┬──────────────┘      │
│           ▼                   │              ▼                      │
│  ┌─────────────────────┐      │  ┌──────────────────────────┐      │
│  │  [NAME] Masking     │      │  │  Robust Text Replacement │      │
│  │  + Detokenization   │      │  │  (Spacing-Aware)         │      │
│  └─────────────────────┘      │  └──────────────────────────┘      │
└───────────────────────────────┴─────────────────────────────────────┘
```

---

## Project Structure

```
project/
├── main.py                     # CLI entry point (single command interface)
├── requirements.txt            # Python dependencies
├── report.md                   # 2-page summary report
├── .gitignore                  # Git exclusion rules
│
├── data/
│   ├── raw/                    # Original WikiNeural dataset
│   │   ├── train.json          #   28,516 training samples
│   │   └── test.json           #   3,650 test samples
│   └── processed/
│       └── train_augmented.json  # Augmented with synthetic emails
│
├── models/
│   └── bert_ner_final/         # Saved fine-tuned BERT weights
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── tokenizer_config.json
│
├── notebooks/
│   └── data_analysis.ipynb     # Exploratory data analysis
│
├── results/
│   ├── bert_results.txt        # BERT training & evaluation logs
│   └── llm_results.txt         # LLM evaluation logs
│
├── src/
│   ├── __init__.py
│   ├── utils.py                # Shared utilities (labels, metrics, paths)
│   │
│   ├── data/                   # Data handling subpackage
│   │   ├── __init__.py
│   │   ├── loader.py           # JSON data loader
│   │   ├── preprocessing.py    # Dataset analysis & statistics
│   │   ├── augmentation.py     # Synthetic email injection
│   │   ├── email_generator.py  # Realistic email address generator
│   │   └── dataset.py          # HuggingFace tokenization & label alignment
│   │
│   ├── bert/                   # Fine-tuned BERT subpackage
│   │   ├── __init__.py
│   │   ├── model.py            # Model factory
│   │   ├── train.py            # Training pipeline
│   │   ├── evaluate.py         # Comprehensive evaluation
│   │   └── inference.py        # Production inference pipeline
│   │
│   └── llm/                    # Zero-shot LLM subpackage
│       ├── __init__.py
│       ├── config.py           # Model name & system prompt
│       ├── loader.py           # LLM model loader
│       ├── prompt.py           # Few-shot prompt builder
│       ├── inference.py        # LLM inference engine
│       ├── evaluate.py         # LLM evaluation module
│       └── run.py              # Demo runner
│
└── outputs/                    # Generated output files
```

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended for training; CPU works for inference)

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd project

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Data Setup

Place the WikiNeural dataset files in the `data/raw/` directory:
- `data/raw/train.json` — Training data (28,516 samples)
- `data/raw/test.json` — Test data (3,650 samples)

---

## CLI Usage

All operations are controlled through a single entry point: `main.py`

### Data Preprocessing

```bash
# Analyze the raw dataset (tag distribution, email presence check)
python main.py preprocess
```

### Data Augmentation

```bash
# Generate augmented training data with synthetic emails
python main.py augment

# Custom email injection ratio (default: 0.3)
python main.py augment --email-ratio 0.5
```

### Model Training

```bash
# Fine-tune BERT on the augmented dataset
python main.py train
```

### Model Evaluation

```bash
# Evaluate BERT on the test set (with detailed metrics)
python main.py evaluate --model bert

# Evaluate LLM on the test set
python main.py evaluate --model llm --num-samples 100

# Side-by-side comparison
python main.py evaluate --model both
```

### Real-World Inference

```bash
# Mask PII using the BERT model
python main.py predict --model bert --text "John Doe emailed jane.doe@gmail.com about the project"

# Mask PII using the LLM
python main.py predict --model llm --text "Contact Sarah Connor at sarah@skynet.com"
```

---

## Pipeline Details

### 1. Data Preparation

**Problem**: The WikiNeural dataset contains **zero email entities** (B-EMAIL, I-EMAIL counts are 0).

**Solution**: Synthetic email augmentation pipeline:
- Generates realistic emails using common name patterns and domains
- Tokenizes emails to match the BIO scheme (splitting on `@` and `.`)
- Injects into random positions in existing sentences
- Preserves all original data integrity

**Result**: ~30% of training samples augmented → 37,205 total samples

### 2. BERT Fine-Tuning

- **Base Model**: `bert-base-uncased` (110M parameters)
- **Task**: Token classification with 5 labels: `O`, `B-PER`, `I-PER`, `B-EMAIL`, `I-EMAIL`
- **Training**: 3 epochs, lr=2e-5, batch_size=8, weight_decay=0.01
- **Validation**: 10% train split (not test set) for epoch-level evaluation

### 3. LLM Zero-Shot Approach

- **Model**: `Qwen/Qwen2.5-1.5B-Instruct` — chosen for strong instruction-following at small size
- **Strategy**: Structured JSON extraction via few-shot prompting
- **Key Design Decision**: Extract entities as JSON rather than generating modified text, avoiding hallucination and structural instability
- **Post-processing**: Hallucination filtering, email repair, regex fallback

---

## Results & Comparison

### BERT Fine-Tuned (Full Test Set, 3650 Samples)

The BERT model was evaluated on the full un-augmented test set to measure its true generalizability.

| Metric | Value |
|--------|-------|
| **Accuracy (Token-level)** | 0.9953 |
| **Precision (Entity-level)** | 0.9698 |
| **Recall (Entity-level)** | 0.9731 |
| **F1-Score (Entity-level)** | 0.9715 |
| **False Positive Rate (FPR)** | 0.0025 |
| **False Negative Rate (FNR)** | 0.0136 |

**Confusion Matrix:**
- **True Positives (TP):** 15,570
- **False Positives (FP):** 210
- **False Negatives (FN):** 214
- **True Negatives (TN):** 84,694

### LLM Zero-Shot (Test Set Subset, Augmented)

*Note: Generative models suffer from boundary mismatches (e.g., getting "John's" instead of "John") and formatting differences (accents, casing). We report three levels of evaluation for names to provide a realistic assessment of true PII detection capability.*

| Metric | Names (Strict Exact Match) | Names (Partial Overlap) | Emails |
|--------|-----------------------------|---------------------------|--------|
| **Precision** | 0.8293 | 0.8699 | 0.8393 |
| **Recall**    | 0.5178 | 0.5271 | 1.0000 |
| **F1-Score**  | 0.6375 | 0.6564 | 0.9126 |
| **FPR**       | 0.0000 | - | 0.0000 |
| **FNR**       | 0.4822 | - | 0.0000 |

### Head-to-Head Comparison

| Dimension | BERT (Fine-Tuned) | LLM (Zero-Shot) |
|-----------|-------------------|------------------|
| **Name F1** | **0.9715** (Strict) | 0.6564 (Partial) |
| **Email F1** | **>0.99** (Hybrid Regex) | 0.9126 (Hybrid Regex) |
| **Training Required** | ✅ Yes (3 epochs, ~7 min) | ❌ No |
| **Inference Speed** | ⚡ Fast (~15 samples/sec) | 🐢 Slow (~1 sample/sec) |
| **Adaptability** | Domain-specific (needs retraining) | **High** (prompt-based updates) |
| **Hallucinations** | None | Mitigated via text-filtering |

---

## Error Analysis

### BERT Model Errors

1. **False Positives on Common Words** — "No PII in this sentence" → "No [NAME]..." (mitigated by confidence threshold + common word filter)
2. **Tokenization Sensitivity** — Performance depends on input segmentation
3. **Unseen Patterns** — Non-standard naming conventions may be missed

### LLM Model Errors

1. **Low Recall for Names** — Generative models frequently miss names that don't match their learned patterns
2. **Span Boundary Issues** — Extracts "John Doe's" instead of "John Doe"
3. **Email Component Confusion** — Sometimes extracts email local parts as person names
4. **Entity Saturation** — Occasionally over-identifies non-human entities

### Mitigation Strategies Implemented

- ✅ Confidence-based filtering (BERT)
- ✅ BIO tag sequence correction (BERT)
- ✅ Hallucination filtering (LLM)
- ✅ Email-component name removal (LLM)
- ✅ Regex fallback for missed emails (both)

---

## Future Improvements

1. **Hybrid System** — Use BERT for fast primary detection + LLM as a fallback for low-confidence segments
2. **Constrained Decoding** — Use `Outlines` or `LLM-Guidance` to force valid JSON output from the LLM
3. **Model Upgrade** — Replace BERT with DeBERTa-v3-base for better contextual understanding
4. **Probability Calibration** — Apply temperature scaling or Platt scaling for better confidence estimates
5. **Diverse Email Generation** — Expand synthetic email patterns to include organizational and international formats
6. **Active Learning** — Identify highest-uncertainty samples for targeted human annotation

---

## License

This project was developed as an internship assessment task.

---

*Built with PyTorch, HuggingFace Transformers, and seqeval.*
