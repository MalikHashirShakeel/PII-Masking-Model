"""
Production PII masking inference pipeline using the fine-tuned BERT model.

Implements a multi-stage pipeline for real-world PII detection and masking:
1. Regex-based email masking (guaranteed catch)
2. Smart tokenization preserving masked email placeholders
3. BERT NER model prediction for person names
4. Confidence-based filtering to reduce false positives
5. BIO tag sequence correction
6. Clean detokenization for readable output
"""

import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from src.utils import get_project_root

# =========================
# CONFIG
# =========================
label_list = ["O", "B-PER", "I-PER", "B-EMAIL", "I-EMAIL"]
id2label = {i: l for i, l in enumerate(label_list)}
label2id = {l: i for i, l in id2label.items()}

CONFIDENCE_THRESHOLD = 0.85

# Regex for production-grade email detection
EMAIL_REGEX = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
)

# Common words to avoid false positives
COMMON_WORDS = {
    "no", "this", "that", "all", "none", "any", "sentence", "text"
}

# =========================
# LAZY MODEL LOADING
# =========================
_tokenizer = None
_model = None


def _load_model():
    """Lazily load the fine-tuned BERT model and tokenizer."""
    global _tokenizer, _model

    if _model is not None:
        return

    model_path = os.path.join(get_project_root(), "models", "bert_ner_final")

    _tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    _model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    _model.eval()


# ─────────────────────────────────────────────
# STEP 1 — HYBRID EMAIL MASKING (CRITICAL)
# ─────────────────────────────────────────────
def mask_emails_regex(text: str):
    """Mask emails using regex BEFORE model inference."""
    return EMAIL_REGEX.sub("[EMAIL]", text)


# ─────────────────────────────────────────────
# STEP 2 — SMART TOKENIZATION
# ─────────────────────────────────────────────
_TOKEN_RE = re.compile(
    r"\[EMAIL\]"                           # keep masked emails intact
    r"|[\w']+"                             # words
    r"|[^\w\s]"                            # punctuation
)

def smart_tokenize(text):
    """Tokenize text while preserving [EMAIL] placeholders as single tokens."""
    return _TOKEN_RE.findall(text)


# ─────────────────────────────────────────────
# STEP 3 — DETOKENIZER
# ─────────────────────────────────────────────
_ATTACH_LEFT = set(".,!?;:'\")-")
_ATTACH_RIGHT = set("(\"'")

def detokenize(tokens):
    """Reconstruct readable text from a list of tokens.

    Handles punctuation attachment (e.g., periods and commas attach left,
    opening parens attach right).
    """
    if not tokens:
        return ""
    out = tokens[0]
    for tok in tokens[1:]:
        if tok in _ATTACH_LEFT or out[-1] in _ATTACH_RIGHT:
            out += tok
        else:
            out += " " + tok
    return out


# ─────────────────────────────────────────────
# STEP 4 — MODEL PREDICTION
# ─────────────────────────────────────────────
def predict_entities(tokens):
    """Run BERT inference on a token list and return predicted labels + confidences.

    Args:
        tokens (list[str]): Input word tokens.

    Returns:
        tuple: (labels: list[str], confidences: list[float])
    """
    _load_model()

    enc = _tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        logits = _model(**enc).logits

    probs = torch.softmax(logits, dim=2)[0]
    pred_ids = logits.argmax(dim=2)[0]
    word_ids = enc.word_ids()

    labels, confs = [], []
    prev_word = None

    for pos, wid in enumerate(word_ids):
        if wid is None or wid == prev_word:
            continue

        lid = pred_ids[pos].item()
        labels.append(id2label[lid])
        confs.append(probs[pos][lid].item())

        prev_word = wid

    return labels, confs


# ─────────────────────────────────────────────
# STEP 5 — CONFIDENCE FILTER + CLEANING
# ─────────────────────────────────────────────
def apply_threshold(tokens, labels, confs, threshold):
    """Filter low-confidence predictions and fix invalid BIO sequences.

    Args:
        tokens (list[str]): Input word tokens.
        labels (list[str]): Predicted BIO labels.
        confs (list[float]): Confidence scores per token.
        threshold (float): Minimum confidence to keep an entity prediction.

    Returns:
        list[str]: Cleaned label sequence.
    """
    filtered = []

    for tok, lbl, conf in zip(tokens, labels, confs):
        # Remove low-confidence predictions
        if lbl != "O" and conf < threshold:
            lbl = "O"

        # Remove common word false positives
        if tok.lower() in COMMON_WORDS:
            lbl = "O"

        filtered.append(lbl)

    # Fix invalid BIO sequences
    for i, lbl in enumerate(filtered):
        if lbl.startswith("I-"):
            prev = filtered[i - 1] if i > 0 else "O"
            entity = lbl[2:]

            if prev not in (f"B-{entity}", f"I-{entity}"):
                filtered[i] = "O"

    return filtered


# ─────────────────────────────────────────────
# STEP 6 — MASK BUILDING
# ─────────────────────────────────────────────
def build_masked_tokens(tokens, labels):
    """Replace entity spans with [NAME] placeholders.

    Consecutive B-PER/I-PER tokens are collapsed into a single [NAME] tag.

    Args:
        tokens (list[str]): Input word tokens.
        labels (list[str]): Cleaned BIO labels.

    Returns:
        list[str]: Tokens with entities replaced by [NAME].
    """
    masked = []
    i = 0

    while i < len(tokens):
        lbl = labels[i]

        if lbl == "B-PER":
            while i < len(tokens) and labels[i] in ("B-PER", "I-PER"):
                i += 1
            masked.append("[NAME]")

        else:
            masked.append(tokens[i])
            i += 1

    return masked


# ─────────────────────────────────────────────
# FINAL PIPELINE
# ─────────────────────────────────────────────
def mask_pii(text, confidence_threshold=CONFIDENCE_THRESHOLD):
    """Mask personally identifiable information in the given text.

    Combines regex-based email detection with BERT NER for names
    to produce a masked version of the input text.

    Args:
        text (str): Raw input text.
        confidence_threshold (float): Min confidence for entity predictions.

    Returns:
        str: Text with [NAME] and [EMAIL] placeholders.
    """
    if not text or not text.strip():
        return text

    # STEP 1: Regex email masking (guaranteed)
    text = mask_emails_regex(text)

    # STEP 2: Tokenize
    tokens = smart_tokenize(text)
    if not tokens:
        return text

    # Skip already masked emails
    if all(tok == "[EMAIL]" for tok in tokens):
        return text

    # STEP 3: Model prediction
    raw_labels, confs = predict_entities(tokens)

    if len(raw_labels) != len(tokens):
        return text

    # STEP 4: Clean predictions
    labels = apply_threshold(tokens, raw_labels, confs, confidence_threshold)

    # STEP 5: Mask names only (emails already handled)
    masked_tokens = build_masked_tokens(tokens, labels)

    # STEP 6: Reconstruct
    return detokenize(masked_tokens)


# ─────────────────────────────────────────────
# TEST SUITE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("core", "Terry Bradshaw met Joe Montana yesterday"),
        ("core", "Contact me at john.doe@gmail.com for details"),
        ("core", "Tom Brady emailed sarah.smith@yahoo.com about the meeting"),
        ("core", "Elon Musk is CEO of Tesla"),
        ("core", "Reach out to michael.jordan@nba.com immediately"),

        ("edge", "Hello, my name is Alice Johnson and I can be reached at alice.j@company.org."),
        ("edge", "No PII in this sentence at all."),
        ("edge", "Multiple people: John Smith, Jane Doe, and Bob Lee."),
        ("edge", "Email me at first.last@domain.co.uk or call me."),
        ("edge", "Send report to dr.strange@avengers.org before Friday."),
        ("edge", ""),

        ("stress", "Dear John Williams, contact support@example.com or billing@example.com."),
        ("stress", "CEO Sundar Pichai emailed larry.page@google.com."),
        ("stress", "PII, API, FBI — none of these are names."),
        ("stress", "My friend O'Brien sent me an email yesterday."),
    ]

    print("\n" + "=" * 65)
    print("🛡️ FINAL PRODUCTION PII MASKING SYSTEM")
    print("=" * 65)

    for group, text in tests:
        masked = mask_pii(text)
        print(f"\n[{group.upper()}]")
        print("IN :", text)
        print("OUT:", masked)
