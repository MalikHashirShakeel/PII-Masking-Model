"""
Shared utilities for the PII Masking pipeline.

Contains label mappings, metric computation functions, and path
resolution helpers used across both BERT and LLM evaluation modules.
"""

import os
import numpy as np
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# ─────────────────────────────────────────────
# Label Mappings (BIO Tagging Scheme)
# ─────────────────────────────────────────────
label_list = ["O", "B-PER", "I-PER", "B-EMAIL", "I-EMAIL"]

label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for i, l in enumerate(label_list)}


# ─────────────────────────────────────────────
# Path Utilities
# ─────────────────────────────────────────────
def get_project_root():
    """Return the absolute path to the project root directory.

    Resolves relative to this file's location: project/src/utils.py → project/

    Returns:
        str: Absolute path to the project root.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────
# Metric Computation
# ─────────────────────────────────────────────
def compute_metrics(p):
    """Compute precision, recall, and F1 from model predictions.

    Uses seqeval for proper entity-level evaluation following the
    CoNLL-2003 evaluation protocol.

    Args:
        p: EvalPrediction object with .predictions and .label_ids arrays.

    Returns:
        dict: Dictionary with 'precision', 'recall', and 'f1' keys.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_preds = []
    true_labels = []

    for pred, label in zip(predictions, labels):
        pred_list = []
        label_list_clean = []

        for p_i, l_i in zip(pred, label):
            if l_i != -100:
                pred_list.append(id2label[p_i])
                label_list_clean.append(id2label[l_i])

        true_preds.append(pred_list)
        true_labels.append(label_list_clean)

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
    }


def compute_detailed_metrics(predictions, labels):
    """Compute comprehensive metrics including per-entity breakdown.

    Extends the basic metrics with:
    - Token-level accuracy
    - Per-entity precision/recall/F1 via seqeval classification_report
    - False Positive Rate (FPR) and False Negative Rate (FNR)

    Args:
        predictions (np.ndarray): Model logits of shape (N, seq_len, num_labels).
        labels (np.ndarray): Ground-truth label IDs of shape (N, seq_len).

    Returns:
        dict: Comprehensive metrics dictionary.
    """
    pred_ids = np.argmax(predictions, axis=2)

    true_preds = []
    true_labels = []
    all_pred_flat = []
    all_label_flat = []

    for pred, label in zip(pred_ids, labels):
        pred_list = []
        label_list_clean = []

        for p_i, l_i in zip(pred, label):
            if l_i != -100:
                pred_tag = id2label[p_i]
                true_tag = id2label[l_i]
                pred_list.append(pred_tag)
                label_list_clean.append(true_tag)
                all_pred_flat.append(pred_tag)
                all_label_flat.append(true_tag)

        true_preds.append(pred_list)
        true_labels.append(label_list_clean)

    # ── Entity-level metrics (seqeval) ──
    entity_precision = precision_score(true_labels, true_preds)
    entity_recall = recall_score(true_labels, true_preds)
    entity_f1 = f1_score(true_labels, true_preds)
    entity_report = classification_report(true_labels, true_preds, digits=4)

    # ── Token-level accuracy ──
    correct = sum(1 for p, l in zip(all_pred_flat, all_label_flat) if p == l)
    total = len(all_pred_flat)
    accuracy = correct / total if total > 0 else 0.0

    # ── FPR / FNR (entity-level, for PER + EMAIL combined) ──
    # FP: predicted entity tag where true is 'O'
    # FN: true entity tag where predicted is 'O'
    # TN: both are 'O'
    tp, fp, fn, tn = 0, 0, 0, 0
    for p, l in zip(all_pred_flat, all_label_flat):
        p_is_entity = (p != "O")
        l_is_entity = (l != "O")

        if p_is_entity and l_is_entity:
            tp += 1
        elif p_is_entity and not l_is_entity:
            fp += 1
        elif not p_is_entity and l_is_entity:
            fn += 1
        else:
            tn += 1

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "entity_precision": entity_precision,
        "entity_recall": entity_recall,
        "entity_f1": entity_f1,
        "token_fpr": fpr,
        "token_fnr": fnr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "classification_report": entity_report,
    }