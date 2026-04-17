"""
BERT NER model evaluation on the held-out test set.

Loads the fine-tuned model and evaluates against the unseen test dataset.
Reports comprehensive metrics including accuracy, precision, recall, F1,
FPR, FNR, and per-entity classification breakdown.
"""

import os
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
)

from src.data.loader import load_json
from src.utils import (
    label2id,
    id2label,
    compute_metrics,
    compute_detailed_metrics,
    get_project_root,
)

MODEL_NAME = "bert-base-uncased"


def tokenize_and_align_labels(example):
    """Tokenize and align labels for evaluation (same logic as training).

    Args:
        example (dict): Single dataset entry with 'tokens' and 'ner_tags'.

    Returns:
        dict: Tokenized inputs with aligned 'labels' field.
    """
    tokens = example["tokens"]
    labels = example["ner_tags"]

    tokenized = _tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True
    )

    word_ids = tokenized.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            label_ids.append(label2id[labels[word_idx]])

        else:
            label_ids.append(label2id[labels[word_idx]])

        previous_word_idx = word_idx

    tokenized["labels"] = label_ids
    return tokenized


# Module-level tokenizer (initialized in run_evaluation)
_tokenizer = None


def run_evaluation():
    """Execute comprehensive evaluation of the fine-tuned BERT NER model.

    Loads the saved model, prepares the test dataset, and computes:
    - Standard metrics: loss, precision, recall, F1
    - Enhanced metrics: accuracy, FPR, FNR, per-entity breakdown
    """
    global _tokenizer

    root = get_project_root()
    model_path = os.path.join(root, "models", "bert_ner_final")
    test_path = os.path.join(root, "data", "raw", "test.json")

    # =========================
    # LOAD MODEL + TOKENIZER
    # =========================
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    # =========================
    # LOAD TEST DATA
    # =========================
    test_data = load_json(test_path)
    print(f"Test samples: {len(test_data)}")

    # =========================
    # TOKENIZE & PREPARE
    # =========================
    test_dataset = Dataset.from_list(test_data)
    test_dataset = test_dataset.map(tokenize_and_align_labels)

    data_collator = DataCollatorForTokenClassification(tokenizer=_tokenizer)

    # =========================
    # STANDARD EVALUATION (Trainer)
    # =========================
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    results = trainer.evaluate(eval_dataset=test_dataset)

    print("\n" + "=" * 50)
    print("📊 STANDARD TEST RESULTS")
    print("=" * 50)

    for k, v in results.items():
        print(f"  {k}: {v}")

    # =========================
    # ENHANCED EVALUATION
    # =========================
    print("\n" + "=" * 50)
    print("📊 DETAILED EVALUATION (Enhanced Metrics)")
    print("=" * 50)

    # Get raw predictions for detailed analysis
    raw_output = trainer.predict(test_dataset)
    detailed = compute_detailed_metrics(raw_output.predictions, raw_output.label_ids)

    print(f"\n  Token-level Accuracy: {detailed['accuracy']:.4f}")
    print(f"  Entity Precision:     {detailed['entity_precision']:.4f}")
    print(f"  Entity Recall:        {detailed['entity_recall']:.4f}")
    print(f"  Entity F1-Score:      {detailed['entity_f1']:.4f}")
    print(f"  False Positive Rate:  {detailed['token_fpr']:.4f}")
    print(f"  False Negative Rate:  {detailed['token_fnr']:.4f}")
    print(f"\n  Confusion (TP={detailed['tp']}, FP={detailed['fp']}, "
          f"FN={detailed['fn']}, TN={detailed['tn']})")

    print(f"\n  Per-Entity Classification Report:\n")
    print(detailed["classification_report"])

    print("\n✅ Evaluation completed on TRUE unseen test set")


if __name__ == "__main__":
    run_evaluation()
