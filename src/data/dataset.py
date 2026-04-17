"""
HuggingFace Dataset tokenization and label alignment.

Provides the tokenization function that aligns BIO NER labels with
BERT's WordPiece sub-tokens, handling the label propagation for
split words and special tokens.
"""

from transformers import AutoTokenizer

from src.utils import label2id

MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_and_align_labels(example):
    """Tokenize input and align NER labels with sub-word tokens.

    For each sub-token produced by WordPiece:
    - Special tokens ([CLS], [SEP]) get label -100 (ignored in loss).
    - First sub-token of a word gets the word's original label.
    - Subsequent sub-tokens of the same word also get the word's label.

    Args:
        example (dict): Single dataset entry with 'tokens' and 'ner_tags'.

    Returns:
        dict: Tokenized inputs with aligned 'labels' field.
    """
    tokens = example["tokens"]
    labels = example["ner_tags"]

    tokenized = tokenizer(
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
