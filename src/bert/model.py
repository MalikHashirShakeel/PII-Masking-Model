"""
BERT model factory for token classification.

Initializes a `BertForTokenClassification` model from the
pre-trained 'bert-base-uncased' checkpoint with the correct
number of output labels for our NER tag set.
"""

from transformers import AutoModelForTokenClassification

from src.utils import label2id, id2label

MODEL_NAME = "bert-base-uncased"


def get_model():
    """Create and return a BERT model configured for NER token classification.

    The classification head is initialized with random weights (MISSING keys
    in the load report are expected — fine-tuning will train them).

    Returns:
        BertForTokenClassification: Model ready for fine-tuning.
    """
    return AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
