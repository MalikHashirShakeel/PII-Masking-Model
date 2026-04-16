from transformers import AutoModelForTokenClassification
from utils import label2id, id2label

model_name = "bert-base-uncased"

def get_model():
    return AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )