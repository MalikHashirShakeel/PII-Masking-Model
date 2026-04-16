from transformers import AutoTokenizer
from utils import label2id

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_and_align_labels(example):
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