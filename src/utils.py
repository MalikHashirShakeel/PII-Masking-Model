import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

label_list = ["O", "B-PER", "I-PER", "B-EMAIL", "I-EMAIL"]

label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for i, l in enumerate(label_list)}


def compute_metrics(p):
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