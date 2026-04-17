"""
BERT NER model training script.

Fine-tunes 'bert-base-uncased' on the augmented WikiNeural dataset
for Named Entity Recognition of person names and email addresses.
Uses HuggingFace Trainer API with validation-based evaluation.
"""

import os

from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification

from src.data.loader import load_json
from src.data.dataset import tokenize_and_align_labels, tokenizer
from src.bert.model import get_model
from src.utils import compute_metrics, get_project_root


def run_training():
    """Execute the full BERT NER training pipeline.

    Steps:
        1. Load augmented training data and test data
        2. Split training data into train/validation (90/10)
        3. Tokenize and align labels
        4. Fine-tune BERT with HuggingFace Trainer
        5. Save the final model to models/bert_ner_final/
    """
    root = get_project_root()

    # =====================
    # LOAD DATA
    # =====================
    train_data = load_json(os.path.join(root, "data", "processed", "train_augmented.json"))
    test_data = load_json(os.path.join(root, "data", "raw", "test.json"))

    print("Raw Train size:", len(train_data))
    print("Test size (final only):", len(test_data))

    # =====================
    # TRAIN / VALID SPLIT
    # =====================
    train_split, val_split = train_test_split(
        train_data,
        test_size=0.1,
        random_state=42
    )

    print("Train split:", len(train_split))
    print("Validation split:", len(val_split))

    # =====================
    # HF DATASETS
    # =====================
    train_dataset = Dataset.from_list(train_split)
    val_dataset = Dataset.from_list(val_split)

    train_dataset = train_dataset.map(tokenize_and_align_labels)
    val_dataset = val_dataset.map(tokenize_and_align_labels)

    # =====================
    # MODEL
    # =====================
    model = get_model()

    # =====================
    # DATA COLLATOR
    # =====================
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # =====================
    # TRAINING CONFIG
    # =====================
    training_args = TrainingArguments(
        output_dir=os.path.join(root, "models", "bert_ner"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none"
    )

    # =====================
    # TRAINER
    # =====================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # =====================
    # TRAIN
    # =====================
    trainer.train()

    # =====================
    # SAVE MODEL
    # =====================
    save_path = os.path.join(root, "models", "bert_ner_final")
    trainer.save_model(save_path)

    print(f"\nTraining complete. Model saved to: {save_path}")


if __name__ == "__main__":
    run_training()
