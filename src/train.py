from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification

from data_loader import load_json
from dataset import tokenize_and_align_labels, tokenizer
from model import get_model
from utils import compute_metrics


# =====================
# LOAD DATA
# =====================
train_data = load_json("../data/processed/train_augmented.json")
test_data = load_json("../data/raw/test.json")

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
test_dataset = Dataset.from_list(test_data)


train_dataset = train_dataset.map(tokenize_and_align_labels)
val_dataset = val_dataset.map(tokenize_and_align_labels)
test_dataset = test_dataset.map(tokenize_and_align_labels)


# =====================
# MODEL
# =====================
model = get_model()


# =====================
# DATA COLLATOR (FIXES YOUR ERROR)
# =====================
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


# =====================
# TRAINING CONFIG
# =====================
training_args = TrainingArguments(
    output_dir="../models/bert_ner",
    eval_strategy="epoch",   # compatible with your version
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
    eval_dataset=val_dataset,   # ✅ validation ONLY
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
trainer.save_model("../models/bert_ner_final")

print("Training complete. Model saved.")