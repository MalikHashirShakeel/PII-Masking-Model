import json
from data_augmentation import augment_dataset


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


train_data = load("../data/raw/train.json")

print("Original size:", len(train_data))

augmented_data = augment_dataset(train_data, email_ratio=0.3)

print("Augmented size:", len(augmented_data))

save("../data/processed/train_augmented.json", augmented_data)

print("Saved augmented dataset!")