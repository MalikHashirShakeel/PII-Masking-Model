# # test_setup.py

# import torch
# import transformers
# import datasets

# print("Torch:", torch.__version__)
# print("Transformers:", transformers.__version__)
# print("Datasets loaded successfully!")  

import json
from collections import Counter


# =========================
# LOAD DATASET
# =========================
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


train_path = "../data/processed/train_augmented.json"
data = load_data(train_path)

print("\n============================")
print("📊 DATASET VALIDATION REPORT")
print("============================\n")

print("Total samples:", len(data))


# =========================
# TAG STATISTICS
# =========================
tag_counter = Counter()

for sample in data:
    tag_counter.update(sample["ner_tags"])

print("\n📌 TAG DISTRIBUTION:")
for tag, count in tag_counter.items():
    print(f"{tag}: {count}")


# =========================
# EMAIL DETECTION CHECK
# =========================
print("\n📧 EMAIL ENTITY CHECK:")

email_tags_present = ("B-EMAIL" in tag_counter and tag_counter["B-EMAIL"] > 0) or \
                     ("I-EMAIL" in tag_counter and tag_counter["I-EMAIL"] > 0)

print("B-EMAIL count:", tag_counter.get("B-EMAIL", 0))
print("I-EMAIL count:", tag_counter.get("I-EMAIL", 0))

if email_tags_present:
    print("✅ Email entities successfully injected into dataset")
else:
    print("❌ ERROR: No email entities found")


# =========================
# ALIGNMENT VALIDATION
# =========================
print("\n🔗 ALIGNMENT CHECK (tokens vs tags):")

bad_samples = 0

for i, sample in enumerate(data):
    tokens = sample["tokens"]
    tags = sample["ner_tags"]

    if len(tokens) != len(tags):
        print(f"❌ Misalignment found in sample {i}")
        print("Tokens:", len(tokens), "Tags:", len(tags))
        bad_samples += 1

print("Total misaligned samples:", bad_samples)

if bad_samples == 0:
    print("✅ All samples properly aligned")
else:
    print("⚠️ Fix alignment issues before training!")


# =========================
# SAMPLE INSPECTION
# =========================
print("\n🔍 SAMPLE WITH EMAIL (if exists):")

found = False

for sample in data:
    if "B-EMAIL" in sample["ner_tags"]:
        print("\nExample sentence:")
        print(" ".join(sample["tokens"]))

        print("\nTags:")
        print(sample["ner_tags"])

        found = True
        break

if not found:
    print("No email sample found (unexpected!)")


# =========================
# FINAL SUMMARY
# =========================
print("\n============================")
print("📊 FINAL VALIDATION RESULT")
print("============================")

if email_tags_present and bad_samples == 0:
    print("✅ DATASET IS READY FOR MODEL TRAINING 🚀")
else:
    print("❌ DATASET NEEDS FIXES BEFORE TRAINING")