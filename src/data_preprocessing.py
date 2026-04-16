import json
from collections import Counter


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_tags(data):
    tag_counter = Counter()
    for example in data:
        tag_counter.update(example["ner_tags"])
    return tag_counter


def print_sample(data):
    example = data[0]
    print("\nSample Data:\n")
    for token, tag in zip(example["tokens"], example["ner_tags"]):
        print(f"{token} --> {tag}")


if __name__ == "__main__":
    train_path = "../data/raw/train.json"
    test_path = "../data/raw/test.json"

    train_data = load_data(train_path)
    test_data = load_data(test_path)

    print("Train samples:", len(train_data))
    print("Test samples:", len(test_data))

    print_sample(train_data)

    tag_stats = analyze_tags(train_data)
    print("\nTag Distribution:\n", tag_stats)

    print("\nEmail Tags Check:")
    print("B-EMAIL:", tag_stats.get("B-EMAIL", 0))
    print("I-EMAIL:", tag_stats.get("I-EMAIL", 0))