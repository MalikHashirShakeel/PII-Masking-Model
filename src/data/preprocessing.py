"""
Data preprocessing and analysis utilities.

Provides functions for loading raw datasets, analyzing entity tag
distributions, and inspecting sample data entries. Used during the
initial data inspection phase to assess the dataset characteristics.
"""

import os
import json
from collections import Counter

from src.utils import get_project_root


def load_data(path):
    """Load data from a JSON file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        list[dict]: Parsed dataset entries.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_tags(data):
    """Count the frequency of each NER tag across all samples.

    Args:
        data (list[dict]): Dataset entries with 'ner_tags' field.

    Returns:
        Counter: Mapping of tag name → count.
    """
    tag_counter = Counter()
    for example in data:
        tag_counter.update(example["ner_tags"])
    return tag_counter


def print_sample(data):
    """Print the first sample in the dataset for visual inspection.

    Args:
        data (list[dict]): Dataset entries with 'tokens' and 'ner_tags'.
    """
    example = data[0]
    print("\nSample Data:\n")
    for token, tag in zip(example["tokens"], example["ner_tags"]):
        print(f"{token} --> {tag}")


def run_preprocessing():
    """Execute the full data preprocessing and analysis pipeline."""
    root = get_project_root()
    train_path = os.path.join(root, "data", "raw", "train.json")
    test_path = os.path.join(root, "data", "raw", "test.json")

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


if __name__ == "__main__":
    run_preprocessing()
