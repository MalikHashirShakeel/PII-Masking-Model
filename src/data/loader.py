"""
Data loader utilities for the PII Masking pipeline.

Provides functions to load JSON datasets from disk.
"""

import json


def load_json(path):
    """Load and return data from a JSON file.

    Args:
        path (str): Absolute or relative path to the JSON file.

    Returns:
        list | dict: Parsed JSON content.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
