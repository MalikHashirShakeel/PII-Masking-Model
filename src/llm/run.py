"""
LLM zero-shot PII masking demo runner.

Loads the LLM, augments test data with synthetic emails, and runs
the full masking pipeline on each sample to demonstrate real-time
PII detection and replacement.
"""

import os
import re

from src.llm.loader import load_llm
from src.llm.inference import run_llm_mask
from src.data.loader import load_json
from src.data.augmentation import augment_dataset
from src.utils import get_project_root


def robust_replace(original_text, entity, tag):
    """Replace an entity in the text with a tag, handling spacing mismatches.

    Handles cases where:
    - Emails are tokenized with spaces: "sarah668 @ gmail . com"
    - LLM returns them with or without spaces
    - Unicode/encoding mismatches in names with accented characters

    Args:
        original_text (str): The text to perform replacement on.
        entity (str): The entity string to replace.
        tag (str): The replacement tag (e.g., '[NAME]', '[EMAIL]').

    Returns:
        str: Text with the entity replaced by the tag.
    """
    if not entity:
        return original_text

    # 1. Try exact match first (fastest, most reliable)
    if entity in original_text:
        return original_text.replace(entity, tag, 1)

    # 2. Build a flexible regex that allows optional spaces between characters
    entity_no_spaces = entity.replace(" ", "")

    if not entity_no_spaces:
        return original_text

    # Escape each character for regex, then join with optional whitespace
    escaped_chars = [re.escape(ch) for ch in entity_no_spaces]
    pattern = r'\s*'.join(escaped_chars)

    try:
        result = re.sub(pattern, tag, original_text, count=1, flags=re.IGNORECASE)
        if result != original_text:
            return result
    except re.error:
        pass

    # 3. Handle encoding mismatches: try replacing non-ASCII with wildcard
    entity_ascii_pattern = ""
    for ch in entity:
        if ord(ch) > 127 or ch == "\ufffd":
            entity_ascii_pattern += r'\S'  # match any non-whitespace char
        elif ch == " ":
            entity_ascii_pattern += r'\s+'
        else:
            entity_ascii_pattern += re.escape(ch)

    try:
        result = re.sub(entity_ascii_pattern, tag, original_text, count=1)
        if result != original_text:
            return result
    except re.error:
        pass

    return original_text


def run_llm_demo(num_samples=50):
    """Run the LLM PII masking demo on augmented test data.

    Args:
        num_samples (int): Number of test samples to process.
    """
    print("\n==========================")
    print("LLM ZERO-SHOT PIPELINE")
    print("==========================\n")

    tokenizer, model = load_llm()

    root = get_project_root()
    test_path = os.path.join(root, "data", "raw", "test.json")
    test_data = load_json(test_path)

    # We augment so we can see emails being masked live
    test_data = augment_dataset(test_data[:num_samples], email_ratio=0.5)

    print("Test samples:", len(test_data))

    for i, sample in enumerate(test_data):
        text = sample["sequence"]

        print(f"[{i}] IN  : {text}")

        output = run_llm_mask(text, tokenizer, model)

        masked_text = text

        # Replace emails FIRST (longer, more specific patterns)
        for email in sorted(output.get('emails', []), key=len, reverse=True):
            masked_text = robust_replace(masked_text, email, "[EMAIL]")

        # Then replace names (sorted longest first to avoid partial matches)
        for name in sorted(output.get('names', []), key=len, reverse=True):
            masked_text = robust_replace(masked_text, name, "[NAME]")

        print(f"[{i}] OUT : {masked_text}")
        print("-" * 80)


if __name__ == "__main__":
    run_llm_demo()