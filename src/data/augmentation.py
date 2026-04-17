"""
Data augmentation module for injecting synthetic email entities.

Takes the raw WikiNeural dataset (which lacks email entities) and
augments it by inserting synthetic email addresses with proper BIO
tagging to enable the NER model to learn email detection.
"""

import random

from src.data.email_generator import generate_email


def insert_email(tokens, tags):
    """Insert a synthetic email into a token sequence and update BIO tags.

    Generates a random email, tokenizes it (splitting on '@' and '.'),
    and inserts it at a random safe position in the sentence.

    Args:
        tokens (list[str]): Original word tokens.
        tags (list[str]): Corresponding BIO NER tags.

    Returns:
        tuple: (new_tokens, new_tags, email_string)
    """
    email = generate_email()

    email_tokens = email.replace("@", " @ ").replace(".", " . ").split()

    # Choose safe insertion point (not breaking entity structure)
    insert_idx = random.randint(1, len(tokens) - 1)

    new_tokens = tokens[:insert_idx] + email_tokens + tokens[insert_idx:]
    new_tags = tags[:insert_idx] + ["B-EMAIL"] + ["I-EMAIL"] * (len(email_tokens) - 1) + tags[insert_idx:]

    return new_tokens, new_tags, email


def augment_dataset(data, email_ratio=0.3):
    """Augment dataset by injecting synthetic emails into a subset of samples.

    Each original sample is preserved. With probability `email_ratio`,
    a copy with an injected email is also added.

    Args:
        data (list[dict]): List of dataset samples with 'tokens' and 'ner_tags'.
        email_ratio (float): Fraction of samples to augment (default: 0.3).

    Returns:
        list[dict]: Augmented dataset (always >= original size).
    """
    augmented_data = []

    for sample in data:
        tokens = sample["tokens"]
        tags = sample["ner_tags"]

        # keep original sample
        augmented_data.append(sample)

        # decide whether to inject email
        if random.random() < email_ratio:
            new_tokens, new_tags, _ = insert_email(tokens, tags)

            augmented_data.append({
                "tokens": new_tokens,
                "ner_tags": new_tags,
                "sequence": " ".join(new_tokens),
                "lang": sample.get("lang", "en")
            })

    return augmented_data
