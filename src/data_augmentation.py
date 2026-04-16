import random
from email_generator import generate_email


def insert_email(tokens, tags):
    """
    Inserts email into a sentence and updates BIO tags properly.
    """

    email = generate_email()

    email_tokens = email.replace("@", " @ ").replace(".", " . ").split()

    # Choose safe insertion point (not breaking entity structure)
    insert_idx = random.randint(1, len(tokens) - 1)

    new_tokens = tokens[:insert_idx] + email_tokens + tokens[insert_idx:]
    new_tags = tags[:insert_idx] + ["B-EMAIL"] + ["I-EMAIL"] * (len(email_tokens) - 1) + tags[insert_idx:]

    return new_tokens, new_tags, email


def augment_dataset(data, email_ratio=0.3):
    """
    Adds synthetic emails to dataset.
    email_ratio = % of sentences to modify
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