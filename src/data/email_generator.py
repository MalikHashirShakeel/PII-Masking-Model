"""
Synthetic email address generator for PII dataset augmentation.

Generates realistic email addresses using common first-name patterns
and popular email domains to create training data for email entity
recognition.
"""

import random

FIRST_NAMES = [
    "john", "michael", "sarah", "david", "emily",
    "daniel", "jessica", "matthew", "ashley", "chris"
]

DOMAINS = [
    "gmail.com", "yahoo.com", "outlook.com",
    "hotmail.com", "company.org"
]


def generate_email():
    """Generate a single realistic synthetic email address.

    Returns:
        str: A randomly generated email address (e.g. 'john42@gmail.com').
    """
    name = random.choice(FIRST_NAMES)
    number = random.randint(10, 999)
    domain = random.choice(DOMAINS)

    patterns = [
        f"{name}{number}@{domain}",
        f"{name}.{random.choice(FIRST_NAMES)}@{domain}",
        f"{name}@{domain}"
    ]

    return random.choice(patterns)
