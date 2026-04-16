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
    name = random.choice(FIRST_NAMES)
    number = random.randint(10, 999)
    domain = random.choice(DOMAINS)

    patterns = [
        f"{name}{number}@{domain}",
        f"{name}.{random.choice(FIRST_NAMES)}@{domain}",
        f"{name}@{domain}"
    ]

    return random.choice(patterns)