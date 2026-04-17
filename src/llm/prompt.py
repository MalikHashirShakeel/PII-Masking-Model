"""
Few-shot prompt builder for LLM-based PII extraction.

Constructs the chat message sequence with system instructions and
few-shot examples that guide the LLM to extract entities in a
structured JSON format. Uses 3 carefully designed examples that
cover the most common failure modes.
"""

from src.llm.config import SYSTEM_PROMPT


def build_prompt(text: str) -> list:
    """Build the chat prompt for PII entity extraction.

    Uses 3 targeted few-shot examples chosen to address specific failure modes:
    1. Multiple names + spaced email (baseline)
    2. No PII (prevents hallucination)
    3. Many diverse names (teaches exhaustive extraction)

    Args:
        text (str): The input text to extract PII from.

    Returns:
        list[dict]: Chat messages formatted for tokenizer.apply_chat_template().
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},

        # Few-shot example 1: Multiple names + spaced email
        {"role": "user", "content": "Tom Brady emailed daniel . david @ company . org about Elon Musk ."},
        {"role": "assistant", "content": '{"names": ["Tom Brady", "Elon Musk"], "emails": ["daniel . david @ company . org"]}'},

        # Few-shot example 2: No PII — teaches the model NOT to hallucinate
        {"role": "user", "content": "The city voted for the measure at Le Dôme Café near Central Park ."},
        {"role": "assistant", "content": '{"names": [], "emails": []}'},

        # Few-shot example 3: Many names including single-word, accented, and hyphenated
        {"role": "user", "content": "Jean-René met Susy , Clara , José Ferrer and O'Brien at the conference ."},
        {"role": "assistant", "content": '{"names": ["Jean-René", "Susy", "Clara", "José Ferrer", "O\'Brien"], "emails": []}'},

        # Actual query
        {"role": "user", "content": text},
    ]