"""
Few-shot prompt builder for LLM-based PII extraction.

Constructs the chat message sequence with system instructions and
few-shot examples that guide the LLM to extract entities in a
structured JSON format.
"""

from src.llm.config import SYSTEM_PROMPT


def build_prompt(text: str) -> list:
    """Build the chat prompt for PII entity extraction.

    Uses 2 concise few-shot examples for accuracy without excessive
    prompt length. Critical for CPU inference where every token adds latency.

    Args:
        text (str): The input text to extract PII from.

    Returns:
        list[dict]: Chat messages formatted for tokenizer.apply_chat_template().
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        # Few-shot example 1: Names + spaced email
        {"role": "user", "content": "Tom Brady emailed daniel . david @ company . org about Elon Musk ."},
        {"role": "assistant", "content": '{"names": ["Tom Brady", "Elon Musk"], "emails": ["daniel . david @ company . org"]}'},
        # Few-shot example 2: No PII (teaches model not to hallucinate)
        {"role": "user", "content": "The city voted for the measure at Le Dôme Café ."},
        {"role": "assistant", "content": '{"names": [], "emails": []}'},
        # Actual query
        {"role": "user", "content": text},
    ]