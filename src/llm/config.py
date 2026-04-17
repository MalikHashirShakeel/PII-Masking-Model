"""
LLM configuration for zero-shot PII masking.

Defines the model name, generation parameters, and the system prompt
used to instruct the LLM for structured entity extraction.
"""

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

MAX_NEW_TOKENS = 200
TOP_P = 1.0

SYSTEM_PROMPT = """\
You are a PII detector. Extract PERSON NAMES and EMAIL ADDRESSES from text.

RULES:
- Extract full person names exactly as written. Include first names, last names, and full names.
- Extract email addresses exactly as written, including any spaces (e.g., "john @ gmail . com").
- NEVER invent names not in the text.
- Exclude organizations, places, bands, cafes, product names.
- Output ONLY valid JSON: {"names": [...], "emails": [...]}
- Use empty lists if none found. No markdown, no explanation."""