"""
LLM inference engine for zero-shot PII entity extraction.

Handles the full pipeline from text input to structured entity output:
1. Prompt construction and tokenization
2. LLM generation with constrained decoding
3. Robust JSON parsing of LLM output
4. Post-processing filters (hallucination removal, email fixing)
5. Regex fallback for missed emails
"""

import torch
import json
import re
import ast
import unicodedata

from src.llm.prompt import build_prompt
from src.llm.config import MAX_NEW_TOKENS, TOP_P


def normalize_text(text):
    """Normalize unicode characters for comparison (handles encoding mismatches)."""
    try:
        text = unicodedata.normalize("NFC", text)
    except Exception:
        pass
    text = text.replace("\ufffd", "?")
    return text


def parse_json_robustly(response_text):
    """Robustly extract a JSON dict with 'names' and 'emails' keys from LLM output.

    Handles markdown code blocks, extra text, and malformed JSON by applying
    multiple parsing strategies in order of reliability.

    Args:
        response_text (str): Raw LLM output string.

    Returns:
        dict: Parsed result with 'names' and 'emails' lists.
    """
    cleaned = response_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            val = json.loads(json_str)
            if isinstance(val, dict):
                return {
                    "names": [n for n in val.get("names", []) if isinstance(n, str) and n.strip()],
                    "emails": [e for e in val.get("emails", []) if isinstance(e, str) and e.strip()]
                }
        except json.JSONDecodeError:
            try:
                val = ast.literal_eval(json_str)
                if isinstance(val, dict):
                    return {
                        "names": [n for n in val.get("names", []) if isinstance(n, str) and n.strip()],
                        "emails": [e for e in val.get("emails", []) if isinstance(e, str) and e.strip()]
                    }
            except Exception:
                pass

    return {"names": [], "emails": []}


def name_in_text(name, text):
    """Check if a name appears in the text, with fuzzy matching for encoding issues.

    Applies multiple matching strategies:
    1. Exact string containment
    2. Case-insensitive regex match
    3. Unicode-normalized comparison
    4. ASCII-only comparison (handles mojibake)

    Args:
        name (str): The name to search for.
        text (str): The text to search in.

    Returns:
        bool: True if the name (or a close variant) is found.
    """
    if name in text:
        return True

    try:
        if re.search(re.escape(name), text, re.IGNORECASE):
            return True
    except re.error:
        pass

    # Handle encoding mismatches: compare normalized forms
    norm_name = normalize_text(name)
    norm_text = normalize_text(text)
    if norm_name in norm_text:
        return True

    # Strip all non-ASCII from both and compare (handles mojibake)
    ascii_name = re.sub(r'[^\x00-\x7f]', '', name).strip()
    ascii_text = re.sub(r'[^\x00-\x7f]', '', text)
    if ascii_name and len(ascii_name) >= 3 and ascii_name in ascii_text:
        return True

    return False


def extract_emails_regex(text):
    """Regex-based fallback to find emails in tokenized text (with spaces around @ and .).

    Matches patterns like: "sarah668 @ gmail . com", "john . doe @ company . org"

    Args:
        text (str): Input text potentially containing spaced-out emails.

    Returns:
        list[str]: Extracted email strings.
    """
    # Match tokenized email patterns: word(s) separated by " . " then " @ " then domain parts
    pattern = r'(\b\w[\w.]*(?:\s*\.\s*\w+)*\s*@\s*\w+(?:\s*\.\s*\w+)+)'

    # For space-tokenized emails, use a more specific pattern
    spaced_pattern = r'(\S+(?:\s*\.\s*\S+)*\s+@\s+\S+(?:\s+\.\s+\S+)+)'

    found = []
    for pat in [spaced_pattern, pattern]:
        for m in re.finditer(pat, text):
            email = m.group(0).strip()
            if '@' in email or '@ ' in email:
                found.append(email)

    return found


def run_llm_mask(text, tokenizer, model):
    """Run the LLM to extract PII entities from the given text.

    Full pipeline: prompt → generate → parse → post-process.

    Args:
        text (str): Input text to analyze.
        tokenizer: HuggingFace tokenizer for the LLM.
        model: HuggingFace causal LM model.

    Returns:
        dict: {'names': list[str], 'emails': list[str]}
    """
    messages = build_prompt(text)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            top_p=TOP_P,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("DEBUG RAW LLM OUTPUT:", repr(decoded))

    result = parse_json_robustly(decoded)

    # --- Post-processing ---

    # 1. Filter hallucinated names (not actually in the text)
    filtered_names = []
    for name in result["names"]:
        if name_in_text(name, text):
            filtered_names.append(name)
        else:
            print(f"  FILTERED hallucinated name: '{name}'")

    # 2. Fix malformed emails (e.g., missing local part)
    fixed_emails = []
    for email in result["emails"]:
        email = email.strip()
        if email.startswith("@") or email.startswith("@ "):
            # LLM missed the local part - reconstruct from text
            at_pos = text.find(email)
            if at_pos > 0:
                before = text[:at_pos].rstrip()
                words_before = before.split()
                if words_before:
                    local_part = words_before[-1]
                    full_email = local_part + " " + email
                    fixed_emails.append(full_email)
                    continue
        fixed_emails.append(email)

    # 3. Regex fallback: catch emails missed by the LLM entirely
    regex_emails = extract_emails_regex(text)
    for re_email in regex_emails:
        # Normalize for comparison
        re_norm = re.sub(r'\s+', '', re_email).lower()
        already_found = any(
            re.sub(r'\s+', '', fe).lower() == re_norm
            for fe in fixed_emails
        )
        if not already_found:
            fixed_emails.append(re_email)

    # 4. Remove any name that is actually part of an email
    email_text_combined = " ".join(fixed_emails).lower()
    final_names = []
    for name in filtered_names:
        name_lower = name.lower()
        # Check if this name is just a component of an email address
        if name_lower in email_text_combined:
            # Double-check: is this name independently present outside emails?
            text_without_emails = text
            for em in fixed_emails:
                text_without_emails = text_without_emails.replace(em, "")
            if name in text_without_emails:
                final_names.append(name)
            else:
                print(f"  FILTERED email-component name: '{name}'")
        else:
            final_names.append(name)

    result["names"] = final_names
    result["emails"] = fixed_emails
    return result