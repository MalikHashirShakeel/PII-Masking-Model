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
    5. Word-by-word containment check (each word must appear in text)

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

    # Word-by-word check: if every word from the name appears in the text,
    # accept it (handles minor spacing/ordering differences)
    name_words = name.split()
    if len(name_words) >= 2:
        text_lower = text.lower()
        if all(w.lower() in text_lower for w in name_words):
            return True

    return False


def _is_valid_email_pattern(text_candidate):
    """Check if a string looks like a plausible email address.

    Validates that the candidate has a proper local part, @ symbol,
    and valid domain with TLD. Filters out garbage regex matches
    like "California peninsula@ yahoo . com".

    Args:
        text_candidate (str): Potential email string.

    Returns:
        bool: True if it looks like a valid email.
    """
    # Reject if original text before @ looks like natural language
    # (multiple words with uppercase letters, e.g., "California peninsula@")
    at_variants = ['@', ' @ ', ' @', '@ ']
    for at in at_variants:
        if at in text_candidate:
            before_at = text_candidate.split(at)[0].strip()
            # If the part before @ has spaces between capitalized words,
            # it's natural language, not an email local part
            words_before = before_at.split()
            if len(words_before) >= 2:
                capitalized = sum(1 for w in words_before if w[0:1].isupper())
                if capitalized >= 1:
                    return False
            break

    # Normalize: remove spaces for validation
    normalized = re.sub(r'\s+', '', text_candidate).lower()

    # Must have exactly one @
    if normalized.count('@') != 1:
        return False

    local, domain = normalized.split('@')

    # Local part must be non-empty and alphanumeric (with dots/underscores)
    if not local or not re.match(r'^[a-z0-9._+\-]+$', local):
        return False

    # Domain must have at least one dot and valid TLD
    if '.' not in domain:
        return False

    parts = domain.split('.')
    tld = parts[-1]
    if len(tld) < 2 or not tld.isalpha():
        return False

    # Domain parts must be alphanumeric
    if not all(re.match(r'^[a-z0-9\-]+$', p) for p in parts):
        return False

    return True


def extract_emails_regex(text):
    """Regex-based fallback to find emails in tokenized text (with spaces around @ and .).

    Matches patterns like: "sarah668 @ gmail . com", "john . doe @ company . org"
    Validates matches to avoid false positives.

    Args:
        text (str): Input text potentially containing spaced-out emails.

    Returns:
        list[str]: Extracted and validated email strings.
    """
    # For space-tokenized emails, use a specific pattern
    spaced_pattern = r'(\S+(?:\s*\.\s*\S+)*\s+@\s+\S+(?:\s+\.\s+\S+)+)'

    # Standard email pattern
    pattern = r'(\b\w[\w.]*(?:\s*\.\s*\w+)*\s*@\s*\w+(?:\s*\.\s*\w+)+)'

    found = []
    seen_normalized = set()

    for pat in [spaced_pattern, pattern]:
        for m in re.finditer(pat, text):
            email = m.group(0).strip()
            if '@' not in email and '@ ' not in email:
                continue

            # Validate the match looks like a real email
            if not _is_valid_email_pattern(email):
                continue

            # Deduplicate
            norm = re.sub(r'\s+', '', email).lower()
            if norm not in seen_normalized:
                seen_normalized.add(norm)
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
        # Skip if the "name" looks like an email address
        if '@' in name or '@ ' in name:
            print(f"  FILTERED email-as-name: '{name}'")
            continue
        # Skip if it contains digits (likely email local part, not a name)
        if re.search(r'\d', name) and not name_in_text(name, text):
            print(f"  FILTERED numeric non-name: '{name}'")
            continue
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
                    if _is_valid_email_pattern(full_email):
                        fixed_emails.append(full_email)
                        continue
        # Validate before accepting
        if _is_valid_email_pattern(email):
            fixed_emails.append(email)
        else:
            print(f"  FILTERED invalid email: '{email}'")

    # 3. Regex fallback: catch emails missed by the LLM entirely
    regex_emails = extract_emails_regex(text)
    for re_email in regex_emails:
        re_norm = re.sub(r'\s+', '', re_email).lower()
        already_found = any(
            re.sub(r'\s+', '', fe).lower() == re_norm
            for fe in fixed_emails
        )
        if not already_found:
            fixed_emails.append(re_email)

    # 4. Remove names that are ONLY present as part of an email local part
    #    (but KEEP names that also appear independently in the text)
    final_names = []
    for name in filtered_names:
        name_lower = name.lower()

        # Build text with emails removed to check independent presence
        text_without_emails = text
        for em in fixed_emails:
            text_without_emails = text_without_emails.replace(em, " ")

        # If the name still exists in the text after removing emails, keep it
        if name_in_text(name, text_without_emails):
            final_names.append(name)
        else:
            # Check if name matches an email local part pattern exactly
            is_email_component = False
            for em in fixed_emails:
                em_normalized = re.sub(r'\s+', '', em).lower()
                if name_lower in em_normalized.split('@')[0]:
                    is_email_component = True
                    break

            if is_email_component:
                print(f"  FILTERED email-component name: '{name}'")
            else:
                # Not in email either — keep it (likely a real name)
                final_names.append(name)

    result["names"] = final_names
    result["emails"] = fixed_emails
    return result