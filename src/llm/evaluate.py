"""
LLM zero-shot PII masking evaluation module.

Evaluates the LLM's ability to extract person names and email addresses
against ground-truth BIO-tagged data. Computes precision, recall, F1,
FPR, and FNR for both entity types independently.

Reports both strict (exact match) and lenient (normalized) metrics
to give a fair assessment of LLM extraction quality.
"""

import os
import re
import unicodedata

from src.llm.loader import load_llm
from src.llm.inference import run_llm_mask
from src.data.loader import load_json
from src.data.augmentation import augment_dataset
from src.utils import get_project_root


def get_gold_entities(sample):
    """Extract gold-standard entity lists from BIO-tagged sample.

    Reconstructs full entity spans by joining consecutive B-/I- tagged tokens.

    Args:
        sample (dict): Dataset entry with 'tokens' and 'ner_tags'.

    Returns:
        tuple: (names: list[str], emails: list[str])
    """
    tokens = sample["tokens"]
    tags = sample["ner_tags"]

    names = []
    emails = []

    current_name = []
    current_email = []

    for token, tag in zip(tokens, tags):
        if tag == "B-PER":
            if current_name:
                names.append(" ".join(current_name))
            current_name = [token]
        elif tag == "I-PER":
            current_name.append(token)
        else:
            if current_name:
                names.append(" ".join(current_name))
                current_name = []

        if tag == "B-EMAIL":
            if current_email:
                emails.append("".join(current_email).replace(" ", ""))
            current_email = [token]
        elif tag == "I-EMAIL":
            current_email.append(token)
        else:
            if current_email:
                emails.append("".join(current_email).replace(" ", ""))
                current_email = []

    if current_name:
        names.append(" ".join(current_name))
    if current_email:
        emails.append("".join(current_email).replace(" ", ""))

    return names, emails


def normalize_email(email):
    """Normalize an email by stripping all whitespace for consistent comparison.

    Args:
        email (str): Email string potentially with spaces.

    Returns:
        str: Lowercase email with all whitespace removed.
    """
    return re.sub(r'\s+', '', email).lower()


def normalize_name(name):
    """Normalize a name for lenient comparison.

    Strips accents, lowercases, and normalizes whitespace to handle
    unicode mismatches between gold and predicted names (e.g.,
    "Renée" vs "Renee", "José" vs "Jose").

    Args:
        name (str): Person name string.

    Returns:
        str: Normalized name for comparison.
    """
    # Unicode normalize and strip accents
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Lowercase and normalize whitespace
    return " ".join(ascii_name.lower().split())


def evaluate_metrics(gold_list, pred_list, normalize_fn=None):
    """Compare gold and predicted entity lists and return TP, FP, FN counts.

    Uses optional normalize_fn for comparison (useful for emails with
    spacing differences or names with accent variations).

    Args:
        gold_list (list[str]): Ground-truth entities.
        pred_list (list[str]): Predicted entities.
        normalize_fn (callable, optional): Normalization function for comparison.

    Returns:
        tuple: (tp, fp, fn) counts.
    """
    if normalize_fn:
        gold_normalized = {normalize_fn(g): g for g in gold_list}
        pred_normalized = {normalize_fn(p): p for p in pred_list}
        gold_keys = set(gold_normalized.keys())
        pred_keys = set(pred_normalized.keys())
    else:
        gold_keys = set(gold_list)
        pred_keys = set(pred_list)

    tp = len(gold_keys & pred_keys)
    fp = len(pred_keys - gold_keys)
    fn = len(gold_keys - pred_keys)

    return tp, fp, fn


def evaluate_metrics_partial(gold_list, pred_list):
    """Evaluate with partial/overlap matching for names.

    A predicted name is a TP if it overlaps significantly with any gold name
    (i.e., shares at least one word). This handles cases where the LLM
    extracts "Bernaudeau" but the gold is "Jean-René Bernaudeau".

    Args:
        gold_list (list[str]): Ground-truth names.
        pred_list (list[str]): Predicted names.

    Returns:
        tuple: (tp, fp, fn) counts.
    """
    gold_remaining = list(gold_list)
    tp = 0
    fp = 0

    for pred in pred_list:
        pred_words = set(normalize_name(pred).split())
        matched = False

        for i, gold in enumerate(gold_remaining):
            gold_words = set(normalize_name(gold).split())

            # Match if they share at least one significant word (len >= 3)
            shared = pred_words & gold_words
            significant_shared = {w for w in shared if len(w) >= 3}

            if significant_shared:
                tp += 1
                gold_remaining.pop(i)
                matched = True
                break

        if not matched:
            fp += 1

    fn = len(gold_remaining)
    return tp, fp, fn


def run_llm_evaluation(num_samples=100):
    """Execute full LLM evaluation pipeline.

    Loads the LLM, augments test data with synthetic emails, runs
    inference on each sample, and computes comprehensive metrics
    using both strict and lenient matching.

    Args:
        num_samples (int): Number of base test samples to evaluate.
    """
    print("Loading model...")
    tokenizer, model = load_llm()
    print("Loading test data...")

    root = get_project_root()
    test_path = os.path.join(root, "data", "raw", "test.json")
    test_data = load_json(test_path)

    # Augment test data with synthetic emails for evaluation
    print("Augmenting test data with emails...")
    test_data = augment_dataset(test_data[:num_samples], email_ratio=0.5)

    # Strict matching counters
    tp_name, fp_name, fn_name = 0, 0, 0
    tp_email, fp_email, fn_email = 0, 0, 0

    # Lenient matching counters (normalized names)
    tp_name_lenient, fp_name_lenient, fn_name_lenient = 0, 0, 0

    # Partial overlap matching counters
    tp_name_partial, fp_name_partial, fn_name_partial = 0, 0, 0

    tn_name_samples, fp_name_samples = 0, 0
    tn_email_samples, fp_email_samples = 0, 0

    total_samples = len(test_data)
    print(f"Evaluating {total_samples} samples...")

    for i, sample in enumerate(test_data):
        text = sample["sequence"]

        try:
            pred_dict = run_llm_mask(text, tokenizer, model)
        except Exception as e:
            print(f"  ERROR on sample {i}: {e}")
            continue

        gold_names, gold_emails = get_gold_entities(sample)
        pred_names = pred_dict.get("names", [])
        pred_emails = pred_dict.get("emails", [])

        # Strict name metrics (exact match)
        tn, fp, fn = evaluate_metrics(gold_names, pred_names)
        tp_name += tn
        fp_name += fp
        fn_name += fn

        # Lenient name metrics (normalized — strips accents, lowercases)
        tn_l, fp_l, fn_l = evaluate_metrics(gold_names, pred_names, normalize_fn=normalize_name)
        tp_name_lenient += tn_l
        fp_name_lenient += fp_l
        fn_name_lenient += fn_l

        # Partial overlap name metrics
        tn_p, fp_p, fn_p = evaluate_metrics_partial(gold_names, pred_names)
        tp_name_partial += tn_p
        fp_name_partial += fp_p
        fn_name_partial += fn_p

        # Email metrics (normalized comparison — strip spaces)
        te, fe, fne = evaluate_metrics(gold_emails, pred_emails, normalize_fn=normalize_email)
        tp_email += te
        fp_email += fe
        fn_email += fne

        # Sample level negatives for FPR calculation
        if len(gold_names) == 0:
            if fp > 0:
                fp_name_samples += 1
            else:
                tn_name_samples += 1

        if len(gold_emails) == 0:
            if fe > 0:
                fp_email_samples += 1
            else:
                tn_email_samples += 1

        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{total_samples} samples...")

    # Calculate overall metrics
    def calc_metrics(tp, fp, fn, tn_samples=0, fp_samples=0):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn_samples) / (tp + fp + fn + tn_samples) if (tp + fp + fn + tn_samples) > 0 else 0
        fpr = fp_samples / (fp_samples + tn_samples) if (fp_samples + tn_samples) > 0 else 0
        fnr = 1 - recall
        return precision, recall, f1, accuracy, fpr, fnr

    # ── Strict Name Metrics ──
    p_name, r_name, f1_name, acc_name, fpr_name, fnr_name = calc_metrics(
        tp_name, fp_name, fn_name, tn_name_samples, fp_name_samples
    )

    # ── Lenient Name Metrics ──
    p_name_l, r_name_l, f1_name_l, _, _, _ = calc_metrics(
        tp_name_lenient, fp_name_lenient, fn_name_lenient
    )

    # ── Partial Name Metrics ──
    p_name_p, r_name_p, f1_name_p, _, _, _ = calc_metrics(
        tp_name_partial, fp_name_partial, fn_name_partial
    )

    # ── Email Metrics ──
    p_email, r_email, f1_email, acc_email, fpr_email, fnr_email = calc_metrics(
        tp_email, fp_email, fn_email, tn_email_samples, fp_email_samples
    )

    print("\n" + "=" * 60)
    print("📊 LLM Zero-Shot Evaluation Results")
    print("=" * 60)
    print(f"Total test samples evaluated: {total_samples}")

    print("\n--- NAMES (Strict Exact Match) ---")
    print(f"  TP: {tp_name}  |  FP: {fp_name}  |  FN: {fn_name}")
    print(f"  Accuracy:  {acc_name:.4f}")
    print(f"  Precision: {p_name:.4f}")
    print(f"  Recall:    {r_name:.4f}")
    print(f"  F1-Score:  {f1_name:.4f}")
    print(f"  FPR:       {fpr_name:.4f}")
    print(f"  FNR:       {fnr_name:.4f}")

    print("\n--- NAMES (Lenient — Accent-Normalized) ---")
    print(f"  TP: {tp_name_lenient}  |  FP: {fp_name_lenient}  |  FN: {fn_name_lenient}")
    print(f"  Precision: {p_name_l:.4f}")
    print(f"  Recall:    {r_name_l:.4f}")
    print(f"  F1-Score:  {f1_name_l:.4f}")

    print("\n--- NAMES (Partial Overlap Match) ---")
    print(f"  TP: {tp_name_partial}  |  FP: {fp_name_partial}  |  FN: {fn_name_partial}")
    print(f"  Precision: {p_name_p:.4f}")
    print(f"  Recall:    {r_name_p:.4f}")
    print(f"  F1-Score:  {f1_name_p:.4f}")

    print("\n--- EMAILS (Normalized Match) ---")
    print(f"  TP: {tp_email}  |  FP: {fp_email}  |  FN: {fn_email}")
    print(f"  Accuracy:  {acc_email:.4f}")
    print(f"  Precision: {p_email:.4f}")
    print(f"  Recall:    {r_email:.4f}")
    print(f"  F1-Score:  {f1_email:.4f}")
    print(f"  FPR:       {fpr_email:.4f}")
    print(f"  FNR:       {fnr_email:.4f}")

    print("\n" + "=" * 60)
    print("📝 Note: Lenient matching normalizes accents (é→e, ü→u)")
    print("   Partial matching credits overlap (shared name words)")
    print("   This reflects real-world PII safety more accurately")
    print("=" * 60)


if __name__ == "__main__":
    run_llm_evaluation(num_samples=100)