#!/usr/bin/env python3
"""
PII Masking Pipeline — CLI Entry Point

A unified command-line interface for the complete PII (Personally
Identifiable Information) detection and masking pipeline, supporting
both a fine-tuned BERT NER model and a zero-shot LLM approach.

Usage:
    python main.py preprocess              # Analyze raw dataset
    python main.py augment                 # Generate augmented training data
    python main.py train                   # Fine-tune BERT NER model
    python main.py evaluate --model bert   # Evaluate BERT on test set
    python main.py evaluate --model llm    # Evaluate LLM on test set
    python main.py predict --model bert --text "John Doe emailed jane@example.com"
    python main.py predict --model llm --text "John Doe emailed jane@example.com"
"""

import os
import sys
import json
import argparse

# Ensure the project root is on the Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def cmd_preprocess(args):
    """Run data preprocessing and analysis."""
    from src.data.preprocessing import run_preprocessing
    run_preprocessing()


def cmd_augment(args):
    """Run synthetic email augmentation on the training dataset."""
    from src.data.loader import load_json
    from src.data.augmentation import augment_dataset

    train_path = os.path.join(PROJECT_ROOT, "data", "raw", "train.json")
    output_path = os.path.join(PROJECT_ROOT, "data", "processed", "train_augmented.json")

    print("Loading raw training data...")
    train_data = load_json(train_path)
    print(f"Original size: {len(train_data)}")

    ratio = args.email_ratio
    augmented_data = augment_dataset(train_data, email_ratio=ratio)
    print(f"Augmented size: {len(augmented_data)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(augmented_data, f, indent=2)

    print(f"Saved augmented dataset to: {output_path}")


def cmd_train(args):
    """Fine-tune the BERT NER model."""
    from src.bert.train import run_training
    run_training()


def cmd_evaluate(args):
    """Evaluate model(s) on the test dataset."""
    model_choice = args.model

    if model_choice in ("bert", "both"):
        print("\n" + "=" * 60)
        print("  BERT Fine-Tuned Model Evaluation")
        print("=" * 60)
        from src.bert.evaluate import run_evaluation
        run_evaluation()

    if model_choice in ("llm", "both"):
        print("\n" + "=" * 60)
        print("  LLM Zero-Shot Model Evaluation")
        print("=" * 60)
        from src.llm.evaluate import run_llm_evaluation
        run_llm_evaluation(num_samples=args.num_samples)

    if model_choice == "both":
        print("\n" + "=" * 60)
        print("  📊 COMPARISON SUMMARY")
        print("=" * 60)
        print("  See detailed metrics above for side-by-side comparison.")
        print("  BERT: Supervised fine-tuning → High precision + recall")
        print("  LLM:  Zero-shot prompting   → High precision, lower recall")
        print("=" * 60)


def cmd_predict(args):
    """Run PII masking on real-world input text."""
    text = args.text
    model_choice = args.model

    if not text:
        print("Error: --text is required for prediction.")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  PII Masking — {model_choice.upper()} Model")
    print(f"{'=' * 60}")
    print(f"\n  Input:  {text}")

    if model_choice == "bert":
        from src.bert.inference import mask_pii
        masked = mask_pii(text)
        print(f"  Output: {masked}")

    elif model_choice == "llm":
        import re
        from src.llm.loader import load_llm
        from src.llm.inference import run_llm_mask
        from src.llm.run import robust_replace

        print("  Loading LLM (this may take a moment)...")
        tokenizer, model = load_llm()

        output = run_llm_mask(text, tokenizer, model)

        masked_text = text

        # Replace emails first (longer patterns)
        for email in sorted(output.get("emails", []), key=len, reverse=True):
            masked_text = robust_replace(masked_text, email, "[EMAIL]")

        # Then replace names (longest first)
        for name in sorted(output.get("names", []), key=len, reverse=True):
            masked_text = robust_replace(masked_text, name, "[NAME]")

        print(f"  Output: {masked_text}")

    print(f"\n{'=' * 60}")


def main():
    """Parse CLI arguments and dispatch to the appropriate command handler."""
    parser = argparse.ArgumentParser(
        prog="pii-masking",
        description="PII Masking Pipeline — Detect and mask names & emails using BERT or LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python main.py preprocess
  python main.py augment --email-ratio 0.3
  python main.py train
  python main.py evaluate --model bert
  python main.py evaluate --model llm --num-samples 100
  python main.py evaluate --model both
  python main.py predict --model bert --text "John Doe emailed jane@example.com"
  python main.py predict --model llm --text "Contact Sarah at sarah@company.org"
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── preprocess ──
    sub_preprocess = subparsers.add_parser(
        "preprocess",
        help="Analyze the raw dataset (tag distribution, email check)"
    )
    sub_preprocess.set_defaults(func=cmd_preprocess)

    # ── augment ──
    sub_augment = subparsers.add_parser(
        "augment",
        help="Generate augmented training data with synthetic emails"
    )
    sub_augment.add_argument(
        "--email-ratio", type=float, default=0.3,
        help="Fraction of samples to augment with emails (default: 0.3)"
    )
    sub_augment.set_defaults(func=cmd_augment)

    # ── train ──
    sub_train = subparsers.add_parser(
        "train",
        help="Fine-tune the BERT NER model on augmented data"
    )
    sub_train.set_defaults(func=cmd_train)

    # ── evaluate ──
    sub_evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate model performance on the test dataset"
    )
    sub_evaluate.add_argument(
        "--model", type=str, choices=["bert", "llm", "both"],
        required=True,
        help="Which model to evaluate: bert, llm, or both"
    )
    sub_evaluate.add_argument(
        "--num-samples", type=int, default=100,
        help="Number of test samples for LLM evaluation (default: 100)"
    )
    sub_evaluate.set_defaults(func=cmd_evaluate)

    # ── predict ──
    sub_predict = subparsers.add_parser(
        "predict",
        help="Mask PII in real-world input text"
    )
    sub_predict.add_argument(
        "--model", type=str, choices=["bert", "llm"],
        required=True,
        help="Which model to use: bert or llm"
    )
    sub_predict.add_argument(
        "--text", type=str, required=True,
        help="Input text to mask PII in"
    )
    sub_predict.set_defaults(func=cmd_predict)

    # Parse and dispatch
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
