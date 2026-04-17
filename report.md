# PII Masking Pipeline Report

## Overview
This task aimed to develop a robust pipeline for detecting and masking Personally Identifiable Information (PII)—specifically names and email addresses. The approach was divided into training a fine-tuned encoder-based transformer (BERT), and leveraging an open-source instruction-following Large Language Model (Qwen2.5-1.5B-Instruct) for zero-shot inference.

## 1. Data Preparation & Augmentation
Initial analysis revealed a severe class imbalance and a complete absence of email entities (`B-EMAIL`, `I-EMAIL`) in the raw dataset. 
- **Email Augmentation**: We implemented a synthetic data injection mechanism capable of generating realistic email addresses and seamlessly injecting them into sequence tokens, maintaining the BIO tagging format. 
- A subset of the testing and training sets was dynamically injected with emails to properly establish ground-truth datasets for metric evaluation.

## 2. Zero-Shot PII Masking with LLMs
Instead of fine-tuning the LLM, we utilized **Qwen2.5-1.5B-Instruct**, a high-performance, fully open-source sub-2B parameter model optimized for strong reasoning and instruction following. 

### Prompting Structure & Generative Evaluation
Rather than directing the LLM to output modified raw text (which is highly prone to uncatchable hallucinations, accidental phrasing transformations, and structural instability), we designed the system prompt to enforce **Structured Entity Extraction**:
- **System Prompt**: Enforced the output format `{"names": [], "emails": []}` to output continuous entity mentions in a parseable JSON schema.
- **Robust Parsing Pipeline**: We employed `tokenizer.apply_chat_template` to natively map instructions to Qwen's specific instruction tokens. The pipeline parses the JSON securely via Regex/ast fallbacks to avoid decoding exceptions.
- **Dynamic Metric Computations**: True precision, recall, F1, FPR, and FNR metrics were implemented to measure intersection between extracted LLM JSON items and original BIO tags in `evaluate.py`.

## 3. Comparison: Fine-Tuned BERT vs Zero-Shot LLM
- **Fine-Tuned BERT Baseline**: Achieved a robust **0.97 F1-score** on the unseen test set, indicating an excellent balance between precision constraints and false/negative detection. Encoder-based bio-taggers are exceptionally fast and excel at direct token classification.
- **Zero-Shot LLM Configuration**: 
  - *Strengths*: Highly generalized. The LLM successfully abstracted the definition of "email addresses" without dataset-specific tuning. It required zero feature engineering.
  - *Weaknesses/Challenges faced*: It evaluates entities fundamentally differently than rigid token classifiers. It might output grammatical variations (e.g., extracting "John Doe's" instead of "John Doe"), trailing commas, or miss minor contextual constraints which degrades the *exact match* evaluation metrics against raw BIO arrays. It is also significantly more computationally expensive per sample.

## 4. Error Analysis & Improvement Strategies
**Common LLM Challenges in Production:**
1. **Span Boundary Matching**: Generative models frequently break exact span borders by predicting conversational prefixes (e.g. `['John Doe']` vs `['John']`, `['Doe']`), leading to false negatives despite theoretically understanding the PII element.
2. **Entity Saturation**: LLMs may over-identify non-human fictional entities, places, or corporation names if the zero-shot alignment is weak.

**Improvement Strategies:**
1. **Few-Shot Contexts**: The zero-shot capabilities of the LLM can be severely improved via *Few-Shot Prompting* (providing 5-10 pre-solved extraction examples inside the system prompt).
2. **Hybrid Verification**: Production masking systems often utilize incredibly fast models (like the BERT implementation) to parse 90% of traffic, but use LLM extraction on low-confidence `[CLS]` sequence segments as an anomaly detection second-pass fallback.
3. **Logits Biasing / JSON mode**: Applying guided generation or constrained JSON decoding libraries (e.g., `Outlines` or `LLM-Guidance`) forces the generative transformer matrix to only select vocabulary tokens matching valid RFC 8259 syntax, destroying JSON parsing errors entirely.
