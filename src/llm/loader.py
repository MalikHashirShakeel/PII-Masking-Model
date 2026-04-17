"""
LLM model loader with optimized inference settings.

Loads the Qwen2.5-1.5B-Instruct model with automatic device placement
and precision selection (FP16 on GPU, FP32 on CPU).
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.llm.config import MODEL_NAME


def load_llm():
    """Load the LLM model and tokenizer with optimized settings.

    Returns:
        tuple: (tokenizer, model) ready for inference.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    return tokenizer, model