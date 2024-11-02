from dataclasses import dataclass

import torch

SUPPORTED_MODELS = {
    "llama_3_1_8b": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",  # Llama-3.1 2x faster
    "llama_3_1_8b_instruct": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "llama_3_1_70b": "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "llama_3_1_405b": "unsloth/Meta-Llama-3.1-405B-bnb-4bit",  # 4bit for 405b!
    "mistral_small_instruct": "unsloth/Mistral-Small-Instruct-2409",  # Mistral 22b 2x faster!
    "mistral_7b_instruct": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "phi_3_5_mini_instruct": "unsloth/Phi-3.5-mini-instruct",  # Phi-3.5 2x faster!
    "phi_3_medium_4k_instruct": "unsloth/Phi-3-medium-4k-instruct",
    "gemma_2_9b": "unsloth/gemma-2-9b-bnb-4bit",
    "gemma_2_27b": "unsloth/gemma-2-27b-bnb-4bit",  # Gemma 2x faster!
    "llama_3_2_1b": "unsloth/Llama-3.2-1B-bnb-4bit",  # NEW! Llama 3.2 models
    "llama_3_2_1b_instruct": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "llama_3_2_3b": "unsloth/Llama-3.2-3B-bnb-4bit",
    "llama_3_2_3b_instruct": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
}  # More models at https://huggingface.co/unsloth


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class FinetuneConfig(Config):
    max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.
    model_name = SUPPORTED_MODELS["llama_3_2_3b"]
