import os

import torch
import transformers
from sentence_transformers import SentenceTransformer

CACHE_DIR = "cache"
TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

def load_model(model_path: str, device: torch.device):
    """Load a model from the Hugging Face Hub or a local file.
    """
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=CACHE_DIR,
        token=TOKEN,
    ).to(device)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=CACHE_DIR,
        token=TOKEN,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer

def load_embedding_model(model_path: str, device: torch.device):
    return SentenceTransformer(model_path, cache_folder=CACHE_DIR, token=TOKEN).to(device)