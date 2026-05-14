import os

import torch
import transformers

CACHE_DIR = "cache"
def load_model(model_path: str, device: torch.device):
    """Load a model from the Hugging Face Hub or a local file.
    """
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token is None:
        raise RuntimeError(
            "Missing Hugging Face token. Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) "
            "in environment/.env before loading gated models."
        )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=CACHE_DIR,
        token=token,
    ).to(device)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=CACHE_DIR,
        token=token,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer