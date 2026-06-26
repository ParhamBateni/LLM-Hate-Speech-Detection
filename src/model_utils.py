import os
import random
from typing import Literal

import numpy as np
import torch
import transformers
from sentence_transformers import SentenceTransformer

CACHE_DIR = "cache"
TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Set RNG seeds for Python, NumPy, and PyTorch without disabling CUDA parallelism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_kind_for_path(model_path: str) -> Literal["causal", "seq2seq"]:
    """Return ``seq2seq`` for encoder–decoder models (T5, BART, …), else ``causal``."""
    config = transformers.AutoConfig.from_pretrained(
        model_path,
        cache_dir=CACHE_DIR,
        token=TOKEN,
    )
    if getattr(config, "is_encoder_decoder", False):
        return "seq2seq"
    return "causal"


def load_model(model_path: str, device: torch.device) -> tuple:
    """Load model, tokenizer, and architecture kind (``causal`` or ``seq2seq``)."""
    kind = model_kind_for_path(model_path)

    if kind == "seq2seq":
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            cache_dir=CACHE_DIR,
            token=TOKEN,
        ).to(device)
    else:
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

    if kind == "causal":
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer, kind


def load_embedding_model(model_path: str, device: torch.device):
    return SentenceTransformer(
        model_path, cache_folder=CACHE_DIR, token=TOKEN, trust_remote_code=True
    ).to(device)
