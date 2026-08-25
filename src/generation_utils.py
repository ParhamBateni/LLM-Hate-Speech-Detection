"""Tokenizer chat templates, generation-token helpers, and prediction parsing."""

from typing import Literal, Optional, Sequence

import numpy as np
import torch


def build_prompt(
    tokenizer,
    system_prompt: str,
    user_text: str,
) -> str:
    """Build a prompt string for the model (chat template or plain text fallback)."""
    ct = getattr(tokenizer, "chat_template", None)
    user_text = user_text.replace('"', "'")
    query = f'QUERY: "{user_text}"'

    if ct is None:
        return f"{system_prompt.strip()}\n\n{query}"

    # Gemma (and similar) reject a separate system role; fold instructions into the user turn.
    if isinstance(ct, str) and "System role not supported" in ct:
        conversation = [
            {"role": "user", "content": f"{system_prompt.strip()}\n\n{query}"},
        ]
    else:
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

    return tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )


def calculate_confidence_score(
    chosen_logits: torch.Tensor,
    chosen_ids: torch.Tensor,
    excluded_token_ids: Optional[torch.Tensor] = None,
) -> float:
    """Geometric mean of p(chosen_token | context) over decoding steps."""
    n_tok = int(chosen_ids.shape[0])
    n_step = int(chosen_logits.shape[0])
    if n_tok == n_step + 1:
        chosen_ids = chosen_ids[1:]
    if n_tok > n_step + 1:
        chosen_ids = chosen_ids[-n_step:]
    else:
        chosen_logits = chosen_logits[-n_tok:]
    if chosen_ids.shape[0] == 0:
        return float("nan")

    log_probs = torch.log_softmax(chosen_logits, dim=-1)
    valid = torch.ones(chosen_ids.shape[0], dtype=torch.bool, device=log_probs.device)
    if excluded_token_ids is not None and excluded_token_ids.numel() > 0:
        ex = excluded_token_ids.to(chosen_ids.device).long()
        valid &= ~torch.isin(chosen_ids, ex)
    if not valid.any().item():
        return float("nan")
    log_probs_tokens = log_probs[valid, chosen_ids[valid]]
    return np.round(float(torch.exp(log_probs_tokens.mean()).item()), 3)


def prediction_start_after_tags(
    new_tokens: torch.Tensor, tag_tensors: Sequence[torch.Tensor]
) -> Optional[int]:
    """Index of first token after a recognized PREDICTION marker, or ``None``."""
    for tag in tag_tensors:
        n = tag.numel()
        if n == 0 or new_tokens.numel() < n:
            continue
        for k in range(new_tokens.numel() - n, -1, -1):
            if torch.all(new_tokens[k : k + n] == tag):
                return k + n
    return None


def parse_prediction_label(
    prediction: str,
) -> Optional[Literal["hateful", "non-hateful"]]:
    if (
        "non-hateful" in prediction
        or "non hateful" in prediction
        or prediction in ("non-hateful", "non hateful")
    ):
        return "non-hateful"
    if (
        prediction == "hateful"
        or prediction.startswith("hateful")
        or prediction in ("hate speech", "hate-speech")
    ):
        return "hateful"
    return None


def build_prediction_tag_tensors(tokenizer) -> list[torch.Tensor]:
    return [
        torch.tensor(
            tokenizer.encode(prefix, add_special_tokens=False), dtype=torch.long
        )
        for prefix in ("PREDICTION:", "\n\nPREDICTION:", "\nPREDICTION:")
    ]


def get_special_token_ids(tokenizer) -> torch.Tensor:
    return torch.tensor(
        sorted(
            {int(t) for t in getattr(tokenizer, "all_special_ids", []) if t is not None}
        ),
        dtype=torch.long,
    )
