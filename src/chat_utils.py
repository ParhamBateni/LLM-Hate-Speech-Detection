def _chat_template_rejects_system_role(tokenizer) -> bool:
    """True when the tokenizer's Jinja chat template errors on ``role: system`` (e.g. Gemma 2 IT)."""
    ct = getattr(tokenizer, "chat_template", None)
    return isinstance(ct, str) and "System role not supported" in ct


def build_chat_messages(tokenizer, system_prompt: str, user_text: str) -> list:
    """Build HF-style chat messages; fold system into user when the template has no system role."""
    if _chat_template_rejects_system_role(tokenizer):
        return [{"role": "user", "content": f"System: {system_prompt.strip()}\n\nUser: {user_text}"}]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
